from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel, Field
from loguru import logger

from src.ade.finacial.service.financial_service import AdeFinancialService
from src.applications.parsing.utils.financial_metadata_extractor import parse_financial_metadata, parse_three_statements
from src.applications.parsing.service.three_statements_frontend_service import (
    extract_three_statements_frontend_by_file_id,
    extract_org_tables_by_file_id,
    manual_merge_tables,
)
from src.applications.parsing.utils.mineru_storage import load_mineru_items
from src.utils import StopWatch

router = APIRouter(prefix="/api/parsing/extract", tags=["财报元数据提取"])


# ---------------------------------------------------------------------------
# Request schemas for manual merge
# ---------------------------------------------------------------------------
class MergeSpec(BaseModel):
    sourceIds: List[str] = Field(..., description="要合并的原始表格 id 列表（有序，按行拼接顺序）")
    text: Optional[str] = Field(None, description="合并后的报表标签（可选，不传则自动推断）")


class ManualMergeRequest(BaseModel):
    fileId: str = Field(..., description="文件唯一标识")
    merges: List[MergeSpec] = Field(..., description="合并指令列表")


@router.get(
    "/financial-metadata/{file_id}",
    summary="基于 file_id 提取财报元数据和财务三大表",
)
async def extract_financial_metadata_by_file_id(
    file_id: str,
    save: bool = Query(False, description="是否将提取结果单独保存到 metadata 目录"),
):
    sw = StopWatch(f"extract_financial_metadata({file_id}, {save})")
    try:
        items = await load_mineru_items(file_id)
        if not items:
            raise FileNotFoundError(f"目标 items 不存在: {file_id}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 items 失败: {e}")

    # 调用元数据提取器
    try:
        with sw.task("extract_financial_metadata"):
            metadata = parse_financial_metadata(items)
        if not isinstance(metadata, dict):
            raise ValueError("元数据提取结果不是字典结构")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"元数据提取失败: {e}")

    # 调用三大表提取器
    try:
        with sw.task("extract_three_statements"):
            three_statements_result = parse_three_statements(items)
        if not isinstance(three_statements_result, dict):
            raise ValueError("三大表提取结果不是字典结构")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"三大表提取失败: {e}")

    metadata_saved_path = None
    three_statements_saved_path = None
    if save:
        try:
            metadata_saved_path = await AdeFinancialService._upload_result_to_minio(
                job_id=file_id,
                result_type="metadata",
                payload=metadata,
            )
            three_statements_saved_path = await AdeFinancialService._upload_result_to_minio(
                job_id=file_id,
                result_type="three_statements",
                payload=three_statements_result,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存结果失败: {e}")
    
    sw.print()
    return {
        "success": True,
        "file_id": file_id,
        "metadata": metadata,
        "three_statements": three_statements_result.get("三大表") if isinstance(three_statements_result, dict) else None,
        "metadata_path": metadata_saved_path or "",
        "three_statements_path": three_statements_saved_path or "",
    }

@router.get(
    "/three-statements/{file_id}",
    summary="基于 file_id 提取财务三大表",
    description="凡属三大表家族（合并/母公司/无前缀及利润表+OCI 等变体）按细分标签收集输出；"
    "同一细分标签下多张 MinerU 表（如多页多张「利润表」）可全部返回，同 expanded_idx 仍择优一条；"
    "同一块内若并列合并/单体金额列再拆两条；若单表内并列 **三个及以上** 不同报告期截止日金额列，则按相邻期间再拆多张两列金额表"
    "子表 ``period`` 按相邻两期间隔推断为年报/半年报/季报等。"
    "顶层 category 归一为资产负债表/利润表/现金流量表；口径见 caliberLabel。"
    "``mergeIds`` 与 ``/org-tables`` 返回的 ``id`` 对齐（含 ``_ss_*`` / ``_mp*`` / ``_grp`` / ``_std`` 等逻辑后缀）；"
    "``primaryElementId`` 为去掉上述后缀后的 MinerU ``element_id``；多表纵向合并时 ``id`` 仍可为派生 UUID。",
)
async def extract_three_statements_by_file_id(
    file_id: str,
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    sw = StopWatch(f"extract_three_statements({file_id}, {accept_language})")
    
    try:
        # 2. 执行核心逻辑并获取返回结果
        with sw.task("extract_three_statements"):
            result = await extract_three_statements_frontend_by_file_id(file_id, locale=accept_language)
        
        return result
    
    except FileNotFoundError as e:
        # 4. 记录文件不存在的错误
        logger.error(
            "三大表提取请求失败 | file_id={} | 错误类型=文件不存在 | 错误信息={}",
            file_id,
            e,
        )
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        # 5. 记录其他所有异常（包含堆栈信息，方便排查）
        logger.exception(
            "三大表提取请求失败 | file_id={} | 错误类型=服务器内部错误 | 错误信息={}",
            file_id,
            e,
        )
        raise HTTPException(status_code=500, detail=f"三大表提取失败: {e}")
    finally:
        sw.print()


@router.get(
    "/org-tables/{file_id}",
    summary="获取所有原始表格块（未合并）",
    description="条数=解析出的逻辑表数（excel 左右并排见 ``…_ss_l`` / ``…_ss_r``）× 是否按多报告期列拆分（``…_mp0`` / ``…_mp1`` …，"
    "仅当单表内可解析出 **三个及以上** 相异报告期截止日时）× 是否再拆合并/单体（``…_grp`` / ``…_std``）。"
    "``id`` 可含后缀以保证唯一；每条 ``mergeIds`` 与同条 ``id`` 一致（含拆分后缀），便于与 ``/three-statements`` 对齐；"
    "``primaryElementId`` 为 MinerU 主键（已剥离派生后缀）。",
)
async def get_org_tables_by_file_id(
    file_id: str,
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    """返回 file_id 对应 PDF 中所有独立表格元素。

    ``id`` 在同一块多次拆分时带 ``_ss_*`` / ``_mp*`` / ``_grp`` / ``_std`` 等后缀以示区分；
    ``mergeIds`` 与 ``/three-statements`` 使用同一套逻辑表 id，可与该接口输出逐条对齐；
    ``primaryElementId`` 为 MinerU 原始 ``element_id``（不含后缀）。
    手动合并 ``sourceIds`` 仍使用 MinerU 原始 id。
    """
    sw = StopWatch(f"get_org_tables_by_file_id({file_id}, {accept_language})")

    try:
        with sw.task("extract_org_tables") as t:
            result = await extract_org_tables_by_file_id(file_id, locale=accept_language)
            t.detail(f"表格数: {len(result)}")
        return result

    except FileNotFoundError as e:
        logger.error(
            "获取原始表格列表失败 | file_id={} | 错误类型=文件不存在 | 错误信息={}",
            file_id,
            e,
        )
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(
            "获取原始表格列表失败 | file_id={} | 错误类型=服务器内部错误 | 错误信息={}",
            file_id,
            e,
        )
        raise HTTPException(status_code=500, detail=f"获取原始表格失败: {e}")
    finally:
        sw.print()


@router.post(
    "/merge-tables",
    summary="手动合并表格",
)
async def merge_tables_manually(
    req: ManualMergeRequest,
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    """按指定的合并指令将多个原始表格块拼接为一张合并表。

    请求体中 ``merges[].sourceIds`` 的顺序即为行拼接顺序（第一个为
    主表）。响应格式与 ``/three-statements`` 一致，包含 ``id`` 与
    ``mergeIds`` 用于回溯。
    """
    sw = StopWatch(f"merge_tables_manually({req.fileId}, {accept_language})")

    try:
        merge_specs = [m.model_dump() for m in req.merges]
        with sw.task("manual_merge_tables") as t:
            result = await manual_merge_tables(req.fileId, merge_specs, locale=accept_language)
            t.detail(f"输出表数: {len(result)}")
        return result

    except FileNotFoundError as e:
        logger.error(
            "手动合并失败 | file_id={} | 错误类型=文件不存在 | 错误信息={}",
            req.fileId,
            e,
        )
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(
            "手动合并失败 | file_id={} | 错误类型=服务器内部错误 | 错误信息={}",
            req.fileId,
            e,
        )
        raise HTTPException(status_code=500, detail=f"手动合并失败: {e}")
    finally:
        sw.print()
