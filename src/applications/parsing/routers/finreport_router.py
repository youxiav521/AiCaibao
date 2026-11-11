from typing import Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Header
from loguru import logger

from src.applications.parsing.service.parse_minerU_service import MinerUParsingService
from src.applications.parsing.utils.storage_utils import upload_file_to_minio_temp
from src.applications.parsing.utils.mineru_storage import extract_pdf_info_from_mineru_result
from src.applications.rag.services.index_service import index_document_items

from src.utils import json_utils
from src.utils.stopwatch import StopWatch

router = APIRouter(prefix="/api/file", tags=["解析给前端展示"])

_TABLE_TYPE_DISPLAY: dict[str, dict[str, str]] = {
    "合并资产负债表": {"zh": "合并资产负债表", "en": "Consolidated Balance Sheet"},
    "母公司资产负债表": {"zh": "单体资产负债表", "en": "Standalone Balance Sheet"},
    "合并利润表": {"zh": "合并利润表", "en": "Consolidated Income Statement"},
    "母公司利润表": {"zh": "单体利润表", "en": "Standalone Income Statement"},
    "合并现金流量表": {"zh": "合并现金流量表", "en": "Consolidated Cashflow Statement"},
    "母公司现金流量表": {"zh": "单体现金流量表", "en": "Standalone Cashflow Statement"},
    "资产负债表": {"zh": "资产负债表", "en": "Balance Sheet"},
    "利润表": {"zh": "利润表", "en": "Income Statement"},
    "现金流量表": {"zh": "现金流量表", "en": "Cashflow Statement"},
    "其他表": {"zh": "其他表", "en": "Other"},
}


def _extract_content_and_page_sizes_from_mineru_result(
    mineru_result: Any,
    *,
    file_id: str,
) -> tuple[Optional[Any], Optional[list[Any]]]:
    filtered_content = None
    pages_size = None

    if isinstance(mineru_result, dict) and "results" in mineru_result:
        results = mineru_result.get("results", {})
        if isinstance(results, dict) and results:
            file_data = results.get(file_id)
            if file_data is None:
                file_data = next(iter(results.values()))
            if isinstance(file_data, dict):
                filtered_content = {"md_content": file_data.get("md_content")}
                pdf_info = extract_pdf_info_from_mineru_result(mineru_result, file_id=file_id)
                if pdf_info:
                    pages_size = [
                        page.get("page_size")
                        for page in pdf_info
                        if isinstance(page, dict) and "page_size" in page
                    ]
    elif isinstance(mineru_result, list):
        filtered_content = [
            {"md_content": item.get("md_content")} for item in mineru_result
        ]
    elif isinstance(mineru_result, dict) and "md_content" in mineru_result:
        filtered_content = {"md_content": mineru_result.get("md_content")}

    return filtered_content, pages_size


def _translate_convert_json(items: List[Any], locale: Optional[str]) -> List[Any]:
    """Translate Chinese table_type labels in convert_json items based on locale."""
    if not items or not isinstance(items, list):
        return items
    lang = "en" if locale and locale.strip().lower().startswith("en") else "zh"
    for item in items:
        if not isinstance(item, dict):
            continue
        table_type = item.get("table_type")
        if isinstance(table_type, str) and table_type in _TABLE_TYPE_DISPLAY:
            item["table_type"] = _TABLE_TYPE_DISPLAY[table_type][lang]
    return items


@router.post(
    "/read",
    summary="上传文件 -> 异步调用 MinerU，生成并保存目标items（推荐）",
)
async def parse_to_json(
    file: UploadFile = File(
        ..., description="支持 .pdf / .docx / .md / .doc / .pptx / .png / .jpg / .jpeg / .txt"
    ),
    folderId: str = Form(None, description="文件夹ID"),
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    sw = StopWatch()
    logger.info("收到解析请求 | folder_id={}", folderId)

    in_path: str | None = None
    try:
        # 持久化到 MinIO，并生成解析用临时文件。
        with sw.task("save_file"):
            file_id, in_path, original_filename = await upload_file_to_minio_temp(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    try:
        with sw.task("parse_mineru"):
            items_paths = await MinerUParsingService().parse_to_items_and_save_async(
                in_path,
                file_id=file_id,
                classify_tables=True,
                original_filename=original_filename,
                folder_id=folderId,
                locale=accept_language,
            )

        # 加载解析结果
        filtered_content, pages_size = _extract_content_and_page_sizes_from_mineru_result(
            items_paths.get("mineru_result"),
            file_id=file_id,
        )

        convert_json = items_paths.get("items")

        if not isinstance(convert_json, list):
            raise ValueError("convert_json 解析结果缺失或格式非法，无法执行索引")

        converted_folder_id = next(
            (item.get("folder_id") for item in convert_json if isinstance(item, dict)),
            None,
        )
        logger.info(
            "解析完成 | converted_json_folder_id={}",
            converted_folder_id,
        )

        with sw.task("index_document_items"):
            indexed_file_id = await index_document_items(
                doc_parse_items=convert_json,
                file_name=original_filename,
                folder_id=folderId,
            )

        if convert_json:
            with sw.task("translate_convert_json"):
             convert_json = _translate_convert_json(convert_json, accept_language)

        response_data = {
            "success": True,
            "file_id": file_id,
            "content": filtered_content,
            "convert_json": convert_json,
            "pages_size": pages_size,
        }
        with sw.task("serialize_response") as t:
            response_length = len(json_utils.dumps(response_data, ensure_ascii=False))
            t.detail(f"length={response_length}")
        return response_data
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError, FileExistsError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"索引失败: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {e}")
    finally:
        if in_path:
            try:
                Path(in_path).unlink(missing_ok=True)
            except Exception:
                logger.warning("临时上传文件清理失败 | path={}", in_path)
        sw.print()
