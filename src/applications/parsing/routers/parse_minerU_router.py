from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Header
from pathlib import Path
from typing import Optional

from loguru import logger

from src.applications.parsing.service.parse_minerU_service import MinerUParsingService
from src.applications.parsing.utils.storage_utils import upload_file_to_minio_temp
from src.applications.parsing.utils.mineru_storage import load_mineru_items
from src.utils import json_utils

router = APIRouter(prefix="/api/parsing/mineru", tags=["MinerU解析"])

@router.post(
    "/upload",
    summary="上传文件并异步触发 MinerU 解析，立即返回可查询 MinIO 解析结果的 file_id",
)
async def upload_and_parse_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="支持 .pdf/.docx/.md/.doc/.pptx/.png/.jpg/.jpeg/.txt"),
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    try:
        # 保存上传文件并启动异步解析
        file_id, in_path, original_filename = await upload_file_to_minio_temp(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    async def _do_parse_and_update(file_id: str, in_path: str, original_name: str, locale: Optional[str] = None):
        try:
            svc = MinerUParsingService()

            # 执行解析
            await svc.parse_to_items_and_save_async(
                in_path,
                file_id=file_id,
                classify_tables=True,
                original_filename=original_name,
                locale=locale,
            )
        except Exception:
            logger.exception("MinerU 后台解析失败 | file_id={}", file_id)
        finally:
            try:
                Path(in_path).unlink(missing_ok=True)
            except Exception:
                logger.warning("临时上传文件清理失败 | path={}", in_path)

    background_tasks.add_task(_do_parse_and_update, file_id, in_path, original_filename, accept_language)

    return {"success": True, "file_id": file_id}


@router.post(
    "/to-json",
    summary="上传文件并同步调用 MinerU，返回 file_id 与解析结果对象信息（推荐）",
)
async def parse_to_json(
    file: UploadFile = File(..., description="支持 .pdf / .docx / .md / .doc / .pptx / .png / .jpg / .jpeg / .txt"),
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    try:
        file_id, in_path, original_filename = await upload_file_to_minio_temp(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    try:
        svc = MinerUParsingService()

        await svc.parse_to_items_and_save_async(
            in_path,
            file_id=file_id,
            classify_tables=True,
            original_filename=original_filename,
            locale=accept_language,
        )

        return {
            "success": True,
            "file_id": file_id,
            "indexing_status": "disabled",
        }
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {e}")
    finally:
        try:
            Path(in_path).unlink(missing_ok=True)
        except Exception:
            logger.warning("临时上传文件清理失败 | path={}", in_path)


@router.get(
    "/result/{file_id}",
    summary="按 file_id 查询 MinerU 解析结果（来自 MinIO 的 converted.json 对象）；未完成则提示等待",
)
async def get_parse_result(
    file_id: str,
):
    try:
        items = await load_mineru_items(file_id)
    except FileNotFoundError:
        return {"success": False, "message": "解析未完成或不存在，请耐心等待"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取解析结果失败: {e}")

    if not items:
        return {"success": False, "message": "解析未完成或不存在，请耐心等待"}

    try:
        json_str = json_utils.dumps(items, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"序列化解析结果失败: {e}")

    return {"success": True, "json": json_str}
