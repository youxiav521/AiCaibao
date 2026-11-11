"""
ADE 结构化提取路由

接口列表：
  POST /openapi/ade/extract/jobs    — 提交异步提取任务
  GET  /openapi/ade/extract/{jobId} — 查询任务状态
  POST /openapi/ade/extract         — 同步提取
  GET  /openapi/ade/extract/openapi.json — OpenAPI 规范
"""
import json
import os
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from src.ade.extract.schema.extract_schema import (
    ExtractRule,
    ExtractionItem,
    ModelEnum,
    QueryJobResponse,
    JobStatus,
    SubmitJobResponse,
    SyncExtractResponse,
)
from src.ade.extract.service.extract_service import ExtractService
from src.base.task_manager import TaskManager, TaskStatus

router = APIRouter(prefix="/openapi/ade/extract", tags=["ADE 结构化提取"])

# 注意：不要在模块级实例化 ExtractService。
# 它的 __init__ 会创建 VlmExtractService → ChatOpenAI，
# langchain_openai 内部会创建 _AsyncHttpxClientWrapper，其 __del__ 在应用关闭时会
# 调用 asyncio.get_running_loop().create_task(self.aclose())，导致
# "Task exception was never retrieved: RuntimeError: Event loop is closed"。
# 改为按请求懒加载，与 src/ade/extract/tasks/ade_extract_tasks.py 中的模式一致。
_extract_service: Optional[ExtractService] = None


def _get_extract_service() -> ExtractService:
    global _extract_service
    if _extract_service is None:
        _extract_service = ExtractService()
    return _extract_service

# 预加载 openapi.json
_openapi_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openapi.json")
_openapi_json_cache = None

# TaskManager 状态 → JobStatus 映射（与 parse 保持一致）
_TASK_STATUS_TO_EXTRACT: dict[TaskStatus, JobStatus] = {
    TaskStatus.PENDING: JobStatus.pending,
    TaskStatus.RUNNING: JobStatus.processing,
    TaskStatus.SUCCESS: JobStatus.completed,
    TaskStatus.FAILED: JobStatus.failed,
}


async def _load_openapi_json():
    global _openapi_json_cache
    if _openapi_json_cache is None:
        async with aiofiles.open(_openapi_json_path, "r", encoding="utf-8") as f:
            content = await f.read()
        _openapi_json_cache = json.loads(content)
    return _openapi_json_cache


# ------------------------------------------------------------------ #
# 0. OpenAPI 规范
# ------------------------------------------------------------------ #
@router.get(
    "/openapi.json",
    summary="获取 ADE Extract OpenAPI 规范",
    include_in_schema=False,
)
async def get_openapi_spec():
    """返回 ADE Extract 模块的 OpenAPI JSON 规范"""
    return JSONResponse(content=await _load_openapi_json())


def _parse_extract_rules(raw: str) -> List[ExtractRule]:
    """将 form 字段中的 JSON 字符串解析为 ExtractRule 列表"""
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"extractRule JSON 解析失败: {e}")

    if not isinstance(items, list) or len(items) == 0:
        raise HTTPException(status_code=400, detail="extractRule 必须为非空数组")

    rules: List[ExtractRule] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"extractRule[{idx}] 不是有效对象")
        if not item.get("extractName"):
            raise HTTPException(
                status_code=400,
                detail=f"extractRule[{idx}]: extractName 必填",
            )
        if "jsonSchema" not in item:
            raise HTTPException(status_code=400, detail=f"extractRule[{idx}]: jsonSchema 必填")
        rules.append(ExtractRule(**item))

    return rules


# ------------------------------------------------------------------ #
# 1. 提交异步提取任务
# ------------------------------------------------------------------ #
@router.post(
    "/jobs",
    summary="提交异步提取任务",
    response_model=SubmitJobResponse,
)
async def submit_extract_task(
    file: UploadFile = File(..., description="待提取的上传文件"),
    extractRule: str = Form(..., description="提取规则数组 JSON 字符串"),
    model: Optional[ModelEnum] = Form(None, description="模型版本: yusys_llm / yusys_vlm"),
):
    """
    上传单个文件并提交异步提取任务，返回 jobId。
    文件先上传到存储，任务队列中只传 file_path + filename + rules + model。
    """
    logger.info(f"收到异步提取请求, file={file.filename}, model={model}")

    rules = _parse_extract_rules(extractRule)
    file_bytes = await file.read()

    try:
        job_id = await _get_extract_service().submit_extract_job(
            file_bytes=file_bytes,
            filename=file.filename,
            rules=rules,
            model=model,
        )
    except Exception as exc:
        logger.exception("ADE submit-extract-job | 入队失败")
        raise HTTPException(status_code=500, detail=f"任务入队失败: {exc}")

    return SubmitJobResponse(
        code=200,
        displayMessage="Operation successful",
        jobId=job_id,
    )


# ------------------------------------------------------------------ #
# 2. 查询任务状态
# ------------------------------------------------------------------ #
@router.get(
    "/{jobId}",
    summary="查询异步提取任务状态",
    response_model=QueryJobResponse,
)
async def query_extract_task(jobId: str):
    """
    通过 jobId 查询任务状态与结果。
    """
    record = await TaskManager.get(jobId)
    if record is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {jobId}")

    job_status = _TASK_STATUS_TO_EXTRACT.get(record.status, JobStatus.pending)

    data: Optional[ExtractionItem] = None
    if record.status == TaskStatus.SUCCESS and record.result:
        try:
            result_obj = record.result
            extraction_path: Optional[str] = None
            if isinstance(result_obj, dict):
                extraction_path = result_obj.get("extraction_path")

            if extraction_path and "/" in extraction_path:
                bucket, obj_name = extraction_path.split("/", 1)
                from src.base.storage_service import storage_service
                raw_bytes = await storage_service.download(bucket, obj_name)
                data = ExtractionItem.model_validate_json(raw_bytes)
            elif isinstance(result_obj, dict) and "extraction" in result_obj:
                data = ExtractionItem(**result_obj)
        except Exception as exc:
            logger.warning("ADE query-extract-job | 结果反序列化失败 | job_id={} err={}", jobId, exc)

    return QueryJobResponse(
        code=200,
        displayMessage="Operation successful",
        jobId=jobId,
        status=job_status,
        data=data,
    )


# ------------------------------------------------------------------ #
# 3. 同步提取
# ------------------------------------------------------------------ #
@router.post(
    "",
    summary="同步结构化提取",
    response_model=SyncExtractResponse,
)
async def sync_extract(
    file: UploadFile = File(..., description="待提取的上传文件"),
    extractRule: str = Form(..., description="提取规则数组 JSON 字符串"),
    model: Optional[ModelEnum] = Form(None, description="模型版本: yusys_llm / yusys_vlm"),
):
    """
    上传单个文件，同步提取并直接返回结果。
    """
    logger.info(f"收到同步提取请求, file={file.filename}, model={model}")

    rules = _parse_extract_rules(extractRule)

    file_bytes = await file.read()
    result = await _get_extract_service().sync_extract(
        file_bytes=file_bytes,
        filename=file.filename,
        rules=rules,
        model=model,
    )
    return SyncExtractResponse(
        code=200,
        displayMessage="Operation successful",
        data=result,
    )
