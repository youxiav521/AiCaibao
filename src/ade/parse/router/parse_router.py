"""
ADE Parse 路由

接口列表：
  POST /openapi/ade/upload-file          — 上传文件（MinIO 持久化 + Redis 元信息）
  POST /openapi/ade/submit-parse-job     — 提交异步解析任务（入队 ARQ）
  POST /openapi/ade/parse                — 同步解析（MinerU 底层解析 → ParseResponse 输出）
  GET  /openapi/ade/parse/jobs           — 查询异步任务列表（按 task_type=ade_parse）
  GET  /openapi/ade/parse/jobs/{job_id}  — 查询异步任务状态/结果

架构设计：
  - 长耗时（>1~2min）的解析走 ARQ 队列，API 仅登记 + 立即返回 jobId
  - 上传文件落 MinIO，fileId → bucket/object 的映射存 Redis
  - 任务结果（ParseResponse）也落 MinIO，Redis 仅保存结果引用路径
  - worker 完成后可向 callback_url 实时通知
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Annotated, Any, Dict, Optional
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse
from loguru import logger

from src.ade.parse.schema import (
    AdeParseDocumentUpload,
    AdeUploadFile,
    JobStatusData,
    JobStatusResponse,
    JobSummary,
    JobsListData,
    JobsListResponse,
    ListParseJobsQuery,
    ParseJobStatus,
    ParseMetadata,
    ParseRequest,
    ParseResponse,
    ParseSyncResponse,
    ParseSyncResponseData,
    SubmitParseJobRequest,
    SubmitParseJobResponse,
    SubmitParseJobResponseData,
    UploadFileResponse,
    UploadFileResponseData,
)
from src.ade.parse.service.ade_upload_store import (
    AdeUploadRecord,
    AdeUploadStore,
)
from src.ade.parse.service.ade_parse_cache_store import (
    AdeParseCacheRecord,
    AdeParseCacheStore,
)
from src.ade.parse.service.parse_service import AdeParseService
from src.base.storage_service import storage_service
from src.base.task_manager import TaskManager, TaskRecord, TaskStatus

router = APIRouter(prefix="/openapi/ade", tags=["ADE Parse"])

# 异步任务 / 文件存储常量
_ADE_TASK_TYPE = "ade_parse"
_ADE_TASK_NAME = "process_ade_parse"
_ADE_BUCKET = os.getenv("ADE_PARSE_BUCKET", "ade-parse")

_openapi_json_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
_openapi_json_cache = None

_parse_service = AdeParseService()


# ── 状态映射 ──────────────────────────────────────────────────────────────────
# TaskManager 内部状态 → ADE 对外状态
_TASK_STATUS_TO_ADE: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "pending",
    TaskStatus.RUNNING: "processing",
    TaskStatus.SUCCESS: "completed",
    TaskStatus.FAILED: "failed",
    TaskStatus.CANCELLED: "cancelled",
}

# ADE 查询过滤状态 → TaskManager 内部状态
_ADE_STATUS_TO_TASK: dict[ParseJobStatus, TaskStatus] = {
    ParseJobStatus.pending: TaskStatus.PENDING,
    ParseJobStatus.processing: TaskStatus.RUNNING,
    ParseJobStatus.completed: TaskStatus.SUCCESS,
    ParseJobStatus.failed: TaskStatus.FAILED,
    ParseJobStatus.cancelled: TaskStatus.CANCELLED,
}


def _progress_of(status: TaskStatus) -> float:
    if status == TaskStatus.SUCCESS:
        return 1.0
    if status == TaskStatus.RUNNING:
        return 0.5
    return 0.0


def _received_at_of(created_at: str) -> int:
    try:
        return int(
            datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        )
    except Exception:
        return 0


def _load_openapi_json():
    global _openapi_json_cache
    if _openapi_json_cache is None:
        with open(_openapi_json_path, "r", encoding="utf-8") as f:
            _openapi_json_cache = json.load(f)
    return _openapi_json_cache


def _calc_sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@router.get(
    "/openapi.json",
    summary="获取 ADE Parse OpenAPI 规范",
    include_in_schema=False,
)
async def get_openapi_spec():
    """返回 ADE Parse 模块的 OpenAPI JSON 规范。"""
    return JSONResponse(content=_load_openapi_json())


def ade_parse_form_fields(
    model: Annotated[Optional[str], Form(description="解析模型版本")] = None,
    document_url: Annotated[
        Optional[str],
        Form(description="待解析文件 URL；与 document（二进制上传）二选一"),
    ] = None,
    split: Annotated[Optional[str], Form(description="分段策略，可选 page")] = None,
    password: Annotated[
        Optional[str],
        Form(description="加密文件密码（仅加密文件需要）"),
    ] = None,
) -> ParseRequest:
    """
    解析 multipart/form-data 中的非文件字段并组装 ParseRequest。

    说明：
    - 路由中的文件字段由 `document: AdeParseDocumentUpload` 单独接收。
    - 这里只负责表单字段（model/document_url/split/password）解析。
    """
    model = None if model == "string" else model
    document_url = None if document_url == "string" else document_url

    if split not in (None, "page"):
        raise HTTPException(status_code=422, detail="split 仅支持 page")
    return ParseRequest(
        model=model,
        document_url=document_url,
        split=split,
        password=password,
    )


def ade_list_parse_jobs_query(
    page: Annotated[int, Query(ge=0, description="页码（0-indexed）")] = 0,
    page_size: Annotated[int, Query(ge=1, le=100, alias="pageSize")] = 10,
    status_value: Annotated[Optional[ParseJobStatus], Query(alias="status")] = None,
) -> ListParseJobsQuery:
    """解析任务列表查询参数并组装 ListParseJobsQuery。"""
    return ListParseJobsQuery(
        page=page,
        pageSize=page_size,
        status=status_value,
    )


# ── 1) 上传文件 ────────────────────────────────────────────────────────────────

@router.post(
    "/upload-file",
    summary="上传文件（仅保存，不解析）",
    response_model=UploadFileResponse,
)
async def upload_file(
    file: AdeUploadFile,
):
    """
    上传文档文件：
      1. 落 MinIO，路径 `{bucket}/uploads/{yyyy_mm_dd}/{file_id}_{filename}`
      2. 元信息写 Redis（供 submit-parse-job 跨进程查找）

    返回 fileId，供后续 `/submit-parse-job` 使用。
    """
    file_id = str(uuid4())
    filename = file.filename or f"{file_id}.bin"
    content_type = file.content_type or "application/octet-stream"

    content = await file.read()
    size = len(content)
    file_hash = _calc_sha256_hex(content)

    date_prefix = datetime.now().strftime("%Y_%m_%d")
    object_name = f"uploads/{date_prefix}/{file_id}_{filename}"

    logger.info(
        "ADE upload-file | file_id={} filename={} size={}B bucket={} object={}",
        file_id,
        filename,
        size,
        _ADE_BUCKET,
        object_name,
    )

    try:
        await storage_service.upload(
            _ADE_BUCKET, object_name, content, content_type
        )
    except Exception as exc:
        logger.exception("ADE upload-file | 上传 MinIO 失败 | file_id={}", file_id)
        raise HTTPException(status_code=500, detail=f"上传文件失败: {exc}")

    record = AdeUploadRecord(
        file_id=file_id,
        filename=filename,
        content_type=content_type,
        size=size,
        bucket=_ADE_BUCKET,
        object_name=object_name,
        file_hash=file_hash,
        created_at=AdeUploadStore.now_iso(),
    )
    await AdeUploadStore.save(record)

    return UploadFileResponse(
        code=200,
        displayMessage="Operation successful",
        data=UploadFileResponseData(fileId=file_id),
    )


# ── 2) 提交异步解析任务 ────────────────────────────────────────────────────────

@router.post(
    "/submit-parse-job",
    summary="创建异步解析任务",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SubmitParseJobResponse,
)
async def submit_parse_job(
    payload: SubmitParseJobRequest,
):
    """
    基于 fileId 提交异步解析任务。

    - 从 Redis 查询 fileId 对应的 MinIO 文件路径
    - 生成 job_id，入队 ARQ 任务 `process_ade_parse`
    - 立即返回 202 + jobId，下游通过 /parse/jobs/{job_id} 拉取结果
    """
    logger.info(
        "ADE submit-parse-job | fileId={} split={} callback_url={}",
        payload.fileId,
        payload.split,
        payload.callback_url,
    )

    upload = await AdeUploadStore.get(payload.fileId)
    if upload is None:
        raise HTTPException(
            status_code=404,
            detail="文件不存在或已过期，请重新上传",
        )

    split_by_page = payload.split is not None and payload.split.value == "page"
    file_hash = upload.file_hash
    if not file_hash:
        try:
            file_bytes = await storage_service.download(upload.bucket, upload.object_name)
            file_hash = _calc_sha256_hex(file_bytes)
            upload.file_hash = file_hash
            await AdeUploadStore.save(upload)
        except Exception:
            logger.exception(
                "ADE submit-parse-job | 计算文件哈希失败，降级继续解析 | fileId={}",
                payload.fileId,
            )

    # 过滤 Swagger 默认占位值 "string" 以及非 http(s) 的非法 URL，避免 worker 端回调炸掉。
    raw_cb = (payload.callback_url or "").strip()
    if raw_cb.lower() in ("", "string") or not raw_cb.lower().startswith(
        ("http://", "https://")
    ):
        if raw_cb:
            logger.warning(
                "ADE submit-parse-job | 忽略非法 callback_url={!r}", payload.callback_url
            )
        callback_url: Optional[str] = None
    else:
        callback_url = raw_cb

    job_id = str(uuid4())

    if file_hash:
        cached = await AdeParseCacheStore.get(file_hash, split_by_page)
        if cached:
            cached_parse = await _load_parse_response_from_minio(cached.parse_response_path)
            if cached_parse is not None:
                await AdeParseCacheStore.touch_hit(file_hash, split_by_page)
                cached_result = {
                    "parse_response_path": cached.parse_response_path,
                    "filename": cached.filename or upload.filename,
                    "page_count": cached.page_count,
                    "duration_ms": cached.duration_ms,
                }
                metadata = {
                    "original_filename": upload.filename,
                    "file_id": payload.fileId,
                    "split_by_page": split_by_page,
                    "cache_hit": True,
                    "file_hash": file_hash,
                }
                await TaskManager.create_finished_with_job_id(
                    job_id,
                    _ADE_TASK_NAME,
                    task_type=_ADE_TASK_TYPE,
                    request={
                        "file_path": upload.file_path,
                        "filename": upload.filename,
                        "file_id": payload.fileId,
                        "split_by_page": split_by_page,
                        "callback_url": callback_url,
                        "file_hash": file_hash,
                    },
                    result=cached_result,
                    metadata=metadata,
                )
                await _notify_callback(callback_url, job_id, "completed", cached_result)
                logger.info(
                    "ADE submit-parse-job | 命中缓存直接完成(无worker) | job_id={} fileId={} hash={} split_by_page={}",
                    job_id,
                    payload.fileId,
                    file_hash,
                    split_by_page,
                )
                return SubmitParseJobResponse(
                    code=200,
                    displayMessage="Operation successful",
                    data=SubmitParseJobResponseData(jobId=job_id),
                )
            await AdeParseCacheStore.delete(file_hash, split_by_page)
    kwargs = {
        "file_path": upload.file_path,
        "filename": upload.filename,
        "file_id": payload.fileId,
        "split_by_page": split_by_page,
        "callback_url": callback_url,
        "file_hash": file_hash,
    }
    metadata = {
        "original_filename": upload.filename,
        "file_id": payload.fileId,
        "split_by_page": split_by_page,
        "file_hash": file_hash,
    }

    try:
        await TaskManager.submit_with_job_id(
            job_id,
            _ADE_TASK_NAME,
            task_type=_ADE_TASK_TYPE,
            kwargs=kwargs,
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception("ADE submit-parse-job | 入队失败 | job_id={}", job_id)
        raise HTTPException(status_code=500, detail=f"任务入队失败: {exc}")

    return SubmitParseJobResponse(
        code=200,
        displayMessage="Operation successful",
        data=SubmitParseJobResponseData(jobId=job_id),
    )


# ── 3) 同步解析 ────────────────────────────────────────────────────────────────

@router.post(
    "/parse",
    summary="同步解析文档",
    response_model=ParseSyncResponse,
)
async def parse_sync(
    form: Annotated[ParseRequest, Depends(ade_parse_form_fields)],
    document: AdeParseDocumentUpload = None,
):
    """
    同步解析文档：上传文件（或提供 document_url），通过 MinerU 解析后
    返回符合 ADE Parse 规范的 ParseResponse。
    """
    has_document = bool(document is not None and document.filename)
    has_url = bool(form.document_url and form.document_url.strip())
    logger.info(
        "ADE parse request received, has_document={}, document_url={}, model={}",
        has_document, form.document_url, form.model,
    )

    if not has_document and not has_url:
        raise HTTPException(
            status_code=422,
            detail="请提供 document（文件上传）或 document_url（文件链接），二者至少传一个",
        )

    split_by_page = form.split is not None and form.split.value == "page"

    try:
        if has_document:
            file_bytes = await document.read()
            filename = document.filename or "unknown"
        else:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(form.document_url)
                resp.raise_for_status()
            file_bytes = resp.content
            url_path = form.document_url.rsplit("?", 1)[0]
            filename = url_path.rsplit("/", 1)[-1] or "document.pdf"
        file_hash = _calc_sha256_hex(file_bytes)

        cached = await AdeParseCacheStore.get(file_hash, split_by_page)
        if cached:
            cached_parse = await _load_parse_response_from_minio(cached.parse_response_path)
            if cached_parse is not None:
                await AdeParseCacheStore.touch_hit(file_hash, split_by_page)
                logger.info(
                    "ADE parse sync | 命中缓存直接返回 | hash={} split_by_page={} path={}",
                    file_hash,
                    split_by_page,
                    cached.parse_response_path,
                )
                return ParseSyncResponse(
                    code=200,
                    displayMessage="Operation successful",
                    data=ParseSyncResponseData(
                        fileId=cached_parse.metadata.file_id or "",
                        parseResult=cached_parse,
                    ),
                )
            await AdeParseCacheStore.delete(file_hash, split_by_page)

        result = await _parse_service.parse_document(
            file_bytes=file_bytes,
            filename=filename,
            split_by_page=split_by_page,
        )
        parse_response_path = await _save_parse_response_to_minio(
            parse_response=result,
            object_name=f"results/{datetime.now().strftime('%Y_%m_%d')}/sync_{uuid4()}.json",
        )
        now_iso = AdeParseCacheStore.now_iso()
        await AdeParseCacheStore.save(
            AdeParseCacheRecord(
                file_hash=file_hash,
                split_by_page=split_by_page,
                parse_response_path=parse_response_path,
                file_id=result.metadata.file_id,
                filename=filename,
                page_count=result.metadata.page_count,
                duration_ms=result.metadata.duration_ms,
                created_at=now_iso,
                updated_at=now_iso,
            )
        )
        resolved_file_id = result.metadata.file_id or ""
        return ParseSyncResponse(
            code=200,
            displayMessage="Operation successful",
            data=ParseSyncResponseData(
                fileId=resolved_file_id,
                parseResult=result,
            ),
        )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"下载 document_url 失败: {e}",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("ADE parse failed")
        raise HTTPException(status_code=500, detail=f"解析失败: {e}")


# ── 4) 异步任务列表 ────────────────────────────────────────────────────────────

@router.get(
    "/parse/jobs",
    summary="查询异步解析任务列表",
    response_model=JobsListResponse,
)
async def list_parse_jobs(
    query: Annotated[ListParseJobsQuery, Depends(ade_list_parse_jobs_query)],
):
    """
    分页查询 ADE 异步解析任务。

    - task_type 固定为 `ade_parse`
    - 按创建时间倒序
    - 支持按 status 过滤
    """
    logger.info(
        "ADE list parse jobs | page={} pageSize={} status={}",
        query.page, query.pageSize, query.status,
    )

    limit = query.pageSize
    offset = query.page * query.pageSize

    raw_limit = limit + 1 if query.status is None else max(limit * 5, 50)

    records: list[TaskRecord] = await TaskManager.get_all(
        task_type=_ADE_TASK_TYPE, limit=raw_limit, offset=offset
    )

    if query.status is not None:
        target_status = _ADE_STATUS_TO_TASK.get(query.status)
        if target_status is None:
            records = []
        else:
            records = [r for r in records if r.status == target_status]

    has_more = len(records) > limit
    records = records[:limit]

    jobs: list[JobSummary] = []
    for record in records:
        jobs.append(
            JobSummary(
                job_id=record.job_id,
                status=_TASK_STATUS_TO_ADE.get(record.status, record.status.value),
                received_at=_received_at_of(record.created_at),
                progress=_progress_of(record.status),
                failure_reason=record.error,
            )
        )

    return JobsListResponse(
        code=200,
        displayMessage="Operation successful",
        data=JobsListData(jobs=jobs, org_id=None, has_more=has_more),
    )


# ── 5) 异步任务详情 ────────────────────────────────────────────────────────────

async def _load_parse_response_from_minio(
    parse_response_path: str,
) -> Optional[ParseResponse]:
    """从 MinIO 反序列化 ParseResponse。"""
    if "/" not in parse_response_path:
        return None
    bucket, object_name = parse_response_path.split("/", 1)
    try:
        raw = await storage_service.download(bucket, object_name)
        return ParseResponse.model_validate_json(raw)
    except Exception as exc:
        logger.exception(
            "加载 ParseResponse 失败 | path={} err={}", parse_response_path, exc
        )
        return None


async def _save_parse_response_to_minio(
    parse_response: ParseResponse,
    object_name: str,
) -> str:
    raw = parse_response.model_dump_json(by_alias=True).encode("utf-8")
    await storage_service.upload(
        _ADE_BUCKET,
        object_name,
        raw,
        "application/json",
    )
    return f"{_ADE_BUCKET}/{object_name}"


async def _notify_callback(
    url: Optional[str],
    job_id: str,
    status_value: str,
    data: Dict[str, Any],
) -> None:
    """异步任务回调（路由侧直返缓存场景使用）。"""
    if not url:
        return
    url_stripped = url.strip()
    if not url_stripped.lower().startswith(("http://", "https://")):
        return
    payload = {"jobId": job_id, "status": status_value, "data": data}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url_stripped, json=payload)
            resp.raise_for_status()
        logger.info(
            "ADE submit-parse-job 缓存命中回调成功 | job_id={} url={}",
            job_id,
            url_stripped,
        )
    except Exception as exc:
        logger.warning(
            "ADE submit-parse-job 缓存命中回调失败（已忽略） | job_id={} url={} err={}",
            job_id,
            url_stripped,
            exc,
        )


@router.get(
    "/parse/jobs/{job_id}",
    summary="查询异步解析任务状态",
    response_model=JobStatusResponse,
)
async def get_parse_job(job_id: str):
    """根据 job_id 查询任务状态。成功时附带 ParseResponse。"""
    logger.info("ADE get parse job | job_id={}", job_id)

    record = await TaskManager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    ade_status = _TASK_STATUS_TO_ADE.get(record.status, record.status.value)
    received_at = _received_at_of(record.created_at)
    progress = _progress_of(record.status)

    data: Optional[ParseResponse] = None
    output_url: Optional[str] = None
    metadata: Optional[ParseMetadata] = None

    result = record.result if isinstance(record.result, dict) else None
    if record.status == TaskStatus.SUCCESS and result:
        parse_response_path = result.get("parse_response_path")
        if parse_response_path:
            output_url = parse_response_path
            data = await _load_parse_response_from_minio(parse_response_path)

    if record.metadata:
        original_filename = record.metadata.get("original_filename") or job_id
        page_count = int(result.get("page_count", 0)) if result else 0
        duration_ms = int(result.get("duration_ms", 0)) if result else 0
        metadata = ParseMetadata(
            filename=original_filename,
            org_id=None,
            page_count=page_count,
            duration_ms=duration_ms,
            credit_usage=float(page_count),
            job_id=job_id,
            version="ade-parse",
            failed_pages=[],
        )

    return JobStatusResponse(
        code=200,
        displayMessage="Operation successful",
        data=JobStatusData(
            job_id=record.job_id,
            status=ade_status,
            received_at=received_at,
            progress=progress,
            org_id=None,
            version="ade-parse",
            parse_result=data,
            output_url=output_url,
            metadata=metadata,
            failure_reason=record.error,
        ),
    )
