"""
ADE Jobs 路由

按 docs/new.md 规范提供 4 段式接口（submit/status/data/images）：
  POST /openapi/ade/jobs/submit                    — 提交解析任务
  GET  /openapi/ade/jobs/{job_id}                  — 查询任务状态
  POST /openapi/ade/jobs/{job_id}/cancel           — 取消待执行任务
  GET  /openapi/ade/jobs/{job_id}/data             — 拉取解析结果
  GET  /openapi/ade/jobs/{job_id}/images/{image_name} — 拉取图片

健康检查在 src/ade/health/router/health_router.py 中实现（路径 /openapi/ade/health）。

设计要点：
- 提交后立即返回 job_id，绝不阻塞到解析完成
- 解析在 ARQ worker 中异步执行，Parsing 侧轮询 /jobs/{job_id}
- completed 后调 /jobs/{job_id}/data 拉结果，/jobs/{job_id}/images/{name} 拉图片
- 业务错误（404/409）按 dev_plan 第 8 节返回对应 HTTP 状态码 + 信封
"""
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, Response
from loguru import logger

from src.ade.jobs.schema.jobs_schema import (
    CancelJobData,
    CancelJobResponse,
    JobDataResponse,
    JobStatus,
    JobStatusResponse,
    SubmitJobData,
    SubmitJobResponse,
)
from src.ade.jobs.service.jobs_service import AdeJobsService

router = APIRouter(prefix="/openapi/ade/jobs", tags=["ADE Jobs"])

# 异步任务 / 文件存储常量
_ADE_TASK_TYPE = "ade_jobs"
_ADE_TASK_NAME = "process_ade_jobs"
_ADE_BUCKET = os.getenv("ADE_JOBS_BUCKET", "ade-jobs")

# 上传文件大小上限（字节）。docs/new.md 第 2 节要求支持 ≥500MB；此处限制 1GB，
# 超出直接 413。storage_service.upload 当前只接受 bytes，无法流式上传，必须在
# router 层挡住超大文件避免 OOM。后续 storage_service 支持流式后可放开。
_ADE_JOBS_MAX_UPLOAD_BYTES = int(os.getenv("ADE_JOBS_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))

_jobs_service = AdeJobsService()


def _envelope(code: int, display_message: str, data: Optional[dict]) -> dict:
    """构造统一响应信封。"""
    return {"code": code, "displayMessage": display_message, "data": data}


def _json_envelope(code: int, display_message: str, data: Optional[dict], http_status: int) -> JSONResponse:
    """构造 JSONResponse，HTTP 状态码与 code 对齐（dev_plan 第 8 节）。"""
    return JSONResponse(
        status_code=http_status,
        content=_envelope(code, display_message, data),
    )


def _get_content_length(upload_file: UploadFile) -> Optional[int]:
    """从 UploadFile 头读取声明的 Content-Length；无法获取时返回 None。"""
    try:
        size = upload_file.size
        if size is not None:
            return int(size)
    except Exception:
        pass
    return None


# ── 接口 1：提交解析任务 ─────────────────────────────────────────────────────────
@router.post(
    "/submit",
    summary="提交解析任务",
    status_code=status.HTTP_200_OK,
    response_model=SubmitJobResponse,
)
async def submit_job(
    file: UploadFile = File(..., description="文件本体，流式接收"),
    file_hash: str = Form(..., description="文件 sha256，用于幂等去重"),
    external_file_id: Optional[str] = Form(
        None, description="Parsing 侧 file_id，原样透传回 metadata"
    ),
    backend: str = Form(
        "pipeline/hybrid-http-client", description="解析后端"
    ),
    ang: str = Form("ch", description="语言，默认 ch"),
    method: str = Form("auto", description="方法，默认 auto"),
    formula_enable: bool = Form(True, description="公式识别，默认 true"),
    table_enable: bool = Form(True, description="表格识别，默认 true"),
    priority: int = Form(0, description="优先级，越大越优先，默认 0"),
    callback_url: Optional[str] = Form(
        None, description="完成/失败时 POST 通知"
    ),
):
    """
    提交一个解析任务，立即返回 job_id，不阻塞解析。

    幂等：
    - 若带 file_hash 且 ADE 已有该 hash 的成功结果，直接返回指向已有结果的 job_id
      （status 立即为 completed），避免重复解析。
    - 同一 job_id 重复提交不产生新任务。

    注意：当前版本 priority 仅记录到 metadata，不实际影响 ARQ 调度顺序。
    """
    logger.info(
        "ADE jobs/submit | filename={} file_hash={} external_file_id={} backend={}",
        file.filename, file_hash, external_file_id, backend,
    )

    # 过滤 Swagger 默认占位值 "string" 与非 http(s) URL
    raw_cb = (callback_url or "").strip()
    if raw_cb.lower() in ("", "string") or not raw_cb.lower().startswith(
        ("http://", "https://")
    ):
        if raw_cb:
            logger.warning("ADE jobs/submit | 忽略非法 callback_url={!r}", callback_url)
        resolved_callback: Optional[str] = None
    else:
        resolved_callback = raw_cb

    # 基于 Content-Length 做大文件预检（避免读到一半才发现超限）
    declared_size = _get_content_length(file)
    if declared_size is not None and declared_size > _ADE_JOBS_MAX_UPLOAD_BYTES:
        logger.warning(
            "ADE jobs/submit | Content-Length 超限 | filename={} size={}B limit={}B",
            file.filename, declared_size, _ADE_JOBS_MAX_UPLOAD_BYTES,
        )
        return _json_envelope(
            413,
            f"文件超过大小限制 {_ADE_JOBS_MAX_UPLOAD_BYTES} 字节",
            None,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    try:
        job_id = await _jobs_service.submit_job(
            file_stream=file.file,
            filename=file.filename,
            content_type=file.content_type,
            file_hash=file_hash,
            external_file_id=external_file_id,
            backend=backend,
            ang=ang,
            method=method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            priority=priority,
            callback_url=resolved_callback,
            bucket=_ADE_BUCKET,
            task_name=_ADE_TASK_NAME,
            task_type=_ADE_TASK_TYPE,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("ADE jobs/submit | 提交失败")
        raise HTTPException(status_code=500, detail=f"提交失败: {exc}")

    return SubmitJobResponse(
        code=200,
        displayMessage="Operation successful",
        data=SubmitJobData(job_id=job_id),
    )


# ── 接口 2：查询任务状态 ─────────────────────────────────────────────────────────
@router.get(
    "/{job_id}",
    summary="查询任务状态",
    response_model=JobStatusResponse,
)
async def get_status(job_id: str):
    """
    查询指定任务 ID 的状态。

    - job_id 不存在返回 HTTP 404 + code=404
    - status 取值：pending / processing / completed / failed
    """
    data = await _jobs_service.get_status(job_id)
    if data is None:
        return _json_envelope(
            404, "task not found", None, status.HTTP_404_NOT_FOUND,
        )

    return JobStatusResponse(
        code=200,
        displayMessage="Operation successful",
        data=data,
    )


@router.post(
    "/{job_id}/cancel",
    summary="取消尚未执行的解析任务",
    response_model=CancelJobResponse,
)
async def cancel_job(job_id: str):
    """仅取消 pending 任务；processing/completed/failed 返回 HTTP 409。"""
    job_status = await _jobs_service.cancel_pending(job_id)
    if job_status is None:
        return _json_envelope(
            404, "task not found", None, status.HTTP_404_NOT_FOUND
        )
    data = CancelJobData(job_id=job_id, status=job_status)
    if job_status != JobStatus.cancelled:
        return _json_envelope(
            409,
            "task already started or finished",
            data.model_dump(),
            status.HTTP_409_CONFLICT,
        )
    return CancelJobResponse(
        code=200,
        displayMessage="Operation successful",
        data=data,
    )


# ── 接口 3：拉取解析结果 ─────────────────────────────────────────────────────────
@router.get(
    "/{job_id}/data",
    summary="拉取解析结果",
    response_model=JobDataResponse,
)
async def get_data(job_id: str):
    """
    拉取指定任务 ID 的解析结果。

    - 未 completed 时返回 HTTP 409 + code=409
    - job_id 不存在返回 HTTP 404 + code=404
    - 结果损坏（converted.json 下载/解析失败）返回 HTTP 500 + code=500
    - content_list.content 保持 converted.json 原结构，字段不裁剪
    """
    try:
        data, ade_status = await _jobs_service.get_data(job_id)
    except RuntimeError as exc:
        logger.exception("ADE jobs/data | 结果加载失败 | job_id={}", job_id)
        return _json_envelope(
            500, str(exc), None, status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if ade_status is None:
        return _json_envelope(
            404, "task not found", None, status.HTTP_404_NOT_FOUND,
        )
    if data is None:
        return _json_envelope(
            409, "task not completed", None, status.HTTP_409_CONFLICT,
        )

    return JobDataResponse(
        code=200,
        displayMessage="Operation successful",
        data=data,
    )


# ── 接口 4：拉取图片 ─────────────────────────────────────────────────────────────
@router.get(
    "/{job_id}/images/{image_name}",
    summary="拉取图片",
    response_class=Response,
)
async def get_image(job_id: str, image_name: str):
    """
    提供指定任务 ID 和图片名称的下载端点。

    - 返回图片二进制，Content-Type: image/*
    - 未 completed 时返回 HTTP 409 + code=409
    - job_id 不存在返回 HTTP 404 + code=404
    - 图片不存在返回 HTTP 404 + code=404
    """
    image_bytes, content_type, ade_status = await _jobs_service.get_image(
        job_id, image_name, _ADE_BUCKET
    )

    if ade_status is None:
        return _json_envelope(
            404, "task not found", None, status.HTTP_404_NOT_FOUND,
        )
    if ade_status != JobStatus.completed:
        return _json_envelope(
            409, "task not completed", None, status.HTTP_409_CONFLICT,
        )
    if image_bytes is None:
        return _json_envelope(
            404, "image not found", None, status.HTTP_404_NOT_FOUND,
        )

    return Response(
        content=image_bytes,
        media_type=content_type or "application/octet-stream",
    )
