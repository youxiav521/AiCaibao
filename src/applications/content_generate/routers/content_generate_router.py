"""内容生成API路由 - 异步回调接口"""

import json
import html
import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, Body, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, ORJSONResponse
from opentelemetry import trace
from pydantic import Field
from typing import Annotated

from src.utils import json_utils
from src.utils.langfuse_tracing import inject_otel_trace_carrier
from loguru import logger

from src.agents.content_generate_agent.models import (
    Document,
    QuestionTask,
)
from src.agents.content_generate_agent.utils import (
    get_localized_message,
    resolve_answer_locale,
)
from src.base.task_manager import TaskManager
from src.applications.content_generate.services.log_redaction import redact_data_uris_for_log
from src.applications.content_generate.services.debug_image_store import DebugImageStore

router = APIRouter(prefix="/api/content_generate", tags=["内容生成"])
_DEBUG_IMAGE_STORE = DebugImageStore()


def _get_current_trace_id() -> Optional[str]:
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        return f"{span_context.trace_id:032x}"
    return None


def redact_callback_payload_for_log(payload: dict) -> dict:
    """Return a callback payload that never writes full image Base64 to logs."""
    redacted = dict(payload)
    data = redacted.get("data")
    if isinstance(data, str):
        try:
            redacted["data"] = json.dumps(redact_data_uris_for_log(json.loads(data)), ensure_ascii=False)
        except (TypeError, ValueError):
            return redact_data_uris_for_log(redacted)
    else:
        redacted = redact_data_uris_for_log(redacted)
    return redacted


@router.post("/callback")
async def content_generate_callback(
    request: Request,
    contentList: Annotated[
        List[Document],
        Field(description="文档列表：每项包含 file_name 以及 content 条目数组"),
    ] = Body(...),
    questionTasks: Annotated[
        List[QuestionTask],
        Field(description="问题任务列表"),
    ] = Body(...),
    taskId: Annotated[
        str,
        Field(description="任务ID，用于标识本次任务，用于后续结果查询"),
    ] = Body(...),
    callbackUrl: Annotated[
        Optional[str],
        Field(
            default=None,
            description="回调URL，用于debug模式，如果不提供则使用默认回调URL",
        ),
    ] = Body(None),
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    seen_ids = set()
    duplicate_ids = set()
    for task in questionTasks:
        qid = task.id
        if qid in seen_ids:
            duplicate_ids.add(qid)
        else:
            seen_ids.add(qid)
    if duplicate_ids:
        dup_sorted = sorted(duplicate_ids)
        logger.error(
            "检测到重复 question_id，拒绝提交后台任务。duplicate_ids={}", dup_sorted
        )
        return ORJSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": f"检测到重复 question_id: {dup_sorted}",
                "taskId": taskId,
            },
        )

    logger.info(
        "接收到QuestionTask taskId={}: [{}]",
        taskId, "".join(t.model_dump_json() for t in questionTasks),
    )

    try:
        context_logger = logger.bind(request_tag="context_text")
        content_list_json = json_utils.dumps(
            [doc.model_dump() for doc in contentList], ensure_ascii=False, indent=2
        )
        context_logger.info(
            "任务ID: {}\n\ncontentList (JSON):\n{}", taskId, content_list_json
        )
    except Exception as e:
        logger.error("记录contentList到日志文件时出错: {}", str(e))
        logger.exception("记录contentList到日志文件时出错")

    raw_accept_language = request.headers.get("Accept-Language")
    logger.info(
        "[i18n] content_generate_callback收到请求头: task_id={} 参数accept_language={!r} 请求头Accept-Language={!r}",
        taskId, accept_language, raw_accept_language,
    )
    answer_lang = resolve_answer_locale(accept_language)
    logger.info(
        "[i18n] content_generate_callback语言归一化完成: task_id={} 归一化后answer_lang={} raw_accept_language={!r}",
        taskId, answer_lang, accept_language,
    )
    trace_id = _get_current_trace_id()
    trace_carrier = inject_otel_trace_carrier()

    await TaskManager.submit_with_job_id(
        #不再用业务taskId做jobId
        job_id=uuid.uuid4().hex,
        task_name="process_content_generate",
        task_type="report_generate",
        kwargs={
            "content_list": [doc.model_dump() for doc in contentList],
            "question_tasks": [t.model_dump() for t in questionTasks],
            "task_id": taskId,
            "callback_url": callbackUrl,
            "answer_lang": answer_lang,
            "trace_id": trace_id,
            "trace_carrier": trace_carrier,
        },
        metadata={
            "business_task_id": taskId,
            "trace_id": trace_id,
        },
    )

    return ORJSONResponse(
        content={
            "success": True,
            "message": get_localized_message("api_task_submitted", answer_lang),
            "taskId": taskId,
        }
    )


@router.post("/callback/receive")
async def receive_callback(request: Request):
    """
    接收回调消息并打印出来
    """
    try:
        body = await request.json()
        task_id = body.get("taskId", "")
        logger.info("callback: {} ", redact_callback_payload_for_log(body))

        interrupt = body.get("interrupt", False)

        return ORJSONResponse(
            content={
                "data": {
                    "taskId": task_id,
                    "interrupt": interrupt,
                }
            }
        )
    except Exception:
        logger.exception("接收回调消息时出错")
        return ORJSONResponse(
            status_code=400,
            content={
                "data": {
                    "taskId": "",
                    "interrupt": False,
                }
            }
        )


@router.post("/debug/image-callback", include_in_schema=False)
async def receive_debug_image_callback(request: Request):
    """Local development callback that stores an image only in API memory."""
    _require_local_debug_access(request)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid callback JSON") from exc

    task_id = str(body.get("taskId") or "").strip()
    stored = _DEBUG_IMAGE_STORE.store_callback(body)
    logger.info("本地图片调试回调 | task_id={} stored={}", task_id, stored)
    return ORJSONResponse(
        content={
            "data": {
                "taskId": task_id,
                "interrupt": False,
                "viewUrl": f"/api/content_generate/debug/images/{task_id}" if stored else None,
            }
        }
    )


@router.get("/debug/images/{task_id}", include_in_schema=False)
async def view_debug_image(task_id: str, request: Request):
    """Render one locally cached callback image in a browser."""
    _require_local_debug_access(request)
    image_data_uri = _DEBUG_IMAGE_STORE.get_image(task_id)
    if image_data_uri is None:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    safe_task_id = html.escape(task_id)
    safe_data_uri = html.escape(image_data_uri, quote=True)
    return HTMLResponse(
        content=(
            "<!doctype html><html><head><meta charset='utf-8'><title>Image Debug</title>"
            "<style>body{margin:24px;font-family:Arial,sans-serif}img{max-width:100%;height:auto;"
            "border:1px solid #ddd}</style></head><body>"
            f"<h3>任务图片：{safe_task_id}</h3><img src='{safe_data_uri}'></body></html>"
        )
    )


def _require_local_debug_access(request: Request) -> None:
    enabled = os.getenv("CONTENT_GENERATE_DEBUG_CALLBACK_ENABLED", "false").strip().lower()
    client_host = request.client.host if request.client else ""
    if enabled not in {"1", "true", "yes"} or client_host not in {"127.0.0.1", "::1"}:
        raise HTTPException(status_code=404, detail="Not found")
