import json
import os
import threading
import traceback
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.agents.policy.policy_retractor.schemas import Framework, Requirement
from loguru import logger

try:
    from src.agents.policy.policy_retractor.agent import PolicyRetractorAgent
    from src.agents.policy.policy_retractor.extractor import ExtractorConfig
except Exception:  # pragma: no cover
    # Fallback: bypass src.agents.policy.__init__ side effects in some envs.
    import sys

    src_dir = Path(__file__).resolve().parents[3]
    module_dir = src_dir / "agents" / "policy" / "policy_retractor"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from agent import PolicyRetractorAgent  # type: ignore
    from extractor import ExtractorConfig  # type: ignore


router = APIRouter(prefix="/api/policy/rule-extract", tags=["政策规则提取"])
DEFAULT_CALLBACK_PATH = "/api/policy/doc/modelcallback"
DEFAULT_CALLBACK_BASE_URL = "http://10.192.72.11:32223"


class RuleExtractSubmitRequest(BaseModel):
    input_path: Optional[str] = Field(default=None, description="MinerU converted json path")
    docContent: Optional[list[dict[str, Any]]] = Field(default=None, description="Java inline MinerU content")
    fileId: Optional[str] = Field(default=None, description="Upstream file id")
    callbackUrl: Optional[str] = Field(default=None, description="Java callback URL")
    output_path: Optional[str] = Field(default=None, description="Optional output json path")
    enable_llm: bool = Field(default=True)
    model_id: str = Field(default="model_qwen35_35b")
    max_section_concurrency: int = Field(default=16, ge=1)
    auto_section_concurrency: bool = Field(default=True)
    stable_max_section_concurrency: int = Field(default=16, ge=1)
    max_chars_per_section: int = Field(default=12000, ge=500)
    section_overlap_chunks: int = Field(default=1, ge=0)


class RuleExtractJob(BaseModel):
    job_id: str
    status: str
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    model_id: str
    source_type: str = "file"
    framework_id: str = ""
    file_id: str = ""
    callback_url: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None


_jobs_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _job_store_dir() -> Path:
    path = Path("storage/policy_retractor/jobs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_file(job_id: str) -> Path:
    return _job_store_dir() / f"{job_id}.json"


def _save_job(job: RuleExtractJob) -> None:
    job_path = _job_file(job.job_id)
    temp_path = job_path.with_suffix(".tmp")
    temp_path.write_text(job.model_dump_json(), encoding="utf-8")
    temp_path.replace(job_path)


def _load_job(job_id: str) -> RuleExtractJob:
    job_path = _job_file(job_id)
    if not job_path.exists():
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return RuleExtractJob.model_validate_json(job_path.read_text(encoding="utf-8"))


def _build_inline_payload(request: RuleExtractSubmitRequest) -> Optional[dict[str, Any]]:
    """Build a loader-compatible in-memory payload from Java request fields."""
    if not request.docContent:
        return None
    payload: dict[str, Any] = {"docContent": request.docContent}
    if request.fileId:
        payload["fileId"] = request.fileId
    return payload


def _resolve_request_file_id(request: RuleExtractSubmitRequest) -> str:
    """Resolve best-effort file id before extraction finishes."""
    if request.fileId:
        return request.fileId
    if request.input_path:
        return Path(request.input_path).stem
    return ""


def _resolve_request_framework_id(request: RuleExtractSubmitRequest) -> str:
    """Resolve best-effort framework id before extraction finishes."""
    file_id = _resolve_request_file_id(request)
    return f"FW-{file_id}" if file_id else ""


def _serialize_source_location(source_location: dict[str, Any]) -> dict[str, Any]:
    """Serialize source location to Java storage-friendly callback fields."""
    element_ids = source_location.get("elementIds") or []
    if isinstance(element_ids, list):
        element_ids_value = ",".join(str(item) for item in element_ids)
    else:
        element_ids_value = str(element_ids or "")
    page_no = source_location.get("pageNo")
    if page_no in ("", []):
        page_no = None
    return {
        "fileId": str(source_location.get("fileId") or ""),
        "elementIds": element_ids_value,
        "pageNo": page_no,
    }


def _serialize_framework_for_java(framework) -> dict[str, Any]:
    """Serialize framework with requirement-level tree for Java persistence."""
    payload = framework.model_dump(mode="json", by_alias=True)
    for requirement in payload.get("requirements", []):
        requirement["sourceLocation"] = _serialize_source_location(requirement.get("sourceLocation") or {})
        for control in requirement.get("controls", []):
            control["sourceLocation"] = _serialize_source_location(control.get("sourceLocation") or {})
    return payload


def _serialize_requirement_payload(
    framework_id: str,
    file_id: str,
    requirement: Requirement,
) -> dict[str, Any]:
    """Serialize one requirement as a framework-shaped payload for incremental callbacks."""
    framework = Framework(
        framework_id=framework_id,
        file_id=file_id,
        name="",
        description="",
        requirements=[requirement],
    )
    return _serialize_framework_for_java(framework)


def _build_java_event_response(
    framework_id: str,
    file_id: str,
    event: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build Java-consumable event wrapper."""
    return {
        "frameworkId": framework_id,
        "fileId": file_id,
        "event": event,
        "data": json.dumps(data or {}, ensure_ascii=False),
    }


def _resolve_callback_url(request: RuleExtractSubmitRequest) -> Optional[str]:
    """Resolve callback target. Java callback interface is /api/policy/doc/modelcallback."""
    if request.callbackUrl:
        return request.callbackUrl
    base_url = (
        os.getenv("POLICY_CALLBACK_BASE_URL")
        or os.getenv("CONTENT_GENERATE_CALLBACK_URL")
        or DEFAULT_CALLBACK_BASE_URL
    )
    return f"{base_url.rstrip('/')}{DEFAULT_CALLBACK_PATH}"


def _send_callback(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    """Send callback payload to Java. Failures are logged but do not fail the job."""
    if not callback_url:
        logger.warning("policy_retractor callback skipped: callback_url is empty")
        return
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception as exc:  # pragma: no cover
        logger.error(
            "policy_retractor callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


async def _send_callback_async(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    """Async variant used for per-requirement callbacks inside extractor execution."""
    if not callback_url:
        logger.warning("policy_retractor async callback skipped: callback_url is empty")
        return
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception as exc:  # pragma: no cover
        logger.error(
            "policy_retractor async callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


def _run_job(job_id: str, req: RuleExtractSubmitRequest) -> None:
    with _jobs_lock:
        job = _load_job(job_id)
        job.status = "RUNNING"
        job.started_at = _utc_now_iso()
        _save_job(job)
        start_payload = _build_java_event_response(
            framework_id=job.framework_id,
            file_id=job.file_id,
            event="start",
        )

    logger.info(
        "policy_retractor job started: job_id={} file_id={} callback_url={}",
        job_id,
        job.file_id,
        job.callback_url or "",
    )
    _send_callback(job.callback_url, start_payload)

    try:
        config = ExtractorConfig(
            enable_llm=req.enable_llm,
            model_id=req.model_id,
            max_section_concurrency=req.max_section_concurrency,
            auto_section_concurrency=req.auto_section_concurrency,
            stable_max_section_concurrency=req.stable_max_section_concurrency,
            max_chars_per_section=req.max_chars_per_section,
            section_overlap_chunks=req.section_overlap_chunks,
        )
        callback_url = job.callback_url
        framework_id = job.framework_id
        file_id = job.file_id

        async def on_requirement(requirement: Requirement) -> None:
            payload = _build_java_event_response(
                framework_id=requirement.framework_id or framework_id,
                file_id=file_id,
                event="message",
                data=_serialize_requirement_payload(
                    framework_id=requirement.framework_id or framework_id,
                    file_id=file_id,
                    requirement=requirement,
                ),
            )
            await _send_callback_async(callback_url, payload)

        agent = PolicyRetractorAgent(config=config, requirement_callback=on_requirement)

        output_path = Path(req.output_path) if req.output_path else None
        payload = _build_inline_payload(req)
        if payload is not None:
            framework = (
                agent.run_payload_to_file(payload, output_path)
                if output_path is not None
                else agent.run_payload(payload)
            )
        else:
            if not req.input_path:
                raise ValueError("input_path is required when docContent is not provided")
            framework = (
                agent.run_to_file(req.input_path, output_path)
                if output_path is not None
                else agent.run(req.input_path)
            )

        complete_payload = _build_java_event_response(
            framework_id=framework.framework_id,
            file_id=framework.file_id,
            event="complete",
            data=_serialize_framework_for_java(framework),
        )

        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "SUCCESS"
            job.finished_at = _utc_now_iso()
            job.framework_id = framework.framework_id
            job.file_id = framework.file_id
            job.result = complete_payload
            _save_job(job)

        _send_callback(job.callback_url, complete_payload)
    except Exception as exc:
        error_payload: dict[str, Any]
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "FAILED"
            job.finished_at = _utc_now_iso()
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            error_payload = _build_java_event_response(
                framework_id=job.framework_id,
                file_id=job.file_id,
                event="error",
                data={"message": job.error},
            )
            job.result = error_payload
            _save_job(job)
        _send_callback(job.callback_url, error_payload)


@router.post("/jobs")
def submit_rule_extract_job(request: RuleExtractSubmitRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    payload = _build_inline_payload(request)
    resolved_input_path: Optional[str] = None
    source_type = "payload" if payload is not None else "file"

    if payload is None:
        if not request.input_path:
            raise HTTPException(status_code=400, detail="input_path or docContent is required")
        input_path = Path(request.input_path)
        if not input_path.exists():
            raise HTTPException(status_code=400, detail=f"input_path not found: {request.input_path}")
        resolved_input_path = str(input_path.resolve())

    if request.output_path:
        output_path = Path(request.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid4())
    job = RuleExtractJob(
        job_id=job_id,
        status="PENDING",
        input_path=resolved_input_path,
        output_path=str(Path(request.output_path).resolve()) if request.output_path else None,
        created_at=_utc_now_iso(),
        model_id=request.model_id,
        source_type=source_type,
        framework_id=_resolve_request_framework_id(request),
        file_id=_resolve_request_file_id(request),
        callback_url=_resolve_callback_url(request),
    )
    with _jobs_lock:
        _save_job(job)

    background_tasks.add_task(_run_job, job_id, request)
    return {
        "jobId": job_id,
        **_build_java_event_response(
            framework_id=job.framework_id,
            file_id=job.file_id,
            event="start",
        ),
    }


@router.get("/jobs/{job_id}")
def get_rule_extract_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _load_job(job_id)
        return json.loads(job.model_dump_json())


@router.get("/jobs/{job_id}/result")
def get_rule_extract_result(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _load_job(job_id)
        if job.status in {"PENDING", "RUNNING"}:
            return _build_java_event_response(
                framework_id=job.framework_id,
                file_id=job.file_id,
                event="message",
            )
        return job.result or _build_java_event_response(
            framework_id=job.framework_id,
            file_id=job.file_id,
            event="error" if job.status == "FAILED" else "complete",
        )
