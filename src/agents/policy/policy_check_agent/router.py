import asyncio
import json
import os
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from pydantic import BaseModel

from src.agents.policy.policy_check_agent.agents.agent import run_policy_check
from src.agents.policy.policy_check_agent.schemas import PolicyCheckRequest
from src.agents.policy.policy_check_agent.single_file_consistency_checks import (
    single_file_consistency_checks_router,
)
from src.agents.policy.policy_check_agent.single_file_completeness_checks import (
    single_file_completeness_checks_router,
)
from src.agents.policy.policy_check_agent.single_file_risk_signal_checks import (
    single_file_risk_signal_checks_router,
)
from src.agents.policy.policy_check_agent.single_file_policy_checks import single_file_policy_router
from loguru import logger


policy_check_router = APIRouter(prefix="/api/policy")
policy_check_router.include_router(single_file_consistency_checks_router)
policy_check_router.include_router(single_file_completeness_checks_router)
policy_check_router.include_router(single_file_risk_signal_checks_router)
policy_check_router.include_router(single_file_policy_router)
DEFAULT_CALLBACK_PATH = "/api/policy/check/result/modelcallback"
DEFAULT_CALLBACK_BASE_URL = "http://10.192.72.13:12223"
_jobs_lock = threading.Lock()


class PolicyCheckJob(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    model_id: str
    max_concurrency: int
    check_report_id: str
    resource_repository_id: str = ""
    framework_id: str = ""
    file_id: str = ""
    callback_url: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _job_store_dir() -> Path:
    path = Path("storage/policy_check/jobs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_file(job_id: str) -> Path:
    return _job_store_dir() / f"{job_id}.json"


def _save_job(job: PolicyCheckJob) -> None:
    job_path = _job_file(job.job_id)
    temp_path = job_path.with_suffix(".tmp")
    temp_path.write_text(job.model_dump_json(), encoding="utf-8")
    temp_path.replace(job_path)


def _load_job(job_id: str) -> PolicyCheckJob:
    job_path = _job_file(job_id)
    if not job_path.exists():
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return PolicyCheckJob.model_validate_json(job_path.read_text(encoding="utf-8"))


def _resolve_request_framework_id(request: PolicyCheckRequest) -> str:
    if request.frameworks:
        return str(request.frameworks[0].frameworkId or "")
    return ""


def _resolve_request_file_id(request: PolicyCheckRequest) -> str:
    if request.frameworks:
        return str(request.frameworks[0].fileId or "")
    return str(request.resource_repository.id or "")


def _resolve_callback_url(request: PolicyCheckRequest) -> Optional[str]:
    if request.callbackUrl:
        return request.callbackUrl
    base_url = (
        os.getenv("POLICY_CALLBACK_BASE_URL")
        or os.getenv("CONTENT_GENERATE_CALLBACK_URL")
        or DEFAULT_CALLBACK_BASE_URL
    )
    return f"{base_url.rstrip('/')}{DEFAULT_CALLBACK_PATH}"


def _build_java_event_response(
    framework_id: str,
    file_id: str,
    event: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "frameworkId": framework_id,
        "fileId": file_id,
        "event": event,
        "data": json.dumps(data or {}, ensure_ascii=False),
    }


def _send_callback(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("policy_check callback skipped: callback_url is empty")
        return
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception as exc:
        logger.error(
            "policy_check callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


async def _send_callback_async(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("policy_check async callback skipped: callback_url is empty")
        return
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception as exc:
        logger.error(
            "policy_check async callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


def _build_completion_summary(
    *,
    request: PolicyCheckRequest,
    control_results_by_key: Dict[Tuple[str, str, str], Dict[str, Any]],
    duration_seconds: float,
) -> Dict[str, Any]:
    total_requirements = sum(len(framework.requirements or []) for framework in request.frameworks)
    total_controls = len(control_results_by_key)
    status_counts = {
        "passed": 0,
        "failed": 0,
        "uncertain": 0,
        "skipped": 0,
    }
    for item in control_results_by_key.values():
        result = str(item.get("result") or "uncertain").lower()
        if result == "pass":
            status_counts["passed"] += 1
        elif result == "fail":
            status_counts["failed"] += 1
        elif result == "skipped":
            status_counts["skipped"] += 1
        else:
            status_counts["uncertain"] += 1
    pass_rate = 0.0 if total_controls <= 0 else (status_counts["passed"] / total_controls) * 100
    return {
        "totalRequirements": total_requirements,
        "totalControls": total_controls,
        "passed": status_counts["passed"],
        "failed": status_counts["failed"],
        "uncertain": status_counts["uncertain"],
        "skipped": status_counts["skipped"],
        "description": (
            f"Completed {total_requirements} requirements and {total_controls} controls. "
            f"Passed: {status_counts['passed']}, failed: {status_counts['failed']}, "
            f"uncertain: {status_counts['uncertain']}, skipped: {status_counts['skipped']}."
        ),
        "durationSeconds": round(duration_seconds, 2),
        "passRate": f"{pass_rate:.1f}%",
    }


def _build_framework_file_map(request: PolicyCheckRequest) -> Dict[str, str]:
    return {
        str(framework.frameworkId or ""): str(framework.fileId or "")
        for framework in request.frameworks
    }


def _sanitize_stream_result_for_callback(result: Dict[str, Any]) -> Dict[str, Any]:
    event = str(result.get("event") or "")
    if event == "requirement_result":
        control_results = []
        for item in result.get("control_results", []) or []:
            normalized = dict(item)
            normalized["degraded"] = bool(normalized.get("degraded"))
            control_results.append(normalized)
        check_results = dict(result.get("check_results", {}) or {})
        if "controlResults" in check_results:
            check_results["controlResults"] = control_results
        return {
            "requirement_id": result.get("requirement_id"),
            "requirement_title": result.get("requirement_title"),
            "framework_id": result.get("framework_id"),
            "status": result.get("status"),
            "progress": result.get("progress"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "check_results": check_results,
            "control_results": control_results,
        }
    return result


def _run_job(job_id: str, request: PolicyCheckRequest) -> None:
    asyncio.run(_run_job_async(job_id, request))


async def _run_job_async(job_id: str, request: PolicyCheckRequest) -> None:
    with _jobs_lock:
        job = _load_job(job_id)
        job.status = "RUNNING"
        job.started_at = _utc_now_iso()
        _save_job(job)
        start_payload = _build_java_event_response(
            framework_id=job.framework_id,
            file_id=job.file_id,
            event="start",
            data={"checkReportId": job.check_report_id},
        )

    logger.info(
        "policy_check job started: job_id={} report_id={} callback_url={}",
        job_id,
        job.check_report_id,
        job.callback_url or "",
    )
    _send_callback(job.callback_url, start_payload)

    framework_file_map = _build_framework_file_map(request)
    control_results_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    started_at = datetime.now(UTC)

    try:
        async for result in run_policy_check(
            resource_repository=request.resource_repository,
            frameworks=request.frameworks,
            model_id=request.model_id,
            max_concurrency=request.max_concurrency,
        ):
            framework_id = str(result.get("framework_id") or job.framework_id or "")
            file_id = framework_file_map.get(framework_id) or job.file_id

            if result.get("event") == "control_result":
                event_key = (
                    framework_id,
                    str(result.get("requirement_id") or ""),
                    str(result.get("control_id") or ""),
                )
                if event_key[2]:
                    control_results_by_key[event_key] = {
                        "result": result.get("result", "uncertain"),
                    }
                continue
            elif result.get("event") == "requirement_result":
                requirement_id = str(result.get("requirement_id") or "")
                for control_result in result.get("control_results", []) or []:
                    control_id = str(control_result.get("controlId") or "")
                    if control_id:
                        control_results_by_key[(framework_id, requirement_id, control_id)] = control_result
                callback_data = {
                    "checkReportId": job.check_report_id,
                    "status": "running",
                    "summary": _build_completion_summary(
                        request=request,
                        control_results_by_key=control_results_by_key,
                        duration_seconds=(datetime.now(UTC) - started_at).total_seconds(),
                    ),
                    "requirement": _sanitize_stream_result_for_callback(result),
                }
                message_payload = _build_java_event_response(
                    framework_id=framework_id,
                    file_id=file_id,
                    event="message",
                    data=callback_data,
                )
                await _send_callback_async(job.callback_url, message_payload)

        duration_seconds = (datetime.now(UTC) - started_at).total_seconds()
        final_data = {
            "event": "check_completed",
            "checkReportId": job.check_report_id,
            "docFolderId": job.resource_repository_id,
            "status": "completed",
            "summary": _build_completion_summary(
                request=request,
                control_results_by_key=control_results_by_key,
                duration_seconds=duration_seconds,
            ),
        }
        complete_payload = _build_java_event_response(
            framework_id=job.framework_id,
            file_id=job.file_id,
            event="complete",
            data=final_data,
        )

        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "SUCCESS"
            job.finished_at = _utc_now_iso()
            job.result = complete_payload
            _save_job(job)

        _send_callback(job.callback_url, complete_payload)
    except Exception as exc:
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "FAILED"
            job.finished_at = _utc_now_iso()
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            error_payload = _build_java_event_response(
                framework_id=job.framework_id,
                file_id=job.file_id,
                event="error",
                data={
                    "checkReportId": job.check_report_id,
                    "docFolderId": job.resource_repository_id,
                    "message": job.error,
                },
            )
            job.result = error_payload
            _save_job(job)
        _send_callback(job.callback_url, error_payload)


def _submit_policy_check_job(request: PolicyCheckRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    job_id = str(uuid4())
    check_report_id = (request.checkReportId or "").strip() or f"rpt_{uuid4().hex}"
    framework_id = _resolve_request_framework_id(request)
    file_id = _resolve_request_file_id(request)
    job = PolicyCheckJob(
        job_id=job_id,
        status="PENDING",
        created_at=_utc_now_iso(),
        model_id=request.model_id,
        max_concurrency=request.max_concurrency,
        check_report_id=check_report_id,
        resource_repository_id=str(request.resource_repository.id or ""),
        framework_id=framework_id,
        file_id=file_id,
        callback_url=_resolve_callback_url(request),
    )
    with _jobs_lock:
        _save_job(job)

    request_for_job = request.model_copy(update={"checkReportId": check_report_id})
    background_tasks.add_task(_run_job, job_id, request_for_job)
    return {
        "success": True,
        "message": "request accepted",
    }


@policy_check_router.post("/check", tags=["policy_check"])
def submit_policy_check_job(
    background_tasks: BackgroundTasks,
    request: PolicyCheckRequest = Body(..., description="Policy check request"),
):
    return _submit_policy_check_job(request, background_tasks)
