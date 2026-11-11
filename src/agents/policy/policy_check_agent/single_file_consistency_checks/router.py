from __future__ import annotations

import asyncio
import json
import os
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from pydantic import BaseModel


from .schemas import SingleDocConsistencyCheckRequest
from .service import SingleDocConsistencyService
from loguru import logger


single_file_consistency_checks_router = APIRouter(tags=["single_file_consistency_checks"])
DEFAULT_CALLBACK_PATH = "/api/policy/check/result/modelcallback"
DEFAULT_CALLBACK_BASE_URL = "http://10.192.72.13:12223"
_jobs_lock = threading.Lock()


class SingleDocConsistencyJob(BaseModel):
    job_id: str
    check_report_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    model_id: str
    file_id: str = ""
    file_name: str = ""
    callback_url: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _job_store_dir() -> Path:
    path = Path("storage/policy_check/single_doc_consistency_jobs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_file(job_id: str) -> Path:
    return _job_store_dir() / f"{job_id}.json"


def _save_job(job: SingleDocConsistencyJob) -> None:
    job_path = _job_file(job.job_id)
    temp_path = job_path.with_suffix(".tmp")
    temp_path.write_text(job.model_dump_json(), encoding="utf-8")
    temp_path.replace(job_path)


def _load_job(job_id: str) -> SingleDocConsistencyJob:
    job_path = _job_file(job_id)
    if not job_path.exists():
        raise HTTPException(status_code=404, detail=f"single_doc_consistency job not found: {job_id}")
    return SingleDocConsistencyJob.model_validate_json(job_path.read_text(encoding="utf-8"))


def _resolve_file_id(request: SingleDocConsistencyCheckRequest) -> str:
    if request.file_id:
        return request.file_id
    if request.input_path:
        return Path(request.input_path).stem
    return ""


def _resolve_file_name(request: SingleDocConsistencyCheckRequest) -> str:
    if request.file_name:
        return request.file_name
    if request.input_path:
        return Path(request.input_path).name
    file_id = _resolve_file_id(request)
    return f"{file_id}.json" if file_id else ""


def _resolve_callback_url(request: SingleDocConsistencyCheckRequest) -> Optional[str]:
    if request.callback_url:
        return request.callback_url
    base_url = (
        os.getenv("POLICY_CALLBACK_BASE_URL")
        or os.getenv("CONTENT_GENERATE_CALLBACK_URL")
        or DEFAULT_CALLBACK_BASE_URL
    )
    return f"{base_url.rstrip('/')}{DEFAULT_CALLBACK_PATH}"


def _build_callback_payload(
    *,
    check_report_id: str,
    file_id: str,
    event: str,
    category: str = "CONSISTENCY",
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "checkReportId": check_report_id,
        "fileId": file_id,
        "event": event,
        "category": category,
        "data": data or {},
    }


def _to_callback_status(status: str) -> str:
    normalized = (status or "").lower()
    if normalized == "inconsistent":
        return "fail"
    if normalized in {"possibly_inconsistent", "insufficient_information"}:
        return "uncertain"
    return "pass"


def _to_callback_level(severity: str, status: str) -> str:
    callback_status = _to_callback_status(status)
    if callback_status == "pass":
        return "NONE"
    normalized = (severity or "").upper()
    if normalized in {"HIGH", "MEDIUM", "LOW"}:
        return normalized
    return "MEDIUM"


def _build_audited_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    file_id = str(result.get("fileId") or "")
    file_name = str(result.get("fileName") or "")
    rows: list[dict[str, Any]] = []
    for issue in result.get("issues", []) or []:
        evidence_rows = []
        for index, evidence in enumerate(issue.get("evidence", []) or [], start=1):
            location = evidence.get("sourceLocation", {}) or {}
            evidence_rows.append(
                {
                    "sequenceId": index,
                    "fileId": location.get("fileId") or file_id,
                    "elementIds": location.get("elementIds") or [],
                    "pageNo": location.get("pageNo"),
                    "segmentId": location.get("segmentId"),
                    "blockType": location.get("blockType"),
                    "role": evidence.get("role") or "",
                    "quote": evidence.get("sourceText") or "",
                    "recommendation": issue.get("reasoning") or "",
                }
            )
        rows.append(
            {
                "scope": "file",
                "category": "CONSISTENCY",
                "fileId": file_id,
                "docFileName": file_name,
                "issueId": issue.get("issueId") or "",
                "issueType": issue.get("issueType") or "",
                "controlResult": {
                    "status": _to_callback_status(str(issue.get("status") or "")),
                    "level": _to_callback_level(str(issue.get("severity") or ""), str(issue.get("status") or "")),
                    "title": issue.get("summary") or "",
                    "detail": issue.get("reasoning") or "",
                    "suggestion": issue.get("suggestion") or "",
                    "degraded": False,
                },
                "attributes": issue.get("attributes") or {},
                "evidence": evidence_rows,
            }
        )
    if not rows:
        rows.append(
            {
                "scope": "file",
                "category": "CONSISTENCY",
                "fileId": file_id,
                "docFileName": file_name,
                "issueId": "",
                "issueType": "NO_ISSUE",
                "controlResult": {
                    "status": "pass",
                    "level": "NONE",
                    "title": result.get("summary") or "未发现一致性问题",
                    "detail": result.get("summary") or "未发现一致性问题",
                    "suggestion": "",
                    "degraded": False,
                },
                "attributes": {"resultType": "summary"},
                "evidence": [],
            }
        )
    return rows


def _build_completed_summary(result: dict[str, Any], duration_seconds: float) -> dict[str, Any]:
    issues = result.get("issues", []) or []
    inconsistent = sum(1 for issue in issues if str(issue.get("status") or "").lower() == "inconsistent")
    possibly_inconsistent = sum(
        1 for issue in issues if str(issue.get("status") or "").lower() == "possibly_inconsistent"
    )
    insufficient_information = sum(
        1 for issue in issues if str(issue.get("status") or "").lower() == "insufficient_information"
    )
    return {
        "issueCount": len(issues),
        "inconsistent": inconsistent,
        "possiblyInconsistent": possibly_inconsistent,
        "insufficientInformation": insufficient_information,
        "consistent": 1 if len(issues) == 0 else 0,
        "durationSeconds": round(duration_seconds, 2),
        "description": result.get("summary") or "",
    }


def _send_callback(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("single_doc_consistency callback skipped: callback_url is empty")
        return
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
            logger.info(
                "single_doc_consistency callback sent: event={} url={} status_code={}",
                payload.get("event"),
                callback_url,
                response.status_code,
            )
    except Exception as exc:
        logger.error(
            "single_doc_consistency callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


async def _send_callback_async(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("single_doc_consistency async callback skipped: callback_url is empty")
        return
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
            logger.info(
                "single_doc_consistency async callback sent: event={} url={} status_code={}",
                payload.get("event"),
                callback_url,
                response.status_code,
            )
    except Exception as exc:
        logger.error(
            "single_doc_consistency async callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


def _run_job(job_id: str, request: SingleDocConsistencyCheckRequest) -> None:
    asyncio.run(_run_job_async(job_id, request))


async def _run_job_async(job_id: str, request: SingleDocConsistencyCheckRequest) -> None:
    with _jobs_lock:
        job = _load_job(job_id)
        job.status = "RUNNING"
        job.started_at = _utc_now_iso()
        _save_job(job)
        start_payload = _build_callback_payload(
            check_report_id=job.check_report_id,
            file_id=job.file_id,
            event="started",
            data={"checkType": "single_doc_consistency", "fileStatus": "P"},
        )

    _send_callback(job.callback_url, start_payload)
    started_at = datetime.now(UTC)
    try:
        service = SingleDocConsistencyService(
            model_id=request.model_id,
            max_candidates=request.max_candidates,
            max_judge_concurrency=request.max_judge_concurrency,
            enable_llm=request.enable_llm,
        )
        result = await service.analyze_request(request)
        result_payload = result.model_dump(mode="json", by_alias=True)
        logger.info(
            "single_doc_consistency job result ready: job_id={} file_id={} overall_status={} issue_count={} callback_url={}",
            job_id,
            result.file_id,
            result.overall_status,
            result.issue_count,
            job.callback_url,
        )
        message_payload = _build_callback_payload(
            check_report_id=job.check_report_id,
            file_id=result.file_id,
            event="audited",
            data={
                "fileId": result.file_id,
                "docFileName": result.file_name,
                "fileStatus": "F" if result.issue_count else "S",
                "results": _build_audited_results(result_payload),
                "status": "running",
                "checkType": "single_doc_consistency",
                "result": result_payload,
            },
        )
        complete_payload = _build_callback_payload(
            check_report_id=job.check_report_id,
            file_id=result.file_id,
            event="completed",
            data={
                **_build_completed_summary(result_payload, (datetime.now(UTC) - started_at).total_seconds()),
                "status": "completed",
                "checkType": "single_doc_consistency",
                "result": result_payload,
            },
        )
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "SUCCESS"
            job.finished_at = _utc_now_iso()
            job.file_id = result.file_id
            job.file_name = result.file_name
            job.result = complete_payload
            _save_job(job)
            logger.info(
                "single_doc_consistency job saved: job_id={} status={} file_id={} issue_count={}",
                job_id,
                job.status,
                job.file_id,
                result.issue_count,
            )
        await _send_callback_async(job.callback_url, message_payload)
        _send_callback(job.callback_url, complete_payload)
    except Exception as exc:
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "FAILED"
            job.finished_at = _utc_now_iso()
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            error_payload = _build_callback_payload(
                check_report_id=job.check_report_id,
                file_id=job.file_id,
                event="error",
                data={
                    "checkType": "single_doc_consistency",
                    "message": job.error,
                },
            )
            job.result = error_payload
            _save_job(job)
        _send_callback(job.callback_url, error_payload)


@single_file_consistency_checks_router.post("/single-doc/consistency/jobs")
def submit_single_doc_consistency_job(
    background_tasks: BackgroundTasks,
    request: SingleDocConsistencyCheckRequest = Body(..., description="Single document consistency check request"),
):
    check_report_id = (request.check_report_id or "").strip()
    if not check_report_id:
        raise HTTPException(status_code=400, detail="checkReportId is required")
    if not request.doc_content and not request.input_path:
        raise HTTPException(status_code=400, detail="input_path or docContent is required")
    job_id = str(uuid4())
    job = SingleDocConsistencyJob(
        job_id=job_id,
        check_report_id=check_report_id,
        status="PENDING",
        created_at=_utc_now_iso(),
        model_id=request.model_id,
        file_id=_resolve_file_id(request),
        file_name=_resolve_file_name(request),
        callback_url=_resolve_callback_url(request),
    )
    with _jobs_lock:
        _save_job(job)
    background_tasks.add_task(_run_job, job_id, request)
    return {
        "jobId": job_id,
        "checkReportId": check_report_id,
        "success": True,
        "message": "request accepted",
    }


@single_file_consistency_checks_router.get("/single-doc/consistency/jobs/{job_id}")
def get_single_doc_consistency_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        return json.loads(_load_job(job_id).model_dump_json())


@single_file_consistency_checks_router.get("/single-doc/consistency/jobs/{job_id}/result")
def get_single_doc_consistency_job_result(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _load_job(job_id)
        if job.status in {"PENDING", "RUNNING"}:
            return {
                "jobId": job.job_id,
                "checkReportId": job.check_report_id,
                "status": job.status,
                "fileId": job.file_id,
                "fileName": job.file_name,
            }
        return job.result or {
            "jobId": job.job_id,
            "checkReportId": job.check_report_id,
            "status": job.status,
            "fileId": job.file_id,
            "fileName": job.file_name,
        }
