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


from .schemas import SingleFileRiskSignalCheckRequest
from .service import SingleFileRiskSignalCheckService
from loguru import logger


single_file_risk_signal_checks_router = APIRouter(tags=["single_file_risk_signal_checks"])
DEFAULT_CALLBACK_PATH = "/api/policy/check/result/modelcallback"
DEFAULT_CALLBACK_BASE_URL = "http://10.192.72.13:12223"
_jobs_lock = threading.Lock()


class SingleFileRiskSignalJob(BaseModel):
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
    path = Path("storage/policy_check/single_file_risk_signal_jobs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_file(job_id: str) -> Path:
    return _job_store_dir() / f"{job_id}.json"


def _save_job(job: SingleFileRiskSignalJob) -> None:
    job_path = _job_file(job.job_id)
    temp_path = job_path.with_suffix(".tmp")
    temp_path.write_text(job.model_dump_json(), encoding="utf-8")
    temp_path.replace(job_path)


def _load_job(job_id: str) -> SingleFileRiskSignalJob:
    job_path = _job_file(job_id)
    if not job_path.exists():
        raise HTTPException(status_code=404, detail=f"single_file_risk_signal job not found: {job_id}")
    return SingleFileRiskSignalJob.model_validate_json(job_path.read_text(encoding="utf-8"))


def _resolve_callback_url(request: SingleFileRiskSignalCheckRequest) -> Optional[str]:
    if request.callback_url:
        return request.callback_url
    base_url = (
        os.getenv("POLICY_CALLBACK_BASE_URL")
        or os.getenv("CONTENT_GENERATE_CALLBACK_URL")
        or DEFAULT_CALLBACK_BASE_URL
    )
    return f"{base_url.rstrip('/')}{DEFAULT_CALLBACK_PATH}"


def _build_callback_payload(*, event: str, check_report_id: str, file_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": event,
        "checkReportId": check_report_id,
        "fileId": file_id,
        "category": "RISK_SIGNAL",
        "data": data,
    }


def _send_callback(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("single_file_risk_signal callback skipped: callback_url is empty")
        return
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
            logger.info(
                "single_file_risk_signal callback sent: event={} url={} status_code={}",
                payload.get("event"),
                callback_url,
                response.status_code,
            )
    except Exception as exc:
        logger.error(
            "single_file_risk_signal callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


async def _send_callback_async(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("single_file_risk_signal async callback skipped: callback_url is empty")
        return
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
            logger.info(
                "single_file_risk_signal async callback sent: event={} url={} status_code={}",
                payload.get("event"),
                callback_url,
                response.status_code,
            )
    except Exception as exc:
        logger.error(
            "single_file_risk_signal async callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


def _to_level(severity: str, status: str) -> str:
    if status == "pass":
        return "NONE"
    return severity.upper()


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
                    "quote": evidence.get("sourceText") or "",
                    "recommendation": issue.get("reasoning") or "",
                    "role": evidence.get("role") or "",
                }
            )
        rows.append(
            {
                "scope": "file",
                "category": "RISK_SIGNAL",
                "fileId": file_id,
                "docFileName": file_name,
                "issueType": issue.get("issueType") or "",
                "controlResult": {
                    "status": issue.get("status") or "uncertain",
                    "level": _to_level(issue.get("severity") or "low", issue.get("status") or "uncertain"),
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
                "category": "RISK_SIGNAL",
                "fileId": file_id,
                "docFileName": file_name,
                "issueType": "NO_ISSUE",
                "controlResult": {
                    "status": "pass",
                    "level": "NONE",
                    "title": result.get("summary") or "未发现风险信号",
                    "detail": result.get("summary") or "未发现风险信号",
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
    return {
        "totalIssues": len(issues),
        "high": sum(1 for issue in issues if issue.get("severity") == "high"),
        "medium": sum(1 for issue in issues if issue.get("severity") == "medium"),
        "low": sum(1 for issue in issues if issue.get("severity") == "low"),
        "passed": 0 if issues else 1,
        "failed": sum(1 for issue in issues if issue.get("status") == "fail"),
        "uncertain": sum(1 for issue in issues if issue.get("status") == "uncertain"),
        "durationSeconds": round(duration_seconds, 2),
        "description": result.get("summary") or "",
    }


def _run_job(job_id: str, request: SingleFileRiskSignalCheckRequest) -> None:
    asyncio.run(_run_job_async(job_id, request))


async def _run_job_async(job_id: str, request: SingleFileRiskSignalCheckRequest) -> None:
    with _jobs_lock:
        job = _load_job(job_id)
        job.status = "RUNNING"
        job.started_at = _utc_now_iso()
        _save_job(job)
        started_payload = _build_callback_payload(
            event="started",
            check_report_id=job.check_report_id,
            file_id=job.file_id,
            data={"fileStatus": "P"},
        )
    _send_callback(job.callback_url, started_payload)
    started_at = datetime.now(UTC)
    try:
        service = SingleFileRiskSignalCheckService(
            model_id=request.model_id,
            max_candidates=request.max_candidates,
            max_judge_concurrency=request.max_judge_concurrency,
            enable_llm=request.enable_llm,
            llm_retry_attempts=request.llm_retry_attempts,
        )
        result = await service.analyze_request(request)
        result_payload = result.model_dump(mode="json", by_alias=True)
        logger.info(
            "single_file_risk_signal job result ready: job_id={} file_id={} overall_status={} issue_count={} callback_url={}",
            job_id,
            result.file_id,
            result.overall_status,
            result.issue_count,
            job.callback_url,
        )
        audited_payload = _build_callback_payload(
            event="audited",
            check_report_id=job.check_report_id,
            file_id=result.file_id,
            data={
                "fileId": result.file_id,
                "docFileName": result.file_name,
                "fileStatus": "F" if result.issue_count else "S",
                "results": _build_audited_results(result_payload),
            },
        )
        completed_payload = _build_callback_payload(
            event="completed",
            check_report_id=job.check_report_id,
            file_id=result.file_id,
            data=_build_completed_summary(result_payload, (datetime.now(UTC) - started_at).total_seconds()),
        )
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "SUCCESS"
            job.finished_at = _utc_now_iso()
            job.file_id = result.file_id
            job.file_name = result.file_name
            job.result = completed_payload
            _save_job(job)
            logger.info(
                "single_file_risk_signal job saved: job_id={} status={} file_id={} issue_count={}",
                job_id,
                job.status,
                job.file_id,
                result.issue_count,
            )
        await _send_callback_async(job.callback_url, audited_payload)
        _send_callback(job.callback_url, completed_payload)
    except Exception as exc:
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "FAILED"
            job.finished_at = _utc_now_iso()
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            error_payload = _build_callback_payload(
                event="error",
                check_report_id=job.check_report_id,
                file_id=job.file_id,
                data={"errorMessage": job.error},
            )
            job.result = error_payload
            _save_job(job)
        _send_callback(job.callback_url, error_payload)


@single_file_risk_signal_checks_router.post("/single-file/risk/jobs")
def submit_single_file_risk_signal_job(
    background_tasks: BackgroundTasks,
    request: SingleFileRiskSignalCheckRequest = Body(..., description="Single file risk signal check request"),
):
    check_report_id = (request.check_report_id or "").strip()
    if not check_report_id:
        raise HTTPException(status_code=400, detail="checkReportId is required")
    if not request.doc_content:
        raise HTTPException(status_code=400, detail="docContent is required")
    job_id = str(uuid4())
    file_id = request.file_id or check_report_id
    file_name = request.file_name or f"{file_id}.json"
    job = SingleFileRiskSignalJob(
        job_id=job_id,
        check_report_id=check_report_id,
        status="PENDING",
        created_at=_utc_now_iso(),
        model_id=request.model_id,
        file_id=file_id,
        file_name=file_name,
        callback_url=_resolve_callback_url(request),
    )
    with _jobs_lock:
        _save_job(job)
    background_tasks.add_task(_run_job, job_id, request.model_copy(update={"file_id": file_id, "file_name": file_name}))
    return {"success": True, "checkReportId": check_report_id, "jobId": job_id, "message": "request accepted"}


@single_file_risk_signal_checks_router.get("/single-file/risk/jobs/{job_id}")
def get_single_file_risk_signal_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        return json.loads(_load_job(job_id).model_dump_json())


@single_file_risk_signal_checks_router.get("/single-file/risk/jobs/{job_id}/result")
def get_single_file_risk_signal_job_result(job_id: str) -> dict[str, Any]:
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
