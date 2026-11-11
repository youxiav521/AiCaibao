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


from .schemas import SingleFileCompletenessCheckRequest
from .service import SingleFileCompletenessCheckService
from loguru import logger


single_file_completeness_checks_router = APIRouter(tags=["single_file_completeness_checks"])
DEFAULT_CALLBACK_PATH = "/api/policy/check/result/modelcallback"
DEFAULT_CALLBACK_BASE_URL = "http://10.192.72.13:12223"
_jobs_lock = threading.Lock()


class SingleFileCompletenessJob(BaseModel):
    job_id: str
    check_report_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    file_id: str = ""
    file_name: str = ""
    callback_url: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _job_store_dir() -> Path:
    path = Path("storage/policy_check/single_file_completeness_jobs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_file(job_id: str) -> Path:
    return _job_store_dir() / f"{job_id}.json"


def _save_job(job: SingleFileCompletenessJob) -> None:
    job_path = _job_file(job.job_id)
    temp_path = job_path.with_suffix(".tmp")
    temp_path.write_text(job.model_dump_json(), encoding="utf-8")
    temp_path.replace(job_path)


def _load_job(job_id: str) -> SingleFileCompletenessJob:
    job_path = _job_file(job_id)
    if not job_path.exists():
        raise HTTPException(status_code=404, detail=f"single_file_completeness job not found: {job_id}")
    return SingleFileCompletenessJob.model_validate_json(job_path.read_text(encoding="utf-8"))


def _resolve_callback_url(request: SingleFileCompletenessCheckRequest) -> Optional[str]:
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
        "category": "COMPLETENESS",
        "data": data,
    }


def _send_callback(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("single_file_completeness callback skipped: callback_url is empty")
        return
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception as exc:
        logger.error(
            "single_file_completeness callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


async def _send_callback_async(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("single_file_completeness async callback skipped: callback_url is empty")
        return
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception as exc:
        logger.error(
            "single_file_completeness async callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


def _to_level(status: str) -> str:
    if status == "fail":
        return "HIGH"
    if status == "uncertain":
        return "MEDIUM"
    return "NONE"


def _build_audited_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    file_id = str(result.get("fileId") or "")
    file_name = str(result.get("fileName") or "")
    rows: list[dict[str, Any]] = []
    for item in result.get("items", []) or []:
        evidence_rows = []
        for index, evidence in enumerate(item.get("evidence", []) or [], start=1):
            location = evidence.get("sourceLocation", {}) or {}
            evidence_rows.append(
                {
                    "sequenceId": index,
                    "fileId": location.get("fileId") or file_id,
                    "elementIds": location.get("elementIds") or [],
                    "pageNo": location.get("pageNo"),
                    "quote": evidence.get("sourceText") or "",
                    "recommendation": item.get("reasoning") or "",
                    "role": evidence.get("role") or "",
                }
            )
        rows.append(
            {
                "scope": "file",
                "category": "COMPLETENESS",
                "fileId": file_id,
                "docFileName": file_name,
                "checkPoint": item.get("checkPoint") or "",
                "sectionName": item.get("sectionName") or "",
                "controlResult": {
                    "status": item.get("status") or "uncertain",
                    "level": _to_level(item.get("status") or "uncertain"),
                    "title": item.get("summary") or "",
                    "detail": item.get("reasoning") or "",
                    "suggestion": "建议补充缺失章节或调整标题命名。" if item.get("status") == "fail" else "",
                    "degraded": False,
                },
                "attributes": item.get("attributes") or {},
                "evidence": evidence_rows,
            }
        )
    return rows


def _build_completed_summary(result: dict[str, Any]) -> dict[str, Any]:
    stats = result.get("statistics", {}) or {}
    return {
        "totalItems": result.get("itemCount") or 0,
        "passed": stats.get("pass", 0),
        "failed": stats.get("fail", 0),
        "uncertain": stats.get("uncertain", 0),
        "description": result.get("summary") or "",
    }


def _run_job(job_id: str, request: SingleFileCompletenessCheckRequest) -> None:
    asyncio.run(_run_job_async(job_id, request))


async def _run_job_async(job_id: str, request: SingleFileCompletenessCheckRequest) -> None:
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
    try:
        service = SingleFileCompletenessCheckService()
        result = await service.analyze_request(request)
        result_payload = result.model_dump(mode="json", by_alias=True)
        audited_payload = _build_callback_payload(
            event="audited",
            check_report_id=job.check_report_id,
            file_id=result.file_id,
            data={
                "fileId": result.file_id,
                "docFileName": result.file_name,
                "fileStatus": "F" if result.overall_status == "fail" else "S",
                "results": _build_audited_results(result_payload),
            },
        )
        completed_payload = _build_callback_payload(
            event="completed",
            check_report_id=job.check_report_id,
            file_id=result.file_id,
            data=_build_completed_summary(result_payload),
        )
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "SUCCESS"
            job.finished_at = _utc_now_iso()
            job.file_id = result.file_id
            job.file_name = result.file_name
            job.result = completed_payload
            _save_job(job)
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


@single_file_completeness_checks_router.post("/single-file/completeness/jobs")
def submit_single_file_completeness_job(
    background_tasks: BackgroundTasks,
    request: SingleFileCompletenessCheckRequest = Body(..., description="Single file completeness check request"),
):
    check_report_id = (request.check_report_id or "").strip()
    if not check_report_id:
        raise HTTPException(status_code=400, detail="checkReportId is required")
    if not request.doc_content:
        raise HTTPException(status_code=400, detail="docContent is required")
    job_id = str(uuid4())
    file_id = request.file_id or check_report_id
    file_name = request.file_name or f"{file_id}.json"
    job = SingleFileCompletenessJob(
        job_id=job_id,
        check_report_id=check_report_id,
        status="PENDING",
        created_at=_utc_now_iso(),
        file_id=file_id,
        file_name=file_name,
        callback_url=_resolve_callback_url(request),
    )
    with _jobs_lock:
        _save_job(job)
    background_tasks.add_task(_run_job, job_id, request.model_copy(update={"file_id": file_id, "file_name": file_name}))
    return {"success": True, "checkReportId": check_report_id, "jobId": job_id, "message": "request accepted"}


@single_file_completeness_checks_router.get("/single-file/completeness/jobs/{job_id}")
def get_single_file_completeness_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        return json.loads(_load_job(job_id).model_dump_json())


@single_file_completeness_checks_router.get("/single-file/completeness/jobs/{job_id}/result")
def get_single_file_completeness_job_result(job_id: str) -> dict[str, Any]:
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
