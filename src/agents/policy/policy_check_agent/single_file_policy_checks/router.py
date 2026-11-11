from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import traceback
from datetime import UTC, datetime
from typing import Any, Dict, Optional
from uuid import uuid4
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from pydantic import BaseModel


from .schemas import (
    SingleFilePolicyAuditedPayload,
    SingleFilePolicyCallbackControlResult,
    SingleFilePolicyCallbackEvidence,
    SingleFilePolicyCallbackResultItem,
    SingleFilePolicyCallbackRule,
    SingleFilePolicyCheckRequest,
    SingleFilePolicyCheckResult,
    SingleFilePolicyCompletedPayload,
)
from .service import SingleFilePolicyCheckService
from loguru import logger


single_file_policy_router = APIRouter(tags=["single_file_policy_check"])
DEFAULT_CALLBACK_PATH = "/api/policy/check/result/modelcallback"
DEFAULT_CALLBACK_BASE_URL = "http://10.192.72.13:12223"
_jobs_lock = threading.Lock()
_ELEMENT_PLACEHOLDER_RE = re.compile(r"\[(?:element|elem|id)[_\-\s]*\d+\]", flags=re.IGNORECASE)


class SingleFilePolicyCheckJob(BaseModel):
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
    path = Path("storage/policy_check/single_file_policy_jobs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_file(job_id: str) -> Path:
    return _job_store_dir() / f"{job_id}.json"


def _save_job(job: SingleFilePolicyCheckJob) -> None:
    job_path = _job_file(job.job_id)
    temp_path = job_path.with_suffix(".tmp")
    temp_path.write_text(job.model_dump_json(), encoding="utf-8")
    temp_path.replace(job_path)


def _load_job(job_id: str) -> SingleFilePolicyCheckJob:
    job_path = _job_file(job_id)
    if not job_path.exists():
        raise HTTPException(status_code=404, detail=f"未找到单文档合规检查任务: {job_id}")
    return SingleFilePolicyCheckJob.model_validate_json(job_path.read_text(encoding="utf-8"))


def _resolve_callback_url(request: SingleFilePolicyCheckRequest) -> str:
    if request.callback_url:
        return request.callback_url
    base_url = (
        os.getenv("POLICY_CALLBACK_BASE_URL")
        or os.getenv("CONTENT_GENERATE_CALLBACK_URL")
        or DEFAULT_CALLBACK_BASE_URL
    )
    return f"{base_url.rstrip('/')}{DEFAULT_CALLBACK_PATH}"


def _resolve_input_file_id(check_report_id: str, index: int, file_id: Optional[str]) -> str:
    value = (file_id or "").strip()
    if value:
        return value
    return f"{check_report_id}_{index + 1}"


def _resolve_input_file_name(file_id: str, file_name: Optional[str]) -> str:
    value = (file_name or "").strip()
    if value:
        return value
    return f"{file_id}.json"


def _normalize_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _control_index(
    request: SingleFilePolicyCheckRequest,
) -> tuple[Dict[tuple[str, str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    index: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    by_control_id: Dict[str, Dict[str, Any]] = {}
    for framework in request.frameworks:
        for requirement in framework.requirements or []:
            for control in requirement.controls or []:
                meta = {
                    "framework": framework,
                    "requirement": requirement,
                    "control": control,
                }
                index[
                    (
                        _normalize_id(framework.frameworkId),
                        _normalize_id(requirement.requirementId),
                        _normalize_id(control.controlId),
                    )
                ] = meta
                by_control_id[_normalize_id(control.controlId)] = meta
    return index, by_control_id


def _resolve_control_meta(
    *,
    by_pair: Dict[tuple[str, str, str], Dict[str, Any]],
    by_control_id: Dict[str, Dict[str, Any]],
    requirement_id: str,
    control_id: str,
) -> Optional[Dict[str, Any]]:
    pair_key = ("", _normalize_id(requirement_id), _normalize_id(control_id))
    for key, meta in by_pair.items():
        if key[1] == pair_key[1] and key[2] == pair_key[2]:
            return meta
    return by_control_id.get(_normalize_id(control_id))


def _risk_level(status: str, control_risk: str) -> str:
    normalized_status = (status or "").lower()
    if normalized_status in {"pass", "skipped"}:
        return "NONE"
    risk = (control_risk or "").upper()
    return risk if risk in {"HIGH", "MEDIUM", "LOW"} else "MEDIUM"


def _build_suggestion(status: str, degraded: bool) -> str:
    normalized_status = (status or "").lower()
    if normalized_status == "fail":
        return "请补充与该规则相关的材料，并人工复核当前结论。"
    if normalized_status == "uncertain":
        return "当前证据不足，请人工补充材料后再次审核。"
    if degraded:
        return "当前结果使用了降级路径，请人工抽样复核。"
    return ""


def _sanitize_text_for_callback(text: str) -> str:
    # Remove raw placeholder tokens like [element_6]; keep normalized numeric refs appended by backend.
    cleaned = _ELEMENT_PLACEHOLDER_RE.sub("", str(text or ""))
    return " ".join(cleaned.split())


def _truncate_text(text: str, limit: int = 160) -> str:
    value = _sanitize_text_for_callback(text).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _status_label(status: str) -> str:
    normalized = (status or "").lower()
    if normalized == "pass":
        return "通过"
    if normalized == "fail":
        return "未通过"
    if normalized == "uncertain":
        return "待人工复核"
    if normalized == "skipped":
        return "已跳过"
    return normalized or "未知"


def _format_material_trace(evidence: list[Any]) -> str:
    refs: list[str] = []
    seen: set[str] = set()
    for item in evidence or []:
        payload = item.model_dump(mode="json", by_alias=True) if hasattr(item, "model_dump") else dict(item)
        file_id = str(payload.get("fileId") or "").strip()
        element_ids = [str(value) for value in (payload.get("elementIds") or []) if str(value).strip()]
        page_no = payload.get("pageNo")
        if not file_id and not element_ids:
            continue
        element_ref = "".join(f"[{eid}]" for eid in element_ids[:6])
        page_ref = f" p{int(page_no)}" if isinstance(page_no, int) else ""
        ref = f"{file_id}{page_ref}{element_ref}".strip()
        if not ref or ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    if not refs:
        return "无可用材料定位信息"
    return "；".join(refs[:3])


def _format_policy_trace(matched_meta: Optional[Dict[str, Any]], fallback_rule: SingleFilePolicyCallbackRule) -> str:
    if not matched_meta:
        return f"frameworkId={fallback_rule.framework_id}, requirementId={fallback_rule.requirement_id}, controlId={fallback_rule.control_id}"
    control = matched_meta.get("control")
    source_location = getattr(control, "sourceLocation", None)
    policy_file_id = str(getattr(source_location, "fileId", "") or "").strip()
    element_ids = [str(value) for value in (getattr(source_location, "elementIds", []) or []) if str(value).strip()]
    page_no = getattr(source_location, "pageNo", None)
    element_ref = "".join(f"[{eid}]" for eid in element_ids[:6])
    page_ref = f" p{int(page_no)}" if isinstance(page_no, int) else ""
    source_ref = f"{policy_file_id}{page_ref}{element_ref}".strip()
    if source_ref:
        return source_ref
    return (
        f"frameworkId={fallback_rule.framework_id}, requirementId={fallback_rule.requirement_id}, "
        f"controlId={fallback_rule.control_id}"
    )


def _build_detail_with_citations(reason: str, evidence: list[Any]) -> str:
    base_reason = _sanitize_text_for_callback(reason).strip()
    citation_lines: list[str] = []
    seen: set[str] = set()
    for item in evidence or []:
        payload = item.model_dump(mode="json", by_alias=True) if hasattr(item, "model_dump") else dict(item)
        quote = _sanitize_text_for_callback(payload.get("quotedText") or "").strip()
        if not quote:
            continue
        element_ids = [str(value) for value in (payload.get("elementIds") or []) if str(value).strip()]
        if not element_ids:
            continue
        # Keep callback detail concise and stable for frontend rendering.
        compact_quote = " ".join(quote.split())
        if len(compact_quote) > 120:
            compact_quote = compact_quote[:117] + "..."
        element_ref = "".join(f"[{eid}]" for eid in element_ids)
        line = f"{compact_quote}{element_ref}"
        if line in seen:
            continue
        seen.add(line)
        citation_lines.append(line)
    if not citation_lines:
        return base_reason
    if base_reason:
        return f"{base_reason}\n引述依据：\n" + "\n".join(citation_lines)
    return "引述依据：\n" + "\n".join(citation_lines)


def _build_readable_detail(
    *,
    status: str,
    reason: str,
    evidence: list[Any],
    matched_meta: Optional[Dict[str, Any]],
    rule: SingleFilePolicyCallbackRule,
) -> str:
    control_name = ""
    rule_requirement = ""
    if matched_meta:
        control = matched_meta.get("control")
        requirement = matched_meta.get("requirement")
        control_name = _truncate_text(
            str(getattr(control, "logicDescription", "") or getattr(requirement, "title", "") or rule.control_id),
            limit=64,
        )
        rule_requirement = _truncate_text(
            str(getattr(control, "sourceText", "") or getattr(requirement, "description", "") or ""),
            limit=120,
        )
    if not control_name:
        control_name = rule.control_id or "未命名规则"
    if not rule_requirement:
        rule_requirement = "请参考对应政策条款原文。"
    reason_with_citations = _build_detail_with_citations(reason, evidence)
    lines = [
        f"规则名称（{control_name}）",
        f"规则要求：{rule_requirement}",
        f"检查结果：{_status_label(status)}",
        f"判定说明：{reason_with_citations or '未提供'}",
        f"材料溯源：{_format_material_trace(evidence)}",
        f"政策溯源：{_format_policy_trace(matched_meta, rule)}",
    ]
    return "\n".join(lines)


def _build_element_page_index(request: SingleFilePolicyCheckRequest) -> dict[str, dict[int, int]]:
    index: dict[str, dict[int, int]] = {}
    for input_file in request.files or []:
        file_id = str(getattr(input_file, "file_id", "") or "")
        if not file_id:
            continue
        element_to_page: dict[int, int] = {}
        for item in list(getattr(input_file, "doc_content", []) or []):
            if not isinstance(item, dict):
                continue
            raw_element_id = item.get("element_id")
            raw_page = item.get("page")
            try:
                element_id = int(raw_element_id)
                page = int(raw_page)
            except (TypeError, ValueError):
                continue
            if element_id > 0 and page > 0:
                element_to_page[element_id] = page
        if element_to_page:
            index[file_id] = element_to_page
    return index


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        ivalue = int(value)
        return ivalue if ivalue > 0 else None
    except (TypeError, ValueError):
        return None


def _resolve_page_no_for_callback(
    *,
    payload: dict[str, Any],
    fallback_file_id: str,
    element_page_index: dict[str, dict[int, int]],
) -> Optional[int]:
    existing = _coerce_optional_int(payload.get("pageNo"))
    if existing is not None:
        return existing
    candidate_file_id = str(payload.get("fileId") or fallback_file_id or "")
    element_ids = payload.get("elementIds") or []
    if not isinstance(element_ids, list) or not candidate_file_id:
        return None
    file_pages = element_page_index.get(candidate_file_id) or {}
    for value in element_ids:
        try:
            element_id = int(value)
        except (TypeError, ValueError):
            continue
        page_no = file_pages.get(element_id)
        if page_no is not None:
            return page_no
    return None


def _normalize_evidence_for_callback(
    evidence: list[Any],
    file_id: str,
    *,
    element_page_index: dict[str, dict[int, int]],
) -> list[SingleFilePolicyCallbackEvidence]:
    normalized: list[SingleFilePolicyCallbackEvidence] = []
    for index, item in enumerate(evidence or [], start=1):
        if hasattr(item, "model_dump"):
            payload = item.model_dump(mode="json", by_alias=True)
        else:
            payload = dict(item)
        resolved_page_no = _resolve_page_no_for_callback(
            payload=payload,
            fallback_file_id=file_id,
            element_page_index=element_page_index,
        )
        element_ids = list(payload.get("elementIds") or [])
        # Callback only keeps traceable evidence positions.
        # Drop document-level placeholders that have no element position and no page.
        if not element_ids and resolved_page_no is None:
            continue
        normalized.append(
            SingleFilePolicyCallbackEvidence(
                sequence_id=len(normalized) + 1,
                file_id=str(payload.get("fileId") or file_id),
                element_ids=element_ids,
                page_no=resolved_page_no,
                recommendation=str(payload.get("summary") or payload.get("recommendation") or ""),
                quote=_sanitize_text_for_callback(payload.get("quotedText") or payload.get("summary") or ""),
            )
        )
    return normalized


def _build_audited_results(
    *,
    request: SingleFilePolicyCheckRequest,
    result: SingleFilePolicyCheckResult,
) -> list[SingleFilePolicyCallbackResultItem]:
    control_meta_by_pair, control_meta_by_control_id = _control_index(request)
    element_page_index = _build_element_page_index(request)
    # Callback only returns reviewed controls for the current file.
    # Skip unrelated controls to avoid flooding downstream with irrelevant "skipped" items.
    flattened = [item for item in result.matched_controls if (item.result or "").lower() != "skipped"]
    callback_results: list[SingleFilePolicyCallbackResultItem] = []
    for item in flattened:
        matched_meta = _resolve_control_meta(
            by_pair=control_meta_by_pair,
            by_control_id=control_meta_by_control_id,
            requirement_id=str(item.requirement_id or ""),
            control_id=str(item.control_id or ""),
        )
        framework_id = str(getattr(matched_meta["framework"], "frameworkId", "") if matched_meta else "")
        requirement_id = str(getattr(matched_meta["requirement"], "requirementId", "") if matched_meta else item.requirement_id)
        control_id = str(getattr(matched_meta["control"], "controlId", "") if matched_meta else item.control_id)
        callback_rule = SingleFilePolicyCallbackRule(
            framework_id=framework_id,
            requirement_id=requirement_id,
            control_id=control_id,
        )
        control_risk = str(getattr(matched_meta["control"], "risk", "") if matched_meta else "")
        title = ""
        if matched_meta:
            title = str(
                getattr(matched_meta["control"], "logicDescription", "")
                or getattr(matched_meta["requirement"], "title", "")
                or control_id
            )
        callback_results.append(
            SingleFilePolicyCallbackResultItem(
                file_id=result.file_id,
                doc_file_name=result.file_name,
                control_result=SingleFilePolicyCallbackControlResult(
                    status=item.result,
                    level=_risk_level(item.result, control_risk),
                    title=title,
                    detail=_build_readable_detail(
                        status=item.result,
                        reason=item.reason,
                        evidence=item.evidence,
                        matched_meta=matched_meta,
                        rule=callback_rule,
                    ),
                    suggestion=_build_suggestion(item.result, item.degraded),
                    degraded=item.degraded,
                ),
                rule=callback_rule,
                evidence=_normalize_evidence_for_callback(
                    item.evidence,
                    result.file_id,
                    element_page_index=element_page_index,
                ),
            )
        )
    return callback_results


def _build_completed_payload(result: SingleFilePolicyCheckResult, duration_seconds: float) -> SingleFilePolicyCompletedPayload:
    stats = result.statistics or {}
    total_controls = int(stats.get("totalControls", 0))
    passed = int(stats.get("passedControls", 0))
    pass_rate = 0.0 if total_controls <= 0 else (passed / total_controls) * 100
    return SingleFilePolicyCompletedPayload(
        total_requirements=int(stats.get("totalRequirements", 0)),
        total_controls=total_controls,
        passed=passed,
        failed=int(stats.get("failedControls", 0)),
        uncertain=int(stats.get("uncertainControls", 0)),
        skipped=int(stats.get("skippedControls", 0)),
        duration_seconds=round(duration_seconds, 2),
        pass_rate=f"{pass_rate:.1f}%",
        description=result.summary,
    )


def _build_callback_payload(
    *,
    event: str,
    job: SingleFilePolicyCheckJob,
    file_id: str,
    category: str = "COMPLIANCE",
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "event": event,
        "checkReportId": job.check_report_id,
        "fileId": file_id,
        "category": category,
        "data": data or {},
    }
    return payload


def _send_callback(callback_url: Optional[str], payload: dict[str, Any]) -> None:
    if not callback_url:
        logger.warning("single_file_policy_check callback skipped: callback_url is empty")
        return
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception as exc:
        logger.error(
            "single_file_policy_check callback failed: url={} error_type={} error={} payload={}",
            callback_url,
            type(exc).__name__,
            str(exc),
            payload,
        )


async def _run_job_async(job_id: str, request: SingleFilePolicyCheckRequest) -> None:
    with _jobs_lock:
        job = _load_job(job_id)
        job.status = "RUNNING"
        job.started_at = _utc_now_iso()
        _save_job(job)
    callback_url = job.callback_url
    started_at = datetime.now(UTC)
    try:
        service = SingleFilePolicyCheckService(
            model_id=request.model_id,
            max_concurrency=request.max_concurrency,
        )
        per_file_results: list[dict[str, Any]] = []
        failed_file_count = 0
        ordered_file_keys: list[tuple[str, str]] = []
        for index, input_file in enumerate(request.files):
            file_id = _resolve_input_file_id(job.check_report_id, index, input_file.file_id)
            file_name = _resolve_input_file_name(file_id, input_file.file_name)
            ordered_file_keys.append((file_id, file_name))
            started_event = _build_callback_payload(
                event="started",
                job=job,
                file_id=file_id,
                data={"fileStatus": "P"},
            )
            _send_callback(callback_url, started_event)
        hybrid_results = await service.analyze_request_hybrid(request)
        results_by_file_id: dict[str, list[SingleFilePolicyCheckResult]] = {}
        for item in hybrid_results:
            results_by_file_id.setdefault(item.file_id, []).append(item)
        for file_id, file_name in ordered_file_keys:
            result_bucket = results_by_file_id.get(file_id) or []
            result = result_bucket.pop(0) if result_bucket else None
            if result is None:
                failed_file_count += 1
                file_error = "Hybrid execution returned no result for current file."
                error_event = _build_callback_payload(
                    event="error",
                    job=job,
                    file_id=file_id,
                    data={"errorMessage": file_error},
                )
                _send_callback(callback_url, error_event)
                per_file_results.append(
                    {
                        "fileId": file_id,
                        "fileName": file_name,
                        "status": "FAILED",
                        "error": file_error,
                        "event": error_event,
                    }
                )
                continue
            audited_payload = SingleFilePolicyAuditedPayload(
                file_id=result.file_id,
                doc_file_name=result.file_name,
                file_status="F" if result.overall_status == "fail" else "S",
                results=_build_audited_results(request=request, result=result),
            ).model_dump(mode="json", by_alias=True)
            duration_seconds = (datetime.now(UTC) - started_at).total_seconds()
            completed_payload = _build_completed_payload(result, duration_seconds).model_dump(mode="json", by_alias=True)
            audited_event = _build_callback_payload(
                event="audited",
                job=job,
                file_id=result.file_id,
                data=audited_payload,
            )
            completed_event = _build_callback_payload(
                event="completed",
                job=job,
                file_id=result.file_id,
                data=completed_payload,
            )
            _send_callback(callback_url, audited_event)
            _send_callback(callback_url, completed_event)
            per_file_results.append(
                {
                    "fileId": result.file_id,
                    "fileName": result.file_name,
                    "status": "SUCCESS",
                    "audited": audited_event,
                    "completed": completed_event,
                }
            )
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "SUCCESS" if failed_file_count == 0 else "FAILED"
            job.finished_at = _utc_now_iso()
            job.result = {"files": per_file_results, "failedFileCount": failed_file_count}
            job.error = "" if failed_file_count == 0 else f"本次任务共有 {failed_file_count} 个文件处理失败。"
            _save_job(job)
    except Exception as exc:
        with _jobs_lock:
            job = _load_job(job_id)
            job.status = "FAILED"
            job.finished_at = _utc_now_iso()
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            error_event = _build_callback_payload(
                event="error",
                job=job,
                file_id=job.file_id,
                data={"errorMessage": job.error},
            )
            job.result = error_event
            _save_job(job)
        _send_callback(callback_url, error_event)


def _run_job(job_id: str, request: SingleFilePolicyCheckRequest) -> None:
    asyncio.run(_run_job_async(job_id, request))


@single_file_policy_router.post("/single-file/check/jobs")
def submit_single_file_policy_check_job(
    background_tasks: BackgroundTasks,
    request: SingleFilePolicyCheckRequest = Body(..., description="Single file policy compliance check request"),
) -> dict[str, Any]:
    check_report_id = (request.check_report_id or "").strip()
    if not check_report_id:
        raise HTTPException(status_code=400, detail="缺少必填参数 checkReportId")
    if not request.frameworks:
        raise HTTPException(status_code=400, detail="缺少必填参数 frameworks")
    if not request.files:
        raise HTTPException(status_code=400, detail="缺少必填参数 files")
    for index, input_file in enumerate(request.files):
        if not input_file.doc_content:
            raise HTTPException(status_code=400, detail=f"缺少必填参数 files[{index}].docContent")

    job_id = str(uuid4())
    first_file = request.files[0]
    file_id = _resolve_input_file_id(check_report_id, 0, first_file.file_id)
    file_name = _resolve_input_file_name(file_id, first_file.file_name)
    job = SingleFilePolicyCheckJob(
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
    background_tasks.add_task(_run_job, job_id, request)
    return {
        "success": True,
        "checkReportId": check_report_id,
        "jobId": job_id,
        "message": "request accepted",
    }


@single_file_policy_router.get("/single-file/check/jobs/{job_id}")
def get_single_file_policy_check_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        return json.loads(_load_job(job_id).model_dump_json())


@single_file_policy_router.get("/single-file/check/jobs/{job_id}/result")
def get_single_file_policy_check_job_result(job_id: str) -> dict[str, Any]:
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


def _build_readable_detail(
    *,
    status: str,
    reason: str,
    evidence: list[Any],
    matched_meta: Optional[Dict[str, Any]],
    rule: SingleFilePolicyCallbackRule,
) -> str:
    """
    Override detail formatter:
    1) 规则简要
    2) 检查结果
    3) 未通过原因（或通过依据）
    """
    control_name = ""
    rule_requirement = ""
    if matched_meta:
        control = matched_meta.get("control")
        requirement = matched_meta.get("requirement")
        control_name = _truncate_text(
            str(getattr(control, "logicDescription", "") or getattr(requirement, "title", "") or rule.control_id),
            limit=64,
        )
        rule_requirement = _truncate_text(
            str(getattr(control, "sourceText", "") or getattr(requirement, "description", "") or ""),
            limit=120,
        )
    if not control_name:
        control_name = rule.control_id or "未命名规则"
    if not rule_requirement:
        rule_requirement = "请参考对应政策条款原文。"

    if rule_requirement and control_name and control_name not in rule_requirement:
        rule_brief = _truncate_text(f"{control_name}；{rule_requirement}", limit=180)
    else:
        rule_brief = _truncate_text(rule_requirement or control_name, limit=160)
    reason_with_citations = _build_detail_with_citations(reason, evidence)
    status_label = _status_label(status)
    normalized_status = str(status or "").lower()
    if normalized_status == "pass":
        tail = f"通过依据：{reason_with_citations or '未提供额外说明。'}"
    else:
        tail = f"未通过原因：{reason_with_citations or '未提供具体原因。'}"

    return "\n".join(
        [
            rule_brief,
            f"检查结果：{status_label}",
            tail,
        ]
    )


def _build_element_refs(evidence: list[Any]) -> str:
    element_ids: list[str] = []
    seen: set[str] = set()
    for item in evidence or []:
        payload = item.model_dump(mode="json", by_alias=True) if hasattr(item, "model_dump") else dict(item)
        ids = payload.get("elementIds") or []
        if not isinstance(ids, list):
            continue
        representative_eid = ""
        for value in ids:
            eid = str(value).strip()
            if not eid:
                continue
            representative_eid = eid
            break
        if not representative_eid or representative_eid in seen:
            continue
        seen.add(representative_eid)
        element_ids.append(representative_eid)
    if not element_ids:
        return ""
    return "".join(f"[{eid}]" for eid in element_ids[:6])


def _build_detail_with_citations(reason: str, evidence: list[Any]) -> str:
    base_reason = _sanitize_text_for_callback(reason).strip()
    refs = _build_element_refs(evidence)
    if refs:
        if base_reason:
            return f"{base_reason}\n引述依据：{refs}"
        return f"引述依据：{refs}"
    return base_reason


def _build_readable_detail(
    *,
    status: str,
    reason: str,
    evidence: list[Any],
    matched_meta: Optional[Dict[str, Any]],
    rule: SingleFilePolicyCallbackRule,
) -> str:
    control_name = ""
    if matched_meta:
        control = matched_meta.get("control")
        requirement = matched_meta.get("requirement")
        control_name = _truncate_text(
            str(
                getattr(control, "content", "")
                or getattr(control, "logicDescription", "")
                or getattr(control, "sourceText", "")
                or getattr(requirement, "title", "")
                or ""
            ),
            limit=180,
        )
    if not control_name:
        control_name = "未提供规则内容"
    rule_brief = _truncate_text(control_name, limit=180)
    reason_with_citations = _build_detail_with_citations(reason, evidence)
    status_label = _status_label(status)
    normalized_status = str(status or "").lower()
    if normalized_status == "pass":
        tail = f"通过依据：{reason_with_citations or '未提供额外说明。'}"
    else:
        tail = f"未通过原因：{reason_with_citations or '未提供具体原因。'}"

    return "\n".join(
        [
            f"规则名称：{rule_brief}",
            f"检查结果：{status_label}",
            tail,
        ]
    )
