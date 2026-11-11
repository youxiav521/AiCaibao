from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from regression_mvp.web_app import data
from regression_mvp.web_app import case_actions
from regression_mvp.web_app import case_edit_actions
from regression_mvp.web_app import candidate_edit_actions
from regression_mvp.web_app import case_update_actions
from regression_mvp.web_app import environment
from regression_mvp.web_app import report_actions
from regression_mvp.web_app import test_runner
from regression_mvp.storage import minio_sync
from regression_mvp.utils.artifact_labels import artifact_label
from regression_mvp.utils.diagnostics import diagnose_failure
from regression_mvp.utils.json_io import load_json
from regression_mvp.utils.reporter import recompare_report_case


WEB_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))

app = FastAPI(title="Regression MVP Dashboard")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
TEMPLATES.env.globals["file_url"] = lambda path: _file_url(str(path or ""))
TEMPLATES.env.globals["pdf_url"] = lambda path: _pdf_url(str(path or ""))
TEMPLATES.env.globals["file_exists"] = lambda path: Path(str(path or "")).is_file()
TEMPLATES.env.globals["artifact_label"] = artifact_label
TEMPLATES.env.filters["beijing_time"] = lambda value: _beijing_time(str(value or ""))


@app.on_event("startup")
def startup_minio_scan() -> None:
    minio_sync.start_background_scan("startup")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    latest = data.latest_endpoint_status()
    cases = data.all_cases(include_preview_artifacts=False)
    run_targets = [
        row for row in data.endpoint_case_summary(cases)
        if row["endpoint"] in test_runner.available_targets()
    ]
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "active": "home",
            "latest": latest,
            "run_targets": run_targets,
            "case_count": len(cases),
            "enabled_case_count": sum(1 for case in cases if case["enabled"]),
            "dashboard_path": data.dashboard_path(),
            "run_status": test_runner.current_status(),
            "environment": environment.check_environment(),
        },
    )


@app.get("/endpoints", response_class=HTMLResponse)
def endpoints(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        "endpoints.html",
        {
            "request": request,
            "active": "endpoints",
            "endpoints": data.endpoint_case_summary(),
            "run_status": test_runner.current_status(),
        },
    )


@app.get("/minio", response_class=HTMLResponse)
def minio_dashboard(
    request: Request,
    message: str = "",
    error: str = "",
    remote_cleanup_plan: bool = False,
    keep_versions: int = 10,
    local_keep_days: int = 30,
) -> HTMLResponse:
    remote_plan = None
    if remote_cleanup_plan:
        try:
            remote_plan = minio_sync.remote_version_cleanup_plan(keep_versions)
        except Exception as exc:
            diagnostic = minio_sync.diagnose_sync_error(exc)
            error = error or f"远端版本清理预览失败：{diagnostic['title']}。{diagnostic['suggestion']}"
    return TEMPLATES.TemplateResponse(
        "minio.html",
        {
            "request": request,
            "active": "minio",
            "overview": minio_sync.management_overview(),
            "remote_cleanup_plan": remote_plan,
            "local_cleanup_plan": minio_sync.local_cleanup_plan(local_keep_days),
            "cleanup_filters": {"keep_versions": keep_versions, "local_keep_days": local_keep_days},
            "message": message,
            "error": error,
        },
    )


@app.post("/minio/refresh")
def minio_refresh_all() -> RedirectResponse:
    try:
        result = minio_sync.refresh_cases_and_candidates()
    except Exception as exc:
        minio_sync.record_sync_event("refresh_all", error=exc)
        diagnostic = minio_sync.diagnose_sync_error(exc)
        return RedirectResponse(f"/minio?error={quote(diagnostic['title'] + '：' + diagnostic['suggestion'])}", status_code=303)
    minio_sync.record_sync_event("refresh_all", result=result)
    minio_sync.scan_sync_status(reason="refresh_all")
    message = f"刷新完成：下载 {result.downloaded}，跳过 {result.skipped}，冲突 {result.conflicts}"
    return RedirectResponse(f"/minio?message={quote(message)}", status_code=303)


@app.post("/minio/upload")
def minio_upload_all() -> RedirectResponse:
    try:
        result = minio_sync.upload_cases_and_candidates()
    except Exception as exc:
        minio_sync.record_sync_event("upload_all", error=exc)
        diagnostic = minio_sync.diagnose_sync_error(exc)
        return RedirectResponse(f"/minio?error={quote(diagnostic['title'] + '：' + diagnostic['suggestion'])}", status_code=303)
    minio_sync.record_sync_event("upload_all", result=result)
    minio_sync.scan_sync_status(reason="upload_all")
    message = f"上传完成：上传 {result.uploaded}，跳过 {result.skipped}，冲突 {result.conflicts}"
    return RedirectResponse(f"/minio?message={quote(message)}", status_code=303)


@app.post("/minio/scan")
def minio_scan_status() -> RedirectResponse:
    status = minio_sync.scan_sync_status(reason="manual")
    message = f"扫描完成：{status.get('message', '')}"
    return RedirectResponse(f"/minio?message={quote(message)}", status_code=303)


@app.post("/minio/cleanup/remote-plan")
async def minio_remote_cleanup_plan(request: Request) -> RedirectResponse:
    form = await request.form()
    keep_versions = _positive_int(str(form.get("keep_versions") or ""), default=10)
    return RedirectResponse(f"/minio?remote_cleanup_plan=true&keep_versions={keep_versions}", status_code=303)


@app.post("/minio/cleanup/remote-apply")
async def minio_remote_cleanup_apply(request: Request) -> RedirectResponse:
    form = await request.form()
    keep_versions = _positive_int(str(form.get("keep_versions") or ""), default=10)
    try:
        result = minio_sync.cleanup_remote_versions(keep_versions)
    except Exception as exc:
        minio_sync.record_sync_event("cleanup_remote_versions", target=f"keep={keep_versions}", error=exc)
        diagnostic = minio_sync.diagnose_sync_error(exc)
        return RedirectResponse(f"/minio?error={quote('远端版本清理失败：' + diagnostic['title'] + '。' + diagnostic['suggestion'])}", status_code=303)
    message = f"远端版本清理完成：删除对象 {result.deleted}，跳过 {result.skipped}"
    return RedirectResponse(f"/minio?message={quote(message)}", status_code=303)


@app.post("/minio/cleanup/local-plan")
async def minio_local_cleanup_plan(request: Request) -> RedirectResponse:
    form = await request.form()
    keep_days = _positive_int(str(form.get("local_keep_days") or ""), default=30)
    return RedirectResponse(f"/minio?local_keep_days={keep_days}", status_code=303)


@app.post("/minio/cleanup/local-apply")
async def minio_local_cleanup_apply(request: Request) -> RedirectResponse:
    form = await request.form()
    keep_days = _positive_int(str(form.get("local_keep_days") or ""), default=30)
    try:
        result = minio_sync.cleanup_local_archives(keep_days)
    except Exception as exc:
        minio_sync.record_sync_event("cleanup_local_archives", target=f"keep_days={keep_days}", error=exc)
        diagnostic = minio_sync.diagnose_sync_error(exc)
        return RedirectResponse(f"/minio?error={quote('本地归档清理失败：' + diagnostic['title'] + '。' + diagnostic['suggestion'])}", status_code=303)
    message = f"本地归档清理完成：删除 {result['deleted']}，失败 {result['failed']}，释放 {result['released_bytes']} B"
    return RedirectResponse(f"/minio?message={quote(message)}&local_keep_days={keep_days}", status_code=303)


@app.get("/minio/conflicts/{report_name}", response_class=HTMLResponse)
def minio_conflict_detail(request: Request, report_name: str) -> HTMLResponse:
    try:
        report = minio_sync.conflict_report_detail(report_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="conflict report not found")
    except (ValueError, minio_sync.SyncError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TEMPLATES.TemplateResponse(
        "minio_conflict_detail.html",
        {
            "request": request,
            "active": "minio",
            "report": report,
        },
    )


@app.get("/minio/events/{event_id}", response_class=HTMLResponse)
def minio_event_detail(request: Request, event_id: str) -> HTMLResponse:
    try:
        event = minio_sync.sync_event_detail(event_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="sync event not found")
    return TEMPLATES.TemplateResponse(
        "minio_event_detail.html",
        {
            "request": request,
            "active": "minio",
            "event": event,
        },
    )


@app.post("/minio/conflicts/{report_name}/{action}")
def minio_resolve_conflict(report_name: str, action: str) -> RedirectResponse:
    labels = {
        "use_remote": "采用远端",
        "force_local": "强制采用本地",
        "mark_handled": "标记已处理",
    }
    try:
        result = minio_sync.resolve_conflict_report(report_name, action)
    except Exception as exc:
        minio_sync.record_sync_event(f"resolve_conflict_{action}", target=report_name, error=exc)
        diagnostic = minio_sync.diagnose_sync_error(exc)
        return RedirectResponse(f"/minio?error={quote('冲突处理失败：' + diagnostic['title'] + '。' + diagnostic['suggestion'])}", status_code=303)
    label = labels.get(action, action)
    sync_result = result.get("result", {})
    minio_sync.record_sync_event(f"resolve_conflict_{action}", target=report_name, result=sync_result)
    minio_sync.scan_sync_status(reason=f"resolve_conflict_{action}")
    message = (
        f"{label}完成：{result.get('object_type')}/{result.get('case_id')}，"
        f"上传 {sync_result.get('uploaded', 0)}，下载 {sync_result.get('downloaded', 0)}，"
        f"跳过 {sync_result.get('skipped', 0)}"
    )
    return RedirectResponse(f"/minio?message={quote(message)}", status_code=303)


@app.post("/run/all")
def run_all_tests() -> RedirectResponse:
    test_runner.start_run("all")
    return RedirectResponse("/run", status_code=303)


@app.post("/run/selected")
async def run_selected_tests(request: Request) -> RedirectResponse:
    form = await request.form()
    targets = [str(target) for target in form.getlist("targets")]
    try:
        test_runner.start_targets(targets)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse("/run", status_code=303)


@app.post("/run/{target}")
def run_target(target: str) -> RedirectResponse:
    try:
        test_runner.start_run(target)
    except ValueError:
        raise HTTPException(status_code=404, detail="test target not found")
    return RedirectResponse("/run", status_code=303)


@app.get("/run", response_class=HTMLResponse)
def run_status(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        "run_status.html",
        {
            "request": request,
            "active": "run",
            "status": test_runner.current_status(),
            "recent_runs": test_runner.recent_runs(),
            "dashboard_path": data.dashboard_path(),
        },
    )


@app.get("/run/status")
def run_status_json() -> dict:
    return test_runner.current_status()


@app.get("/cases", response_class=HTMLResponse)
def cases(request: Request, endpoint: str = "", enabled: str = "", keyword: str = "") -> HTMLResponse:
    all_case_rows = data.all_cases()
    rows = all_case_rows
    if endpoint:
        rows = [row for row in rows if row["endpoint"] == endpoint]
    if enabled:
        status = enabled.lower()
        if status == "deleted":
            rows = [row for row in rows if row.get("deleted")]
        elif status in {"true", "enabled"}:
            rows = [row for row in rows if row["enabled"] and not row.get("deleted")]
        elif status in {"false", "disabled"}:
            rows = [row for row in rows if not row["enabled"] and not row.get("deleted")]
    else:
        rows = [row for row in rows if not row.get("deleted")]
    if keyword:
        text = keyword.lower()
        rows = [
            row for row in rows
            if text in row["case_name"].lower()
            or text in row["endpoint"].lower()
            or text in row["description"].lower()
        ]
    endpoints = sorted({row["endpoint"] for row in all_case_rows})
    return TEMPLATES.TemplateResponse(
        "cases.html",
        {
            "request": request,
            "active": "cases",
            "cases": rows,
            "endpoints": endpoints,
            "filters": {"endpoint": endpoint, "enabled": enabled, "keyword": keyword},
            "supports_visual_editor": case_edit_actions.supports_visual_editor,
            "run_status": test_runner.current_status(),
            "sync_info": minio_sync.sync_info(),
        },
    )


@app.post("/cases/sync-minio")
def sync_cases_from_minio() -> RedirectResponse:
    try:
        minio_sync.refresh_cases_and_candidates()
        minio_sync.scan_sync_status(reason="refresh_all")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MinIO sync failed: {exc}")
    return RedirectResponse("/cases", status_code=303)


@app.post("/cases/upload-minio")
def upload_cases_to_minio() -> RedirectResponse:
    try:
        minio_sync.upload_cases_and_candidates()
        minio_sync.scan_sync_status(reason="upload_all")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MinIO upload failed: {exc}")
    return RedirectResponse("/cases", status_code=303)


@app.post("/cases/{endpoint}/{case_name}/set-enabled")
async def set_case_enabled(endpoint: str, case_name: str, request: Request) -> RedirectResponse:
    form = await request.form()
    enabled = str(form.get("enabled") or "").lower() in {"1", "true", "on", "yes"}
    try:
        case_actions.set_case_enabled(endpoint, case_name, enabled)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(f"/cases?endpoint={endpoint}", status_code=303)


@app.post("/cases/{endpoint}/{case_name}/archive")
def archive_case(endpoint: str, case_name: str) -> RedirectResponse:
    try:
        case_actions.archive_case(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(f"/cases?endpoint={endpoint}", status_code=303)


@app.post("/cases/{endpoint}/{case_name}/restore")
def restore_case(endpoint: str, case_name: str) -> RedirectResponse:
    try:
        case_actions.restore_case(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(f"/cases?endpoint={endpoint}", status_code=303)


@app.post("/cases/{endpoint}/{case_name}/hard-delete")
def hard_delete_case(endpoint: str, case_name: str) -> RedirectResponse:
    try:
        case_actions.hard_delete_case(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(f"/cases?endpoint={endpoint}&enabled=deleted", status_code=303)


@app.post("/cases/{endpoint}/{case_name}/run")
def run_case(endpoint: str, case_name: str) -> RedirectResponse:
    try:
        case_actions.case_metadata(endpoint, case_name)
        test_runner.start_case(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse("/run", status_code=303)


@app.get("/cases/{endpoint}/{case_name}/runs", response_class=HTMLResponse)
def case_run_history(request: Request, endpoint: str, case_name: str) -> HTMLResponse:
    try:
        case = case_actions.case_metadata(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TEMPLATES.TemplateResponse(
        "case_runs.html",
        {
            "request": request,
            "active": "cases",
            "case": case,
            "runs": data.case_run_history(endpoint, case_name),
        },
    )


@app.get("/cases/{endpoint}/{case_name}/edit-meta", response_class=HTMLResponse)
def edit_case_metadata(request: Request, endpoint: str, case_name: str) -> HTMLResponse:
    try:
        case = case_actions.case_metadata(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TEMPLATES.TemplateResponse(
        "case_edit_meta.html",
        {
            "request": request,
            "active": "cases",
            "case": case,
            "error": "",
        },
    )


@app.post("/cases/{endpoint}/{case_name}/edit-meta", response_class=HTMLResponse)
async def save_case_metadata(request: Request, endpoint: str, case_name: str):
    form = await request.form()
    new_case_name = str(form.get("case_name") or "")
    description = str(form.get("description") or "")
    try:
        target_dir = case_actions.update_case_metadata(endpoint, case_name, new_case_name, description)
    except (ValueError, FileExistsError) as exc:
        return TEMPLATES.TemplateResponse(
            "case_edit_meta.html",
            {
                "request": request,
                "active": "cases",
                "case": {
                    "endpoint": endpoint,
                    "case_name": new_case_name,
                    "description": description,
                    "case_dir": "",
                },
                "error": str(exc),
            },
            status_code=400,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    return RedirectResponse(f"/cases?endpoint={endpoint}&keyword={target_dir.name}", status_code=303)


@app.get("/cases/{endpoint}/{case_name}/update", response_class=HTMLResponse)
def case_update(request: Request, endpoint: str, case_name: str) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        "case_update.html",
        {
            "request": request,
            "active": "cases",
            "endpoint": endpoint,
            "case_name": case_name,
            "job": case_update_actions.current_status(endpoint, case_name),
            "error": "",
            "error_diagnostic": None,
        },
    )


@app.post("/cases/{endpoint}/{case_name}/update/generate")
async def generate_case_update(endpoint: str, case_name: str, request: Request):
    form = await request.form()
    upload = form.get("upload_file")
    try:
        uploaded_path = None
        if getattr(upload, "filename", ""):
            uploaded_path = case_actions.save_uploaded_file(upload.filename, upload.file)
        case_update_actions.start_update(endpoint, case_name, str(uploaded_path) if uploaded_path else None)
    except Exception as exc:
        diagnostic = diagnose_failure(error_message=str(exc))
        return TEMPLATES.TemplateResponse(
            "case_update.html",
            {
                "request": request,
                "active": "cases",
                "endpoint": endpoint,
                "case_name": case_name,
                "job": case_update_actions.current_status(endpoint, case_name),
                "error": str(exc),
                "error_diagnostic": diagnostic.to_dict() if diagnostic else None,
            },
            status_code=400,
        )
    return RedirectResponse(f"/cases/{endpoint}/{case_name}/update", status_code=303)


@app.post("/cases/{endpoint}/{case_name}/update/apply")
def apply_case_update(endpoint: str, case_name: str) -> RedirectResponse:
    try:
        case_update_actions.apply_update(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="update candidate not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(f"/cases/{endpoint}/{case_name}/update", status_code=303)


@app.get("/cases/{endpoint}/{case_name}/edit-visual", response_class=HTMLResponse)
def edit_case_visual(request: Request, endpoint: str, case_name: str) -> HTMLResponse:
    try:
        payload = case_edit_actions.visual_editor(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case expected file not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _case_visual_response(request, payload, "")


@app.post("/cases/{endpoint}/{case_name}/edit-visual", response_class=HTMLResponse)
async def save_case_visual(request: Request, endpoint: str, case_name: str):
    form = await request.form()
    try:
        case_edit_actions.save_visual_editor(endpoint, case_name, form)
    except ValueError as exc:
        payload = case_edit_actions.visual_editor(endpoint, case_name)
        return _case_visual_response(request, payload, str(exc), status_code=400)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case expected file not found")
    return RedirectResponse(f"/cases?endpoint={endpoint}", status_code=303)


def _case_visual_response(
    request: Request,
    payload: dict,
    error: str,
    *,
    status_code: int = 200,
) -> HTMLResponse:
    context = {"request": request, "active": "cases", "error": error, **payload}
    return TEMPLATES.TemplateResponse(payload["template"], context, status_code=status_code)


@app.get("/cases/{endpoint}/{case_name}/edit-json", response_class=HTMLResponse)
def edit_case_json(request: Request, endpoint: str, case_name: str) -> HTMLResponse:
    try:
        case = case_edit_actions.case_detail(endpoint, case_name)
        json_text = case_edit_actions.expected_json_text(endpoint, case_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case expected file not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TEMPLATES.TemplateResponse(
        "candidate_edit_json.html",
        {
            "request": request,
            "active": "cases",
            "candidate": case,
            "json_text": json_text,
            "error": "",
            "title": "编辑用例标准答案 JSON",
            "cancel_url": f"/cases?endpoint={endpoint}",
        },
    )


@app.post("/cases/{endpoint}/{case_name}/edit-json", response_class=HTMLResponse)
async def save_case_json(request: Request, endpoint: str, case_name: str):
    form = await request.form()
    json_text = str(form.get("json_text") or "")
    try:
        case_edit_actions.save_expected_json(endpoint, case_name, json_text)
    except ValueError as exc:
        return TEMPLATES.TemplateResponse(
            "candidate_edit_json.html",
            {
                "request": request,
                "active": "cases",
                "candidate": case_edit_actions.case_detail(endpoint, case_name),
                "json_text": json_text,
                "error": str(exc),
                "title": "编辑用例标准答案 JSON",
                "cancel_url": f"/cases?endpoint={endpoint}",
            },
            status_code=400,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case expected file not found")
    return RedirectResponse(f"/cases?endpoint={endpoint}", status_code=303)


@app.get("/cases/{endpoint}/{case_name}/compare-exclusions/new", response_class=HTMLResponse)
def new_case_compare_exclusion(
    request: Request,
    endpoint: str,
    case_name: str,
    page: str = "",
    item_type: str = "",
    bbox: str = "",
    diff_path: str = "",
    run_id: str = "",
) -> HTMLResponse:
    try:
        payload = case_edit_actions.compare_exclusion_editor(
            endpoint,
            case_name,
            page=page,
            item_type=item_type,
            bbox_text=bbox,
            diff_path=diff_path,
            run_id=run_id,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _compare_exclusion_response(request, payload, "")


@app.post("/cases/{endpoint}/{case_name}/compare-exclusions", response_class=HTMLResponse)
async def save_case_compare_exclusion(request: Request, endpoint: str, case_name: str):
    form = await request.form()
    form_values = {key: str(value or "") for key, value in form.items()}
    try:
        case_edit_actions.save_compare_exclusion(endpoint, case_name, form_values)
        run_id = form_values.get("run_id", "").strip()
        if run_id:
            config = load_json(data.ROOT_DIR / "config.json")
            refreshed = recompare_report_case(
                data.REPORT_ROOT,
                endpoint,
                run_id,
                case_name,
                config.get("compare") or {},
            )
            if refreshed:
                report_url = (
                    f"/files/reports/{quote(endpoint, safe='')}/runs/"
                    f"{quote(run_id, safe='')}/report.html?compare_rules_updated=1"
                )
                return RedirectResponse(report_url, status_code=303)
        return RedirectResponse(f"/cases?endpoint={quote(endpoint, safe='')}", status_code=303)
    except (ValueError, FileNotFoundError) as exc:
        payload = case_edit_actions.compare_exclusion_editor(
            endpoint,
            case_name,
            page=form_values.get("page", ""),
            item_type=form_values.get("item_type", ""),
            bbox_text=form_values.get("bbox", ""),
            diff_path=form_values.get("diff_path", ""),
            run_id=form_values.get("run_id", ""),
        )
        return _compare_exclusion_response(request, payload, str(exc), status_code=400)


def _compare_exclusion_response(
    request: Request,
    payload: dict,
    error: str,
    *,
    status_code: int = 200,
) -> HTMLResponse:
    endpoint = str(payload["case"]["endpoint"])
    run_id = str(payload.get("run_id") or "")
    cancel_url = f"/cases?endpoint={quote(endpoint, safe='')}"
    if run_id:
        cancel_url = (
            f"/files/reports/{quote(endpoint, safe='')}/runs/"
            f"{quote(run_id, safe='')}/report.html"
        )
    return TEMPLATES.TemplateResponse(
        "case_compare_exclusion.html",
        {
            "request": request,
            "active": "cases",
            "error": error,
            "cancel_url": cancel_url,
            **payload,
        },
        status_code=status_code,
    )


@app.get("/case-candidates", response_class=HTMLResponse)
def case_candidates(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        "case_candidates.html",
        {
            "request": request,
            "active": "candidates",
            "candidates": data.all_candidates(),
            "endpoints": sorted(case_actions.ENDPOINTS),
            "job": case_actions.current_status(),
            "error": "",
            "error_diagnostic": None,
            "sync_info": minio_sync.sync_info(),
        },
    )


@app.post("/case-candidates/sync-minio")
def sync_candidates_from_minio() -> RedirectResponse:
    try:
        minio_sync.refresh_cases_and_candidates()
        minio_sync.scan_sync_status(reason="refresh_all")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MinIO sync failed: {exc}")
    return RedirectResponse("/case-candidates", status_code=303)


@app.post("/case-candidates/upload-minio")
def upload_candidates_to_minio() -> RedirectResponse:
    try:
        minio_sync.upload_cases_and_candidates()
        minio_sync.scan_sync_status(reason="upload_all")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MinIO upload failed: {exc}")
    return RedirectResponse("/case-candidates", status_code=303)


@app.post("/case-candidates/generate")
async def generate_case_candidate(request: Request):
    form = await request.form()
    endpoint = str(form.get("endpoint") or "")
    input_file = str(form.get("input_file") or "")
    uploads = [
        upload for upload in form.getlist("upload_file")
        if getattr(upload, "filename", "")
    ]
    case_name = str(form.get("case_name") or "")
    description = str(form.get("description") or "")
    try:
        jobs: list[tuple[str, str, str]] = []
        if uploads:
            total = len(uploads)
            for index, upload in enumerate(uploads, start=1):
                uploaded_path = case_actions.save_uploaded_file(upload.filename, upload.file)
                generated_case_name = case_actions.batch_case_name(case_name, upload.filename, index, total)
                jobs.append((str(uploaded_path), generated_case_name, description))
        else:
            generated_case_name = case_name.strip() or case_actions.batch_case_name("", Path(input_file).name, 1, 1)
            jobs.append((input_file, generated_case_name, description))
        case_actions.start_generate_batch(endpoint, jobs)
    except Exception as exc:
        diagnostic = diagnose_failure(error_message=str(exc))
        return TEMPLATES.TemplateResponse(
            "case_candidates.html",
            {
                "request": request,
                "active": "candidates",
                "candidates": data.all_candidates(),
                "endpoints": sorted(case_actions.ENDPOINTS),
                "job": case_actions.current_status(),
                "error": str(exc),
                "error_diagnostic": diagnostic.to_dict() if diagnostic else None,
                "sync_info": minio_sync.sync_info(),
            },
            status_code=400,
        )
    return RedirectResponse("/case-candidates", status_code=303)


@app.post("/case-candidates/{candidate_name}/promote")
async def promote_case_candidate(candidate_name: str, request: Request):
    form = await request.form()
    force = str(form.get("force") or "").lower() in {"1", "true", "on", "yes"}
    try:
        case_actions.promote(candidate_name, force=force)
    except Exception as exc:
        diagnostic = diagnose_failure(error_message=str(exc))
        return TEMPLATES.TemplateResponse(
            "case_candidates.html",
            {
                "request": request,
                "active": "candidates",
                "candidates": data.all_candidates(),
                "endpoints": sorted(case_actions.ENDPOINTS),
                "job": case_actions.current_status(),
                "error": str(exc),
                "error_diagnostic": diagnostic.to_dict() if diagnostic else None,
                "sync_info": minio_sync.sync_info(),
            },
            status_code=400,
        )
    return RedirectResponse("/case-candidates", status_code=303)


@app.post("/case-candidates/{candidate_name}/discard")
def discard_case_candidate(candidate_name: str) -> RedirectResponse:
    try:
        case_actions.discard_candidate(candidate_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="candidate not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse("/case-candidates", status_code=303)


@app.get("/case-candidates/{candidate_name}", response_class=HTMLResponse)
def candidate_detail(request: Request, candidate_name: str) -> HTMLResponse:
    try:
        candidate = candidate_edit_actions.candidate_detail(candidate_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="candidate not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TEMPLATES.TemplateResponse(
        "candidate_detail.html",
        {
            "request": request,
            "active": "candidates",
            "candidate": candidate,
            "error": "",
            "merge_error": "",
        },
    )


@app.get("/case-candidates/{candidate_name}/edit-json", response_class=HTMLResponse)
def edit_candidate_json(request: Request, candidate_name: str) -> HTMLResponse:
    try:
        candidate = candidate_edit_actions.candidate_detail(candidate_name)
        json_text = candidate_edit_actions.expected_json_text(candidate_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="candidate not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TEMPLATES.TemplateResponse(
        "candidate_edit_json.html",
        {
            "request": request,
            "active": "candidates",
            "candidate": candidate,
            "json_text": json_text,
            "error": "",
        },
    )


@app.post("/case-candidates/{candidate_name}/edit-json", response_class=HTMLResponse)
async def save_candidate_json(request: Request, candidate_name: str):
    form = await request.form()
    json_text = str(form.get("json_text") or "")
    try:
        candidate_edit_actions.save_expected_json(candidate_name, json_text)
    except Exception as exc:
        return TEMPLATES.TemplateResponse(
            "candidate_edit_json.html",
            {
                "request": request,
                "active": "candidates",
                "candidate": candidate_edit_actions.candidate_detail(candidate_name),
                "json_text": json_text,
                "error": str(exc),
            },
            status_code=400,
        )
    return RedirectResponse(f"/case-candidates/{candidate_name}", status_code=303)


@app.get("/case-candidates/{candidate_name}/edit-visual", response_class=HTMLResponse)
def edit_candidate_visual(request: Request, candidate_name: str) -> HTMLResponse:
    try:
        payload = candidate_edit_actions.visual_editor(candidate_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="candidate expected file not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _case_visual_response(request, payload, "")


@app.post("/case-candidates/{candidate_name}/edit-visual", response_class=HTMLResponse)
async def save_candidate_visual(request: Request, candidate_name: str):
    form = await request.form()
    try:
        candidate_edit_actions.save_visual_editor(candidate_name, form)
    except ValueError as exc:
        payload = candidate_edit_actions.visual_editor(candidate_name)
        return _case_visual_response(request, payload, str(exc), status_code=400)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="candidate expected file not found")
    return RedirectResponse(f"/case-candidates/{candidate_name}", status_code=303)


@app.post("/case-candidates/{candidate_name}/edit-meta", response_class=HTMLResponse)
async def save_candidate_metadata(request: Request, candidate_name: str):
    form = await request.form()
    try:
        candidate_edit_actions.save_candidate_metadata(
            candidate_name,
            str(form.get("case_name") or ""),
            str(form.get("description") or ""),
        )
    except (FileNotFoundError, ValueError) as exc:
        try:
            candidate = candidate_edit_actions.candidate_detail(candidate_name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="candidate not found")
        candidate["case_name"] = str(form.get("case_name") or "")
        candidate["description"] = str(form.get("description") or "")
        return TEMPLATES.TemplateResponse(
            "candidate_detail.html",
            {
                "request": request,
                "active": "candidates",
                "candidate": candidate,
                "error": str(exc),
                "merge_error": "",
            },
            status_code=400,
        )
    return RedirectResponse(f"/case-candidates/{candidate_name}", status_code=303)


@app.post("/case-candidates/{candidate_name}/merge-tables", response_class=HTMLResponse)
async def regenerate_candidate_merge_tables(request: Request, candidate_name: str):
    form = await request.form()
    try:
        candidate_edit_actions.regenerate_merge_tables(candidate_name, form)
    except (FileNotFoundError, ValueError) as exc:
        try:
            candidate = candidate_edit_actions.candidate_detail(candidate_name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="candidate not found")
        return TEMPLATES.TemplateResponse(
            "candidate_detail.html",
            {
                "request": request,
                "active": "candidates",
                "candidate": candidate,
                "error": "",
                "merge_error": str(exc),
            },
            status_code=400,
        )
    return RedirectResponse(f"/case-candidates/{candidate_name}", status_code=303)


@app.get("/case-candidates/{candidate_name}/edit-subject-match", response_class=HTMLResponse)
def edit_candidate_subject_match(request: Request, candidate_name: str) -> HTMLResponse:
    try:
        payload = candidate_edit_actions.subject_match_editor(candidate_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="candidate not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TEMPLATES.TemplateResponse(
        "candidate_edit_subject_match.html",
        {
            "request": request,
            "active": "candidates",
            "candidate": payload["candidate"],
            "rows": payload["rows"],
            "subjects": payload["subjects"],
            "error": "",
        },
    )


@app.post("/case-candidates/{candidate_name}/edit-subject-match", response_class=HTMLResponse)
async def save_candidate_subject_match(request: Request, candidate_name: str):
    form = await request.form()
    try:
        candidate_edit_actions.save_subject_match_editor(candidate_name, form)
    except Exception as exc:
        payload = candidate_edit_actions.subject_match_editor(candidate_name)
        return TEMPLATES.TemplateResponse(
            "candidate_edit_subject_match.html",
            {
                "request": request,
                "active": "candidates",
                "candidate": payload["candidate"],
                "rows": payload["rows"],
                "subjects": payload["subjects"],
                "error": str(exc),
            },
            status_code=400,
        )
    return RedirectResponse(f"/case-candidates/{candidate_name}", status_code=303)


@app.get("/reports", response_class=HTMLResponse)
def reports(request: Request, endpoint: str = "") -> HTMLResponse:
    rows = data.recent_runs(limit=200)
    if endpoint:
        rows = [row for row in rows if row["endpoint"] == endpoint]
    endpoints = sorted({row["endpoint"] for row in data.recent_runs(limit=1000)})
    aggregate_reports = [
        row for row in test_runner.recent_runs(limit=100)
        if row.get("summary_report_path") and Path(str(row["summary_report_path"])).is_file()
    ]
    response = TEMPLATES.TemplateResponse(
        "reports.html",
        {
            "request": request,
            "active": "reports",
            "reports": rows,
            "aggregate_reports": aggregate_reports,
            "endpoints": endpoints,
            "filters": {"endpoint": endpoint},
        },
    )
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/reports/{endpoint}/{run_id}", response_class=HTMLResponse)
def report_detail(request: Request, endpoint: str, run_id: str) -> HTMLResponse:
    run = data.get_run(endpoint, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    case_runs = data.case_runs_for_run(endpoint, run_id)
    report_payload = data.load_report_json(endpoint, run_id)
    return TEMPLATES.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "active": "reports",
            "run": run,
            "case_runs": case_runs,
            "report_payload": report_payload,
        },
    )


@app.post("/reports/{endpoint}/{run_id}/archive")
def archive_report(endpoint: str, run_id: str) -> RedirectResponse:
    try:
        report_actions.archive_endpoint_report(endpoint, run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="report not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _redirect_to_reports()


@app.post("/reports/aggregate/archive")
async def archive_aggregate_report(request: Request) -> RedirectResponse:
    form = await request.form()
    summary_report_path = str(form.get("summary_report_path") or "")
    try:
        report_actions.archive_aggregate_report(summary_report_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="aggregate report not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _redirect_to_reports()


def _redirect_to_reports() -> RedirectResponse:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S%f")
    response = RedirectResponse(f"/reports?refresh={stamp}", status_code=303)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/pdf/{file_path:path}", response_class=HTMLResponse)
def pdf_preview(request: Request, file_path: str) -> HTMLResponse:
    root = data.ROOT_DIR.resolve()
    target = (root / file_path).resolve()
    if not _is_under(target, root) or not target.is_file() or target.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="pdf not found")
    return TEMPLATES.TemplateResponse(
        "pdf_preview.html",
        {
            "request": request,
            "active": "",
            "filename": target.name,
            "file_path": str(target),
            "file_url": _file_url(str(target)),
        },
    )


@app.get("/files/{file_path:path}")
def files(file_path: str) -> FileResponse:
    root = data.ROOT_DIR.resolve()
    target = (root / file_path).resolve()
    if not _is_under(target, root) or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    headers = {"Cache-Control": "no-store, max-age=0"}
    if target.suffix.lower() == ".pdf":
        headers["Content-Disposition"] = f'inline; filename="{target.name}"'
        headers["X-Content-Type-Options"] = "nosniff"
    return FileResponse(target, headers=headers)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _file_url(path: str) -> str:
    if not path:
        return "#"
    try:
        target = _display_path(path)
        rel = target.relative_to(data.ROOT_DIR).as_posix()
        version = ""
        if target.is_file():
            stat = target.stat()
            version = f"?v={int(stat.st_mtime)}-{stat.st_size}"
        return f"/files/{rel}{version}"
    except Exception:
        return "#"


def _pdf_url(path: str) -> str:
    if not path:
        return "#"
    try:
        target = _display_path(path)
        if target.suffix.lower() != ".pdf":
            return _file_url(path)
        rel = target.relative_to(data.ROOT_DIR).as_posix()
        return f"/pdf/{rel}"
    except Exception:
        return "#"


def _display_path(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = data.ROOT_DIR / target
    return target.absolute()


def _beijing_time(value: str) -> str:
    if not value:
        return "-"
    text = value.strip()
    try:
        if text.endswith("Z"):
            dt = datetime.fromisoformat(text[:-1] + "+00:00")
        else:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _positive_int(value: str, *, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default
