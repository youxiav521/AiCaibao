from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from regression_mvp.utils.json_io import dump_json, load_json
from regression_mvp.utils.artifact_renderer import render_content_preview, render_overlay_preview
from regression_mvp.utils.financial_statement_artifacts import (
    write_merge_tables_artifact,
    write_three_statements_artifacts,
)
from regression_mvp.utils.metrics import (
    file_fingerprint,
    financial_merge_tables_metrics,
    result_metrics,
    upload_summary,
)
from regression_mvp.utils.runners.financial_merge_tables_runner import run_financial_merge_tables
from regression_mvp.utils.subject_match_artifacts import write_subject_match_table_artifact
from regression_mvp.storage import minio_sync
from regression_mvp.web_app import case_edit_actions


ROOT_DIR = Path(__file__).resolve().parents[1]
CANDIDATES_DIR = ROOT_DIR / "case_candidates"
CONFIG_PATH = ROOT_DIR / "config.json"


def candidate_detail(candidate_name: str) -> dict[str, Any]:
    candidate_dir = _candidate_dir(candidate_name)
    raw = _read_simple_yaml(candidate_dir / "candidate.yaml")
    expected_path = _candidate_file(candidate_dir, raw.get("expected_file") or "")
    request_path = candidate_dir / "subject_match_request.json"
    actual_path = candidate_dir / "actual_response.json"
    return {
        "name": candidate_dir.name,
        "path": str(candidate_dir),
        "endpoint": raw.get("endpoint_type", ""),
        "case_name": raw.get("case_name", candidate_dir.name),
        "description": raw.get("description", ""),
        "review_status": raw.get("review_status", ""),
        "expected_file": str(expected_path) if expected_path else "",
        "has_expected_json": bool(expected_path and expected_path.is_file()),
        "has_visual_editor": bool(expected_path and expected_path.is_file())
        and case_edit_actions.supports_visual_editor(raw.get("endpoint_type", "")),
        "has_subject_editor": raw.get("endpoint_type") == "financial_subject_match"
        and expected_path is not None
        and expected_path.is_file()
        and request_path.is_file(),
        "has_merge_table_editor": raw.get("endpoint_type") == "financial_merge_tables",
        "merge_config": _merge_config(raw),
        "merge_table_options": _merge_table_options(candidate_dir, raw),
        "artifacts": _candidate_artifacts(candidate_dir),
    }


def expected_json_text(candidate_name: str) -> str:
    expected_path = _expected_path(candidate_name)
    value = load_json(expected_path)
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def save_expected_json(candidate_name: str, raw_text: str) -> None:
    expected_path = _expected_path(candidate_name)
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误: line={exc.lineno}, column={exc.colno}, message={exc.msg}") from exc
    dump_json(expected_path, value)
    _refresh_subject_match_artifact(_candidate_dir(candidate_name))
    _try_upload_candidate(_candidate_dir(candidate_name))


def save_candidate_metadata(candidate_name: str, case_name: str, description: str) -> None:
    candidate_dir = _candidate_dir(candidate_name)
    safe_case_name = _validate_case_name(case_name)
    yaml_path = candidate_dir / "candidate.yaml"
    raw = _read_simple_yaml(yaml_path)
    raw["case_name"] = safe_case_name
    raw["description"] = description.strip()
    _write_simple_yaml(yaml_path, raw)
    _try_upload_candidate(candidate_dir)


def regenerate_merge_tables(candidate_name: str, form_data: Any) -> None:
    candidate_dir = _candidate_dir(candidate_name)
    yaml_path = candidate_dir / "candidate.yaml"
    raw = _read_simple_yaml(yaml_path)
    if raw.get("endpoint_type") != "financial_merge_tables":
        raise ValueError("only financial_merge_tables candidates support merge source selection")

    source_indices = _selected_merge_indices(form_data)
    merge_text = str(form_data.get("merge_text") or "").strip()
    input_path = _candidate_file(candidate_dir, raw.get("input_file") or "")
    expected_path = _candidate_file(candidate_dir, raw.get("expected_file") or "")
    if input_path is None or not input_path.is_file():
        raise FileNotFoundError(f"candidate input file not found: {candidate_dir}")
    if expected_path is None:
        raise FileNotFoundError(f"candidate expected file not configured: {candidate_dir}")

    config = load_json(CONFIG_PATH)
    run_result = run_financial_merge_tables(
        config["api"],
        input_path,
        source_indices=source_indices,
        merge_text=merge_text,
    )

    actual_items_path = candidate_dir / (raw.get("actual_items_file") or "actual_items.json")
    actual_response_path = candidate_dir / (raw.get("actual_response_file") or "actual_response.json")
    metrics_path = candidate_dir / (raw.get("metrics_file") or "metrics.json")
    org_tables_path = candidate_dir / (raw.get("org_tables_file") or "org_tables.json")
    preview_md_path = candidate_dir / (raw.get("preview_md") or "preview.md")
    preview_html_path = candidate_dir / (raw.get("preview_html") or "preview.html")

    dump_json(actual_items_path, run_result.mineru_items)
    dump_json(actual_response_path, run_result.actual_response)
    dump_json(expected_path, run_result.actual_response)
    dump_json(org_tables_path, run_result.org_tables)

    render_content_preview(run_result.mineru_items, preview_md_path, preview_html_path)
    overlay_path = render_overlay_preview(input_path, run_result.mineru_items, candidate_dir)
    statement_artifacts = write_three_statements_artifacts(
        run_result.actual_response,
        run_result.actual_response,
        candidate_dir,
        raw.get("case_name") or candidate_dir.name,
    )
    statement_artifacts.update(
        write_merge_tables_artifact(
            run_result.org_tables,
            run_result.actual_response,
            run_result.merge_request,
            candidate_dir,
            raw.get("case_name") or candidate_dir.name,
        )
    )

    metrics = {
        "file": file_fingerprint(input_path),
        "upload": {
            "status_code": run_result.merge_tables_status_code,
            "elapsed_seconds": round(run_result.merge_tables_elapsed_seconds, 3),
            "summary": upload_summary({"merge_tables": run_result.actual_response}, str(actual_items_path)),
        },
        "result": result_metrics(run_result.mineru_items),
    }
    metrics["result"]["financial_merge_tables"] = financial_merge_tables_metrics(
        run_result.actual_response,
        run_result.merge_request,
    )
    metrics["result"]["financial_merge_tables_artifacts"] = statement_artifacts
    dump_json(metrics_path, metrics)

    actual_indices = run_result.merge_request.get("debug", {}).get("source_indices") or source_indices
    raw["merge_source_indices"] = ",".join(str(index) for index in actual_indices)
    raw["merge_text"] = merge_text
    raw["org_tables_file"] = org_tables_path.name
    raw["actual_items_file"] = actual_items_path.name
    raw["actual_response_file"] = actual_response_path.name
    raw["metrics_file"] = metrics_path.name
    raw["preview_md"] = preview_md_path.name
    raw["preview_html"] = preview_html_path.name
    raw["overlay_html"] = overlay_path.name if overlay_path else raw.get("overlay_html", "")
    _write_simple_yaml(yaml_path, raw)
    _try_upload_candidate(candidate_dir)


def visual_editor(candidate_name: str) -> dict[str, Any]:
    candidate = candidate_detail(candidate_name)
    endpoint = candidate["endpoint"]
    if not case_edit_actions.supports_visual_editor(endpoint):
        raise ValueError(f"unsupported visual editor endpoint: {endpoint}")
    expected = load_json(_expected_path(candidate_name))
    case = _candidate_as_case(candidate)

    if endpoint in case_edit_actions.ITEM_ENDPOINTS:
        if not isinstance(expected, list):
            raise ValueError("expected file is not an items array")
        return {"template": "case_edit_visual.html", "case": case, "items": expected, **_editor_links(candidate_name)}
    if endpoint in case_edit_actions.FINANCIAL_TABLE_ENDPOINTS:
        if not isinstance(expected, list):
            raise ValueError("expected file is not a financial table array")
        return {"template": "case_edit_financial_tables.html", "case": case, "tables": expected, **_editor_links(candidate_name)}
    if endpoint == case_edit_actions.SUBJECT_MATCH_ENDPOINT:
        if not isinstance(expected, list):
            raise ValueError("expected file is not a subject-match table")
        return {
            "template": "case_edit_subject_match.html",
            "case": case,
            "rows": expected,
            "subjects": _subject_options(_subject_match_request_payload(candidate_name)),
            **_editor_links(candidate_name),
        }
    if endpoint == case_edit_actions.METADATA_ENDPOINT:
        if not isinstance(expected, dict):
            raise ValueError("expected file is not a metadata object")
        return {
            "template": "case_edit_metadata.html",
            "case": case,
            "response": expected,
            "metadata_rows": case_edit_actions._metadata_rows(expected.get("metadata")),
            **_editor_links(candidate_name),
        }
    raise ValueError(f"unsupported visual editor endpoint: {endpoint}")


def save_visual_editor(candidate_name: str, form_data: Any) -> None:
    candidate = candidate_detail(candidate_name)
    endpoint = candidate["endpoint"]
    if not case_edit_actions.supports_visual_editor(endpoint):
        raise ValueError(f"unsupported visual editor endpoint: {endpoint}")
    expected_path = _expected_path(candidate_name)
    expected = load_json(expected_path)

    if endpoint in case_edit_actions.ITEM_ENDPOINTS:
        if not isinstance(expected, list):
            raise ValueError("expected file is not an items array")
        for item_index, item in enumerate(expected):
            if isinstance(item, dict):
                case_edit_actions._apply_item_fields(item, form_data, item_index)
                case_edit_actions._apply_table_cells(item, form_data, item_index)
    elif endpoint in case_edit_actions.FINANCIAL_TABLE_ENDPOINTS:
        if not isinstance(expected, list):
            raise ValueError("expected file is not a financial table array")
        for table_index, table in enumerate(expected):
            if isinstance(table, dict):
                case_edit_actions._apply_financial_table_fields(table, form_data, table_index)
                case_edit_actions._apply_financial_table_cells(table, form_data, table_index)
    elif endpoint == case_edit_actions.SUBJECT_MATCH_ENDPOINT:
        if not isinstance(expected, list):
            raise ValueError("expected file is not a subject-match table")
        case_edit_actions._apply_subject_match_cells(expected, form_data)
    elif endpoint == case_edit_actions.METADATA_ENDPOINT:
        if not isinstance(expected, dict):
            raise ValueError("expected file is not a metadata object")
        case_edit_actions._apply_metadata_fields(expected, form_data)
    else:
        raise ValueError(f"unsupported visual editor endpoint: {endpoint}")

    dump_json(expected_path, expected)
    _refresh_subject_match_artifact(_candidate_dir(candidate_name))
    _try_upload_candidate(_candidate_dir(candidate_name))


def subject_match_editor(candidate_name: str) -> dict[str, Any]:
    candidate_dir = _candidate_dir(candidate_name)
    expected_path = _expected_path(candidate_name)
    request_path = candidate_dir / "subject_match_request.json"
    expected_rows = load_json(expected_path)
    request_payload = load_json(request_path)
    if not isinstance(expected_rows, list):
        raise ValueError("expected response is not a subject-match rows array")
    if not isinstance(request_payload, dict):
        raise ValueError("subject_match_request.json is not an object")
    return {
        "candidate": candidate_detail(candidate_name),
        "rows": expected_rows,
        "subjects": _subject_options(request_payload),
    }


def save_subject_match_editor(candidate_name: str, form_data: Any) -> None:
    candidate_dir = _candidate_dir(candidate_name)
    expected_path = _expected_path(candidate_name)
    expected_rows = load_json(expected_path)
    if not isinstance(expected_rows, list):
        raise ValueError("expected response is not a subject-match rows array")

    for row_index, row in enumerate(expected_rows):
        if not isinstance(row, list):
            continue
        for col_index, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            key = f"subject__{row_index}__{col_index}"
            subject_id = str(form_data.get(key) or "").strip()
            cell["subjectId"] = subject_id or None

    dump_json(expected_path, expected_rows)
    _refresh_subject_match_artifact(candidate_dir)
    _try_upload_candidate(candidate_dir)


def _refresh_subject_match_artifact(candidate_dir: Path) -> None:
    request_path = candidate_dir / "subject_match_request.json"
    actual_path = candidate_dir / "actual_response.json"
    try:
        request_payload = load_json(request_path) if request_path.is_file() else None
        actual_rows = load_json(actual_path) if actual_path.is_file() else None
        expected_rows = load_json(_expected_path(candidate_dir.name))
    except Exception:
        return
    write_subject_match_table_artifact(
        actual_rows,
        expected_rows,
        candidate_dir,
        request_payload=request_payload if isinstance(request_payload, dict) else None,
    )


def _candidate_as_case(candidate: dict[str, Any]) -> dict[str, str]:
    return {
        "endpoint": str(candidate.get("endpoint") or ""),
        "case_name": str(candidate.get("case_name") or candidate.get("name") or ""),
        "name": f"候选/{candidate.get('name')}",
        "path": str(candidate.get("path") or ""),
        "expected_file": str(candidate.get("expected_file") or ""),
    }


def _editor_links(candidate_name: str) -> dict[str, str]:
    return {
        "cancel_url": f"/case-candidates/{candidate_name}",
        "json_url": f"/case-candidates/{candidate_name}/edit-json",
    }


def _subject_match_request_payload(candidate_name: str) -> dict[str, Any]:
    candidate_dir = _candidate_dir(candidate_name)
    request_path = candidate_dir / "subject_match_request.json"
    if not request_path.is_file():
        return {}
    value = load_json(request_path)
    return value if isinstance(value, dict) else {}


def _candidate_dir(candidate_name: str) -> Path:
    safe_name = Path(candidate_name).name
    candidate_dir = (CANDIDATES_DIR / safe_name).resolve()
    try:
        candidate_dir.relative_to(CANDIDATES_DIR.resolve())
    except ValueError:
        raise ValueError(f"invalid candidate name: {candidate_name}")
    if not (candidate_dir / "candidate.yaml").is_file():
        raise FileNotFoundError(f"candidate.yaml not found: {candidate_dir}")
    return candidate_dir


def _expected_path(candidate_name: str) -> Path:
    candidate_dir = _candidate_dir(candidate_name)
    raw = _read_simple_yaml(candidate_dir / "candidate.yaml")
    expected_path = _candidate_file(candidate_dir, raw.get("expected_file") or "")
    if expected_path is None or not expected_path.is_file():
        raise FileNotFoundError(f"expected file not found in candidate: {candidate_dir}")
    return expected_path


def _candidate_file(candidate_dir: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = candidate_dir / path
    path = path.resolve()
    try:
        path.relative_to(candidate_dir.resolve())
    except ValueError:
        raise ValueError(f"invalid candidate file path: {value}")
    return path


def _candidate_artifacts(candidate_dir: Path) -> list[dict[str, str]]:
    patterns = [
        ("candidate.yaml", "candidate.yaml"),
        ("input", "input.*"),
        ("expected", "expected_*.candidate.json"),
        ("actual_response", "actual_response.json"),
        ("org_tables", "org_tables.json"),
        ("subject_match_request", "subject_match_request.json"),
        ("subject_match_source_table", "subject_match_source_table.html"),
        ("subject_match_table", "subject_match_table.html"),
        ("preview", "preview.html"),
        ("overlay", "overlay.html"),
        ("merge_tables_preview", "*.merge_tables_preview.html"),
        ("statement_grid_preview", "*.statement_grid_preview.html"),
        ("metrics", "metrics.json"),
    ]
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, pattern in patterns:
        for path in sorted(candidate_dir.glob(pattern)):
            resolved = str(path.resolve())
            if path.is_file() and resolved not in seen:
                artifacts.append({"label": label, "path": resolved})
                seen.add(resolved)
    return artifacts


def _subject_options(request_payload: dict[str, Any]) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for report in request_payload.get("reportList") or []:
        if not isinstance(report, dict):
            continue
        report_name = str(report.get("reportName") or "")
        for subject in report.get("subjectList") or []:
            if not isinstance(subject, dict):
                continue
            subject_id = str(subject.get("subjectId") or "").strip()
            subject_name = str(subject.get("subjectName") or "").strip()
            if subject_id:
                options.append(
                    {
                        "id": subject_id,
                        "name": subject_name,
                        "label": f"{subject_id} / {subject_name}" if subject_name else subject_id,
                        "report": report_name,
                    }
                )
    return sorted(options, key=lambda item: (item["report"], item["id"]))


def _merge_config(raw: dict[str, str]) -> dict[str, Any]:
    return {
        "source_indices": _parse_indices(raw.get("merge_source_indices") or ""),
        "source_indices_text": raw.get("merge_source_indices") or "",
        "merge_text": raw.get("merge_text") or "",
    }


def _merge_table_options(candidate_dir: Path, raw: dict[str, str]) -> list[dict[str, Any]]:
    if raw.get("endpoint_type") != "financial_merge_tables":
        return []
    org_tables_path = _candidate_file(candidate_dir, raw.get("org_tables_file") or "org_tables.json")
    if org_tables_path is None or not org_tables_path.is_file():
        return []
    try:
        org_tables = load_json(org_tables_path)
    except Exception:
        return []
    if not isinstance(org_tables, list):
        return []
    selected = set(_parse_indices(raw.get("merge_source_indices") or ""))
    options: list[dict[str, Any]] = []
    for index, table in enumerate(org_tables):
        if not isinstance(table, dict):
            continue
        rows = table.get("rows")
        row_count = len(rows) if isinstance(rows, list) else 0
        options.append(
            {
                "index": index,
                "selected": index in selected,
                "id": str(table.get("id") or ""),
                "primary_element_id": str(table.get("primaryElementId") or table.get("element_id") or ""),
                "page": str(table.get("pageAt") or table.get("page_no") or table.get("page") or ""),
                "text": _short_text(str(table.get("text") or table.get("table_type") or table.get("md") or "")),
                "row_count": row_count,
            }
        )
    return options


def _selected_merge_indices(form_data: Any) -> list[int]:
    selected = [str(value) for value in form_data.getlist("source_indices") if str(value).strip()]
    if selected:
        raw = ",".join(selected)
    else:
        raw = str(form_data.get("source_indices_text") or "")
    indices = _parse_indices(raw)
    if len(indices) < 2:
        raise ValueError("请至少选择两个要合并的表格")
    return indices


def _parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            index = int(text)
        except ValueError as exc:
            raise ValueError(f"表格序号必须是数字: {text}") from exc
        if index < 0:
            raise ValueError("表格序号不能小于 0")
        if index not in indices:
            indices.append(index)
    return indices


def _short_text(value: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _read_simple_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"').strip("'").replace('\\"', '"').replace("\\\\", "\\")
    return result


def _write_simple_yaml(path: Path, values: dict[str, Any]) -> None:
    lines = []
    for key, value in values.items():
        text = "" if value is None else str(value)
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}: "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_upload_candidate(candidate_dir: Path) -> None:
    try:
        minio_sync.upload_candidate_dir(candidate_dir)
    except Exception as exc:
        from loguru import logger

        minio_sync.record_sync_event("auto_upload_candidate", target=str(candidate_dir), error=exc)
        logger.warning("Regression MinIO candidate sync failed | candidate_dir={} err={}", candidate_dir, exc)


def _validate_case_name(case_name: str) -> str:
    value = case_name.strip()
    if not value:
        raise ValueError("case name is required")
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("case name cannot contain path separators")
    if not re.fullmatch(r"[\w.\-\u4e00-\u9fff]+", value):
        raise ValueError("case name only supports Chinese, letters, numbers, underscore, dash and dot")
    return value
