from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from regression_mvp.utils.json_io import dump_json, load_json
from regression_mvp.utils.case_exclusions import DEFAULT_EXCLUSIONS_FILE, load_compare_exclusions
from regression_mvp.utils.metrics import utc_timestamp
from regression_mvp.storage import minio_sync


ROOT_DIR = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT_DIR / "cases"
CASE_HISTORY_DIR = ROOT_DIR / "case_history"
ITEM_ENDPOINTS = {"file_read", "mineru_to_json", "mineru_upload_async", "ade_jobs_async"}
FINANCIAL_TABLE_ENDPOINTS = {
    "financial_three_statements",
    "financial_org_tables",
    "financial_merge_tables",
}
SUBJECT_MATCH_ENDPOINT = "financial_subject_match"
METADATA_ENDPOINT = "financial_metadata"


def case_detail(endpoint: str, case_name: str) -> dict[str, str]:
    case_dir = _case_dir(endpoint, case_name)
    raw = _read_simple_yaml(case_dir / "case.yaml")
    expected_path = _expected_path(endpoint, case_name)
    return {
        "endpoint": endpoint,
        "case_name": raw.get("case_name", case_dir.name),
        "name": f"{endpoint}/{raw.get('case_name', case_dir.name)}",
        "path": str(case_dir),
        "expected_file": str(expected_path),
    }


def expected_json_text(endpoint: str, case_name: str) -> str:
    value = load_json(_expected_path(endpoint, case_name))
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def save_expected_json(endpoint: str, case_name: str, raw_text: str) -> Path:
    expected_path = _expected_path(endpoint, case_name)
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误: line={exc.lineno}, column={exc.colno}, message={exc.msg}") from exc

    backup_dir = _backup_expected(endpoint, case_name, expected_path)
    dump_json(expected_path, value)
    _try_upload_case(_case_dir(endpoint, case_name))
    return backup_dir


def compare_exclusion_editor(
    endpoint: str,
    case_name: str,
    *,
    page: str = "",
    item_type: str = "",
    bbox_text: str = "",
    diff_path: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    case = case_detail(endpoint, case_name)
    bbox = _parse_bbox(bbox_text, required=False)
    return {
        "case": case,
        "page": _optional_int(page),
        "item_type": item_type.strip(),
        "bbox_text": json.dumps(bbox, ensure_ascii=False) if bbox else "",
        "diff_path": diff_path.strip(),
        "run_id": run_id.strip(),
        "rules": load_compare_exclusions(_case_dir(endpoint, case_name)),
    }


def save_compare_exclusion(endpoint: str, case_name: str, form_data: Any) -> dict[str, Any]:
    case_dir = _case_dir(endpoint, case_name)
    page = _optional_int(form_data.get("page"))
    if page is None or page <= 0:
        raise ValueError("请选择有效的页码后再保存排除规则")
    scope = str(form_data.get("scope") or "item").strip()
    if scope not in {"item", "page"}:
        raise ValueError("排除范围只能是单项或整页")
    reason = str(form_data.get("reason") or "").strip()
    if not reason:
        raise ValueError("请填写排除原因，便于后续审查")
    item_type = str(form_data.get("item_type") or "").strip()
    bbox = _parse_bbox(str(form_data.get("bbox") or ""), required=scope == "item")
    if scope == "item" and not item_type:
        raise ValueError("单项排除需要保留解析对象类型")

    raw = _read_simple_yaml(case_dir / "case.yaml")
    exclusions_name = raw.get("compare_exclusions_file") or DEFAULT_EXCLUSIONS_FILE
    exclusions_path = (case_dir / exclusions_name).resolve()
    try:
        exclusions_path.relative_to(case_dir.resolve())
    except ValueError as exc:
        raise ValueError("排除规则文件路径不合法") from exc

    payload: dict[str, Any] = {"rules": []}
    if exclusions_path.is_file():
        loaded = load_json(exclusions_path)
        if isinstance(loaded, dict) and isinstance(loaded.get("rules"), list):
            payload = loaded
    selector: dict[str, Any] = {"page": page}
    if item_type:
        selector["type"] = item_type
    if bbox:
        selector["bbox"] = bbox
    rule = {
        "id": _next_exclusion_id(),
        "status": "active",
        "scope": scope,
        "selector": selector,
        "reason": reason,
        "created_at": utc_timestamp(),
        "source": {
            "run_id": str(form_data.get("run_id") or "").strip(),
            "diff_path": str(form_data.get("diff_path") or "").strip(),
        },
    }
    payload["rules"].append(rule)
    dump_json(exclusions_path, payload)
    raw["compare_exclusions_file"] = exclusions_name
    _write_simple_yaml(case_dir / "case.yaml", raw)
    _try_upload_case(case_dir)
    return rule


def supports_visual_editor(endpoint: str) -> bool:
    return endpoint in ITEM_ENDPOINTS | FINANCIAL_TABLE_ENDPOINTS | {SUBJECT_MATCH_ENDPOINT, METADATA_ENDPOINT}


def visual_editor(endpoint: str, case_name: str) -> dict[str, Any]:
    if not supports_visual_editor(endpoint):
        raise ValueError(f"unsupported visual editor endpoint: {endpoint}")
    case = case_detail(endpoint, case_name)
    expected = load_json(_expected_path(endpoint, case_name))
    if endpoint in ITEM_ENDPOINTS:
        if not isinstance(expected, list):
            raise ValueError("expected file is not an items array")
        return {"template": "case_edit_visual.html", "case": case, "items": expected}
    if endpoint in FINANCIAL_TABLE_ENDPOINTS:
        if not isinstance(expected, list):
            raise ValueError("expected file is not a financial table array")
        return {"template": "case_edit_financial_tables.html", "case": case, "tables": expected}
    if endpoint == SUBJECT_MATCH_ENDPOINT:
        if not isinstance(expected, list):
            raise ValueError("expected file is not a subject-match table")
        return {
            "template": "case_edit_subject_match.html",
            "case": case,
            "rows": expected,
            "subjects": _subject_options(_case_input_payload(endpoint, case_name)),
        }
    if endpoint == METADATA_ENDPOINT:
        if not isinstance(expected, dict):
            raise ValueError("expected file is not a metadata object")
        return {
            "template": "case_edit_metadata.html",
            "case": case,
            "response": expected,
            "metadata_rows": _metadata_rows(expected.get("metadata")),
        }
    raise ValueError(f"unsupported visual editor endpoint: {endpoint}")


def save_visual_editor(endpoint: str, case_name: str, form_data: Any) -> Path:
    if not supports_visual_editor(endpoint):
        raise ValueError(f"unsupported visual editor endpoint: {endpoint}")
    expected_path = _expected_path(endpoint, case_name)
    expected = load_json(expected_path)

    if endpoint in ITEM_ENDPOINTS:
        if not isinstance(expected, list):
            raise ValueError("expected file is not an items array")
        for item_index, item in enumerate(expected):
            if not isinstance(item, dict):
                continue
            _apply_item_fields(item, form_data, item_index)
            _apply_table_cells(item, form_data, item_index)
    elif endpoint in FINANCIAL_TABLE_ENDPOINTS:
        if not isinstance(expected, list):
            raise ValueError("expected file is not a financial table array")
        for table_index, table in enumerate(expected):
            if not isinstance(table, dict):
                continue
            _apply_financial_table_fields(table, form_data, table_index)
            _apply_financial_table_cells(table, form_data, table_index)
    elif endpoint == SUBJECT_MATCH_ENDPOINT:
        if not isinstance(expected, list):
            raise ValueError("expected file is not a subject-match table")
        _apply_subject_match_cells(expected, form_data)
    elif endpoint == METADATA_ENDPOINT:
        if not isinstance(expected, dict):
            raise ValueError("expected file is not a metadata object")
        _apply_metadata_fields(expected, form_data)
    else:
        raise ValueError(f"unsupported visual editor endpoint: {endpoint}")

    backup_dir = _backup_expected(endpoint, case_name, expected_path)
    dump_json(expected_path, expected)
    _try_upload_case(_case_dir(endpoint, case_name))
    return backup_dir


def _case_dir(endpoint: str, case_name: str) -> Path:
    safe_endpoint = Path(endpoint.strip()).name
    safe_case_name = Path(case_name.strip()).name
    case_dir = (CASES_DIR / safe_endpoint / safe_case_name).resolve()
    try:
        case_dir.relative_to(CASES_DIR.resolve())
    except ValueError:
        raise ValueError(f"invalid case path: {endpoint}/{case_name}")
    if (case_dir / "case.yaml").is_file():
        return case_dir

    endpoint_dir = (CASES_DIR / safe_endpoint).resolve()
    for candidate_yaml in endpoint_dir.glob("*/case.yaml"):
        raw = _read_simple_yaml(candidate_yaml)
        if raw.get("case_name") == case_name:
            return candidate_yaml.parent.resolve()
    raise FileNotFoundError(f"case.yaml not found: {case_dir}")


def _expected_path(endpoint: str, case_name: str) -> Path:
    case_dir = _case_dir(endpoint, case_name)
    raw = _read_simple_yaml(case_dir / "case.yaml")
    expected_name = raw.get("expected_file") or "expected_response.json"
    expected_path = (case_dir / expected_name).resolve()
    try:
        expected_path.relative_to(case_dir.resolve())
    except ValueError:
        raise ValueError(f"invalid expected file path in case.yaml: {expected_name}")
    if not expected_path.is_file():
        raise FileNotFoundError(f"expected file not found: {expected_path}")
    return expected_path


def _next_backup_dir(endpoint: str, case_name: str) -> Path:
    CASE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = utc_timestamp().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    base = CASE_HISTORY_DIR / f"{stamp}_{endpoint}_{case_name}_manual_expected_edit"
    target = base
    index = 1
    while target.exists():
        index += 1
        target = CASE_HISTORY_DIR / f"{base.name}_{index}"
    return target


def _backup_expected(endpoint: str, case_name: str, expected_path: Path) -> Path:
    backup_dir = _next_backup_dir(endpoint, case_name)
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(expected_path, backup_dir / expected_path.name)
    return backup_dir


def _apply_item_fields(item: dict[str, Any], form_data: Any, item_index: int) -> None:
    for field in ("type", "value", "md"):
        key = f"item__{item_index}__{field}"
        if key in form_data:
            item[field] = str(form_data.get(key) or "")

    page_key = f"item__{item_index}__page"
    if page_key in form_data:
        item["page"] = _optional_int(form_data.get(page_key))

    heading_key = f"item__{item_index}__headingLevel"
    if heading_key in form_data:
        item["headingLevel"] = _optional_int(form_data.get(heading_key))

    bbox = item.get("bBox")
    if isinstance(bbox, dict):
        for field in ("l", "t", "r", "b"):
            key = f"item__{item_index}__bbox__{field}"
            if key in form_data:
                bbox[field] = _optional_float(form_data.get(key))


def _apply_table_cells(item: dict[str, Any], form_data: Any, item_index: int) -> None:
    for section in ("rows", "headerCells", "bodyCells", "footerCells"):
        rows = item.get(section)
        if not isinstance(rows, list):
            continue
        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            for col_index, cell in enumerate(row):
                if not isinstance(cell, dict):
                    continue
                key = f"cell__{item_index}__{section}__{row_index}__{col_index}__text"
                if key in form_data:
                    cell["text"] = str(form_data.get(key) or "")
    if isinstance(item.get("rows"), list):
        _sync_table_sections_from_rows(item)
        item["html"] = _table_html(item["rows"])
        item["md"] = _table_markdown(item["rows"])


def _apply_financial_table_fields(table: dict[str, Any], form_data: Any, table_index: int) -> None:
    for field in ("id", "category", "currencyUnit", "caliberLabel", "period", "text"):
        key = f"table__{table_index}__{field}"
        if key in form_data:
            table[field] = str(form_data.get(key) or "")

    page_key = f"table__{table_index}__pageAt"
    if page_key in form_data:
        table["pageAt"] = _optional_int(form_data.get(page_key))


def _apply_financial_table_cells(table: dict[str, Any], form_data: Any, table_index: int) -> None:
    rows = table.get("rows")
    if not isinstance(rows, list):
        return
    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for col_index, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            key = f"ftcell__{table_index}__{row_index}__{col_index}__text"
            if key in form_data:
                cell["text"] = str(form_data.get(key) or "")


def _apply_subject_match_cells(rows: list[Any], form_data: Any) -> None:
    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for col_index, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            text_key = f"smcell__{row_index}__{col_index}__text"
            if text_key in form_data:
                cell["text"] = str(form_data.get(text_key) or "")
            subject_key = f"smcell__{row_index}__{col_index}__subjectId"
            if subject_key in form_data:
                subject_id = str(form_data.get(subject_key) or "").strip()
                cell["subjectId"] = subject_id or None


def _apply_metadata_fields(response: dict[str, Any], form_data: Any) -> None:
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        response["metadata"] = metadata
    for key in list(form_data.keys()):
        if not key.startswith("metadata__"):
            continue
        parts = key.split("__")
        if len(parts) < 3:
            continue
        field_path = parts[1:-1]
        leaf = parts[-1]
        target = _ensure_metadata_node(metadata, field_path)
        if leaf == "value":
            target["value"] = str(form_data.get(key) or "") or None
        elif leaf == "source_element_id":
            target["source_element_id"] = _optional_int(form_data.get(key))


def _ensure_metadata_node(metadata: dict[str, Any], field_path: list[str]) -> dict[str, Any]:
    current: Any = metadata
    for part in field_path:
        if not isinstance(current, dict):
            raise ValueError(f"metadata path is not editable: {'/'.join(field_path)}")
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        raise ValueError(f"metadata field is not editable: {'/'.join(field_path)}")
    return current


def _sync_table_sections_from_rows(item: dict[str, Any]) -> None:
    rows = item.get("rows")
    if not isinstance(rows, list):
        return
    text_by_position: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for col_index, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            rowstart = _optional_int(cell.get("rowstart"))
            colstart = _optional_int(cell.get("colstart"))
            key = (rowstart if rowstart is not None else row_index, colstart if colstart is not None else col_index)
            text_by_position[key] = str(cell.get("text") or "")

    for section in ("headerCells", "bodyCells", "footerCells"):
        section_rows = item.get(section)
        if not isinstance(section_rows, list):
            continue
        for row_index, row in enumerate(section_rows):
            if not isinstance(row, list):
                continue
            for col_index, cell in enumerate(row):
                if not isinstance(cell, dict):
                    continue
                rowstart = _optional_int(cell.get("rowstart"))
                colstart = _optional_int(cell.get("colstart"))
                key = (rowstart if rowstart is not None else row_index, colstart if colstart is not None else col_index)
                if key in text_by_position:
                    cell["text"] = text_by_position[key]


def _table_html(rows: list[Any]) -> str:
    parts = ["<table>"]
    for row in rows:
        if not isinstance(row, list):
            continue
        parts.append("<tr>")
        for cell in row:
            if not isinstance(cell, dict):
                continue
            colspan = _positive_int(cell.get("colspan"), 1)
            rowspan = _positive_int(cell.get("rowspan"), 1)
            attrs = []
            if colspan > 1:
                attrs.append(f' colspan="{colspan}"')
            if rowspan > 1:
                attrs.append(f' rowspan="{rowspan}"')
            parts.append(f"<td{''.join(attrs)}>{_escape_html(str(cell.get('text') or ''))}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _table_markdown(rows: list[Any]) -> str:
    expanded_rows = [_expanded_row(row) for row in rows if isinstance(row, list)]
    if not expanded_rows:
        return ""
    width = max(len(row) for row in expanded_rows)
    normalized = [row + [""] * (width - len(row)) for row in expanded_rows]
    lines = []
    for index, row in enumerate(normalized):
        escaped = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
        if index == 0:
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _expanded_row(row: list[Any]) -> list[str]:
    values: list[str] = []
    for cell in row:
        if not isinstance(cell, dict):
            continue
        text = str(cell.get("text") or "")
        colspan = _positive_int(cell.get("colspan"), 1)
        values.extend([text] * colspan)
    return values


def _optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return int(text)


def _optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def _parse_bbox(raw_text: str, *, required: bool) -> dict[str, float] | None:
    text = raw_text.strip()
    if not text:
        if required:
            raise ValueError("当前差异没有可用坐标，请改为按整页排除")
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("坐标信息格式错误，请返回报告重新选择差异项") from exc
    if not isinstance(value, dict):
        raise ValueError("坐标信息格式错误，请返回报告重新选择差异项")
    try:
        bbox = {key: float(value[key]) for key in ("l", "t", "r", "b")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("坐标信息不完整，请改为按整页排除") from exc
    if bbox["r"] <= bbox["l"] or bbox["b"] <= bbox["t"]:
        raise ValueError("坐标范围无效，请改为按整页排除")
    return bbox


def _next_exclusion_id() -> str:
    stamp = utc_timestamp().replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    return f"exclude_{stamp}_{uuid4().hex[:8]}"


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _case_input_payload(endpoint: str, case_name: str) -> dict[str, Any]:
    case_dir = _case_dir(endpoint, case_name)
    raw = _read_simple_yaml(case_dir / "case.yaml")
    input_name = raw.get("input_file") or "input_request.json"
    input_path = (case_dir / input_name).resolve()
    try:
        input_path.relative_to(case_dir.resolve())
    except ValueError:
        raise ValueError(f"invalid input file path in case.yaml: {input_name}")
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        value = load_json(input_path)
        return value if isinstance(value, dict) else {}
    return {}


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


def _metadata_rows(metadata: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(metadata, dict):
        _collect_metadata_rows(metadata, [], rows)
    return rows


def _collect_metadata_rows(value: Any, path: list[str], rows: list[dict[str, Any]]) -> None:
    if _is_metadata_leaf(value):
        rows.append(
            {
                "path": path,
                "name": " / ".join(path),
                "form_key": "__".join(path),
                "value": "" if value.get("value") is None else value.get("value"),
                "source_element_id": "" if value.get("source_element_id") is None else value.get("source_element_id"),
            }
        )
        return
    if isinstance(value, list):
        rows.append(
            {
                "path": path,
                "name": " / ".join(path),
                "form_key": "__".join(path),
                "value": json.dumps(value, ensure_ascii=False),
                "source_element_id": "",
                "readonly": True,
            }
        )
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _collect_metadata_rows(value[key], path + [str(key)], rows)


def _is_metadata_leaf(value: Any) -> bool:
    return isinstance(value, dict) and ("value" in value or "source_element_id" in value)


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
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}: "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_upload_case(case_dir: Path) -> None:
    try:
        minio_sync.upload_case_dir(case_dir)
    except Exception as exc:
        from loguru import logger

        minio_sync.record_sync_event("auto_upload_case", target=str(case_dir), error=exc)
        logger.warning("Regression MinIO case sync failed | case_dir={} err={}", case_dir, exc)
