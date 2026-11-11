from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any

SEVERITY_ORDER = ("blocker", "critical", "major", "minor", "info")
SEVERITY_RANK = {severity: index for index, severity in enumerate(SEVERITY_ORDER)}


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_id_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def validate_input_fingerprint(expected_sha256: str | None, actual_sha256: str) -> list[str]:
    if not expected_sha256:
        return [
            "case.yaml does not define input_sha256; baseline ownership cannot be fully verified",
        ]
    if expected_sha256.lower() != actual_sha256.lower():
        return [
            f"input_sha256 mismatch: case.yaml={expected_sha256.lower()}, actual={actual_sha256.lower()}",
        ]
    return []


def result_metrics(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {
            "item_count": 0,
            "type_counts": {},
            "page_count": 0,
            "table_count": 0,
            "text_count": 0,
            "image_count": 0,
            "empty_text_count": 0,
            "empty_text_ratio": 0,
            "items_per_page": {},
        }

    type_counts: Counter[str] = Counter()
    page_counts: Counter[str] = Counter()
    empty_text_count = 0
    table_row_counts: list[int] = []

    for item in items:
        if not isinstance(item, dict):
            type_counts["<non-object>"] += 1
            continue

        item_type = str(item.get("type") or "<missing>")
        type_counts[item_type] += 1

        page = item.get("page")
        if page is not None:
            page_counts[str(page)] += 1

        text = item.get("value") if item.get("value") is not None else item.get("md")
        if isinstance(text, str) and not text.strip():
            empty_text_count += 1

        rows = item.get("rows")
        if isinstance(rows, list):
            table_row_counts.append(len(rows))

    item_count = len(items)
    return {
        "item_count": item_count,
        "type_counts": dict(sorted(type_counts.items())),
        "page_count": len(page_counts),
        "table_count": type_counts.get("table", 0),
        "text_count": type_counts.get("text", 0),
        "image_count": type_counts.get("image", 0),
        "empty_text_count": empty_text_count,
        "empty_text_ratio": round(empty_text_count / item_count, 4) if item_count else 0,
        "items_per_page": dict(sorted(page_counts.items(), key=lambda pair: _page_sort_key(pair[0]))),
        "max_table_rows": max(table_row_counts) if table_row_counts else 0,
    }


def financial_metadata_metrics(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {
            "success": False,
            "metadata_field_count": 0,
            "metadata_non_empty_field_count": 0,
            "metadata_source_element_count": 0,
            "three_statements_count": 0,
            "change_explanation_count": 0,
            "recognized_fields": [],
            "missing_core_fields": [],
        }

    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    flattened = _flatten_metadata_fields(metadata)
    recognized_fields = sorted(name for name, value in flattened.items() if _has_value(value))
    source_element_count = sum(1 for value in flattened.values() if _has_source_element(value))

    core_fields = [
        "企业全称",
        "当前期数",
        "审计",
        "证券代码.A股",
        "证券代码.港股",
        "证券代码.美股",
        "交易简称.A股",
        "交易简称.港股",
    ]
    missing_core_fields = [name for name in core_fields if not _has_value(flattened.get(name))]

    change_explanations = metadata.get("指标变动说明")
    three_statements = response.get("three_statements")

    return {
        "success": response.get("success") is True,
        "metadata_field_count": len(flattened),
        "metadata_non_empty_field_count": len(recognized_fields),
        "metadata_source_element_count": source_element_count,
        "three_statements_count": len(three_statements) if isinstance(three_statements, list) else 0,
        "change_explanation_count": len(change_explanations) if isinstance(change_explanations, list) else 0,
        "recognized_fields": recognized_fields,
        "missing_core_fields": missing_core_fields,
        "has_company_full_name": _has_value(flattened.get("企业全称")),
        "has_current_period": _has_value(flattened.get("当前期数")),
        "has_audit": _has_value(flattened.get("审计")),
    }


def financial_three_statements_metrics(response: Any) -> dict[str, Any]:
    if not isinstance(response, list):
        return {
            "statement_count": 0,
            "statement_names": [],
            "currency_units": [],
            "total_rows": 0,
            "total_lines": 0,
            "tables_with_rows": 0,
            "tables_with_lines": 0,
            "pages": [],
        }

    statement_names: list[str] = []
    currency_units: list[str] = []
    pages: set[str] = set()
    total_rows = 0
    total_lines = 0
    tables_with_rows = 0
    tables_with_lines = 0
    row_counts: list[int] = []

    for item in response:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if text is not None:
            statement_names.append(str(text))

        currency_unit = item.get("currencyUnit")
        if currency_unit is not None:
            currency_units.append(str(currency_unit))

        page_at = item.get("pageAt")
        if page_at is not None:
            pages.add(str(page_at))

        rows = item.get("rows")
        if isinstance(rows, list):
            row_count = len(rows)
            row_counts.append(row_count)
            total_rows += row_count
            if row_count:
                tables_with_rows += 1

        lines = item.get("lines")
        if isinstance(lines, list):
            line_count = len(lines)
            total_lines += line_count
            if line_count:
                tables_with_lines += 1

    return {
        "statement_count": len(response),
        "statement_names": sorted(dict.fromkeys(statement_names)),
        "currency_units": sorted(dict.fromkeys(currency_units)),
        "total_rows": total_rows,
        "total_lines": total_lines,
        "tables_with_rows": tables_with_rows,
        "tables_with_lines": tables_with_lines,
        "max_rows": max(row_counts) if row_counts else 0,
        "pages": sorted(pages, key=_page_sort_key),
    }


def financial_org_tables_metrics(response: Any) -> dict[str, Any]:
    if not isinstance(response, list):
        return {
            "org_table_count": 0,
            "table_names": [],
            "currency_units": [],
            "total_rows": 0,
            "tables_with_bbox": 0,
            "bbox_coverage_ratio": 0,
            "financial_statement_table_count": 0,
            "pages": [],
        }

    table_names: list[str] = []
    currency_units: list[str] = []
    pages: set[str] = set()
    row_counts: list[int] = []
    tables_with_bbox = 0
    financial_statement_table_count = 0

    for item in response:
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        if text is not None:
            table_names.append(str(text))

        currency_unit = item.get("currencyUnit")
        if currency_unit is not None:
            currency_units.append(str(currency_unit))

        page_at = item.get("pageAt")
        if page_at is not None:
            pages.add(str(page_at))

        rows = item.get("rows")
        if isinstance(rows, list):
            row_counts.append(len(rows))

        bbox = item.get("bBox") or item.get("bbox")
        if isinstance(bbox, dict) and all(key in bbox for key in ("l", "t", "r", "b")):
            tables_with_bbox += 1

        if item.get("isFinancialStatementTable") is True:
            financial_statement_table_count += 1

    org_table_count = len(response)
    return {
        "org_table_count": org_table_count,
        "table_names": sorted(dict.fromkeys(table_names)),
        "currency_units": sorted(dict.fromkeys(currency_units)),
        "total_rows": sum(row_counts),
        "max_rows": max(row_counts) if row_counts else 0,
        "tables_with_bbox": tables_with_bbox,
        "bbox_coverage_ratio": round(tables_with_bbox / org_table_count, 4) if org_table_count else 0,
        "financial_statement_table_count": financial_statement_table_count,
        "pages": sorted(pages, key=_page_sort_key),
    }


def financial_merge_tables_metrics(response: Any, merge_request: Any) -> dict[str, Any]:
    statement_metrics = financial_three_statements_metrics(response)
    merges = merge_request.get("merges") if isinstance(merge_request, dict) else None
    first_merge = merges[0] if isinstance(merges, list) and merges and isinstance(merges[0], dict) else {}
    source_ids = first_merge.get("sourceIds") if isinstance(first_merge, dict) else []
    if not isinstance(source_ids, list):
        source_ids = []

    return {
        "merge_count": len(merges) if isinstance(merges, list) else 0,
        "source_id_count": len(source_ids),
        "source_ids": [str(value) for value in source_ids],
        "merge_text": str(first_merge.get("text") or "") if isinstance(first_merge, dict) else "",
        "output_table_count": statement_metrics.get("statement_count", 0),
        "output_total_rows": statement_metrics.get("total_rows", 0),
        "output_table_names": statement_metrics.get("statement_names", []),
    }


def financial_subject_match_metrics(response: Any) -> dict[str, Any]:
    if not isinstance(response, list):
        return {
            "row_count": 0,
            "cell_count": 0,
            "matched_subject_count": 0,
            "matched_subject_rate": 0,
            "empty_subject_count": 0,
            "unique_subject_count": 0,
            "matched_subject_ids": [],
        }

    cell_count = 0
    matched_subject_ids: list[str] = []
    for row in response:
        if not isinstance(row, list):
            continue
        for cell in row:
            if not isinstance(cell, dict):
                continue
            cell_count += 1
            subject_id = cell.get("subjectId")
            if isinstance(subject_id, str) and subject_id.strip():
                matched_subject_ids.append(subject_id.strip())

    matched_subject_count = len(matched_subject_ids)
    return {
        "row_count": len(response),
        "cell_count": cell_count,
        "matched_subject_count": matched_subject_count,
        "matched_subject_rate": round(matched_subject_count / cell_count, 4) if cell_count else 0,
        "empty_subject_count": cell_count - matched_subject_count,
        "unique_subject_count": len(set(matched_subject_ids)),
        "matched_subject_ids": sorted(set(matched_subject_ids)),
    }


def diff_metrics(diffs: list[dict[str, Any]]) -> dict[str, Any]:
    kind_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    top_paths: list[str] = []

    for diff in sorted(diffs, key=_diff_sort_key):
        kind = classify_diff(diff)
        kind_counts[kind] += 1
        severity_counts[diff_severity(diff)] += 1
        path = str(diff.get("path") or "")
        field_counts[_field_name(path)] += 1
        if len(top_paths) < 20:
            top_paths.append(path)

    return {
        "total": len(diffs),
        "by_severity": {
            severity: severity_counts[severity]
            for severity in SEVERITY_ORDER
            if severity_counts.get(severity)
        },
        "by_kind": dict(sorted(kind_counts.items())),
        "top_fields": dict(field_counts.most_common(10)),
        "sample_paths": top_paths,
    }


def diff_severity(diff: dict[str, Any]) -> str:
    severity = str(diff.get("severity") or "major").lower()
    return severity if severity in SEVERITY_RANK else "major"


def _diff_sort_key(diff: dict[str, Any]) -> tuple[int, str]:
    return (SEVERITY_RANK.get(diff_severity(diff), SEVERITY_RANK["major"]), str(diff.get("path") or ""))


def classify_diff(diff: dict[str, Any]) -> str:
    diff_type = diff.get("type")
    if isinstance(diff_type, str) and diff_type:
        return diff_type

    expected = diff.get("expected")
    actual = diff.get("actual")
    if expected == "<missing>":
        return "extra_field"
    if actual == "<missing>":
        return "missing_field"
    if type(expected) is not type(actual):
        return "type_changed"
    return "value_changed"


def classify_failure(error_message: str, diff_total: int) -> str:
    text = error_message.lower()
    if diff_total:
        return "DIFF_FAILED"
    if "error code: 502" in text or "502 bad gateway" in text or "status=502" in text:
        return "UPSTREAM_502_FAILED"
    if "timeout" in text:
        return "TIMEOUT"
    if "server disconnected without sending a response" in text:
        return "PROXY_OR_NETWORK_FAILED"
    if "connection" in text or "connect" in text or "proxy" in text:
        return "ENV_OR_NETWORK_FAILED"
    if "status=500" in text or "status code 500" in text:
        return "API_5XX_FAILED"
    if "status=4" in text or "status code 4" in text:
        return "API_4XX_FAILED"
    if "schema" in text or "not a list" in text or "not an object" in text:
        return "SCHEMA_FAILED"
    if error_message:
        return "CASE_EXECUTION_FAILED"
    return ""


def upload_summary(raw_response: dict[str, Any], actual_items_path: str) -> dict[str, Any]:
    content = raw_response.get("content")
    pages_size = raw_response.get("pages_size")
    return {
        "success": raw_response.get("success"),
        "file_id": raw_response.get("file_id") or raw_response.get("fileId"),
        "has_convert_json": isinstance(raw_response.get("convert_json"), list),
        "convert_json_count": len(raw_response.get("convert_json") or [])
        if isinstance(raw_response.get("convert_json"), list)
        else 0,
        "content_keys": sorted(content.keys()) if isinstance(content, dict) else [],
        "pages_size_count": len(pages_size) if isinstance(pages_size, list) else 0,
        "actual_items_path": actual_items_path,
    }


def build_run_summary(entries: list[Any]) -> dict[str, Any]:
    status_counts = Counter(entry.status for entry in entries)
    failure_counts = Counter(entry.failure_category for entry in entries if entry.failure_category)
    total_cases = len(entries)
    warning_cases = status_counts.get("warning", 0)
    suspected_cases = status_counts.get("suspected", 0)
    passed_cases = status_counts.get("passed", 0) + warning_cases
    failed_cases = status_counts.get("failed", 0) + suspected_cases
    total_duration = sum(entry.duration_seconds for entry in entries)
    total_diffs = sum(entry.diff_metrics.get("total", 0) for entry in entries)
    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "warning_cases": warning_cases,
        "suspected_cases": suspected_cases,
        "pass_rate": round(passed_cases / total_cases, 4) if total_cases else 0,
        "total_duration_seconds": round(total_duration, 3),
        "total_diffs": total_diffs,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def _field_name(path: str) -> str:
    if not path:
        return "<unknown>"
    tail = path.rsplit(".", 1)[-1]
    if "[" in tail:
        return tail.split("[", 1)[0] or "<list>"
    return tail


def _page_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def _flatten_metadata_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, dict) and "value" not in value and "source_element_id" not in value:
            for child_key, child_value in value.items():
                result[f"{key}.{child_key}"] = child_value
        else:
            result[key] = value
    return result


def _has_value(value: Any) -> bool:
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | dict):
        return bool(value)
    return True


def _has_source_element(value: Any) -> bool:
    return isinstance(value, dict) and value.get("source_element_id") is not None
