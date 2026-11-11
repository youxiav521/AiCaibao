from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


MINERU_ENDPOINTS = {"mineru_to_json", "mineru_upload_async", "ade_jobs_async"}
DEFAULT_THRESHOLDS = {
    "min_text_coverage": 0.94,
    "min_table_text_coverage": 0.92,
    "min_number_coverage": 0.98,
    "max_added_text_ratio": 0.12,
    "min_page_text_coverage": 0.70,
    "meaningful_page_min_chars": 40,
    "max_duplicate_text_ratio": 0.50,
}
_NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_WHITESPACE_RE = re.compile(r"\s+")


def assess_mineru_regression(
    expected: Any,
    actual: Any,
    diff_detail: list[dict[str, Any]],
    *,
    input_file: Path | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if isinstance(profile, dict):
        thresholds.update({key: value for key, value in profile.items() if value is not None})

    expected_items = expected if isinstance(expected, list) else []
    actual_items = actual if isinstance(actual, list) else []
    expected_pages = _page_content(expected_items)
    actual_pages = _page_content(actual_items)
    contract = _contract_assessment(expected_items, actual_items, expected_pages, actual_pages, thresholds, input_file)
    content = _content_assessment(expected_pages, actual_pages, diff_detail, thresholds)
    structure = _structure_assessment(expected_items, actual_items, diff_detail)

    if contract["issues"]:
        status = "failed"
        failure_category = "MINERU_CONTRACT_FAILED"
    elif content["issues"]:
        status = "suspected"
        failure_category = "MINERU_CONTENT_SUSPECTED"
    elif structure["warnings"]:
        status = "warning"
        failure_category = ""
    else:
        status = "passed"
        failure_category = ""

    return {
        "version": "mineru_assessment_v1",
        "status": status,
        "should_fail": status in {"failed", "suspected"},
        "failure_category": failure_category,
        "summary": _summary(status, contract, content, structure),
        "thresholds": thresholds,
        "contract": contract,
        "content": content,
        "structure": structure,
    }


def _contract_assessment(
    expected_items: list[Any],
    actual_items: list[Any],
    expected_pages: dict[str, dict[str, Any]],
    actual_pages: dict[str, dict[str, Any]],
    thresholds: dict[str, Any],
    input_file: Path | None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not actual_items:
        issues.append(_issue("empty_result", "MinerU returned no parsed items"))

    invalid_items = _schema_issues(actual_items)
    issues.extend(invalid_items)

    expected_tables = sum(page["table_count"] for page in expected_pages.values())
    actual_tables = sum(page["table_count"] for page in actual_pages.values())
    expected_images = sum(page["image_count"] for page in expected_pages.values())
    actual_images = sum(page["image_count"] for page in actual_pages.values())
    if expected_tables and not actual_tables:
        issues.append(_issue("tables_disappeared", "All expected table items disappeared"))
    if expected_images and not actual_images:
        issues.append(_issue("images_disappeared", "All expected image items disappeared"))

    meaningful_min_chars = int(thresholds["meaningful_page_min_chars"])
    for page, expected_page in expected_pages.items():
        expected_meaningful = expected_page["text_chars"] >= meaningful_min_chars or expected_page["table_count"] > 0
        if not expected_meaningful:
            continue
        actual_page = actual_pages.get(page)
        if actual_page is None or (actual_page["text_chars"] == 0 and actual_page["table_count"] == 0):
            issues.append(_issue("meaningful_page_missing", f"Expected meaningful page {page} has no parsed content", page=page))

    source_page_count = _pdf_page_count(input_file)
    actual_page_count = len(actual_pages)
    if source_page_count and actual_page_count == 0:
        issues.append(_issue("output_pages_empty", "Input PDF has pages but MinerU produced no page assignment"))

    duplicate_ratio = _duplicate_text_ratio(actual_pages)
    if duplicate_ratio > float(thresholds["max_duplicate_text_ratio"]):
        issues.append(
            _issue(
                "duplicate_text_excessive",
                f"Repeated text ratio {duplicate_ratio:.1%} exceeds threshold",
            )
        )

    return {
        "passed": not issues,
        "issues": issues,
        "expected_page_count": len(expected_pages),
        "actual_page_count": actual_page_count,
        "input_pdf_page_count": source_page_count,
        "expected_table_count": expected_tables,
        "actual_table_count": actual_tables,
        "expected_image_count": expected_images,
        "actual_image_count": actual_images,
        "duplicate_text_ratio": duplicate_ratio,
    }


def _content_assessment(
    expected_pages: dict[str, dict[str, Any]],
    actual_pages: dict[str, dict[str, Any]],
    diff_detail: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    page_metrics: list[dict[str, Any]] = []
    expected_text_chars = 0
    actual_text_chars = 0
    matched_text_chars = 0
    expected_table_chars = 0
    matched_table_chars = 0
    expected_numbers: Counter[str] = Counter()
    actual_numbers: Counter[str] = Counter()

    for page in sorted(set(expected_pages) | set(actual_pages), key=_page_sort_key):
        expected_page = expected_pages.get(page, _empty_page())
        actual_page = actual_pages.get(page, _empty_page())
        text_match = _text_match(expected_page["text"], actual_page["text"])
        table_match = _text_match(expected_page["table_text"], actual_page["table_text"])
        expected_text_chars += text_match["expected_chars"]
        actual_text_chars += text_match["actual_chars"]
        matched_text_chars += text_match["matched_chars"]
        expected_table_chars += table_match["expected_chars"]
        matched_table_chars += table_match["matched_chars"]
        expected_numbers.update(expected_page["numbers"])
        actual_numbers.update(actual_page["numbers"])
        page_metrics.append(
            {
                "page": page,
                "text_coverage": text_match["coverage"],
                "text_similarity": text_match["similarity"],
                "added_text_ratio": text_match["added_ratio"],
                "expected_text_chars": text_match["expected_chars"],
                "actual_text_chars": text_match["actual_chars"],
                "expected_table_chars": table_match["expected_chars"],
                "table_text_coverage": table_match["coverage"],
                "expected_tables": expected_page["table_count"],
                "actual_tables": actual_page["table_count"],
                "expected_images": expected_page["image_count"],
                "actual_images": actual_page["image_count"],
            }
        )

    text_coverage = _ratio(matched_text_chars, expected_text_chars, default=1.0)
    added_text_ratio = _ratio(max(actual_text_chars - matched_text_chars, 0), actual_text_chars, default=0.0)
    table_text_coverage = _ratio(matched_table_chars, expected_table_chars, default=1.0)
    number_coverage = _counter_coverage(expected_numbers, actual_numbers)
    issues: list[dict[str, Any]] = []
    if text_coverage < float(thresholds["min_text_coverage"]):
        issues.append(_issue("text_coverage_low", f"Text coverage {text_coverage:.1%} below threshold"))
    if table_text_coverage < float(thresholds["min_table_text_coverage"]):
        issues.append(_issue("table_content_coverage_low", f"Table text coverage {table_text_coverage:.1%} below threshold"))
    if expected_numbers and number_coverage < float(thresholds["min_number_coverage"]):
        issues.append(_issue("number_coverage_low", f"Table number coverage {number_coverage:.1%} below threshold"))
    if added_text_ratio > float(thresholds["max_added_text_ratio"]):
        issues.append(_issue("added_text_excessive", f"Added text ratio {added_text_ratio:.1%} exceeds threshold"))

    page_min_chars = int(thresholds["meaningful_page_min_chars"])
    for page in page_metrics:
        if page["expected_text_chars"] < page_min_chars:
            continue
        if page["text_coverage"] < float(thresholds["min_page_text_coverage"]):
            issues.append(
                _issue(
                    "page_text_coverage_low",
                    f"Page {page['page']} text coverage {page['text_coverage']:.1%} below threshold",
                    page=page["page"],
                )
            )

    critical_number_diffs = sum(
        1
        for diff in diff_detail
        if diff.get("type") == "financial_numeric_changed" and diff.get("severity") in {"critical", "major"}
    )
    if critical_number_diffs:
        issues.append(
            _issue(
                "financial_numeric_changed",
                f"Detected {critical_number_diffs} changed financial numeric cell(s)",
            )
        )

    focus_pages = sorted(
        page_metrics,
        key=lambda item: (
            item["text_coverage"],
            item["table_text_coverage"],
            -item["added_text_ratio"],
        ),
    )[:8]
    return {
        "passed": not issues,
        "issues": issues,
        "text_coverage": text_coverage,
        "added_text_ratio": added_text_ratio,
        "table_text_coverage": table_text_coverage,
        "number_coverage": number_coverage,
        "expected_text_chars": expected_text_chars,
        "actual_text_chars": actual_text_chars,
        "expected_table_chars": expected_table_chars,
        "critical_number_diff_count": critical_number_diffs,
        "focus_pages": focus_pages,
    }


def _structure_assessment(
    expected_items: list[Any],
    actual_items: list[Any],
    diff_detail: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_count = len(expected_items)
    actual_count = len(actual_items)
    item_delta_ratio = _ratio(abs(actual_count - expected_count), max(expected_count, actual_count), default=0.0)
    expected_types = _type_counts(expected_items)
    actual_types = _type_counts(actual_items)
    split_merge_count = sum(diff.get("type") == "text_split_merged" for diff in diff_detail)
    bbox_change_count = sum(diff.get("type") == "bbox_changed" for diff in diff_detail)
    minor_diff_count = sum(diff.get("severity") == "minor" for diff in diff_detail)
    warnings: list[dict[str, Any]] = []
    if expected_count != actual_count:
        warnings.append(_issue("item_count_changed", f"Item count {expected_count} -> {actual_count}"))
    if expected_types != actual_types:
        warnings.append(_issue("item_type_counts_changed", "Item type distribution changed"))
    if split_merge_count:
        warnings.append(_issue("text_split_or_merged", f"Detected {split_merge_count} local text split/merge event(s)"))
    if bbox_change_count:
        warnings.append(_issue("bbox_changed", f"Detected {bbox_change_count} coordinate drift event(s)"))
    return {
        "warnings": warnings,
        "expected_item_count": expected_count,
        "actual_item_count": actual_count,
        "item_delta_ratio": item_delta_ratio,
        "expected_type_counts": dict(sorted(expected_types.items())),
        "actual_type_counts": dict(sorted(actual_types.items())),
        "text_split_merge_count": split_merge_count,
        "bbox_change_count": bbox_change_count,
        "minor_diff_count": minor_diff_count,
    }


def _page_content(items: list[Any]) -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = defaultdict(_empty_page)
    for item in items:
        if not isinstance(item, dict):
            continue
        page = _item_page(item)
        if not page:
            continue
        page_data = pages[page]
        item_type = str(item.get("type") or "")
        page_data["type_counts"][item_type or "<missing>"] += 1
        if item_type == "table" or isinstance(item.get("rows"), list):
            table_text = _table_text(item)
            if table_text:
                page_data["table_parts"].append(table_text)
                page_data["numbers"].update(_numbers(table_text))
            page_data["table_count"] += 1
            continue
        if item_type == "image":
            page_data["image_count"] += 1
            continue
        text = _item_text(item)
        if text:
            page_data["text_parts"].append(text)
            page_data["numbers"].update(_numbers(text))
    for page_data in pages.values():
        page_data["text"] = "\n".join(page_data.pop("text_parts"))
        page_data["table_text"] = "\n".join(page_data.pop("table_parts"))
        page_data["text_chars"] = len(page_data["text"])
    return dict(pages)


def _empty_page() -> dict[str, Any]:
    return {
        "text_parts": [],
        "table_parts": [],
        "text": "",
        "table_text": "",
        "text_chars": 0,
        "table_count": 0,
        "image_count": 0,
        "numbers": Counter(),
        "type_counts": Counter(),
    }


def _schema_issues(items: list[Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(_issue("item_not_object", f"Item {index} is not an object"))
            continue
        if not item.get("type"):
            issues.append(_issue("item_type_missing", f"Item {index} is missing type"))
        if not _item_page(item):
            issues.append(_issue("item_page_missing", f"Item {index} is missing page"))
        if item.get("type") == "table" and not isinstance(item.get("rows"), list):
            issues.append(_issue("table_rows_missing", f"Table item {index} is missing rows"))
    return issues[:20]


def _item_page(item: dict[str, Any]) -> str:
    value = item.get("page") if item.get("page") is not None else item.get("pageAt")
    return str(value).strip() if value is not None else ""


def _item_text(item: dict[str, Any]) -> str:
    for key in ("value", "text", "md"):
        value = item.get(key)
        if value not in (None, ""):
            return _normalize_text(str(value))
    return ""


def _table_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    rows = item.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, list):
                continue
            cells = []
            for cell in row:
                value = cell.get("text") if isinstance(cell, dict) else cell
                text = _normalize_text(str(value or ""))
                if text:
                    cells.append(text)
            if cells:
                parts.append("|".join(cells))
    return "\n".join(parts)


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return _WHITESPACE_RE.sub("", text).strip()


def _numbers(value: str) -> Counter[str]:
    return Counter(match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(value))


def _text_match(expected: str, actual: str) -> dict[str, Any]:
    if not expected and not actual:
        return {"expected_chars": 0, "actual_chars": 0, "matched_chars": 0, "coverage": 1.0, "similarity": 1.0, "added_ratio": 0.0}
    if not expected:
        return {"expected_chars": 0, "actual_chars": len(actual), "matched_chars": 0, "coverage": 1.0, "similarity": 0.0, "added_ratio": 1.0}
    if not actual:
        return {"expected_chars": len(expected), "actual_chars": 0, "matched_chars": 0, "coverage": 0.0, "similarity": 0.0, "added_ratio": 0.0}

    matcher = SequenceMatcher(None, expected, actual, autojunk=False)
    matched_chars = sum(block.size for block in matcher.get_matching_blocks())
    expected_chars = len(expected)
    actual_chars = len(actual)
    return {
        "expected_chars": expected_chars,
        "actual_chars": actual_chars,
        "matched_chars": matched_chars,
        "coverage": round(_ratio(matched_chars, expected_chars, default=1.0), 4),
        "similarity": round(_ratio(matched_chars * 2, expected_chars + actual_chars, default=1.0), 4),
        "added_ratio": round(_ratio(max(actual_chars - matched_chars, 0), actual_chars, default=0.0), 4),
    }


def _duplicate_text_ratio(pages: dict[str, dict[str, Any]]) -> float:
    chunks = [part for page in pages.values() for part in page["text"].split("\n") if len(part) >= 20]
    if not chunks:
        return 0.0
    counts = Counter(chunks)
    total_chars = sum(len(chunk) for chunk in chunks)
    repeated_chars = sum(len(chunk) * (count - 1) for chunk, count in counts.items() if count > 1)
    return round(_ratio(repeated_chars, total_chars, default=0.0), 4)


def _counter_coverage(expected: Counter[str], actual: Counter[str]) -> float:
    expected_total = sum(expected.values())
    matched = sum(min(count, actual.get(value, 0)) for value, count in expected.items())
    return round(_ratio(matched, expected_total, default=1.0), 4)


def _type_counts(items: list[Any]) -> Counter[str]:
    return Counter(str(item.get("type") or "<missing>") for item in items if isinstance(item, dict))


def _pdf_page_count(input_file: Path | None) -> int | None:
    if input_file is None or input_file.suffix.lower() != ".pdf" or not input_file.is_file():
        return None
    try:
        import fitz

        with fitz.open(input_file) as document:
            return document.page_count
    except Exception:
        return None


def _issue(code: str, message: str, *, page: str | None = None) -> dict[str, str]:
    result = {"code": code, "message": message}
    if page:
        result["page"] = page
    return result


def _ratio(numerator: int | float, denominator: int | float, *, default: float) -> float:
    return float(numerator) / float(denominator) if denominator else default


def _page_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def _summary(status: str, contract: dict[str, Any], content: dict[str, Any], structure: dict[str, Any]) -> str:
    if status == "failed":
        return f"基础契约校验失败：发现 {len(contract['issues'])} 项问题"
    if status == "suspected":
        return f"内容疑似回归：发现 {len(content['issues'])} 项需要复核的问题"
    if status == "warning":
        return f"内容指标在阈值内，但存在 {len(structure['warnings'])} 项结构性提示"
    return "基础契约和内容指标均通过"
