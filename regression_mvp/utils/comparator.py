from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any

from regression_mvp.utils.case_exclusions import filter_compare_exclusions


BBOX_TOLERANCE = 4.0
FLOAT_TOLERANCE = 0.001
MAX_DIFFS = 500
TEXT_MATCH_THRESHOLD = 0.86
TEXT_MINOR_DIFF_THRESHOLD = 0.92
BBOX_MATCH_THRESHOLD = 0.72
MIN_MATCH_SCORE = 0.62
FLOAT_TOLERANCE_FIELDS = {"score", "confidence"}
TEXT_FIELDS = {"text", "md", "value"}
DERIVED_TABLE_FIELDS = {"hasHeader", "headerCells", "standard2D", "table_caption", "two_d_data"}
SECONDARY_ITEM_FIELDS = {"headingLevel", "table_type"}
SEVERITY_ORDER = ("blocker", "critical", "major", "minor", "info")


@dataclass(frozen=True)
class CompareResult:
    passed: bool
    summary: str
    diffs: list[dict[str, Any]]
    ignored: list[dict[str, Any]] = field(default_factory=list)


def compare_items(
    expected: Any,
    actual: Any,
    *,
    exclusions: list[dict[str, Any]] | None = None,
) -> CompareResult:
    expected, actual, ignored = filter_compare_exclusions(expected, actual, exclusions)
    diffs: list[dict[str, Any]] = []
    _compare(expected, actual, "items", diffs)
    if not diffs:
        return CompareResult(True, "no differences", [], ignored)
    summary = _diff_summary(diffs)
    if len(diffs) >= MAX_DIFFS:
        summary += f" (truncated at {MAX_DIFFS})"
    return CompareResult(False, summary, diffs, ignored)


def _diff_summary(diffs: list[dict[str, Any]]) -> str:
    severity_counts = Counter(_diff_severity(diff) for diff in diffs)
    parts = [
        f"{severity_counts[severity]} {severity}"
        for severity in SEVERITY_ORDER
        if severity_counts.get(severity)
    ]
    if not parts:
        return f"{len(diffs)} differences"
    return f"{len(diffs)} differences: " + ", ".join(parts)


def _compare(expected: Any, actual: Any, path: str, diffs: list[dict[str, Any]]) -> None:
    if _diff_limit_reached(diffs):
        return

    if _is_bbox_number_path(path):
        if _numbers_close(expected, actual, BBOX_TOLERANCE):
            return
        _add_diff(diffs, path, expected, actual, diff_type="bbox_changed", severity="minor")
        return
    if _is_row_position_path(path):
        return
    if _compare_by_field_strategy(expected, actual, path, diffs):
        return

    if type(expected) is not type(actual):
        _add_diff(diffs, path, expected, actual, diff_type="type_changed")
        return

    if isinstance(expected, dict):
        if _is_top_level_item(expected, actual):
            _compare_item(expected, actual, path, diffs)
            return
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys | actual_keys):
            if _skip_redundant_table_field(key, expected, actual):
                continue
            if _skip_ordinal_identifier_field(key, expected, actual):
                continue
            child_path = f"{path}.{key}"
            if key not in expected:
                _add_diff(diffs, child_path, "<missing>", actual[key], diff_type="added_field")
            elif key not in actual:
                _add_diff(diffs, child_path, expected[key], "<missing>", diff_type="removed_field")
            else:
                _compare(expected[key], actual[key], child_path, diffs)
        return

    if isinstance(expected, list):
        _compare_list(expected, actual, path, diffs)
        return

    if expected != actual:
        _add_diff(diffs, path, expected, actual, diff_type="changed")


def _compare_by_field_strategy(expected: Any, actual: Any, path: str, diffs: list[dict[str, Any]]) -> bool:
    field = _path_field_name(path)
    if field in FLOAT_TOLERANCE_FIELDS and _is_number_like(expected) and _is_number_like(actual):
        if _numbers_close(expected, actual, FLOAT_TOLERANCE):
            return True
        _add_diff(diffs, path, expected, actual, diff_type="float_changed", severity="minor")
        return True

    if field in TEXT_FIELDS and isinstance(expected, str) and isinstance(actual, str):
        _compare_text_content(expected, actual, path, diffs)
        return True

    if field in SECONDARY_ITEM_FIELDS and expected != actual:
        _add_diff(diffs, path, expected, actual, diff_type="metadata_changed", severity="minor")
        return True

    return False


def _is_top_level_item(expected: dict[str, Any], actual: Any) -> bool:
    return isinstance(actual, dict) and ("type" in expected or "type" in actual)


def _compare_item(expected: dict[str, Any], actual: dict[str, Any], path: str, diffs: list[dict[str, Any]]) -> None:
    item_type = _item_type(expected) if _item_type(expected) != "unknown" else _item_type(actual)
    expected_keys = set(expected)
    actual_keys = set(actual)
    content_fields = set(TEXT_FIELDS) | {"html"}

    for key in sorted((expected_keys | actual_keys) - content_fields):
        if _skip_derived_table_field(key, expected, actual):
            continue
        if _skip_ordinal_identifier_field(key, expected, actual):
            continue
        child_path = f"{path}.{key}"
        if key in SECONDARY_ITEM_FIELDS:
            expected_value = expected.get(key, "<missing>")
            actual_value = actual.get(key, "<missing>")
            if expected_value != actual_value:
                _add_diff(diffs, child_path, expected_value, actual_value, diff_type="metadata_changed", severity="minor")
            continue
        if key in {"bBox", "bbox"} and (key not in expected or key not in actual):
            _add_diff(
                diffs,
                child_path,
                expected.get(key, "<missing>"),
                actual.get(key, "<missing>"),
                diff_type="bbox_changed",
                severity="minor",
            )
            continue
        if key not in expected:
            _add_diff(diffs, child_path, "<missing>", actual[key], diff_type="added_field")
        elif key not in actual:
            _add_diff(diffs, child_path, expected[key], "<missing>", diff_type="removed_field")
        else:
            _compare(expected[key], actual[key], child_path, diffs)

    if item_type == "table" and (isinstance(expected.get("rows"), list) or isinstance(actual.get("rows"), list)):
        return
    if item_type == "image":
        _compare_image_description(expected, actual, path, diffs)
        return

    expected_value, actual_value = _primary_item_content(expected), _primary_item_content(actual)
    if expected_value is None and actual_value is None:
        return
    _compare_text_content(expected_value or "", actual_value or "", f"{path}.value", diffs)


def _skip_derived_table_field(key: str, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if key not in DERIVED_TABLE_FIELDS:
        return False
    return isinstance(expected.get("rows"), list) or isinstance(actual.get("rows"), list)


def _primary_item_content(item: dict[str, Any]) -> str | None:
    for key in ("value", "text", "md", "html"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _compare_image_description(expected: dict[str, Any], actual: dict[str, Any], path: str, diffs: list[dict[str, Any]]) -> None:
    expected_description = expected.get("md") or expected.get("value") or expected.get("text") or ""
    actual_description = actual.get("md") or actual.get("value") or actual.get("text") or ""
    if expected_description == actual_description:
        return
    _add_diff(
        diffs,
        f"{path}.image_description",
        expected_description,
        actual_description,
        diff_type="image_description_changed",
        severity="minor",
    )


def _compare_text_content(expected: str, actual: str, path: str, diffs: list[dict[str, Any]]) -> None:
    if expected == actual:
        return
    if _compact_text(expected) == _compact_text(actual):
        _add_diff(diffs, path, expected, actual, diff_type="text_whitespace_changed", severity="minor")
        return
    numeric = _financial_number_pair(expected, actual)
    if numeric is not None:
        expected_number, actual_number = numeric
        if expected_number == actual_number:
            _add_diff(diffs, path, expected, actual, diff_type="financial_number_format_changed", severity="minor")
            return
        _add_diff(
            diffs,
            path,
            expected,
            actual,
            diff_type="financial_numeric_changed",
            severity=_financial_number_severity(expected, actual, expected_number, actual_number),
        )
        return
    similarity = _text_similarity(expected, actual)
    severity = _text_diff_severity(expected, actual, similarity)
    _add_diff(diffs, path, expected, actual, diff_type="text_changed", severity=severity)


def _compare_list(expected: list[Any], actual: list[Any], path: str, diffs: list[dict[str, Any]]) -> None:
    if _should_match_dict_list(expected, actual, path):
        _compare_matched_dict_list(expected, actual, path, diffs)
        return
    if _should_match_table_rows(expected, actual, path):
        _compare_matched_table_rows(expected, actual, path, diffs)
        return

    if len(expected) != len(actual):
        _add_diff(diffs, f"{path}.length", len(expected), len(actual), diff_type="length_changed")
    for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        _compare(expected_item, actual_item, f"{path}[{index}]", diffs)


def _should_match_dict_list(expected: list[Any], actual: list[Any], path: str) -> bool:
    if path != "items" and not path.endswith(".items"):
        return False
    if not expected and not actual:
        return False
    return all(isinstance(item, dict) for item in expected + actual)


def _should_match_table_rows(expected: list[Any], actual: list[Any], path: str) -> bool:
    if not path.endswith(".rows"):
        return False
    if not expected and not actual:
        return False
    return all(isinstance(row, list) for row in expected + actual)


def _compare_matched_dict_list(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    path: str,
    diffs: list[dict[str, Any]],
) -> None:
    pairs, groups, table_groups, removed, added = _match_dict_items(expected, actual)
    if removed or added:
        _add_diff(
            diffs,
            f"{path}.length",
            len(expected),
            len(actual),
            diff_type="length_changed",
            severity=_list_length_severity(len(expected), len(actual), has_removed=bool(removed)),
        )

    for expected_index in removed:
        _add_diff(
            diffs,
            f"{path}[{expected_index}]",
            expected[expected_index],
            "<missing>",
            diff_type="removed_item",
            severity=_item_presence_severity(expected[expected_index]),
        )
    for actual_index in added:
        _add_diff(
            diffs,
            f"{path}[+{actual_index}]",
            "<missing>",
            actual[actual_index],
            diff_type="added_item",
            severity=_item_presence_severity(actual[actual_index]),
        )
    for group in groups:
        expected_indexes = group["expected_indexes"]
        actual_indexes = group["actual_indexes"]
        _add_diff(
            diffs,
            f"{path}[{expected_indexes[0]}~{actual_indexes[0]}]",
            f"{len(expected_indexes)} text item(s)",
            f"{len(actual_indexes)} text item(s)",
            diff_type="text_split_merged",
            severity="minor",
        )
    table_groups_by_expected = {group["expected_index"]: group for group in table_groups}
    for group in table_groups:
        _add_diff(
            diffs,
            f"{path}[{group['expected_index']}~{group['actual_indexes'][0]}]",
            f"1 table / {group['expected_row_count']} rows",
            f"{len(group['actual_indexes'])} tables / {group['actual_row_count']} rows",
            diff_type="table_split_merged",
            severity="minor",
        )
    for pair in pairs:
        match_path = f"{path}[{pair['expected_index']}~{pair['actual_index']}]"
        table_group = table_groups_by_expected.get(pair["expected_index"])
        if table_group is not None:
            _compare_split_table_items(
                expected[pair["expected_index"]],
                actual,
                table_group,
                match_path,
                diffs,
            )
            continue
        _compare(expected[pair["expected_index"]], actual[pair["actual_index"]], match_path, diffs)


def _match_dict_items(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    pairs: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    matched_expected: set[int] = set()
    matched_actual: set[int] = set()

    _match_by_exact_key(expected, actual, pairs, matched_expected, matched_actual)
    groups.extend(_match_local_text_groups(expected, actual, matched_expected, matched_actual))

    candidates: list[tuple[float, int, int, str]] = []
    for expected_index, expected_item in enumerate(expected):
        if expected_index in matched_expected:
            continue
        for actual_index, actual_item in enumerate(actual):
            if actual_index in matched_actual:
                continue
            score, strategy = _match_score(expected_item, actual_item)
            if score >= MIN_MATCH_SCORE:
                candidates.append((score, expected_index, actual_index, strategy))

    for score, expected_index, actual_index, strategy in sorted(candidates, reverse=True):
        if expected_index in matched_expected or actual_index in matched_actual:
            continue
        pairs.append(
            {
                "expected_index": expected_index,
                "actual_index": actual_index,
                "score": round(score, 4),
                "strategy": strategy,
            }
        )
        matched_expected.add(expected_index)
        matched_actual.add(actual_index)

    table_groups = _match_cross_page_table_continuations(expected, actual, pairs, matched_actual)
    removed = [index for index in range(len(expected)) if index not in matched_expected]
    added = [index for index in range(len(actual)) if index not in matched_actual]
    return sorted(pairs, key=lambda item: item["expected_index"]), groups, table_groups, removed, added


def _match_cross_page_table_continuations(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    matched_actual: set[int],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for pair in pairs:
        expected_index = pair["expected_index"]
        actual_index = pair["actual_index"]
        expected_item = expected[expected_index]
        actual_item = actual[actual_index]
        if not _is_table_item(expected_item) or not _is_table_item(actual_item):
            continue

        expected_rows = expected_item["rows"]
        primary_rows = actual_item["rows"]
        primary_page = _item_page(actual_item)
        continuation_indexes: list[int] = []
        for candidate_index in range(actual_index + 1, min(actual_index + 4, len(actual))):
            if candidate_index in matched_actual:
                continue
            candidate_item = actual[candidate_index]
            candidate_page = _item_page(candidate_item)
            if not _is_table_item(candidate_item) or not candidate_page or candidate_page == primary_page:
                continue

            expected_continuation_rows = _rows_from_source_page(expected_rows, candidate_page)
            if not expected_continuation_rows:
                continue
            candidate_rows = _without_repeated_table_header(candidate_item["rows"], primary_rows)
            _, removed_rows, added_rows = _match_table_rows(expected_continuation_rows, candidate_rows)
            if removed_rows or added_rows:
                continue

            continuation_indexes.append(candidate_index)
            matched_actual.add(candidate_index)

        if continuation_indexes:
            groups.append(
                {
                    "expected_index": expected_index,
                    "actual_indexes": [actual_index, *continuation_indexes],
                    "expected_row_count": len(expected_rows),
                    "actual_row_count": sum(
                        len(actual[index]["rows"])
                        - (1 if index != actual_index and _has_repeated_table_header(actual[index]["rows"], primary_rows) else 0)
                        for index in [actual_index, *continuation_indexes]
                    ),
                }
            )
    return groups


def _compare_split_table_items(
    expected_item: dict[str, Any],
    actual: list[dict[str, Any]],
    group: dict[str, Any],
    path: str,
    diffs: list[dict[str, Any]],
) -> None:
    primary_actual = actual[group["actual_indexes"][0]]
    expected_metadata = dict(expected_item)
    actual_metadata = dict(primary_actual)
    expected_metadata["rows"] = []
    actual_metadata["rows"] = []
    _compare_item(expected_metadata, actual_metadata, path, diffs)

    combined_rows = list(primary_actual["rows"])
    for actual_index in group["actual_indexes"][1:]:
        combined_rows.extend(_without_repeated_table_header(actual[actual_index]["rows"], primary_actual["rows"]))
    diff_start = len(diffs)
    _compare_matched_table_rows(expected_item["rows"], combined_rows, f"{path}.rows", diffs)
    _downgrade_split_table_layout_diffs(diffs[diff_start:])


def _is_table_item(item: dict[str, Any]) -> bool:
    return _item_type(item) == "table" and isinstance(item.get("rows"), list)


def _rows_from_source_page(rows: list[Any], page: Any) -> list[list[Any]]:
    return [row for row in rows if isinstance(row, list) and _row_has_source_page(row, page)]


def _row_has_source_page(row: list[Any], page: Any) -> bool:
    page_text = str(page)
    return any(
        isinstance(cell, dict)
        and str(cell.get("source_page_no") or cell.get("sourcePageNo") or "") == page_text
        for cell in row
    )


def _without_repeated_table_header(rows: list[Any], primary_rows: list[Any]) -> list[Any]:
    if _has_repeated_table_header(rows, primary_rows):
        return rows[1:]
    return rows


def _has_repeated_table_header(rows: list[Any], primary_rows: list[Any]) -> bool:
    if not rows or not primary_rows or not isinstance(rows[0], list) or not isinstance(primary_rows[0], list):
        return False
    return _row_match_score(primary_rows[0], rows[0]) >= 0.98


def _downgrade_split_table_layout_diffs(diffs: list[dict[str, Any]]) -> None:
    layout_fields = {"length", "colspan", "colSpan", "rowspan", "rowSpan", "colstart", "colStart", "rowstart", "rowStart"}
    for diff in diffs:
        if _path_field_name(str(diff.get("path") or "")) not in layout_fields:
            continue
        diff["type"] = "table_layout_changed"
        diff["severity"] = "minor"


def _match_local_text_groups(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    matched_expected: set[int],
    matched_actual: set[int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for expected_start in range(len(expected)):
        for expected_width in range(1, 4):
            expected_indexes = list(range(expected_start, expected_start + expected_width))
            if not _valid_text_group(expected, expected_indexes, matched_expected):
                continue
            expected_page = _item_page(expected[expected_start])
            expected_text = _group_text(expected, expected_indexes)
            if len(expected_text) < 8:
                continue
            for actual_start in range(len(actual)):
                for actual_width in range(1, 4):
                    if expected_width == actual_width == 1:
                        continue
                    actual_indexes = list(range(actual_start, actual_start + actual_width))
                    if not _valid_text_group(actual, actual_indexes, matched_actual):
                        continue
                    if _item_page(actual[actual_start]) != expected_page:
                        continue
                    actual_text = _group_text(actual, actual_indexes)
                    if len(actual_text) < 8:
                        continue
                    similarity = _text_similarity(expected_text, actual_text)
                    if similarity < 0.97:
                        continue
                    candidates.append(
                        {
                            "expected_indexes": expected_indexes,
                            "actual_indexes": actual_indexes,
                            "score": round(similarity, 4),
                            "strategy": "local_text_group",
                        }
                    )

    selected: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["score"],
            len(item["expected_indexes"]) + len(item["actual_indexes"]),
        ),
        reverse=True,
    ):
        if any(index in matched_expected for index in candidate["expected_indexes"]):
            continue
        if any(index in matched_actual for index in candidate["actual_indexes"]):
            continue
        selected.append(candidate)
        matched_expected.update(candidate["expected_indexes"])
        matched_actual.update(candidate["actual_indexes"])
    return selected


def _valid_text_group(items: list[dict[str, Any]], indexes: list[int], matched: set[int]) -> bool:
    if not indexes or indexes[-1] >= len(items) or any(index in matched for index in indexes):
        return False
    page = _item_page(items[indexes[0]])
    return bool(page) and all(
        _is_textual_item(items[index]) and _item_page(items[index]) == page
        for index in indexes
    )


def _is_textual_item(item: dict[str, Any]) -> bool:
    return _item_type(item) in {"text", "list"}


def _group_text(items: list[dict[str, Any]], indexes: list[int]) -> str:
    return "".join(_item_text(items[index]) for index in indexes)


def _compare_matched_table_rows(
    expected: list[list[Any]],
    actual: list[list[Any]],
    path: str,
    diffs: list[dict[str, Any]],
) -> None:
    pairs, removed, added = _match_table_rows(expected, actual)

    for expected_index in removed:
        _add_diff(
            diffs,
            f"{path}[{expected_index}]",
            expected[expected_index],
            "<missing>",
            diff_type="removed_row",
        )
    for actual_index in added:
        _add_diff(
            diffs,
            f"{path}[+{actual_index}]",
            "<missing>",
            actual[actual_index],
            diff_type="added_row",
            severity="minor",
        )
    for pair in pairs:
        expected_index = pair["expected_index"]
        actual_index = pair["actual_index"]
        if _row_exact_key(expected[expected_index]) == _row_exact_key(actual[actual_index]):
            continue
        match_path = f"{path}[{expected_index}~{actual_index}]"
        _compare(expected[expected_index], actual[actual_index], match_path, diffs)


def _match_table_rows(
    expected: list[list[Any]],
    actual: list[list[Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    pairs: list[dict[str, Any]] = []
    matched_expected: set[int] = set()
    matched_actual: set[int] = set()

    actual_by_key: dict[str, list[int]] = {}
    for actual_index, actual_row in enumerate(actual):
        key = _row_exact_key(actual_row)
        if key:
            actual_by_key.setdefault(key, []).append(actual_index)

    for expected_index, expected_row in enumerate(expected):
        key = _row_exact_key(expected_row)
        if not key:
            continue
        actual_index = next((index for index in actual_by_key.get(key, []) if index not in matched_actual), None)
        if actual_index is None:
            continue
        pairs.append(
            {
                "expected_index": expected_index,
                "actual_index": actual_index,
                "score": 1.0,
                "strategy": "row_exact_key",
            }
        )
        matched_expected.add(expected_index)
        matched_actual.add(actual_index)

    candidates: list[tuple[float, int, int]] = []
    for expected_index, expected_row in enumerate(expected):
        if expected_index in matched_expected:
            continue
        for actual_index, actual_row in enumerate(actual):
            if actual_index in matched_actual:
                continue
            score = _row_match_score(expected_row, actual_row)
            if score >= TEXT_MATCH_THRESHOLD:
                candidates.append((score, expected_index, actual_index))

    for score, expected_index, actual_index in sorted(candidates, reverse=True):
        if expected_index in matched_expected or actual_index in matched_actual:
            continue
        pairs.append(
            {
                "expected_index": expected_index,
                "actual_index": actual_index,
                "score": round(score, 4),
                "strategy": "row_similarity",
            }
        )
        matched_expected.add(expected_index)
        matched_actual.add(actual_index)

    removed = [index for index in range(len(expected)) if index not in matched_expected]
    added = [index for index in range(len(actual)) if index not in matched_actual]
    return sorted(pairs, key=lambda item: item["expected_index"]), removed, added


def _row_exact_key(row: list[Any]) -> str:
    texts = _row_cell_texts(row)
    while texts and not texts[-1]:
        texts.pop()
    return "|".join(texts)


def _row_similarity(expected: list[Any], actual: list[Any]) -> float:
    expected_text = " | ".join(_row_cell_texts(expected))
    actual_text = " | ".join(_row_cell_texts(actual))
    return _text_similarity(expected_text, actual_text)


def _row_match_score(expected: list[Any], actual: list[Any]) -> float:
    full_score = _row_similarity(expected, actual)
    expected_anchor = _row_anchor_text(expected)
    actual_anchor = _row_anchor_text(actual)
    if not expected_anchor or not actual_anchor:
        return full_score

    anchor_score = _text_similarity(expected_anchor, actual_anchor)
    if anchor_score >= 0.96:
        return max(full_score, 0.9 + anchor_score * 0.1)
    if anchor_score >= 0.82 and full_score >= 0.35:
        return max(full_score, anchor_score * 0.78 + full_score * 0.22)
    return full_score


def _row_anchor_text(row: list[Any]) -> str:
    for text in _row_cell_texts(row):
        if text:
            return text
    return ""


def _row_cell_texts(row: list[Any]) -> list[str]:
    texts: list[str] = []
    for cell in row:
        if isinstance(cell, dict):
            value = cell.get("text")
        else:
            value = cell
        texts.append(_clean_text(value))
    return texts


def _match_by_exact_key(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    matched_expected: set[int],
    matched_actual: set[int],
) -> None:
    actual_by_key: dict[tuple[Any, ...], list[int]] = {}
    for actual_index, actual_item in enumerate(actual):
        key = _exact_match_key(actual_item)
        if key:
            actual_by_key.setdefault(key, []).append(actual_index)

    for expected_index, expected_item in enumerate(expected):
        key = _exact_match_key(expected_item)
        if not key:
            continue
        candidates = actual_by_key.get(key) or []
        actual_index = next((index for index in candidates if index not in matched_actual), None)
        if actual_index is None:
            continue
        pairs.append(
            {
                "expected_index": expected_index,
                "actual_index": actual_index,
                "score": 1.0,
                "strategy": "exact_key",
            }
        )
        matched_expected.add(expected_index)
        matched_actual.add(actual_index)


def _exact_match_key(item: dict[str, Any]) -> tuple[Any, ...] | None:
    stable_id = item.get("element_id") or item.get("primaryElementId")
    if stable_id and _is_stable_item_identifier(stable_id):
        return ("id", stable_id)

    table_key = _financial_table_key(item)
    if table_key:
        return ("financial_table", *table_key)

    text = _item_text(item)
    if text:
        return ("content", _item_page(item), _item_type(item), text)
    return None


def _is_stable_item_identifier(value: Any) -> bool:
    """MinerU's sequential element_id is an ordinal, not a cross-run identity."""
    if isinstance(value, int):
        return False
    text = str(value or "").strip()
    return bool(text) and not text.isdecimal()


def _financial_table_key(item: dict[str, Any]) -> tuple[Any, ...] | None:
    if not isinstance(item.get("rows"), list):
        return None
    category = item.get("category") or ""
    text = item.get("text") or ""
    page = item.get("pageAt") or item.get("page") or ""
    if category or text:
        return (category, text, page)
    return None


def _match_score(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[float, str]:
    if _item_page(expected) != _item_page(actual):
        return 0.0, "page_mismatch"

    expected_type = _item_type(expected)
    actual_type = _item_type(actual)
    if expected_type == actual_type == "image":
        bbox_score = _bbox_similarity(
            expected.get("bBox") or expected.get("bbox"),
            actual.get("bBox") or actual.get("bbox"),
        )
        if bbox_score >= BBOX_MATCH_THRESHOLD:
            return 0.78 + bbox_score * 0.12, "image_bbox"
        return 0.64, "image_same_page"

    text_score = _text_similarity(_item_text(expected), _item_text(actual))
    bbox_score = _bbox_similarity(expected.get("bBox") or expected.get("bbox"), actual.get("bBox") or actual.get("bbox"))
    if expected_type != actual_type:
        if bbox_score >= 0.93:
            return 0.78 + min(text_score, 0.2) * 0.1, "bbox_type_changed"
        return 0.0, "type_mismatch"

    if bbox_score >= BBOX_MATCH_THRESHOLD and text_score >= 0.45:
        return 0.65 * bbox_score + 0.35 * text_score, "bbox_text"
    if text_score >= TEXT_MATCH_THRESHOLD:
        return text_score, "text_similarity"
    if bbox_score >= 0.9:
        return bbox_score * 0.85, "bbox"
    return max(text_score * 0.9, bbox_score * 0.7), "weak"


def _skip_redundant_table_field(key: str, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if key not in {"html", "md", "text", "value"}:
        return False
    return isinstance(expected.get("rows"), list) or isinstance(actual.get("rows"), list)


def _skip_ordinal_identifier_field(key: str, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if key not in {"element_id", "primaryElementId"}:
        return False
    return key in expected and key in actual and not _is_stable_item_identifier(expected[key]) and not _is_stable_item_identifier(actual[key])


def _item_presence_severity(item: dict[str, Any]) -> str:
    if _item_type(item) == "table":
        return "critical"
    if _item_type(item) in {"text", "image"}:
        return "major"
    return "major"


def _item_type(item: dict[str, Any]) -> str:
    if item.get("type"):
        return str(item["type"])
    if isinstance(item.get("rows"), list):
        return "table"
    return "unknown"


def _item_page(item: dict[str, Any]) -> Any:
    return item.get("page") or item.get("pageAt") or ""


def _item_text(item: dict[str, Any]) -> str:
    for key in ("value", "text", "md", "html", "category"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    rows = item.get("rows")
    if isinstance(rows, list):
        texts: list[str] = []
        for row in rows[:5]:
            if isinstance(row, list):
                for cell in row[:6]:
                    if isinstance(cell, dict) and cell.get("text") not in (None, ""):
                        texts.append(str(cell["text"]))
        return " ".join(texts)
    return ""


def _text_similarity(expected: str, actual: str) -> float:
    expected_clean = _compact_text(expected)
    actual_clean = _compact_text(actual)
    if not expected_clean and not actual_clean:
        return 1.0
    if not expected_clean or not actual_clean:
        return 0.0
    return SequenceMatcher(None, expected_clean, actual_clean).ratio()


def _text_diff_severity(expected: str, actual: str, similarity: float) -> str:
    if similarity >= TEXT_MINOR_DIFF_THRESHOLD:
        return "minor"
    if _small_non_numeric_text_change(expected, actual):
        return "minor"
    return "major"


def _small_non_numeric_text_change(expected: str, actual: str) -> bool:
    if _contains_digit(expected) or _contains_digit(actual):
        return False
    expected_clean = _compact_text(expected)
    actual_clean = _compact_text(actual)
    if not expected_clean or not actual_clean:
        return False
    if max(len(expected_clean), len(actual_clean)) > 80:
        return False
    return _edit_distance_limited(expected_clean, actual_clean, 2) <= 2


def _edit_distance_limited(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
            )
            row_min = min(row_min, current[-1])
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _financial_number_pair(expected: str, actual: str) -> tuple[Decimal, Decimal] | None:
    expected_number = _parse_financial_number(expected)
    actual_number = _parse_financial_number(actual)
    if expected_number is None or actual_number is None:
        return None
    return expected_number, actual_number


def _parse_financial_number(value: str) -> Decimal | None:
    text = str(value or "").strip()
    if not text or not _contains_digit(text):
        return None
    normalized = (
        text.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("￥", "")
        .replace("$", "")
        .replace("¥", "")
    )
    is_percent = normalized.endswith("%") or normalized.endswith("％")
    normalized = normalized.rstrip("%％")
    negative = False
    if (normalized.startswith("(") and normalized.endswith(")")) or (
        normalized.startswith("（") and normalized.endswith("）")
    ):
        negative = True
        normalized = normalized[1:-1]
    normalized = normalized.replace("+", "")
    match = re.fullmatch(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return None
    if negative and number > 0:
        number = -number
    if is_percent:
        return number
    return number


def _financial_number_severity(
    expected: str,
    actual: str,
    expected_number: Decimal,
    actual_number: Decimal,
) -> str:
    if (expected_number < 0 <= actual_number) or (actual_number < 0 <= expected_number):
        return "critical"
    if _first_significant_digit(expected_number) != _first_significant_digit(actual_number):
        return "critical"
    delta = abs(actual_number - expected_number)
    baseline = max(abs(expected_number), abs(actual_number), Decimal("1"))
    if delta / baseline >= Decimal("0.01"):
        return "critical"
    if _number_format_only(expected, actual):
        return "minor"
    return "major"


def _first_significant_digit(value: Decimal) -> str:
    text = format(abs(value), "f").replace(".", "").lstrip("0")
    return text[:1] or "0"


def _number_format_only(expected: str, actual: str) -> bool:
    left = _parse_financial_number(expected)
    right = _parse_financial_number(actual)
    return left is not None and right is not None and left == right


def _contains_digit(value: str) -> bool:
    return any(ch.isdigit() for ch in str(value or ""))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _bbox_similarity(expected: Any, actual: Any) -> float:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return 0.0
    expected_center = _bbox_center(expected)
    actual_center = _bbox_center(actual)
    expected_size = _bbox_size(expected)
    actual_size = _bbox_size(actual)
    if expected_center is None or actual_center is None or expected_size is None or actual_size is None:
        return 0.0

    center_distance = abs(expected_center[0] - actual_center[0]) + abs(expected_center[1] - actual_center[1])
    size_distance = abs(expected_size[0] - actual_size[0]) + abs(expected_size[1] - actual_size[1])
    return max(0.0, 1.0 - (center_distance / 80.0) - (size_distance / 160.0))


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float] | None:
    try:
        left = float(bbox["l"])
        right = float(bbox["r"])
        top = float(bbox["t"])
        bottom = float(bbox["b"])
    except (KeyError, TypeError, ValueError):
        return None
    return ((left + right) / 2, (top + bottom) / 2)


def _bbox_size(bbox: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return (abs(float(bbox["r"]) - float(bbox["l"])), abs(float(bbox["b"]) - float(bbox["t"])))
    except (KeyError, TypeError, ValueError):
        return None


def _is_bbox_number_path(path: str) -> bool:
    parts = path.split(".")
    if not parts:
        return False
    return parts[-1] in {"l", "t", "r", "b"} and any(part in {"bBox", "bbox"} for part in parts)


def _is_row_position_path(path: str) -> bool:
    if ".rows[" not in path:
        return False
    return path.rsplit(".", 1)[-1] in {"rowStart", "rowstart"}


def _numbers_close(expected: Any, actual: Any, tolerance: float) -> bool:
    try:
        return abs(float(expected) - float(actual)) <= tolerance
    except (TypeError, ValueError):
        return False


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return not isinstance(value, bool)


def _path_field_name(path: str) -> str:
    tail = path.rsplit(".", 1)[-1]
    if "[" in tail:
        return tail.split("[", 1)[0]
    return tail


def _add_diff(
    diffs: list[dict[str, Any]],
    path: str,
    expected: Any,
    actual: Any,
    *,
    diff_type: str,
    severity: str = "major",
) -> None:
    if _diff_limit_reached(diffs):
        return
    diffs.append(
        {
            "path": path,
            "expected": expected,
            "actual": actual,
            "type": diff_type,
            "severity": severity,
        }
    )


def _diff_limit_reached(diffs: list[dict[str, Any]]) -> bool:
    return len(diffs) >= MAX_DIFFS


def _diff_severity(diff: dict[str, Any]) -> str:
    severity = str(diff.get("severity") or "major").lower()
    return severity if severity in SEVERITY_ORDER else "major"


def _list_length_severity(expected_len: int, actual_len: int, *, has_removed: bool) -> str:
    if expected_len <= 0 and actual_len > 0:
        return "minor"
    if expected_len > 0 and actual_len <= 0:
        return "critical"
    delta = abs(actual_len - expected_len)
    ratio = delta / max(expected_len, actual_len, 1)
    if has_removed and ratio >= 0.5:
        return "critical"
    if has_removed:
        return "major"
    return "minor"
