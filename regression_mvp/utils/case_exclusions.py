from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_EXCLUSIONS_FILE = "compare_exclusions.json"
ITEM_SCOPE = "item"
PAGE_SCOPE = "page"


def load_case_compare_exclusions(case: Any) -> list[dict[str, Any]]:
    expected_file = getattr(case, "expected_file", None)
    if not isinstance(expected_file, Path):
        return []
    return load_compare_exclusions(expected_file.parent)


def load_compare_exclusions(case_dir: Path) -> list[dict[str, Any]]:
    config_path = case_dir / "case.yaml"
    file_name = _read_simple_yaml(config_path).get("compare_exclusions_file", DEFAULT_EXCLUSIONS_FILE)
    exclusions_path = _resolve_exclusions_path(case_dir, file_name)
    if exclusions_path is None or not exclusions_path.is_file():
        return []
    try:
        payload = json.loads(exclusions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rules = payload.get("rules") if isinstance(payload, dict) else payload
    if not isinstance(rules, list):
        return []
    return [rule for rule in rules if _is_active_rule(rule)]


def filter_compare_exclusions(
    expected: Any,
    actual: Any,
    exclusions: list[dict[str, Any]] | None,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Return comparison inputs with approved page/item rules removed on both sides."""
    if not exclusions or not _is_item_list(expected) or not _is_item_list(actual):
        return expected, actual, []

    filtered_expected = list(expected)
    filtered_actual = list(actual)
    applied: list[dict[str, Any]] = []
    for rule in exclusions:
        if not _is_active_rule(rule):
            continue
        kept_expected, removed_expected = _filter_side(filtered_expected, rule)
        kept_actual, removed_actual = _filter_side(filtered_actual, rule)
        filtered_expected = kept_expected
        filtered_actual = kept_actual
        if removed_expected or removed_actual:
            applied.append(
                {
                    "id": str(rule.get("id") or ""),
                    "scope": str(rule.get("scope") or ITEM_SCOPE),
                    "reason": str(rule.get("reason") or "未填写原因"),
                    "expected_count": removed_expected,
                    "actual_count": removed_actual,
                    "selector": dict(rule.get("selector") or {}),
                }
            )
    return filtered_expected, filtered_actual, applied


def _filter_side(items: list[dict[str, Any]], rule: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for item in items:
        if _item_matches_rule(item, rule):
            removed += 1
        else:
            kept.append(item)
    return kept, removed


def _item_matches_rule(item: dict[str, Any], rule: dict[str, Any]) -> bool:
    selector = rule.get("selector")
    if not isinstance(selector, dict):
        return False
    expected_page = _positive_int(selector.get("page"))
    if expected_page is None or _item_page(item) != expected_page:
        return False
    scope = str(rule.get("scope") or ITEM_SCOPE)
    if scope == PAGE_SCOPE:
        return True
    if scope != ITEM_SCOPE:
        return False
    expected_type = str(selector.get("type") or "").strip()
    if expected_type and _item_type(item) != expected_type:
        return False
    expected_bbox = selector.get("bbox")
    actual_bbox = item.get("bBox") or item.get("bbox")
    return isinstance(expected_bbox, dict) and isinstance(actual_bbox, dict) and _bbox_matches(expected_bbox, actual_bbox)


def _bbox_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_values = _bbox_values(expected)
    actual_values = _bbox_values(actual)
    if expected_values is None or actual_values is None:
        return False
    if all(abs(left - right) <= 20.0 for left, right in zip(expected_values, actual_values)):
        return True
    left = max(expected_values[0], actual_values[0])
    top = max(expected_values[1], actual_values[1])
    right = min(expected_values[2], actual_values[2])
    bottom = min(expected_values[3], actual_values[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    expected_area = max(0.0, expected_values[2] - expected_values[0]) * max(0.0, expected_values[3] - expected_values[1])
    actual_area = max(0.0, actual_values[2] - actual_values[0]) * max(0.0, actual_values[3] - actual_values[1])
    return overlap > 0 and overlap / max(min(expected_area, actual_area), 1.0) >= 0.5


def _bbox_values(bbox: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(bbox[key]) for key in ("l", "t", "r", "b"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError):
        return None


def _item_page(item: dict[str, Any]) -> int | None:
    for key in ("page", "page_no", "pageAt", "source_page_no", "sourcePageNo"):
        value = _positive_int(item.get(key))
        if value is not None:
            return value
    return None


def _item_type(item: dict[str, Any]) -> str:
    value = str(item.get("type") or "").strip()
    if value:
        return value
    return "table" if isinstance(item.get("rows"), list) else "unknown"


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _is_item_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _is_active_rule(rule: Any) -> bool:
    return isinstance(rule, dict) and str(rule.get("status") or "active").lower() == "active"


def _resolve_exclusions_path(case_dir: Path, file_name: Any) -> Path | None:
    candidate = (case_dir / str(file_name or DEFAULT_EXCLUSIONS_FILE)).resolve()
    try:
        candidate.relative_to(case_dir.resolve())
    except ValueError:
        return None
    return candidate


def _read_simple_yaml(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
