from __future__ import annotations

import re
import unicodedata
from typing import Any


DEFAULT_IGNORE_FIELDS = {
    "trace_id",
    "request_id",
    "created_at",
    "updated_at",
    "task_id",
    "file_id",
    "fileId",
    "folder_id",
    "elapsed",
    "elapsed_ms",
    "duration",
    "duration_ms",
    "url",
    "uri",
    "path",
    "tmp_path",
    "temp_path",
    "img_path",
    "image_path",
    "source_path",
    "metadata_path",
}

INVISIBLE_CHARS_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "＂": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "＇": "'",
    }
)


def normalize_items(value: Any, compare_config: dict[str, Any]) -> Any:
    ignore_fields = DEFAULT_IGNORE_FIELDS | set(compare_config.get("ignore_fields") or [])
    normalized = _normalize(value, ignore_fields)

    if (
        isinstance(normalized, list)
        and not compare_config.get("strict_order", True)
        and all(isinstance(item, dict) for item in normalized)
    ):
        normalized = sorted(normalized, key=_stable_item_key)

    return normalized


def _normalize(value: Any, ignore_fields: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize(value[key], ignore_fields)
            for key in sorted(value)
            if key not in ignore_fields
        }

    if isinstance(value, list):
        return [_normalize(item, ignore_fields) for item in value]

    if isinstance(value, str):
        return _normalize_text(value)

    if value is None:
        return ""

    return value


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.translate(QUOTE_TRANSLATION)
    text = INVISIBLE_CHARS_RE.sub("", text)
    text = CONTROL_CHARS_RE.sub("", text)
    return text.strip()


def _stable_item_key(item: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, Any]:
    bbox = item.get("bBox") or item.get("bbox") or {}
    return (
        item.get("page") or item.get("pageAt") or "",
        item.get("type") or "",
        _safe_float(bbox.get("t")),
        _safe_float(bbox.get("l")),
        item.get("element_id", ""),
        item.get("text") or item.get("value") or item.get("md") or "",
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
