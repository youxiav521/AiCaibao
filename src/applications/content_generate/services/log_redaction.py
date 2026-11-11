"""Safe logging helpers for content-generation payloads."""

from __future__ import annotations

import hashlib
import re
from typing import Any


_DATA_URI_PATTERN = re.compile(r"^data:(image/[^;,]+);base64,", re.IGNORECASE)


def redact_data_uris_for_log(value: Any) -> Any:
    """Replace image Data URIs with compact, non-reversible audit metadata."""
    if isinstance(value, dict):
        return {key: redact_data_uris_for_log(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_data_uris_for_log(item) for item in value]
    if not isinstance(value, str):
        return value

    match = _DATA_URI_PATTERN.match(value)
    if match is None:
        return value
    return {
        "redacted": True,
        "mime": match.group(1).lower(),
        "chars": len(value),
        "sha256_12": hashlib.sha256(value.encode("utf-8")).hexdigest()[:12],
    }
