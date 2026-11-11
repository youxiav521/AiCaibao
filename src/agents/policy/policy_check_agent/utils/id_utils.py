from __future__ import annotations

import hashlib


def stable_id(prefix: str, *parts: object) -> str:
    raw = "||".join(str(part) for part in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
