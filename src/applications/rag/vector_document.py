from __future__ import annotations


def utf8_len(value: str) -> int:
    return len(value.encode("utf-8"))


def truncate_utf8(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    raw = value.encode("utf-8")
    if len(raw) <= max_bytes:
        return value
    return raw[:max_bytes].decode("utf-8", errors="ignore")
