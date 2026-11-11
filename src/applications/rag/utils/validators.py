from __future__ import annotations

from typing import Any


_INVALID_FILENAME_CHARS = set('<>:"/\\\\|?*')


def validate_file_id(value: Any) -> str:
    """验证 Milvus 字段和本地存储文件命名的 file_id。"""
    file_id = "" if value is None else str(value).strip()
    if not file_id:
        raise ValueError("file_id is empty")
    if len(file_id) > 256:
        raise ValueError("file_id is too long (max 256)")
    if any((ch in _INVALID_FILENAME_CHARS) for ch in file_id):
        raise ValueError('file_id contains invalid filename characters: <>:"/\\\\|?*')
    if any(ord(ch) < 32 for ch in file_id):
        raise ValueError("file_id contains control characters")
    if file_id.endswith("."):
        raise ValueError("file_id must not end with '.'")
    return file_id



