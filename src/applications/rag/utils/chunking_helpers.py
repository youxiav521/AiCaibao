from __future__ import annotations

import hashlib
from typing import Any


def hash_id(text: str, *, length: int = 12) -> str:
    """生成稳定的短哈希片段（用于切片/元素 ID 拼接）。"""
    text = text or ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[: int(length)]


def cell_text(cell: Any) -> str:
    """抽取单元格文本。支持 dict/其他类型。"""
    if isinstance(cell, dict):
        return str(cell.get("text", "")).strip()
    return str(cell or "").strip()

