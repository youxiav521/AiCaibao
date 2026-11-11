from __future__ import annotations

import re
import unicodedata
from typing import Any, List, Optional


def _is_cjk_char(ch: str) -> bool:
    """判断是否为常见中文（CJK）字符。"""
    if not ch:
        return False
    cp = ord(ch)
    # CJK Unified Ideographs + Extension A + Compatibility Ideographs
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0xF900 <= cp <= 0xFAFF)
    )


def _char_group_for_spacing(ch: str) -> str:
    """将字符归类为：cjk / en / punct / other。"""
    if _is_cjk_char(ch):
        return "cjk"
    cat = unicodedata.category(ch) or ""
    head = cat[:1]
    if head in {"P", "S"}:
        return "punct"
    # 字母/数字/组合符号：统一按“英文/数字”侧处理（便于与中文/标点分开）
    if head in {"L", "N", "M"}:
        return "en"
    return "other"


def add_spacing_for_tokenizer(text: Any) -> str:
    """在中文与英文/数字相邻处插入空格，便于基于空格的分词器切分。

    规则：
    - 保留换行（\\n），并清理换行前后的多余空格
    - 其他空白字符压缩为单个空格
    - 仅在 中文(cjk) ↔ 英文/数字(en) 切换时插入空格
    - 标点与其他字符之间不插入空格，避免破坏 SDIS_T_1_2、1.2 等连贯 token
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text

    # 统一 CRLF/CR 为 LF，避免后续处理出现重复换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    out: List[str] = []
    prev_group: Optional[str] = None

    for ch in text:
        # 全角空格统一成半角空格
        if ch == "\u3000":
            ch = " "

        # 保留换行，但去掉换行前后的冗余空格
        if ch == "\n":
            if out and out[-1] == " ":
                out.pop()
            out.append("\n")
            prev_group = None
            continue

        # 其他空白（空格/Tab等）压缩为单个空格
        if ch.isspace():
            if out and out[-1] not in {" ", "\n"}:
                out.append(" ")
            prev_group = None
            continue

        group = _char_group_for_spacing(ch)
        # 仅在 中文↔英文/数字 边界插入空格，标点不触发
        if out and out[-1] not in {" ", "\n"} and prev_group and group and prev_group != group:
            if {prev_group, group} == {"cjk", "en"}:
                out.append(" ")

        out.append(ch)
        prev_group = group

    res = "".join(out)
    # 再做一次轻量清理：去掉换行两侧空格、压缩多空格
    res = re.sub(r"[ \t]+\n", "\n", res)
    res = re.sub(r"\n[ \t]+", "\n", res)
    res = re.sub(r"[ \t]{2,}", " ", res)
    return res.strip()


