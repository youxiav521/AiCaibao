from __future__ import annotations

import re

from src.utils.chinese_converter import traditional_to_simplified
from src.applications.parsing.utils.semantic_matcher import (
    normalize_accounting_terms,
    repair_truncation,
)

TRAILING_COLON_CHARS: tuple[str, ...] = (":", "：", "﹕", "︰", "∶", "꞉", "˸")
SUMMARY_SUFFIXES: tuple[str, ...] = ("合计", "小计", "总计")
NET_TYPE_SUFFIXES: tuple[str, ...] = ("净额", "净值", "净损失", "净收益")
FAMILY_SUFFIXES_DEDUP: tuple[str, ...] = (
    "净额", "净值", "原值", "摊销",
    "减值准备", "跌价准备", "坏账准备", "准备",
    "合计", "小计", "总计",
)


def normalize_subject_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^[一二三四五六七八九十]+、\s*", "", text)
    text = re.sub(r"^[（(][一二三四五六七八九十\d]+[）)]\s*", "", text)
    text = re.sub(r"^\d+[.、)）]\s*", "", text)
    text = re.sub(r"^[（(]\d+[）)]\s*", "", text)
    text = re.sub(r"^(其中|加|减|其中：|加：|减：)\s*[:：]?\s*", "", text)
    text = re.sub(r"[（(][^）)]*[）)]$", "", text)
    text = re.sub(r"[:：]\s*$", "", text)
    return text.strip()


def deep_normalize(text: str) -> str:
    value = normalize_subject_text(text)
    value = re.sub(r"\s+", "", value)
    value = traditional_to_simplified(value)
    value = repair_truncation(value)
    value = normalize_accounting_terms(value)
    return value


def template_text_key(text: str) -> str:
    if not text:
        return ""
    value = normalize_subject_text(text)
    value = traditional_to_simplified(value)
    return re.sub(r"\s+", "", value)


def has_trailing_heading_punctuation(text: str) -> bool:
    stripped = (text or "").rstrip()
    return bool(stripped) and stripped[-1] in TRAILING_COLON_CHARS


def classify_family_pair(deep_a: str, deep_b: str) -> str | None:
    if not deep_a or not deep_b:
        return None
    if deep_a == deep_b:
        return "identical"

    shorter, longer = (deep_a, deep_b) if len(deep_a) <= len(deep_b) else (deep_b, deep_a)
    if not longer.startswith(shorter):
        return None

    suffix = longer[len(shorter):]
    if suffix in FAMILY_SUFFIXES_DEDUP:
        return "suffix"
    if suffix and suffix[0] in ("及", "和", "与"):
        return "compound"
    return None
