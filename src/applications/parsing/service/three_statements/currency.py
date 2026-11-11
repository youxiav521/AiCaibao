from __future__ import annotations

import re
from typing import Any, Callable, Optional

from src.applications.parsing.service.three_statements.cells import (
    get_raw_cell_text,
    normalize_raw_rows,
)
from src.applications.parsing.service.three_statements.i18n import (
    CURRENCY_UNIT_ZH_TO_EN,
    locale_prefers_english,
)
from src.utils.chinese_converter import traditional_to_simplified

CURRENCY_UNIT_KIND_ONE = "one"
CURRENCY_UNIT_KIND_THOUSANDS = "thousands"
CURRENCY_UNIT_KIND_MILLIONS = "millions"
CURRENCY_UNIT_KIND_BILLIONS = "billions"

CURRENCY_UNIT_KIND_TO_ZH: dict[str, str] = {
    CURRENCY_UNIT_KIND_ONE: "元",
    CURRENCY_UNIT_KIND_THOUSANDS: "千元",
    CURRENCY_UNIT_KIND_MILLIONS: "万元",
    CURRENCY_UNIT_KIND_BILLIONS: "亿元",
}
CURRENCY_UNIT_ZH_TO_KIND: dict[str, str] = {
    value: key for key, value in CURRENCY_UNIT_KIND_TO_ZH.items()
}
CURRENCY_UNIT_EN_TO_KIND: dict[str, str] = {
    value.lower(): CURRENCY_UNIT_ZH_TO_KIND[key]
    for key, value in CURRENCY_UNIT_ZH_TO_EN.items()
}

RM_TEN_THOUSAND_RE = re.compile(r"[Rr][Mm]\s*'\s*10\s*[,.']?\s*000(?![0-9])")
RM_10000_COMPACT_RE = re.compile(r"[Rr][Mm]\s*'\s*10000(?![0-9])")
RM_4Z_RE = re.compile(r"[Rr][Mm]\s*'\s*0{4}(?![0-9])")
RM_3Z_RE = re.compile(r"[Rr][Mm]\s*'\s*0{3}(?![0-9])")
RM_1_RE = re.compile(r"[Rr][Mm]\s*'\s*1(?![0-9])")
RM_0_RE = re.compile(r"[Rr][Mm]\s*'\s*0(?![0-9])")

RE_BILLIONS_WORD = re.compile(r"\bBillions\b", re.I)
RE_MILLIONS_WORD = re.compile(r"\bMillions\b", re.I)
RE_THOUSANDS_WORD = re.compile(r"\bThousands\b", re.I)
RE_ONE_WORD = re.compile(r"\bOne\b", re.I)

TableCaptionGetter = Callable[[dict[str, Any]], str]


def normalize_apostrophes_for_rm_scan(text: str) -> str:
    out: list[str] = []
    for char in text:
        if char in ("\u2019", "\u2018", "\u2032", "`"):
            out.append("'")
        else:
            out.append(char)
    return "".join(out)


def collect_currency_scan_text(table_el: dict[str, Any], extra_text: str = "") -> str:
    parts: list[str] = []
    md_text = table_el.get("md") or ""
    if isinstance(md_text, str) and md_text.strip():
        parts.append(md_text)
    rows = normalize_raw_rows(table_el.get("rows"))
    for row in rows[:12]:
        for cell in row:
            cell_text = get_raw_cell_text(cell)
            if cell_text:
                parts.append(cell_text)
    if extra_text.strip():
        parts.append(extra_text)
    return "\n".join(parts)


def scan_blob_for_currency_kind(blob: str) -> Optional[str]:
    if not blob.strip():
        return None

    normalized = normalize_apostrophes_for_rm_scan(blob)
    if (
        RM_TEN_THOUSAND_RE.search(normalized)
        or RM_10000_COMPACT_RE.search(normalized)
        or RM_4Z_RE.search(normalized)
    ):
        return CURRENCY_UNIT_KIND_MILLIONS
    if RM_3Z_RE.search(normalized):
        return CURRENCY_UNIT_KIND_THOUSANDS
    if RM_1_RE.search(normalized):
        return CURRENCY_UNIT_KIND_ONE
    if RM_0_RE.search(normalized):
        return CURRENCY_UNIT_KIND_ONE

    if RE_BILLIONS_WORD.search(normalized):
        return CURRENCY_UNIT_KIND_BILLIONS
    if RE_MILLIONS_WORD.search(normalized):
        return CURRENCY_UNIT_KIND_MILLIONS
    if RE_THOUSANDS_WORD.search(normalized):
        return CURRENCY_UNIT_KIND_THOUSANDS
    if RE_ONE_WORD.search(normalized):
        return CURRENCY_UNIT_KIND_ONE

    zh = traditional_to_simplified(blob)
    if "亿元" in zh:
        return CURRENCY_UNIT_KIND_BILLIONS
    if "万元" in zh:
        return CURRENCY_UNIT_KIND_MILLIONS
    if "千元" in zh:
        return CURRENCY_UNIT_KIND_THOUSANDS
    return None


def detect_currency_unit_kind(
    table_el: dict[str, Any],
    context_above: str = "",
    previous_unit: Optional[str] = None,
    *,
    table_caption_getter: Optional[TableCaptionGetter] = None,
) -> str:
    caption_text = table_caption_getter(table_el) if table_caption_getter else ""
    if caption_text:
        kind = scan_blob_for_currency_kind(caption_text)
        if kind:
            return kind
        if "元" in caption_text:
            return CURRENCY_UNIT_KIND_ONE

    context_above = (context_above or "")[-100:]
    if context_above:
        kind = scan_blob_for_currency_kind(context_above)
        if kind:
            return kind

    blob = collect_currency_scan_text(table_el, extra_text="")
    if blob:
        kind = scan_blob_for_currency_kind(blob)
        if kind:
            return kind

    if previous_unit:
        previous_kind = CURRENCY_UNIT_ZH_TO_KIND.get(previous_unit)
        if not previous_kind:
            previous_kind = CURRENCY_UNIT_EN_TO_KIND.get(previous_unit.lower())
        if not previous_kind:
            previous_kind = scan_blob_for_currency_kind(previous_unit)
        if previous_kind:
            return previous_kind

    return CURRENCY_UNIT_KIND_ONE


def currency_unit_zh_for_table(
    table_el: dict[str, Any],
    context_above: str = "",
    previous_unit: Optional[str] = None,
    *,
    table_caption_getter: Optional[TableCaptionGetter] = None,
) -> str:
    kind = detect_currency_unit_kind(
        table_el,
        context_above=context_above,
        previous_unit=previous_unit,
        table_caption_getter=table_caption_getter,
    )
    return CURRENCY_UNIT_KIND_TO_ZH.get(kind, "元")


def currency_unit_for_api(
    table_el: dict[str, Any],
    locale: Optional[str],
    context_above: str = "",
    previous_unit: Optional[str] = None,
    *,
    table_caption_getter: Optional[TableCaptionGetter] = None,
) -> str:
    zh = currency_unit_zh_for_table(
        table_el,
        context_above=context_above,
        previous_unit=previous_unit,
        table_caption_getter=table_caption_getter,
    )
    if locale_prefers_english(locale):
        return CURRENCY_UNIT_ZH_TO_EN.get(zh, "One")
    return zh
