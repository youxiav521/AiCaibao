from __future__ import annotations

from typing import Any, Optional

STATEMENT_DISPLAY_MAP: dict[str, dict[str, str]] = {
    "合并资产负债表": {"zh": "合并资产负债表", "en": "Consolidated Balance Sheet"},
    "母公司资产负债表": {"zh": "单体资产负债表", "en": "Standalone Balance Sheet"},
    "合并利润表": {"zh": "合并利润表", "en": "Consolidated Income Statement"},
    "母公司利润表": {"zh": "单体利润表", "en": "Standalone Income Statement"},
    "合并现金流量表": {"zh": "合并现金流量表", "en": "Consolidated Cashflow Statement"},
    "母公司现金流量表": {"zh": "单体现金流量表", "en": "Standalone Cashflow Statement"},
    "资产负债表": {"zh": "资产负债表", "en": "Balance Sheet"},
    "利润表": {"zh": "利润表", "en": "Income Statement"},
    "合并利润表及其他综合收益": {"zh": "合并利润表及其他综合收益", "en": "Consolidated P&L and OCI"},
    "母公司利润表及其他综合收益": {"zh": "单体利润表及其他综合收益", "en": "Standalone P&L and OCI"},
    "利润表及其他综合收益": {"zh": "利润表及其他综合收益", "en": "Statement of profit or loss and OCI"},
    "现金流量表": {"zh": "现金流量表", "en": "Cashflow Statement"},
    "合并所有者权益变动表": {"zh": "合并所有者权益变动表", "en": "Consolidated Statement of Changes in Equity"},
    "母公司所有者权益变动表": {"zh": "单体所有者权益变动表", "en": "Standalone Statement of Changes in Equity"},
    "所有者权益变动表": {"zh": "所有者权益变动表", "en": "Statement of Changes in Equity"},
    "其他表": {"zh": "其他表", "en": "Other"},
    "合并其他表": {"zh": "合并其他表", "en": "Consolidated Other"},
    "母公司其他表": {"zh": "母公司其他表", "en": "Standalone Other"},
}

FILING_PERIOD_I18N: dict[str, dict[str, str]] = {
    "年报": {"zh": "年报", "en": "Annual Report"},
    "半年报": {"zh": "半年报", "en": "Semi-Annual Report"},
    "一季度报": {"zh": "一季度报", "en": "First Quarter Report"},
    "三季度报": {"zh": "三季度报", "en": "Third Quarter Report"},
    "季报": {"zh": "季报", "en": "Quarterly Report"},
    "月报": {"zh": "月报", "en": "Monthly Report"},
}

CALIBER_LABEL_I18N: dict[str, dict[str, str]] = {
    "合并": {"zh": "合并", "en": "Consolidated"},
    "单体": {"zh": "单体", "en": "Standalone"},
}

CURRENCY_UNIT_ZH_TO_EN: dict[str, str] = {
    "元": "One",
    "千元": "Thousands",
    "万元": "Millions",
    "亿元": "Billions",
}


def locale_prefers_english(locale: Optional[str]) -> bool:
    if not locale or not isinstance(locale, str):
        return False
    return locale.strip().lower().startswith("en")


def translate_statement_label(label: str, locale: Optional[str]) -> str:
    mapping = STATEMENT_DISPLAY_MAP.get(label)
    if not mapping:
        return label
    if locale_prefers_english(locale):
        return mapping["en"]
    return mapping["zh"]


def translate_filing_period_for_response(period: Any, locale: Optional[str]) -> str:
    if period is None:
        return ""
    key = str(period).strip()
    if not key:
        return ""
    row = FILING_PERIOD_I18N.get(key)
    if not row:
        return key
    return row["en"] if locale_prefers_english(locale) else row["zh"]


def translate_caliber_label_for_response(zh: Optional[str], locale: Optional[str]) -> str:
    if zh is None:
        return ""
    key = str(zh).strip()
    if not key:
        return ""
    row = CALIBER_LABEL_I18N.get(key)
    if not row:
        return key
    return row["en"] if locale_prefers_english(locale) else row["zh"]


def apply_i18n_to_statement_table_response(entry: dict[str, Any], locale: Optional[str]) -> None:
    if not locale_prefers_english(locale):
        return
    currency_unit = entry.get("currencyUnit")
    if isinstance(currency_unit, str):
        mapped_currency_unit = CURRENCY_UNIT_ZH_TO_EN.get(currency_unit)
        if mapped_currency_unit is not None:
            entry["currencyUnit"] = mapped_currency_unit
    if "period" in entry:
        entry["period"] = translate_filing_period_for_response(entry.get("period"), locale)
    orig_cat = entry.get("category")
    if isinstance(orig_cat, str) and orig_cat.strip():
        mapped_cat = translate_statement_label(orig_cat, locale)
        if mapped_cat != orig_cat:
            entry["category"] = mapped_cat
            entry["text"] = mapped_cat
    caliber_label = entry.get("caliberLabel")
    if caliber_label is not None and caliber_label != "":
        entry["caliberLabel"] = translate_caliber_label_for_response(str(caliber_label), locale)
    rows = entry.get("rows")
    if not isinstance(rows, list):
        return
    for row in rows:
        cells = row if isinstance(row, list) else (row.get("cells") if isinstance(row, dict) else None)
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            cell_caliber_label = cell.get("caliberLabel")
            if cell_caliber_label is None or cell_caliber_label == "":
                continue
            cell["caliberLabel"] = translate_caliber_label_for_response(str(cell_caliber_label), locale)
