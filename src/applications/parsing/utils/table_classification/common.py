from __future__ import annotations

from typing import Any, Optional, Tuple

from src.utils.chinese_converter import traditional_to_simplified

LOCALE_ZH_CN: str = "zh_cn"
LOCALE_EN_US: str = "en_us"


def normalize_locale(locale: Any) -> str:
    """
    将 Accept-Language 或任意 locale 字符串归一为 zh_cn / en_us。
    未识别或空值默认 zh_cn（业务以中文报表为主，兼容历史行为）。
    """
    if not isinstance(locale, str):
        return LOCALE_ZH_CN
    text = locale.strip().lower().replace("-", "_")
    if not text:
        return LOCALE_ZH_CN
    primary = text.split(",")[0].split(";")[0].strip()
    if primary.startswith("en"):
        return LOCALE_EN_US
    return LOCALE_ZH_CN


def is_en_locale(locale: Optional[str]) -> bool:
    return normalize_locale(locale) == LOCALE_EN_US


STATEMENT_KEYWORDS: Tuple[str, ...] = ("资产负债表", "利润表", "现金流量表")
PROFIT_STATEMENT_VARIANTS: Tuple[str, ...] = (
    "合并利润表及其他综合收益",
    "母公司利润表及其他综合收益",
    "利润表及其他综合收益",
)

STATEMENT_TABLE_TYPE_LABELS: Tuple[str, ...] = (
    "合并资产负债表",
    "母公司资产负债表",
    "资产负债表",
    "合并利润表",
    "母公司利润表",
    "利润表",
    "合并现金流量表",
    "母公司现金流量表",
    "现金流量表",
    "合并所有者权益变动表",
    "母公司所有者权益变动表",
    "所有者权益变动表",
)
STATEMENT_TABLE_TYPE_LABELS_SET = set(STATEMENT_TABLE_TYPE_LABELS) | set(PROFIT_STATEMENT_VARIANTS)
ALLOWED_TABLE_TYPE_LABELS: Tuple[str, ...] = (
    "合并资产负债表",
    "母公司资产负债表",
    "资产负债表",
    "合并利润表",
    "母公司利润表",
    "利润表",
    "合并现金流量表",
    "母公司现金流量表",
    "现金流量表",
    "合并所有者权益变动表",
    "母公司所有者权益变动表",
    "所有者权益变动表",
    "其他表",
)
ALLOWED_TABLE_TYPE_LABELS_SET = set(ALLOWED_TABLE_TYPE_LABELS) | set(PROFIT_STATEMENT_VARIANTS)

STATEMENT_CONTENT_KEYWORDS_ZH: Tuple[str, ...] = (
    "资产", "负债", "权益", "利润", "净利", "收入",
    "现金流", "经营活动", "投资活动", "筹资活动",
)
STATEMENT_CONTENT_KEYWORDS_EN: Tuple[str, ...] = (
    "assets", "liabilities", "equity", "profit", "revenue",
    "cash flow", "operating", "investing", "financing",
)

LARGE_TABLE_SET_THRESHOLD: int = 40
MIN_ROW_COUNT_FOR_STATEMENT: int = 10
MIN_COL_COUNT_FOR_STATEMENT: int = 2


def compact_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return "".join(text.split())


def contains_any(text: str, terms: Tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def count_contains(text: str, terms: Tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in text)


def has_statement_content_keywords(table_md: str, locale: Any = None) -> bool:
    """
    按 locale 分流检查表格内容是否含三大表关键词。
    locale 显式指定 en_us / zh_cn 时只检查对应词表；
    locale 缺省（None/空）时同时检查中英文，兼容历史行为。
    """
    if not isinstance(table_md, str) or not table_md.strip():
        return False
    compact_lower = "".join(traditional_to_simplified(table_md).split()).lower()
    if isinstance(locale, str) and locale.strip():
        _locale = normalize_locale(locale)
        if _locale == LOCALE_EN_US:
            return any(kw in compact_lower for kw in STATEMENT_CONTENT_KEYWORDS_EN)
        return any(kw in compact_lower for kw in STATEMENT_CONTENT_KEYWORDS_ZH)
    if any(kw in compact_lower for kw in STATEMENT_CONTENT_KEYWORDS_ZH):
        return True
    if any(kw in compact_lower for kw in STATEMENT_CONTENT_KEYWORDS_EN):
        return True
    return False


def normalize_profit_statement_label(label: str) -> str:
    """
    归一化利润表标签：将各种"利润表及其他综合收益"变体统一为对应的标准标签
    """
    if not isinstance(label, str):
        return label
    if label == "合并利润表及其他综合收益":
        return "合并利润表"
    if label == "母公司利润表及其他综合收益":
        return "母公司利润表"
    if label == "利润表及其他综合收益":
        return "利润表"
    return label


def coerce_allowed_label(label: Any) -> str:
    if not isinstance(label, str):
        return "其他表"
    stripped = label.strip()
    if stripped in ALLOWED_TABLE_TYPE_LABELS_SET:
        normalized = normalize_profit_statement_label(stripped)
        return normalized
    return "其他表"
