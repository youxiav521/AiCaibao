from __future__ import annotations

import re
from typing import Optional

from src.applications.parsing.service.three_statements.statement_labels.common import STATEMENT_ORDER

EN_BALANCE_SHEET_TITLES: tuple[str, ...] = (
    "balance sheet", "statement of financial position",
)
EN_INCOME_STATEMENT_TITLES: tuple[str, ...] = (
    "income statement", "profit and loss", "profit or loss",
    "statement of profit or loss",
    # "statement of comprehensive income" 不算主三大表（属于综合收益表，独立一类）；
    # P&L+OCI 合并展示由 normalize_pl_oci_combined_label 单独处理
    "analysis of income and expense", "analysis of revenue and expense",
)
EN_CASH_FLOW_TITLES: tuple[str, ...] = (
    "cash flow statement", "statement of cash flows",
    "cashflow statement", "cash flows",
)
EN_CHANGES_IN_EQUITY_TITLES: tuple[str, ...] = (
    "statement of changes in equity",
    "statements of changes in equity",
)
EN_SCOPE_CONSOLIDATED: tuple[str, ...] = (
    "consolidated",
)
EN_SCOPE_STANDALONE: tuple[str, ...] = (
    "standalone", "parent company", "parent only", "unconsolidated",
    "parent-only",
)
EN_POISON_PATTERNS: tuple[str, ...] = ("notes to", "note to", "annex to")


def apply_en_scope_tokens_to_pl_oci(text_lower: str) -> str:
    has_consolidated = any(kw in text_lower for kw in EN_SCOPE_CONSOLIDATED)
    has_standalone = any(kw in text_lower for kw in EN_SCOPE_STANDALONE)
    if has_consolidated and not has_standalone:
        return "合并利润表及其他综合收益"
    if has_standalone and not has_consolidated:
        return "母公司利润表及其他综合收益"
    return "利润表及其他综合收益"


def normalize_pl_oci_combined_label(blob: str) -> Optional[str]:
    if not blob:
        return None
    low = re.sub(r"\s+", " ", blob.lower())
    if "statement of cash flows" in low or "statement of cash flow" in low:
        return None
    # 只接受标题中明确写出 "profit or loss AND other comprehensive income" 的合并表。
    # 独立 OCI 表（标题如 "Statement of comprehensive income - Net of tax"）不在此列，
    # 应交由 LLM/规则归到"其他表"。
    if "statement of profit or loss and other comprehensive income" in low:
        return apply_en_scope_tokens_to_pl_oci(low)
    if "profit or loss and other comprehensive income" in low:
        return apply_en_scope_tokens_to_pl_oci(low)
    return None


def normalize_english_statement_label(text: str) -> Optional[str]:
    text_lower = text.lower()
    if any(pattern in text_lower for pattern in EN_POISON_PATTERNS):
        return None

    base_zh: Optional[str] = None
    for title in EN_CHANGES_IN_EQUITY_TITLES:
        if title in text_lower:
            base_zh = "所有者权益变动表"
            break
    if not base_zh:
        for title in EN_BALANCE_SHEET_TITLES:
            if title in text_lower:
                base_zh = "资产负债表"
                break
    if not base_zh:
        for title in EN_INCOME_STATEMENT_TITLES:
            if title in text_lower:
                base_zh = "利润表"
                break
    if not base_zh:
        for title in EN_CASH_FLOW_TITLES:
            if title in text_lower:
                base_zh = "现金流量表"
                break
    if not base_zh:
        return None

    scope_zh: Optional[str] = None
    for keyword in EN_SCOPE_CONSOLIDATED:
        if keyword in text_lower:
            scope_zh = "合并"
            break
    if not scope_zh:
        for keyword in EN_SCOPE_STANDALONE:
            if keyword in text_lower:
                scope_zh = "母公司"
                break

    if scope_zh:
        normalized = f"{scope_zh}{base_zh}"
        return normalized if normalized in STATEMENT_ORDER else None
    return base_zh if base_zh in STATEMENT_ORDER else None
