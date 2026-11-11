from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from src.applications.parsing.utils.table_classification.common import (
    STATEMENT_TABLE_TYPE_LABELS_SET,
    count_contains,
    contains_any,
)

EN_BALANCE_TERMS: Tuple[str, ...] = (
    "total assets", "total liabilities", "total equity",
    "current assets", "non-current assets",
    "current liabilities", "non-current liabilities",
    "shareholders' equity", "stockholders' equity",
    "total liabilities and equity", "total equity and liabilities",
    "owners' equity",
)
EN_PROFIT_TERMS: Tuple[str, ...] = (
    "total revenue", "operating revenue",
    "net income", "net profit", "profit for the year",
    "profit for the period", "profit for period",
    "gross profit", "operating income", "operating profit",
    "total comprehensive income", "earnings per share",
    "profit before tax", "profit before income tax",
    "income tax expense",
    "net sales", "turnover", "sales",
    "net loss", "loss for the year",
    "loss for the period", "profit from operations",
)
EN_CASH_TERMS: Tuple[str, ...] = (
    "operating activities",
    "investing activities",
    "financing activities",
    "cash flows from operating",
    "cash flows from investing",
    "cash flows from financing",
    "cash flow from operating",
    "cash flow from investing",
    "cash flow from financing",
    "net cash from operating",
    "net cash used in",
    "cash and cash equivalents",
    "cash at end of",
    "cash at beginning of",
    "net increase in cash",
    "net change in cash",
)
IFRS_PARENS_RE = re.compile(r"\s*\((?:loss(?:es)?|gain(?:s)?)\)", re.IGNORECASE)
EN_SCOPE_CONSOLIDATED_KW: Tuple[str, ...] = ("consolidated",)
EN_SCOPE_STANDALONE_KW: Tuple[str, ...] = (
    "standalone", "parent company", "parent only", "unconsolidated", "parent-only",
)
EN_OTHER_DIRECT_PHRASES: Tuple[str, ...] = (
    "statement of changes in equity",
    "changes in equity",
    "changes in shareholders equity",
    "changes in owners equity",
    "movements during the financial year",
    "risk area and rationale",
    "our response",
    "maturity analysis",
    "ageing analysis",
    "fair value of financial instruments",
    "financial instruments carried at fair value",
    "key management personnel",
    "related party",
    "directors fees and remuneration",
    "share based payment",
    "contingent liabilities",
    "capital commitments",
    "operating segments",
    "business segments",
    "geographical segments",
    "rating agency",
    "rating classification",
    "principal activities",
    "name of company",
)
EN_EQUITY_CHANGE_OTHER_TERMS: Tuple[str, ...] = (
    "attributable to owners",
    "owners of the company",
    "attributable to equity holders",
    "share capital",
    "retained earnings",
    "non controlling interests",
    "total equity",
    "reserve",
    "reserves",
    "translation reserve",
    "treasury shares",
    "dividends",
    "balance at",
    "comprehensive income",
)
EN_EQUITY_CHANGE_ANCHORS: Tuple[str, ...] = (
    "attributable to owners",
    "attributable to equity holders",
    "owners of the company",
    "balance at",
    "dividends",
    "translation reserve",
    "treasury shares",
)
EN_FINANCIAL_NOTE_OTHER_TERMS: Tuple[str, ...] = (
    "financial instruments",
    "risk management",
    "fair value",
    "credit risk",
    "liquidity risk",
    "sensitivity analysis",
    "expected credit losses",
    "stage 1",
    "stage 2",
    "stage 3",
    "12 month ecl",
    "lifetime ecl",
)
EN_DETAIL_NOTE_OTHER_TERMS: Tuple[str, ...] = (
    "property plant and equipment",
    "right of use assets",
    "investment properties",
    "inventories",
    "trade and other receivables",
    "trade and other payables",
    "borrowings",
    "lease liabilities",
    "deferred tax",
    "income tax",
    "commitments",
    "contingencies",
    "remuneration",
    "allowance",
    "impairment",
    "carrying amount",
    "reconciliation",
    "movement",
)
EN_DETAIL_NOTE_ANCHORS: Tuple[str, ...] = (
    "carrying amount",
    "reconciliation",
    "movement",
    "allowance",
    "impairment",
    "fair value",
    "as at",
)
EN_PRIMARY_STATEMENT_TITLE_PHRASES: Tuple[str, ...] = (
    "statement of financial position",
    "balance sheet",
    "income statement",
    "statement of profit or loss",
    # "statement of comprehensive income" 不算主三大表（属于综合收益表，独立一类，
    # 仅当 P&L+OCI 合并展示时才由 normalize_pl_oci_combined_label 归到"利润表+OCI"）
    "statement of operations",
    "statement of cash flow",
    "statement of cash flows",
    "cash flow statement",
    "statement of changes in equity",
    "statement of changes in shareholders equity",
)

# 对齐中文 NON_STATEMENT_EXACT_TITLES：标题恰好等于这些词时直接判其他表
EN_NON_STATEMENT_EXACT_TITLES: frozenset[str] = frozenset({
    "cash flow",
    "cash flows",
})

# 对齐中文 NON_FINANCIAL_TITLE_TERMS：标题中出现这些短语时判其他表（仅在 use_extended_rules 时启用）
EN_NON_FINANCIAL_TITLE_TERMS: Tuple[str, ...] = (
    "comprehensive income breakdown",
    "revenue breakdown",
    "expense breakdown",
    "cost breakdown",
    "composition",
    "introduction",
    "overview",
    "customers",
    "suppliers",
    "ranking",
    "competition",
    "core technology",
    "r and d expenditure",
    "research and development",
    "bad debt provision",
    "long term equity investment",
    "deferred tax",
    "bonds payable",
    "contract assets",
    "contract liabilities",
    "inventories",
    "prepayments",
    "other receivables",
    "goodwill",
    "intangible assets",
    "employee benefits payable",
    "short term compensation",
    "post employment benefits",
    "associates",
    "joint ventures",
    "investments in subsidiaries",
    "non recurring profit or loss",
    "principal business",
    "fair value",
    "independent directors",
    "board of directors",
    "supervisory board",
    "audit committee",
    "remuneration and nomination",
    "guarantees",
    "accounting firm",
    "shareholding",
    "restricted shares",
    "total share capital",
    "key audit matters",
    "definitions",
    "changes in shareholders equity",
    "changes in owners equity",
    "changes in equity",
)

# 对齐中文 derivative_markers：标题含主表关键短语 + 这些尾缀 → 衍生标题
EN_DERIVATIVE_TAIL_MARKERS: Tuple[str, ...] = (
    "notes",
    "note to",
    "notes to",
    "schedule",
    "schedules",
    "breakdown",
    "composition",
    "analysis",
    "analyses",
    "movement",
    "movements",
    "reconciliation",
    "reconciliations",
    "details",
    "detail",
    "supplementary",
    "supplemental",
    "supplement",
    "disclosure",
    "disclosures",
    "explanation",
    "explanations",
    "description",
)


def strip_ifrs_parens(text: str) -> str:
    return re.sub(r"\s+", " ", IFRS_PARENS_RE.sub(" ", text)).strip()


def normalize_en_rule_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    normalized = text.lower().replace("&", " and ")
    normalized = re.sub(r"[’'`\u2018\u2019]", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def has_primary_statement_title_phrase(text: str) -> bool:
    normalized = normalize_en_rule_text(text)
    if not normalized:
        return False
    return any(phrase in normalized for phrase in EN_PRIMARY_STATEMENT_TITLE_PHRASES)


def is_non_statement_exact_title_en(label: Any) -> bool:
    """对齐 zh_rules.is_non_statement_exact_title：英文标题恰好等于排除词时为 True。"""
    normalized = normalize_en_rule_text(label)
    if not normalized:
        return False
    return normalized in EN_NON_STATEMENT_EXACT_TITLES


def is_non_financial_title_en(label: Any) -> bool:
    """对齐 zh_rules.is_non_financial_title：标题命中非财务术语时为 True。"""
    normalized = normalize_en_rule_text(label)
    if not normalized:
        return False
    return any(term in normalized for term in EN_NON_FINANCIAL_TITLE_TERMS)


def is_derivative_statement_title_en(label: Any) -> bool:
    """
    对齐 zh_rules.is_derivative_statement_title：
    标题包含主表短语（balance sheet / income statement / cash flow statement 等），
    且尾部仍有 notes/schedule/breakdown/movement 等衍生词，则视为衍生标题。
    """
    normalized = normalize_en_rule_text(label)
    if not normalized:
        return False
    if not has_primary_statement_title_phrase(normalized):
        return False
    return any(marker in normalized for marker in EN_DERIVATIVE_TAIL_MARKERS)


def is_extended_non_financial_english_table(label: str, table_md: str) -> bool:
    label_text = normalize_en_rule_text(label)
    table_text = normalize_en_rule_text(table_md)
    combined = f"{label_text} {table_text}".strip()
    if not combined:
        return False

    if has_primary_statement_title_phrase(label_text):
        return False

    if any(phrase in combined for phrase in EN_OTHER_DIRECT_PHRASES):
        return True

    equity_score = count_contains(combined, EN_EQUITY_CHANGE_OTHER_TERMS)
    has_equity_anchor = contains_any(combined, EN_EQUITY_CHANGE_ANCHORS)
    if has_equity_anchor and equity_score >= 3:
        return True

    financial_note_score = count_contains(combined, EN_FINANCIAL_NOTE_OTHER_TERMS)
    if financial_note_score >= 2:
        return True

    detail_note_score = count_contains(combined, EN_DETAIL_NOTE_OTHER_TERMS)
    has_detail_anchor = contains_any(combined, EN_DETAIL_NOTE_ANCHORS)
    if has_detail_anchor and detail_note_score >= 2:
        return True

    return False


def apply_en_scope(base: str, text_lower: str) -> str:
    has_consolidated = any(kw in text_lower for kw in EN_SCOPE_CONSOLIDATED_KW)
    has_standalone = any(kw in text_lower for kw in EN_SCOPE_STANDALONE_KW)

    if has_consolidated:
        scoped = f"合并{base}"
        if scoped in STATEMENT_TABLE_TYPE_LABELS_SET:
            return scoped
    if has_standalone and not has_consolidated:
        scoped = f"母公司{base}"
        if scoped in STATEMENT_TABLE_TYPE_LABELS_SET:
            return scoped
    return base


def is_full_en_statement_table_md(table_md: str, label: str) -> bool:
    raw_lower = re.sub(r"\s+", " ", (table_md or "").lower())
    if not raw_lower.strip():
        return False
    text_lower = strip_ifrs_parens(raw_lower)

    if "资产负债表" in label:
        has_total_assets = "total assets" in text_lower
        has_liab_and_equity = any(t in text_lower for t in (
            "total liabilities and equity",
            "total equity and liabilities",
            "total liabilities and shareholders",
        ))
        has_total_liab = "total liabilities" in text_lower
        has_total_equity = any(t in text_lower for t in (
            "total equity", "shareholders' equity",
            "stockholders' equity", "owners' equity",
        ))
        return has_total_assets and (has_liab_and_equity or (has_total_liab and has_total_equity))

    if "利润表" in label:
        has_revenue = any(t in text_lower for t in (
            "revenue", "total revenue", "operating revenue",
            "sales", "net sales", "turnover",
        ))
        has_net = any(t in text_lower for t in (
            "net income", "net profit", "profit for the year",
            "profit for the period", "profit for period",
            "net loss", "loss for the year",
            "loss for the period", "loss for period",
        ))
        has_extra = any(t in text_lower for t in (
            "profit before tax", "profit before income tax",
            "earnings per share", "total comprehensive income",
            "income tax", "income tax expense", "operating profit",
            "gross profit", "profit from operations",
        ))
        return has_revenue and has_net and has_extra

    if "现金流量表" in label:
        has_operating = (
            "operating activities" in text_lower
            or "cash flows from operating" in text_lower
            or "cash flow from operating" in text_lower
        )
        has_investing = (
            "investing activities" in text_lower
            or "cash flows from investing" in text_lower
            or "cash flow from investing" in text_lower
        )
        has_financing = (
            "financing activities" in text_lower
            or "cash flows from financing" in text_lower
            or "cash flow from financing" in text_lower
        )
        activity_sections = sum(1 for x in (has_operating, has_investing, has_financing) if x)
        has_cf_anchor = any(
            t in text_lower
            for t in (
                "net increase in cash",
                "net decrease in cash",
                "increase in cash and cash equivalents",
                "cash and cash equivalents at end",
                "cash and cash equivalents at beginning",
                "net cash generated from operating activities",
                "statement of cash flows",
                "cash flow statement",
            )
        )
        return activity_sections >= 3 or (activity_sections >= 2 and has_cf_anchor)

    return False


def detect_statement_type_by_content_en(table_md: str) -> Optional[str]:
    if not table_md:
        return None
    text_lower = strip_ifrs_parens(re.sub(r"\s+", " ", table_md.lower()))

    for label in ("资产负债表", "利润表", "现金流量表"):
        if is_full_en_statement_table_md(table_md, label):
            return apply_en_scope(label, text_lower)

    balance_score = sum(1 for t in EN_BALANCE_TERMS if t in text_lower)
    profit_score = sum(1 for t in EN_PROFIT_TERMS if t in text_lower)
    cash_score = sum(1 for t in EN_CASH_TERMS if t in text_lower)

    best_score = max(balance_score, profit_score, cash_score)
    if best_score < 3:
        if cash_score >= 2 and cash_score > balance_score and cash_score > profit_score:
            return apply_en_scope("现金流量表", text_lower)
        if profit_score >= 2 and ("comprehensive income" in text_lower or "items that may be reclassified" in text_lower):
            return apply_en_scope("利润表", text_lower)
        return None

    base: Optional[str] = None
    if balance_score >= 3 and balance_score > max(profit_score, cash_score):
        base = "资产负债表"
    elif cash_score >= 3 and cash_score > max(balance_score, profit_score):
        base = "现金流量表"
    elif profit_score >= 3 and profit_score > max(balance_score, cash_score):
        base = "利润表"

    if base is None:
        return None
    return apply_en_scope(base, text_lower)
