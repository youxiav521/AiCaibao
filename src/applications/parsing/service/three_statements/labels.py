from __future__ import annotations

import re
from typing import Any, Optional

from src.utils.chinese_converter import traditional_to_simplified

from src.applications.parsing.service.three_statements.statement_labels.common import (
    STATEMENT_ORDER,
    THREE_MAIN_BASE_TYPES,
)
from src.applications.parsing.service.three_statements.statement_labels.en import (
    EN_BALANCE_SHEET_TITLES,
    EN_CASH_FLOW_TITLES,
    EN_CHANGES_IN_EQUITY_TITLES,
    EN_INCOME_STATEMENT_TITLES,
    EN_POISON_PATTERNS,
    EN_SCOPE_CONSOLIDATED,
    EN_SCOPE_STANDALONE,
    apply_en_scope_tokens_to_pl_oci,
    normalize_english_statement_label,
    normalize_pl_oci_combined_label,
)
from src.applications.parsing.service.three_statements.statement_labels.zh import (
    normalize_chinese_statement_label,
)


def normalize_statement_label(label: Any) -> Optional[str]:
    if not isinstance(label, str):
        return None
    text = traditional_to_simplified(label.strip())
    if not text:
        return None

    if re.search(r"[a-zA-Z]", text):
        english_label = normalize_english_statement_label(text)
        if english_label:
            return english_label

    return normalize_chinese_statement_label(text)


def get_base_type(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    if "资产负债表" in label:
        return "资产负债表"
    if "利润表" in label:
        return "利润表"
    if "现金流量表" in label:
        return "现金流量表"
    if "所有者权益变动" in label:
        return "所有者权益变动表"
    return None


def has_scope_prefix(label: Optional[str]) -> bool:
    if not label:
        return False
    return "合并" in label or "母公司" in label


def strip_scope_prefix_from_category(category: Optional[str]) -> str:
    if not category:
        return category or ""
    if category.startswith("合并"):
        return category[len("合并"):]
    if category.startswith("母公司"):
        return category[len("母公司"):]
    return category


def normalize_category_to_three_main(category: Optional[str]) -> str:
    if not category:
        return category or ""
    base = get_base_type(category)
    if base in THREE_MAIN_BASE_TYPES:
        return base
    return category


def is_three_main_label(label: Optional[str]) -> bool:
    return get_base_type(label) in THREE_MAIN_BASE_TYPES


__all__ = [
    "EN_BALANCE_SHEET_TITLES",
    "EN_CASH_FLOW_TITLES",
    "EN_CHANGES_IN_EQUITY_TITLES",
    "EN_INCOME_STATEMENT_TITLES",
    "EN_POISON_PATTERNS",
    "EN_SCOPE_CONSOLIDATED",
    "EN_SCOPE_STANDALONE",
    "STATEMENT_ORDER",
    "THREE_MAIN_BASE_TYPES",
    "apply_en_scope_tokens_to_pl_oci",
    "get_base_type",
    "has_scope_prefix",
    "is_three_main_label",
    "normalize_category_to_three_main",
    "normalize_english_statement_label",
    "normalize_pl_oci_combined_label",
    "normalize_statement_label",
    "strip_scope_prefix_from_category",
]
