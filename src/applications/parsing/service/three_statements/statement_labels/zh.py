from __future__ import annotations

import re
from typing import Any, Optional

from src.applications.parsing.service.three_statements.statement_labels.common import STATEMENT_ORDER
from src.applications.parsing.utils.table_classification_utils import is_derivative_statement_title
from src.utils.chinese_converter import traditional_to_simplified


def normalize_chinese_statement_label(label: Any) -> Optional[str]:
    if not isinstance(label, str):
        return None
    text = traditional_to_simplified(label.strip())
    if not text:
        return None

    compact = re.sub(r"\s+", "", text)
    if any(poison_word in compact for poison_word in ("注释", "注解", "附注")):
        return None

    if is_derivative_statement_title(text):
        return None

    text = text.replace("金流量表", "现金流量表")
    base: Optional[str] = None
    if "资产负债表" in text:
        base = "资产负债表"
    elif "利润表及其他综合收益" in text:
        base = "利润表及其他综合收益"
    elif "利润表" in text or ("利润" in text and "分配表" in text):
        base = "利润表"
    elif "现金流量表" in text:
        base = "现金流量表"
    elif "所有者权益变动" in text or "权益变动表" in text:
        base = "所有者权益变动表"
    if base is None:
        return None

    scope: Optional[str] = None
    if "合并" in text:
        scope = "合并"
    elif "母公司" in text or ("母" in text and "公司" in text):
        scope = "母公司"
    if scope is None:
        return base if base in STATEMENT_ORDER else None

    normalized = f"{scope}{base}"
    return normalized if normalized in STATEMENT_ORDER else None
