from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from src.applications.parsing.service.three_statements.cells import get_page_at, get_raw_cell_text, get_text, normalize_raw_rows

logger = logging.getLogger(__name__)

NormalizeStatementLabel = Callable[[str], Optional[str]]

CASHFLOW_SUPPLEMENTARY_HEADERS: tuple[str, ...] = (
    "补充资料",
    "将净利润调节为经营活动现金流量",
    "将净利润调节为经营活动的现金流量",
    "将净利润调节为经营活动现金流",
    "不涉及现金收支的重大投资和筹资活动",
    "不涉及现金收支的投资和筹资活动",
    "不涉及现金收支的重大交易",
    "现金及现金等价物净变动情况",
    "现金及现金等价物净变动",
    "现金及现金等价物的净增加额",
)


def truncate_cashflow_supplementary(rows: Any) -> Any:
    normalized = normalize_raw_rows(rows)
    if not normalized:
        return rows

    cut_index = -1
    for row_index, row_cells in enumerate(normalized):
        first_text = ""
        for cell in row_cells:
            text = get_raw_cell_text(cell)
            if text:
                first_text = text
                break
        if not first_text:
            continue
        compact = re.sub(r"\s+", "", first_text)
        if any(header in compact for header in CASHFLOW_SUPPLEMENTARY_HEADERS):
            cut_index = row_index
            break

    if cut_index <= 0:
        return rows

    if isinstance(rows, list) and len(rows) > cut_index:
        marker = get_raw_cell_text(normalized[cut_index][0]) if normalized[cut_index] else ""
        logger.info(
            "Cash flow supplementary section truncated at row %d (marker=%r), kept %d of %d rows",
            cut_index, marker, cut_index, len(rows),
        )
        return rows[:cut_index]
    return rows


def find_nearest_statement_title(
    items: list[Any],
    *,
    table_index: int,
    normalize_statement_label: NormalizeStatementLabel,
    max_back: int = 50,
    max_page_gap: int = 2,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if table_index <= 0 or table_index >= len(items):
        return None, None

    table = items[table_index] if isinstance(items[table_index], dict) else {}
    table_page = get_page_at(table)
    effective_max_back = max_back if table_page is not None else 5

    checked = 0
    for index in range(table_index - 1, -1, -1):
        item = items[index]
        if not isinstance(item, dict):
            continue

        if item.get("type") in ("table", "excel_table"):
            table_type = item.get("table_type") or ""
            table_type_lower = table_type.lower()
            zh_break = any(statement in table_type for statement in ("资产负债表", "利润表", "现金流量表", "所有者权益变动"))
            en_break = any(statement in table_type_lower for statement in (
                "balance sheet",
                "income statement",
                "statement of profit or loss",
                "statement of comprehensive income",
                "cash flow",
                "changes in equity",
            ))
            if zh_break or en_break:
                break
            continue

        item_page = get_page_at(item)
        if table_page is not None and item_page is not None:
            if table_page - item_page > max_page_gap:
                break

        if item.get("type") != "text":
            continue

        checked += 1
        if checked > effective_max_back:
            break

        normalized = normalize_statement_label(get_text(item))
        if normalized:
            return item, normalized

    return None, None
