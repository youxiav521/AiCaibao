from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.applications.parsing.service.three_statements.cells import get_raw_cell_text, normalize_raw_rows
from src.applications.parsing.service.three_statements.row_text import get_row_first_cell_text
from src.applications.parsing.service.three_statements.statement_content import (
    BASE_TYPE_START_KEYWORDS,
    content_matches_base_type,
    get_terminal_keywords,
)
from src.applications.parsing.service.three_statements.two_block_rows import row_has_any_text

logger = logging.getLogger(__name__)


def try_split_combined_statement(
    table_el: dict[str, Any], base_type: str, detected_label: str
) -> Optional[list[tuple[str, dict[str, Any]]]]:
    rows = table_el.get("rows")
    if not isinstance(rows, list) or len(rows) < 10:
        return None

    terminal_keywords = get_terminal_keywords(base_type)
    if not terminal_keywords:
        return None

    raw_rows = normalize_raw_rows(rows)
    if not raw_rows:
        return None

    end_indices: list[int] = []
    for row_index, row in enumerate(raw_rows):
        text = get_row_first_cell_text(row).replace(" ", "")
        if any(keyword in text for keyword in terminal_keywords):
            if "归属于" in text or "其中" in text:
                continue
            end_indices.append(row_index)

    if len(end_indices) < 2:
        return None

    min_split_gap = 15
    gap = end_indices[-1] - end_indices[0]
    if gap < min_split_gap:
        logger.info(
            "Split rejected for %r: terminal keyword gap=%d < %d (indices=%s)",
            detected_label, gap, min_split_gap, end_indices,
        )
        return None

    split_point = end_indices[0] + 1
    first_rows = rows[:split_point]
    second_rows = rows[split_point:]

    stripped = 0
    while second_rows and stripped < 5:
        raw = normalize_raw_rows([second_rows[0]])
        if not raw or not raw[0] or not row_has_any_text(raw[0]):
            second_rows = second_rows[1:]
            stripped += 1
            continue
        first_text = get_row_first_cell_text(raw[0]).strip()
        footer_keywords = (
            "资产负债表", "利润表", "现金流量表", "所有者权益",
            "编制单位", "法定代表人", "财务负责人", "负责人",
            "主管会计", "编制", "单位：", "金额单位",
        )
        if any(keyword in first_text for keyword in footer_keywords):
            second_rows = second_rows[1:]
            stripped += 1
            continue
        break

    if not second_rows or len(second_rows) < 5:
        return None

    second_raw = normalize_raw_rows(second_rows)
    start_keywords = BASE_TYPE_START_KEYWORDS.get(base_type, ())
    if start_keywords:
        head_rows = second_raw[:8]
        found_start = any(
            any(keyword in get_row_first_cell_text(row).replace(" ", "") for keyword in start_keywords)
            for row in head_rows
        )
        if not found_start:
            logger.info(
                "Split rejected for %r: second half lacks start keyword %s in first %d rows",
                detected_label, start_keywords, len(head_rows),
            )
            return None

    second_has_terminal = any(
        any(keyword in get_row_first_cell_text(row).replace(" ", "") for keyword in terminal_keywords)
        for row in second_raw
    )
    if not second_has_terminal:
        return None

    first_raw_for_check = normalize_raw_rows(first_rows)
    second_raw_for_check = normalize_raw_rows(second_rows)
    first_compact = "".join(
        get_row_first_cell_text(row).replace(" ", "") for row in first_raw_for_check
    )
    second_compact = "".join(
        get_row_first_cell_text(row).replace(" ", "") for row in second_raw_for_check
    )
    if not content_matches_base_type(first_compact, base_type) or not content_matches_base_type(second_compact, base_type):
        return None

    def row_full_signature(row: list[Any]) -> str:
        return "|".join(re.sub(r"\s+", "", get_raw_cell_text(cell)) for cell in row)

    first_sigs = set(row_full_signature(row) for row in first_raw_for_check if row)
    second_sigs = [row_full_signature(row) for row in second_raw_for_check if row]
    if second_sigs:
        dup_count = sum(1 for signature in second_sigs if signature in first_sigs)
        dup_ratio = dup_count / len(second_sigs)
        if dup_ratio > 0.5:
            logger.info(
                "Split rejected for %r: %.0f%% of second-half rows (%d/%d) are exact "
                "duplicates of first-half rows, likely a row-duplication artifact",
                detected_label, dup_ratio * 100, dup_count, len(second_sigs),
            )
            return None

    if "合并" in detected_label:
        first_label = detected_label
        second_label = detected_label.replace("合并", "母公司")
    elif "母公司" in detected_label:
        first_label = detected_label
        second_label = detected_label.replace("母公司", "合并")
    else:
        first_label = f"合并{base_type}"
        second_label = f"母公司{base_type}"

    table_el_1 = dict(table_el)
    table_el_1["rows"] = first_rows
    table_el_2 = dict(table_el)
    table_el_2["rows"] = second_rows

    logger.info(
        "Split succeeded for %r → %r (%d rows) + %r (%d rows), gap=%d, indices=%s",
        detected_label, first_label, len(first_rows),
        second_label, len(second_rows), gap, end_indices,
    )
    return [(first_label, table_el_1), (second_label, table_el_2)]
