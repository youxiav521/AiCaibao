from __future__ import annotations

import re
from collections import defaultdict

from loguru import logger

from ...schemas.subject_match_schema import Cell, Table
from .en.column_role_rules import (
    EN_AMOUNT_HEADER_KEYWORDS,
    EN_AMOUNT_HEADER_RE,
    EN_SUBJECT_HEADER_KEYWORDS,
)
from .zh.column_role_rules import (
    ZH_AMOUNT_HEADER_KEYWORDS,
    ZH_AMOUNT_HEADER_RE,
    ZH_SUBJECT_HEADER_KEYWORDS,
)


_SUBJECT_HEADER_KEYWORDS: frozenset[str] = frozenset({
    *ZH_SUBJECT_HEADER_KEYWORDS,
    *EN_SUBJECT_HEADER_KEYWORDS,
})
_AMOUNT_HEADER_KEYWORDS: frozenset[str] = frozenset({
    *ZH_AMOUNT_HEADER_KEYWORDS,
    *EN_AMOUNT_HEADER_KEYWORDS,
})
_AMOUNT_HEADER_RE = re.compile(
    rf"(?:{ZH_AMOUNT_HEADER_RE.pattern})|(?:{EN_AMOUNT_HEADER_RE.pattern})",
    re.IGNORECASE,
)

def detect_column_roles(table: Table) -> dict[int, str]:
    """Detect column roles (subject / amount / other) from table structure.

    Supports both single-row headers (项目|金额|项目|金额) and multi-row
    headers (项目|合并|合并 / 空|期末余额|期初余额). The first several rows are
    merged by column before role classification, so a column can be recognized
    as amount even when its year/caliber/unit and amount label are split across
    multiple header rows.
    """
    _VALUE_CELL_RE = re.compile(r'^[\d,.\-—\s%()（）±+×÷/＋－＊／]+$')

    max_cols = 0
    for row in table.rows:
        if row:
            max_cols = max(max_cols, len(row))
    if max_cols == 0:
        return {}

    def _cell_col_index(cell: Cell, fallback_idx: int) -> int:
        if isinstance(cell.colNum, int) and cell.colNum >= 0:
            if cell.colNum >= max_cols and cell.colNum - 1 >= 0:
                return cell.colNum - 1
            return cell.colNum
        return fallback_idx

    def _normalize_header_text(text: str) -> str:
        return re.sub(r'\s+', '', (text or '').strip()).lower()

    def _is_subject_header(text: str) -> bool:
        compact = _normalize_header_text(text)
        return compact in _SUBJECT_HEADER_KEYWORDS

    def _is_amount_header(text: str) -> bool:
        compact = _normalize_header_text(text)
        if not compact:
            return False
        return compact in _AMOUNT_HEADER_KEYWORDS or bool(_AMOUNT_HEADER_RE.search(compact))

    scan_limit = min(5, len(table.rows))

    # Pass 1: merge possible multi-row headers by column. This handles tables like:
    # row0: 项目 | 合并 | 合并 | 母公司 | 母公司
    # row1:      | 期末余额 | 期初余额 | 期末余额 | 期初余额
    column_header_parts: dict[int, list[str]] = defaultdict(list)
    saw_subject_header = False
    saw_amount_header = False

    for row_idx in range(scan_limit):
        row = table.rows[row_idx] if row_idx < len(table.rows) else []
        if not row:
            continue

        row_has_header_signal = False
        for fallback_idx, cell in enumerate(row):
            txt = _normalize_header_text(cell.text)
            if not txt:
                continue
            if _is_subject_header(txt) or _is_amount_header(txt):
                row_has_header_signal = True
                if _is_subject_header(txt):
                    saw_subject_header = True
                if _is_amount_header(txt):
                    saw_amount_header = True

        if not row_has_header_signal and saw_subject_header and saw_amount_header:
            break

        for fallback_idx, cell in enumerate(row):
            col_idx = _cell_col_index(cell, fallback_idx)
            txt = _normalize_header_text(cell.text)
            if txt:
                column_header_parts[col_idx].append(txt)

    roles: dict[int, str] = {}
    for ci in range(max_cols):
        parts = column_header_parts.get(ci, [])
        merged = ''.join(parts)
        if any(_is_subject_header(p) for p in parts):
            roles[ci] = "subject"
        elif any(_is_amount_header(p) for p in parts) or _is_amount_header(merged):
            roles[ci] = "amount"
        else:
            roles[ci] = "other"

    subject_cols = [c for c, r in roles.items() if r == "subject"]
    amount_cols = [c for c, r in roles.items() if r == "amount"]
    if subject_cols and amount_cols:
        logger.info(
            "[column-roles] Detected from merged headers: subject_cols={}, amount_cols={}",
            subject_cols, amount_cols,
        )
        return roles

    # Pass 2: content-pattern fallback. This covers tables where header rows are
    # missing after OCR/table extraction, including parallel subject/amount blocks.
    col_text_counts: dict[int, int] = defaultdict(int)
    col_num_counts: dict[int, int] = defaultdict(int)
    sample_rows = min(20, len(table.rows))
    for row in table.rows[:sample_rows]:
        for fallback_idx, cell in enumerate(row):
            col_idx = _cell_col_index(cell, fallback_idx)
            txt = (cell.text or '').strip()
            if not txt:
                continue
            if _VALUE_CELL_RE.match(txt):
                col_num_counts[col_idx] += 1
            elif len(txt) >= 2:
                col_text_counts[col_idx] += 1

    roles = {}
    for ci in range(max_cols):
        text_cnt = col_text_counts.get(ci, 0)
        num_cnt = col_num_counts.get(ci, 0)
        total = text_cnt + num_cnt
        if total == 0:
            roles[ci] = "other"
        elif text_cnt / max(total, 1) >= 0.6:
            roles[ci] = "subject"
        elif num_cnt / max(total, 1) >= 0.6:
            roles[ci] = "amount"
        else:
            roles[ci] = "other"

    subject_cols = [c for c, r in roles.items() if r == "subject"]
    amount_cols = [c for c, r in roles.items() if r == "amount"]
    if subject_cols and amount_cols:
        logger.info(
            "[column-roles] Inferred from content: subject_cols={}, amount_cols={}",
            subject_cols, amount_cols,
        )
        return roles

    return {}
