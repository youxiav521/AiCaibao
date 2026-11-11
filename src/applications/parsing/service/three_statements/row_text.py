from __future__ import annotations

from typing import Any, Optional

from src.applications.parsing.service.three_statements.cells import (
    get_raw_cell_text,
    normalize_raw_rows,
    resolve_cell_column_key,
)
from src.applications.parsing.service.three_statements.headers import looks_like_amount_text


def fill_null_accounting_year_from_document(rows: list[list[Any]], doc_year: Optional[int]) -> None:
    if doc_year is None:
        return
    for row in rows:
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            if cell.get("accountingYear") is not None:
                continue
            col_key = resolve_cell_column_key(cell, col_idx)
            if col_key <= 0:
                continue
            cell["accountingYear"] = doc_year


def row_has_amount_like_text(row: Any) -> bool:
    normalized = normalize_raw_rows([row])
    if not normalized:
        return False
    for cell in normalized[0]:
        cell_text = get_raw_cell_text(cell).strip()
        if looks_like_amount_text(cell_text):
            return True
    return False


def get_row_first_cell_text(row: Any) -> str:
    if isinstance(row, list) and row:
        return get_raw_cell_text(row[0])
    if isinstance(row, dict) and isinstance(row.get("cells"), list) and row["cells"]:
        return get_raw_cell_text(row["cells"][0])
    return ""


def rows_to_compact_text(rows: Any) -> str:
    raw = normalize_raw_rows(rows) if isinstance(rows, list) else []
    parts: list[str] = []
    for row in raw:
        for cell in row:
            text = get_raw_cell_text(cell).strip()
            if text:
                parts.append(text)
    return " ".join(parts)
