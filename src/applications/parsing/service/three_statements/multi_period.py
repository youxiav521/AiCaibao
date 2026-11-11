from __future__ import annotations

import copy
import re
from datetime import date
from typing import Any, Callable

from src.applications.parsing.service.three_statements.cells import resolve_cell_column_key

ParseDate = Callable[[str], date | None]


def all_logical_col_keys_in_rows(rows: list[list[Any]]) -> set[int]:
    keys: set[int] = set()
    for row in rows:
        if not isinstance(row, list):
            continue
        for col_idx, cell in enumerate(row):
            if isinstance(cell, dict):
                keys.add(resolve_cell_column_key(cell, col_idx))
    return keys


def filing_period_label_for_reporting_date_pair(d0: date, d1: date) -> str:
    if d1 <= d0:
        return "年报"
    days = (d1 - d0).days
    if 300 <= days <= 430:
        return "年报"
    if 150 <= days <= 210:
        return "半年报"
    if 75 <= days <= 115:
        end_month = d1.month
        if end_month == 3:
            return "一季度报"
        if end_month == 9:
            return "三季度报"
        return "季报"
    return "年报"


def format_cn_reporting_date(value: date) -> str:
    return "{}年{}月{}日".format(value.year, value.month, value.day)


def slice_rows_to_ordered_logical_keys(
    rows: list[list[Any]],
    ordered_keys: list[int],
) -> list[list[Any]]:
    key_set = set(ordered_keys)
    out: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cells_by_key: dict[int, Any] = {}
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            column_key = resolve_cell_column_key(cell, col_idx)
            if column_key in key_set:
                cells_by_key[column_key] = cell
        new_row: list[Any] = []
        for new_index, column_key in enumerate(ordered_keys):
            source = cells_by_key.get(column_key)
            if source is None:
                new_row.append({"text": "", "colStart": new_index, "colSpan": 1})
            else:
                new_cell = copy.deepcopy(source)
                new_cell["colStart"] = new_index
                new_cell["colSpan"] = 1
                new_row.append(new_cell)
        out.append(new_row)
    return out


def slice_header_cells_to_ordered_logical_keys(
    header_cells: list[list[Any]],
    ordered_keys: list[int],
) -> list[list[Any]]:
    """按 ordered_keys 列号切片 ``headerCells``，并将每行 cell 的 colStart 重排为新位置。

    展开 colSpan 时仅保留落在 ordered_keys 内的列段；多列段连续命中时合并为一个 cell（保持 colSpan）。
    与 :func:`slice_rows_to_ordered_logical_keys` 配套使用，确保切片后表头列与数据列对齐。
    """
    if not header_cells:
        return []
    new_index_by_key = {key: i for i, key in enumerate(ordered_keys)}
    out: list[list[Any]] = []
    for row in header_cells:
        if not isinstance(row, list):
            continue
        # 先按 colStart/colSpan 把每个 cell 展开成 (column_key -> cell_template)
        col_cursor = 0
        cell_per_key: dict[int, dict[str, Any]] = {}
        for cell in row:
            if not isinstance(cell, dict):
                col_cursor += 1
                continue
            cstart = cell.get("colStart")
            if cstart is None:
                cstart = cell.get("colstart")
            if isinstance(cstart, int):
                col_cursor = cstart
            cspan_raw = cell.get("colSpan")
            if cspan_raw is None:
                cspan_raw = cell.get("colspan", 1)
            try:
                cspan = max(int(cspan_raw or 1), 1)
            except (TypeError, ValueError):
                cspan = 1
            for offset in range(cspan):
                cell_per_key[col_cursor + offset] = cell
            col_cursor += cspan

        # 按 ordered_keys 顺序重组：连续命中相同 cell 的列段合并为一个 cell（合并 colSpan）
        new_row: list[dict[str, Any]] = []
        run_cell: Optional[dict[str, Any]] = None
        run_start: int = 0
        run_count: int = 0

        def _flush() -> None:
            nonlocal run_cell, run_start, run_count
            if run_cell is None or run_count <= 0:
                return
            if run_cell.get("text") or run_cell.get("value") or run_cell.get("md"):
                new_cell = copy.deepcopy(run_cell)
            else:
                new_cell = copy.deepcopy(run_cell) if isinstance(run_cell, dict) else {"text": ""}
            new_cell["colStart"] = run_start
            new_cell["colSpan"] = run_count
            # 同步小写键，兼容老代码
            new_cell["colstart"] = run_start
            new_cell["colspan"] = run_count
            new_row.append(new_cell)
            run_cell = None
            run_count = 0

        for new_index, column_key in enumerate(ordered_keys):
            source = cell_per_key.get(column_key)
            if source is None:
                _flush()
                new_row.append({
                    "text": "",
                    "colStart": new_index,
                    "colSpan": 1,
                    "colstart": new_index,
                    "colspan": 1,
                })
                continue
            if run_cell is None:
                run_cell = source
                run_start = new_index
                run_count = 1
                continue
            if source is run_cell:
                run_count += 1
                continue
            _flush()
            run_cell = source
            run_start = new_index
            run_count = 1
        _flush()

        out.append(new_row)
    return out


def compact_multiline_header_cell_text(raw: str) -> str:
    parts = [part.strip() for part in re.split(r"[\r\n]+", (raw or "").strip()) if part.strip()]
    if not parts:
        return ""
    return "".join(parts)


def preferred_period_column_header_display(
    raw_header: str,
    target_date: date,
    *,
    parse_statement_column_date_from_text: ParseDate,
) -> str:
    raw = (raw_header or "").strip()
    if not raw:
        return format_cn_reporting_date(target_date)
    lines = [line.strip() for line in re.split(r"[\r\n]+", raw) if line.strip()]
    if lines:
        first = compact_multiline_header_cell_text(lines[0])
        if first and parse_statement_column_date_from_text(first) == target_date:
            return first
        accumulated = ""
        for line in lines:
            accumulated = (accumulated + line) if accumulated else line
            candidate = compact_multiline_header_cell_text(accumulated)
            if candidate and parse_statement_column_date_from_text(candidate) == target_date:
                return candidate
    compact = compact_multiline_header_cell_text(raw)
    if compact and parse_statement_column_date_from_text(compact) == target_date:
        return compact
    return format_cn_reporting_date(target_date)


def patch_pair_header_period_labels(
    rows: list[list[Any]],
    d0: date,
    d1: date,
    *,
    display_left: str,
    display_right: str,
    looks_like_column_header_row: Callable[[Any], bool],
    parse_statement_column_date_from_text: ParseDate,
) -> None:
    for row in rows[:8]:
        if not isinstance(row, list) or len(row) < 2:
            continue
        cell_0, cell_1 = row[-2], row[-1]
        if not isinstance(cell_0, dict) or not isinstance(cell_1, dict):
            continue
        text_0 = str(cell_0.get("text") or "")
        text_1 = str(cell_1.get("text") or "")
        looks_header = looks_like_column_header_row(row)
        looks_like_dates = bool(
            parse_statement_column_date_from_text(text_0) and parse_statement_column_date_from_text(text_1)
        )
        if not looks_header and not looks_like_dates:
            continue
        cell_0["text"] = display_left
        cell_1["text"] = display_right
        return


def touch_accounting_years_for_pair_columns(rows: list[list[Any]], d0: date, d1: date) -> None:
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        cell_0, cell_1 = row[-2], row[-1]
        if isinstance(cell_0, dict) and resolve_cell_column_key(cell_0, len(row) - 2) > 0:
            cell_0["accountingYear"] = d0.year
        if isinstance(cell_1, dict) and resolve_cell_column_key(cell_1, len(row) - 1) > 0:
            cell_1["accountingYear"] = d1.year
