from __future__ import annotations

from typing import Any, Optional

from src.applications.parsing.service.three_statements.cells import (
    find_raw_cell_covering_col,
    get_raw_cell_col_span,
    get_raw_cell_col_start,
    get_raw_cell_text,
    infer_table_total_columns,
    normalize_raw_rows,
)
from src.applications.parsing.service.three_statements.headers import looks_like_subject
from src.applications.parsing.service.three_statements.row_text import rows_to_compact_text
from src.applications.parsing.service.three_statements.two_block_rows import (
    _looks_like_line_no_header,
    _looks_like_project_header,
    _maybe_split_multi_block_rows,
    maybe_split_two_block_rows,
    row_has_any_text,
)


def _looks_like_folded_two_block_header(raw_rows: list[list[Any]], total_cols: int) -> bool:
    if total_cols < 4 or total_cols % 2 != 0:
        return False
    right_start = total_cols // 2
    for row in raw_rows[:6]:
        left_project = get_raw_cell_text(find_raw_cell_covering_col(row, 0))
        right_project = get_raw_cell_text(find_raw_cell_covering_col(row, right_start))
        if not (_looks_like_project_header(left_project) and _looks_like_project_header(right_project)):
            continue
        left_next = get_raw_cell_text(find_raw_cell_covering_col(row, 1)) if total_cols > 1 else ""
        right_next = get_raw_cell_text(find_raw_cell_covering_col(row, right_start + 1)) if right_start + 1 < total_cols else ""
        if _looks_like_line_no_header(left_next) == _looks_like_line_no_header(right_next):
            return True
    return False


def _as_single_split_table(table_el: dict[str, Any], rows: list[list[Any]]) -> tuple[dict[str, Any], ...]:
    base = {key: value for key, value in table_el.items() if key not in ("rows", "md", "html")}
    return ({**base, "rows": rows, "md": rows_to_compact_text(rows)},)


def try_split_side_by_side_table(
    table_el: dict[str, Any],
) -> Optional[tuple[dict[str, Any], ...]]:
    rows = table_el.get("rows")
    multi_block_rows = _maybe_split_multi_block_rows(rows)
    if multi_block_rows:
        base = {key: value for key, value in table_el.items() if key not in ("rows", "md", "html")}
        multi_table_el = {**base, "rows": multi_block_rows, "md": rows_to_compact_text(multi_block_rows)}
        return (multi_table_el,)

    raw_rows = normalize_raw_rows(rows)
    if not raw_rows:
        return None

    total_cols = infer_table_total_columns(raw_rows)
    if _looks_like_folded_two_block_header(raw_rows, total_cols):
        folded_rows = maybe_split_two_block_rows(rows)
        if isinstance(folded_rows, list) and folded_rows is not rows:
            return _as_single_split_table(table_el, folded_rows)

    if total_cols < 6:
        return None

    left_end: Optional[int] = None
    right_start: Optional[int] = None
    separator_col: Optional[int] = None

    if total_cols % 2 == 0:
        left_end = total_cols // 2
        right_start = left_end
    else:
        candidate_left = total_cols // 2
        candidate_right = total_cols - candidate_left - 1
        candidate_sep = candidate_left
        if candidate_left >= 3 and candidate_left == candidate_right:
            empty_sep = sum(
                1 for row in raw_rows
                if not get_raw_cell_text(find_raw_cell_covering_col(row, candidate_sep))
            )
            if empty_sep / max(len(raw_rows), 1) >= 0.7:
                left_end = candidate_left
                right_start = candidate_left + 1
                separator_col = candidate_sep

    if left_end is None or right_start is None:
        return None

    both_subject_rows = 0
    for row in raw_rows:
        left_cell = find_raw_cell_covering_col(row, 0)
        right_cell = find_raw_cell_covering_col(row, right_start)
        if (
            looks_like_subject(get_raw_cell_text(left_cell))
            and looks_like_subject(get_raw_cell_text(right_cell))
        ):
            both_subject_rows += 1
            if both_subject_rows >= 2:
                break

    if both_subject_rows < 2:
        return None

    left_rows: list[list[Any]] = []
    right_rows: list[list[Any]] = []

    for row in raw_rows:
        left_cells: list[Any] = []
        right_cells: list[Any] = []
        col_cursor = 0
        crossed = False

        for cell in row:
            col_span = get_raw_cell_col_span(cell)
            col_start_val = get_raw_cell_col_start(cell)
            start = col_start_val if col_start_val is not None else col_cursor
            end = start + col_span

            if separator_col is not None:
                if start <= separator_col < end and not (
                    start == separator_col and end == separator_col + 1
                ):
                    crossed = True
                    break
                if start == separator_col and end == separator_col + 1:
                    col_cursor = max(col_cursor, end) if col_start_val is not None else end
                    continue

            if start < left_end < end or start < right_start < end:
                crossed = True
                break

            if end <= left_end:
                left_cells.append(cell)
            elif start >= right_start:
                if col_start_val is not None and isinstance(cell, dict):
                    cell = dict(cell)
                    for key in ("colstart", "colStart", "col_start"):
                        if key in cell:
                            cell[key] = col_start_val - right_start
                right_cells.append(cell)

            col_cursor = max(col_cursor, end) if col_start_val is not None else end

        if crossed:
            if row_has_any_text(row):
                left_rows.append(row)
            continue

        if row_has_any_text(left_cells):
            left_rows.append(left_cells)
        if row_has_any_text(right_cells):
            right_rows.append(right_cells)

    if not left_rows or not right_rows:
        return None

    left_md = rows_to_compact_text(left_rows)
    right_md = rows_to_compact_text(right_rows)

    base = {key: value for key, value in table_el.items() if key not in ("rows", "md", "html")}
    left_table_el = {**base, "rows": left_rows, "md": left_md}
    right_table_el = {**base, "rows": right_rows, "md": right_md}

    return left_table_el, right_table_el

