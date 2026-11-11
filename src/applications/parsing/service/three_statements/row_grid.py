from __future__ import annotations

from typing import Any, Optional

from src.applications.parsing.service.three_statements.cells import to_int


def strip_col_start_keys(raw_rows: list[list[Any]]) -> list[list[Any]]:
    stripped_rows: list[list[Any]] = []
    for row in raw_rows:
        new_row: list[Any] = []
        for cell in row:
            if not isinstance(cell, dict):
                new_row.append(cell)
                continue
            new_cell = dict(cell)
            new_cell.pop("colStart", None)
            new_cell.pop("colstart", None)
            new_cell.pop("col_start", None)
            new_row.append(new_cell)
        stripped_rows.append(new_row)
    return stripped_rows


def expand_rows_to_cell_grid(raw_rows: list[list[Any]]) -> list[list[Optional[dict[str, Any]]]]:
    pending_cells_by_row: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    cell_grid: list[list[Optional[dict[str, Any]]]] = []

    def ensure_list_length(values: list[Any], target_index: int) -> None:
        if target_index >= len(values):
            values.extend([None] * (target_index + 1 - len(values)))

    for row_index, row in enumerate(raw_rows):
        row_cells: list[Optional[dict[str, Any]]] = []
        column_cursor = 0

        if row_index in pending_cells_by_row:
            for column_index, cell_reference in pending_cells_by_row.pop(row_index):
                ensure_list_length(row_cells, column_index)
                row_cells[column_index] = cell_reference

        for cell in row:
            if not isinstance(cell, dict):
                cell = {"text": "" if cell is None else str(cell)}

            row_span = to_int(cell.get("rowSpan") or cell.get("rowspan") or cell.get("row_span")) or 1
            column_span = to_int(cell.get("colSpan") or cell.get("colspan") or cell.get("col_span")) or 1
            row_span = max(row_span, 1)
            column_span = max(column_span, 1)

            declared_col_start = to_int(cell.get("colStart") or cell.get("colstart") or cell.get("col_start"))
            if declared_col_start is not None and declared_col_start >= 0:
                start_column = declared_col_start
                ensure_list_length(row_cells, start_column)
                while start_column < len(row_cells) and row_cells[start_column] is not None:
                    start_column += 1
            else:
                while True:
                    if column_cursor >= len(row_cells):
                        row_cells.append(None)
                        break
                    if row_cells[column_cursor] is None:
                        break
                    column_cursor += 1
                start_column = column_cursor

            for span_offset in range(column_span):
                target_column_index = start_column + span_offset
                ensure_list_length(row_cells, target_column_index)
                row_cells[target_column_index] = cell

            if row_span > 1:
                for row_offset in range(1, row_span):
                    target_row_index = row_index + row_offset
                    pending_list = pending_cells_by_row.setdefault(target_row_index, [])
                    for span_offset in range(column_span):
                        pending_list.append((start_column + span_offset, cell))

            column_cursor = max(column_cursor, start_column + column_span)

        cell_grid.append(row_cells)

    max_columns = max((len(row) for row in cell_grid), default=0)
    for row in cell_grid:
        if len(row) < max_columns:
            row.extend([None] * (max_columns - len(row)))
    return cell_grid
