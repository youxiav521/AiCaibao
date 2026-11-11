from __future__ import annotations

from typing import Any, Optional


def resolve_cell_column_key(cell: Any, fallback_idx: int) -> int:
    if isinstance(cell, dict):
        value = cell.get("colStart") or cell.get("colstart") or cell.get("col_start")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return fallback_idx


def to_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def get_page_at(obj: Any) -> Optional[int]:
    if not isinstance(obj, dict):
        return None
    return to_int(obj.get("source_page_no")) or to_int(obj.get("pageAt")) or to_int(obj.get("page"))


def get_text(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    for key in ("md", "value", "text"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def coerce_bbox(bbox: Any) -> dict[str, Any]:
    if not isinstance(bbox, dict):
        return {
            "l": 0,
            "t": 0,
            "r": 0,
            "b": 0,
            "coordOrigin": "TOPLEFT",
        }

    origin = bbox.get("coordOrigin") or bbox.get("coord_origin")
    if isinstance(origin, str):
        origin = origin.upper()

    return {
        "l": bbox.get("l", 0),
        "t": bbox.get("t", 0),
        "r": bbox.get("r", 0),
        "b": bbox.get("b", 0),
        "coordOrigin": origin or "TOPLEFT",
    }


def coerce_cell(cell: Any, *, fallback_page_at: Optional[int]) -> dict[str, Any]:
    if not isinstance(cell, dict):
        return {
            "text": "" if cell is None else str(cell),
            "rowSpan": 1,
            "colSpan": 1,
            "pageAt": fallback_page_at,
            "bBox": {},
        }

    page_at = get_page_at(cell) or fallback_page_at
    row_span = cell.get("rowSpan") or cell.get("rowspan") or cell.get("row_span") or 1
    col_span = cell.get("colSpan") or cell.get("colspan") or cell.get("col_span") or 1
    bbox_raw = cell.get("bBox") or cell.get("bbox")
    bbox = coerce_bbox(bbox_raw) if isinstance(bbox_raw, dict) else {}

    out: dict[str, Any] = {
        "text": str(cell.get("text") or ""),
        "rowSpan": int(row_span) if str(row_span).strip().isdigit() else 1,
        "colSpan": int(col_span) if str(col_span).strip().isdigit() else 1,
        "pageAt": page_at,
        "bBox": bbox,
    }
    row_start = to_int(cell.get("rowstart") or cell.get("rowStart"))
    col_start = to_int(cell.get("colstart") or cell.get("colStart"))
    if row_start is not None:
        out["rowStart"] = row_start
    if col_start is not None:
        out["colStart"] = col_start
    return out


def coerce_rows(rows: Any, *, fallback_page_at: Optional[int]) -> list[list[dict[str, Any]]]:
    if not isinstance(rows, list):
        return []

    out: list[list[dict[str, Any]]] = []
    for row in rows:
        if isinstance(row, list):
            out.append([coerce_cell(cell, fallback_page_at=fallback_page_at) for cell in row])
            continue
        if isinstance(row, dict) and isinstance(row.get("cells"), list):
            out.append([coerce_cell(cell, fallback_page_at=fallback_page_at) for cell in (row.get("cells") or [])])
            continue
    return out


def _logical_row_width(row: list[dict[str, Any]]) -> int:
    cursor = 0
    width = 0
    for cell in row:
        col_span = get_raw_cell_col_span(cell)
        col_start = get_raw_cell_col_start(cell)
        start = col_start if col_start is not None else cursor
        width = max(width, start + col_span)
        cursor = max(cursor, start + col_span)
    return width


def _alignment_column_count(rows: list[list[dict[str, Any]]]) -> int:
    physical_width = max((len(row) for row in rows), default=0)
    logical_widths = [
        _logical_row_width(row)
        for row in rows
        if len(row) > 1
    ]
    return max([physical_width, *logical_widths], default=0)


def _empty_aligned_cell(
    column_index: int,
    *,
    page_at: Optional[int],
    row_span: int = 1,
) -> dict[str, Any]:
    return {
        "text": "",
        "rowSpan": row_span,
        "colSpan": 1,
        "colStart": column_index,
        "pageAt": page_at,
        "bBox": {},
    }


def align_table_rows(rows: list[list[dict[str, Any]]]) -> None:
    """按表格逻辑列原地补齐缺失单元格，同时保留合法的跨行占位。"""
    if not rows:
        return
    column_count = _alignment_column_count(rows)
    if column_count <= 1:
        return

    active_row_spans: dict[int, int] = {}
    for row_index, row in enumerate(rows):
        occupied_columns = set(active_row_spans)
        placements: list[tuple[int, int, dict[str, Any]]] = []
        cursor = 0

        for raw_cell in row:
            cell = raw_cell if isinstance(raw_cell, dict) else {"text": str(raw_cell or "")}
            col_span = get_raw_cell_col_span(cell)
            declared_start = get_raw_cell_col_start(cell)
            start = declared_start if declared_start is not None else cursor
            while any(column in occupied_columns for column in range(start, start + col_span)):
                start += 1

            placements.append((start, col_span, cell))
            occupied_columns.update(range(start, start + col_span))
            cursor = start + col_span

        page_at = next(
            (
                to_int(cell.get("pageAt") or cell.get("page"))
                for _, _, cell in placements
                if to_int(cell.get("pageAt") or cell.get("page")) is not None
            ),
            None,
        )
        placements_by_start = {start: (span, cell) for start, span, cell in placements}
        aligned_row: list[dict[str, Any]] = []
        next_row_spans = {
            column: remaining - 1
            for column, remaining in active_row_spans.items()
            if remaining > 1
        }

        column = 0
        while column < column_count:
            placement = placements_by_start.get(column)
            if placement is None:
                if column not in active_row_spans:
                    aligned_row.append(
                        _empty_aligned_cell(column, page_at=page_at)
                    )
                column += 1
                continue

            col_span, source_cell = placement
            visible_col_span = min(col_span, column_count - column)
            row_span = to_int(
                source_cell.get("rowSpan")
                or source_cell.get("rowspan")
                or source_cell.get("row_span")
            ) or 1
            row_span = max(row_span, 1)
            for offset in range(visible_col_span):
                target_column = column + offset
                if offset == 0:
                    output_cell = dict(source_cell)
                    output_cell["colStart"] = target_column
                    output_cell["colSpan"] = 1
                else:
                    output_cell = _empty_aligned_cell(
                        target_column,
                        page_at=page_at,
                        row_span=row_span,
                    )
                aligned_row.append(output_cell)
                if row_span > 1:
                    next_row_spans[target_column] = max(
                        next_row_spans.get(target_column, 0),
                        row_span - 1,
                    )
            column += visible_col_span

        rows[row_index] = aligned_row
        active_row_spans = next_row_spans


def expand_colspan_header_rows(rows: list[list[dict[str, Any]]]) -> None:
    """兼容旧调用：展开跨列表头，并统一补齐所有表格行。"""
    align_table_rows(rows)


def get_raw_cell_text(cell: Any) -> str:
    if not isinstance(cell, dict):
        return "" if cell is None else str(cell).strip()
    value = cell.get("text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for key in ("md", "value"):
        value = cell.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_raw_cell_col_span(cell: Any) -> int:
    if not isinstance(cell, dict):
        return 1
    for key in ("colSpan", "colspan", "col_span"):
        value = cell.get(key)
        int_value = to_int(value)
        if int_value is not None:
            return max(int_value, 1)
    return 1


def get_raw_cell_col_start(cell: Any) -> Optional[int]:
    if not isinstance(cell, dict):
        return None
    for key in ("colStart", "colstart", "col_start"):
        int_value = to_int(cell.get(key))
        if int_value is not None:
            return int_value
    return None


def normalize_raw_rows(rows: Any) -> list[list[Any]]:
    if not isinstance(rows, list):
        return []

    out: list[list[Any]] = []
    for row in rows:
        if isinstance(row, list):
            out.append(row)
            continue
        if isinstance(row, dict) and isinstance(row.get("cells"), list):
            out.append(row.get("cells") or [])
            continue
    return out


def infer_table_total_columns(raw_rows: list[list[Any]]) -> int:
    max_end = 0
    for row in raw_rows:
        col_cursor = 0
        row_end = 0
        for cell in row:
            col_span = get_raw_cell_col_span(cell)
            col_start = get_raw_cell_col_start(cell)
            start = col_start if col_start is not None else col_cursor
            end = start + col_span
            row_end = max(row_end, end)
            col_cursor = max(col_cursor, end) if col_start is not None else end
        max_end = max(max_end, row_end)
    return max_end


def find_raw_cell_covering_col(row: list[Any], target_col: int) -> Optional[Any]:
    col_cursor = 0
    for cell in row:
        col_span = get_raw_cell_col_span(cell)
        col_start = get_raw_cell_col_start(cell)
        start = col_start if col_start is not None else col_cursor
        end = start + col_span
        if start <= target_col < end:
            return cell
        col_cursor = max(col_cursor, end) if col_start is not None else end
    return None
