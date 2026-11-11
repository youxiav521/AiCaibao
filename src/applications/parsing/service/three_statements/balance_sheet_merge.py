from __future__ import annotations

import re
from typing import Any, Callable, Optional

from src.applications.parsing.service.three_statements.cells import (
    coerce_bbox,
    get_raw_cell_text,
)
from src.applications.parsing.service.three_statements.row_text import rows_to_compact_text
from src.applications.parsing.utils.financial_recognition_and_extraction_utils import (
    infer_column_index_to_date_from_header_cells,
    parse_statement_column_date_from_text,
)

IsFullStatement = Callable[[str, str], bool]


def table_blob_lower(table_el: dict[str, Any]) -> str:
    md = str(table_el.get("md") or "").strip()
    if md:
        return md.lower()
    return rows_to_compact_text(table_el.get("rows")).lower()


def statement_scope_key(label: str) -> str:
    if label.startswith("合并"):
        return "合并"
    if label.startswith("母公司"):
        return "母公司"
    return ""


def statement_labels_merge_compatible(a: str, b: str) -> bool:
    if a != b:
        return False
    return statement_scope_key(a) == statement_scope_key(b)


def looks_like_independent_same_page_tables(
    prev_cand: dict[str, Any], next_cand: dict[str, Any]
) -> bool:
    prev_page = prev_cand.get("page")
    next_page = next_cand.get("page")
    if not (isinstance(prev_page, int) and isinstance(next_page, int) and prev_page == next_page):
        return False
    t1 = prev_cand.get("table") if isinstance(prev_cand.get("table"), dict) else {}
    t2 = next_cand.get("table") if isinstance(next_cand.get("table"), dict) else {}
    bbox1 = coerce_bbox(t1.get("bBox") or t1.get("bbox"))
    bbox2 = coerce_bbox(t2.get("bBox") or t2.get("bbox"))
    try:
        l1, t1y, r1, b1 = float(bbox1.get("l")), float(bbox1.get("t")), float(bbox1.get("r")), float(bbox1.get("b"))
        l2, t2y, r2, b2 = float(bbox2.get("l")), float(bbox2.get("t")), float(bbox2.get("r")), float(bbox2.get("b"))
    except (TypeError, ValueError):
        return False
    h_overlap = max(0.0, min(r1, r2) - max(l1, l2))
    v_overlap = max(0.0, min(b1, b2) - max(t1y, t2y))
    h_threshold = 0.3 * min(max(r1 - l1, 1.0), max(r2 - l2, 1.0))
    v_threshold = 0.3 * min(max(b1 - t1y, 1.0), max(b2 - t2y, 1.0))
    if h_overlap < h_threshold:
        return True
    if v_overlap < v_threshold:
        return True
    return False


def looks_like_independent_balance_sheet_halves(
    prev_cand: dict[str, Any], next_cand: dict[str, Any]
) -> bool:
    prev_page = prev_cand.get("page")
    next_page = next_cand.get("page")
    if not (isinstance(prev_page, int) and isinstance(next_page, int) and prev_page == next_page):
        return False
    t1 = prev_cand.get("table") if isinstance(prev_cand.get("table"), dict) else {}
    t2 = next_cand.get("table") if isinstance(next_cand.get("table"), dict) else {}
    b1 = table_blob_lower(t1)
    b2 = table_blob_lower(t2)
    if "total assets" not in b1:
        return False
    if "total equity and liabilities" in b1:
        return False
    if (
        "total equity and liabilities" not in b2
        and "total liabilities and shareholders' equity" not in b2
    ):
        return False
    bbox1 = coerce_bbox(t1.get("bBox") or t1.get("bbox"))
    bbox2 = coerce_bbox(t2.get("bBox") or t2.get("bbox"))
    try:
        l1, r1 = float(bbox1.get("l")), float(bbox1.get("r"))
        l2, r2 = float(bbox2.get("l")), float(bbox2.get("r"))
    except (TypeError, ValueError):
        return True
    horiz_overlap = max(0.0, min(r1, r2) - max(l1, l2))
    if horiz_overlap > min(r1 - l1, r2 - l2) * 0.3:
        return False
    return True


def can_merge_adjacent_balance_sheet_halves(
    prev_cand: dict[str, Any], next_cand: dict[str, Any]
) -> bool:
    prev_label = str(prev_cand.get("label") or "")
    next_label = str(next_cand.get("label") or "")
    if "资产负债表" not in prev_label or "资产负债表" not in next_label:
        return False
    if not statement_labels_merge_compatible(prev_label, next_label):
        return False
    prev_index = prev_cand.get("expanded_idx")
    next_index = next_cand.get("expanded_idx")
    if not isinstance(prev_index, int) or not isinstance(next_index, int) or next_index != prev_index + 1:
        return False
    prev_page = prev_cand.get("page")
    next_page = next_cand.get("page")
    if isinstance(prev_page, int) and isinstance(next_page, int) and prev_page == next_page:
        if looks_like_independent_balance_sheet_halves(prev_cand, next_cand):
            return False
    t1 = prev_cand.get("table") if isinstance(prev_cand.get("table"), dict) else {}
    t2 = next_cand.get("table") if isinstance(next_cand.get("table"), dict) else {}
    b1 = table_blob_lower(t1)
    b2 = table_blob_lower(t2)
    if "total assets" not in b1:
        return False
    if "total equity and liabilities" in b1:
        return False
    if "total equity and liabilities" not in b2 and "total liabilities and shareholders' equity" not in b2:
        return False
    return True


def row_text_signature_for_merge(row: Any) -> tuple[str, ...]:
    if isinstance(row, list):
        cells = row
    elif isinstance(row, dict):
        cells = row.get("cells") or []
    else:
        cells = []
    return tuple(
        re.sub(r"\s+", " ", (get_raw_cell_text(cell) or "").strip().lower())
        for cell in cells
    )


def looks_like_column_header_row(row: Any) -> bool:
    if isinstance(row, list):
        cells = row
    elif isinstance(row, dict):
        cells = row.get("cells") or []
    else:
        return False
    blob = " ".join((get_raw_cell_text(cell) or "").strip() for cell in cells).lower()
    if not blob.strip():
        return False
    if re.search(r"\b(19|20)\d{2}\b", blob):
        return True
    if re.search(r"\b(group|company|bank|note|rm)\b", blob):
        return True
    if re.search(r"千元|万元|亿元|金额", blob):
        return True
    return False


def detect_merge_header_skip(row_1: list[Any], row_2: list[Any]) -> int:
    if not row_1 or not row_2:
        return 0
    skip = 0
    max_skip = min(len(row_1), len(row_2), 3)
    for index in range(max_skip):
        sig1 = row_text_signature_for_merge(row_1[index])
        sig2 = row_text_signature_for_merge(row_2[index])
        if sig1 and sig1 == sig2 and looks_like_column_header_row(row_2[index]):
            skip += 1
            continue
        break
    if skip == 0 and looks_like_column_header_row(row_2[0]):
        skip = 1
    return skip


def _is_statement_auxiliary_header(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith(("note", "notes", "附注", "注释", "备注"))


def _copy_rows_with_column_positions(rows: list[Any]) -> list[Any]:
    copied_rows: list[Any] = []
    for row in rows:
        if not isinstance(row, list):
            copied_rows.append(row)
            continue
        copied_row: list[Any] = []
        for column, cell in enumerate(row):
            if not isinstance(cell, dict):
                copied_row.append(cell)
                continue
            copied_cell = dict(cell)
            copied_cell["colStart"] = column
            copied_cell["colstart"] = column
            copied_row.append(copied_cell)
        copied_rows.append(copied_row)
    return copied_rows


def _date_columns_from_header(rows: list[Any], header_cells: Any) -> dict[int, Any]:
    if isinstance(header_cells, list) and header_cells:
        dates = infer_column_index_to_date_from_header_cells(header_cells)
        if dates:
            return dates
    if not rows or not isinstance(rows[0], list):
        return {}
    return {
        index: parsed_date
        for index, cell in enumerate(rows[0])
        if (parsed_date := parse_statement_column_date_from_text(get_raw_cell_text(cell)))
        is not None
    }


def _unique_date_columns(column_dates: dict[int, Any]) -> Optional[dict[Any, int]]:
    dates_to_columns: dict[Any, int] = {}
    for column, value in column_dates.items():
        if value in dates_to_columns:
            return None
        dates_to_columns[value] = column
    return dates_to_columns


def _can_anchor_subject_column(primary_header: list[Any], continuation_header: list[Any]) -> bool:
    if not primary_header or not continuation_header:
        return False
    primary_text = get_raw_cell_text(primary_header[0])
    continuation_text = get_raw_cell_text(continuation_header[0])
    return (
        bool(primary_text and continuation_text)
        and parse_statement_column_date_from_text(primary_text) is None
        and parse_statement_column_date_from_text(continuation_text) is None
        and not _is_statement_auxiliary_header(primary_text)
        and not _is_statement_auxiliary_header(continuation_text)
    )


def align_continuation_rows_to_primary_layout(
    primary_rows: list[Any],
    continuation_rows: list[Any],
    *,
    primary_header_cells: Any = None,
    continuation_header_cells: Any = None,
) -> tuple[list[Any], list[Any]]:
    """按科目列和日期表头将续页列映射到主表列。"""
    if not primary_rows or not continuation_rows:
        return primary_rows, continuation_rows
    primary_header = primary_rows[0]
    continuation_header = continuation_rows[0]
    if not isinstance(primary_header, list) or not isinstance(continuation_header, list):
        return primary_rows, continuation_rows

    primary_width = len(primary_header)
    continuation_width = len(continuation_header)
    if primary_width <= continuation_width or not _can_anchor_subject_column(
        primary_header, continuation_header
    ):
        return primary_rows, continuation_rows
    if any(
        not isinstance(row, list) or len(row) != continuation_width
        for row in continuation_rows
    ):
        return primary_rows, continuation_rows

    primary_dates = _unique_date_columns(
        _date_columns_from_header(primary_rows, primary_header_cells)
    )
    continuation_dates = _unique_date_columns(
        _date_columns_from_header(continuation_rows, continuation_header_cells)
    )
    if not primary_dates or not continuation_dates:
        return primary_rows, continuation_rows

    column_mapping: dict[int, int] = {0: 0}
    for report_date, continuation_column in continuation_dates.items():
        primary_column = primary_dates.get(report_date)
        if primary_column is None:
            return primary_rows, continuation_rows
        column_mapping[continuation_column] = primary_column

    primary_auxiliary_columns = [
        index
        for index, cell in enumerate(primary_header)
        if _is_statement_auxiliary_header(get_raw_cell_text(cell))
    ]
    continuation_auxiliary_columns = [
        index
        for index, cell in enumerate(continuation_header)
        if _is_statement_auxiliary_header(get_raw_cell_text(cell))
    ]
    if len(primary_auxiliary_columns) == len(continuation_auxiliary_columns) == 1:
        column_mapping[continuation_auxiliary_columns[0]] = primary_auxiliary_columns[0]

    if set(column_mapping) != set(range(continuation_width)):
        return primary_rows, continuation_rows
    if len(set(column_mapping.values())) != len(column_mapping):
        return primary_rows, continuation_rows

    source_column_by_target = {target: source for source, target in column_mapping.items()}
    aligned_rows: list[Any] = []
    for row in continuation_rows:
        aligned_row: list[Any] = []
        for column in range(primary_width):
            source_index = source_column_by_target.get(column)
            if source_index is None:
                aligned_row.append(
                    {"text": "", "colStart": column, "colstart": column}
                )
                continue
            cell = row[source_index]
            copied_cell = dict(cell) if isinstance(cell, dict) else {"text": str(cell or "")}
            copied_cell["colStart"] = column
            copied_cell["colstart"] = column
            aligned_row.append(copied_cell)
        aligned_rows.append(aligned_row)
    return _copy_rows_with_column_positions(primary_rows), aligned_rows


def merge_balance_sheet_split_candidates(
    prev_cand: dict[str, Any],
    next_cand: dict[str, Any],
    *,
    locale: Optional[str],
    is_full_statement_table_md: Callable[[str, str, Optional[str]], bool],
) -> dict[str, Any]:
    t1 = prev_cand["table"]
    t2 = next_cand["table"]
    row_1 = t1.get("rows") if isinstance(t1.get("rows"), list) else []
    row_2 = t2.get("rows") if isinstance(t2.get("rows"), list) else []
    row_1, row_2 = align_continuation_rows_to_primary_layout(
        row_1,
        row_2,
        primary_header_cells=t1.get("headerCells"),
        continuation_header_cells=t2.get("headerCells"),
    )
    header_skip = detect_merge_header_skip(row_1, row_2)
    if len(row_2) <= header_skip:
        return prev_cand
    merged_el = dict(t1)
    merged_rows = list(row_1) + list(row_2[header_skip:])
    merged_el["rows"] = merged_rows
    md_merged = "\n".join(
        value for value in (str(t1.get("md") or "").strip(), str(t2.get("md") or "").strip()) if value
    )
    merged_el["md"] = md_merged
    continuation = list(prev_cand.get("continuation_indices") or [])
    continuation.append(int(next_cand["expanded_idx"]))
    label = str(prev_cand.get("label") or "资产负债表")
    is_full = is_full_statement_table_md(md_merged, label, locale)
    return {
        "label": label,
        "table": merged_el,
        "rows_count": len(merged_rows),
        "is_full": is_full,
        "expanded_idx": prev_cand["expanded_idx"],
        "page": prev_cand.get("page"),
        "continuation_indices": continuation,
    }
