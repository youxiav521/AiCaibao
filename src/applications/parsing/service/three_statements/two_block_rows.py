from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger

from src.applications.parsing.service.three_statements.cells import (
    find_raw_cell_covering_col,
    get_raw_cell_col_span,
    get_raw_cell_col_start,
    get_raw_cell_text,
    infer_table_total_columns,
    normalize_raw_rows,
)
from src.applications.parsing.service.three_statements.headers import (
    is_group_with_bank_or_company_dual_scope_table,
    is_header_like_row,
    looks_like_amount_text,
    looks_like_subject,
)


def row_has_any_text(cells: list[Any]) -> bool:
    return any(get_raw_cell_text(cell) for cell in cells)


def row_prefix_signature(cells: list[Any]) -> tuple[str, ...]:
    normalized = [re.sub(r"\s+", "", get_raw_cell_text(cell)) for cell in cells]
    while normalized and not normalized[-1]:
        normalized.pop()
    return tuple(normalized)


def _normalize_header_token(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _looks_like_line_no_header(text: str) -> bool:
    token = _normalize_header_token(text)
    return token in {"行次", "行号", "行项目", "line", "lineno", "no.", "no"}


def _looks_like_project_header(text: str) -> bool:
    token = _normalize_header_token(text)
    return token in {"项目", "项", "item", "items", "project"}


def _find_repeating_statement_block_starts(raw_rows: list[list[Any]], total_cols: int) -> list[int]:
    best_starts: list[int] = []
    best_score = 0
    for row in raw_rows[:12]:
        starts: list[int] = []
        for col_idx in range(total_cols):
            text = get_raw_cell_text(find_raw_cell_covering_col(row, col_idx))
            if looks_like_subject(text):
                starts.append(col_idx)
        if len(starts) < 3:
            continue
        scored_starts: list[int] = []
        for start in starts:
            next_text = get_raw_cell_text(find_raw_cell_covering_col(row, start + 1)) if start + 1 < total_cols else ""
            amount_text = get_raw_cell_text(find_raw_cell_covering_col(row, start + 2)) if start + 2 < total_cols else ""
            score = 1
            if _looks_like_line_no_header(next_text):
                score += 1
            if looks_like_amount_text(amount_text):
                score += 1
            if looks_like_amount_text(next_text):
                score += 2
            if score >= 2:
                scored_starts.append(start)
        if len(scored_starts) >= 3 and len(scored_starts) > best_score:
            best_starts = scored_starts
            best_score = len(scored_starts)
    return best_starts


def _detect_header_row_multi_block(raw_rows: list[list[Any]], total_cols: int) -> list[int]:
    """通过表头行检测多排结构，寻找重复出现的"项目"列。"""
    for row in raw_rows[:5]:
        project_cols: list[int] = []
        for col_idx in range(total_cols):
            text = get_raw_cell_text(find_raw_cell_covering_col(row, col_idx))
            if _looks_like_project_header(text):
                project_cols.append(col_idx)
        
        if len(project_cols) >= 3:
            widths = []
            for i in range(len(project_cols) - 1):
                widths.append(project_cols[i + 1] - project_cols[i])
            if widths and len(set(widths)) == 1 and widths[0] >= 2:
                return project_cols
    return []


def _copy_cell_with_shifted_colstart(cell: Any, block_start: int) -> Any:
    if not isinstance(cell, dict):
        return cell
    shifted = dict(cell)
    col_start = get_raw_cell_col_start(cell)
    if col_start is None:
        return shifted
    new_start = col_start - block_start
    for key in ("colstart", "colStart", "col_start"):
        if key in shifted:
            shifted[key] = new_start
    return shifted


def _slice_row_by_col_range(row: list[Any], start_col: int, end_col: int) -> Optional[list[Any]]:
    cells: list[Any] = []
    col_cursor = 0
    for cell in row:
        col_span = get_raw_cell_col_span(cell)
        col_start = get_raw_cell_col_start(cell)
        start = col_start if col_start is not None else col_cursor
        end = start + col_span
        if start < end_col and end > start_col:
            if start < start_col or end > end_col:
                return None
            cells.append(_copy_cell_with_shifted_colstart(cell, start_col))
        col_cursor = max(col_cursor, end) if col_start is not None else end
    return cells


def _looks_like_repeated_statement_header_row(cells: list[Any]) -> bool:
    text = "".join(_normalize_header_token(get_raw_cell_text(cell)) for cell in cells)
    return any(token in text for token in ("项目", "行次", "金额", "item", "line", "amount"))


def _drop_repeated_header_rows(rows: list[list[Any]], reference_signature: tuple[str, ...]) -> list[list[Any]]:
    if not rows:
        return rows
    out = list(rows)
    while out and reference_signature and row_prefix_signature(out[0]) == reference_signature:
        out = out[1:]
    if out and is_header_like_row(out[0]) and _looks_like_repeated_statement_header_row(out[0]):
        out = out[1:]
    return out


def _maybe_split_multi_block_rows(rows: Any) -> Optional[list[list[Any]]]:
    raw_rows = normalize_raw_rows(rows)
    if not raw_rows:
        return None
    total_cols = infer_table_total_columns(raw_rows)
    if total_cols < 6:
        return None

    block_starts = _find_repeating_statement_block_starts(raw_rows, total_cols)
    
    if len(block_starts) < 3:
        block_starts = _detect_header_row_multi_block(raw_rows, total_cols)
    
    if len(block_starts) < 3:
        return None

    block_ranges: list[tuple[int, int]] = []
    for idx, start in enumerate(block_starts):
        end = block_starts[idx + 1] if idx + 1 < len(block_starts) else total_cols
        if end - start < 2:
            return None
        block_ranges.append((start, end))

    blocks: list[list[list[Any]]] = [[] for _ in block_ranges]
    for row in raw_rows:
        for block_idx, (start, end) in enumerate(block_ranges):
            sliced = _slice_row_by_col_range(row, start, end)
            if sliced is None:
                continue
            if row_has_any_text(sliced):
                blocks[block_idx].append(sliced)

    if sum(1 for block in blocks if block) < 3:
        return None

    first_block = blocks[0]
    if not first_block:
        return None
    header_signature = row_prefix_signature(first_block[0])
    merged: list[list[Any]] = []
    for block_idx, block in enumerate(blocks):
        block_rows = block if block_idx == 0 else _drop_repeated_header_rows(block, header_signature)
        merged.extend(block_rows)
    
    if merged:
        logger.info(
            "[multi-block-split] total_cols={} block_starts={} input_rows={} output_rows={}",
            total_cols, block_starts, len(raw_rows), len(merged)
        )
    
    return merged or None


def maybe_split_two_block_rows(rows: Any) -> Any:
    multi_block_rows = _maybe_split_multi_block_rows(rows)
    if multi_block_rows:
        return multi_block_rows

    raw_rows = normalize_raw_rows(rows)
    if not raw_rows:
        return rows

    total_cols = infer_table_total_columns(raw_rows)
    if total_cols < 4:
        return rows

    preview_grid: list[list[Optional[dict[str, Any]]]] = []
    for row in raw_rows[:6]:
        preview_grid.append([cell if isinstance(cell, dict) else None for cell in row])
    if preview_grid and is_group_with_bank_or_company_dual_scope_table(
        preview_grid, min(4, len(preview_grid))
    ):
        return rows

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
        if candidate_left >= 2 and candidate_left == candidate_right:
            empty_sep = 0
            for row in raw_rows:
                sep_cell = find_raw_cell_covering_col(row, candidate_sep)
                if not get_raw_cell_text(sep_cell):
                    empty_sep += 1
            if empty_sep / max(len(raw_rows), 1) >= 0.7:
                left_end = candidate_left
                right_start = candidate_left + 1
                separator_col = candidate_sep

    if left_end is None or right_start is None:
        return rows

    both_subject_rows = 0
    for row in raw_rows:
        left_cell = find_raw_cell_covering_col(row, 0)
        right_cell = find_raw_cell_covering_col(row, right_start)
        if looks_like_subject(get_raw_cell_text(left_cell)) and looks_like_subject(get_raw_cell_text(right_cell)):
            both_subject_rows += 1
            if both_subject_rows >= 2:
                break

    if both_subject_rows < 2:
        return rows

    left_rows: list[list[Any]] = []
    right_rows: list[list[Any]] = []

    for row in raw_rows:
        left_cells: list[Any] = []
        right_cells: list[Any] = []
        col_cursor = 0
        crossed = False

        for cell in row:
            col_span = get_raw_cell_col_span(cell)
            col_start = get_raw_cell_col_start(cell)
            start = col_start if col_start is not None else col_cursor
            end = start + col_span

            if separator_col is not None:
                if start <= separator_col < end and not (start == separator_col and end == separator_col + 1):
                    crossed = True
                    break
                if start == separator_col and end == separator_col + 1:
                    col_cursor = max(col_cursor, end) if col_start is not None else end
                    continue

            if start < left_end < end or start < right_start < end:
                if left_end == right_start:
                    # 跨边界合并cell：按上一行推断text归属，拆成左右两个cell
                    left_span = left_end - start
                    right_span = end - left_end
                    text = get_raw_cell_text(cell)
                    text_to_right = False
                    # 向上回看最近5行，找到第一个非空cell判断是数字还是文本
                    for prev_row in reversed(left_rows[-5:]):
                        prev_cell = find_raw_cell_covering_col(prev_row, left_end - 1)
                        prev_text = get_raw_cell_text(prev_cell) if prev_cell else ""
                        if prev_text:
                            if looks_like_amount_text(prev_text):
                                text_to_right = True
                            break

                    if left_span > 0:
                        left_cell = dict(cell) if isinstance(cell, dict) else {"text": str(cell or "")}
                        left_cell["colspan"] = left_span
                        if "colSpan" in left_cell:
                            left_cell["colSpan"] = left_span
                        left_cell["text"] = "" if text_to_right else text
                        left_cells.append(left_cell)
                    if right_span > 0:
                        right_cell = dict(cell) if isinstance(cell, dict) else {"text": str(cell or "")}
                        right_cell["colspan"] = right_span
                        if "colSpan" in right_cell:
                            right_cell["colSpan"] = right_span
                        right_cell["text"] = text if text_to_right else ""
                        # 右块内列号从 0 起（right_start 是右块物理起点）
                        right_cell["colstart"] = left_end - right_start
                        right_cell["colStart"] = left_end - right_start
                        right_cells.append(right_cell)

                    col_cursor = end
                    continue
                else:
                    crossed = True
                    break

            if end <= left_end:
                left_cells.append(cell)
            elif start >= right_start:
                right_cells.append(_copy_cell_with_shifted_colstart(cell, right_start))

            col_cursor = max(col_cursor, end) if col_start is not None else end

        if crossed:
            if row_has_any_text(row):
                left_rows.append(row)
            continue

        if row_has_any_text(left_cells):
            left_rows.append(left_cells)
        if row_has_any_text(right_cells):
            right_rows.append(right_cells)

    prefix_len = 0
    while (
        prefix_len < len(left_rows)
        and prefix_len < len(right_rows)
        and row_prefix_signature(left_rows[prefix_len])
        and row_prefix_signature(left_rows[prefix_len]) == row_prefix_signature(right_rows[prefix_len])
    ):
        prefix_len += 1
    if prefix_len:
        right_rows = right_rows[prefix_len:]
    elif len(right_rows) >= 1 and is_header_like_row(right_rows[0]):
        right_rows = right_rows[1:]
    
    result = left_rows + right_rows
    if len(result) > len(raw_rows):
        logger.info(
            "[two-block-split] total_cols={} left_end={} input_rows={} output_rows={}",
            total_cols, left_end, len(raw_rows), len(result)
        )
    
    return result
