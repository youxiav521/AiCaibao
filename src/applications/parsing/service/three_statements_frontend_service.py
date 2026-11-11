from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import uuid
from datetime import date
from collections import defaultdict
from typing import Any, Optional

from src.applications.parsing.service.three_statements_row_context import build_lines_context

from src.applications.parsing.service.three_statements.caliber import (
    coerce_cell_caliber_to_binary as _coerce_cell_caliber_to_binary,
    finalize_caliber_label_for_output as _finalize_caliber_label_for_output,
)
from src.applications.parsing.service.three_statements.balance_sheet_merge import (
    can_merge_adjacent_balance_sheet_halves as _can_merge_adjacent_balance_sheet_halves,
    detect_merge_header_skip as _detect_merge_header_skip,
    align_continuation_rows_to_primary_layout as _align_continuation_rows_to_primary_layout,
    looks_like_column_header_row as _looks_like_column_header_row,
    looks_like_independent_balance_sheet_halves as _looks_like_independent_balance_sheet_halves,
    looks_like_independent_same_page_tables as _looks_like_independent_same_page_tables,
    merge_balance_sheet_split_candidates as _merge_balance_sheet_split_candidates_impl,
    row_text_signature_for_merge as _row_text_signature_for_merge,
    statement_labels_merge_compatible as _statement_labels_merge_compatible,
    statement_scope_key as _statement_scope_key,
    table_blob_lower as _table_blob_lower,
)
from src.applications.parsing.service.three_statements.candidate_selection import (
    estimate_table_rows_count as _estimate_table_rows_count,
    pick_better_statement_candidate as _pick_better_statement_candidate,
    promote_no_prefix_in_multi as _promote_no_prefix_in_multi_impl,
    push_three_statement_candidate as _push_three_statement_candidate_impl,
    update_best_candidate as _update_best_candidate,
)
from src.applications.parsing.service.three_statements.cells import (
    coerce_bbox as _coerce_bbox,
    coerce_cell as _coerce_cell,
    coerce_rows as _coerce_rows,
    expand_colspan_header_rows as _expand_colspan_header_rows,
    get_page_at as _get_page_at,
    get_raw_cell_text as _get_raw_cell_text,
    get_text as _get_text,
    normalize_raw_rows as _normalize_raw_rows,
    resolve_cell_column_key as _resolve_cell_column_key,
    to_int as _to_int,
)
from src.applications.parsing.service.three_statements.column_filter import (
    build_table_column_filter_prompt as _build_table_column_filter_prompt,
    build_table_column_profiles as _build_table_column_profiles,
    call_table_column_filter_llm as _call_table_column_filter_llm,
    choose_keep_column_indices_by_llm as _choose_keep_column_indices_by_llm,
    choose_keep_column_indices_by_rule as _choose_keep_column_indices_by_rule,
)
from src.applications.parsing.service.three_statements.combined_split import (
    try_split_combined_statement as _try_split_combined_statement,
)
from src.applications.parsing.service.three_statements.dual_caliber_split import (
    allowed_data_keys_for_merge_standalone_scope as _allowed_data_keys_for_merge_standalone_scope,
    logical_column_caliber_map_complete as _logical_column_caliber_map_complete,
    rewrite_caliber_labels_after_scope_split as _rewrite_caliber_labels_after_scope_split,
    scoped_category_for_split as _scoped_category_for_split,
    should_split_table_by_merge_vs_standalone_columns as _should_split_table_by_merge_vs_standalone_columns,
    slice_table_rows_by_logical_data_keys as _slice_table_rows_by_logical_data_keys,
)
from src.applications.parsing.service.three_statements.currency import (
    CURRENCY_UNIT_EN_TO_KIND as _CURRENCY_UNIT_EN_TO_KIND,
    CURRENCY_UNIT_KIND_BILLIONS as _CURRENCY_UNIT_KIND_BILLIONS,
    CURRENCY_UNIT_KIND_MILLIONS as _CURRENCY_UNIT_KIND_MILLIONS,
    CURRENCY_UNIT_KIND_ONE as _CURRENCY_UNIT_KIND_ONE,
    CURRENCY_UNIT_KIND_THOUSANDS as _CURRENCY_UNIT_KIND_THOUSANDS,
    CURRENCY_UNIT_KIND_TO_ZH as _CURRENCY_UNIT_KIND_TO_ZH,
    CURRENCY_UNIT_ZH_TO_KIND as _CURRENCY_UNIT_ZH_TO_KIND,
    collect_currency_scan_text as _collect_currency_scan_text,
    currency_unit_for_api as _currency_unit_for_api_impl,
    currency_unit_zh_for_table as _currency_unit_zh_for_table_impl,
    detect_currency_unit_kind as _detect_currency_unit_kind_impl,
    normalize_apostrophes_for_rm_scan as _normalize_apostrophes_for_rm_scan,
    scan_blob_for_currency_kind as _scan_blob_for_currency_kind,
)
from src.applications.parsing.service.three_statements.headers import (
    AMOUNT_RE as _AMOUNT_RE,
    ensure_keep_columns_include_all_dual_scope_amounts as _ensure_keep_columns_include_all_dual_scope_amounts,
    guess_header_row_count as _guess_header_row_count,
    identify_auxiliary_column_indices as _identify_auxiliary_column_indices,
    is_auxiliary_statement_column_header as _is_auxiliary_statement_column_header,
    is_group_with_bank_or_company_dual_scope_table as _is_group_with_bank_or_company_dual_scope_table,
    is_header_like_row as _is_header_like_row,
    is_statement_date_fragment as _is_statement_date_fragment,
    is_statement_meta_fragment as _is_statement_meta_fragment,
    is_title_row as _is_title_row,
    looks_like_amount_text as _looks_like_amount_text,
    looks_like_date_header_text as _looks_like_date_header_text,
    looks_like_sequence_number_text as _looks_like_sequence_number_text,
    normalize_header_text as _normalize_header_text,
)
from src.applications.parsing.service.three_statements.header_prepend import (
    prepend_clean_statement_header_row as _prepend_clean_statement_header_row_impl,
    drop_note_column_by_first_row as _drop_note_column_by_first_row,
)
from src.applications.parsing.service.three_statements.i18n import (
    CURRENCY_UNIT_ZH_TO_EN as _CURRENCY_UNIT_ZH_TO_EN,
    apply_i18n_to_statement_table_response as _apply_i18n_to_statement_table_response,
    locale_prefers_english as _locale_prefers_english,
    translate_caliber_label_for_response as _translate_caliber_label_for_response,
    translate_filing_period_for_response as _translate_filing_period_for_response,
    translate_statement_label as _translate_statement_label,
)
from src.applications.parsing.service.three_statements.labels import (
    STATEMENT_ORDER,
    apply_en_scope_tokens_to_pl_oci as _apply_en_scope_tokens_to_pl_oci,
    get_base_type as _get_base_type,
    has_scope_prefix as _has_scope_prefix,
    is_three_main_label as _is_three_main_label,
    normalize_category_to_three_main as _normalize_category_to_three_main,
    normalize_statement_label as _normalize_statement_label,
    strip_scope_prefix_from_category as _strip_scope_prefix_from_category,
)
from src.applications.parsing.service.three_statements.continuation import (
    base_statement_type as _base_statement_type,
    collect_continuation_rows as _collect_continuation_rows_impl,
    collect_continuation_rows_backward as _collect_continuation_rows_backward_impl,
    collect_continuation_rows_forward as _collect_continuation_rows_forward_impl,
    deduplicate_rows as _deduplicate_rows,
    is_safe_continuation as _is_safe_continuation_impl,
    row_signature as _row_signature,
    strip_residual_header_rows as _strip_residual_header_rows,
)
from src.applications.parsing.service.three_statements.multi_period import (
    all_logical_col_keys_in_rows as _all_logical_col_keys_in_rows,
    compact_multiline_header_cell_text as _compact_multiline_header_cell_text,
    filing_period_label_for_reporting_date_pair as _filing_period_label_for_reporting_date_pair,
    format_cn_reporting_date as _format_cn_reporting_date,
    patch_pair_header_period_labels as _patch_pair_header_period_labels_impl,
    preferred_period_column_header_display as _preferred_period_column_header_display_impl,
    slice_header_cells_to_ordered_logical_keys as _slice_header_cells_to_ordered_logical_keys,
    slice_rows_to_ordered_logical_keys as _slice_rows_to_ordered_logical_keys,
    touch_accounting_years_for_pair_columns as _touch_accounting_years_for_pair_columns,
)
from src.applications.parsing.service.three_statements.notes_context import (
    find_nearest_statement_title as _find_nearest_statement_title_impl,
    truncate_cashflow_supplementary as _truncate_cashflow_supplementary,
)
from src.applications.parsing.service.three_statements.org_table_ids import (
    ORG_TABLE_SPLIT_KEY as _ORG_TABLE_SPLIT_KEY,
    build_org_table_merge_ids as _build_org_table_merge_ids,
    get_element_id_str as _get_element_id_str,
    iter_org_tables_with_source_item_index as _iter_org_tables_with_source_item_index_impl,
    org_tables_element_id as _org_tables_element_id,
    source_merge_ids_for_table_element as _source_merge_ids_for_table_element,
    strip_org_table_derivative_suffixes as _strip_org_table_derivative_suffixes,
)
from src.applications.parsing.service.three_statements.row_grid import (
    expand_rows_to_cell_grid as _expand_rows_to_cell_grid,
    strip_col_start_keys as _strip_col_start_keys,
)
from src.applications.parsing.service.three_statements.row_preprocess import (
    fix_missing_spaces_in_rows as _fix_missing_spaces_in_rows,
    merge_continuation_rows as _merge_continuation_rows,
)
from src.applications.parsing.service.three_statements.row_text import (
    fill_null_accounting_year_from_document as _fill_null_accounting_year_from_document,
    get_row_first_cell_text as _get_row_first_cell_text,
    rows_to_compact_text as _rows_to_compact_text,
)
from src.applications.parsing.service.three_statements.statement_content import (
    content_matches_base_type as _content_matches_base_type,
    content_matches_other_base_type as _content_matches_other_base_type,
)
from src.applications.parsing.service.three_statements.side_by_side import (
    try_split_side_by_side_table as _try_split_side_by_side_table,
)
from src.applications.parsing.service.three_statements.two_block_rows import (
    maybe_split_two_block_rows as _maybe_split_two_block_rows,
    row_has_any_text as _row_has_any_text,
    row_prefix_signature as _row_prefix_signature,
)
from src.applications.parsing.utils.financial_recognition_and_extraction_utils import (
    augment_column_years_for_comparative_period_labels,
    collect_neighbor_text_context,
    detect_caliber_label,
    detect_gaap_type,
    infer_column_index_to_caliber_for_table,
    infer_column_index_to_caliber_from_header_cells,
    infer_column_index_to_date_for_table,
    infer_column_index_to_date_from_header_cells,
    infer_column_index_to_gaap_for_table,
    infer_column_index_to_gaap_from_header_cells,
    infer_column_index_to_statement_type_for_table,
    infer_column_index_to_statement_type_from_header_cells,
    infer_column_index_to_year_for_table,
    infer_document_years_and_period,
    infer_single_table_filing_period,
    infer_single_table_reporting_year,
    interpolate_balance_sheet_reporting_years_by_peer_pages,
    map_to_statement_type,
    parse_statement_column_date_from_text,
    table_element_markdown,
    _infer_column_header_texts,
    _extract_column_header_texts_from_header_cells,
)
from src.applications.parsing.utils.mineru_storage import load_mineru_items
from src.applications.parsing.utils.table_classification_utils import (
    detect_statement_type_by_content,
    is_derivative_statement_title,
    is_full_statement_table_md,
)
from src.utils.chinese_converter import traditional_to_simplified

_logger = logging.getLogger(__name__)


_MERGED_STATEMENT_LABELS: tuple[str, ...] = (
    "合并资产负债表",
    "合并利润表",
    "合并现金流量表",
    "合并所有者权益变动表",
)
_NO_PREFIX_STATEMENT_LABELS: tuple[str, ...] = (
    "资产负债表",
    "利润表",
    "现金流量表",
    "所有者权益变动表",
)


def _prepend_clean_statement_header_row(
    rows: Any,
    table_el: dict[str, Any],
    *,
    fallback_page_at: Optional[int],
    locale: Optional[str] = None,
) -> Optional[list[list[Any]]]:
    return _prepend_clean_statement_header_row_impl(
        rows,
        table_el,
        fallback_page_at=fallback_page_at,
        locale=locale,
    )


def _filter_statement_table_rows(rows: Any) -> Any:
    raw_row_cells = _normalize_raw_rows(rows)
    if len(raw_row_cells) < 2:
        return rows

    raw_row_cells = _strip_col_start_keys(raw_row_cells)
    cell_grid = _expand_rows_to_cell_grid(raw_row_cells)
    header_row_count = _guess_header_row_count(cell_grid, max_rows=6)

    _logger.info(
        "Filter table: total_rows=%d, total_cols=%d, header_row_count=%d",
        len(cell_grid), len(cell_grid[0]) if cell_grid else 0, header_row_count,
    )

    kept_column_indices = None
    if header_row_count > 0:
        kept_column_indices = _choose_keep_column_indices_by_llm(cell_grid, header_row_count=header_row_count)
    if not kept_column_indices:
        kept_column_indices = _choose_keep_column_indices_by_rule(cell_grid, header_row_count=header_row_count)
    if not kept_column_indices and header_row_count == 0:
        kept_column_indices = _choose_keep_column_indices_by_llm(cell_grid, header_row_count=0)
    if not kept_column_indices:
        _logger.warning("Column selection failed, returning raw rows (total=%d)", len(raw_row_cells))
        return rows

    total_cols = len(cell_grid[0])
    kept_column_indices = _ensure_keep_columns_include_all_dual_scope_amounts(
        cell_grid,
        header_row_count=header_row_count,
        keep_column_indices=list(kept_column_indices),
        total_columns=total_cols,
    )

    _logger.info("Kept column indices: %s", kept_column_indices)

    # Preserve leading title/meta rows (company name, date, unit, etc.) for front-end display.
    # These rows are useful to downstream rendering/extraction even if they are not table body rows.
    title_row_end = 0
    for row_index in range(min(8, len(cell_grid))):
        row_cells = [cell_grid[row_index][ci] for ci in kept_column_indices if ci < len(cell_grid[row_index])]
        real_cells = [c for c in row_cells if c is not None]
        if real_cells and _is_title_row(real_cells):
            title_row_end = row_index + 1
        else:
            break

    preserved_title_row_count = title_row_end
    if preserved_title_row_count > 0:
        _logger.info("Preserving %d title row(s) at start of statement table", preserved_title_row_count)

    start_row = title_row_end
    while start_row < len(cell_grid):
        row_cells = [cell_grid[start_row][ci] for ci in kept_column_indices if ci < len(cell_grid[start_row])]
        row_texts = [_get_raw_cell_text(c).strip() for c in row_cells if c is not None and _get_raw_cell_text(c).strip()]
        if row_texts and all(_looks_like_sequence_number_text(text) for text in row_texts):
            start_row += 1
            continue
        break

    output_row_count = preserved_title_row_count + len(cell_grid) - start_row
    if output_row_count <= 0:
        _logger.warning(
            "All %d rows were skipped as title/header! Falling back to start_row=title_row_end=%d",
            len(cell_grid), title_row_end,
        )
        start_row = title_row_end

    filtered_rows: list[list[dict[str, Any]]] = []
    for row_index in list(range(0, preserved_title_row_count)) + list(range(start_row, len(cell_grid))):
        filtered_row: list[dict[str, Any]] = []
        for column_index in kept_column_indices:
            cell_reference = cell_grid[row_index][column_index] if column_index < len(cell_grid[row_index]) else None
            if isinstance(cell_reference, dict):
                bounding_box = cell_reference.get("bBox") or cell_reference.get("bbox") or cell_reference.get("coordinate")
                page_number = _get_page_at(cell_reference)
                filtered_row.append(
                    {
                        "text": _get_raw_cell_text(cell_reference),
                        "rowSpan": 1,
                        "colSpan": 1,
                        "pageAt": page_number,
                        "bBox": bounding_box if isinstance(bounding_box, dict) else {},
                    }
                )
            else:
                filtered_row.append(
                    {"text": "", "rowSpan": 1, "colSpan": 1, "pageAt": None, "bBox": {}}
                )
        filtered_rows.append(filtered_row)
    return filtered_rows

def _detect_currency_unit_kind(
    table_el: dict[str, Any],
    context_above: str = "",
    previous_unit: Optional[str] = None,
) -> str:
    return _detect_currency_unit_kind_impl(
        table_el,
        context_above=context_above,
        previous_unit=previous_unit,
        table_caption_getter=_table_caption_joined,
    )


def _currency_unit_zh_for_table(
    table_el: dict[str, Any],
    context_above: str = "",
    previous_unit: Optional[str] = None,
) -> str:
    return _currency_unit_zh_for_table_impl(
        table_el,
        context_above=context_above,
        previous_unit=previous_unit,
        table_caption_getter=_table_caption_joined,
    )


def _currency_unit_for_api(
    table_el: dict[str, Any],
    locale: Optional[str],
    context_above: str = "",
    previous_unit: Optional[str] = None,
) -> str:
    return _currency_unit_for_api_impl(
        table_el,
        locale,
        context_above=context_above,
        previous_unit=previous_unit,
        table_caption_getter=_table_caption_joined,
    )




def _find_nearest_statement_title(
    items: list[Any], *, table_index: int, max_back: int = 50, max_page_gap: int = 2
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    return _find_nearest_statement_title_impl(
        items,
        table_index=table_index,
        normalize_statement_label=_normalize_statement_label,
        max_back=max_back,
        max_page_gap=max_page_gap,
    )














































def _merge_balance_sheet_split_candidates(
    prev_cand: dict[str, Any],
    next_cand: dict[str, Any],
    *,
    locale: Optional[str],
) -> dict[str, Any]:
    return _merge_balance_sheet_split_candidates_impl(
        prev_cand,
        next_cand,
        locale=locale,
        is_full_statement_table_md=is_full_statement_table_md,
    )


def _push_three_statement_candidate(
    label_to_candidates: defaultdict[str, list[dict[str, Any]]],
    siblings: list[dict[str, Any]],
    label: str,
    candidate: dict[str, Any],
    *,
    locale: Optional[str],
    statement_strategy: Any = None,
) -> None:
    return _push_three_statement_candidate_impl(
        label_to_candidates,
        siblings,
        label,
        candidate,
        locale=locale,
        statement_strategy=statement_strategy,
        looks_like_independent_balance_sheet_halves=_looks_like_independent_balance_sheet_halves,
        can_merge_adjacent_balance_sheet_halves=_can_merge_adjacent_balance_sheet_halves,
        merge_balance_sheet_split_candidates=_merge_balance_sheet_split_candidates,
        looks_like_independent_same_page_tables=_looks_like_independent_same_page_tables,
    )


def _promote_no_prefix_in_multi(label_to_candidates: defaultdict[str, list[dict[str, Any]]]) -> None:
    return _promote_no_prefix_in_multi_impl(
        label_to_candidates,
        no_prefix_statement_labels=_NO_PREFIX_STATEMENT_LABELS,
        merged_statement_labels=_MERGED_STATEMENT_LABELS,
        table_caption_joined=_table_caption_joined,
    )


def _is_safe_continuation(
    items: list[Any], idx: int, primary_base: str, primary_label: str
) -> bool:
    return _is_safe_continuation_impl(
        items,
        idx,
        primary_base,
        primary_label,
        normalize_statement_label=_normalize_statement_label,
        find_nearest_statement_title=_find_nearest_statement_title,
    )


def _collect_continuation_rows_forward(
    items: list[Any],
    primary_idx: int,
    primary_label: str,
    primary_page: Optional[int],
    primary_base: str,
    *,
    max_page_gap: int = 2,
    max_look: int = 60,
    max_tables: int = 8,
) -> tuple[list[Any], list[int]]:
    return _collect_continuation_rows_forward_impl(
        items,
        primary_idx,
        primary_label,
        primary_page,
        primary_base,
        normalize_statement_label=_normalize_statement_label,
        find_nearest_statement_title=_find_nearest_statement_title,
        max_page_gap=max_page_gap,
        max_look=max_look,
        max_tables=max_tables,
    )


def _collect_continuation_rows_backward(
    items: list[Any],
    primary_idx: int,
    primary_label: str,
    primary_page: Optional[int],
    primary_base: str,
    *,
    max_page_gap: int = 2,
    max_look: int = 60,
    max_tables: int = 8,
) -> tuple[list[Any], list[int]]:
    return _collect_continuation_rows_backward_impl(
        items,
        primary_idx,
        primary_label,
        primary_page,
        primary_base,
        normalize_statement_label=_normalize_statement_label,
        find_nearest_statement_title=_find_nearest_statement_title,
        max_page_gap=max_page_gap,
        max_look=max_look,
        max_tables=max_tables,
    )


def _collect_continuation_rows(
    items: list[Any],
    primary_idx: int,
    primary_label: str,
    *,
    max_page_gap: int = 2,
    max_look: int = 60,
    max_tables: int = 8,
) -> tuple[list[Any], list[Any], list[int]]:
    return _collect_continuation_rows_impl(
        items,
        primary_idx,
        primary_label,
        normalize_statement_label=_normalize_statement_label,
        find_nearest_statement_title=_find_nearest_statement_title,
        max_page_gap=max_page_gap,
        max_look=max_look,
        max_tables=max_tables,
    )


def _merge_forward_statement_continuations(
    candidate: dict[str, Any],
    expanded_items: list[Any],
    *,
    locale: Optional[str],
    excluded_indices: set[int],
) -> tuple[dict[str, Any], list[int]]:
    """合并当前财报表格后紧邻的跨页续表，并返回已消费的源表索引。"""
    table = candidate.get("table") if isinstance(candidate.get("table"), dict) else {}
    primary_idx = candidate.get("expanded_idx")
    label = str(candidate.get("label") or "")
    base_type = _get_base_type(label)
    if not table or not isinstance(primary_idx, int) or not label or not base_type:
        return candidate, []

    _, continuation_indices = _collect_continuation_rows_forward(
        expanded_items,
        primary_idx,
        label,
        _get_page_at(table),
        base_type,
    )
    continuation_indices = [
        index for index in continuation_indices if index not in excluded_indices
    ]
    if not continuation_indices:
        return candidate, []

    merged_table = dict(table)
    merged_rows = list(table.get("rows") or [])
    # 合并前先对各表做 Note 列预清洗，确保主表和续表列结构一致，
    # 避免后续按索引删列时误删续表数据列。
    # 预清洗后需清除 headerCells，否则后续 _prepend_clean_statement_header_row
    # 会用旧的多列 headerCells 处理已删列的 rows，导致列错位。
    _pre_cleaned = _drop_note_column_by_first_row(merged_rows)
    if _pre_cleaned is not None:
        merged_rows = _pre_cleaned
        merged_table["headerCells"] = None
    markdown_parts = [str(table.get("md") or "").strip()]
    for continuation_idx in continuation_indices:
        continuation = expanded_items[continuation_idx]
        if not isinstance(continuation, dict):
            continue
        continuation_rows = continuation.get("rows")
        if isinstance(continuation_rows, list) and continuation_rows:
            _pre_cleaned_cont = _drop_note_column_by_first_row(continuation_rows)
            if _pre_cleaned_cont is not None:
                continuation_rows = _pre_cleaned_cont
            merged_rows, continuation_rows = _align_continuation_rows_to_primary_layout(
                merged_rows,
                continuation_rows,
                primary_header_cells=merged_table.get("headerCells"),
                continuation_header_cells=continuation.get("headerCells"),
            )
            header_skip = _detect_merge_header_skip(merged_rows, continuation_rows)
            merged_rows.extend(continuation_rows[header_skip:])
        markdown = str(continuation.get("md") or "").strip()
        if markdown:
            markdown_parts.append(markdown)

    merged_table["rows"] = merged_rows
    merged_table["md"] = "\n".join(part for part in markdown_parts if part)
    merged_candidate = dict(candidate)
    merged_candidate["table"] = merged_table
    merged_candidate["rows_count"] = len(merged_rows)
    merged_candidate["is_full"] = is_full_statement_table_md(
        merged_table["md"], label, locale=locale
    )
    merged_candidate["continuation_indices"] = continuation_indices
    return merged_candidate, continuation_indices


def _preferred_period_column_header_display(raw_header: str, d_target: date) -> str:
    return _preferred_period_column_header_display_impl(
        raw_header,
        d_target,
        parse_statement_column_date_from_text=parse_statement_column_date_from_text,
    )


def _patch_pair_header_period_labels(
    rows: list[list[Any]],
    d0: date,
    d1: date,
    *,
    display_left: str,
    display_right: str,
) -> None:
    return _patch_pair_header_period_labels_impl(
        rows,
        d0,
        d1,
        display_left=display_left,
        display_right=display_right,
        looks_like_column_header_row=_looks_like_column_header_row,
        parse_statement_column_date_from_text=parse_statement_column_date_from_text,
    )


def _iter_org_tables_with_source_item_index(
    items: list[Any],
):
    return _iter_org_tables_with_source_item_index_impl(
        items,
        try_split_side_by_side_table=_try_split_side_by_side_table,
    )


def _table_caption_joined(table_el: dict[str, Any]) -> str:
    cap = table_el.get("table_caption") or table_el.get("tableCaption")
    lines: list[str] = []
    sheet_name = table_el.get("sheetName") or table_el.get("sheet_name")
    if isinstance(sheet_name, str) and sheet_name.strip():
        lines.append(sheet_name.strip())
    if isinstance(cap, list):
        for x in cap:
            if isinstance(x, str) and x.strip():
                lines.append(x.strip())
            elif isinstance(x, dict):
                text = str(x.get("content") or x.get("text") or "").strip()
                if text:
                    lines.append(text)
    elif isinstance(cap, str) and cap.strip():
        lines.append(cap.strip())
    return "\n".join(lines)


def _extract_table_inline_title(table_el: dict[str, Any]) -> str:
    """取表第一行第一格非空文本作为内嵌标题（监管财报/离岸 PDF 常见把标题塞进表内）。"""
    rows = table_el.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return ""
    first_row = rows[0]
    cells = first_row if isinstance(first_row, list) else first_row.get("cells", [])
    if not isinstance(cells, list):
        return ""
    for c in cells:
        if not isinstance(c, dict):
            continue
        t = (c.get("text") or "").strip()
        if t:
            return t
    return ""


def _infer_scope_prefix_from_evidence(*text_parts: str) -> Optional[str]:
    blob = "\n".join(p for p in text_parts if p).lower()
    has_merge = any(t in blob for t in ("合并", "consolidated"))
    has_parent = any(t in blob for t in ("母公司", "单体", "standalone", "parent only", "parent company"))
    if has_merge and not has_parent:
        return "合并"
    if has_parent and not has_merge:
        return "母公司"
    return None


def _apply_scope_prefix_to_category(category: str, *evidence_parts: str) -> str:
    if category not in _NO_PREFIX_STATEMENT_LABELS:
        return category
    scope_prefix = _infer_scope_prefix_from_evidence(*evidence_parts)
    if scope_prefix == "合并":
        return f"合并{category}"
    if scope_prefix == "母公司":
        return f"母公司{category}"
    return category


def _strip_multiline_suffix_from_numeric_amount_cells(rows: list[list[dict[str, Any]]]) -> None:
    """金额列首行已为典型数值时，仅保留首行（去掉 MinerU 等合并进单元格的换行及后续片段）。"""
    for row in rows:
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            if _resolve_cell_column_key(cell, col_idx) <= 0:
                continue
            t = str(cell.get("text") or "")
            if "\n" not in t:
                continue
            first = t.split("\n")[0].strip()
            if _looks_like_amount_text(first):
                cell["text"] = first






















































def expand_financial_table_if_multi_period(
    table: dict[str, Any],
    *,
    locale: Optional[str] = None,
    stable_id_prefix: Optional[str] = None,
    allow_split: bool = True,
) -> list[dict[str, Any]]:
    """单表内并列 **三个及以上** 不同报告期截止日金额列时，按相邻期间拆成多张两列金额表。

    规则：表头日期排序后，每相邻两期为一张子表（较早为期初列、较晚为期末列）；
    间隔约 1 年 → ``年报``，约半年 → ``半年报``，约 3 个月 → ``一季度报`` / ``三季度报`` / ``季报``。

    输出顺序：在时间上**较新**的相邻对先输出（``_mp0`` 对应含最晚报告期的那一对），与常见横向列报
    （左新右旧）及监管附件阅读习惯一致；校验仍按时间升序相邻对进行。

    仅在「财报表 + 至少 3 个可解析的相异报告期」时生效；否则原样返回单元素列表，
    以便在 ``expand_financial_table_if_dual_caliber`` 之前调用且不改变旧行为。
    """
    _ = locale
    if not allow_split:
        return [table]
    if not table.get("isFinancialStatementTable"):
        return [table]
    rows = table.get("rows") or []
    if not isinstance(rows, list) or len(rows) < 2:
        return [table]

    cleaned_header_cells = table.get("cleanedHeaderCells")
    if cleaned_header_cells:
        col_date = infer_column_index_to_date_from_header_cells(cleaned_header_cells)
    else:
        col_date = infer_column_index_to_date_for_table(
            rows,
            context_above="",
            context_below="",
            report_name=str(table.get("category") or table.get("text") or ""),
        )
    dated = [(ck, d) for ck, d in col_date.items() if ck > 0 and d is not None]
    if len(dated) < 3:
        return [table]

    dated_sorted = sorted(dated, key=lambda x: (x[1], x[0]))
    uniq: list[tuple[int, date]] = []
    seen: set[date] = set()
    for ck, d in dated_sorted:
        if d in seen:
            continue
        seen.add(d)
        uniq.append((ck, d))
    if len(uniq) < 3:
        return [table]

    for i in range(len(uniq) - 1):
        d0, d1 = uniq[i][1], uniq[i + 1][1]
        if d1 <= d0:
            return [table]
        days = (d1 - d0).days
        if days < 25 or days > 450:
            return [table]

    min_ck = min(ck for ck, _ in uniq)
    all_keys = _all_logical_col_keys_in_rows(rows)
    prefix_keys = {k for k in all_keys if k < min_ck}
    stable = stable_id_prefix or str(table.get("id") or "") or uuid.uuid4().hex
    if cleaned_header_cells:
        col_header_raw = _extract_column_header_texts_from_header_cells(cleaned_header_cells)
    else:
        col_header_raw = _infer_column_header_texts(rows, header_rows=4)

    # 输出顺序：先拆「较新」的相邻期（与常见 PDF 从左到右 2025/2024/2023 及附件示意一致），
    # ``_mp0`` 始终对应该表内时间上最晚的一对相邻报告期。
    pair_count = len(uniq) - 1
    parts: list[dict[str, Any]] = []
    for out_i in range(pair_count):
        pair_idx = pair_count - 1 - out_i
        ck0, d0 = uniq[pair_idx]
        ck1, d1 = uniq[pair_idx + 1]
        # 较晚报告期在左（期末列）、较早在右（期初列），与常见 PDF 列报顺序（新→旧）一致。
        pair_keys_chrono = [ck1, ck0]
        leading: list[int] = []
        if 0 in prefix_keys:
            leading.append(0)
        leading.extend(sorted(k for k in prefix_keys if k > 0))
        ordered_keys = leading + pair_keys_chrono

        new_rows = _slice_rows_to_ordered_logical_keys(rows, ordered_keys)
        disp0 = _preferred_period_column_header_display(col_header_raw.get(ck0, ""), d0)
        disp1 = _preferred_period_column_header_display(col_header_raw.get(ck1, ""), d1)
        _patch_pair_header_period_labels(new_rows, d0, d1, display_left=disp1, display_right=disp0)
        _touch_accounting_years_for_pair_columns(new_rows, d1, d0)

        new_id = "{}_mp{}".format(stable, out_i)
        period_label = _filing_period_label_for_reporting_date_pair(d0, d1)
        split_entry: dict[str, Any] = {}
        for k, v in table.items():
            if k in ("rows", "years", "period", "id", "cleanedHeaderCells"):
                continue
            split_entry[k] = v
        split_entry["rows"] = new_rows
        # 同步切片 cleanedHeaderCells，保持与 rows 列对齐，供下游推断使用。
        if cleaned_header_cells:
            split_entry["cleanedHeaderCells"] = _slice_header_cells_to_ordered_logical_keys(
                cleaned_header_cells, ordered_keys
            )
        split_entry["years"] = d1.year
        split_entry["period"] = period_label
        split_entry["id"] = new_id
        mids = table.get("mergeIds") or []
        if len(mids) <= 1:
            split_entry["mergeIds"] = [new_id]
        else:
            split_entry["mergeIds"] = list(mids)
        pep = table.get("primaryElementId")
        if pep is not None:
            split_entry["primaryElementId"] = pep
        else:
            split_entry["primaryElementId"] = _strip_org_table_derivative_suffixes(stable)
        parts.append(split_entry)

    return parts


def expand_financial_table_if_dual_caliber(
    table: dict[str, Any],
    *,
    locale: Optional[str] = None,
    stable_id_prefix: Optional[str] = None,
    allow_split: bool = True,
    use_scope_prefix_category: bool = False,
) -> list[dict[str, Any]]:
    """同一张表内并列「合并口径金额列」与「单体口径金额列」时拆成两张表（先合并后单体）。

    按逻辑列号（colStart）切片，避免 MinerU 稀疏列与数组下标错位导致漏拆或列错乱。

    参数：
        use_scope_prefix_category:
            - ``False`` (默认，三大表接口契约)：拆出的两张子表 ``category`` 统一去掉
              口径前缀并归并到三大主表（资产负债表 / 利润表 / 现金流量表）。
            - ``True`` (org-tables 接口契约)：保留原始基础类型并按目标口径拼接前缀
              （``合并X`` / ``母公司X``），确保输出 ``category`` 落在
              ``STATEMENT_ORDER`` 全 15 个受控值之内。
    """
    if not allow_split:
        return [table]
    rows = table.get("rows") or []
    if not rows:
        return [table]

    logical_map = _logical_column_caliber_map_complete(rows)
    if not _should_split_table_by_merge_vs_standalone_columns(logical_map):
        return [table]

    merge_keys = _allowed_data_keys_for_merge_standalone_scope(logical_map, "合并")
    std_keys = _allowed_data_keys_for_merge_standalone_scope(logical_map, "单体")
    if not merge_keys or not std_keys:
        return [table]

    prefix_boundary = min(merge_keys | std_keys)

    parts: list[dict[str, Any]] = []
    original_category = str(table.get("category") or "")
    # 三大表接口契约：先去口径前缀，再归并到三大主表枚举。
    three_main_category = _normalize_category_to_three_main(
        _strip_scope_prefix_from_category(original_category)
    )
    # org-tables 契约：保留基础类型，由目标口径决定加 ``合并`` / ``母公司`` 前缀。
    base_no_prefix = (
        _strip_scope_prefix_from_category(original_category) or original_category
    )
    cleaned_header_cells = table.get("cleanedHeaderCells")
    for target, sfx, keys in (
        ("合并", "_grp", merge_keys),
        ("单体", "_std", std_keys),
    ):
        new_rows = _slice_table_rows_by_logical_data_keys(
            rows, keys, first_scope_data_logical_col=prefix_boundary
        )
        _rewrite_caliber_labels_after_scope_split(new_rows, target)
        new_id = f"{stable_id_prefix}{sfx}" if stable_id_prefix else uuid.uuid4().hex
        split_entry = {"years": table.get("years"), "period": table.get("period")}
        for k, v in table.items():
            if k in ("rows", "years", "period", "cleanedHeaderCells"):
                continue
            split_entry[k] = v
        split_entry["id"] = new_id
        if use_scope_prefix_category:
            # 目标口径 -> category 前缀映射：合并 → 合并X；单体 → 母公司X。
            scope_prefix = "合并" if target == "合并" else "母公司"
            new_category = (
                f"{scope_prefix}{base_no_prefix}" if base_no_prefix else original_category
            )
        else:
            new_category = three_main_category
        split_entry["text"] = _translate_statement_label(new_category, locale)
        split_entry["category"] = new_category
        split_entry["caliberLabel"] = target
        split_entry["rows"] = new_rows
        # 同步切片 cleanedHeaderCells（保持与 rows 列对齐）
        if cleaned_header_cells:
            # ordered_keys: subject 列（逻辑列号 < prefix_boundary）按升序，再追加 keys 升序
            hc_keys: set[int] = set()
            for hc_row in cleaned_header_cells:
                if not isinstance(hc_row, list):
                    continue
                cursor = 0
                for cell in hc_row:
                    if not isinstance(cell, dict):
                        cursor += 1
                        continue
                    cstart = cell.get("colStart")
                    if cstart is None:
                        cstart = cell.get("colstart")
                    if isinstance(cstart, int):
                        cursor = cstart
                    cspan_raw = cell.get("colSpan")
                    if cspan_raw is None:
                        cspan_raw = cell.get("colspan", 1)
                    try:
                        cspan = max(int(cspan_raw or 1), 1)
                    except (TypeError, ValueError):
                        cspan = 1
                    for offset in range(cspan):
                        hc_keys.add(cursor + offset)
                    cursor += cspan
            subject_keys = sorted({k for k in hc_keys if k < prefix_boundary})
            data_keys = sorted(keys)
            ordered_keys = subject_keys + data_keys
            split_entry["cleanedHeaderCells"] = _slice_header_cells_to_ordered_logical_keys(
                cleaned_header_cells, ordered_keys
            )
        # 与 ``/org-tables`` 契约一致：单列拆分后 ``mergeIds`` 指向该逻辑块 ``id``（含 _grp/_std）。
        mids = table.get("mergeIds") or []
        if len(mids) <= 1:
            split_entry["mergeIds"] = [new_id]
            split_entry["primaryElementId"] = _strip_org_table_derivative_suffixes(new_id)
        parts.append(split_entry)

    if len(parts) != 2:
        return [table]
    # 清理拆分后全空的数据列（如 colSpan 覆盖产生的空列）
    for p in parts:
        _rows = p.get("rows")
        if not isinstance(_rows, list) or not _rows:
            continue
        _data_keys = {
            _resolve_cell_column_key(_cell, _fallback_index)
            for _row in _rows
            if isinstance(_row, list)
            for _fallback_index, _cell in enumerate(_row)
            if isinstance(_cell, dict)
            and _resolve_cell_column_key(_cell, _fallback_index) > 0
        }
        if len(_data_keys) <= 1:
            continue
        _nonempty_data_keys = {
            _resolve_cell_column_key(_cell, _fallback_index)
            for _row in _rows
            if isinstance(_row, list)
            for _fallback_index, _cell in enumerate(_row)
            if isinstance(_cell, dict)
            and (_cell.get("text") or "").strip()
            and _resolve_cell_column_key(_cell, _fallback_index) > 0
        }
        _empty_cols = _data_keys - _nonempty_data_keys
        if not _empty_cols:
            continue
        for _row in _rows:
            if not isinstance(_row, list):
                continue
            _kept_cells = []
            for _fallback_index, _cell in enumerate(_row):
                if not isinstance(_cell, dict):
                    _kept_cells.append(_cell)
                    continue
                _old_index = _resolve_cell_column_key(_cell, _fallback_index)
                if _old_index in _empty_cols:
                    continue
                _new_index = _old_index - sum(
                    _removed_index < _old_index
                    for _removed_index in _empty_cols
                )
                _cell["colStart"] = _new_index
                _cell["colstart"] = _new_index
                _kept_cells.append(_cell)
            _row[:] = _kept_cells
    return parts


def extract_three_statements_frontend(
    items: list[Any],
    *,
    locale: Optional[str] = None,
    statement_strategy: Any = None,
) -> list[dict[str, Any]]:
    """每种细分财报标签下可输出多张 MinerU 表（如同一文档多张「利润表」）；同 expanded_idx 仍择优一条。

    excel 左右并排先拆；块内合并/单体并列再拆。资产负债表若为「资产块 + 权益/负债块」两个相邻表会先纵向拼接。
    同一基础类型下「合并X」与无前缀「X」可同时保留；顶层 ``category`` 固定为三大主表之一。
    """
    expanded_items: list[Any] = []
    for it in items:
        if isinstance(it, dict) and it.get("type") == "excel_table":
            side_split = _try_split_side_by_side_table(it)
            if side_split:
                split_tags = ("ss_l", "ss_r") if len(side_split) == 2 else tuple(
                    f"ss_{split_idx}" for split_idx in range(len(side_split))
                )
                for part, tag in zip(side_split, split_tags):
                    split_part = dict(part)
                    split_part[_ORG_TABLE_SPLIT_KEY] = tag
                    expanded_items.append(split_part)
                continue
        expanded_items.append(it)

    doc_years, doc_period = infer_document_years_and_period(expanded_items)
    bs_gap_year_by_element_id = interpolate_balance_sheet_reporting_years_by_peer_pages(
        expanded_items
    )

    label_to_candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    # 同页左右对开的独立 BS 半张：不与 label_to_candidates 抢位，但仍应单独输出。
    siblings: list[dict[str, Any]] = []
    consumed_continuation_indices: set[int] = set()
    for idx, it in enumerate(expanded_items):
        if not isinstance(it, dict) or it.get("type") not in {"table", "excel_table"}:
            continue
        if idx in consumed_continuation_indices:
            continue

        table_type_text = it.get("table_type")
        normalized = _normalize_statement_label(table_type_text)

        table_md = it.get("md") or ""
        # caption / 内嵌标题硬覆盖救回归类已移除：表格分类（``table_type``）在解析阶段
        # 已经完成（含 LLM + 规则 + OCI caption 救回 + 页码主簇过滤）。三大表流程
        # 不再用 caption 兜底，避免附注里的 "by funds" / "by segment" / "(continued)"
        # 子表被误识别为主表。
        # 注：P&L+OCI 合并表（"Statement of profit or loss and other comprehensive
        # income"）的 caption 救回已前移到 ``classify_tables_async`` 里，element
        # ["table_type"] 在此阶段已是 ``利润表及其他综合收益`` 家族标签。

        _, nearest_normalized = _find_nearest_statement_title(expanded_items, table_index=idx)

        table_md = it.get("md") or ""
        table_type_has_scope = _has_scope_prefix(table_type_text) if table_type_text else False

        # 图3修复：当 table_type 给出无口径基础标签（如"资产负债表"）但页面标题
        # 带有口径信息（如"合并资产负债表"），且基础类型相同时，优先使用带口径的标题。
        # 原因：LLM分类有时会丢失"合并"/"母公司"前缀，导致口径识别错误。
        #
        # 图2修复：当 table_type 与页面标题的基础类型不同时（如 table_type="现金流量表"
        # 但标题="母公司利润表"），通过内容验证判断哪个更可信——若表格内容符合标题类型
        # 而不符合 table_type 类型，则采用标题标签。
        table_type_has_scope = _has_scope_prefix(table_type_text) if table_type_text else False

        if normalized and nearest_normalized:
            base_n = _get_base_type(normalized)
            base_nn = _get_base_type(nearest_normalized)
            if base_n == base_nn and base_n is not None:
                if _has_scope_prefix(nearest_normalized) and not _has_scope_prefix(normalized):
                    if not table_type_has_scope:
                        normalized = nearest_normalized
            # elif base_n != base_nn:
                # content_matches_nearest = is_full_statement_table_md(table_md, nearest_normalized, locale=locale)
                # content_matches_type = is_full_statement_table_md(table_md, normalized, locale=locale)
                # if content_matches_nearest and not content_matches_type:
                    # if not table_type_has_scope:
                        # normalized = nearest_normalized
        # elif not normalized:
            # normalized = nearest_normalized

        # if not normalized:
            # normalized = detect_statement_type_by_content(table_md, locale=locale)
            # # Content detection cannot determine scope (合并/母公司); if the
            # # nearest title has the same base type AND carries scope info,
            # # adopt it — the title is explicit document text, far more
            # # reliable than guessing.
            # if normalized and nearest_normalized:
                # base_det = _get_base_type(normalized)
                # base_nn = _get_base_type(nearest_normalized)
                # if base_det == base_nn and _has_scope_prefix(nearest_normalized):
                    # normalized = nearest_normalized

        # 内嵌标题英文/中文映射已移除：表格分类（``table_type``）在解析阶段已经
        # 完成（含 LLM + 规则 + 页码主簇过滤），三大表流程不再做"内嵌标题硬覆盖"
        # 二次救回，避免把附注里的小表（如 Allianz 100+ 页 "Income Statement
        # of segment X"）误识别成主三大表。
        if not normalized:
            continue

        base_type = _get_base_type(normalized)
        split_result = _try_split_combined_statement(it, base_type, normalized) if base_type else None

        if split_result:
            for split_label, split_table_el in split_result:
                split_md = _rows_to_compact_text(split_table_el.get("rows"))
                split_is_full = is_full_statement_table_md(split_md, split_label, locale=locale)
                split_rows_count = len(split_table_el.get("rows") or [])
                split_page = _get_page_at(split_table_el)
                split_candidate = {
                    "label": split_label,
                    "table": split_table_el,
                    "rows_count": split_rows_count,
                    "is_full": split_is_full,
                    "expanded_idx": idx,
                    "page": split_page,
                }
                _push_three_statement_candidate(
                    label_to_candidates, siblings, split_label, split_candidate, locale=locale,
                    statement_strategy=statement_strategy,
                )
        else:
            is_full = is_full_statement_table_md(str(table_md or ""), normalized, locale=locale)
            rows_count = _estimate_table_rows_count(it)
            table_page = _get_page_at(it)

            candidate = {
                "label": normalized,
                "table": it,
                "rows_count": rows_count,
                "is_full": is_full,
                "expanded_idx": idx,
                "page": table_page,
            }
            candidate, continuation_indices = _merge_forward_statement_continuations(
                candidate,
                expanded_items,
                locale=locale,
                excluded_indices=consumed_continuation_indices,
            )
            consumed_continuation_indices.update(continuation_indices)

            _push_three_statement_candidate(
                label_to_candidates, siblings, normalized, candidate, locale=locale,
                statement_strategy=statement_strategy,
            )

    # 不再做「合并X」与无前缀「X」二选一剔除：三大表接口要求凡属 BS/IS/CFS 家族的表
    #（合并/母公司/无前缀、含 P&L+OCI 变体）均保留；顶层 category 已在输出阶段归一到三大主表。
    _promote_no_prefix_in_multi(label_to_candidates)

    out: list[dict[str, Any]] = []
    previous_currency_unit: Optional[str] = None
    for label in STATEMENT_ORDER:
        # 三大表接口仅输出 资产负债表 / 利润表 / 现金流量表 三类，
        # SOCE（所有者权益变动表）与三大表并列、语义不同，不在此接口产出。
        if not _is_three_main_label(label):
            continue
        for picked in label_to_candidates.get(label) or []:
            if not picked:
                continue
            entries = _build_output_entries_for_picked(
                picked,
                label=label,
                expanded_items=expanded_items,
                doc_years=doc_years,
                doc_period=doc_period,
                locale=locale,
                bs_gap_year_by_element_id=bs_gap_year_by_element_id,
                previous_currency_unit=previous_currency_unit,
            )
            out.extend(entries)
            if entries:
                previous_currency_unit = str(entries[-1].get("currencyUnit") or previous_currency_unit)

    # 同页左右对开 BS 等「独立两半」候选：不抢主列表，但仍要单独输出一遍（各自做 合并/单体 拆分）。
    for sibling in siblings:
        sibling_label = str(sibling.get("label") or "")
        if not sibling_label:
            continue
        # 同样过滤非三大表家族。
        if not _is_three_main_label(sibling_label):
            continue
        entries = _build_output_entries_for_picked(
            sibling,
            label=sibling_label,
            expanded_items=expanded_items,
            doc_years=doc_years,
            doc_period=doc_period,
            locale=locale,
            bs_gap_year_by_element_id=bs_gap_year_by_element_id,
            previous_currency_unit=previous_currency_unit,
        )
        out.extend(entries)
        if entries:
            previous_currency_unit = str(entries[-1].get("currencyUnit") or previous_currency_unit)

    return out


def _build_output_entries_for_picked(
    picked: dict[str, Any],
    *,
    label: str,
    expanded_items: list[Any],
    doc_years: Any,
    doc_period: Any,
    locale: Optional[str],
    bs_gap_year_by_element_id: Optional[dict[Any, int]] = None,
    previous_currency_unit: Optional[str] = None,
) -> list[dict[str, Any]]:
    """把单个候选（best 或 sibling）走完后续输出流水线（含 合并/单体 拆分）。"""
    table_el = picked.get("table") if isinstance(picked.get("table"), dict) else {}
    expanded_idx = picked.get("expanded_idx", -1)

    page_at = _get_page_at(table_el)
    table_bbox = _coerce_bbox(table_el.get("bBox") or table_el.get("bbox"))

    primary_rows = table_el.get("rows")
    merged = primary_rows if isinstance(primary_rows, list) else []
    continuation_indices: list[int] = list(picked.get("continuation_indices") or [])

    if "现金流量表" in label:
        merged = _truncate_cashflow_supplementary(merged)

    # 先做 headerCells 驱动的表头清洗（适用于 hasHeader=True 的表）：
    # 清洗元信息行、删除辅助列（行次/序号/Note）、拍平多行表头、重排 colstart 插入 rows[0]。
    # 清洗后 colstart 连续，_maybe_split_two_block_rows 才能正确推断列数并识别左右分界。
    cleaned_rows = _prepend_clean_statement_header_row(
        merged, table_el, fallback_page_at=page_at, locale=locale
    )
    # lineHeaderRowCount 描述的是最终传给 lines 的 rows，而不是原始 MinerU 表是否有表头。
    line_header_row_count = 1 if cleaned_rows is not None else 0
    raw_rows = cleaned_rows if cleaned_rows is not None else merged
    # 兜底：若上游 LLM 未识别到表头（headerCells 缺失），仍按 rows[0] 推断 NOTE 列并清洗。
    if cleaned_rows is None:
        note_dropped = _drop_note_column_by_first_row(raw_rows)
        if note_dropped is not None:
            raw_rows = note_dropped
    raw_rows = _merge_continuation_rows(raw_rows)
    raw_rows = _strip_residual_header_rows(raw_rows)
    raw_rows = _fix_missing_spaces_in_rows(raw_rows)
    raw_rows = _maybe_split_two_block_rows(raw_rows)
    # raw_rows = _filter_statement_table_rows(raw_rows)
    if expanded_idx >= 0:
        ctx_a, ctx_b = collect_neighbor_text_context(expanded_items, expanded_idx)
    else:
        ctx_a, ctx_b = "", ""
    currency_unit = _currency_unit_zh_for_table(
        table_el,
        context_above=ctx_a,
        previous_unit=previous_currency_unit,
    )
    merge_ids = _build_org_table_merge_ids(expanded_items, expanded_idx, continuation_indices)
    entry_id = merge_ids[0] if len(merge_ids) == 1 else uuid.uuid4().hex

    cap_j = _table_caption_joined(table_el)
    md_b = table_element_markdown(table_el)
    category_effective = _apply_scope_prefix_to_category(
        label, label, ctx_a, ctx_b, md_b, cap_j
    )
    gaap = detect_gaap_type(ctx_a, ctx_b, md_b, cap_j)
    stmt = map_to_statement_type(category_effective, category_effective)
    caliber_label = detect_caliber_label(category_effective, ctx_a, ctx_b, md_b, cap_j)
    if caliber_label == "未知" and "合并" not in category_effective and "母公司" not in category_effective:
        caliber_label = "单体"
    rows_out = _coerce_rows(raw_rows, fallback_page_at=page_at)
    col_year = infer_column_index_to_year_for_table(
        rows_out,
        context_above="\n".join(x for x in (ctx_a, md_b, cap_j) if x),
        context_below=ctx_b,
        report_name=category_effective,
    )
    doc_y_int = doc_years if isinstance(doc_years, int) else None
    year_ctx = "\n".join(x for x in (ctx_a, md_b, cap_j) if x)
    table_y_int = infer_single_table_reporting_year(
        rows_out,
        column_year_by_index=dict(col_year),
        context_above=year_ctx,
        context_below=ctx_b,
        report_name=category_effective,
        md=str(md_b or ""),
        caption=str(cap_j or ""),
        fallback_doc_year=None,
    )
    table_y_out = table_y_int
    table_period = infer_single_table_filing_period(
        rows=rows_out,
        context_above=ctx_a,
        context_below=ctx_b,
        report_name=category_effective,
        md=str(md_b or ""),
        caption=str(cap_j or ""),
        fallback_doc_period=str(doc_period or ""),
    )
    if bs_gap_year_by_element_id and stmt in ("BS", "IS", "CF"):
        eid = table_el.get("element_id")
        gap_y = bs_gap_year_by_element_id.get(eid)
        if gap_y is None and eid is not None:
            gap_y = bs_gap_year_by_element_id.get(str(eid))
        if gap_y is not None:
            table_y_out = gap_y
    if table_y_out is None:
        table_y_out = doc_y_int
    col_year = augment_column_years_for_comparative_period_labels(
        rows_out,
        col_year,
        doc_year=table_y_out,
        doc_period=table_period,
    )
    context_blob = "\n".join(x for x in (ctx_a, ctx_b, md_b, cap_j) if x)
    cleaned_header_cells_for_infer = table_el.get("cleanedHeaderCells")
    if cleaned_header_cells_for_infer:
        col_caliber = infer_column_index_to_caliber_from_header_cells(
            cleaned_header_cells_for_infer,
            table_caliber=caliber_label,
            report_name=category_effective,
        )
        col_stmt = infer_column_index_to_statement_type_from_header_cells(
            cleaned_header_cells_for_infer,
            report_name=category_effective,
            fallback_statement_type=stmt,
        )
        col_gaap = infer_column_index_to_gaap_from_header_cells(
            cleaned_header_cells_for_infer,
            report_name=category_effective,
            context_blob=context_blob,
            fallback_gaap=gaap,
        )
    else:
        col_caliber = infer_column_index_to_caliber_for_table(
            rows_out,
            table_caliber=caliber_label,
            report_name=category_effective,
        )
        col_stmt = infer_column_index_to_statement_type_for_table(
            rows_out,
            report_name=category_effective,
            fallback_statement_type=stmt,
        )
        col_gaap = infer_column_index_to_gaap_for_table(
            rows_out,
            report_name=category_effective,
            context_blob=context_blob,
            fallback_gaap=gaap,
        )
    if category_effective in _NO_PREFIX_STATEMENT_LABELS:
        if cleaned_header_cells_for_infer:
            explicit_col_caliber = infer_column_index_to_caliber_from_header_cells(
                cleaned_header_cells_for_infer,
                table_caliber="未知",
                report_name=category_effective,
            )
        else:
            explicit_col_caliber = infer_column_index_to_caliber_for_table(
                rows_out,
                table_caliber="未知",
                report_name=category_effective,
            )
        explicit_data_col_calibers = {
            v for k, v in explicit_col_caliber.items() if k > 0 and v in {"合并", "单体"}
        }
        if "合并" in explicit_data_col_calibers:
            category_effective = f"合并{category_effective}"
        elif "单体" in explicit_data_col_calibers:
            category_effective = f"母公司{category_effective}"
    data_col_calibers = {v for k, v in col_caliber.items() if k > 0 and v in {"合并", "单体", "混合"}}
    if "合并" in data_col_calibers and "单体" in data_col_calibers:
        table_caliber_effective = "混合"
    elif "合并" in data_col_calibers:
        table_caliber_effective = "合并"
    elif "单体" in data_col_calibers:
        table_caliber_effective = "单体"
    else:
        table_caliber_effective = caliber_label
    for row in rows_out:
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            col_key = _resolve_cell_column_key(cell, col_idx)
            cell_stmt = col_stmt.get(col_key) if col_key > 0 else stmt
            cell_gaap = col_gaap.get(col_key) if col_key > 0 else gaap
            cell_caliber = col_caliber.get(col_key) if col_key > 0 else table_caliber_effective
            cell_year = col_year.get(col_key) if col_key > 0 else None
            cell["statementType"] = cell_stmt or stmt
            cell["gaapType"] = cell_gaap or gaap
            cell["caliberLabel"] = cell_caliber or table_caliber_effective
            cell["accountingYear"] = cell_year

    _fill_null_accounting_year_from_document(rows_out, table_y_out)
    _strip_multiline_suffix_from_numeric_amount_cells(rows_out)

    # 新契约：顶层 ``category`` 不带口径前缀，口径独立放到 ``caliberLabel``，
    # 并归并到三大主表枚举（资产负债表 / 利润表 / 现金流量表）。
    base_category = _normalize_category_to_three_main(
        _strip_scope_prefix_from_category(category_effective)
    )
    # ``caliberLabel`` 受控为 合并 / 单体 二选一；混合 / 未知按 category_effective
    # 前缀回填以保留已识别口径信号。
    final_caliber_label = _finalize_caliber_label_for_output(
        table_caliber_effective, category_effective
    )
    _coerce_cell_caliber_to_binary(rows_out, final_caliber_label)
    built = {
        "years": table_y_out if table_y_out is not None else doc_years,
        "period": table_period,
        "id": entry_id,
        "mergeIds": merge_ids,
        "primaryElementId": _strip_org_table_derivative_suffixes(merge_ids[0])
        if merge_ids
        else None,
        "text": _translate_statement_label(base_category, locale),
        "caliberLabel": final_caliber_label,
        "category": base_category,
        "isFinancialStatementTable": True,
        "pageAt": page_at,
        "bBox": table_bbox,
        "hasHeader": table_el.get("hasHeader"),
        "lineHeaderRowCount": line_header_row_count,
        "img_path": table_el.get("img_path"),
        "currencyUnit": currency_unit,
        "rows": rows_out,
    }
    if cleaned_header_cells_for_infer:
        # 透传给下游 expand_financial_table_if_multi_period / dual_caliber，
        # 让后续推断与切片不再依赖 rows[0] 拍平表头。
        built["cleanedHeaderCells"] = cleaned_header_cells_for_infer
    stable_prefix = merge_ids[0] if len(merge_ids) == 1 else None
    out_parts: list[dict[str, Any]] = []
    for chunk in expand_financial_table_if_multi_period(
        built, locale=locale, stable_id_prefix=stable_prefix
    ):
        out_parts.extend(
            expand_financial_table_if_dual_caliber(
                chunk,
                locale=locale,
                stable_id_prefix=chunk.get("id"),
            )
        )
    for p in out_parts:
        rows = p.get("rows")
        if isinstance(rows, list):
            _expand_colspan_header_rows(rows)
        _apply_i18n_to_statement_table_response(p, locale)
    return out_parts


async def extract_three_statements_frontend_by_file_id(
    file_id: str, *, locale: Optional[str] = None
) -> list[dict[str, Any]]:
    items = await load_mineru_items(file_id)
    if not items:
        raise FileNotFoundError(f"目标 items 不存在: {file_id}")
    from src.applications.parsing.service.three_statements.pipeline import (
        extract_three_statements_with_strategy,
    )

    result = extract_three_statements_with_strategy(items, locale=locale)
    await _append_lines_context_to_tables(result, file_id=file_id, locale=locale, source_items=items)
    return result


async def _append_lines_context_to_tables(
    tables: list[dict[str, Any]],
    *,
    file_id: str,
    locale: Optional[str] = None,
    financial_only: bool = False,
    source_items: Optional[list[Any]] = None,
) -> None:
    source_by_id = _line_context_source_by_id(source_items or [])
    should_build_lines = isinstance(locale, str) and locale.strip().lower() == "en_us"
    pending_line_tables: list[tuple[dict[str, Any], Any]] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        if financial_only and not table.get("isFinancialStatementTable"):
            continue
        rows = table.get("rows")
        if not isinstance(rows, list):
            continue
        if not should_build_lines:
            table["lines"] = []
            continue
        line_table_el = _line_context_table_el(table, source_by_id)
        pending_line_tables.append((table, build_lines_context(rows, line_table_el, file_id=file_id)))

    if not pending_line_tables:
        return

    lines_results = await asyncio.gather(*(pending for _, pending in pending_line_tables))
    for (table, _), lines in zip(pending_line_tables, lines_results):
        table["lines"] = lines


def _line_context_source_by_id(items: list[Any]) -> dict[str, dict[str, Any]]:
    source_by_id: dict[str, dict[str, Any]] = {}
    for _, table_el in _iter_org_tables_with_source_item_index(items):
        if not isinstance(table_el, dict):
            continue
        table_id = _org_tables_element_id(table_el)
        source_by_id[table_id] = table_el
        source_by_id.setdefault(_strip_org_table_derivative_suffixes(table_id), table_el)
    return source_by_id


def _line_context_table_el(
    table: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = _line_context_source_table(table, source_by_id)
    return {
        "hasHeader": table.get("hasHeader") if table.get("hasHeader") is not None else source.get("hasHeader"),
        # 下游只依据 lineHeaderRowCount 判 HEADER；不再用 hasHeader 猜当前 rows[0]。
        "lineHeaderRowCount": table.get("lineHeaderRowCount") or 0,
        "pageAt": table.get("pageAt"),
        "bBox": table.get("bBox"),
        "bbox": table.get("bbox"),
        "img_path": table.get("img_path") or source.get("img_path"),
        "table_type": source.get("table_type") or table.get("category") or table.get("text"),
        "text": table.get("text"),
    }


def _line_context_source_table(
    table: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_ids: list[Any] = []
    merge_ids = table.get("mergeIds")
    if isinstance(merge_ids, list):
        candidate_ids.extend(merge_ids)
    candidate_ids.extend([table.get("primaryElementId"), table.get("id")])

    for candidate_id in candidate_ids:
        if candidate_id is None:
            continue
        candidate = str(candidate_id)
        source = source_by_id.get(candidate) or source_by_id.get(
            _strip_org_table_derivative_suffixes(candidate)
        )
        if source is not None:
            return source
    return {}


# ---------------------------------------------------------------------------
# Org-tables: return all individual table elements with their element ids
# ---------------------------------------------------------------------------

def extract_org_tables(items: list[Any], *, locale: Optional[str] = None) -> list[dict[str, Any]]:
    """把 MinerU ``items`` 里解析出的每张逻辑表打成扁平列表。

    - 条数随「原解析表」数量变化：MinerU 有几个 ``table``/``excel_table`` 块（excel 左右并排会先拆成两张），
      这里就处理几条；再在同一张表内若并列合并/单体金额列则拆成 ``_grp`` / ``_std`` 两条。
    - 与 ``extract_three_statements_frontend`` 一致，先对 ``excel_table`` 做左右并排检测拆分。
    """
    out: list[dict[str, Any]] = []
    previous_currency_unit: Optional[str] = None
    doc_years, doc_period = infer_document_years_and_period(items)
    bs_gap_year_by_element_id = interpolate_balance_sheet_reporting_years_by_peer_pages(items)
    for original_idx, it in _iter_org_tables_with_source_item_index(items):
        table_id = _org_tables_element_id(it)
        work = dict(it)
        work.pop(_ORG_TABLE_SPLIT_KEY, None)

        page_at = _get_page_at(work)
        table_bbox = _coerce_bbox(work.get("bBox") or work.get("bbox"))
        ctx_a, ctx_b = collect_neighbor_text_context(items, original_idx)
        text = work.get("table_type") or ""
        rows = work.get("rows")

        normalized = _normalize_statement_label(text) if isinstance(text, str) else None
        _, nearest_normalized = _find_nearest_statement_title(items, table_index=original_idx)
        if normalized and nearest_normalized:
            base_n = _get_base_type(normalized)
            base_nn = _get_base_type(nearest_normalized)
            if (
                base_n == base_nn
                and base_n is not None
                and _has_scope_prefix(nearest_normalized)
                and not _has_scope_prefix(normalized)
            ):
                normalized = nearest_normalized
        # elif not normalized and nearest_normalized:
            # normalized = nearest_normalized

        category = normalized or (str(text).strip() if str(text).strip() else "其他表")
        stmt = map_to_statement_type(str(text or ""), normalized or category)
        is_fs = bool(stmt and category != "其他表")

        currency_unit = _currency_unit_zh_for_table(
            work,
            context_above=ctx_a,
            previous_unit=previous_currency_unit,
        )
        md_b = table_element_markdown(work)
        cap_j = _table_caption_joined(work)
        context_blob = "\n".join(x for x in (ctx_a, ctx_b, md_b, cap_j) if x)
        gaap = detect_gaap_type(ctx_a, ctx_b, md_b, cap_j)
        caliber_label = detect_caliber_label(category, ctx_a, ctx_b, md_b, cap_j)
        if caliber_label == "未知" and "合并" not in category and "母公司" not in category:
            caliber_label = "单体"
        normalized_rows = rows
        line_header_row_count = 0
        if is_fs:
            # org-tables 也对财报表应用与 three-statements 一致的结构清洗，
            # 避免左右镜像（项目/行次/金额 x2）导致“重复列”输出。
            cleaned_rows = _prepend_clean_statement_header_row(
                normalized_rows, work, fallback_page_at=page_at, locale=locale
            )
            line_header_row_count = 1 if cleaned_rows is not None else 0
            if cleaned_rows is not None:
                normalized_rows = cleaned_rows
            normalized_rows = _maybe_split_two_block_rows(normalized_rows)
            # normalized_rows = _filter_statement_table_rows(normalized_rows)
        rows_out = _coerce_rows(normalized_rows, fallback_page_at=page_at)
        col_year = infer_column_index_to_year_for_table(
            rows_out,
            context_above="\n".join(x for x in (ctx_a, md_b, cap_j) if x),
            context_below=ctx_b,
            report_name=category,
        )
        doc_y_int = doc_years if isinstance(doc_years, int) else None
        year_ctx = "\n".join(x for x in (ctx_a, md_b, cap_j) if x)
        table_y_int = (
            infer_single_table_reporting_year(
                rows_out,
                column_year_by_index=dict(col_year),
                context_above=year_ctx,
                context_below=ctx_b,
                report_name=category,
                md=str(md_b or ""),
                caption=str(cap_j or ""),
                fallback_doc_year=None,
            )
            if is_fs
            else None
        )
        table_y_out = table_y_int
        if bs_gap_year_by_element_id and stmt in ("BS", "IS", "CF"):
            eid = work.get("element_id")
            gy = bs_gap_year_by_element_id.get(eid)
            if gy is None and eid is not None:
                gy = bs_gap_year_by_element_id.get(str(eid))
            if gy is not None:
                table_y_out = gy
        if table_y_out is None:
            table_y_out = doc_y_int
        table_period = infer_single_table_filing_period(
            rows=rows_out,
            context_above=ctx_a,
            context_below=ctx_b,
            report_name=category,
            md=str(md_b or ""),
            caption=str(cap_j or ""),
            fallback_doc_period=str(doc_period or ""),
        )
        col_year = augment_column_years_for_comparative_period_labels(
            rows_out,
            col_year,
            doc_year=table_y_out,
            doc_period=table_period,
        )
        cleaned_header_cells_for_infer = work.get("cleanedHeaderCells")
        if cleaned_header_cells_for_infer:
            col_caliber = infer_column_index_to_caliber_from_header_cells(
                cleaned_header_cells_for_infer,
                table_caliber=caliber_label,
                report_name=category,
            )
            col_stmt = infer_column_index_to_statement_type_from_header_cells(
                cleaned_header_cells_for_infer,
                report_name=category,
                fallback_statement_type=stmt,
            )
            col_gaap = infer_column_index_to_gaap_from_header_cells(
                cleaned_header_cells_for_infer,
                report_name=category,
                context_blob=context_blob,
                fallback_gaap=gaap,
            )
        else:
            col_caliber = infer_column_index_to_caliber_for_table(
                rows_out,
                table_caliber=caliber_label,
                report_name=category,
            )
            col_stmt = infer_column_index_to_statement_type_for_table(
                rows_out,
                report_name=category,
                fallback_statement_type=stmt,
            )
            col_gaap = infer_column_index_to_gaap_for_table(
                rows_out,
                report_name=category,
                context_blob=context_blob,
                fallback_gaap=gaap,
            )
        # 与 three-statements 保持一致：优先使用标题/上下文里的显式口径证据补前缀。
        category = _apply_scope_prefix_to_category(
            category, str(text or ""), ctx_a, ctx_b, md_b, cap_j, nearest_normalized or ""
        )

        # 仅在“显式列证据”存在时按列再提升到带口径标签（避免兜底口径污染 category）。
        # 注意：col_caliber 会受 table_caliber 兜底影响，不能直接用于 category 提升判断。
        if cleaned_header_cells_for_infer:
            explicit_col_caliber = infer_column_index_to_caliber_from_header_cells(
                cleaned_header_cells_for_infer,
                table_caliber="未知",
                report_name=category,
            )
        else:
            explicit_col_caliber = infer_column_index_to_caliber_for_table(
                rows_out,
                table_caliber="未知",
                report_name=category,
            )
        if category in _NO_PREFIX_STATEMENT_LABELS:
            data_col_calibers_for_category = {
                v for k, v in explicit_col_caliber.items() if k > 0 and v in {"合并", "单体"}
            }
            if "合并" in data_col_calibers_for_category:
                category = f"合并{category}"
            elif "单体" in data_col_calibers_for_category:
                category = f"母公司{category}"
        # 注意：``category`` 仍保留带前缀形态供后续 ``table_caliber_effective`` 等内部判断；
        # 真正写入输出的 ``text`` / ``category`` 在末尾用 ``_strip_scope_prefix_from_category``
        # 去前缀后写入（见 ``base_category`` / ``base_display_text``）。
        data_col_calibers = {v for k, v in col_caliber.items() if k > 0 and v in {"合并", "单体", "混合"}}
        if "合并" in data_col_calibers and "单体" in data_col_calibers:
            table_caliber_effective = "混合"
        elif "合并" in data_col_calibers:
            table_caliber_effective = "合并"
        elif "单体" in data_col_calibers:
            table_caliber_effective = "单体"
        else:
            table_caliber_effective = caliber_label
        for row in rows_out:
            for col_idx, cell in enumerate(row):
                if not isinstance(cell, dict):
                    continue
                col_key = _resolve_cell_column_key(cell, col_idx)
                cell_stmt = col_stmt.get(col_key) if col_key > 0 else stmt
                cell_gaap = col_gaap.get(col_key) if col_key > 0 else gaap
                cell_caliber = col_caliber.get(col_key) if col_key > 0 else table_caliber_effective
                cell_year = col_year.get(col_key) if col_key > 0 else None
                cell["statementType"] = cell_stmt or stmt
                cell["gaapType"] = cell_gaap or gaap
                cell["caliberLabel"] = cell_caliber or table_caliber_effective
                cell["accountingYear"] = cell_year

        _fill_null_accounting_year_from_document(rows_out, table_y_out)
        _strip_multiline_suffix_from_numeric_amount_cells(rows_out)

        merge_ids_source = _source_merge_ids_for_table_element(work)
        # org-tables 契约：``category`` 保留 ``STATEMENT_ORDER`` 全 15 个受控值
        # （含「合并X」/「母公司X」/无前缀「X」），非财报场景（is_fs=False）保持
        # 原 ``category`` 不动（``其他表`` 或 LLM 给出的 table_type）。
        output_category = category
        output_display_text = (
            _translate_statement_label(output_category, locale)
            if is_fs and output_category in STATEMENT_ORDER
            else str(text or output_category)
        )
        # caliberLabel 受控为 合并 / 单体 二选一；按 ``category`` 前缀回填非二值。
        final_caliber_label = _finalize_caliber_label_for_output(
            table_caliber_effective, output_category
        )
        _coerce_cell_caliber_to_binary(rows_out, final_caliber_label)
        entry = {
            "years": table_y_out if table_y_out is not None else doc_years,
            "period": table_period,
            "id": table_id,
            "mergeIds": merge_ids_source,
            "primaryElementId": merge_ids_source[0],
            "text": output_display_text,
            "caliberLabel": final_caliber_label,
            "category": output_category,
            "isFinancialStatementTable": is_fs,
            "pageAt": page_at,
            "bBox": table_bbox,
            "hasHeader": work.get("hasHeader"),
            "lineHeaderRowCount": line_header_row_count,
            "img_path": work.get("img_path"),
            "currencyUnit": currency_unit,
            "rows": rows_out,
        }
        if cleaned_header_cells_for_infer:
            # 透传给下游 expand_financial_table_if_multi_period / dual_caliber，
            # 让后续推断与切片不再依赖 rows[0] 拍平表头。
            entry["cleanedHeaderCells"] = cleaned_header_cells_for_infer
        expanded: list[dict[str, Any]] = []
        for chunk in expand_financial_table_if_multi_period(
            entry, locale=locale, stable_id_prefix=table_id
        ):
            expanded.extend(
                expand_financial_table_if_dual_caliber(
                    chunk,
                    locale=locale,
                    stable_id_prefix=chunk.get("id"),
                    allow_split=True,
                    use_scope_prefix_category=True,
                )
            )
        for p in expanded:
            _apply_i18n_to_statement_table_response(p, locale)
        out.extend(expanded)
        if expanded:
            previous_currency_unit = str(expanded[-1].get("currencyUnit") or previous_currency_unit)
    return out


async def extract_org_tables_by_file_id(
    file_id: str, *, locale: Optional[str] = None
) -> list[dict[str, Any]]:
    items = await load_mineru_items(file_id)
    if not items:
        raise FileNotFoundError(f"目标 items 不存在: {file_id}")
    result = extract_org_tables(items, locale=locale)
    await _append_lines_context_to_tables(result, file_id=file_id, locale=locale, financial_only=True, source_items=items)
    return result


# ---------------------------------------------------------------------------
# Manual merge: combine caller-specified table elements into merged tables
# ---------------------------------------------------------------------------

async def manual_merge_tables(
    file_id: str,
    merges: list[dict[str, Any]],
    *,
    locale: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Merge table elements as instructed by the caller.

    *merges* is a list of dicts, each with:
      - ``sourceIds``  – ordered list of original element id strings
      - ``text``       – (optional) label for the merged table

    Returns a list in the same format as the three-statements endpoint.
    """
    items = await load_mineru_items(file_id)
    if not items:
        raise FileNotFoundError(f"目标 items 不存在: {file_id}")

    id_to_item: dict[str, dict[str, Any]] = {}
    id_to_index: dict[str, int] = {}
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if it.get("type") not in {"table", "excel_table"}:
            continue
        eid = _get_element_id_str(it)
        id_to_item[eid] = it
        id_to_index[eid] = idx

    out: list[dict[str, Any]] = []
    previous_currency_unit: Optional[str] = None
    for merge_spec in merges:
        source_ids: list[str] = [str(s) for s in merge_spec.get("sourceIds") or []]
        label: str = merge_spec.get("text") or ""

        if not source_ids:
            continue

        combined_rows: list[Any] = []
        first_el: Optional[dict[str, Any]] = None
        first_sid: Optional[str] = None
        for sid in source_ids:
            el = id_to_item.get(sid)
            if el is None:
                _logger.warning("manual_merge_tables: source id %r not found", sid)
                continue
            if first_el is None:
                first_el = el
                first_sid = sid
            el_rows = el.get("rows")
            if isinstance(el_rows, list):
                combined_rows.extend(el_rows)

        if not first_el:
            continue

        combined_rows = _deduplicate_rows(combined_rows)

        first_idx = id_to_index.get(first_sid, -1) if first_sid else -1
        if first_idx >= 0:
            ctx_a, ctx_b = collect_neighbor_text_context(items, first_idx)
        else:
            ctx_a, ctx_b = "", ""

        if not label:
            md_text = str(first_el.get("table_type") or first_el.get("md") or "")
            normalized = detect_statement_type_by_content(md_text, locale=locale) or ""
            label = _translate_statement_label(normalized, locale) if normalized else ""

        page_at = _get_page_at(first_el)
        table_bbox = _coerce_bbox(first_el.get("bBox") or first_el.get("bbox"))
        currency_unit = _currency_unit_for_api(
            first_el,
            locale,
            context_above=ctx_a,
            previous_unit=previous_currency_unit,
        )

        raw_rows = _filter_statement_table_rows(combined_rows)

        out.append(
            {
                "id": uuid.uuid4().hex,
                "mergeIds": source_ids,
                "text": label,
                "pageAt": page_at,
                "bBox": table_bbox,
                "currencyUnit": currency_unit,
                "rows": _coerce_rows(raw_rows, fallback_page_at=page_at),
            }
        )
        previous_currency_unit = currency_unit

    return out
