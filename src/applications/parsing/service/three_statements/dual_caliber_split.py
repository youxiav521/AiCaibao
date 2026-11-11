from __future__ import annotations

import copy
from typing import Any, Optional

from src.applications.parsing.service.three_statements.cells import resolve_cell_column_key
from src.applications.parsing.service.three_statements.labels import get_base_type


def all_positive_logical_column_keys_in_rows(rows: list[list[Any]]) -> set[int]:
    keys: set[int] = set()
    for row in rows:
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            column_key = resolve_cell_column_key(cell, col_idx)
            if column_key > 0:
                keys.add(column_key)
    return keys


def vote_logical_column_caliber_labels(rows: list[list[Any]]) -> dict[int, str]:
    tallies: dict[int, dict[str, int]] = {}
    for row in rows:
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            column_key = resolve_cell_column_key(cell, col_idx)
            if column_key <= 0:
                continue
            caliber_label = cell.get("caliberLabel")
            if not isinstance(caliber_label, str) or not caliber_label:
                continue
            if caliber_label not in ("合并", "单体", "混合", "未知"):
                continue
            tallies.setdefault(column_key, {})
            tallies[column_key][caliber_label] = tallies[column_key].get(caliber_label, 0) + 1
    priority = {"合并": 3, "单体": 3, "混合": 2, "未知": 1}
    out: dict[int, str] = {}
    for column_key, counter in tallies.items():
        best = max(counter.items(), key=lambda item: (item[1], priority.get(item[0], 0)))
        out[column_key] = best[0]
    return out


def logical_column_caliber_map_complete(rows: list[list[Any]]) -> dict[int, str]:
    all_keys = all_positive_logical_column_keys_in_rows(rows)
    voted = vote_logical_column_caliber_labels(rows)
    out = dict(voted)
    for column_key in all_keys:
        out.setdefault(column_key, "未知")
    return out


def should_split_table_by_merge_vs_standalone_columns(
    logical_map: dict[int, str],
) -> bool:
    labels = {logical_map[key] for key in logical_map if key > 0}
    return "合并" in labels and "单体" in labels


def allowed_data_keys_for_merge_standalone_scope(logical_map: dict[int, str], target: str) -> set[int]:
    opposite = "单体" if target == "合并" else "合并"
    allowed: set[int] = set()
    for column_key, label in logical_map.items():
        if column_key <= 0:
            continue
        if label == opposite:
            continue
        allowed.add(column_key)
    return allowed


def slice_table_rows_by_logical_data_keys(
    rows: list[list[Any]],
    allowed_data_keys: set[int],
    *,
    first_scope_data_logical_col: Optional[int] = None,
) -> list[list[Any]]:
    boundary = first_scope_data_logical_col
    if boundary is None and allowed_data_keys:
        boundary = min(allowed_data_keys)

    subject_keys: set[int] = set()
    for row in rows:
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            column_key = resolve_cell_column_key(cell, col_idx)
            is_subject = (
                column_key < boundary
                if boundary is not None
                else column_key <= 0
            )
            if is_subject:
                subject_keys.add(column_key)

    ordered_keys = sorted(subject_keys) + sorted(allowed_data_keys)
    new_index_by_key = {
        column_key: new_index
        for new_index, column_key in enumerate(ordered_keys)
    }

    out: list[list[Any]] = []
    for row in rows:
        subject_ordered: list[tuple[int, int, Any]] = []
        data_by_key: dict[int, Any] = {}
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            column_key = resolve_cell_column_key(cell, col_idx)
            if boundary is not None:
                if column_key < boundary:
                    subject_ordered.append(
                        (column_key, col_idx, copy.deepcopy(cell))
                    )
                elif column_key in allowed_data_keys:
                    data_by_key[column_key] = copy.deepcopy(cell)
            elif column_key <= 0:
                subject_ordered.append(
                    (column_key, col_idx, copy.deepcopy(cell))
                )
            elif column_key in allowed_data_keys:
                data_by_key[column_key] = copy.deepcopy(cell)
        subject_ordered.sort(key=lambda item: item[1])
        selected_cells = [
            (column_key, cell)
            for column_key, _, cell in subject_ordered
        ] + [
            (column_key, data_by_key[column_key])
            for column_key in sorted(data_by_key)
        ]
        new_row: list[Any] = []
        for column_key, cell in selected_cells:
            cell["colStart"] = new_index_by_key[column_key]
            cell["colstart"] = new_index_by_key[column_key]
            new_row.append(cell)
        out.append(new_row)
    return out


def rewrite_caliber_labels_after_scope_split(rows: list[list[Any]], target_caliber: str) -> None:
    for row in rows:
        for cell in row:
            if not isinstance(cell, dict):
                continue
            cell["caliberLabel"] = target_caliber


def scoped_category_for_split(category: str, target_caliber: str) -> str:
    base = get_base_type(category)
    if not base:
        return category
    if target_caliber == "合并":
        return f"合并{base}"
    return f"母公司{base}"
