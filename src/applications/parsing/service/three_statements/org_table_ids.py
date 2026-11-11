from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Iterable

GetElementId = Callable[[Any], str]
TrySplitSideBySide = Callable[[dict[str, Any]], Any]

ORG_TABLE_SPLIT_KEY = "_org_table_split"
ORG_TABLE_ID_SUFFIXES: tuple[str, ...] = ("_ss_l", "_ss_r", "_grp", "_std")


def get_element_id_str(item: Any) -> str:
    if isinstance(item, dict):
        element_id = item.get("element_id")
        if element_id is not None:
            return str(element_id)
    return uuid.uuid4().hex


def source_merge_ids_for_table_element(table_el: dict[str, Any]) -> list[str]:
    element_id = table_el.get("element_id")
    if element_id is not None:
        return [str(element_id)]
    return [get_element_id_str(table_el)]


def strip_org_table_derivative_suffixes(org_id: str) -> str:
    value = str(org_id)
    changed = True
    while changed:
        changed = False
        for suffix in ORG_TABLE_ID_SUFFIXES:
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                changed = True
                break
        if changed:
            continue
        match = re.match(r"^(.*)_mp\d+$", value)
        if match:
            value = match.group(1)
            changed = True
    return value


def org_tables_element_id(table_el: dict[str, Any]) -> str:
    base = get_element_id_str(table_el)
    tag = table_el.get(ORG_TABLE_SPLIT_KEY)
    if isinstance(tag, str) and tag:
        return f"{base}_{tag}"
    return base


def iter_org_tables_with_source_item_index(
    items: list[Any],
    *,
    try_split_side_by_side_table: TrySplitSideBySide,
) -> Iterable[tuple[int, dict[str, Any]]]:
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "excel_table":
            side_split = try_split_side_by_side_table(item)
            if side_split:
                if len(side_split) == 2:
                    left, right = side_split
                    for part, tag in ((left, "ss_l"), (right, "ss_r")):
                        element = dict(part)
                        element[ORG_TABLE_SPLIT_KEY] = tag
                        yield index, element
                    continue
                for split_idx, part in enumerate(side_split):
                    element = dict(part)
                    element[ORG_TABLE_SPLIT_KEY] = f"mp{split_idx}"
                    yield index, element
                continue
        if item.get("type") in {"table", "excel_table"}:
            yield index, item


def build_org_table_merge_ids(
    items: list[Any], primary_idx: int, continuation_indices: list[int]
) -> list[str]:
    all_indices = sorted({primary_idx} | set(continuation_indices))
    out: list[str] = []
    for index in all_indices:
        if not (0 <= index < len(items)):
            continue
        element = items[index]
        if isinstance(element, dict):
            out.append(org_tables_element_id(element))
        else:
            out.append(str(index))
    return out
