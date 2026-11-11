from __future__ import annotations

from typing import Any, Sequence

from loguru import logger

from src.agents.content_generate_agent.hierarchical_summary.tree_builder import Node

from .parent_summary_service import ParentSummaryRecord, load_usable_parent_summaries_by_file
from .summary_service_utils import (
    coerce_int,
    extract_origin_content_file_ids,
    get_field_value,
    group_origin_content_by_file_id,
    normalize_nonempty_string,
)


def _split_heading_path(heading_path: str) -> tuple[str, ...]:
    """拆分标题路径字符串。

    Args:
        heading_path: 以 `/` 分隔的标题路径。

    Returns:
        tuple[str, ...]: 去除空白和空片段后的标题层级元组。
    """
    if not heading_path:
        return ()
    return tuple(part.strip() for part in str(heading_path).split("/") if part.strip())


def _build_file_item_lookup(file_items: Sequence[Any]) -> dict[int, list[tuple[int, Any]]]:
    """为单个文件构建 `element_id -> 内容项` 查找表。

    Args:
        file_items: 单个文件下的原始内容项列表。

    Returns:
        dict[int, list[tuple[int, Any]]]: 以 `element_id` 为键的查找表，
        值中保留原始位置，便于后续恢复顺序。
    """
    item_lookup: dict[int, list[tuple[int, Any]]] = {}
    for position, item in enumerate(file_items):
        element_id = coerce_int(get_field_value(item, "element_id"))
        if element_id is None:
            continue
        item_lookup.setdefault(element_id, []).append((position, item))
    return item_lookup


def _resolve_record_items(
    record: ParentSummaryRecord,
    item_lookup: dict[int, list[tuple[int, Any]]],
) -> list[Any]:
    """将父切片摘要记录映射回原始内容项。

    Args:
        record: 单条父切片摘要记录。
        item_lookup: 当前文件的 `element_id` 查找表。

    Returns:
        list[Any]: 与记录来源元素对应的原始内容项列表。

    Raises:
        RuntimeError: 当记录无法映射回任何原始内容项时抛出。
    """
    matched_entries: list[tuple[int, Any]] = []
    for element_id in record.source_element_ids:
        matched_entries.extend(item_lookup.get(element_id, ()))
    if not matched_entries:
        raise RuntimeError(
            f"父切片记录无法映射到 origin_content: file_id={record.file_id}, "
            f"parent_id={record.parent_id}, source_element_ids={record.source_element_ids}"
        )
    matched_entries.sort(key=lambda entry: entry[0])
    return [item for _, item in matched_entries]


def build_summary_tree_from_parent_records(
    records: Sequence[ParentSummaryRecord],
    origin_content: Sequence[Any],
) -> Node:
    """基于父切片摘要记录装配摘要树。

    Args:
        records: 已加载完成的父切片摘要记录列表。
        origin_content: 原始内容项序列，用于回填叶子节点内容。

    Returns:
        Node: 供后续摘要检索和内容生成使用的根节点。

    Raises:
        RuntimeError: 当原始内容缺少文件 ID 或记录无法完整装配时抛出。
    """
    ordered_file_ids, file_to_items = group_origin_content_by_file_id(origin_content)
    if not ordered_file_ids:
        raise RuntimeError("origin_content 中未提取到可用的 file_id，无法装配摘要树")

    records_by_file_id: dict[str, list[ParentSummaryRecord]] = {file_id: [] for file_id in ordered_file_ids}
    for record in records:
        if record.file_id in records_by_file_id:
            records_by_file_id[record.file_id].append(record)

    global_root = Node()
    for file_id in ordered_file_ids:
        file_records = sorted(records_by_file_id.get(file_id) or [], key=lambda record: record.sort_key_within_file)
        if not file_records:
            raise RuntimeError(f"file_id={file_id} 缺少可用父切片记录，无法装配摘要树")

        file_root = Node()
        setattr(file_root, "file_id", file_id)
        global_root.childs.append(file_root)

        item_lookup = _build_file_item_lookup(file_to_items.get(file_id) or [])
        path_to_node: dict[tuple[str, ...], Node] = {(): file_root}

        for record in file_records:
            parent_node = file_root
            current_path: tuple[str, ...] = ()
            for heading_part in _split_heading_path(record.heading_path):
                current_path = (*current_path, heading_part)
                heading_node = path_to_node.get(current_path)
                if heading_node is None:
                    heading_node = Node()
                    setattr(heading_node, "heading_path", "/".join(current_path))
                    parent_node.childs.append(heading_node)
                    path_to_node[current_path] = heading_node
                parent_node = heading_node

            leaf_items = _resolve_record_items(record, item_lookup)
            leaf_node = Node(content=list(leaf_items), summary=record.summary)
            setattr(leaf_node, "parent_id", record.parent_id)
            setattr(leaf_node, "file_id", record.file_id)
            setattr(leaf_node, "heading_path", record.heading_path)
            setattr(leaf_node, "heading_node_ids", list(record.heading_node_ids))
            setattr(leaf_node, "source_element_ids", list(record.source_element_ids))
            setattr(leaf_node, "metadata", dict(record.metadata))
            parent_node.childs.append(leaf_node)

    return global_root


async def load_summary_tree_if_available(
    origin_content: Sequence[Any],
) -> Node | None:
    """Load a summary tree only from files with fully usable parent summaries.

    Files whose summary data is missing, blank, or otherwise unusable are
    silently skipped.  When *no* file has usable summary data the function
    returns ``None`` so the caller can continue without summary retrieval.
    """
    file_ids = extract_origin_content_file_ids(origin_content)
    if not file_ids:
        return None

    usable_records_by_file = await load_usable_parent_summaries_by_file(file_ids)
    if not usable_records_by_file:
        return None

    usable_file_ids = set(usable_records_by_file)
    filtered_origin_content = [
        item
        for item in origin_content
        if normalize_nonempty_string(get_field_value(item, "file_id")) in usable_file_ids
    ]

    ordered_records: list[ParentSummaryRecord] = []
    for file_id in file_ids:
        ordered_records.extend(usable_records_by_file.get(file_id, []))

    return build_summary_tree_from_parent_records(
        ordered_records,
        filtered_origin_content,
    )
