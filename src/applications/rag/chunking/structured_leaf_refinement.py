from __future__ import annotations

"""结构化叶子块细化工具。

基于标题层级、表格按行拆分和顺序回退策略，将过大的叶子块递归细化为更适合
RAG 索引的片段。
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.agents.content_generate_agent.input_models import (
    ExcelTableItem,
    ImageItem,
    Item,
    TableItem,
    TextItem,
    extract_text_yusys_ai,
)

from ..config.rag_settings import RAG_SETTINGS
from .table_rows import split_table_by_rows
from loguru import logger


SUMMARY_FALLBACK_TARGET_CHARS = 2000


def _normalize_text_lines(values: Sequence[Any] | Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        normalized = [str(value or "").strip() for value in values]
        return [value for value in normalized if value]
    text = str(values or "").strip()
    return [text] if text else []


def _build_image_index_text(item: ImageItem) -> str:
    captions = _normalize_text_lines(item.image_caption)
    footnotes = _normalize_text_lines(item.image_footnote)
    image_name = Path(str(item.img_path or "")).name if str(item.img_path or "").strip() else ""

    parts: List[str] = []
    if captions:
        parts.append("图片标题: " + "；".join(captions))
    if footnotes:
        parts.append("图片注释: " + "；".join(footnotes))

    location_parts = [f"页码: {item.page}", f"元素ID: {item.element_id}"]
    if image_name:
        location_parts.append(f"图片文件: {image_name}")
    parts.append("图片定位: " + "，".join(location_parts))
    return "\n".join(parts).strip()


def extract_indexable_text(item: Item) -> Optional[str]:
    text = extract_text_yusys_ai(item)
    if text:
        return text
    if isinstance(item, ImageItem):
        image_text = _build_image_index_text(item)
        return image_text or None
    return None


def infer_leaf_type(items: Sequence[Item]) -> Optional[str]:
    normalized_types = {
        str(getattr(item, "type", "") or "").strip()
        for item in items or []
        if str(getattr(item, "type", "") or "").strip()
    }
    if len(normalized_types) == 1:
        return next(iter(normalized_types))
    return None


@dataclass
class RefinedLeafInfo:
    """描述细化后的叶子块信息。

    Attributes:
        items: 当前叶子块关联的原始 Item 列表。
        heading_path: 当前叶子块对应的标题路径文本。
        heading_node_ids: 标题路径上各节点的元素 ID。
        floor_level: 当前细化层的基准标题层级。
        origin_leaf_id: 原始叶子块 ID，用于追踪来源。
        summary: 预留的摘要内容。
        text_override: 直接覆盖叶子块文本的内容。
        split_strategy: 当前叶子块的切分策略标记。
        table_row_start: 表格按行切分时的起始行号。
        table_row_end: 表格按行切分时的结束行号。
        type_override: 覆盖后的叶子块类型。
    """

    items: List[Item]
    heading_path: List[str]
    heading_node_ids: List[int]
    floor_level: int
    origin_leaf_id: str
    summary: Optional[str] = None
    text_override: Optional[str] = None
    split_strategy: Optional[str] = None
    table_row_start: Optional[int] = None
    table_row_end: Optional[int] = None
    type_override: Optional[str] = None


@dataclass
class _LocalNode:
    """局部标题树中的内部节点。

    Attributes:
        level: 当前节点对应的标题层级。
        heading_item: 当前节点自身的标题 Item。
        heading_path: 从根到当前节点的标题路径。
        heading_node_ids: 标题路径对应的元素 ID 列表。
        own_items: 直接挂载在当前节点下的内容项。
        children: 当前节点的子标题节点。
    """

    level: int
    heading_item: Optional[Item]
    heading_path: List[str]
    heading_node_ids: List[int]
    own_items: List[Item] = field(default_factory=list)
    children: List["_LocalNode"] = field(default_factory=list)


def extract_items_text(items: Iterable[Item]) -> str:
    """提取一组 Item 的文本并按段落拼接。

    Args:
        items: 待提取文本的内容项序列。

    Returns:
        str: 使用空行连接后的纯文本内容。
    """

    parts: List[str] = []
    for item in items:
        text = extract_indexable_text(item)
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def extract_refined_leaf_text(leaf: RefinedLeafInfo) -> str:
    """获取细化叶子块的最终文本。

    优先使用 `text_override`，否则回退到原始 `items` 的文本拼接结果。

    Args:
        leaf: 待读取文本的叶子块。

    Returns:
        str: 当前叶子块的可索引文本。
    """

    if leaf.text_override is not None:
        return str(leaf.text_override).strip()
    return extract_items_text(leaf.items)


def is_table_like_item(item: Item) -> bool:
    """判断给定 Item 是否属于表格类型。

    Args:
        item: 待判断的内容项。

    Returns:
        bool: 当内容项为普通表格或 Excel 表格时返回 `True`。
    """

    return isinstance(item, (TableItem, ExcelTableItem))


def get_heading_level(item: Item) -> Optional[int]:
    """提取内容项的标题层级。

    Args:
        item: 待读取层级的内容项。

    Returns:
        Optional[int]: 标题层级；如果内容项不是标题或层级不可解析则返回 `None`。
    """

    if isinstance(item, TextItem) and item.headingLevel is not None:
        try:
            return int(item.headingLevel)
        except Exception:
            return None
    return None


def is_heading_item(item: Item) -> bool:
    """判断内容项是否为标题节点。

    Args:
        item: 待判断的内容项。

    Returns:
        bool: 能解析出标题层级时返回 `True`。
    """

    return get_heading_level(item) is not None


def get_heading_text(item: Item) -> str:
    """提取标题节点的显示文本。

    Args:
        item: 标题内容项。

    Returns:
        str: 去除首尾空白后的标题文本。
    """

    return str(extract_text_yusys_ai(item) or "").strip()


def get_item_element_id(item: Item) -> Optional[int]:
    """读取内容项的元素 ID。

    Args:
        item: 待读取 ID 的内容项。

    Returns:
        Optional[int]: 有效的整数类型元素 ID；否则返回 `None`。
    """

    value = getattr(item, "element_id", None)
    return value if isinstance(value, int) else None


def _build_table_split_entry(item: Item) -> Optional[Dict[str, Any]]:
    """将表格 Item 归一化为按行切分所需的字典结构。

    Args:
        item: 原始表格内容项。

    Returns:
        Optional[Dict[str, Any]]: 可供 `split_table_by_rows` 使用的表格结构；
        如果内容项不是可序列化的表格则返回 `None`。
    """

    if not is_table_like_item(item) or not hasattr(item, "model_dump"):
        return None
    raw = item.model_dump(mode="python")
    header_cells = raw.get("headerCells")
    if not isinstance(header_cells, list):
        header_cells = raw.get("header_cells")
    return {
        "type": raw.get("type"),
        "rows": raw.get("rows") or [],
        "hasHeader": raw.get("hasHeader") if raw.get("hasHeader") is not None else raw.get("has_header"),
        "headerCells": header_cells if isinstance(header_cells, list) else None,
    }


def _collect_deeper_heading_levels(items: Sequence[Item], floor_level: int) -> List[int]:
    """收集超过当前基准层级的更深标题层级。

    Args:
        items: 当前叶子块内的内容项。
        floor_level: 当前细化层的基准标题层级。

    Returns:
        List[int]: 升序排列后的更深标题层级列表。
    """

    levels = {
        level
        for level in (get_heading_level(item) for item in items)
        if level is not None and level > floor_level
    }
    return sorted(levels)


def _has_meaningful_own_items(own_items: Sequence[Item], heading_item: Optional[Item]) -> bool:
    """判断节点直属内容是否足以形成独立叶子块。

    Args:
        own_items: 当前节点直接持有的内容项。
        heading_item: 当前节点自身的标题项。

    Returns:
        bool: 当直属内容不只是标题占位，且具备独立语义时返回 `True`。
    """

    if not own_items:
        return False
    if heading_item is None:
        return len(own_items) > 1 or any(not is_heading_item(item) for item in own_items)
    if len(own_items) > 1:
        return True
    return get_item_element_id(own_items[0]) != get_item_element_id(heading_item)


def _collect_local_refined_leaves(node: _LocalNode, *, origin_leaf_id: str) -> List[RefinedLeafInfo]:
    """从局部标题树中收集最终叶子块。

    Args:
        node: 当前遍历到的局部树节点。
        origin_leaf_id: 原始叶子块 ID。

    Returns:
        List[RefinedLeafInfo]: 从该节点及其子树提取出的细化叶子块列表。
    """

    leaves: List[RefinedLeafInfo] = []

    if node.children:
        if _has_meaningful_own_items(node.own_items, node.heading_item):
            leaves.append(
                RefinedLeafInfo(
                    items=list(node.own_items),
                    heading_path=list(node.heading_path),
                    heading_node_ids=list(node.heading_node_ids),
                    floor_level=node.level,
                    origin_leaf_id=origin_leaf_id,
                    type_override=infer_leaf_type(node.own_items),
                )
            )
        for child in node.children:
            leaves.extend(_collect_local_refined_leaves(child, origin_leaf_id=origin_leaf_id))
        return leaves

    if node.own_items:
        leaves.append(
            RefinedLeafInfo(
                items=list(node.own_items),
                heading_path=list(node.heading_path),
                heading_node_ids=list(node.heading_node_ids),
                floor_level=node.level,
                origin_leaf_id=origin_leaf_id,
                type_override=infer_leaf_type(node.own_items),
            )
        )
    return leaves


def _build_local_structured_leaves(
    *,
    items: Sequence[Item],
    floor_level: int,
    base_heading_path: Sequence[str],
    base_heading_node_ids: Sequence[int],
    origin_leaf_id: str,
) -> List[RefinedLeafInfo]:
    """基于局部标题结构构造新的叶子块。

    仅使用比 `floor_level` 更深的前两级标题参与重建，避免局部递归时一次性拆得
    过细。

    Args:
        items: 当前叶子块内的原始内容项。
        floor_level: 当前细化层的基准标题层级。
        base_heading_path: 上游已确定的标题路径。
        base_heading_node_ids: 上游标题路径对应的元素 ID。
        origin_leaf_id: 原始叶子块 ID。

    Returns:
        List[RefinedLeafInfo]: 基于局部标题树重建出的叶子块列表。
    """

    deeper_levels = _collect_deeper_heading_levels(items, floor_level=floor_level)
    if not deeper_levels:
        logger.debug("No deeper headings found during leaf refinement: origin_leaf_id={}", origin_leaf_id)
        return []

    # 只暴露更深的前两级标题，避免局部结构拆分过于激进。
    visible_levels = set(deeper_levels[:2])
    root = _LocalNode(
        level=floor_level,
        heading_item=None,
        heading_path=list(base_heading_path),
        heading_node_ids=list(base_heading_node_ids),
    )
    stack: List[_LocalNode] = [root]

    for item in items:
        level = get_heading_level(item)
        is_structural_heading = level is not None and level in visible_levels
        if is_structural_heading:
            while len(stack) > 1 and stack[-1].level >= int(level):
                stack.pop()
            parent = stack[-1]
            heading_path = list(parent.heading_path)
            heading_text = get_heading_text(item)
            if heading_text:
                heading_path.append(heading_text)
            heading_node_ids = list(parent.heading_node_ids)
            heading_id = get_item_element_id(item)
            if heading_id is not None:
                heading_node_ids.append(heading_id)
            node = _LocalNode(
                level=int(level),
                heading_item=item,
                heading_path=heading_path,
                heading_node_ids=heading_node_ids,
                own_items=[item],
            )
            parent.children.append(node)
            stack.append(node)
            continue
        stack[-1].own_items.append(item)

    refined_leaves = _collect_local_refined_leaves(root, origin_leaf_id=origin_leaf_id)
    logger.debug(
        "Structured local leaf refinement complete: origin_leaf_id={}, refined_leaf_count={}",
        origin_leaf_id,
        len(refined_leaves),
    )
    return refined_leaves


def _split_text_to_target_char_pieces(text: str, *, target_chars: int) -> List[tuple[str, str]]:
    """按目标字符数切分文本，并标记切分策略。

    优先以段落为单位聚合；当单段本身过长时，再退化为定长字符切分。

    Args:
        text: 待切分的原始文本。
        target_chars: 目标字符数上限。

    Returns:
        List[tuple[str, str]]: `(片段文本, 切分策略)` 元组列表。
    """

    normalized_text = str(text or "").strip()
    if not normalized_text:
        return []

    char_limit = max(1, int(target_chars))
    paragraphs = [
        segment.strip()
        for segment in re.split(r"(?:\r?\n){2,}", normalized_text)
        if segment and segment.strip()
    ]
    if not paragraphs:
        paragraphs = [normalized_text]

    pieces: List[tuple[str, str]] = []
    paragraph_buffer: List[str] = []
    paragraph_buffer_chars = 0

    def flush_paragraph_buffer() -> None:
        nonlocal paragraph_buffer, paragraph_buffer_chars
        if not paragraph_buffer:
            return
        piece_text = "\n\n".join(paragraph_buffer).strip()
        if piece_text:
            pieces.append((piece_text, "paragraph"))
        paragraph_buffer = []
        paragraph_buffer_chars = 0

    for paragraph_text in paragraphs:
        paragraph_chars = len(paragraph_text)
        if paragraph_chars > char_limit:
            flush_paragraph_buffer()
            for start in range(0, paragraph_chars, char_limit):
                piece_text = paragraph_text[start : start + char_limit].strip()
                if piece_text:
                    pieces.append((piece_text, "char"))
            continue

        separator_chars = 2 if paragraph_buffer else 0
        if paragraph_buffer and paragraph_buffer_chars + separator_chars + paragraph_chars > char_limit:
            flush_paragraph_buffer()
        paragraph_buffer.append(paragraph_text)
        paragraph_buffer_chars += separator_chars + paragraph_chars

    flush_paragraph_buffer()
    return pieces or [(normalized_text, "char")]


def _split_single_item_sequential(
    *,
    item: Item,
    floor_level: int,
    heading_path: Sequence[str],
    heading_node_ids: Sequence[int],
    origin_leaf_id: str,
    target_chars: int,
) -> List[RefinedLeafInfo]:
    """将单个超长内容项按顺序切分为多个叶子块。

    Args:
        item: 待切分的单个内容项。
        floor_level: 当前细化层的基准标题层级。
        heading_path: 当前继承的标题路径。
        heading_node_ids: 当前继承的标题节点 ID 列表。
        origin_leaf_id: 原始叶子块 ID。
        target_chars: 目标字符数上限。

    Returns:
        List[RefinedLeafInfo]: 拆分后的叶子块列表。
    """

    item_text = str(extract_indexable_text(item) or "")
    item_type = str(getattr(item, "type", None) or "text").strip() or "text"
    refined_chunks: List[RefinedLeafInfo] = []
    for piece_text, split_strategy in _split_text_to_target_char_pieces(item_text, target_chars=target_chars):
        refined_chunks.append(
            RefinedLeafInfo(
                items=[item],
                heading_path=list(heading_path),
                heading_node_ids=list(heading_node_ids),
                floor_level=floor_level,
                origin_leaf_id=origin_leaf_id,
                text_override=piece_text,
                split_strategy=split_strategy,
                type_override=item_type,
            )
        )
    return refined_chunks


def split_items_sequential(
    *,
    items: Sequence[Item],
    floor_level: int,
    heading_path: Sequence[str],
    heading_node_ids: Sequence[int],
    origin_leaf_id: str,
    target_chars: int = SUMMARY_FALLBACK_TARGET_CHARS,
) -> List[RefinedLeafInfo]:
    """按内容顺序切分叶子块中的 Item 列表。

    常规文本按字符数阈值累积；超长表格优先走按行切分；单个超长内容项则退化为
    文本片段切分。

    Args:
        items: 待切分的内容项序列。
        floor_level: 当前细化层的基准标题层级。
        heading_path: 当前继承的标题路径。
        heading_node_ids: 当前继承的标题节点 ID 列表。
        origin_leaf_id: 原始叶子块 ID。
        target_chars: 目标字符数上限。

    Returns:
        List[RefinedLeafInfo]: 顺序切分后的叶子块列表。
    """

    if not items:
        return []

    table_split_threshold = RAG_SETTINGS.table_split_threshold
    table_row_text_max_length = RAG_SETTINGS.table_row_text_max_length

    refined_chunks: List[RefinedLeafInfo] = []
    current_chunk: List[Item] = []
    current_chars = 0

    def flush_current_chunk() -> None:
        nonlocal current_chunk, current_chars
        if not current_chunk:
            return
        refined_chunks.append(
            RefinedLeafInfo(
                items=list(current_chunk),
                heading_path=list(heading_path),
                heading_node_ids=list(heading_node_ids),
                floor_level=floor_level,
                origin_leaf_id=origin_leaf_id,
                type_override=infer_leaf_type(current_chunk),
            )
        )
        current_chunk = []
        current_chars = 0

    for item in items:
        item_text = str(extract_indexable_text(item) or "")
        item_chars = len(item_text)

        # 超长表格优先按行拆分，尽量保留表头和行级上下文。
        if is_table_like_item(item) and item_text and (
            len(item_text.encode("utf-8")) > table_split_threshold or item_chars > target_chars
        ):
            flush_current_chunk()
            table_entry = _build_table_split_entry(item)
            table_pieces = (
                split_table_by_rows(
                    table_entry,
                    text_limit=table_row_text_max_length,
                    element_id=get_item_element_id(item),
                )
                if table_entry
                else []
            )
            if table_pieces:
                for piece in table_pieces:
                    piece_text = str(piece.get("text") or "").strip()
                    if not piece_text:
                        continue
                    refined_chunks.append(
                        RefinedLeafInfo(
                            items=[item],
                            heading_path=list(heading_path),
                            heading_node_ids=list(heading_node_ids),
                            floor_level=floor_level,
                            origin_leaf_id=origin_leaf_id,
                            text_override=piece_text,
                            split_strategy="table_row",
                            table_row_start=piece.get("row_start"),
                            table_row_end=piece.get("row_end"),
                            type_override="table_row",
                        )
                    )
                continue

        if item_chars > target_chars:
            flush_current_chunk()
            refined_chunks.extend(
                _split_single_item_sequential(
                    item=item,
                    floor_level=floor_level,
                    heading_path=heading_path,
                    heading_node_ids=heading_node_ids,
                    origin_leaf_id=origin_leaf_id,
                    target_chars=target_chars,
                )
            )
            continue

        if current_chunk and current_chars >= target_chars:
            flush_current_chunk()
        current_chunk.append(item)
        current_chars += item_chars

    flush_current_chunk()
    logger.debug(
        "Sequential item split complete: origin_leaf_id={}, refined_chunk_count={}, target_chars={}",
        origin_leaf_id,
        len(refined_chunks),
        target_chars,
    )
    return refined_chunks


def refine_leaf(
    leaf: RefinedLeafInfo,
    *,
    leaf_max_chars: int,
    leaf_max_recursion: int,
    recursion_depth: int = 0,
) -> List[RefinedLeafInfo]:
    """递归细化单个叶子块。

    先尝试利用局部标题结构递归拆分；当标题结构不足、递归达到上限或结果仍然过粗
    时，回退到顺序切分策略。

    Args:
        leaf: 待细化的叶子块。
        leaf_max_chars: 单个叶子块允许的最大字符数。
        leaf_max_recursion: 允许的最大递归深度。
        recursion_depth: 当前递归深度。

    Returns:
        List[RefinedLeafInfo]: 细化后的叶子块列表。
    """

    leaf_text = extract_refined_leaf_text(leaf)
    if len(leaf_text) <= leaf_max_chars:
        return [leaf]

    if recursion_depth >= leaf_max_recursion:
        logger.debug(
            "Leaf refinement hit recursion limit, falling back to sequential split: origin_leaf_id={}",
            leaf.origin_leaf_id,
        )
        return split_items_sequential(
            items=leaf.items,
            floor_level=leaf.floor_level,
            heading_path=leaf.heading_path,
            heading_node_ids=leaf.heading_node_ids,
            origin_leaf_id=leaf.origin_leaf_id,
            target_chars=leaf_max_chars,
        )

    structured_leaves = _build_local_structured_leaves(
        items=leaf.items,
        floor_level=leaf.floor_level,
        base_heading_path=leaf.heading_path,
        base_heading_node_ids=leaf.heading_node_ids,
        origin_leaf_id=leaf.origin_leaf_id,
    )
    if len(structured_leaves) <= 1:
        logger.debug(
            "Structured refinement produced too few leaves, falling back to sequential split: origin_leaf_id={}",
            leaf.origin_leaf_id,
        )
        return split_items_sequential(
            items=leaf.items,
            floor_level=leaf.floor_level,
            heading_path=leaf.heading_path,
            heading_node_ids=leaf.heading_node_ids,
            origin_leaf_id=leaf.origin_leaf_id,
            target_chars=leaf_max_chars,
        )

    refined: List[RefinedLeafInfo] = []
    for child_leaf in structured_leaves:
        child_text = extract_refined_leaf_text(child_leaf)
        if len(child_text) <= leaf_max_chars:
            refined.append(child_leaf)
            continue
        refined.extend(
            refine_leaf(
                child_leaf,
                leaf_max_chars=leaf_max_chars,
                leaf_max_recursion=leaf_max_recursion,
                recursion_depth=recursion_depth + 1,
            )
        )

    if len(refined) <= 1 and len(leaf_text) > leaf_max_chars:
        logger.debug(
            "Recursive refinement still produced too few leaves, falling back to sequential split: origin_leaf_id={}",
            leaf.origin_leaf_id,
        )
        return split_items_sequential(
            items=leaf.items,
            floor_level=leaf.floor_level,
            heading_path=leaf.heading_path,
            heading_node_ids=leaf.heading_node_ids,
            origin_leaf_id=leaf.origin_leaf_id,
            target_chars=leaf_max_chars,
        )

    logger.debug(
        "Leaf refinement complete: origin_leaf_id={}, refined_leaf_count={}, recursion_depth={}",
        leaf.origin_leaf_id,
        len(refined),
        recursion_depth,
    )
    return refined


__all__ = [
    "RefinedLeafInfo",
    "extract_items_text",
    "extract_indexable_text",
    "extract_refined_leaf_text",
    "get_heading_level",
    "get_heading_text",
    "get_item_element_id",
    "infer_leaf_type",
    "is_heading_item",
    "refine_leaf",
    "split_items_sequential",
]
