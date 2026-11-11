from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.agents.content_generate_agent.hierarchical_summary.summary_generator import (
    SummaryGenerator,
)
from src.agents.content_generate_agent.hierarchical_summary.tree_builder import (
    MAX_HEADING_LEVEL,
    Node,
    build_tree,
)
from src.agents.content_generate_agent.input_models import (
    BBox,
    DocumentItems,
    ImageItem,
    Item,
    TextItem,
    extract_text_yusys_ai,
)
from src.applications.rag.chunking.structured_leaf_refinement import (
    RefinedLeafInfo,
    extract_items_text as _extract_refined_items_text,
    extract_refined_leaf_text as _extract_refined_leaf_text,
    get_heading_level as _get_heading_level,
    get_heading_text as _get_heading_text,
    get_item_element_id as _get_item_element_id,
    infer_leaf_type as _infer_leaf_type,
    refine_leaf as _refine_leaf,
)
from src.applications.rag.config.rag_settings import RAG_SETTINGS
from src.applications.rag.schemas import (
    ChildChunk,
    ChunkFields,
    GrandchildChunk,
    ParentChunk,
)
from src.applications.rag.services.image_summary_service import (
    resolve_image_storage_path,
)
from src.applications.rag.utils.chunking_helpers import hash_id
from loguru import logger


SUMMARY_PARENT_PREFIX = "sp"
SUMMARY_CHILD_PREFIX = "sc"
SUMMARY_GRANDCHILD_PREFIX = "sg"


@dataclass
class OriginalLeafInfo:
    node: Node
    heading_path: List[str]
    heading_node_ids: List[int]
    floor_level: int
    origin_leaf_id: str


def _normalize_mineru_items(
    mineru_json: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for raw in mineru_json or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        if item.get("type") == "table":
            item.setdefault("standard2D", False)
            item.setdefault("hasHeader", False)
        normalized.append(item)
    return normalized


def _load_items_tolerant(mineru_json: Sequence[Dict[str, Any]]) -> List[Item]:
    document_items = DocumentItems.model_validate(_normalize_mineru_items(mineru_json))
    return list(document_items.root)


def _extract_items_text(items: Iterable[Item]) -> str:
    return _extract_refined_items_text(items)


def _extract_node_text(node: Node) -> str:
    return _extract_items_text(node.content)


def _should_use_text_summary(items: Sequence[Item], original_text: str) -> bool:
    if not original_text.strip():
        return False
    return not any(extract_text_yusys_ai(item) for item in items)


def _should_emit_generic_summary_grandchild(item_type: str | None) -> bool:
    normalized_type = str(item_type or "").strip().lower()
    return normalized_type != "image"


def _is_excel_like_source(
    items: Sequence[Dict[str, Any]],
    file_name: str | None,
) -> bool:
    if file_name:
        suffix = Path(file_name).suffix.lower()
        if suffix in {".xls", ".xlsx", ".csv"}:
            return True
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() == "excel_table":
            return True
        raw = item.get("source_type") or item.get("file_type") or item.get("sourceType") or ""
        source_type = str(raw).strip().lower()
        if source_type in {"excel", "xlsx", "xls", "csv", "spreadsheet"}:
            return True
    return False


def _unique_ints(values: Iterable[Optional[int]]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _get_source_element_ids(items: Sequence[Item]) -> List[int]:
    return _unique_ints(_get_item_element_id(item) for item in items)


def _infer_node_heading(node: Node) -> tuple[str, Optional[int], Optional[int]]:
    for item in node.content:
        level = _get_heading_level(item)
        if level is None or level > MAX_HEADING_LEVEL:
            continue
        return _get_heading_text(item), level, _get_item_element_id(item)
    return "", None, None


def _make_origin_leaf_id(
    *,
    file_id: str,
    heading_path: Sequence[str],
    source_element_ids: Sequence[int],
) -> str:
    first_id = source_element_ids[0] if source_element_ids else -1
    last_id = source_element_ids[-1] if source_element_ids else -1
    seed = "|".join(
        [
            file_id,
            ",".join(heading_path),
            str(first_id),
            str(last_id),
            str(len(source_element_ids)),
        ]
    )
    return f"sl_{hash_id(seed, length=16)}"


def _first_source_element_id(items: Sequence[Item]) -> int:
    source_element_ids = _get_source_element_ids(items)
    if source_element_ids:
        return int(source_element_ids[0])
    return 1 << 60


def _collect_orphan_root_leaf_infos(
    root: Node,
    *,
    existing_leaf_infos: Sequence[OriginalLeafInfo],
    file_id: str,
) -> List[OriginalLeafInfo]:
    # root.content 中可能残留“未挂到任何标题叶子”的内容，
    # 例如首个标题前的引言段落，或标题结构之外的孤立块。
    covered_source_element_ids: set[int] = set()
    for leaf_info in existing_leaf_infos:
        covered_source_element_ids.update(
            _get_source_element_ids(leaf_info.node.content)
        )

    orphan_leaf_infos: List[OriginalLeafInfo] = []
    current_items: List[Item] = []

    def _flush_current_items() -> None:
        if not current_items:
            return

        # 保持 root 层原始连续块的边界，避免把相隔较远的 orphan 内容错误拼成一个叶子。
        source_element_ids = _get_source_element_ids(current_items)
        orphan_leaf_infos.append(
            OriginalLeafInfo(
                node=Node(content=list(current_items)),
                heading_path=[],
                heading_node_ids=[],
                floor_level=0,
                origin_leaf_id=_make_origin_leaf_id(
                    file_id=file_id,
                    heading_path=[],
                    source_element_ids=source_element_ids,
                ),
            )
        )
        current_items.clear()

    for item in root.content:
        element_id = _get_item_element_id(item)
        # 命中已被标题叶子覆盖的元素时，结束当前 orphan 连续块。
        if element_id is not None and element_id in covered_source_element_ids:
            _flush_current_items()
            continue
        current_items.append(item)

    _flush_current_items()
    return orphan_leaf_infos


def _collect_original_leaf_infos(
    root: Node,
    *,
    file_id: str,
) -> List[OriginalLeafInfo]:
    leaf_infos: List[OriginalLeafInfo] = []

    def _walk(
        node: Node,
        heading_path: List[str],
        heading_node_ids: List[int],
        floor_level: int,
    ) -> None:
        heading_text, node_level, heading_element_id = _infer_node_heading(node)
        next_heading_path = list(heading_path)
        next_heading_node_ids = list(heading_node_ids)
        next_floor_level = floor_level

        if heading_text:
            next_heading_path.append(heading_text)
        if heading_element_id is not None:
            next_heading_node_ids.append(heading_element_id)
        if node_level is not None:
            next_floor_level = node_level

        if not node.childs:
            source_element_ids = _get_source_element_ids(node.content)
            leaf_infos.append(
                OriginalLeafInfo(
                    node=node,
                    heading_path=next_heading_path,
                    heading_node_ids=next_heading_node_ids,
                    floor_level=next_floor_level,
                    origin_leaf_id=_make_origin_leaf_id(
                        file_id=file_id,
                        heading_path=next_heading_path,
                        source_element_ids=source_element_ids,
                    ),
                )
            )
            return

        for child in node.childs:
            _walk(child, next_heading_path, next_heading_node_ids, next_floor_level)

    for child in root.childs:
        _walk(child, [], [], 0)

    orphan_root_leaf_infos = _collect_orphan_root_leaf_infos(
        root,
        existing_leaf_infos=leaf_infos,
        file_id=file_id,
    )
    if orphan_root_leaf_infos:
        leaf_infos.extend(orphan_root_leaf_infos)

    leaf_infos.sort(
        key=lambda leaf_info: _first_source_element_id(leaf_info.node.content)
    )

    logger.debug(
        "已收集原始标题叶子节点: file_id={}, count={}", file_id, len(leaf_infos)
    )
    return leaf_infos


async def _run_summaries(nodes: Sequence[Node]) -> None:
    if not nodes:
        return
    logger.debug("开始为节点批量生成摘要: count={}", len(nodes))
    generator = SummaryGenerator(root=Node())
    await generator._llm_summarize_batch(list(nodes))


async def _run_text_summaries(texts: Sequence[str]) -> List[str]:
    if not texts:
        return []
    logger.debug("开始为文本批量生成摘要: count={}", len(texts))
    nodes = [
        Node(
            content=[
                TextItem(
                    page=1,
                    type="text",
                    bBox=BBox(l=0, t=0, r=0, b=0),
                    value=text,
                    md=text,
                    file_id="summary_index_virtual_file",
                    element_id=index + 1,
                )
            ]
        )
        for index, text in enumerate(texts)
    ]
    await _run_summaries(nodes)
    return [str(node.summary or "") for node in nodes]


def _heading_path_to_text(heading_path: Sequence[str]) -> str:
    return "/".join(
        str(part).strip() for part in heading_path if str(part or "").strip()
    )


def _extract_image_metadata(
    items: Sequence[Item],
    *,
    file_id: str,
) -> Dict[str, Any]:
    image_items = [item for item in items if isinstance(item, ImageItem)]
    if len(image_items) != 1:
        return {}

    image_item = image_items[0]
    raw_img_path = str(image_item.img_path or "").strip()
    image_storage_path = resolve_image_storage_path(raw_img_path, file_id=file_id)

    metadata: Dict[str, Any] = {}
    if image_storage_path:
        metadata["image_storage_path"] = image_storage_path
    return metadata

def _apply_refined_leaf_metadata_overrides(
    chunk: ChunkFields,
    refined_leaf: RefinedLeafInfo,
) -> ChunkFields:
    chunk.type = str(refined_leaf.type_override or chunk.type or "text").strip() or "text"
    if refined_leaf.split_strategy:
        chunk.split_strategy = refined_leaf.split_strategy
    if refined_leaf.table_row_start is not None:
        chunk.table_row_start = refined_leaf.table_row_start
    if refined_leaf.table_row_end is not None:
        chunk.table_row_end = refined_leaf.table_row_end
    return chunk


class SummaryIndexBuilder:
    def __init__(
        self,
        mineru_json: Sequence[Dict[str, Any]],
        file_id: str,
        folder_id: Optional[str],
        file_name: Optional[str],
        build_summaries: bool = True,
        emit_summary_grandchildren: bool = False,
    ):
        self.mineru_json = mineru_json
        self.file_id = file_id
        self.folder_id = folder_id
        self.file_name = file_name
        self.build_summaries = build_summaries
        self.leaf_max_chars = RAG_SETTINGS.summary_leaf_max_chars
        self.leaf_max_recursion = RAG_SETTINGS.summary_leaf_max_recursion
        self.emit_summary_grandchildren = (
            emit_summary_grandchildren
            and RAG_SETTINGS.summary_index_enabled
            and not _is_excel_like_source(mineru_json, file_name)
        )

    async def _build_parent_summaries(self) -> Dict[str, str]:
        parent_summaries: Dict[str, str] = {}
        nodes_to_summarize: List[Node] = []
        node_leaf_ids: List[str] = []
        text_summary_inputs: List[str] = []
        text_summary_leafs: List[OriginalLeafInfo] = []

        for leaf in self.original_leafs:
            original_text = self.original_text_by_leaf_id.get(leaf.origin_leaf_id, "")
            if not original_text.strip():
                parent_summaries[leaf.origin_leaf_id] = ""
                leaf.node.summary = ""
                continue
            if _should_use_text_summary(leaf.node.content, original_text):
                text_summary_inputs.append(original_text)
                text_summary_leafs.append(leaf)
                continue
            if len(original_text) <= self.leaf_max_chars:
                nodes_to_summarize.append(leaf.node)
                node_leaf_ids.append(leaf.origin_leaf_id)
                continue
            text_summary_inputs.append(original_text)
            text_summary_leafs.append(leaf)

        if nodes_to_summarize:
            await _run_summaries(nodes_to_summarize)
            for leaf_id, node in zip(node_leaf_ids, nodes_to_summarize):
                parent_summaries[leaf_id] = str(node.summary or "")

        if text_summary_inputs:
            text_summaries = await _run_text_summaries(text_summary_inputs)
            for leaf, summary in zip(text_summary_leafs, text_summaries):
                normalized_summary = str(summary or "")
                parent_summaries[leaf.origin_leaf_id] = normalized_summary
                leaf.node.summary = normalized_summary

        for leaf in self.original_leafs:
            parent_summaries.setdefault(leaf.origin_leaf_id, "")
        return parent_summaries

    async def build_leaf(self, leaf: OriginalLeafInfo):
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        grandchildren: list[GrandchildChunk] = []
        
        original_text = self.original_text_by_leaf_id.get(leaf.origin_leaf_id, "")
        parent_summary = self.parent_summary_by_leaf_id.get(leaf.origin_leaf_id, "")
        source_element_ids = _get_source_element_ids(leaf.node.content)
        leaf_item_type = _infer_leaf_type(leaf.node.content)
        parent_id = f"{SUMMARY_PARENT_PREFIX}_{hash_id(f'{leaf.origin_leaf_id}|parent', length=16)}"
        parent_heading_path_text = _heading_path_to_text(leaf.heading_path)
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                text=original_text,
                summary=str(parent_summary or ""),
                file_id=self.file_id,
                folder_id=self.folder_id or "",
                file_name=self.file_name or "",
                type=str(leaf_item_type or "text").strip() or "text",
                heading_path=parent_heading_path_text,
                heading_node_ids=list(leaf.heading_node_ids),
                source_element_ids=list(source_element_ids),
            )
        )

        if len(original_text) <= self.leaf_max_chars:
            refined_leafs = [
                RefinedLeafInfo(
                    items=list(leaf.node.content),
                    heading_path=list(leaf.heading_path),
                    heading_node_ids=list(leaf.heading_node_ids),
                    floor_level=leaf.floor_level,
                    origin_leaf_id=leaf.origin_leaf_id,
                    summary=(
                        leaf.node.summary
                        if self.emit_summary_grandchildren
                        and _should_emit_generic_summary_grandchild(leaf_item_type)
                        else None
                    ),
                    type_override=leaf_item_type,
                )
            ]
        else:
            refined_leafs = _refine_leaf(
                RefinedLeafInfo(
                    items=list(leaf.node.content),
                    heading_path=list(leaf.heading_path),
                    heading_node_ids=list(leaf.heading_node_ids),
                    floor_level=leaf.floor_level,
                    origin_leaf_id=leaf.origin_leaf_id,
                ),
                leaf_max_chars=self.leaf_max_chars,
                leaf_max_recursion=self.leaf_max_recursion,
            )

            if self.emit_summary_grandchildren:
                nodes_to_summarize: List[Node] = []
                node_summary_targets: List[RefinedLeafInfo] = []
                text_summary_targets: List[RefinedLeafInfo] = []
                text_summary_inputs: List[str] = []
                for refined_leaf in refined_leafs:
                    refined_text = _extract_refined_leaf_text(refined_leaf)
                    if not refined_text:
                        refined_leaf.summary = ""
                        continue
                    if not _should_emit_generic_summary_grandchild(
                        refined_leaf.type_override
                    ):
                        refined_leaf.summary = None
                        continue
                    if (
                        refined_leaf.text_override is not None
                        or _should_use_text_summary(
                            refined_leaf.items,
                            refined_text,
                        )
                    ):
                        text_summary_targets.append(refined_leaf)
                        text_summary_inputs.append(refined_text)
                        continue
                    nodes_to_summarize.append(Node(content=list(refined_leaf.items)))
                    node_summary_targets.append(refined_leaf)
                if nodes_to_summarize:
                    await _run_summaries(nodes_to_summarize)
                    for refined_leaf, node in zip(
                        node_summary_targets, nodes_to_summarize
                    ):
                        refined_leaf.summary = node.summary
                if text_summary_inputs:
                    text_summaries = await _run_text_summaries(text_summary_inputs)
                    for refined_leaf, summary in zip(
                        text_summary_targets, text_summaries
                    ):
                        refined_leaf.summary = summary

        for index, refined_leaf in enumerate(refined_leafs):
            child_text = _extract_refined_leaf_text(refined_leaf)
            if not child_text.strip():
                continue

            refined_source_ids = _get_source_element_ids(refined_leaf.items)
            child_seed = "|".join(
                [
                    leaf.origin_leaf_id,
                    "child",
                    str(index),
                    ",".join(str(value) for value in refined_source_ids),
                    ",".join(refined_leaf.heading_path),
                    str(refined_leaf.split_strategy or ""),
                    str(refined_leaf.table_row_start or ""),
                    str(refined_leaf.table_row_end or ""),
                ]
            )
            child_id = f"{SUMMARY_CHILD_PREFIX}_{hash_id(child_seed, length=16)}"
            child_heading_path_text = _heading_path_to_text(refined_leaf.heading_path)
            child_payload = ChildChunk(
                child_id=child_id,
                parent_id=parent_id,
                text=child_text,
                file_id=self.file_id,
                folder_id=self.folder_id or "",
                file_name=self.file_name or "",
                type=str(refined_leaf.type_override or "text").strip() or "text",
                heading_path=child_heading_path_text,
                heading_node_ids=list(refined_leaf.heading_node_ids),
                source_element_ids=list(refined_source_ids),
            )
            for key, value in _extract_image_metadata(refined_leaf.items, file_id=self.file_id).items():
                setattr(child_payload, key, value)
            child_payload = _apply_refined_leaf_metadata_overrides(
                child_payload,
                refined_leaf,
            )
            children.append(child_payload)

            if not self.emit_summary_grandchildren:
                continue

            summary_text = str(refined_leaf.summary or "").strip()
            if not summary_text:
                continue
            grandchild_id = f"{SUMMARY_GRANDCHILD_PREFIX}_{hash_id(f'{child_seed}|summary', length=16)}"
            grandchild_heading_path_text = _heading_path_to_text(refined_leaf.heading_path)
            grandchild_payload = GrandchildChunk(
                grandchild_id=grandchild_id,
                parent_id=parent_id,
                child_id=child_id,
                text=summary_text,
                file_id=self.file_id,
                folder_id=self.folder_id or "",
                file_name=self.file_name or "",
                type=str(refined_leaf.type_override or "text").strip() or "text",
                heading_path=grandchild_heading_path_text,
                heading_node_ids=list(refined_leaf.heading_node_ids),
                source_element_ids=list(refined_source_ids),
                doc_kind="summary",
                element_id=refined_source_ids[0] if refined_source_ids else None,
            )
            grandchild_payload = _apply_refined_leaf_metadata_overrides(
                grandchild_payload,
                refined_leaf,
            )
            grandchildren.append(grandchild_payload)
        
        return parents, children, grandchildren

    async def build(self):
        items = _load_items_tolerant(self.mineru_json)
        root = build_tree(items)
        self.original_leafs = _collect_original_leaf_infos(root, file_id=self.file_id)

        if not self.original_leafs:
            logger.warning(
                "未收集到可用于摘要索引的标题叶子节点: file_id={}", self.file_id
            )
            return [], [], []

        original_text_by_leaf_id = {
            leaf.origin_leaf_id: _extract_node_text(leaf.node)
            for leaf in self.original_leafs
        }
        self.original_text_by_leaf_id = original_text_by_leaf_id

        if self.build_summaries:
            parent_summary_by_leaf_id = await self._build_parent_summaries()
        else:
            parent_summary_by_leaf_id = {}
            for leaf in self.original_leafs:
                leaf.node.summary = ""
        self.parent_summary_by_leaf_id = parent_summary_by_leaf_id

        self.parents: list[ParentChunk] = []
        self.children: list[ChildChunk] = []
        self.grandchildren: list[GrandchildChunk] = []

        tasks = [self.build_leaf(leaf) for leaf in self.original_leafs]
        results = await asyncio.gather(*tasks)

        for leaf_parents, leaf_children, leaf_grandchildren in results:
            self.parents.extend(leaf_parents)
            self.children.extend(leaf_children)
            self.grandchildren.extend(leaf_grandchildren)

        return self.parents, self.children, self.grandchildren


async def build_summary_index(
    mineru_json: Sequence[Dict[str, Any]],
    *,
    file_id: str,
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
    build_summaries: bool = True,
    emit_summary_grandchildren: bool = False,
) -> Tuple[list[ParentChunk], list[ChildChunk], list[GrandchildChunk]]:

    start_time = time.perf_counter()
    builder = SummaryIndexBuilder(
        mineru_json=mineru_json,
        file_id=file_id,
        folder_id=folder_id,
        file_name=file_name,
        build_summaries=build_summaries,
        emit_summary_grandchildren=emit_summary_grandchildren,
    )
    parents, children, grandchildren = await builder.build()
    elapsed_time = time.perf_counter() - start_time
    return parents, children, grandchildren


async def build_summary_grandchildren_from_children(
    children: Sequence[ChildChunk],
    *,
    file_id: str,
    folder_id: str | None = None,
    file_name: str | None = None,
) -> list[GrandchildChunk]:
    """从已提交的子切片生成 summary 孙切片。

    仅从子切片文本和元数据派生摘要，不依赖 mineru_json 重建。
    生成的 summary 孙切片使用稳定 ID，确保重试幂等。
    """
    if not children:
        return []

    eligible_children: list[ChildChunk] = []
    texts_to_summarize: list[str] = []
    for child in children:
        child_text = str(child.text or "").strip()
        if not child_text:
            continue
        if not _should_emit_generic_summary_grandchild(child.type):
            continue
        eligible_children.append(child)
        texts_to_summarize.append(child_text)

    if not texts_to_summarize:
        return []

    summaries = await _run_text_summaries(texts_to_summarize)

    result: list[GrandchildChunk] = []
    for child, summary in zip(eligible_children, summaries):
        summary_text = str(summary or "").strip()
        if not summary_text:
            continue

        child_id = child.child_id
        parent_id = child.parent_id
        grandchild_id = f"{SUMMARY_GRANDCHILD_PREFIX}_{hash_id(f'{child_id}|summary', length=16)}"

        source_element_ids = list(child.source_element_ids or [])
        heading_path = str(child.heading_path or "")
        heading_node_ids = list(child.heading_node_ids or [])
        type_val = str(child.type or "text").strip() or "text"

        gc = GrandchildChunk(
            grandchild_id=grandchild_id,
            parent_id=parent_id,
            child_id=child_id,
            text=summary_text,
            file_id=file_id,
            folder_id=folder_id or str(child.folder_id or ""),
            file_name=file_name or str(child.file_name or ""),
            type=type_val,
            heading_path=heading_path,
            heading_node_ids=heading_node_ids,
            source_element_ids=source_element_ids,
            doc_kind="summary",
            element_id=source_element_ids[0] if source_element_ids else None,
            split_strategy=str(child.split_strategy or ""),
            table_row_start=child.table_row_start,
            table_row_end=child.table_row_end,
        )
        result.append(gc)

    return result
