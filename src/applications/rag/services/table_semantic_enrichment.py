from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import hashlib
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger

from src.agents.content_generate_agent.utils import (
    ainvoke_llm_with_prompt as content_generate_ainvoke_llm_with_prompt,
)
from src.applications.rag.schemas import (
    ChildChunk,
    ChunkFields,
    GrandchildChunk,
)

from ..config.rag_settings import RAG_SETTINGS
from ..utils.text_normalize import add_spacing_for_tokenizer
from ..vector_document import truncate_utf8


_TABLE_TYPES = {"table", "excel_table"}
_TABLE_CONTEXT_BREAK_TYPES = {"table", "excel_table", "table_row"}
_TITLE_TYPES = {"title"}
_DEFAULT_TABLE_MODEL_ID = "model_bailian_table_header_aided"
_DEFAULT_TABLE_TIMEOUT_S = 120.0


@dataclass
class TableEnrichmentInput:
    element_id: Any
    element_id_string: str
    table_md: str
    raw_table_caption: str
    table_caption: str
    context_texts: List[str]
    context_element_ids: List[str]
    source_element_ids: List[Any]


@dataclass
class PreparedTableSemanticNode:
    element_id: Any
    element_id_string: str
    semantic_representation_text: str
    table_caption: str
    table_caption_source: str
    table_caption_context_element_ids: List[str]
    source_element_ids: List[Any]


def _normalize_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            item_text = str(item or "").strip()
            if item_text:
                parts.append(item_text)
        return "\n".join(parts).strip()
    return str(value).strip()


def _normalize_table_caption(value: Any) -> str:
    caption_text = _normalize_str(value)
    if not caption_text:
        return ""
    caption_text = caption_text.splitlines()[0].strip()
    caption_text = caption_text.strip('"').strip("'")
    caption_text = re.sub(r"^(table_caption|caption|title)\s*[:：]\s*", "", caption_text, flags=re.IGNORECASE)
    caption_text = re.sub(r"^\s*表格标题\s*[:：]\s*", "", caption_text)
    caption_text = re.sub(r"\s+", " ", caption_text).strip()
    if len(caption_text) > 120:
        caption_text = caption_text[:120].rstrip()
    return caption_text


def _resolve_element_id(element: Dict[str, Any], *, element_index: int) -> tuple[Any, str]:
    element_id = element.get("element_id")
    if element_id is None:
        element_id = element_index + 1
    return element_id, str(element_id).strip()


def _extract_text_element_content(element: Dict[str, Any]) -> str:
    if not isinstance(element, dict):
        return ""
    return _normalize_str(element.get("md") or element.get("value") or element.get("text"))


def _coerce_positive_heading_level(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        heading_level = int(value)
    except (TypeError, ValueError):
        return None
    if heading_level <= 0:
        return None
    return heading_level


def _is_heading_element(element: Dict[str, Any]) -> bool:
    element_type = str(element.get("type") or "").lower()
    if element_type in _TITLE_TYPES:
        return True
    if element_type != "text":
        return False
    return _coerce_positive_heading_level(element.get("headingLevel")) is not None


def _build_table_enrichment_prompt(
    *,
    context_texts: List[str],
    table_md: str,
    existing_caption: str,
) -> str:
    context_lines: List[str] = []
    for index, context_text in enumerate(context_texts):
        cleaned_context_text = str(context_text or "").strip()
        if cleaned_context_text:
            context_lines.append(f"[{index + 1}] {cleaned_context_text}")

    context_block = "\n".join(context_lines).strip() if context_lines else "（无）"
    caption_block = str(existing_caption or "").strip() or "（无）"
    table_block = str(table_md or "").strip() or "（无）"

    return (
        "你是一名表格语义增强助手。\n"
        "请基于提供的内容，生成一行简洁的中文检索描述。\n"
        "要求：\n"
        "- 只输出一行。\n"
        "- 仅使用提供内容中的事实。\n"
        "- 现有表格标题有帮助时优先使用。\n"
        "- 不要编造数据。\n\n"
        f"表格标题：\n{caption_block}\n\n"
        f"前文上下文：\n{context_block}\n\n"
        f"表格 Markdown：\n{table_block}\n\n"
        "答案："
    )

def _invoke_table_enrichment_llm(prompt: str, *, model_id: str, timeout_s: float = _DEFAULT_TABLE_TIMEOUT_S) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        coro = content_generate_ainvoke_llm_with_prompt(prompt, model_id=model_id)
        payload = asyncio.run(asyncio.wait_for(coro, timeout=timeout_s) if timeout_s is not None else coro)
        return str(payload or "").strip()
    raise RuntimeError(
        "_invoke_table_enrichment_llm 不能在运行中的事件循环内调用，请改用 _enrich_table"
    )


async def _enrich_table(
    prompt: str,
    *,
    model_id: str,
    timeout_s: float = _DEFAULT_TABLE_TIMEOUT_S,
) -> str:
    coro = content_generate_ainvoke_llm_with_prompt(prompt, model_id=model_id)
    if timeout_s is not None:
        payload = await asyncio.wait_for(coro, timeout=timeout_s)
    else:
        payload = await coro
    return str(payload or "").strip()


def _collect_table_enrichment_input(items: List[Dict[str, Any]]) -> List[TableEnrichmentInput]:
    if not items:
        return []

    max_previous_elements = RAG_SETTINGS.table_semantic_enrichment_llm_caption_prev_max

    collected_inputs: List[TableEnrichmentInput] = []
    seen_table_element_ids: set[str] = set()

    for element_index, element in enumerate(items or []):
        if not isinstance(element, dict):
            continue
        if str(element.get("type") or "").lower() not in _TABLE_TYPES:
            continue

        element_id, element_id_string = _resolve_element_id(element, element_index=element_index)
        if not element_id_string or element_id_string in seen_table_element_ids:
            continue
        seen_table_element_ids.add(element_id_string)

        table_md = _normalize_str(element.get("md"))
        raw_table_caption = _normalize_str(element.get("table_caption") or element.get("tableCaption"))
        table_caption = _normalize_table_caption(raw_table_caption)

        if not table_md and not table_caption:
            continue

        context_texts: List[str] = []
        context_element_ids: List[str] = []
        if max_previous_elements > 0:
            for candidate_index in range(element_index - 1, -1, -1):
                candidate_element = items[candidate_index]
                if not isinstance(candidate_element, dict):
                    continue

                candidate_type = str(candidate_element.get("type") or "").lower()
                if candidate_type in _TABLE_CONTEXT_BREAK_TYPES:
                    break
                if _is_heading_element(candidate_element):
                    break
                if candidate_type != "text":
                    continue

                candidate_text = _extract_text_element_content(candidate_element)
                if not candidate_text:
                    continue

                _, candidate_element_id_string = _resolve_element_id(
                    candidate_element,
                    element_index=candidate_index,
                )
                if not candidate_element_id_string:
                    continue

                context_texts.append(candidate_text)
                context_element_ids.append(candidate_element_id_string)
                if len(context_texts) >= max_previous_elements:
                    break

            context_texts.reverse()
            context_element_ids.reverse()

        collected_inputs.append(
            TableEnrichmentInput(
                element_id=element_id,
                element_id_string=element_id_string,
                table_md=table_md,
                raw_table_caption=raw_table_caption,
                table_caption=table_caption,
                context_texts=context_texts,
                context_element_ids=context_element_ids,
                source_element_ids=[element_id] if element_id is not None else [],
            )
        )

    return collected_inputs


def _generate_table_semantic_nodes(
    items: List[Dict[str, Any]],
    *,
    caption_generator: Optional[Callable[[str], str]] = None,
) -> List[PreparedTableSemanticNode]:
    collected_inputs = _collect_table_enrichment_input(items)
    if not collected_inputs:
        return []

    llm_enabled = RAG_SETTINGS.table_semantic_enrichment_llm_caption_enabled
    model_id = RAG_SETTINGS.table_semantic_enrichment_llm_caption_model_id
    max_context_bytes = RAG_SETTINGS.table_semantic_enrichment_llm_caption_prev_max_bytes
    max_table_bytes = RAG_SETTINGS.table_semantic_enrichment_llm_caption_table_max_bytes

    prepared_nodes: List[PreparedTableSemanticNode] = []
    for collected_input in collected_inputs:
        caption = str(collected_input.table_caption or "").strip()
        caption_source = "document" if caption else ""
        context_element_ids: List[str] = []

        if llm_enabled:
            prompt = _build_table_enrichment_prompt(
                context_texts=[
                    truncate_utf8(context_text, max_context_bytes)
                    for context_text in collected_input.context_texts
                    if context_text
                ],
                table_md=truncate_utf8(collected_input.table_md, max_table_bytes),
                existing_caption=truncate_utf8(collected_input.raw_table_caption, max_context_bytes),
            )
            try:
                raw_caption_text = (
                    caption_generator(prompt)
                    if caption_generator is not None
                    else _invoke_table_enrichment_llm(prompt, model_id=model_id)
                )
                generated_caption = _normalize_table_caption(raw_caption_text)
                if generated_caption:
                    caption = generated_caption
                    caption_source = "llm"
                    context_element_ids = list(collected_input.context_element_ids)
            except Exception as caption_error:
                logger.warning(
                    "表格语义增强 LLM 调用失败: file_element_id={}, error_type={}, error={}",
                    collected_input.element_id_string,
                    type(caption_error).__name__,
                    caption_error,
                )

        semantic_representation_text = add_spacing_for_tokenizer(caption)
        if not semantic_representation_text:
            continue

        prepared_nodes.append(
            PreparedTableSemanticNode(
                element_id=collected_input.element_id,
                element_id_string=collected_input.element_id_string,
                semantic_representation_text=semantic_representation_text,
                table_caption=caption,
                table_caption_source=caption_source,
                table_caption_context_element_ids=context_element_ids,
                source_element_ids=list(collected_input.source_element_ids),
            )
        )

    return prepared_nodes


async def _build_table_nodes(
    items: List[Dict[str, Any]]
) -> List[PreparedTableSemanticNode]:
    collected_inputs = _collect_table_enrichment_input(items)
    if not collected_inputs:
        return []

    llm_enabled = RAG_SETTINGS.table_semantic_enrichment_llm_caption_enabled
    model_id = RAG_SETTINGS.table_semantic_enrichment_llm_caption_model_id
    max_context_bytes = RAG_SETTINGS.table_semantic_enrichment_llm_caption_prev_max_bytes
    max_table_bytes = RAG_SETTINGS.table_semantic_enrichment_llm_caption_table_max_bytes

    prepared_nodes: List[PreparedTableSemanticNode] = []
    
    async def _create_node(collected_input):
        caption = str(collected_input.table_caption or "").strip()
        caption_source = "document" if caption else ""
        context_element_ids: List[str] = []

        if llm_enabled:
            prompt = _build_table_enrichment_prompt(
                context_texts=[
                    truncate_utf8(context_text, max_context_bytes)
                    for context_text in collected_input.context_texts
                    if context_text
                ],
                table_md=truncate_utf8(collected_input.table_md, max_table_bytes),
                existing_caption=truncate_utf8(collected_input.raw_table_caption, max_context_bytes),
            )
            try:
                raw_caption_text = await _enrich_table(prompt, model_id=model_id)
                generated_caption = _normalize_table_caption(raw_caption_text)
                if generated_caption:
                    caption = generated_caption
                    caption_source = "llm"
                    context_element_ids = list(collected_input.context_element_ids)
            except Exception as caption_error:
                logger.warning(
                    "表格语义增强 LLM 调用失败: file_element_id={}, error_type={}, error={}",
                    collected_input.element_id_string,
                    type(caption_error).__name__,
                    caption_error,
                )

        semantic_representation_text = add_spacing_for_tokenizer(caption)
        if not semantic_representation_text:
            return

        prepared_nodes.append(
            PreparedTableSemanticNode(
                element_id=collected_input.element_id,
                element_id_string=collected_input.element_id_string,
                semantic_representation_text=semantic_representation_text,
                table_caption=caption,
                table_caption_source=caption_source,
                table_caption_context_element_ids=context_element_ids,
                source_element_ids=list(collected_input.source_element_ids),
            )
        )
        
    # 并行处理所有表格节点创建任务
    tasks = [_create_node(collected_input) for collected_input in collected_inputs]
    await asyncio.gather(*tasks)

    return prepared_nodes

def _select_representative_child(candidates: List[Dict[str, Any]]) -> str:
    inf = float("inf")

    def _sort_key(candidate: Dict[str, Any]) -> tuple:
        strategy = candidate.get("split_strategy")
        row_start = candidate.get("table_row_start")
        row_end = candidate.get("table_row_end")
        has_valid_range = False
        row_start_value = inf
        row_end_value = inf
        if strategy == "table_row" and row_start is not None and row_end is not None:
            try:
                row_start_int = int(row_start)
                row_end_int = int(row_end)
                if row_start_int >= 0 and row_end_int >= row_start_int:
                    has_valid_range = True
                    row_start_value = float(row_start_int)
                    row_end_value = float(row_end_int)
            except (TypeError, ValueError):
                pass
        return (
            0 if has_valid_range else 1,
            row_start_value,
            row_end_value,
        )

    if not candidates:
        return ""
    return str(sorted(candidates, key=_sort_key)[0].get("child_id") or "")


def _build_representative_child_id_by_element_id(
    children: Sequence[ChildChunk],
) -> Dict[str, str]:
    candidate_children_by_element_id: Dict[str, List[Dict[str, Any]]] = {}
    for child_document in children or []:
        child_metadata = child_document
        source_element_ids = list(child_metadata.source_element_ids)
        child_id_text = child_document.child_id.strip()
        if not child_id_text:
            continue

        for source_element_id in source_element_ids:
            if source_element_id is None or isinstance(source_element_id, bool):
                continue
            element_id_string = str(source_element_id).strip()
            if not element_id_string:
                continue
            candidate_children_by_element_id.setdefault(element_id_string, []).append(
                {
                    "child_id": child_id_text,
                    "split_strategy": child_metadata.split_strategy,
                    "table_row_start": child_metadata.table_row_start,
                    "table_row_end": child_metadata.table_row_end,
                }
            )

    representative_child_id_by_element_id: Dict[str, str] = {}
    for element_id_string, candidates in candidate_children_by_element_id.items():
        representative_child_id_by_element_id[element_id_string] = _select_representative_child(candidates)
    return representative_child_id_by_element_id


def _bind_table_semantic_nodes(
    prepared_nodes: List[PreparedTableSemanticNode],
    *,
    file_id: str,
    children: Sequence[ChildChunk],
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
) -> List[GrandchildChunk]:
    if not prepared_nodes:
        return []

    representative_child_id_by_element_id = _build_representative_child_id_by_element_id(children)
    child_metadata_by_child_id: Dict[str, ChunkFields] = {}
    for child_document in children or []:
        child_metadata = child_document
        child_id_text = child_document.child_id.strip()
        if child_id_text:
            child_metadata_by_child_id[child_id_text] = child_metadata

    semantic_grandchildren: List[GrandchildChunk] = []
    for prepared_node in prepared_nodes:
        representative_child_id = representative_child_id_by_element_id.get(prepared_node.element_id_string)
        if not representative_child_id:
            raise ValueError(
                "表格语义增强缺少代表性 child: "
                f"file_id={file_id}, element_id={prepared_node.element_id_string}"
            )

        child_metadata = child_metadata_by_child_id.get(representative_child_id) or ChunkFields()
        heading_path = str(child_metadata.heading_path or "")
        heading_node_ids = list(child_metadata.heading_node_ids)

        semantic_chunk_strategy = "table_caption"
        grandchild_id_hash = hashlib.sha1(
            f"{representative_child_id}:{prepared_node.element_id_string}:"
            f"{prepared_node.semantic_representation_text[:200]}".encode("utf-8")
        ).hexdigest()[:12]
        grandchild_id = f"{file_id}:g:{semantic_chunk_strategy}:{grandchild_id_hash}"
        semantic_grandchildren.append(
            GrandchildChunk(
                grandchild_id=grandchild_id,
                parent_id="",
                child_id=representative_child_id,
                text=prepared_node.semantic_representation_text,
                file_id=str(file_id),
                folder_id=folder_id or "",
                file_name=file_name or "",
                type="table",
                source_element_ids=list(prepared_node.source_element_ids),
                element_id=prepared_node.element_id,
                heading_path=heading_path,
                heading_node_ids=list(heading_node_ids),
                table_caption_context_element_ids=list(prepared_node.table_caption_context_element_ids),
            )
        )

    return semantic_grandchildren


def build_table_semantic_nodes(
    items: List[Dict[str, Any]],
    *,
    file_id: str,
    children: Sequence[ChildChunk],
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
    caption_generator: Optional[Callable[[str], str]] = None,
    prepared_nodes: Optional[List[PreparedTableSemanticNode]] = None,
) -> List[GrandchildChunk]:
    if prepared_nodes is None:
        prepared_nodes = _generate_table_semantic_nodes(
            items,
            caption_generator=caption_generator,
        )

    return _bind_table_semantic_nodes(
        prepared_nodes,
        file_id=file_id,
        children=children,
        folder_id=folder_id,
        file_name=file_name,
    )


async def build_table_semantic_nodes_async(
    items: List[Dict[str, Any]],
    *,
    file_id: str,
    children: Sequence[ChildChunk],
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
    prepared_nodes: Optional[List[PreparedTableSemanticNode]] = None,
) -> List[GrandchildChunk]:
    if prepared_nodes is None:
        prepared_nodes = await _build_table_nodes(items)

    return _bind_table_semantic_nodes(
        prepared_nodes,
        file_id=file_id,
        children=children,
        folder_id=folder_id,
        file_name=file_name,
    )


def propagate_table_caption_metadata(
    *,
    children: Sequence[ChildChunk],
    grandchildren: Sequence[GrandchildChunk],
    semantic_grandchildren: Sequence[GrandchildChunk],
) -> None:
    # key: 表格元素ID，value: 包含关联上下文ID的字典
    caption_by_element_id: Dict[str, Dict[str, Any]] = {}

    for doc in semantic_grandchildren or []:
        metadata = doc
        element_id = metadata.element_id
        if element_id is None:
            continue
        context_ids = list(metadata.table_caption_context_element_ids)
        caption_by_element_id[str(element_id)] = {
            "table_caption_context_element_ids": list(context_ids),
        }

    # 如果没有提取到任何表格标题，直接返回无需传播
    if not caption_by_element_id:
        return

    def _apply(metadata: ChunkFields, element_id_text: str) -> None:
        payload = caption_by_element_id.get(element_id_text)
        if not payload:
            return
        if not metadata.table_caption_context_element_ids:
            metadata.table_caption_context_element_ids = list(payload["table_caption_context_element_ids"])

    for child_document in children or []:
        metadata = child_document
        source_element_ids = list(metadata.source_element_ids)

        table_candidates: List[str] = []
        for raw_id in source_element_ids:
            if raw_id is None or isinstance(raw_id, bool):
                continue
            raw_text = str(raw_id).strip()
            if raw_text and raw_text in caption_by_element_id:
                table_candidates.append(raw_text)
        unique_candidates = list(dict.fromkeys(table_candidates))
        if len(unique_candidates) == 1:
            _apply(metadata, unique_candidates[0])

    for grandchild_document in grandchildren or []:
        metadata = grandchild_document
        element_id = metadata.element_id
        if element_id is None:
            continue
        _apply(metadata, str(element_id))
