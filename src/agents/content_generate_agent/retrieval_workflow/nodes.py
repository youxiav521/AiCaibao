from __future__ import annotations

import asyncio
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json

from langgraph.types import Command

from .state import RetrieverState, SelectedContent, SelectedPiece
from .context_reduction_utils import build_item_key_to_item, resolve_piece_text
from .context_reduction_utils import dedupe_selected_content_list, filter_long_context_if_needed
from ..config import (
    create_default_configs,
    FORCE_ALL_TO_DOCUMENT_READING,
)
from .llm_retriever import LLMRetriever
from ..hierarchical_summary.tree_builder import Node
from ..utils import (
    apply_prompt_template,
    ainvoke_llm_with_structured_output,
    abatch_llm_with_structured_output,
    count_tokens,
    truncate_to_token_limit,
    ensure_within_token_limit,
    log_error_type_and_message,
    format_error_type_and_message,
)
from ..models import QuestionTask, ChunkIndexRetrievalTool, GenericDataSource, OtherDataSource
from src.applications.rag.services.summary_tree_service import load_summary_tree_if_available
from loguru import logger

# 统一上下文阈值配置
_CFG = create_default_configs()
_CTX_LIMIT = _CFG.context_token_limit
_ENABLE_SUMMARY_INDEX_RETRIEVAL = _CFG.enable_summary_index_retrieval
_ENABLE_DOCUMENT_READING_RETRIEVAL = _CFG.enable_document_reading_retrieval
_ENABLE_FILTER_LONG_CONTEXT = _CFG.enable_filter_long_context


def _get_retrieval_query_for_task(task: QuestionTask) -> str:
    """
    获取用于检索阶段的统一 query：
    - 若 QuestionTask.retrieval_query 已设置，则优先使用；
    - 否则回退为原始的 description。
    """
    retrieval_query = getattr(task, "retrieval_query", None)
    if retrieval_query:
        return retrieval_query

    logger.warning(f"【WARNING】 问题 {task.id} 未配置 retrieval_query，回退使用原始 description")
    return task.description


def _ensure_thinking_entry(thinking_map: Dict[str, Dict[str, Any]], question_id: str) -> Dict[str, Any]:
    """确保thinking_steps_map中存在对应问题的条目（不预填充任何阶段）。"""
    if question_id not in thinking_map:
        thinking_map[question_id] = {}
    return thinking_map[question_id]


def _pieces_to_resources(pieces: List[SelectedPiece], item_map: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """将SelectedPiece列表转换为资源列表（file_id + page）的工具函数。"""
    resources: List[Dict[str, Any]] = []
    for piece in pieces:
        if not piece.element_ids:
            continue
        first_element_id = piece.element_ids[0]
        key = (str(piece.file_id), first_element_id)
        item = item_map.get(key)
        page = getattr(item, "page", None) if item else None
        if page is None:
            logger.warning(f"[thinking_steps] 未找到 page: file_id={piece.file_id}, element_id={first_element_id}")
            continue
        resources.append({"file_id": str(piece.file_id), "page": int(page)})
    return resources


def _update_thinking_resources(
    thinking_map: Dict[str, Dict[str, Any]],
    question_id: str,
    stage: str,
    query: Optional[str],
    pieces: List[SelectedPiece],
    item_map: Dict[Any, Any],
) -> Dict[str, Dict[str, Any]]:
    """更新thinking_map中指定阶段的query和resources（只在阶段实际执行时创建条目）。"""
    entry = _ensure_thinking_entry(thinking_map, question_id)
    stage_key = stage if stage in ("chunk", "summary", "document") else None
    # 保守检查
    if stage_key is None:
        logger.error(f"[thinking_steps] 阶段{stage}不存在，不更新thinking_steps: question_id={question_id}")
        return thinking_map
    
    # 如果阶段的thinking_steps尚不存在，则创建阶段条目
    if stage_key not in entry:
        entry[stage_key] = {"query": None, "resources": [], "analysis": ""}
    
    if query is not None:
        entry[stage_key]["query"] = query
    entry[stage_key]["resources"].extend(_pieces_to_resources(pieces, item_map))
    return thinking_map


def _update_thinking_analysis(
    thinking_map: Dict[str, Dict[str, Any]],
    question_id: str,
    current_retrieval_phase: str,
    reasoning: str,
) -> Dict[str, Dict[str, Any]]:
    """根据阶段写入对应的analysis（只在阶段实际执行时创建条目）。"""
    entry = _ensure_thinking_entry(thinking_map, question_id)
    if current_retrieval_phase == "chunk":
        # chunk后的judge结果写入summary的analysis
        if "summary" not in entry:
            entry["summary"] = {"query": None, "resources": [], "analysis": ""}
        entry["summary"]["analysis"] = reasoning
    elif current_retrieval_phase == "summary":
        # summary后的judge结果写入document的analysis
        if "document" not in entry:
            entry["document"] = {"query": None, "resources": [], "analysis": ""}
        entry["document"]["analysis"] = reasoning
    return thinking_map


def merge_selected_content(existing_content: List[SelectedContent], new_content: List[SelectedContent]) -> List[SelectedContent]:
    """合并两个SelectedContent列表，将相同问题的selected_pieces合并。
    
    Args:
        existing_content: 现有的SelectedContent列表
        new_content: 新的SelectedContent列表
        
    Returns:
        合并后的SelectedContent列表
    """
    # 创建问题到SelectedContent的映射
    question_to_content: Dict[str, SelectedContent] = {}
    
    # 处理现有的内容
    for content in existing_content:
        question_to_content[content.question] = content
    
    # 处理新内容，合并相同问题的selected_pieces
    for content in new_content:
        if content.question in question_to_content:
            # 如果问题已存在，合并selected_pieces
            existing_content_obj = question_to_content[content.question]
            merged_pieces = existing_content_obj.selected_pieces + content.selected_pieces
            question_to_content[content.question] = SelectedContent(
                question=content.question,
                selected_pieces=merged_pieces
            )
        else:
            # 如果问题不存在，直接添加
            question_to_content[content.question] = content
    
    # 返回合并后的列表
    return list(question_to_content.values())


def merge_selected_content_by_id(existing_content: List[SelectedContent], new_content: List[SelectedContent]) -> List[SelectedContent]:
    """基于question_id合并SelectedContent列表，避免相同question_text的问题被合并。
    
    Args:
        existing_content: 现有的SelectedContent列表
        new_content: 新的SelectedContent列表
        
    Returns:
        合并后的SelectedContent列表
    """
    # 创建question_id到SelectedContent的映射
    id_to_content: Dict[str, SelectedContent] = {}
    
    # 处理现有的内容
    for content in existing_content:
            id_to_content[content.question_id] = content
    
    # 处理新内容，合并相同question_id的selected_pieces
    for content in new_content:
        key = content.question_id
        if key in id_to_content:
            # 如果问题已存在，合并selected_pieces
            existing_content_obj = id_to_content[key]
            merged_pieces = existing_content_obj.selected_pieces + content.selected_pieces
            id_to_content[key] = SelectedContent(
                question=content.question,
                question_id=content.question_id,
                selected_pieces=merged_pieces
            )
        else:
            # 如果问题不存在，直接添加
            id_to_content[key] = content
    
    # 返回合并后的列表
    return list(id_to_content.values())


class ChapterSelection(BaseModel):
    """单个章节选择结果"""
    chapter_index: int = Field(..., description="章节索引（从1开始）")
    relevance_score: int = Field(..., ge=1, le=10, description="相关性评分（1-10分）")
    selection_reason: str = Field(..., description="选择理由，尽量简短")


class ChapterSelectionResponse(BaseModel):
    """LLM返回的章节选择结果"""
    selected_chapters: List[ChapterSelection] = Field(..., description="选中的章节列表")


class BatchChapterSelectionForQuestion(BaseModel):
    """批量章节选择中单个问题的结果"""
    question_index: int = Field(..., description="问题序号（从0开始，对应输入问题列表的索引）")
    selected_chapters: List[ChapterSelection] = Field(..., description="该问题选中的章节列表")


class BatchChapterSelectionResponse(BaseModel):
    """LLM返回的批量章节选择结果，一次调用处理多个问题。
    
    通过将多个问题放在同一个 prompt 中发送，使大模型可以复用
    文档结构（titles_text）的 KV cache，减少重复计算。
    """
    results: List[BatchChapterSelectionForQuestion] = Field(..., description="各问题的章节选择结果")


class ContentSufficiencyResponse(BaseModel):
    """LLM返回的内容充分性判断结果"""
    had_enough_content: bool = Field(..., description="是否有足够的内容回答问题")
    reasoning: str = Field(..., description="判断理由")


def _collect_leaf_nodes(node: Node) -> List[Node]:
    """后序遍历收集所有叶子节点"""
    leaf_nodes = []
    
    def _post_order_traverse(current_node: Node):
        if not current_node.childs:  # 叶子节点
            leaf_nodes.append(current_node)
        else:  # 非叶子节点，递归处理子节点
            for child in current_node.childs:
                _post_order_traverse(child)
    
    _post_order_traverse(node)
    return leaf_nodes


def _build_selected_piece_from_leaf_node(
    leaf_node: Node,
    *,
    chapter_index: int,
    relevance_score: int,
    selection_reason: str,
) -> SelectedPiece:
    node_items = getattr(leaf_node, "content", []) or []
    file_id = str(getattr(leaf_node, "file_id", "") or "").strip()
    element_ids: list[int] = []

    for item in node_items:
        element_id = getattr(item, "element_id", None)
        current_file_id = getattr(item, "file_id", None)
        if element_id is not None:
            element_ids.append(int(element_id))
        if current_file_id is not None:
            current_file_id = str(current_file_id)
            if not file_id:
                file_id = current_file_id
            elif file_id != current_file_id:
                raise ValueError(
                    f"summary_index_retrieval_core 期望叶节点来自同一文件，但检测到多个 file_id: {file_id} 和 {current_file_id}"
                )

    if not element_ids:
        raw_source_element_ids = getattr(leaf_node, "source_element_ids", []) or []
        element_ids = [
            int(element_id)
            for element_id in raw_source_element_ids
            if isinstance(element_id, int) or (isinstance(element_id, str) and element_id.strip().isdigit())
        ]

    doc_id = str(getattr(leaf_node, "parent_id", "") or f"chapter_{chapter_index + 1}")
    return SelectedPiece(
        doc_id=doc_id,
        element_ids=element_ids,
        file_id=file_id,
        relevance_score=relevance_score,
        selection_reason=selection_reason,
        source_stage="summary",
    )


async def chunk_index_retrieval_core(
    question_tasks: List[QuestionTask],
    origin_content: List,
    top_k: int = 30,
    queries: Optional[List[str]] = None, 
    file_ids: Optional[List[str]] = None,
    return_child_chunks: bool = True,
    rerank_enabled: bool = False,
    hybrid_enabled: bool = True,
) -> tuple[List[SelectedContent], Dict[str, str]]:
    """chunk检索的核心逻辑，返回检索结果和实际使用的query映射。
    
    Args:
        question_tasks: QuestionTask对象列表
        origin_content: 原始内容列表（Item对象列表）
        top_k: 检索返回的top_k数量，默认30
        queries: 检索查询文本；为 None 时使用 QuestionTask.retrieval_query 或 description
        file_ids: 检索文件ID列表；为 None 时使用 origin_content 中的唯一 file_id
        return_child_chunk: 是否返回子chunk，默认True
        rerank_enabled: 是否启用重排序，默认False
        hybrid_enabled: 是否启用混合检索，默认True
        
    Returns:
        tuple[List[SelectedContent], Dict[str, str]]: (检索结果列表, question_id到实际query的映射)
    """
    from src.applications.rag.services.query_service import run_batch_query
    from src.applications.rag.schemas import BatchQueryRequest, QuestionItem, BatchQueryItem
    
    logger.info(f"chunk_index_retrieval_core开始检索，问题数量: {len(question_tasks)}")
    logger.info(f"[阶段标识] 进入检索阶段: chunk_index_retrieval_core, 问题列表: {[task.id for task in question_tasks]}")
    
    # 1. 构建输入
    if file_ids:
        file_ids = [str(fid) for fid in file_ids if str(fid).strip()]
        logger.info(f"使用显式传入的 file_ids 作为检索范围，共 {len(file_ids)} 个: {file_ids}")
    else:
        file_ids = list({str(item.file_id) for item in origin_content})
        logger.info(f"提取到 {len(file_ids)} 个唯一的file_id: {file_ids}")
    
    # 构建问题列表，优先使用传入的 queries；如果没有提供 queries，则使用 QuestionTask.retrieval_query 或 description
    questions_list = []
    if queries is None:
        questions_list = [
            QuestionItem(
                question_id=task.id,
                question_text=_get_retrieval_query_for_task(task),
            )
            for task in question_tasks
        ]
    else:
        logger.info(f"使用传入的 queries: {queries}")
        questions_list = [
            QuestionItem(question_id=task.id, question_text=query)
            for task, query in zip(question_tasks, queries)
        ]
    
    # 构建BatchQueryRequest对象
    batch_request = BatchQueryRequest(
        questions=questions_list,
        top_k=top_k,
        file_ids=file_ids,
        return_child_chunks=return_child_chunks,
        rerank_enabled=rerank_enabled,
        hybrid_enabled=hybrid_enabled,
    )
    logger.info(f"构建RAG输入，问题数量: {len(batch_request.questions)}, 文件数量: {len(file_ids)}, top_k: {top_k}")
    
    # 2. 调用RAG接口
    retrieval_results = await run_batch_query(body=batch_request)
    # 转化为json用于输出
    # json_retrieval_results = [result.model_dump() for result in retrieval_results]
    # logger.info(f"RAG接口调用完成，检索结果: {json_retrieval_results}") # TODO: 后续需要移除

    # 3. 处理输出，转换为SelectedContent格式
    all_selected_content = []
    
    # 按question_id分组结果
    results_by_question_id: Dict[str, List[BatchQueryItem]] = {}
    for result in retrieval_results:
        # BatchQueryItem 是 Pydantic BaseModel 对象，使用属性访问
        qid = result.question_id
        if qid:
            if qid not in results_by_question_id:
                results_by_question_id[qid] = []
            results_by_question_id[qid].append(result)
    
    # 为每个问题创建SelectedContent
    for task in question_tasks:
        question_id = task.id
        question_text = task.description
        query = None
        for question_item in questions_list:
            if question_item.question_id == question_id:
                query = question_item.question_text
                break
        if query is None:
            logger.error(f"问题 '{question_id}' 没有检索用的query?请检查问题配置！")
            raise ValueError(f"问题 '{question_id}' 没有检索用的query?请检查问题配置！")

        # 获取该问题的检索结果
        question_results = results_by_question_id.get(question_id, [])
        
        if not question_results:
            logger.warning(f"问题 '{query}' (ID: {question_id}) 没有检索到任何结果")
            continue
        
        logger.info(f"query: '{query}' (ID: {question_id}) 检索到 {len(question_results)} 个结果")
        # 转换为SelectedPiece列表
        selected_pieces = []
        for result in question_results:
            # BatchQueryItem 是 Pydantic BaseModel 对象，使用属性访问
            element_ids = result.element_ids
            file_id = result.file_id
            # 处理 score：可能为 None，需要除以100得到0-10之间的分数，并转换为整数
            relevance_score = None
            if result.score is not None:
                score_float = result.score / 100
                # 四舍五入并限制在0-10范围内，然后转换为整数
                relevance_score = int(round(max(0, min(10, score_float))))
            # 不再使用 result.text，通过 file_id + element_ids 回查获取
            
            selected_piece = SelectedPiece(
                doc_id=f"{question_id}_RAGChunk_{len(selected_pieces)}",
                element_ids=element_ids,
                file_id=file_id,
                relevance_score=relevance_score,
                selection_reason="无",  # chunk检索通常不提供选择理由
                source_stage="chunk",
                raw_text=(result.text or None) if (not file_id or not element_ids) else None,
            )
            selected_pieces.append(selected_piece)
        
        # 创建SelectedContent对象
        if selected_pieces:
            selected_content = SelectedContent(
                question=question_text,  # 使用question_text
                question_id=question_id,
                selected_pieces=selected_pieces
            )
            all_selected_content.append(selected_content)
    
    # 构建question_id到实际query的映射
    query_map = {item.question_id: item.question_text for item in questions_list}
    
    logger.info(f"chunk_index_retrieval_core完成，共处理 {len(all_selected_content)} 个问题的检索结果")
    return all_selected_content, query_map


async def chunk_index_retrival(state: RetrieverState) -> Command[str]:
    """利用chunk索引进行一次检索。

    通过RAG接口进行向量检索，获取与问题相关的文档片段。
    这是原有流程的wrapper函数，调用core函数并处理状态更新。
    """
    start_time = time.perf_counter()
    unfinished_question_tasks = state["unfinished_question_tasks"]
    origin_content = state["origin_content"]

    logger.info(f"chunk_index_retrival开始检索，当前未完成问题数量: {len(unfinished_question_tasks)}")

    try:
        # 调用core函数执行检索，获取检索结果和实际使用的query映射
        all_selected_content, query_map = await chunk_index_retrieval_core(
            question_tasks=unfinished_question_tasks,
            origin_content=origin_content,
            top_k=30
        )

        # 合并到state的selected_content中
        merged_content = merge_selected_content_by_id(state["selected_content"], all_selected_content)
        # chunk后无需去重，因为chunk的检索结果是第一个处理的，不会出现重复的element_ids
        merged_content = dedupe_selected_content_list(merged_content) # TODO: 后续可以移除

        # 更新thinking_steps_map（使用实际使用的query）
        thinking_steps_map = dict(state.get("thinking_steps_map", {}))
        item_map = build_item_key_to_item(origin_content)
        for content in all_selected_content:
            query = query_map.get(content.question_id)
            thinking_steps_map = _update_thinking_resources(
                thinking_steps_map,
                content.question_id,
                "chunk",
                query,
                content.selected_pieces,
                item_map,
            )

        elapsed_time = time.perf_counter() - start_time
        node_timings = dict(state.get("node_timings", {}))
        node_timings["chunk_index_retrival"] = elapsed_time

        update = {
            "selected_content": merged_content,
            "chunk_index_retrival_done": True,
            "thinking_steps_map": thinking_steps_map,
            "node_timings": node_timings,
        }

        logger.info(f"chunk_index_retrival完成，共处理 {len(all_selected_content)} 个问题的检索结果，耗时: {elapsed_time:.3f}s")

        return Command(update=update, goto="judge_if_continue")

    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        node_timings = dict(state.get("node_timings", {}))
        node_timings["chunk_index_retrival"] = elapsed_time

        log_error_type_and_message(
            logger=logger,
            context="[chunk_index_retrival] RAG调用过程中发生未知错误，跳过此阶段继续流程",
            e=e,
            exc_info=True,
            max_message_chars=200,
        )
        # 错误处理：记录错误后跳过此阶段，继续流程
        # 保持原有的selected_content不变，标记chunk_index_retrival_done为True，继续到下一节点
        update = {
            "selected_content": state.get("selected_content", []),
            "chunk_index_retrival_done": True,
            "thinking_steps_map": state.get("thinking_steps_map", {}),
            "node_timings": node_timings,
        }
        return Command(update=update, goto="judge_if_continue")


async def judge_if_continue(state: RetrieverState) -> Command[str]:
    """判断是否需要继续下一阶段检索。

    使用LLM评估当前已检索到的内容是否足以回答问题。
    如果内容充分，则结束检索流程；否则继续下一阶段检索。

    流式处理模式：检测到finished_question时进行标记
    """
    start_time = time.perf_counter()
    unfinished_question_tasks = state["unfinished_question_tasks"]
    selected_content = state["selected_content"]
    origin_content = state["origin_content"]
    model_id = state["model_id"]
    # 使用专门的judge模型，如果没有则fallback到普通模型
    model_id_for_judge = state.get("model_id_for_judge", model_id)
    answer_lang = state.get("answer_lang", "zh_CN") or "zh_CN"
    # 获取完整 prompt 的最大 token 限制
    context_token_limit = state.get("context_token_limit", _CTX_LIMIT)  # 优先从 state 中获取，如果没有则使用默认值
    
    # 构建 item_map 用于回查文本
    item_map = build_item_key_to_item(origin_content)
    thinking_steps_map = dict(state.get("thinking_steps_map", {}))
    node_timings = dict(state.get("node_timings", {}))

    # 确定当前是第几次 judge 调用
    judge_count = sum(1 for k in node_timings.keys() if k.startswith("judge_if_continue#")) + 1
    judge_key = f"judge_if_continue#{judge_count}"

    logger.info(f"judge_if_continue开始评估（第{judge_count}次），当前未完成问题数量: {len(unfinished_question_tasks)}")
    logger.info(
        f"使用judge模型: {model_id_for_judge}, 上下文限制: {context_token_limit} tokens, "
        f"answer_lang={answer_lang}"
    )
    
    # 确定当前判断阶段：chunk后、summary后、还是document_reading后
    current_retrieval_phase = "unknown"
    if state.get("document_reading_retrival_done", False):
        current_retrieval_phase = "document_reading"
    elif state.get("summary_index_retrival_done", False):
        current_retrieval_phase = "summary"
    elif state.get("chunk_index_retrival_done", False):
        current_retrieval_phase = "chunk"
    
    logger.info(f"[阶段标识] 当前判断阶段: {current_retrieval_phase}后")

    # ========= 卫语句：可确定直接结束的场景，跳过无意义的筛选/LLM调用以节省token =========
    def _end_with_remaining_questions(*, config_msg: str, phase_msg: str) -> Command[str]:
        remaining_ids = [t.id for t in unfinished_question_tasks if getattr(t, "id", None)]
        elapsed_time = time.perf_counter() - start_time
        node_timings[judge_key] = elapsed_time
        logger.info(
            f"{config_msg}"
            f"judge_if_continue 直接结束检索流程（跳过上下文筛选与LLM评估），将 {len(remaining_ids)} 个问题按当前已有上下文产出"
            f"，耗时: {elapsed_time:.3f}s"
        )
        for qid in remaining_ids:
            logger.info(
                f"[问题路径分析] question_id={qid}, 阶段={phase_msg}, 使用当前上下文回答, 已加入finished_question_ids"
            )
        return Command(
            update={
                "unfinished_question_tasks": [],
                "thinking_steps_map": thinking_steps_map,
                "finished_question_ids": remaining_ids,
                "selected_content": selected_content,
                "node_timings": node_timings,
            },
            goto="__end__",
        )

    # 1) 配置跳过 summary：chunk 后直接结束检索流程（即便后续还有 judge 逻辑也不再需要）
    if not _ENABLE_SUMMARY_INDEX_RETRIEVAL:
        return _end_with_remaining_questions(
            config_msg="[配置] ENABLE_SUMMARY_INDEX_RETRIEVAL=False，",
            phase_msg="chunk后(跳过summary)",
        )

    # 2) summary 已完成但配置跳过 document_reading：直接结束
    if (
        state.get("summary_index_retrival_done", False)
        and not _ENABLE_DOCUMENT_READING_RETRIEVAL
    ):
        return _end_with_remaining_questions(
            config_msg="[配置] ENABLE_DOCUMENT_READING_RETRIEVAL=False，",
            phase_msg="summary后(跳过document_reading)",
        )

    # 在模型调用前，按配置决定是否筛选超过 context_token_limit 的上下文
    filter_start_time = time.perf_counter()
    if _ENABLE_FILTER_LONG_CONTEXT:
        logger.info(f"[judge_if_continue] 在模型调用前，开始筛选上下文")
        question_ids = [task.id for task in unfinished_question_tasks if getattr(task, "id", None)]
        selected_content = await filter_long_context_if_needed(
            selected_content=selected_content,
            question_ids=question_ids,
            origin_content=origin_content,
            model_id=model_id,
            token_threshold=context_token_limit,
        )
    else:
        logger.info(f"[judge_if_continue] enable_filter_long_context=False，跳过上下文筛选")
    filter_elapsed = time.perf_counter() - filter_start_time
    node_timings[f"filter_context_in_{judge_key}"] = filter_elapsed
    
    # 为每个问题评估内容充分性（仅按 ID 精确对应）
    finished_question_ids: List[str] = []

    # 允许开关：是否对 LLM 调用使用批量处理
    use_batch_for_llm: bool = state.get("use_batch_for_llm", False)
    
    logger.info(f"judge_if_continue 节点，是否开启批量调用: {use_batch_for_llm}")

    # 预构建 prompts 与映射
    prompts: List[str] = []
    prompt_meta: List[Dict[str, str]] = []  # {question, qid}

    for task in unfinished_question_tasks:
        qid = task.id
        retrieval_question = _get_retrieval_query_for_task(task)
        logger.info(f"开始评估问题: question_id={qid}, query='{retrieval_question[:50]}...'")
        
        # 构建该问题对应的已检索内容：仅按 question_id 匹配
        question_content: List[str] = []
        for content in selected_content:
            if getattr(content, "question_id", None) != qid:
                continue
            for piece in content.selected_pieces:
                # 通过回查机制获取文本
                piece_text = resolve_piece_text(piece, item_map)
                question_content.append(
                    f"文档ID: {piece.doc_id}\n内容: {piece_text}\n相关性评分: {piece.relevance_score}\n选择理由: {piece.selection_reason}\n"
                )
        
        if not question_content:
            logger.warning(f"问题 '{retrieval_question}' 没有检索到任何相关内容")
            logger.info(f"[问题路径分析] question_id={qid}, 阶段={current_retrieval_phase}后, 判断结果=不足(无内容), 原因=没有检索到任何相关内容")
            continue
        
        logger.info(f"问题 '{retrieval_question}' 检索到 {len(question_content)} 个内容片段")
        
        # 构建prompt
        selected_content_text = "\n".join(question_content)

        # 计算空模板的 token 开销（仅包含指令与问题，不含 selected_content）
        base_prompt = apply_prompt_template(
            "content_sufficiency",
            {
                "question": retrieval_question,
                "selected_content": "",
            },
            locale=answer_lang,
        )
        base_tokens = count_tokens(base_prompt)
        available_tokens_for_context = context_token_limit - base_tokens
        if available_tokens_for_context <= 0:
            logger.error(
                f"[judge_if_continue] content_sufficiency 基础prompt token数({base_tokens})已超过限制({context_token_limit})，"
                f"question_id={qid}，将不附加任何检索内容。"
            )
            safe_selected_content_text = ""
        else:
            # 在“可用于上下文的 token 预算”内截断 selected_content
            safe_selected_content_text = ensure_within_token_limit(
                selected_content_text, max_tokens=available_tokens_for_context
            )

        prompt = apply_prompt_template(
            "content_sufficiency",
            {
                "question": retrieval_question,
                "selected_content": safe_selected_content_text,
            },
            locale=answer_lang,
        )

        if use_batch_for_llm:
            prompts.append(prompt)
            prompt_meta.append({"question": retrieval_question, "qid": qid})
        else:
            # 非批量模式使用 model_id_for_judge
            try:
                response = await ainvoke_llm_with_structured_output(
                    prompt=prompt,
                    model_id=model_id_for_judge,  # 使用judge专用模型
                    response_type=ContentSufficiencyResponse
                )
                thinking_steps_map = _update_thinking_analysis(
                    thinking_steps_map,
                    qid,
                    current_retrieval_phase,
                    response.reasoning,
                )
                if not response.had_enough_content:
                    logger.info(f"问题 '{question}' 内容不足: {response.reasoning}")
                    logger.info(f"[问题路径分析] question_id={qid}, 阶段={current_retrieval_phase}后, 判断结果=不足, 原因={response.reasoning[:200]}")
                else:
                    logger.info(f"问题 '{question}' 内容充足: {response.reasoning}")
                    logger.info(f"[问题路径分析] question_id={qid}, 阶段={current_retrieval_phase}后, 判断结果=充足, 原因={response.reasoning[:200]}")
                    if not FORCE_ALL_TO_DOCUMENT_READING:
                        finished_question_ids.append(qid)
                    else:
                        logger.info(f"[FORCE_ALL_TO_DOCUMENT_READING] 暂不标记完成，强制进入下一阶段: question_id={qid}")
            except Exception as e:
                log_error_type_and_message(
                    logger=logger,
                    context=f"内容充分性评估过程中出错: question_id={qid}, 阶段={current_retrieval_phase}",
                    e=e,
                    exc_info=True,
                    max_message_chars=200,
                )
                err_type, err_msg = format_error_type_and_message(
                    e, max_message_chars=200
                )
                logger.error(
                    f"[问题路径分析] question_id={qid}, 阶段={current_retrieval_phase}后, 判断结果=错误, 原因=评估过程出错: {err_type}: {err_msg}"
                )

    # 批量路径：一次性调用并处理所有响应
    if use_batch_for_llm and prompts:
        try:
            responses = await abatch_llm_with_structured_output(
                prompts=prompts,
                model_id=model_id,
                response_type=ContentSufficiencyResponse,
            )
            for meta, resp in zip(prompt_meta, responses):
                question = meta["question"]
                qid = meta["qid"]
                thinking_steps_map = _update_thinking_analysis(
                    thinking_steps_map,
                    qid,
                    current_retrieval_phase,
                    resp.reasoning,
                )
                if not resp.had_enough_content:
                    logger.info(f"问题 '{question}' 内容不足: {resp.reasoning}")
                    logger.info(f"[问题路径分析] question_id={qid}, 阶段={current_retrieval_phase}后, 判断结果=不足, 原因={resp.reasoning[:200]}")
                else:
                    logger.info(f"问题 '{question}' 内容充足: {resp.reasoning}")
                    logger.info(f"[问题路径分析] question_id={qid}, 阶段={current_retrieval_phase}后, 判断结果=充足, 原因={resp.reasoning[:200]}")
                    if not FORCE_ALL_TO_DOCUMENT_READING:
                        finished_question_ids.append(qid)
                    else:
                        logger.info(f"[FORCE_ALL_TO_DOCUMENT_READING] 暂不标记完成，强制进入下一阶段: question_id={qid}")
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context=f"批量内容充分性评估出错: 阶段={current_retrieval_phase}, model_id={model_id}",
                e=e,
                exc_info=True,
                max_message_chars=200,
            )
            # 出错时不标记完成，默认继续检索
            # 为批量处理中的问题记录错误
            err_type, err_msg = format_error_type_and_message(
                e, max_message_chars=200
            )
            for meta in prompt_meta:
                qid = meta["qid"]
                logger.error(
                    f"[问题路径分析] question_id={qid}, 阶段={current_retrieval_phase}后, 判断结果=错误, 原因=批量评估出错: {err_type}: {err_msg}"
                )
            
    
    # 决定下一步：先算出剩余未完成任务，再决定 goto
    remaining_question_tasks = [t for t in unfinished_question_tasks if t.id not in finished_question_ids]

    # 注意：跳过 summary / 跳过 document_reading 的场景已在函数前部用卫语句直接 return，避免无意义的筛选与LLM调用。
    if len(finished_question_ids) == len(unfinished_question_tasks):
        goto = "__end__"
    elif state.get("summary_index_retrival_done", False) is True:
        # summary 已完成：此处意味着 enable_document_reading_retrieval=True（否则已在卫语句提前结束）
        goto = "document_reading_retrival"
    else:
        goto = "summary_index_retrival"
    
    # 记录节点耗时
    elapsed_time = time.perf_counter() - start_time
    node_timings[judge_key] = elapsed_time

    # 构建更新状态
    update = {
        "unfinished_question_tasks": remaining_question_tasks,
        "thinking_steps_map": thinking_steps_map,
        "node_timings": node_timings,
    }
    
    # 流式处理模式：如果有finished_question，则标记到状态中供消费者使用
    if finished_question_ids:
        logger.info(f"流式模式下，将finished_question_ids添加到状态: {finished_question_ids}")
        # 为每个finished_question记录最终完成状态
        for qid in finished_question_ids:
            logger.info(f"[问题路径分析] question_id={qid}, 阶段={current_retrieval_phase}后, 判断结果=充足(已完成), 已加入finished_question_ids")
        update["finished_question_ids"] = finished_question_ids
        # 并不是真正的更新，但是需要传出selected_content供消费者使用
        update["selected_content"] = selected_content
        logger.info(f"已产出finished_question资源，等待消费者处理")
    
    logger.info(f"本次judge_if_continue完成，剩余未完成问题: {len(remaining_question_tasks)}，耗时: {elapsed_time:.3f}s")
    
    return Command(update=update, goto=goto)


async def summary_index_retrieval_core(
    question_tasks: List[QuestionTask],
    origin_content: List,
    model_id: str,
    summary_tree_provider: Optional[Any] = None,
    answer_lang: str = "zh_CN",
    use_batch_for_llm: bool = False,
    score_threshold: int = 6
) -> tuple[List[SelectedContent], Optional[Node]]:
    """summary检索的核心逻辑，返回检索结果和摘要树（用于缓存）。
    
    Args:
        question_tasks: QuestionTask对象列表
        origin_content: 原始内容列表（Item对象列表）
        model_id: 模型ID
        summary_tree_provider: 可选的请求内摘要树 provider，用于避免同一次请求重复读向量库
        use_batch_for_llm: 是否使用批量LLM调用
        score_threshold: 章节选择分数阈值，默认6
        
    Returns:
        tuple[List[SelectedContent], Optional[Node]]: (检索结果列表, 摘要树（用于缓存）)
    """
    if summary_tree_provider is not None:
        root = await summary_tree_provider(origin_content)
    else:
        root = await load_summary_tree_if_available(origin_content)

    if root is None:
        logger.info("summary tree unavailable, skip summary retrieval")
        return [], None

    # 1. 后序遍历收集所有叶子节点
    leaf_nodes = _collect_leaf_nodes(root)
    
    # 2. 构建章节摘要结构 [chapter {i}] {summary_i}
    chapter_summaries = []
    for i, leaf_node in enumerate(leaf_nodes, 1):
        if leaf_node.summary:
            chapter_summaries.append(f"[chapter {i}] {leaf_node.summary}")
    
    # 构建章节摘要文本（后续按完整 prompt 的 token 限制进行截断）
    chapters_text = "\n\n".join(chapter_summaries)

    # 为每个问题执行章节选择
    all_selected_content = []
    
    logger.info(f"[阶段标识] 进入检索阶段: summary_index_retrieval_core, 问题列表: {[task.id for task in question_tasks]}")
    logger.info(
        f"summary_index_retrieval_core 节点，是否开启批量调用: {use_batch_for_llm}, "
        f"answer_lang={answer_lang}"
    )

    if use_batch_for_llm:
        # 批量路径：构建所有问题的prompts，一次性调用
        prompts = []
        prompt_meta = []  # 存储每个prompt对应的元数据
        
        for task in question_tasks:
            retrieval_question = _get_retrieval_query_for_task(task)

            # 计算空模板的 token 开销（仅包含指令与question，不含 chapters_text）
            base_prompt = apply_prompt_template(
                "chapter_selection",
                {
                    "question": retrieval_question,
                    "chapters_text": "",
                },
                locale=answer_lang,
            )
            base_tokens = count_tokens(base_prompt)
            available_tokens_for_context = _CTX_LIMIT - base_tokens
            if available_tokens_for_context <= 0:
                logger.error(
                    "[summary_index_retrieval_core] chapter_selection 基础prompt token数"
                    f"({base_tokens})已超过限制({_CTX_LIMIT})，question_id={task.id}，将不附加任何章节摘要内容。"
                )
                safe_chapters_text = ""
            else:
                # 在“可用于上下文的 token 预算”内截断章节摘要文本
                safe_chapters_text = ensure_within_token_limit(
                    chapters_text, max_tokens=available_tokens_for_context
                )

            prompt = apply_prompt_template(
                "chapter_selection",
                {
                    "question": retrieval_question,
                    "chapters_text": safe_chapters_text,
                },
                locale=answer_lang,
            )
            prompts.append(prompt)
            prompt_meta.append(
                {
                    "question": retrieval_question,
                    "task": task,
                }
            )
        
        if prompts:
            try:
                # 批量调用LLM
                responses = await abatch_llm_with_structured_output(
                    prompts=prompts,
                    model_id=model_id,
                    response_type=ChapterSelectionResponse
                )
                
                # 处理每个响应
                for meta, response in zip(prompt_meta, responses):
                    task = meta["task"]
                    question_text = task.description  # 用于后续 SelectedContent.question
                    question_id = task.id
                    
                    # 筛选出分数大于等于score_threshold的章节
                    selected_chapters = []
                    for chapter_selection in response.selected_chapters:
                        if chapter_selection.relevance_score >= score_threshold:
                            # 获取对应的叶子节点
                            chapter_index = chapter_selection.chapter_index - 1  # 转换为0基索引
                            if 0 <= chapter_index < len(leaf_nodes):
                                selected_piece = _build_selected_piece_from_leaf_node(
                                    leaf_nodes[chapter_index],
                                    chapter_index=chapter_index,
                                    relevance_score=chapter_selection.relevance_score,
                                    selection_reason=chapter_selection.selection_reason,
                                )
                                selected_chapters.append(selected_piece)
                    
                    # 创建SelectedContent对象
                    if selected_chapters:
                        selected_content = SelectedContent(
                            question=question_text,
                            question_id=question_id,
                            selected_pieces=selected_chapters,
                        )
                        all_selected_content.append(selected_content)
                        
            except Exception as e:
                log_error_type_and_message(
                    logger=logger,
                    context=f"批量章节选择过程中出错: model_id={model_id}",
                    e=e,
                    exc_info=True,
                    max_message_chars=200,
                )
                # 出错时继续处理，不中断流程
    else:
        # 非批量路径：逐个问题处理（原有逻辑）
        for task in question_tasks:
            question_text = task.description  # 用于后续 SelectedContent.question
            question_id = task.id
            retrieval_question = _get_retrieval_query_for_task(task)
            
            # 3. 在完整 prompt 的 token 限制下构建章节选择 prompt
            base_prompt = apply_prompt_template(
                "chapter_selection",
                {
                    "question": retrieval_question,
                    "chapters_text": "",
                },
                locale=answer_lang,
            )
            base_tokens = count_tokens(base_prompt)
            available_tokens_for_context = _CTX_LIMIT - base_tokens
            if available_tokens_for_context <= 0:
                logger.error(
                    "[summary_index_retrieval_core] chapter_selection 基础prompt token数"
                    f"({base_tokens})已超过限制({_CTX_LIMIT})，question_id={question_id}，将不附加任何章节摘要内容。"
                )
                safe_chapters_text = ""
            else:
                safe_chapters_text = ensure_within_token_limit(
                    chapters_text, max_tokens=available_tokens_for_context
                )

            prompt = apply_prompt_template(
                "chapter_selection",
                {
                    "question": retrieval_question,
                    "chapters_text": safe_chapters_text,
                },
                locale=answer_lang,
            )

            try:
                # 使用结构化输出调用LLM
                response = await ainvoke_llm_with_structured_output(
                    prompt=prompt,
                    model_id=model_id,
                    response_type=ChapterSelectionResponse
                )
                
                # 4. 筛选出分数大于等于score_threshold的章节
                selected_chapters = []
                for chapter_selection in response.selected_chapters:
                    if chapter_selection.relevance_score >= score_threshold:
                        # 获取对应的叶子节点
                        chapter_index = chapter_selection.chapter_index - 1  # 转换为0基索引
                        if 0 <= chapter_index < len(leaf_nodes):
                            selected_piece = _build_selected_piece_from_leaf_node(
                                leaf_nodes[chapter_index],
                                chapter_index=chapter_index,
                                relevance_score=chapter_selection.relevance_score,
                                selection_reason=chapter_selection.selection_reason,
                            )
                            selected_chapters.append(selected_piece)
                
                # 创建SelectedContent对象
                if selected_chapters:
                    selected_content = SelectedContent(
                        question=question_text,
                        question_id=question_id,
                        selected_pieces=selected_chapters,
                    )
                    all_selected_content.append(selected_content)
                    
            except Exception as e:
                log_error_type_and_message(
                    logger=logger,
                    context="章节选择过程中出错",
                    e=e,
                    exc_info=False,
                    max_message_chars=200,
                )
                continue
    
    return all_selected_content, root


async def summary_index_retrival(state: RetrieverState) -> Command[str]:
    """基于 summary 索引进行检索。

    通过summary将文章的结构呈现给LLM来进行检索：
    1. 后序遍历收集所有叶子节点，并得到每个叶子节点的summary
    2. 用结构化输出的方式对LLM提问，让其返回几个对回答问题可能会有帮助的章节
    3. 用筛选器筛出分数大于等于score_threshold的章节，将其加入selected_content

    这是原有流程的wrapper函数，调用core函数并处理状态更新。
    """
    start_time = time.perf_counter()
    contents = state["origin_content"]
    unfinished_question_tasks = state["unfinished_question_tasks"]
    model_id = state["model_id"]
    use_batch_for_llm = state.get("use_batch_for_llm", False)
    # 调用core函数执行检索
    all_selected_content, updated_tree = await summary_index_retrieval_core(
        question_tasks=unfinished_question_tasks,
        origin_content=contents,
        model_id=model_id,
        summary_tree_provider=state.get("summary_tree_provider"),
        answer_lang=state.get("answer_lang", "zh_CN") or "zh_CN",
        use_batch_for_llm=use_batch_for_llm,
        score_threshold=6
    )
    
    # thinking_steps：记录summary阶段的query和资源
    thinking_steps_map = dict(state.get("thinking_steps_map", {}))
    item_map = build_item_key_to_item(contents)
    query_lookup = {
        task.id: _get_retrieval_query_for_task(task)
        for task in unfinished_question_tasks
        if getattr(task, "id", None)
    }
    for content in all_selected_content:
        ts_query = query_lookup.get(content.question_id)
        thinking_steps_map = _update_thinking_resources(
            thinking_steps_map,
            content.question_id,
            "summary",
            ts_query,
            content.selected_pieces,
            item_map,
        )

    elapsed_time = time.perf_counter() - start_time
    node_timings = dict(state.get("node_timings", {}))
    node_timings["summary_index_retrival"] = elapsed_time

    # 更新状态（始终把本步使用的树写回，供 workflow final_state 产出）
    update = {
        "selected_content": dedupe_selected_content_list(
            merge_selected_content_by_id(state["selected_content"], all_selected_content)
        ),
        "summary_index_retrival_done": True,
        "thinking_steps_map": thinking_steps_map,
        "node_timings": node_timings,
    }
    del updated_tree
    logger.info(f"summary_index_retrival完成，耗时: {elapsed_time:.3f}s")
    return Command(update=update, goto="judge_if_continue")


async def document_reading_retrieval_core(
    question_tasks: List[QuestionTask],
    origin_content: List,
    model_id: str,
    answer_lang: str = "zh_CN",
    is_table_retrieval: bool = False,
    simple: bool = True
) -> List[SelectedContent]:
    """document_reading检索的核心逻辑，返回检索结果。
    
    Args:
        question_tasks: QuestionTask对象列表
        origin_content: 原始内容列表（Item对象列表）
        model_id: 模型ID
        is_table_retrieval: 是否进行表格检索，默认False
        simple: 是否使用简单模式，默认True
        
    Returns:
        List[SelectedContent]: 检索结果列表
    """
    test_cfg = create_default_configs()
    retriever = LLMRetriever(
        test_config=test_cfg,
        model_id=model_id,
        isTableRetrieval=is_table_retrieval,
        answer_lang=answer_lang,
    )
    
    # 提取questions和question_ids用于LLMRetriever.run_batch（保持兼容性）
    questions = [_get_retrieval_query_for_task(task) for task in question_tasks]
    question_ids = [task.id for task in question_tasks]
    logger.info(f"[阶段标识] 进入检索阶段: document_reading_retrieval_core, 问题列表: {question_ids}")
    
    selected_contents = await retriever.run_batch(
        content_list=origin_content,
        questions=questions,
        question_ids=question_ids,
        simple=simple
    )
    
    return selected_contents


async def document_reading_retrival(state: RetrieverState) -> Command[str]:
    """
    文档级阅读检索（精读）。
    1. 使用LLMRetriever进行检索
    2. 将检索结果合并到状态中
    3. 更新状态，标记文档级阅读检索完成
    
    这是原有流程的wrapper函数，调用core函数并处理状态更新。
    """
    start_time = time.perf_counter()
    unfinished_question_tasks = state["unfinished_question_tasks"]
    content = state["origin_content"]
    model_id = state["model_id"]
    
    # 调用core函数执行检索
    selected_contents = await document_reading_retrieval_core(
        question_tasks=unfinished_question_tasks,
        origin_content=content,
        model_id=model_id,
        answer_lang=state.get("answer_lang", "zh_CN") or "zh_CN",
        is_table_retrieval=False,
        simple=True
    )

    # 更新thinking_steps_map：记录document阶段的query与资源
    thinking_steps_map = dict(state.get("thinking_steps_map", {}))
    item_map = build_item_key_to_item(content)
    query_lookup = {
        task.id: _get_retrieval_query_for_task(task)
        for task in unfinished_question_tasks
        if getattr(task, "id", None)
    }
    for selected in selected_contents:
        ts_query = query_lookup.get(selected.question_id)
        thinking_steps_map = _update_thinking_resources(
            thinking_steps_map,
            selected.question_id,
            "document",
            ts_query,
            selected.selected_pieces,
            item_map,
        )

    elapsed_time = time.perf_counter() - start_time
    node_timings = dict(state.get("node_timings", {}))
    node_timings["document_reading_retrival"] = elapsed_time

    update = {
        "selected_content": dedupe_selected_content_list(
            merge_selected_content_by_id(state["selected_content"], selected_contents)
        ),
        "document_reading_retrival_done": True,
        "thinking_steps_map": thinking_steps_map,
        "node_timings": node_timings,
    }

    # 流式处理模式：标记所有剩余的问题到状态中供消费者使用
    if unfinished_question_tasks:
        # 提取所有question_ids
        all_question_ids = [task.id for task in unfinished_question_tasks]
        update["finished_question_ids"] = all_question_ids
        logger.info(f"流式模式下，将剩余问题标记到状态: {all_question_ids}")

    logger.info(f"document_reading_retrival完成，耗时: {elapsed_time:.3f}s")
    return Command(update=update, goto="__end__")

async def custom_retrieval_workflow(state: RetrieverState) -> Command[str]:
    """可配置检索流程（新方案）。

    严格根据每个问题的 data_sources 配置决定如何检索，并将结果合并。
    - data_sources 中的 chunk_index_retrieval 项走 RAG 检索；
    - 其它类型项（GenericDataSource）直接将 data 作为 SelectedPiece.raw_text 注入上下文。

    注意：不再支持基于 tools 的检索配置，tools 仅用于 AnswerAgent 行为控制。

    Args:
        state: 检索工作流状态

    Returns:
        Command: 更新状态并跳转到结束节点
    """
    unfinished_question_tasks = state["unfinished_question_tasks"]
    origin_content = state["origin_content"]
    model_id = state.get("model_id", "model_for_content_generation")
    thinking_steps_map = dict(state.get("thinking_steps_map", {}))
    item_map = build_item_key_to_item(origin_content)
    
    logger.info(f"custom_retrieval_workflow开始执行，问题数量: {len(unfinished_question_tasks)}")
    
    all_results: List[SelectedContent] = []
    # 遍历所有问题，根据其 data_sources 配置决定如何检索
    for task in unfinished_question_tasks:
        logger.info(f"处理问题: {task.id}, description: {task.description[:50]}...")

        # === 新方案：强制要求 data_sources，完全不再依赖 tools 进行检索配置 ===
        # 注：仅 knowledge/tools 无 data_sources 的题目由上层在 custom_answer_generator 中走 run_streaming，不会进入本节点
        try:
            resolved_data_sources = task.get_resolved_data_sources()
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context=f"解析问题 {task.id} 的 data_sources 失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            raise

        if not resolved_data_sources:
            logger.error(f"问题 {task.id} 未配置 data_sources，无法确定检索策略，请检查问题配置！")
            raise ValueError(f"问题 {task.id} 未配置 data_sources，无法确定检索策略，请检查问题配置！")

        logger.info(f"问题 {task.id} 使用 data_sources 配置，数量: {len(resolved_data_sources)}")
        pieces_for_task: List[SelectedPiece] = []
        ds_index = 0
        for ds in resolved_data_sources:
            ds_index += 1
            # 1) chunk_index_retrieval: 走 RAG 检索
            if isinstance(ds, ChunkIndexRetrievalTool):
                # 优先使用 data_source 显式配置的 query，否则使用入口处已统一生成的 task.retrieval_query
                resolved_query = ds.query or _get_retrieval_query_for_task(task)
                logger.info(
                    f"问题 {task.id} data_source[{ds_index}] 使用 ChunkIndexRetrievalTool 进行检索: "
                    f"query={resolved_query[:50] if len(resolved_query) > 50 else resolved_query}..., "
                    f"top_k={ds.top_k}, "
                    f"return_child_chunks={ds.return_child_chunk}, "
                    f"rerank_enabled={ds.rerank_enabled}, "
                    f"hybrid_enabled={ds.hybrid_enabled}"
                )

                call_kwargs = {
                    "question_tasks": [task],
                    "origin_content": origin_content,
                    "queries": [resolved_query],
                    "top_k": ds.top_k,
                    "file_ids": ds.file_ids,
                    "return_child_chunks": ds.return_child_chunk,
                    "rerank_enabled": ds.rerank_enabled,
                    "hybrid_enabled": ds.hybrid_enabled,
                }

                task_results, query_map = await chunk_index_retrieval_core(**call_kwargs)
                for content_result in task_results:
                    pieces_for_task.extend(content_result.selected_pieces)
                    actual_query = query_map.get(content_result.question_id, resolved_query)
                    thinking_steps_map = _update_thinking_resources(
                        thinking_steps_map,
                        content_result.question_id,
                        "chunk",
                        actual_query,
                        content_result.selected_pieces,
                        item_map,
                    )
                logger.info(f"问题 {task.id} data_source[{ds_index}] chunk 检索完成，获得 {len(task_results)} 个检索结果")
            # 2) 其它类型：根据 data_source 类型直接注入上下文
            elif isinstance(ds, OtherDataSource):
                # 固定占位数据源：直接作为原始文本片段注入上下文（无真实文档来源）
                logger.info(
                    f"问题 {task.id} data_source[{ds_index}] 类型=other_datasource 作为占位文本片段注入上下文"
                )
                synthetic_piece = SelectedPiece(
                    doc_id=f"{task.id}_DataSource_{ds_index}",
                    element_ids=[],
                    file_id="",
                    relevance_score=None,
                    selection_reason=f"来自 data_source: {ds.type}",
                    source_stage="document",
                    raw_text=str(ds.data),
                    raw_text_only=True,
                )
                pieces_for_task.append(synthetic_piece)
            elif isinstance(ds, GenericDataSource):
                # 针对 extract 类型：data 结构为包含 objectSource 的复杂对象，需要从中解析 value / fileId / element_id
                if ds.type == "extract":
                    try:
                        # 按新 schema：data 为 list，项为 { tagName, files: [{ fileId, objectSource }, ...] }
                        if not isinstance(ds.data, list):
                            logger.error(
                                f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 data 应为 list，当前为 {type(ds.data).__name__}，已跳过"
                            )
                            continue

                        data_items = ds.data
                        if not data_items:
                            logger.error(f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 data 为空列表，已跳过")
                            continue

                        # 对每个 file 单独生成一个 SelectedPiece，避免跨文件聚合 file_id/element_ids
                        piece_counter = 0

                        for item in data_items:
                            if not isinstance(item, dict):
                                logger.error(f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 data 项非 dict，已跳过")
                                continue

                            tag_name = item.get("tagName")
                            files = item.get("files") or []
                            if not isinstance(files, list):
                                logger.error(
                                    f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 files 应为 list，当前为 {type(files).__name__}，已跳过"
                                )
                                continue

                            for f in files:
                                if not isinstance(f, dict):
                                    continue

                                file_id_local = f.get("fileId")
                                if not file_id_local:
                                    logger.error(f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 fileId 为空，已跳过")
                                    continue

                                object_source_raw = f.get("objectSource")
                                if not object_source_raw:
                                    logger.error(f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 objectSource 为空，已跳过")
                                    continue

                                if not isinstance(object_source_raw, list):
                                    logger.error(
                                        f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 objectSource 应为 list，当前为 {type(object_source_raw).__name__}，已跳过"
                                    )
                                    continue

                                object_source_list = object_source_raw

                                # 每个 objectSource 项单独生成一个 SelectedPiece（一个 value 对应一个片段）
                                for src in object_source_list:
                                    if not isinstance(src, dict):
                                        logger.error(f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 objectSource 项非 dict，已跳过")
                                        continue
                                    
                                    if "value" not in src:
                                        logger.error(f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 objectSource 项无 value，已跳过")
                                        continue

                                    value_local = src.get("value")
                                    element_ids_local = []  # 仅当前项的 element_id
                                    el = src.get("element_id")
                                    if el is not None:
                                        try:
                                            eid = int(el)
                                        except Exception:
                                            try:
                                                eid = int(str(el))
                                            except Exception:
                                                logger.error(
                                                    f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 element_id 非法: {el}"
                                                )
                                                continue
                                        if eid not in element_ids_local:
                                            element_ids_local.append(eid)
                                        else:
                                            logger.error(f"问题 {task.id} data_source[{ds_index}] 类型=extract 的 element_id 已存在（不应出现）")
                                            continue

                                    if value_local is None:
                                        logger.error(
                                            f"问题 {task.id} data_source[{ds_index}] 类型=extract 未从 objectSource 中解析到任何 value，已跳过当前 file"
                                        )
                                        continue

                                    # value 作为 raw_text 的内容：
                                    # - str: 直接使用（避免 json.dumps 引入额外引号）
                                    # - 其它类型: 序列化为 json 字符串
                                    json_value_str = (
                                        value_local
                                        if isinstance(value_local, str)
                                        else json.dumps(value_local, ensure_ascii=False)
                                    )
                                    raw_text_with_prefix = (
                                        f"提取结果（tag={tag_name}）：\n{json_value_str}"
                                        if tag_name
                                        else f"提取结果：\n{json_value_str}"
                                    )

                                    piece_counter += 1
                                    tag_info = f", tagName={tag_name}" if tag_name else ""
                                    logger.info(
                                        f"问题 {task.id} data_source[{ds_index}] 类型=extract 为第 {piece_counter} 个 objectSource 项生成片段，"
                                        f"element_ids 数量={len(element_ids_local)}, file_id={file_id_local}{tag_info}"
                                    )

                                    synthetic_piece = SelectedPiece(
                                        doc_id=f"{task.id}_DataSource_{ds_index}_{piece_counter}",
                                        element_ids=element_ids_local,
                                        file_id=file_id_local or "",
                                        relevance_score=None,
                                        selection_reason=f"来自 data_source: {ds.type}{tag_info}",
                                        source_stage=None,
                                        raw_text=raw_text_with_prefix,
                                        raw_text_only=False,
                                    )
                                    pieces_for_task.append(synthetic_piece)
                    except Exception as e:
                        log_error_type_and_message(
                            logger=logger,
                            context=f"问题 {task.id} data_source[{ds_index}] 类型=extract 处理失败",
                            e=e,
                            exc_info=True,
                            max_message_chars=200,
                        )
                        # 出错时，不中断整个问题，只跳过当前 data_source
                        continue
                else:
                    # 其它通用 data_source：仍然按原有逻辑，直接将 data 转为字符串注入上下文
                    raw_text = ""
                    try:
                        raw_text = "" if ds.data is None else str(ds.data)
                    except Exception:
                        logger.error(
                            f"问题 {task.id} data_source[{ds_index}] 类型={ds.type} 的 data 转换为字符串失败，已跳过"
                        )
                        raw_text = ""

                    if not raw_text:
                        logger.error(
                            f"问题 {task.id} data_source[{ds_index}] 类型={ds.type} 的 data 为空，已跳过"
                        )
                        continue

                    logger.info(
                        f"问题 {task.id} data_source[{ds_index}] 类型={ds.type} 作为原始文本片段注入上下文"
                    )
                    synthetic_piece = SelectedPiece(
                        doc_id=f"{task.id}_DataSource_{ds_index}",
                        element_ids=[],
                        file_id="",
                        relevance_score=None,
                        selection_reason=f"来自 data_source: {ds.type}",
                        source_stage="document",
                        raw_text=raw_text,
                        raw_text_only=True,
                    )
                    pieces_for_task.append(synthetic_piece)
            else:
                # 理论上不应该出现，但为健壮性保留
                logger.error(
                    f"问题 {task.id} data_source[{ds_index}] 类型未识别({ds.__class__.__name__})，已跳过"
                )

        # 每个问题只对应一个 SelectedContent，在 data_sources 循环外汇总后构建
        selected_content = SelectedContent(
            question=task.description,
            question_id=task.id,
            selected_pieces=pieces_for_task,
        )
        all_results.append(selected_content)

    logger.info(
        f"data_sources 检索阶段完成，共处理 {len(unfinished_question_tasks)} 个问题，获得 {len(all_results)} 个检索结果"
    )
    
    # 合并所有结果
    merged_content = merge_selected_content_by_id(state["selected_content"], all_results)
    
    # 构建更新状态
    # 将所有问题标记为已完成（因为custom_retrieval_workflow不依赖judge_if_continue判断）
    all_question_ids = [task.id for task in unfinished_question_tasks]
    
    update = {
        "selected_content": merged_content,
        "unfinished_question_tasks": [],  # 所有问题都已处理完成
        "finished_question_ids": all_question_ids,  # 标记所有问题为已完成
        "thinking_steps_map": thinking_steps_map,
    }
    
    logger.info(f"custom_retrieval_workflow完成，共处理 {len(unfinished_question_tasks)} 个问题，"
               f"获得 {len(all_results)} 个问题的检索结果，所有问题已标记为完成")
    
    return Command(update=update, goto="__end__")
