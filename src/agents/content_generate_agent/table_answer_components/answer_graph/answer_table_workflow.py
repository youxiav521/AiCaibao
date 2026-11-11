from src.agents.content_generate_agent.table_answer_components.table_retrieval.table_extract_filter import TableDataFilterResult, RelevantSegment, extract_table_keywords, filter_relevant_table_data
import json
from typing_extensions import List
from src.agents.content_generate_agent.utils import log_error_type_and_message

from src.agents.content_generate_agent.input_models import Item, TableItem
from src.agents.content_generate_agent.models import Citation, GenericDataSource
from src.agents.content_generate_agent.models import ChunkIndexRetrievalTool
from src.agents.content_generate_agent.table_answer_components.answer_graph.answer_table_state import FillResult, TableAnswerState
from src.agents.data_analyst_agent.coding_agent.graph.table_agent.fill_table_graph.fill_table_agent import FillTableAgentService
from src.agents.content_generate_agent.utils import contains_cjk, get_localized_message, normalize_locale
from src.agents.content_generate_agent.table_answer_components.table_retrieval.table_retrieval import (
    retrieve_table_rag_single,
    query_table_rag_custom_single,
)
from src.agents.content_generate_agent.table_answer_components.table_retrieval.table_enhance import enhance_table_items_context
from loguru import logger

def _handle_generic_data_source(
    resolved: List,
    relative_docs: List[dict]
) -> tuple[List[dict], List[Citation]]:
    """
    处理 GenericDataSource 类型的数据源，生成 citations 和 relative_docs。
    
    Args:
        resolved: 已解析的数据源列表
        relative_docs: 当前的相关文档列表（用于计算 ID 递增）
    
    Returns:
        tuple: (updated_relative_docs, new_citations)
    """
    new_citations: List[Citation] = []
    updated_relative_docs = list(relative_docs)
    
    generic_data_sources = [ds for ds in resolved if isinstance(ds, GenericDataSource)]
    
    # 计算 relative_docs 当前的最大 id，用于递增
    current_max_id = 0
    for doc in relative_docs:
        doc_id = doc.get("id", "")
        try:
            doc_id_int = int(doc_id)
            if doc_id_int > current_max_id:
                current_max_id = doc_id_int
        except (ValueError, TypeError):
            pass
    
    for ds in generic_data_sources:
        if ds.type == "extract":
            extract_data = ds.data if hasattr(ds, "data") else []
            for data_item in extract_data:
                tag_name = data_item.get("tagName", "")
                files = data_item.get("files", [])
                for file in files:
                    file_id = file.get("fileId", "")
                    object_source = file.get("objectSource", [])
                    source_element_ids = []
                    combined_text_parts = []
                    for object_source_item in object_source:
                        item_element_id = object_source_item.get("element_id", "")
                        item_value = object_source_item.get("value", "")
                        if item_element_id != "" and item_element_id is not None:
                            item_element_id = int(item_element_id)
                            source_element_ids.append(item_element_id)
                        if item_value is not None and item_value != "":
                            if isinstance(item_value, dict):
                                for k, v in item_value.items():
                                    combined_text_parts.append(f"{k}：{v}")
                            elif isinstance(item_value, list):
                                combined_text_parts.append(str(item_value))
                            else:
                                combined_text_parts.append(str(item_value))
                    
                    combined_text = "\n".join(combined_text_parts)
                    
                    current_max_id += 1
                    updated_relative_docs.append({
                        "id": str(current_max_id),
                        "title": tag_name + f"{current_max_id}",
                        "text": combined_text
                    })
                    
                    citation = Citation(
                        id=len(new_citations) + 1,
                        element_ids=source_element_ids,
                        file_id=file_id,
                        quote=f"配置的提取内容{tag_name}"
                    )
                    new_citations.append(citation)
    
    return updated_relative_docs, new_citations

async def table_entry_node(state: TableAnswerState):
    # 初始化表格文档，使用原始表格HTML
    task = state["question_task"]
    question_id = task.id
    table_html = task.html or ""
    
    # 只处理一个表格，不进行拆分
    if table_html:
        table_with_documents = {
            "table_name": f"table_{question_id}",
            "table_html": table_html
    }
    else:
        table_with_documents = {
            "table_name": f"table_{question_id}",
            "table_html": ""
        }
    state["table_with_documents"] = table_with_documents
    return state

def should_use_data_sources_retrieval(state: TableAnswerState) -> str:
    """
    条件边：若 QuestionTask 的 data_sources 中存在至少一个 ChunkIndexRetrievalTool，
    则走 table_retrieval_data_sources_node，否则走 table_retrieval_node。
    """
    task = state.get("question_task")
    logger.info(f"--开始处理questionTask--: {task}")
    logger.info(f"--questionTask的data_sources--: 判断是否配置了数据源检索")
    if not task:
        return "table_retrieval_node"
    resolved = task.get_resolved_data_sources()
    if not resolved:
        return "table_retrieval_node"
    has_chunk = any(isinstance(ds, ChunkIndexRetrievalTool) for ds in resolved)
    return "table_retrieval_data_sources_node" if has_chunk else "table_retrieval_node"

async def table_retrieval_node(state: TableAnswerState):
    # 做问题表格相关检索
    logger.info(f"table_retrieval_node开始")

    task = state["question_task"]
    table_description = task.description or ""
    table_prompt = task.prompt or ""
    table_question_id = task.id
    content_list = state.get("content_list", [])
    table_with_documents = state.get("table_with_documents", {})
    model_id = state.get("model_id", "")
    disable_keyword_extract = state.get("disable_keyword_extract", False)
    
    if not table_with_documents:
        state["table_items"] = []
        return state
    
    table_name = table_with_documents.get("table_name", "table")

    # 如果table_prompt不为空，则使用table_prompt作为检索问题，否则使用table_description
    retrieval_query = table_prompt if table_prompt else table_description
    # 根据 disable_keyword_extract 参数决定是否使用关键词提取
    if not disable_keyword_extract:
        retrieval_query = table_description + "\n" + table_prompt
        retrieval_query = await extract_table_keywords(retrieval_query, model_id=model_id)
    logger.info(f"--检索问题--: {retrieval_query}")
    
    retrieval_result, file_id_to_name = await retrieve_table_rag_single(
        question=retrieval_query,
        question_id=table_question_id,
        typed_items=content_list
    )
    
    # 优化日志确保在 retrieval_result 为空 或 table_items 为空数组时都能打印相应日志
    if not retrieval_result or not isinstance(retrieval_result, dict):
        logger.error(f"表格 {table_name} (问题ID: {task.id}) 的检索结果为空或格式错误")
    elif not retrieval_result.get("table_items") or len(retrieval_result.get("table_items", [])) == 0:
        logger.error(f"表格 {table_name} (问题ID: {task.id}) 的检索结果中 table_items 为空")
    # 检查检索结果及table_items是否为空或格式错误
    if not retrieval_result or not isinstance(retrieval_result, dict) or not retrieval_result.get("table_items", []):
        if not retrieval_result or not isinstance(retrieval_result, dict):
            logger.error(f"表格 {table_name} (问题ID: {task.id}) 的检索结果为空或格式错误")
        else:
            logger.error(f"表格 {table_name} (问题ID: {task.id}) 的检索结果中 table_items 为空")
        # 即使检索失败，也设置空的 relative_docs，让 table_filter_node 处理
        table_with_documents["relative_docs"] = []
        state["table_items"] = []
        state["table_with_documents"] = table_with_documents
        return state
    
    table_items_data: List[TableItem] = retrieval_result.get("table_items", [])
    state["table_items"] = table_items_data
    logger.info(
        f"检索到 {len(table_items_data)} 个 table_items。" if table_items_data else "未检索到 table_items。"
    )
    
    # TODO 增强table数据上下文，将所有检索到的table_items的前后文作为增强信息添加到相关上下文
    enhanced_table_items_data: List[Item] = enhance_table_items_context(content_list,table_items_data)
    logger.info(f"增强后的table_items_data数量: {len(enhanced_table_items_data)}")
    # 保存所有检索到的数据，供 table_filter_node 进行过滤
    retrieval_data_list = []
    for idx, item in enumerate(enhanced_table_items_data):
        item_md = item.md
        item_element_id = item.element_id
        item_file_id = item.file_id
        # 安全访问 file_id_to_name，如果不存在则使用 file_id 作为默认值，并打印日志
        if item_file_id not in file_id_to_name:
            logger.warning(f"file_id {item_file_id} not found in file_id_to_name:{file_id_to_name}")
        item_file_name = file_id_to_name.get(item_file_id, item_file_id)
        if item_md:
            table_data = {
                "id": str(idx+1),
                "title": f"来自{item_file_name}的相关文本_{idx+1}",
                "text": str(item_md),
                "element_id": item_element_id,
                "file_id": item_file_id
            }
            retrieval_data_list.append(table_data)
    
    table_with_documents["relative_docs"] = retrieval_data_list
    state["table_with_documents"] = table_with_documents

    return state

async def table_retrieval_data_sources_node(state: TableAnswerState):
    """
    当 QuestionTask 的 data_sources 中存在 ChunkIndexRetrievalTool 时使用本节点替代 table_retrieval_node。
    依次用每个 ChunkIndexRetrievalTool 调用 query_table_rag_custom_single，合并去重 table_items 后与 table_retrieval_node 同样写入 state 并进入后续步骤。
    """
    logger.info("table_retrieval_data_sources_node 开始")
    task = state["question_task"]
    logger.info(f"[table_retrieval_data_sources_node] task: {task}")
    content_list = state.get("content_list", [])
    table_with_documents = state.get("table_with_documents", {})
    model_id = state.get("model_id", "model_for_content_generation")

    if not table_with_documents:
        state["table_items"] = []
        return state

    resolved = task.get_resolved_data_sources()
    chunk_tools = [ds for ds in (resolved or []) if isinstance(ds, ChunkIndexRetrievalTool)]
    if not chunk_tools:
        logger.warning("table_retrieval_data_sources_node: 无 ChunkIndexRetrievalTool，设置空检索结果")
        table_with_documents["relative_docs"] = []
        state["table_items"] = []
        state["table_with_documents"] = table_with_documents
        return state

    # 依次用每个 ChunkIndexRetrievalTool 调用 query_table_rag_custom_single，合并 table_items 并按 (file_id, element_id) 去重
    seen_keys: set = set()
    merged_table_items: List[Item] = []
    for ds in chunk_tools:
        search_type = "table" if ds.search_table else None
        # 优先使用 data_source 显式配置的 query，否则使用入口处已统一生成的 task.retrieval_query
        query = ds.query or getattr(task, "retrieval_query", None) or ""
        single_result = await query_table_rag_custom_single(
            question=query,
            question_id=task.id,
            typed_items=content_list,
            top_k=ds.top_k,
            file_ids=ds.file_ids,
            folder_id=None,
            return_child_chunks=ds.return_child_chunk,
            rerank_enabled=ds.rerank_enabled,
            hybrid_enabled=ds.hybrid_enabled,
            search_type=search_type,
        )
        for item in single_result.get("table_items", []):
            key = (str(item.file_id), item.element_id)
            if key not in seen_keys:
                seen_keys.add(key)
                merged_table_items.append(item)
    table_items_data = merged_table_items
    state["table_items"] = table_items_data
    logger.info(
        f"data_sources 共 {len(chunk_tools)} 个 ChunkIndexRetrievalTool，合并去重后 {len(table_items_data)} 个 table_items。"
    )

    if not table_items_data:
        table_with_documents["relative_docs"] = []
        state["table_with_documents"] = table_with_documents
        return state

    # 与 table_retrieval_node 一致：增强上下文并构建 relative_docs（用 file_id 兜底）
    enhanced_table_items_data: List[Item] = enhance_table_items_context(content_list, table_items_data)
    logger.info(f"增强后的 table_items_data 数量: {len(enhanced_table_items_data)}")
    file_id_to_name: dict = {}
    retrieval_data_list = []
    for idx, item in enumerate(enhanced_table_items_data):
        item_md = getattr(item, "md", None)
        item_element_id = item.element_id
        item_file_id = item.file_id
        item_file_name = file_id_to_name.get(item_file_id, item_file_id)
        if item_md:
            table_data = {
                "id": str(idx + 1),
                "title": f"来自{item_file_name}的相关文本_{idx+1}",
                "text": str(item_md),
                "element_id": item_element_id,
                "file_id": item_file_id,
            }
            retrieval_data_list.append(table_data)
    table_with_documents["relative_docs"] = retrieval_data_list
    state["table_with_documents"] = table_with_documents
    return state

async def table_filter_node(state: TableAnswerState):
    # 过滤检索到的表格，只保留与填充表格相关的部分
    logger.info(f"table_filter_node开始")
    task = state["question_task"]
    table_description = task.description or ""
    table_prompt = task.prompt or ""
    table_with_documents = state.get("table_with_documents", {})
    model_id = state.get("model_id", "")

    # TODO 获取是否配置了GenericDataSource
    resolved = task.get_resolved_data_sources()
    if resolved is None:
        has_generic_data_source = False
    else:
        has_generic_data_source = any(isinstance(ds, GenericDataSource) for ds in resolved)
    
    # 使用 table_description 或 table_prompt 作为问题文本
    question_text = table_description if table_description else table_prompt
    question_citations: list[Citation] = []
    
    # 只处理一个表格
    if not table_with_documents:
        state["question_citations"] = []
        return state
    
    table_name = table_with_documents.get("table_name", "table")
    relative_docs = table_with_documents.get("relative_docs", [])
    
    try:
        # 准备过滤数据
        table_data_segments = []
        for doc in relative_docs:
            table_data_segments.append({
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "content": doc.get("text", "")
            })
        # 调用过滤方法
        filter_result: TableDataFilterResult = await filter_relevant_table_data(
            question_text=question_text,
            table_data_segments=table_data_segments,
            model_id=model_id
        )
        
        # 日志记录过滤情况
        logger.info(
            f"表格 {table_name} 过滤完成: 总片段数={len(relative_docs)}, "
            f"相关片段数={len(filter_result.relevant_segments)}, "
            f"has_relevant_data={filter_result.has_relevant_data}, "
            f"description={filter_result.description}"
        )
        
        # 根据过滤结果更新 relative_docs
        if filter_result.has_relevant_data and filter_result.relevant_segments:
            # 构建ID到理由的映射
            id_to_reason = {seg.id: seg.reason for seg in filter_result.relevant_segments}
            relevant_ids = {seg.id for seg in filter_result.relevant_segments}
            # 只保留相关的片段
            filtered_docs = [
                doc for doc in relative_docs 
                if doc.get("id") in relevant_ids
            ]
            table_with_documents["relative_docs"] = filtered_docs
            
            # 为过滤后的文档创建 citations，使用选择理由作为 quote
            for doc in filtered_docs:
                doc_id = doc.get("id", "")
                # 从过滤结果中获取该片段的选择理由，如果没有则使用空字符串
                selection_reason = id_to_reason.get(doc_id, "")
                citation = Citation(
                    id=len(question_citations)+1,
                    element_ids=[doc.get("element_id")] if doc.get("element_id") is not None else [],
                    file_id=doc.get("file_id", ""),
                    quote=selection_reason
                )
                question_citations.append(citation)
            
            # 保存过滤结果对象
            table_with_documents["filter_result"] = filter_result
            state["skip_fill"] = False
        else:
            # 没有相关片段，清空 relative_docs
            logger.warning(
                f"表格 {table_name} 过滤后没有相关片段: {filter_result.description}"
            )
            # 保存过滤结果对象
            table_with_documents["filter_result"] = filter_result
            state["skip_fill"] = True
            # 过滤判断所有检索结果不相关时,为所有文档创建 citations
            # TODO: 当有generic_data_source传入时，应该进行填表
            if has_generic_data_source:
                state["skip_fill"] = False
            for doc in relative_docs:
                citation = Citation(
                    id=len(question_citations)+1,
                    element_ids=[doc.get("element_id")] if doc.get("element_id") is not None else [],
                    file_id=doc.get("file_id", ""),
                    quote=""
                )
                question_citations.append(citation)
        
        # TODO: 处理 GenericDataSource 类型的数据源
        filtered_relative_docs = table_with_documents.get("relative_docs", [])
        # 处理 GenericDataSource 类型的数据源
        if has_generic_data_source:
            new_relative_docs, new_citations = _handle_generic_data_source(
                resolved=resolved,
                relative_docs=filtered_relative_docs
            )
            # 更新 relative_docs 和 question_citations
            question_citations.extend(new_citations)
            table_with_documents["relative_docs"] = new_relative_docs
    
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context=f"表格 {table_name} 过滤失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        # 过滤失败时，保留所有检索结果，创建 TableDataFilterResult 对象
        error_relevant_segments = [
            RelevantSegment(
                id=doc.get("id", ""),
                reason="保留所有检索结果"
            )
            for doc in relative_docs
            if doc.get("id")
        ]
        table_with_documents["filter_result"] = TableDataFilterResult(
            has_relevant_data=True,  # 假设保留所有结果
            relevant_segments=error_relevant_segments,
            description=""  # 当has_relevant_data=true时，description必须为空字符串
        )
        state["skip_fill"] = False
        # 为所有文档创建 citations
        for doc in relative_docs:
            citation = Citation(
                id=len(question_citations)+1,
                element_ids=[doc.get("element_id")] if doc.get("element_id") is not None else [],
                file_id=doc.get("file_id", ""),
                quote=""
            )
            question_citations.append(citation)

    state["table_with_documents"] = table_with_documents
    state["question_citations"] = question_citations
    return state

def should_skip_fill_node(state: TableAnswerState) -> str:
    """
    判断是否应该跳过 fill 节点
    返回下一个节点的名称
    """
    skip_fill = state.get("skip_fill", False)
    if skip_fill:
        return "table_output_node"
    else:
        return "table_fill_node"
    
async def table_fill_node(state: TableAnswerState):
    # 填充表格数据
    try:
        task = state["question_task"]
        table_description = task.description or ""
        table_with_documents = state.get("table_with_documents", {})
        history_messages = state.get("messages",[])
        model_id = state.get("model_id", "")
        max_iterations = state.get("max_iterations", 5)
        answer_lang = state.get("answer_lang", "zh_CN")

        fill_service = FillTableAgentService()
        
        # 只处理一个表格
        if not table_with_documents:
            state["table_with_documents"] = []
            return state
        
        table_html = str(table_with_documents.get("table_html", ""))
        table_name = str(table_with_documents.get("table_name", ""))
        relative_docs = table_with_documents.get("relative_docs", [])
        
        logger.info(f"table_fill_node: 表格 {table_name} 填充，相关片段数量: {len(relative_docs)}")
        
        # 如果 relative_docs 为空，记录错误并跳过填充（使用规范错误结构）
        if not relative_docs:
            warning_msg = get_localized_message(
                "table_retrieval_empty",
                answer_lang,
                table_name=table_name,
            )
            logger.warning(warning_msg)
            table_with_documents["fill_result"] = FillResult.error_result(warning_msg, table_html)
            state["table_with_documents"] = table_with_documents
            return state

        table_result = await fill_service.run_fill_table_with_token_limit(
            question=table_description + get_localized_message("table_fill_question_suffix", answer_lang) + table_html,
            documents=relative_docs,
            model_id=model_id,
            history_messages=history_messages,
            max_iterations=max_iterations,
            answer_lang=answer_lang,
        )
        output_table_data = table_result.get("output_table_data", "")
        # 错误结果时补充 filled_table_html 便于前端展示原始表格
        if FillResult.is_error_dict(output_table_data):
            output_table_data = FillResult.ensure_filled_html(output_table_data, table_html)
        table_with_documents["fill_result"] = output_table_data

        state["table_with_documents"] = table_with_documents
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="table_fill_node: 表格填充出现错误",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        # 异常时写入规范错误结果，保留当前表格 HTML 用于展示
        table_with_documents = state.get("table_with_documents", {})
        table_html_fallback = table_with_documents.get("table_html", "") if table_with_documents else ""
        table_with_documents["fill_result"] = FillResult.error_result(
            get_localized_message("table_fill_error", state.get("answer_lang", "zh_CN"), error=str(e)),
            table_html_fallback,
        )
        state["table_with_documents"] = table_with_documents
    return state


async def table_output_node(state: TableAnswerState):
    # 输出表格填充结果
    table_with_documents = state.get("table_with_documents", {})
    task = state["question_task"]
    target_table_html = task.html or ""
    skip_fill = state.get("skip_fill", False)
    answer_lang = state.get("answer_lang", "zh_CN")
    
    # 检查是否因无相关数据而跳过填充
    if skip_fill:
        no_relevant_description = ""
        if table_with_documents:
            filter_result: TableDataFilterResult = table_with_documents.get("filter_result")
            if filter_result and not filter_result.has_relevant_data:
                no_relevant_description = filter_result.description
        logger.warning("由于没有相关数据，返回无法填表的描述信息")
        detail = no_relevant_description
        if not detail or (
            normalize_locale(answer_lang) == "en_US" and contains_cjk(detail)
        ):
            detail = get_localized_message("table_missing_related_info_default_detail", answer_lang)
        state["merged_table_result"] = FillResult.error_result(
            get_localized_message("table_missing_related_info", answer_lang, description=detail),
            target_table_html,
        )
        return state

    # 无表格填充上下文时返回规范错误
    if not table_with_documents:
        state["merged_table_result"] = FillResult.error_result(
            get_localized_message("table_no_fill_result", answer_lang),
            "",
        )
        return state

    fill_result = table_with_documents.get("fill_result", "")
    # 错误结果时确保有 filled_table_html（用原始表格兜底）
    if FillResult.is_error_dict(fill_result):
        fill_result = FillResult.ensure_filled_html(fill_result, target_table_html)
    state["merged_table_result"] = fill_result

    return state
    
