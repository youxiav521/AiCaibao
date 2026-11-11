"""
表格检索模块

该模块提供了两种表格检索接口：
1. 基于 RAG接口 的单个问题表格检索
2. 基于 LLMRetriever 的批量问题表格检索
"""

from typing import List, Dict, Any, Optional

from ...input_models import Item
from src.applications.rag.services.query_service import run_batch_query
from src.applications.rag.schemas import (
    BatchQueryRequest,
    QuestionItem,
    BatchQueryItem,
    BatchQueryItemHttp,
    batch_query_item_to_http,
)

from src.base.langfuse_observe import trace_span
from loguru import logger

@trace_span("query_table_rag_custom")
async def query_table_rag_custom(
    questions: List[str],
    question_ids: List[str],
    typed_items: List[Item],
    *,
    top_k: int = 5,
    file_ids: Optional[List[str]] = None,
    folder_id: Optional[str] = None,
    return_child_chunks: bool = False,
    rerank_enabled: Optional[bool] = False,
    hybrid_enabled: Optional[bool] = True,
    search_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    自定义参数的表格 RAG 批量查询接口。

    可自定义 BatchQueryRequest 的全部参数，调用 RAG 批量查询后，
    根据 element_id 匹配 typed_items，返回 items_list。

    参数
    ----------
    questions : List[str]
        问题文本列表。
    question_ids : List[str]
        问题 ID 列表，与 questions 一一对应。
    typed_items : List[Item]
        已校验的文档项列表，用于提取默认 file_ids 及按 element_id 匹配返回 Item。
    top_k : int, 默认 5
        每条问题返回的最大检索条数。
    file_ids : Optional[List[str]], 默认 None
        按 metadata.file_id 过滤；为 None 时从 typed_items 中提取唯一 file_id。
    folder_id : Optional[str], 默认 None
        按 metadata.folder_id 过滤（资源库 ID）。
    return_child_chunks : bool, 默认 False
        是否检索孙切片但返回子切片。
    rerank_enabled : Optional[bool], 默认 False
        是否启用重排。
    hybrid_enabled : Optional[bool], 默认 True
        是否启用混合检索（向量 + BM25）。

    返回值
    -------
    List[Dict[str, Any]]
        items_list，每项为 {"question_id", "question", "table_items": List[Item]}。
    """
    if file_ids is None:
        file_ids = list({str(item.file_id) for item in typed_items})
    logger.info(f"自定义表格检索: {len(questions)} 个问题, file_ids={file_ids}, top_k={top_k}")

    batch_request = BatchQueryRequest(
        questions=[
            QuestionItem(question_id=qid, question_text=q)
            for qid, q in zip(question_ids, questions)
        ],
        top_k=top_k,
        file_ids=file_ids,
        folder_id=folder_id,
        return_child_chunks=return_child_chunks,
        rerank_enabled=rerank_enabled,
        hybrid_enabled=hybrid_enabled,
        search_type=search_type,
    )
    raw_results: List[BatchQueryItem] = await run_batch_query(body=batch_request)
    # 将 BatchQueryItem 转为 BatchQueryItemHttp 供后续使用
    retrieval_results: List[BatchQueryItemHttp] = [
        batch_query_item_to_http(r) for r in raw_results
    ]
    logger.info(f"RAG 查询完成，返回 {len(retrieval_results)} 条结果")

    # 根据检索结果匹配 typed_items，构建 items_list（仅返回列表）
    table_items_list: List[Dict[str, Any]] = []
    for question_id, question in zip(question_ids, questions):
        table_items_list.append({
            "question_id": question_id,
            "question": question,
            "table_items": []
        })
    question_dict = {d["question_id"]: d for d in table_items_list}
    added_items = {qid: set() for qid in question_ids}
    item_key_to_item = {(str(item.file_id), item.element_id): item for item in typed_items}

    for result in retrieval_results:
        target_dict = question_dict.get(result.questionId)
        if target_dict and result.elementIds:
            for element_id in result.elementIds:
                item_key = (str(result.fileId), element_id)
                if item_key not in added_items[result.questionId]:
                    if item_key in item_key_to_item:
                        target_dict["table_items"].append(item_key_to_item[item_key])
                        added_items[result.questionId].add(item_key)
                    else:
                        logger.warning(
                            f"问题 {result.questionId} 的检索结果中，fileId={result.fileId}, "
                            f"element_id={element_id} 在 typed_items 中未找到匹配项"
                        )
        else:
            if not target_dict:
                logger.warning(f"检索结果中的 questionId '{result.questionId}' 在输入的问题列表中未找到匹配")
            elif not result.elementIds:
                logger.warning(f"问题 {result.questionId} 的检索结果中没有 elementIds")

    logger.info(f"自定义表格检索，返回 {len(table_items_list)} 条结果")
    return table_items_list


@trace_span("query_table_rag_custom_single")
async def query_table_rag_custom_single(
    question: str,
    question_id: str,
    typed_items: List[Item],
    *,
    top_k: int = 5,
    file_ids: Optional[List[str]] = None,
    folder_id: Optional[str] = None,
    return_child_chunks: bool = False,
    rerank_enabled: Optional[bool] = False,
    hybrid_enabled: Optional[bool] = True,
    search_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    单次调用的自定义表格 RAG 查询接口，适配单问题、单组参数场景。
    """
    items_list = await query_table_rag_custom(
        questions=[question],
        question_ids=[question_id],
        typed_items=typed_items,
        top_k=top_k,
        file_ids=file_ids,
        folder_id=folder_id,
        return_child_chunks=return_child_chunks,
        rerank_enabled=rerank_enabled,
        hybrid_enabled=hybrid_enabled,
        search_type=search_type,
    )
    return items_list[0] if items_list else {}


@trace_span("retrieve_table_rag_batch")
async def retrieve_table_rag_batch(
    questions: List[str],
    question_ids: List[str],
    typed_items: List[Item],
    top_k: int = 5,
    hybrid_enabled: bool = True,
):
    """
    批量问题的表格检索接口
    
    使用 RAG 系统从文档中检索与问题相关的表格元素。
    该函数会：
    1. 从文档项中提取唯一的 file_id
    2. 构建批量查询请求（仅包含一个问题）
    3. 调用 RAG 接口进行检索
    4. 根据检索结果中的 element_id 匹配并返回对应的 Item 列表
    
    参数
    ----------
    questions: List[str]
        问题列表，可以是自然语言问题或表格的 HTML 结构。
    question_ids: List[str]
        问题ID列表，必须与 questions 一一对应，长度相同。
        要检索的问题文本。可以是自然语言问题，也可以是表格的 HTML 结构。
    typed_items: List[Item]
        已校验的文档项列表，用于提取 file_id 和后续匹配 element_id。
    top_k: int, 默认值 5
        检索返回的最大结果数量。
    hybrid_enabled: bool, 默认值 True
        是否启用混合检索（向量检索 + BM25 检索）。
    return_child_chunks: bool, 默认值 False
        是否返回子切片。如果为 False，则直接返回检索到的结果。
    
    返回值
    -------
    List[Dict[str, Any]]
        匹配到的表格项列表，按照检索相关性排序。
        每个结果包含：
        - question_id: 问题ID
        - question: 问题文本
        - table_items: 表格项列表
    Dict[str, str]
        文件ID到文件名的映射字典。
    """
    # 从文档项中提取唯一的 file_id
    file_ids = list({str(item.file_id) for item in typed_items})
    logger.info(f"提取到 {len(file_ids)} 个唯一的file_id: {file_ids}")
    
    # 构建批量查询请求
    batch_request = BatchQueryRequest(
        questions=[
            QuestionItem(question_id=question_id, question_text=question)
            for question_id, question in zip(question_ids, questions)
        ],
        top_k=top_k,
        file_ids=file_ids,
        hybrid_enabled=hybrid_enabled,
        return_child_chunks=False,
        search_type="table"  # 指定检索类型为表格
    )
    
    # 调用 RAG 接口进行检索，并将 BatchQueryItem 转为 BatchQueryItemHttp
    raw_results: List[BatchQueryItem] = await run_batch_query(body=batch_request)
    retrieval_results: List[BatchQueryItemHttp] = [
        batch_query_item_to_http(r) for r in raw_results
    ]
    logger.info(f"RAG 查询完成，返回 {len(retrieval_results)} 条结果")
    for result in retrieval_results:
        logger.debug(f"检索结果: questionId={result.questionId}, elementIds={result.elementIds}")

    # 对每个问题，根据检索结果中的 element_id 匹配并返回对应的 Item 列表
    table_items_list: List[Dict[str, Any]] = []
    for question_id, question in zip(question_ids, questions):
        table_items_list.append({
            "question_id": question_id,
            "question": question,
            "table_items": []
        })

    # 创建一个字典，用于快速查找问题对应的结果字典
    question_dict = {table_dict["question_id"]: table_dict for table_dict in table_items_list}
    # 使用 BatchQueryItemHttp 的驼峰字段构建 file_id -> file_name 映射
    file_id_to_name = {item.fileId: item.fileName for item in retrieval_results}

    # 用于问题内去重：同一个问题内，避免同一个表格项被多次添加
    added_items = {qid: set() for qid in question_ids}
    item_key_to_item = {(str(item.file_id), item.element_id): item for item in typed_items}

    for result in retrieval_results:
        target_dict = question_dict.get(result.questionId)
        if target_dict and result.elementIds:
            for element_id in result.elementIds:
                item_key = (str(result.fileId), element_id)
                if item_key not in added_items[result.questionId]:
                    if item_key in item_key_to_item:
                        target_dict["table_items"].append(item_key_to_item[item_key])
                        added_items[result.questionId].add(item_key)
                    else:
                        logger.warning(
                            f"问题 {result.questionId} 的检索结果中，fileId={result.fileId}, "
                            f"element_id={element_id} 在 typed_items 中未找到匹配项"
                        )
        else:
            if not target_dict:
                logger.warning(f"检索结果中的 questionId '{result.questionId}' 在输入的问题列表中未找到匹配")
            elif not result.elementIds:
                logger.warning(f"问题 {result.questionId} 的检索结果中没有 elementIds")
    
    return table_items_list, file_id_to_name

@trace_span("retrieve_table_rag_single")
async def retrieve_table_rag_single(
    question: str,
    question_id: str,
    typed_items: List[Item],
    top_k: int = 30,
    hybrid_enabled: bool = True,
):
    """
    单个问题的表格检索接口
    """
    results, file_id_to_name = await retrieve_table_rag_batch([question], [question_id], typed_items, top_k, hybrid_enabled)
    return results[0] if results else {}, file_id_to_name
