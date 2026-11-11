"""
Thinking Steps 构建器模块

用于统一管理 Thinking Steps 的构建逻辑，包括：
- 构建检索阶段信息（chunk/summary/document）
- 将检索阶段的 thinking_steps 信息附加到 FinalAnswer
"""
from typing import Optional, Dict, Any, Type, TypeVar
from loguru import logger

from .models import (
    ThinkingSteps,
    FinalAnswer,
    ChunkRetrievalInfo,
    SummaryRetrievalInfo,
    DocumentRetrievalInfo,
    RetrievalQueryItem,
    RetrievalResourcesItem,
    RetrievalAnalysisItem,
    RetrievalResourceItem,
)


T = TypeVar('T', ChunkRetrievalInfo, SummaryRetrievalInfo, DocumentRetrievalInfo)


def build_retrieval_stage_info(
    raw: Optional[Dict[str, Any]],
    stage_type: str,
    cls: Type[T]
) -> Optional[T]:
    """
    构建thinking_steps的检索阶段的信息，并返回检索阶段的信息的类实例。
    
    Args:
        raw: 检索阶段的信息的原始数据（包含 query, resources, analysis）
        stage_type: 检索阶段类型（"chunkRetrieval", "summaryRetrieval", "documentRetrieval"）
        cls: 检索阶段的信息的类（ChunkRetrievalInfo, SummaryRetrievalInfo, DocumentRetrievalInfo）
        
    Returns:
        检索阶段的信息的类实例，如果原始数据为空则返回None
    """
    if not raw:
        logger.warning(f"[build_retrieval_stage_info] {cls.__name__}的检索阶段的信息的原始数据为空，不写入thinking_steps")
        return None
    
    query = raw.get("query")
    resources_raw = raw.get("resources", [])
    analysis = raw.get("analysis", "")
    
    items = []
    
    # 1. analysis放在前面（如果有）
    if analysis:
        items.append(RetrievalAnalysisItem(text=analysis))

    # 2. query固定放在第二个item
    if query:
        items.append(RetrievalQueryItem(text=query))
    
    # 3. resources放在第三个item
    resource_items = []
    for r in resources_raw:
        fid = r.get("file_id")
        page = r.get("page")
        if fid is None or page is None:
            logger.error(f"[build_retrieval_stage_info] file_id或page为空，跳过该资源")
            continue
        # page转换为字符串
        resource_items.append(RetrievalResourceItem(file_id=str(fid), page=f"第{page}页"))
    
    if resource_items:
        items.append(RetrievalResourcesItem(items=resource_items))
    
    # 如果没有任何内容，返回None
    if not items:
        return None
    
    return cls(type=stage_type, items=items)


def attach_thinking_steps_to_answer(
    answer: FinalAnswer,
    thinking_entry: Optional[Dict[str, Any]]
) -> FinalAnswer:
    """
    将检索阶段的thinking_steps信息写入FinalAnswer。
    
    Args:
        answer: 最终答案对象
        thinking_entry: 检索阶段的thinking_steps原始数据（包含chunk/summary/document三个阶段的字典）
        
    Returns:
        更新后的FinalAnswer对象
    """
    if not thinking_entry:
        logger.warning(f"thinking_entry为空，保留原有thinking_steps")
        # 如果thinking_steps不存在，才创建新的
        if answer.thinking_steps is None:
            answer.thinking_steps = ThinkingSteps()
        return answer

    # 构建三个检索阶段的信息
    chunk_stage = build_retrieval_stage_info(
        thinking_entry.get("chunk"),
        "chunkRetrieval",
        ChunkRetrievalInfo
    )
    summary_stage = build_retrieval_stage_info(
        thinking_entry.get("summary"),
        "summaryRetrieval",
        SummaryRetrievalInfo
    )
    document_stage = build_retrieval_stage_info(
        thinking_entry.get("document"),
        "documentRetrieval",
        DocumentRetrievalInfo
    )

    # 保留原有的answer_agent_steps，避免被覆盖
    existing_answer_agent_steps = None
    if answer.thinking_steps and answer.thinking_steps.answer_agent_steps:
        existing_answer_agent_steps = answer.thinking_steps.answer_agent_steps

    answer.thinking_steps = ThinkingSteps(
        chunk_retrieval=chunk_stage,
        summary_retrieval=summary_stage,
        document_retrieval=document_stage,
        answer_agent_steps=existing_answer_agent_steps  # 保留原有的answer_agent_steps
    )
    return answer

