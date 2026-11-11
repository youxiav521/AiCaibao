"""
表格关键词提炼模块

该模块提供了从表格问题提示词中提炼真正需要的相关数据关键词的功能。
用于在检索前对问题进行优化，提高检索精度。
并支持按 token 上限对 table_data_segments 做滑动窗口过滤，避免单次 prompt 超长。
"""

import traceback
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.base.prompt_template_factory import PromptManager

from ...utils import ainvoke_llm_with_prompt, ainvoke_llm_with_structured_output
from src.agents.content_generate_agent.utils import count_tokens, truncate_to_token_limit
from loguru import logger

# 默认 token 上限与划分值（与 fill_table / agent_full 的 token_utils 对齐，避免单次 prompt 超模型上下文）
DEFAULT_FILTER_TOKEN_LIMIT = 100000
DEFAULT_FILTER_PARTITION_TOKEN_VALUE = 90000

class RelevantSegment(BaseModel):
    """
    相关片段信息模型，包含片段ID和选择理由
    """
    id: str = Field(
        description="表格数据片段的唯一标识符（字符串）"
    )
    reason: str = Field(
        description="该片段被选中的理由说明（字符串）。应简洁明确地说明该片段为什么与问题相关，包含哪些关键信息等"
    )

class TableDataFilterResult(BaseModel):
    """
    表格数据过滤结果的结构化输出模型
    """
    has_relevant_data: bool = Field(
        description="是否存在与问题相关的表格数据片段。如果所有片段都不相关，则为false；如果有相关片段，则为true"
    )
    relevant_segments: List[RelevantSegment] = Field(
        default_factory=list,
        description="与问题相关的表格数据片段列表，每个元素包含片段ID和选择理由。当has_relevant_data为true时，此列表包含所有相关片段的信息；当has_relevant_data为false时，此列表为空"
    )
    description: str = Field(
        default="",
        description="描述信息。当has_relevant_data为false时，应详细说明片段中缺少哪些关键信息（如时间、实体、指标等），且不得提及具体的片段ID；当has_relevant_data为true时，必须设置为空字符串"
    )


def _segment_to_prompt_text(seg: Dict[str, Any]) -> str:
    """将单个片段格式化为与 filter_prompt 中一致的字符串，用于 token 统计。"""
    return f"片段ID: {seg.get('id', '')}\n片段内容:{seg.get('title', '')}\n{seg.get('content', '')}\n"


def count_segment_tokens(seg: Dict[str, Any]) -> int:
    """计算单个表格数据片段在 prompt 中的 token 数。"""
    try:
        return count_tokens(_segment_to_prompt_text(seg))
    except Exception as e:
        logger.warning(f"count_tokens 失败: {e}，使用字符数近似")
        return len(_segment_to_prompt_text(seg))


def count_segments_tokens(segments: List[Dict[str, Any]]) -> int:
    """计算表格数据片段列表在 prompt 中的总 token 数。"""
    total = 0
    for seg in (segments or []):
        total += count_segment_tokens(seg)
    return total


def split_segments_by_token_limit(
    segments: List[Dict[str, Any]],
    token_limit: int = DEFAULT_FILTER_TOKEN_LIMIT,
    partition_token_value: int = DEFAULT_FILTER_PARTITION_TOKEN_VALUE,
) -> List[List[Dict[str, Any]]]:
    """
    按 token 上限将片段列表划分为多块，不拆开单个片段。
    当总 token 不超过 token_limit 时返回 [segments]；超过时按每块不超过 partition_token_value 划分。
    """
    if not segments:
        return []

    total_tokens = count_segments_tokens(segments)
    if total_tokens <= token_limit:
        return [segments]

    seg_tokens = [count_segment_tokens(seg) for seg in segments]
    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_tokens = 0

    for seg, tokens in zip(segments, seg_tokens):
        if current_chunk and (current_tokens + tokens > partition_token_value):
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(seg)
        current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk)

    logger.info(
        f"table_data_segments 总 token 数 {total_tokens} 超过上限 {token_limit}，"
        f"已按 partition_token_value={partition_token_value} 划分为 {len(chunks)} 块"
    )
    return chunks


async def extract_table_keywords(
    question_prompt: str,
    model_id: str = "model_for_content_generation_bailian"
) -> str:
    # 验证并修复 model_id，如果为 None 或空字符串，使用默认值
    if not model_id or not model_id.strip():
        logger.warning(f"model_id 为空或无效（值为: {model_id}），使用默认值 'model_for_content_generation_bailian'")
        model_id = "model_for_content_generation_bailian"
    
    if not question_prompt or not question_prompt.strip():
        logger.warning("问题提示词为空，返回空字符串")
        return ""
    # 构建关键词提取的提示词
    extraction_prompt = f"""请从以下表格问题提示词中提取出真正需要的、用于检索相关数据的关键词。

要求：
1. 只提取与数据检索相关的核心关键词和短语
2. 去除HTML标签、表格结构等格式信息
3. 去除冗余的描述性文字
4. 保留具体的数据需求关键词（如：客户名称、销售额、年份等）
5. 如果问题中包含具体的数值要求（如"前五大"、"最近三年"等），请保留
6. 返回简洁的关键词字符串，用空格或逗号分隔，不要添加解释性文字

任务：
- 分析表格生成要求和表格内容，从提示词中提取出用于填表的信息
- 从描述表格的问题中找出需要填入表格的项目等名称或者数据名称
- 从表格问题提示词中提取出真正需要的、用于检索相关数据的关键词。
- 不要将表格中的每个字段都提取出来，而是总结当前表格的填表要求和内容。

表格问题提示词：
{question_prompt}

请直接返回提炼后的关键词（不要添加任何前缀或说明）：
"""
    try:
        # 调用 LLM 提取关键词
        keywords = await ainvoke_llm_with_prompt(
            prompt=extraction_prompt,
            model_id=model_id
        )
        
        # 清理返回结果（去除可能的引号、换行等）
        keywords = keywords.strip().strip('"').strip("'")
        # 去除可能的说明性文字（如果模型返回了说明）
        if "关键词" in keywords or "：" in keywords or ":" in keywords:
            # 尝试提取冒号后的内容
            if "：" in keywords:
                keywords = keywords.split("：", 1)[-1].strip()
            elif ":" in keywords:
                keywords = keywords.split(":", 1)[-1].strip()
        
        logger.info(f"关键词提取完成，原始问题长度: {len(question_prompt)}，提炼后长度: {len(keywords)}")
        logger.debug(f"提炼的关键词: {keywords}")
        
        return keywords if keywords else question_prompt  # 如果提取失败，返回原始问题
    
    except Exception as e:
        logger.error(f"关键词提取失败: {e}")
        logger.error(f"详细异常堆栈: {traceback.format_exc()}")
        # 提取失败时返回原始问题，保证流程可以继续
        logger.warning("关键词提取失败，返回原始问题提示词")
        return question_prompt


async def _filter_relevant_table_data_single(
    question_text: str,
    table_data_segments: List[Dict[str, Any]],
    model_id: str,
) -> TableDataFilterResult:
    """单次调用：对一段 table_data_segments 做相关性过滤（内部使用，不处理空输入）。"""
    formatted_segments = table_data_segments
    segments_text = "\n\n".join([
        f"片段ID: {seg.get('id', '')}\n片段内容:{seg.get('title', '')}\n{seg.get('content', '')}\n"
        for seg in formatted_segments
    ])
    filter_prompt = PromptManager.getPrompt("table_filter_relevant_table", {
        "question_text": question_text,
        "segments_text": segments_text
    })

    result: TableDataFilterResult = await ainvoke_llm_with_structured_output(
        prompt=filter_prompt,
        model_id=model_id,
        response_type=TableDataFilterResult
    )
    valid_ids = {seg.get("id", "") for seg in formatted_segments}
    filtered_relevant_segments = [
        seg for seg in result.relevant_segments
        if seg.id in valid_ids
    ]
    invalid_segments = [seg for seg in result.relevant_segments if seg.id not in valid_ids]
    if invalid_segments:
        invalid_ids = [seg.id for seg in invalid_segments]
        logger.warning(f"模型返回了不存在的片段ID: {invalid_ids}，已过滤")
    if result.has_relevant_data and not filtered_relevant_segments:
        logger.warning("模型认为有相关数据，但所有返回的片段ID都无效，将has_relevant_data设为false")
        result.has_relevant_data = False
        if not result.description or "相关" in result.description:
            result.description = "模型返回的相关片段ID无效，所有片段都不相关"
    result.relevant_segments = filtered_relevant_segments
    if result.has_relevant_data and result.relevant_segments:
        for seg in result.relevant_segments:
            if not seg.reason or not seg.reason.strip():
                logger.warning(f"片段 {seg.id} 缺少选择理由，将生成默认理由")
                seg.reason = "该片段包含与问题相关的关键信息"
    if not result.has_relevant_data and not result.description:
        result.description = "所有表格数据片段都与问题文本不相关，无法用于回答该问题"
    return result


def _merge_filter_results(
    acc: Optional[TableDataFilterResult],
    chunk_result: TableDataFilterResult,
) -> TableDataFilterResult:
    """合并两个过滤结果（滑动窗口多块合并）。"""
    if acc is None:
        return chunk_result
    merged_segments = list(acc.relevant_segments) + list(chunk_result.relevant_segments)
    has_relevant = acc.has_relevant_data or chunk_result.has_relevant_data
    if has_relevant:
        description = ""
    else:
        parts = [acc.description or "", chunk_result.description or ""]
        description = "；".join(p for p in parts if p).strip() or (chunk_result.description or "")
    return TableDataFilterResult(
        has_relevant_data=has_relevant,
        relevant_segments=merged_segments,
        description=description,
    )


async def filter_relevant_table_data(
    question_text: str,
    table_data_segments: List[Dict[str, Any]],
    model_id: str = "model_for_content_generation_bailian",
    token_limit: int = DEFAULT_FILTER_TOKEN_LIMIT,
    partition_token_value: int = DEFAULT_FILTER_PARTITION_TOKEN_VALUE,
) -> TableDataFilterResult:
    """
    使用模型判断多段表格数据是否与问题文本相关，并返回相关片段的ID。
    当 table_data_segments 总 token 超过 token_limit 时，按 partition_token_value 划分为多块，
    逐块调用模型后合并结果（滑动窗口，与 fill_table 的 run_fill_table_with_token_limit 一致）。
    """
    if not model_id or not model_id.strip():
        logger.warning(f"model_id 为空或无效（值为: {model_id}），使用默认值 'model_for_content_generation_bailian'")
        model_id = "model_for_content_generation_bailian"

    if not question_text or not question_text.strip() :
        logger.error("缺少问题文本，无法回答")
        return TableDataFilterResult(
            has_relevant_data=False,
            relevant_segments=[],
            description="缺少问题文本，无法进行回答"
        )
    
    if not table_data_segments or len(table_data_segments) == 0:
        logger.error("缺少表格数据片段，无法回答")
        return TableDataFilterResult(
            has_relevant_data=False,
            relevant_segments=[],
            description="缺少表格数据片段，无法回答"
        )

    try:
        formatted_segments = table_data_segments
        chunks = split_segments_by_token_limit(
            formatted_segments,
            token_limit=token_limit,
            partition_token_value=partition_token_value,
        )
        last_result: Optional[TableDataFilterResult] = None
        for i, chunk in enumerate(chunks):
            chunk_tokens = count_segments_tokens(chunk)
            logger.info(
                f"---filter_relevant_table_data 滑动窗口---: "
                f"窗口 {i + 1}/{len(chunks)}，本块片段数 {len(chunk)}，本块 token 约 {chunk_tokens}"
            )
            chunk_result = await _filter_relevant_table_data_single(
                question_text=question_text,
                table_data_segments=chunk,
                model_id=model_id,
            )
            last_result = _merge_filter_results(last_result, chunk_result)

        return last_result if last_result is not None else TableDataFilterResult(
            has_relevant_data=False,
            relevant_segments=[],
            description="无表格数据片段",
        )

    except Exception as e:
        logger.error(f"表格数据过滤失败: {e}")
        logger.error(f"详细异常堆栈: {traceback.format_exc()}")
        logger.warning("表格数据过滤失败，返回无相关数据的结果")
        return TableDataFilterResult(
            has_relevant_data=False,
            relevant_segments=[],
            description="表格数据过滤失败，检查检索过程重试",
        )