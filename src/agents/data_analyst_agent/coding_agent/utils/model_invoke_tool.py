from typing import List, Optional, Type, TypeVar, Any
from langgraph.graph.message import BaseMessage
from langchain.tools import BaseTool

from ..config.invoke_model_config import model_invoke_config
from .code_extractor import remove_think_blocks
from loguru import logger
from src.agents.content_generate_agent.utils import (
    ainvoke_llm_with_messages,
    ainvoke_llm_with_messages_and_tools,
    ainvoke_llm_with_messages_structured_output,
    _log_llm_stats_before,
    _log_llm_stats_after,
    count_tokens,
)

T = TypeVar("T")

# 默认 token 上限（保留余量给模型输出）
DEFAULT_MESSAGES_TOKEN_LIMIT = 120000

def _count_messages_tokens(messages: List[BaseMessage]) -> int:
    """
    计算 messages 列表的总 token 数。
    """
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        total += count_tokens(content)
    return total


def _truncate_messages_by_token_limit(
    messages: List[BaseMessage],
    token_limit: int = DEFAULT_MESSAGES_TOKEN_LIMIT
) -> List[BaseMessage]:
    """
    截断 messages 使其 token 数不超过限制。

    策略：保留 SystemMessage，保留最后两条 HumanMessage（及其之间的 AIMessage），
    移除中间的消息以控制 token 数量。
    """
    if not messages:
        return messages

    total_tokens = _count_messages_tokens(messages)
    if total_tokens <= token_limit:
        return messages

    logger.warning(
        f"---_truncate_messages_by_token_limit---: "
        f"消息总 token 数 {total_tokens} 超过上限 {token_limit}，"
        f"将截断消息保留 SystemMessage 和最后两条消息"
    )

    # 分离 SystemMessage 和其他消息
    system_messages = [msg for msg in messages if msg.type == "system"]
    other_messages = [msg for msg in messages if msg.type != "system"]

    # 保留最后两条消息（HumanMessage）以及它们之间的 AIMessage
    if len(other_messages) <= 2:
        return system_messages + other_messages

    # 获取最后两条 HumanMessage 的索引
    human_indices = [
        i for i, msg in enumerate(other_messages)
        if msg.type == "human"
    ]

    if len(human_indices) < 2:
        # 如果 HumanMessage 不足两条，保留最后两条消息
        kept_messages = other_messages[-2:]
    else:
        # 保留最后两个 HumanMessage 及其之间的所有消息
        last_two_human_idx = human_indices[-2:]
        start_idx = last_two_human_idx[0]
        kept_messages = other_messages[start_idx:]

    return system_messages + kept_messages


def _convert_base_messages_to_tuples(messages: List[BaseMessage]) -> List[tuple]:
    """
    将 LangChain BaseMessage 列表转换为 (role, content) 元组列表
    """
    result = []
    for msg in messages:
        role = msg.type if hasattr(msg, 'type') else str(type(msg).__name__).lower()
        content = msg.content if hasattr(msg, 'content') else str(msg)
        result.append((role, content))
    return result


def _should_retry(result) -> bool:
    """判断是否需要重试"""
    has_invalid = hasattr(result, 'invalid_tool_calls') and result.invalid_tool_calls
    has_malformed = hasattr(result, 'content') and result.content and \
                    ("<tool_call>" in result.content or "</tool_call>" in result.content)
    return has_invalid or has_malformed


async def ainvoke_model_with_message(
    messages: List[BaseMessage],
    model_id: Optional[str] = None,
    tools: Optional[List[BaseTool]] = None
):
    """
    调用utils方法请求大模型，处理 tool_calls 和 invalid_tool_calls，并去除 think 块。
    返回模型调用结果和调用用时（毫秒）
    """
    import time
    if not model_id:
        model_id = model_invoke_config.get_model_id()
    if not model_id:
        raise ValueError("model_id error, please set model_id")

    # 截断消息以控制 token 数量
    # messages = _truncate_messages_by_token_limit(messages, DEFAULT_MESSAGES_TOKEN_LIMIT)

    # tuple_messages = _convert_base_messages_to_tuples(messages)
    max_retry = 5

    async def _call_with_tools():
        start_time = time.time()
        result = await ainvoke_llm_with_messages_and_tools(
            messages=messages,
            model_id=model_id,
            tools=tools
        )
        end_time = time.time()
        return result, (end_time - start_time) * 1000

    async def _call_without_tools():
        start_time = time.time()
        result = await ainvoke_llm_with_messages_and_tools(
            messages=messages,
            model_id=model_id
        )
        end_time = time.time()
        return result, (end_time - start_time) * 1000

    # 根据是否有 tools 选择调用函数
    call_func = _call_with_tools if tools else _call_without_tools

    # 调用模型，带重试逻辑
    result = None
    total_time = 0
    retries = 0
    while retries <= max_retry:
        call_result, call_time = await call_func()
        total_time += call_time

        # 去除 think 块
        if hasattr(call_result, 'content') and call_result.content:
            call_result.content = remove_think_blocks(call_result.content)

        # 检查是否需要重试
        if retries < max_retry and _should_retry(call_result):
            if hasattr(call_result, 'invalid_tool_calls') and call_result.invalid_tool_calls:
                logger.warning(f"---model-generate---: 大模型返回的 tool 调用的 invalid_tool_calls，尝试重新生成")
            else:
                logger.warning(f"---model-generate---: 大模型的 tool_call 错误生成在 result.content 中，尝试重新生成")
            retries += 1
            continue

        result = call_result
        break

    logger.debug(f"---model-generate---: 模型调用完成，总用时: {total_time:.2f}ms")
    return result, total_time


async def ainvoke_model_with_structured_output(
    messages: List[BaseMessage],
    response_type: Type[T],
    model_id: Optional[str] = None,
):
    """
    使用 with_structured_output 要求模型按照给定的 response_type 返回结构化结果。
    返回模型调用结果和调用用时（毫秒）
    """
    import time
    if not model_id:
        model_id = model_invoke_config.get_model_id()
    if not model_id:
        raise ValueError("model_id error, please set model_id")

    start_time = time.time()
    result = await ainvoke_llm_with_messages_structured_output(
        messages=_convert_base_messages_to_tuples(messages),
        model_id=model_id,
        response_type=response_type,
    )
    end_time = time.time()
    total_time = (end_time - start_time) * 1000

    logger.debug(f"---model-generate---: 结构化输出模型调用完成，总用时: {total_time:.2f}ms")
    return result, total_time

