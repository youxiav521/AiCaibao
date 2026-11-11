from __future__ import annotations

import asyncio
from typing import Optional, Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate

from src.agents.content_generate_agent.utils import _retry_with_backoff
from src.utils.model_factory import ModelFactory

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY_S = 2.0
DEFAULT_MAX_DELAY_S = 30.0


def _retry_model_id_for_gate(model_id: str) -> str:
    """
    `src.agents.content_generate_agent.utils._retry_with_backoff` 只对特定 model_id 开启重试；
    这里通过加前缀的方式确保重试逻辑对任意 model_id 生效，而无需修改原文件。
    """
    return f"model_bailian::{model_id}"


async def _with_timeout(coro, timeout_s: Optional[float]):
    if timeout_s is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"LLM call timed out after {timeout_s}s") from e


async def ainvoke_llm_with_prompt(
    prompt: str,
    *,
    model_id: str,
    timeout_s: Optional[float] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY_S,
    max_delay: float = DEFAULT_MAX_DELAY_S,
    **model_overrides,
) -> str:
    async def _call_llm():
        model = ModelFactory.get_model(key=model_id, **(model_overrides or {}))
        prompt_template = ChatPromptTemplate.from_messages([("human", "{prompt}")])
        chain = prompt_template | model
        resp = await chain.ainvoke({"prompt": prompt})
        return resp.content if hasattr(resp, "content") else str(resp)

    coro = _retry_with_backoff(
        _call_llm,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        model_id=_retry_model_id_for_gate(model_id),
    )
    return str(await _with_timeout(coro, timeout_s=timeout_s) or "")


async def ainvoke_llm_with_structured_output(
    prompt: str,
    *,
    model_id: str,
    response_type: Type[T],
    timeout_s: Optional[float] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY_S,
    max_delay: float = DEFAULT_MAX_DELAY_S,
    rate_limited: bool = True,
    **model_overrides,
) -> T:
    async def _call_llm():
        model = ModelFactory.get_model(key=model_id, rate_limited=rate_limited, **(model_overrides or {}))
        prompt_template = ChatPromptTemplate.from_messages([("human", "{prompt}")])
        if ModelFactory.is_deepseek_model_for_key(model_id):
            structured = model.with_structured_output(response_type, method="json_mode")
        else:
            structured = model.with_structured_output(response_type)
        chain = prompt_template | structured
        return await chain.ainvoke({"prompt": prompt})

    coro = _retry_with_backoff(
        _call_llm,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        model_id=_retry_model_id_for_gate(model_id),
    )
    return await _with_timeout(coro, timeout_s=timeout_s)
