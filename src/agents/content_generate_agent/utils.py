from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, List, Callable, Awaitable, TYPE_CHECKING, TypedDict
# ContextVar：上下文局部变量。同一进程内多个协程/任务并发时，各自有一份独立「当前值」，
# 互不覆盖；适合在本模块内传递「当前这一次内容生成任务」的统计对象，而不用全局变量或到处传参。
# 常用 API：.get() 读当前上下文值；.set(value) 写入并返回 token；.reset(token) 恢复到 set 之前。
from contextvars import ContextVar
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .models import QuestionTask
import json_repair
import json
import asyncio
import threading
import random


@dataclass
class ContentGenerateLlmTaskStats:
    """单次内容生成后台任务内的 LLM 统计（由 contextvars 挂到协程上下文）。"""

    # 仅保护下面两个计数器的自增，避免未来引入跨线程/让出控制权导致的竞态丢计数。
    # 临界区不包含 await，且只使用这一把锁，不与其它锁嵌套，避免死锁风险。
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # LLM 调用“单元”尝试次数：单次 ainvoke 计 1；abatch 计 n；发生重试时会再次累加（放大）。
    llm_api_attempts: int = 0
    # 与历史 [LLM调用统计] 口径一致：单次 ainvoke 计 1，abatch 计 n
    llm_scheduled_units: int = 0


# 当前协程是否处于「内容生成后台任务」的统计会话中：在会话内为 ContentGenerateLlmTaskStats 实例，否则为 None。
_CG_CONTENT_GENERATE_LLM_STATS: ContextVar[Optional[ContentGenerateLlmTaskStats]] = ContextVar(
    "cg_content_generate_llm_stats", default=None
)


# @asynccontextmanager：把「单入口 async 生成器函数」包装成异步上下文管理器，使其可用 `async with`。
# - 进入 `async with` 时：执行 yield 之前的代码（此处：创建 stats、set ContextVar）。
# - `async with` 块内拿到的对象：yield 出的值（此处：stats）。
# - 离开块时（正常结束、return、异常）：执行 finally，保证清理（此处：reset ContextVar，避免泄漏到其他协程任务）。
@asynccontextmanager
async def content_generate_llm_stats_session():
    """
    供内容生成路由层包裹单次后台任务，用于聚合 LLM 调用次数。
    与 utils 内统一 LLM 入口（经 _retry_with_backoff）配合计数。

    使用方式：`async with content_generate_llm_stats_session() as stats:`；
    等价于手写 `__aenter__` / `__aexit__` 的异步上下文管理器类，代码更短。
    """
    stats = ContentGenerateLlmTaskStats()
    token = _CG_CONTENT_GENERATE_LLM_STATS.set(stats)
    try:
        yield stats
    finally:
        _CG_CONTENT_GENERATE_LLM_STATS.reset(token)


def _log_llm_stats_before(mode: str, count: int) -> None:
    """mode: ainvoke | abatch"""
    bag = _CG_CONTENT_GENERATE_LLM_STATS.get()
    if bag is not None:
        with bag._lock:
            bag.llm_scheduled_units += count
    logger.debug(f"[LLM调用统计] ({mode})调用了{count}次LLM")


def _log_llm_stats_after(mode: str, count: int, success: int, failure: int) -> None:
    logger.debug(
        f"[LLM调用统计] ({mode})调用了{count}次LLM，成功了{success}次，失败了{failure}次"
    )


from functools import lru_cache
from jinja2 import Environment, FileSystemLoader
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.base.langfuse_observe import observe
from ...utils.model_factory import ModelFactory
from .config import create_default_configs
from loguru import logger

_CFG = create_default_configs()


def format_error_type_and_message(
    e: Exception,
    *,
    max_message_chars: int | None = 200,
) -> tuple[str, str]:
    """
    Extract (error_type, error_message) from exception for consistent logging.
    """
    err_type = type(e).__name__
    err_msg = str(e)
    if max_message_chars is not None:
        err_msg = err_msg[:max_message_chars]
    return err_type, err_msg


def log_error_type_and_message(
    *,
    logger,
    context: str,
    e: Exception,
    max_message_chars: int | None = 200,
    exc_info: bool = False,
) -> None:
    """
    Unified error logger:
    - always prints error_type and error_message (optionally truncated)
    - optionally includes traceback via exc_info=True
    """
    err_type, err_msg = format_error_type_and_message(
        e, max_message_chars=max_message_chars
    )
    logger.opt(exception=exc_info).error(
        "{} | error_type={} | error_message={}",
        context,
        err_type,
        err_msg,
    )


T = TypeVar('T')

# ============= Token utilities =============
from src.utils.tokenizer import Tokenizer, DEFAULT_MAX_TOKENS

# 模块级 tokenizer 实例，进程启动时初始化；tiktoken_cache/ 缺失则直接抛出异常。
_tokenizer = Tokenizer()


def get_default_tokenizer() -> Tokenizer:
    """返回模块级 Tokenizer 实例（cl100k_base 编码，向后兼容入口）。"""
    return _tokenizer


def count_tokens(text: str, tokenizer: Tokenizer | None = None) -> int:
    """计算文本的 token 数量。

    Args:
        text: 待计算的文本。
        tokenizer: 可选 Tokenizer 实例；未提供时使用模块默认（gpt-4o）。
    """
    tok = tokenizer if isinstance(tokenizer, Tokenizer) else _tokenizer
    return tok(text)


def truncate_to_token_limit(
    text: str,
    tokenizer: Tokenizer | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """将文本截断至不超过 max_tokens 个 token。

    Args:
        text: 待截断的文本。
        tokenizer: 可选 Tokenizer 实例；未提供时使用模块默认（gpt-4o）。
        max_tokens: token 上限。
    """
    tok = tokenizer if isinstance(tokenizer, Tokenizer) else _tokenizer
    return tok.limit(text, max_tokens)


def ensure_within_token_limit(
    context_text: str,
    tokenizer: Tokenizer | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """超限时截断并记录警告，否则原样返回。

    Args:
        context_text: 待检查的文本。
        tokenizer: 可选 Tokenizer 实例；未提供时使用模块默认（gpt-4o）。
        max_tokens: token 上限。
    """
    tok = tokenizer if isinstance(tokenizer, Tokenizer) else _tokenizer
    return tok.ensure(context_text, max_tokens)


# ============= 重试机制辅助函数 =============

def _is_connection_error(error: Exception) -> bool:
    """检查异常是否是连接错误（可重试的错误）"""
    error_type_name = type(error).__name__
    if "ConnectionError" in error_type_name or "ConnectError" in error_type_name:
        return True
    error_str = str(error).lower()
    connection_keywords = ["connection error", "connection refused", "connection reset",
                           "connect timeout", "name or service not known", "no route to host"]
    return any(keyword in error_str for keyword in connection_keywords)


def _is_rate_limit_error(error: Exception) -> bool:
    """
    检查异常是否是RPM或TPM限制错误（可重试的错误）
    
    Args:
        error: 异常对象
        
    Returns:
        bool: 如果是RPM或TPM限制错误则返回True
    """
    error_str = str(error).lower()
    error_repr = repr(error).lower()
    
    # 检查错误消息中是否包含RPM/TPM限制相关的关键词
    rate_limit_keywords = [
        "exceeded your current request limit",
        "limit_requests",
        "rate limit",
        "request limit",
        "too many requests",
        "rpm",
        "quota exceeded",
        "exceeded your current quota",
        # TPM/token-limit 相关关键词
        "token-limit",
        "token limit",
        "tpm",
        "token per minute",
        "tokens per minute",
    ]
    
    # 检查 HTTP 429 状态码（通常表示限流）
    if "429" in error_str or "429" in error_repr:
        return True
    
    for keyword in rate_limit_keywords:
        if keyword in error_str or keyword in error_repr:
            return True
    
    # 检查异常类型和属性
    if hasattr(error, 'type') and error.type == 'limit_requests':
        return True
    
    # 检查异常的状态码属性
    if hasattr(error, 'status_code') and error.status_code == 429:
        return True
    
    if hasattr(error, 'body'):
        try:
            body_str = str(error.body).lower()
            if any(keyword in body_str for keyword in rate_limit_keywords):
                return True
            # 检查 body 中是否包含 429 状态码
            if "429" in body_str:
                return True
        except:
            pass
    
    # 检查异常的错误代码属性
    if hasattr(error, 'code') and str(error.code) == "429":
        return True
    
    return False


def _is_service_unavailable_error(error: Exception) -> bool:
    """
    检查异常是否是 503 Service Temporarily Unavailable（可重试的错误）。
    """
    error_str = str(error).lower()
    error_repr = repr(error).lower()
    retryable_status_codes = {"500", "502", "503", "504"}
    transient_keywords = [
        "internal server error",
        "bad gateway",
        "service temporarily unavailable",
        "gateway timeout",
    ]
    if any(code in error_str or code in error_repr for code in retryable_status_codes):
        return True
    if any(keyword in error_str or keyword in error_repr for keyword in transient_keywords):
        return True
    if hasattr(error, "status_code") and str(error.status_code) in retryable_status_codes:
        return True
    if hasattr(error, "code") and str(error.code) in retryable_status_codes:
        return True
    if hasattr(error, "body"):
        try:
            body_str = str(error.body).lower()
            if any(code in body_str for code in retryable_status_codes) or any(
                keyword in body_str for keyword in transient_keywords
            ):
                return True
        except Exception:
            pass
    return False


def _is_length_limit_error(error: Exception) -> bool:
    """
    检查异常是否是输出长度限制错误（可重试的错误）。
    """
    error_str = str(error).lower()
    error_repr = repr(error).lower()

    length_limit_keywords = [
        "length limit was reached",
        "completion_tokens limit",
        "output length limit",
    ]

    for keyword in length_limit_keywords:
        if keyword in error_str or keyword in error_repr:
            return True

    if hasattr(error, "body"):
        try:
            body_str = str(error.body).lower()
            if any(keyword in body_str for keyword in length_limit_keywords):
                return True
        except Exception:
            pass

    return False


def _should_retry_for_model(model_id: str) -> bool:
    """
    判断是否应该为重试机制启用（暂时认为都应该进行重试）
    
    Args:
        model_id: 模型ID
        
    Returns:
        bool: 如果应该启用重试则返回True
    """
    return True #都应该进行重试
    # return model_id.startswith("model_bailian")

async def _retry_with_backoff(
    func,
    max_retries: int = 5,
    initial_delay: float = 20.0,
    max_delay: float = 300.0,
    backoff_factor: float = 2.0,
    model_id: Optional[str] = None,
    on_retry_wait: Optional[Callable[[], Awaitable[None]]] = None,
    attempt_units: int = 1,
):
    """
    带指数退避的重试机制，特别处理RPM和TPM限制错误
    
    Args:
        func: 要执行的异步函数
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        backoff_factor: 退避因子
        model_id: 模型ID，用于判断是否需要重试
        on_retry_wait: 可选的回调函数，在等待后、重试前调用（用于调整并发数等）
        
    Returns:
        函数执行结果
        
    Raises:
        最后一次尝试的异常
    """

    async def _invoke_once():
        bag = _CG_CONTENT_GENERATE_LLM_STATS.get()
        if bag is not None:
            with bag._lock:
                bag.llm_api_attempts += attempt_units
        return await func()

    if model_id and not _should_retry_for_model(model_id):
        logger.warning(f"模型({model_id})不需要重试，直接执行不重试")
        # 如果不是需要重试的模型，直接执行不重试
        return await _invoke_once()

    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await _invoke_once()
        except Exception as e:
            last_exception = e
            
            is_rate_limit = _is_rate_limit_error(e)
            is_connection = _is_connection_error(e)
            is_service_unavailable = _is_service_unavailable_error(e)
            # is_length_limit = _is_length_limit_error(e)

            if is_rate_limit or is_connection or is_service_unavailable:
                if is_service_unavailable:
                    error_category = "503服务暂不可用"
                elif is_rate_limit:
                    error_category = "RPM/TPM限制"
                # elif is_length_limit:
                #     error_category = "输出长度限制"
                else:
                    error_category = "连接"
                if attempt < max_retries:
                    # 添加不超过10%的随机抖动，避免所有重试在同一时间进行
                    jitter = random.uniform(-0.1, 0.1)  # ±10% 的随机抖动
                    # 连接错误使用较短的重试延迟；429/503 使用完整退避
                    current_delay = delay if (is_rate_limit or is_service_unavailable) else min(delay, 5.0)
                    delay_with_jitter = current_delay * (1 + jitter)
                    logger.warning(
                        "检测到{}错误（模型: {}），当前第 {}/{} 次重试，"
                        "等待 {:.1f} 秒后将进行第 {} 次调用尝试（基础延迟: {:.1f} 秒）。",
                        error_category,
                        model_id,
                        attempt + 1,
                        max_retries,
                        delay_with_jitter,
                        attempt + 2,
                        current_delay,
                    )
                    await asyncio.sleep(delay_with_jitter)
                    
                    # 等待后调用回调函数（用于调整并发数等）
                    if on_retry_wait:
                        await on_retry_wait()
                    
                    delay = min(delay * backoff_factor, max_delay)
                    continue
                else:
                    log_error_type_and_message(
                        logger=logger,
                        context=f"{error_category}错误，已达到最大重试次数 {max_retries}（模型: {model_id}）。最后一次错误",
                        e=e,
                        exc_info=False,
                        max_message_chars=200,
                    )
            else:
                # 非可重试错误，直接抛出
                log_error_type_and_message(
                    logger=logger,
                    context=f"LLM调用出现非RPM/TPM限制错误（模型: {model_id}）",
                    e=e,
                    exc_info=False,
                    max_message_chars=200,
                )
                raise
    
    # 如果所有重试都失败，抛出最后一次的异常
    raise last_exception

# **异步**使用统一的 ChatModelFactory 按给定 model_id 以单条字符串 prompt 调用模型
@observe(name="nodes._ainvoke_llm_with_prompt", as_type="span")
async def ainvoke_llm_with_prompt(
    prompt: str,
    model_id: Optional[str] = None,
) -> str:
    _log_llm_stats_before("ainvoke", 1)
    try:
        async def _call_llm():
            model = ModelFactory.get_model(key=model_id)
            prompt_template = ChatPromptTemplate.from_messages([
                ("human", "{prompt}")
            ])
            chain = prompt_template | model
            resp = await chain.ainvoke({"prompt": prompt})
            return resp.content if hasattr(resp, "content") else str(resp)

        out = await _retry_with_backoff(_call_llm, model_id=model_id)
        _log_llm_stats_after("ainvoke", 1, 1, 0)
        return out
    except Exception:
        _log_llm_stats_after("ainvoke", 1, 0, 1)
        raise

# **异步**使用统一的 ChatModelFactory 按给定 model_id 以多轮对话消息列表调用模型
@observe(name="nodes._ainvoke_llm_with_messages", as_type="span")
async def ainvoke_llm_with_messages(
    messages: List[tuple],
    model_id: str,
) -> str:
    """
    异步多轮对话调用模型，支持系统消息、用户消息和助手消息。
    这里直接将已渲染好的消息列表传给模型，避免再经过 ChatPromptTemplate 的变量插值，
    从而允许消息内容中安全地包含花括号（如 JSON 示例）。
    """

    # 将 (role, content) 元组转换为 LangChain 的消息对象
    def _convert_messages(raw_messages: List[tuple]):
        lc_messages = []
        for role, content in raw_messages:
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role in ("human", "user"):
                lc_messages.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                lc_messages.append(AIMessage(content=content))
            else:
                # 未知角色，一律按人类消息处理，避免抛异常
                logger.warning(f"未知角色: {role}，按人类消息处理: {content}")
                lc_messages.append(HumanMessage(content=str(content)))
        return lc_messages

    _log_llm_stats_before("ainvoke", 1)
    try:
        async def _call_llm():
            model = ModelFactory.get_model(key=model_id)
            lc_messages = _convert_messages(messages)
            resp = await model.ainvoke(lc_messages)
            return resp.content if hasattr(resp, "content") else str(resp)

        out = await _retry_with_backoff(_call_llm, model_id=model_id)
        _log_llm_stats_after("ainvoke", 1, 1, 0)
        return out
    except Exception:
        _log_llm_stats_after("ainvoke", 1, 0, 1)
        raise

# **异步**使用统一的 ChatModelFactory 按给定 model_id 以多轮对话消息列表调用模型，支持工具调用
@observe(name="nodes._ainvoke_llm_with_messages_and_tools", as_type="span")
async def ainvoke_llm_with_messages_and_tools(
    messages: List[Any],
    model_id: str,
    tools: Optional[List[BaseTool]] = None,
) -> Any:
    """
    异步多轮对话调用模型，支持系统消息、用户消息和助手消息，并绑定工具供模型调用。
    """

    # 将 (role, content) 元组转换为 LangChain 的消息对象
    def _convert_messages(raw_messages: List[tuple]):
        lc_messages = []
        for role, content in raw_messages:
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role in ("human", "user"):
                lc_messages.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                lc_messages.append(AIMessage(content=content))
            else:
                # 未知角色，一律按人类消息处理，避免抛异常
                logger.warning(f"未知角色: {role}，按人类消息处理: {content}")
                lc_messages.append(HumanMessage(content=str(content)))
        return lc_messages
    _log_llm_stats_before("ainvoke", 1)
    try:
        async def _call_llm():
            model = ModelFactory.get_model(key=model_id)
            # 检查 messages 是否已经是 LangChain 消息列表（检查第一个元素）
            if messages and hasattr(messages[0], 'type') and hasattr(messages[0], 'content'):
                logger.info(f"messages已经是 LangChain 消息列表，直接使用")
                lc_messages = messages
            else:
                logger.info(f"messages不是 LangChain 消息列表，需要转换")
                lc_messages = _convert_messages(messages)
            # 绑定工具并调用模型
            if tools:
                model_with_tools = model.bind_tools(tools)
            else:
                model_with_tools = model
            resp = await model_with_tools.ainvoke(lc_messages)
            return resp

        out = await _retry_with_backoff(_call_llm, model_id=model_id)
        _log_llm_stats_after("ainvoke", 1, 1, 0)
        return out
    except Exception:
        _log_llm_stats_after("ainvoke", 1, 0, 1)
        raise

# **异步 Batch** 方法，避免同步批量调用耗时过长导致超时
@observe(name="nodes.abatch_llm_with_prompt", as_type="span")
async def abatch_llm_with_prompt(
    prompts: List[str],
    model_id: Optional[str] = None,
) -> List[str]:
    """
    异步批量调用 LLM，底层模型已实现并发控制和重试机制。
    Args:
        prompts: 提示词列表
        model_id: 模型 ID
    """
    n = len(prompts)
    if n == 0:
        logger.warning("提示词列表为空，直接返回空列表")
        return []

    _log_llm_stats_before("abatch", n)

    # 并发提交多个 ainvoke，底层 RateLimitedChatModel 会处理并发控制和重试
    tasks = [
        ainvoke_llm_with_prompt(
            prompt=p,
            model_id=model_id,
        )
        for p in prompts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success = sum(1 for r in results if not isinstance(r, BaseException))
    failure = n - success
    _log_llm_stats_after("abatch", n, success, failure)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return results


def _with_structured_output_compat(model: Any, model_id: str, response_type: Type[T]) -> Any:
    """DeepSeek 兼容接口不支持默认的 json_schema response_format，需使用 json_mode。"""
    if ModelFactory.is_deepseek_model_for_key(model_id):
        return model.with_structured_output(response_type, method="json_mode")
    return model.with_structured_output(response_type)


@observe(name="nodes._ainvoke_llm_with_structured_output", as_type="span")
async def ainvoke_llm_with_structured_output(
    prompt: str,
    model_id: str,
    response_type: Type[T],
) -> T:
    """异步结构化输出调用模型，返回指定类型对象（单条字符串 prompt，兼容旧逻辑）"""
    _log_llm_stats_before("ainvoke", 1)
    try:
        async def _call_llm():
            model = ModelFactory.get_model(key=model_id)
            # Some gateways require the word "json" in messages when response_format=json_object is used.
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Return a valid json object only."),
                ("human", "{prompt}")
            ])
            chain = prompt_template | _with_structured_output_compat(model, model_id, response_type)
            resp = await chain.ainvoke({"prompt": prompt})
            return resp

        out = await _retry_with_backoff(_call_llm, model_id=model_id)
        _log_llm_stats_after("ainvoke", 1, 1, 0)
        return out
    except Exception:
        _log_llm_stats_after("ainvoke", 1, 0, 1)
        raise


@observe(name="nodes._ainvoke_llm_with_messages_structured_output", as_type="span")
async def ainvoke_llm_with_messages_structured_output(
    messages: List[Any],
    model_id: str,
    response_type: Type[T],
) -> T:
    """
    异步多轮对话 + 结构化输出调用模型，返回指定类型对象。
    与 `ainvoke_llm_with_messages` 一样，直接使用消息对象调用模型，
    避免 ChatPromptTemplate 对花括号进行变量解析。
    """

    def _convert_messages(raw_messages: List[tuple]):
        lc_messages = []
        for role, content in raw_messages:
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role in ("human", "user"):
                lc_messages.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                lc_messages.append(AIMessage(content=content))
            else:
                logger.warning(f"未知角色: {role}，按人类消息处理: {content}")
                lc_messages.append(HumanMessage(content=str(content)))
        return lc_messages

    def _contains_json_word(raw_messages: List[Any]) -> bool:
        for item in raw_messages:
            if isinstance(item, tuple) and len(item) >= 2:
                content = str(item[1] or "")
            else:
                content = str(item or "")
            if "json" in content.lower():
                return True
        return False

    _log_llm_stats_before("ainvoke", 1)
    try:
        async def _call_llm():
            model = ModelFactory.get_model(key=model_id)
            structured_model = _with_structured_output_compat(model, model_id, response_type)
            normalized_messages = list(messages)
            if not _contains_json_word(normalized_messages):
                normalized_messages.insert(0, ("system", "Return a valid json object only."))
            lc_messages = _convert_messages(normalized_messages)
            resp = await structured_model.ainvoke(lc_messages)
            return resp

        out = await _retry_with_backoff(_call_llm, model_id=model_id)
        _log_llm_stats_after("ainvoke", 1, 1, 0)
        return out
    except Exception:
        _log_llm_stats_after("ainvoke", 1, 0, 1)
        raise

# **异步 Batch** 方法，避免同步批量调用耗时过长导致超时
@observe(name="nodes.abatch_llm_with_structured_output", as_type="span")
async def abatch_llm_with_structured_output(
    prompts: List[str],
    model_id: str,
    response_type: Type[T],
) -> List[T]:
    """
    异步批量使用结构化输出调用模型，返回指定类型的对象列表。
    底层模型已实现并发控制和重试机制。
    Args:
        prompts: 输入提示词列表
        model_id: 模型ID
        response_type: 期望的返回类型（Pydantic模型类）
    
    Returns:
        指定类型的结构化响应对象列表
    """
    n = len(prompts)
    if n == 0:
        logger.warning("提示词列表为空，直接返回空列表")
        return []

    _log_llm_stats_before("abatch", n)

    # 并发提交多个 ainvoke，底层 RateLimitedChatModel 会处理并发控制和重试
    tasks = [
        ainvoke_llm_with_structured_output(
            prompt=p,
            model_id=model_id,
            response_type=response_type,
        )
        for p in prompts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success = sum(1 for r in results if not isinstance(r, BaseException))
    failure = n - success
    _log_llm_stats_after("abatch", n, success, failure)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return results


# 尝试从 LLM 输出中提取 JSON 片段：优先解析 ```json 代码块，其次回退到花括号范围
def extract_json(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    # 优先解析 ```json 代码块
    start = lower.find("```json")
    if start != -1:
        real_start = text.find("```", start) + 3
        end = text.find("```", real_start)
        return text[real_start:end].strip() if end != -1 else text[real_start:].strip()
    # 其次回退到花括号范围
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        return text[l:r + 1]
    return ""

def repair_json_output(content: str) -> str:
    """
    Repair and normalize JSON output.

    Args:
        content (str): String content that may contain JSON

    Returns:
        str: Repaired JSON string, or original content if not JSON
    """
    content = content.strip()
    try:
        # Try to repair and parse JSON
        repaired_content = json_repair.loads(content)
        if not isinstance(repaired_content, dict) and not isinstance(
            repaired_content, list
        ):
            logger.warning("Repaired content is not a valid JSON object or array.")
            return content
        content = json.dumps(repaired_content, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"JSON repair failed: {e}")

    return content

# 回答语言 locale -> 模型指令文案（用于 prompt 中要求答案使用何种语言）
ANSWER_LANGUAGE_INSTRUCTIONS: Dict[str, str] = {
    "zh_CN": "请用中文回答。",
    "en_US": "Please answer in English.",
}
SUPPORTED_ANSWER_LOCALES = frozenset(ANSWER_LANGUAGE_INSTRUCTIONS.keys())


class AnsweringPromptLocaleVars(TypedDict):
    related_data_tag: str
    related_data_label: str


ANSWERING_PROMPT_LOCALE_VARS: Dict[str, AnsweringPromptLocaleVars] = {
    "zh_CN": {
        "related_data_tag": "相关数据计算结果",
        "related_data_label": "相关数据计算结果",
    },
    "en_US": {
        "related_data_tag": "related_data_calculation_results",
        "related_data_label": "related data calculation results",
    },
}


LOCALIZED_MESSAGES: Dict[str, Dict[str, str]] = {
    "zh_CN": {
        "reporter_generation_error": "[ERROR] 生成答案时出现错误，请稍后重试。{error}",
        "content_generation_example_detailed_answer": "公司产品市场营销情况及可持续性分析显示，经营收入从2020年的57.17亿元[doc1]增长至2024年的85.40亿元[doc2]，年复合增长率约为8,6%[doc3]，表明市场需求稳定增长，产品具有较强的市场竞争力。",
        "content_generation_example_simple_answer": "直接采购模式[doc1]",
        "unanswerable_example_answer": "信息不足，请再次重试或需补充资料。",
        "sliding_windows_empty": "无法创建滑动窗口，上下文为空。",
        "retrieval_result_missing": "未找到该问题的检索结果，无法生成答案。",
        "retrieval_result_empty": "检索结果为空，无法生成答案。",
        "retrieval_stream_incomplete": "检索流程未产出该问题的完成信号，无法生成答案。",
        "sliding_window_processing_error": "窗口 {window_id} 处理失败: {error}",
        "sliding_windows_processing_error": "[ERROR] 滑动窗口处理失败: {error}",
        "coder_processing_error": "[ERROR] Coder节点处理时发生错误: {error}",
        "omitted_context": "（省略）",
        "window_answer_label": "【窗口 {window_id} 答案】",
        "generic_question_processing_error": "[ERROR] 处理问题时发生错误: {error}",
        "api_task_submitted": "任务已提交，正在后台处理",
        "api_operation_success": "操作成功",
        "api_operation_failed": "操作失败: {error}",
        "api_stream_error": "流式处理失败: {error}",
        "analysis_previous_window_result": "上一窗口的完整分析结果如下，请在此基础上继续或补充：\n{answer}",
        "model_service_error": "抱歉，在调用大模型服务时遇到了问题。可能是输入内容过长或服务暂时不可用，请稍后重试或尝试简化问题。",
        "callback_start": "开始处理",
        "callback_empty_input": "输入内容为空，无法处理。",
        "callback_validation_failed": "内容列表校验失败，无法继续解析流程：{error}",
        "callback_success": "内容生成成功，处理了 {count} 个问题",
        "callback_unknown_error": "内容生成过程中发生未知错误: {error}",
        "table_result_empty": "结果为空",
        "table_unknown_result_type": "未知结果类型",
        "table_unknown_error": "未知错误",
        "table_retrieval_empty": "表格 {table_name} 检索为空，无法进行填充",
        "table_fill_error": "表格填充出现错误：{error}",
        "table_missing_related_info": "缺失相关信息：{description}",
        "table_missing_related_info_default_detail": "检索结果中缺少完成该表格所需的关键信息。",
        "table_no_fill_result": "没有表格填充结果，请检查输入",
        "table_missing_related_info_prefix": "缺少相关信息：",
        "table_generated_description_label": "生成描述",
        "table_execution_result_label": "执行结果",
        "table_missing_key_info": "缺少表格关键信息",
        "table_read_saved_error": "读取保存的表格失败: {error}",
        "table_history_previous_output": "之前已经完成的填表输出结果：\n{output}\n请根据之前的填表结果继续填表",
        "table_fill_question_suffix": "\n请按照如下表格结构填充数据：",
        "table_agent_failed": "[ERROR] 表格回答生成失败: {error}",
    },
    "en_US": {
        "reporter_generation_error": "[ERROR] An error occurred while generating the answer. Please try again later. {error}",
        "content_generation_example_detailed_answer": "The analysis of the company's product marketing and sustainability shows that operating revenue increased from RMB 5.717 billion in 2020 [doc1] to RMB 8.540 billion in 2024 [doc2], with a compound annual growth rate of about 8.6% [doc3], indicating stable market demand growth and strong product competitiveness.",
        "content_generation_example_simple_answer": "Direct procurement model [doc1]",
        "unanswerable_example_answer": "The information provided is incomplete. Please retry or submit supplementary materials.",
        "sliding_windows_empty": "Unable to create sliding windows because the context is empty.",
        "retrieval_result_missing": "No retrieval result was found for this question, so an answer cannot be generated.",
        "retrieval_result_empty": "The retrieval result is empty, so an answer cannot be generated.",
        "retrieval_stream_incomplete": "The retrieval workflow did not emit a completion signal for this question, so an answer cannot be generated.",
        "sliding_window_processing_error": "Window {window_id} processing failed: {error}",
        "sliding_windows_processing_error": "[ERROR] Sliding-window processing failed: {error}",
        "coder_processing_error": "[ERROR] An error occurred while processing the coder node: {error}",
        "omitted_context": "[omitted]",
        "window_answer_label": "[Window {window_id} Answer]",
        "generic_question_processing_error": "[ERROR] An error occurred while processing the question: {error}",
        "api_task_submitted": "The task has been submitted and is being processed in the background.",
        "api_operation_success": "Operation succeeded.",
        "api_operation_failed": "Operation failed: {error}",
        "api_stream_error": "Stream processing failed: {error}",
        "analysis_previous_window_result": "The complete analysis result from the previous window is below. Continue or extend the analysis based on it:\n{answer}",
        "model_service_error": "Sorry, we ran into an issue while calling the model service. The input may be too long, or the service may be temporarily unavailable. Please try again later or simplify the question.",
        "callback_start": "Processing started",
        "callback_empty_input": "The input content is empty and cannot be processed.",
        "callback_validation_failed": "Content validation failed, unable to continue: {error}",
        "callback_success": "Content generation completed successfully. Processed {count} questions.",
        "callback_unknown_error": "An unexpected error occurred during content generation: {error}",
        "table_result_empty": "The result is empty.",
        "table_unknown_result_type": "Unknown result type.",
        "table_unknown_error": "Unknown error.",
        "table_retrieval_empty": "No relevant information was retrieved for table {table_name}, so the table cannot be filled.",
        "table_fill_error": "An error occurred while filling the table: {error}",
        "table_missing_related_info": "Missing relevant information: {description}",
        "table_missing_related_info_default_detail": "The retrieved context does not contain the key information required to complete this table.",
        "table_no_fill_result": "No table fill result was generated. Please check the input.",
        "table_missing_related_info_prefix": "Missing relevant information:",
        "table_generated_description_label": "Generated description",
        "table_execution_result_label": "Execution result",
        "table_missing_key_info": "Missing key table information.",
        "table_read_saved_error": "Failed to read the saved table: {error}",
        "table_history_previous_output": "Previously completed table-fill output:\n{output}\nPlease continue filling the table based on the previous result.",
        "table_fill_question_suffix": "\nPlease fill the data according to the following table structure:",
        "table_agent_failed": "[ERROR] Failed to generate the table answer: {error}",
    },
}


LOCALE_ALIASES: Dict[str, str] = {
    "zh": "zh_CN",
    "en": "en_US",
}


def _normalize_locale_candidate(locale: Optional[str]) -> Optional[str]:
    """Normalize a single locale candidate and return None when unsupported."""
    if not locale or not locale.strip():
        return None

    raw = locale.strip().replace("-", "_")
    lowered_raw = raw.lower()
    if lowered_raw in LOCALE_ALIASES:
        normalized = LOCALE_ALIASES[lowered_raw]
    else:
        parts = raw.split("_", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            normalized = f"{parts[0].lower()}_{parts[1].upper()}"
        else:
            return None

    if normalized not in SUPPORTED_ANSWER_LOCALES:
        return None

    return normalized


def normalize_locale(locale: Optional[str], default: str = "zh_CN") -> str:
    """
    Normalize locale strings such as en-US / en_us / EN-us to a stable form.
    Unknown locales fall back to the provided default.
    """
    if not locale or not locale.strip():
        logger.warning("未指定 locale，回退为默认语言")
        return default

    normalized = _normalize_locale_candidate(locale)
    if normalized is None:
        logger.warning(f"不支持的 locale: {locale}，回退为 {default}")
        return default

    return normalized


def resolve_answer_locale(accept_language: Optional[str], default: str = "zh_CN") -> str:
    """
    Resolve locale from either a single locale token or a standard Accept-Language header.

    Selection rule:
    - Prefer higher q value.
    - For equal q values, prefer earlier entries.
    - Return the first supported locale among zh_CN / en_US.
    - Fall back to default when nothing matches.
    """
    if not accept_language or not accept_language.strip():
        logger.warning("未指定 Accept-Language，回退为默认语言")
        return default

    normalized = _normalize_locale_candidate(accept_language)
    if normalized is not None:
        return normalized

    candidates: List[tuple[float, int, str]] = []
    for index, raw_part in enumerate(accept_language.split(",")):
        part = raw_part.strip()
        if not part:
            continue

        lang_range, *params = [segment.strip() for segment in part.split(";") if segment.strip()]
        q_value = 1.0
        for param in params:
            key, sep, value = param.partition("=")
            if sep and key.strip().lower() == "q":
                try:
                    q_value = float(value.strip())
                except ValueError:
                    q_value = 0.0
                break

        if q_value <= 0:
            continue
        candidates.append((q_value, index, lang_range))

    for _, _, candidate in sorted(candidates, key=lambda item: (-item[0], item[1])):
        normalized = _normalize_locale_candidate(candidate)
        if normalized is not None:
            return normalized

    logger.warning(f"Accept-Language 未命中受支持的 locale: {accept_language}，回退为 {default}")
    return default


def get_answer_language_instruction(locale: str) -> str:
    """
    根据 locale（如 zh_CN / en_US）返回用于 prompt 的「回答语言」说明文案。
    未知 locale 时回退为中文。
    """
    normalized = resolve_answer_locale(locale)
    result = ANSWER_LANGUAGE_INSTRUCTIONS.get(normalized, ANSWER_LANGUAGE_INSTRUCTIONS["zh_CN"])
    logger.info(f"回答语言指令: {result}")
    return result


def get_answering_prompt_locale_vars(locale: str) -> AnsweringPromptLocaleVars:
    normalized = resolve_answer_locale(locale)
    return ANSWERING_PROMPT_LOCALE_VARS.get(
        normalized,
        ANSWERING_PROMPT_LOCALE_VARS["zh_CN"],
    )


def get_localized_message(message_key: str, locale: Optional[str] = None, **kwargs: Any) -> str:
    """Return a localized runtime message string."""
    normalized = resolve_answer_locale(locale)
    locale_messages = LOCALIZED_MESSAGES.get(normalized, LOCALIZED_MESSAGES["zh_CN"])
    template = locale_messages.get(message_key) or LOCALIZED_MESSAGES["zh_CN"].get(message_key)
    if template is None:
        raise KeyError(f"Unknown localized message key: {message_key}")
    return template.format(**kwargs)

def build_prompt_state(prompt_family: str, locale: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Build runtime state for shared prompt template rendering."""
    _ = prompt_family
    _ = locale
    return dict(kwargs)


def contains_cjk(text: Optional[str]) -> bool:
    """Return whether the text contains CJK Unified Ideographs."""
    if not text:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


@lru_cache(maxsize=1) # 第一次调用时创建 Jinja2 Environment 对象，后续调用直接返回缓存的同一个对象
def _get_jinja_env() -> Environment:
    """获取 Jinja2 环境，加载当前目录下的 prompts 文件夹。"""
    current_dir = Path(__file__).parent
    prompts_dir = current_dir / "prompts"
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        autoescape=False,
        keep_trailing_newline=True,
    )


def _resolve_prompt_template_name(prompt_name: str, locale: Optional[str] = None) -> str:
    prompts_dir = Path(__file__).parent / "prompts"
    if locale:
        normalized_locale = normalize_locale(locale)
        localized_rel_path = Path(normalized_locale) / f"{prompt_name}.md"
        if (prompts_dir / localized_rel_path).exists():
            return localized_rel_path.as_posix()
    return f"{prompt_name}.md"


def apply_prompt_template(prompt_name: str, state: Dict[str, Any], locale: Optional[str] = None) -> str:
    """
    使用 Jinja2 渲染 prompts 目录下的 .md 模板，返回渲染后的字符串。

    Args:
        prompt_name: 模板文件名（不含 .md）
        state: 用于渲染模板的变量字典
        configurable: 可选的配置对象（若为 dataclass 将并入渲染变量）
        locale: 可选 locale；若存在对应的 prompts/<locale>/<prompt_name>.md，则优先使用本地化模板

    Returns:
        渲染后的系统提示词字符串
    """
    # 渲染模板
    try:
        env = _get_jinja_env()
        template_name = _resolve_prompt_template_name(prompt_name, locale=locale)
        template = env.get_template(template_name)
        system_prompt: str = template.render(**state)
        return system_prompt
    except FileNotFoundError:
        current_dir = Path(__file__).parent
        raise FileNotFoundError(f"Prompt模板文件不存在: {current_dir / 'prompts' / (prompt_name + '.md')}")
    except Exception as e:
        raise ValueError(f"模板渲染失败: {e}")


async def generate_retrieval_query_from_description_and_prompt(
    description: str,
    prompt: Optional[str],
    model_id: str,
    locale: Optional[str] = None,
) -> str:
    """
    使用大模型从问题的 description 与 prompt 中提炼适合作为检索使用的 query。
    
    - 调用方式与本模块中其它 LLM 调用保持一致（ChatModelFactory + ainvoke_llm_with_prompt）
    - 若调用失败或模型返回结果为空，则回退为原始 description
    """
    desc = (description or "").strip()
    extra = (prompt or "").strip()
    # 若两者都为空，直接返回空字符串，避免无意义调用
    if not desc and not extra:
        return ""

    if not model_id:
        logger.error("未指定模型ID，无法生成检索 query")
        return ""

    # 使用 Jinja2 模板构建用于 LLM 的完整提示词
    normalized_locale = resolve_answer_locale(locale) if locale else "zh_CN"
    logger.info(
        f"开始生成 retrieval_query: locale={normalized_locale}, "
        f"description_len={len(desc)}, prompt_len={len(extra)}"
    )

    try:
        full_prompt = apply_prompt_template(
            "retrieval_query_generation",
            {
                "description": desc,
                "prompt": extra,
            },
            locale=normalized_locale,
        )
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="渲染检索 query 模板失败，将回退使用原始 description",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        return desc or extra

    try:
        raw_output = await ainvoke_llm_with_prompt(
            prompt=full_prompt,
            model_id=model_id,
        )
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context=(
                "调用大模型生成检索 query 失败，将回退使用原始 description"
                f" locale={normalized_locale}"
            ),
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        return desc or extra

    if not raw_output:
        logger.error("模型返回结果为空，无法生成检索 query")
        return desc or extra

    # 统一为字符串并去掉首尾空白
    candidate = str(raw_output).strip()

    # 若模型误返回多行，只取第一行作为最终 query
    if "\n" in candidate:
        logger.warning(f"模型返回结果为多行: {candidate}，只取第一行作为最终 query")
        candidate = candidate.splitlines()[0].strip()

    # 最终兜底：如果为空，则回退到 description 或 prompt
    if not candidate:
        logger.error("模型返回结果为空，无法生成检索 query")
        return desc or extra

    return candidate


async def ensure_retrieval_queries_for_tasks(
    question_tasks: List["QuestionTask"],
    model_id: str,
    locale: Optional[str] = None,
) -> None:
    """
    在检索开始前，为所有 QuestionTask 统一生成 retrieval_query（若尚未设置）。
    使用大模型从 description + prompt 提炼检索用 query，供整个检索阶段统一使用。
    可在内容生成 agent、表格 agent 等所有流程入口调用，保证行为一致。
    """
    from .models import QuestionTask

    # 过滤出需要生成 retrieval_query 的任务
    pending_tasks: List[QuestionTask] = [
        t for t in question_tasks if not getattr(t, "retrieval_query", None)
    ]
    if not pending_tasks:
        return

    normalized_locale = resolve_answer_locale(locale) if locale else "zh_CN"
    logger.info(
        f"批量生成 retrieval_query: pending_count={len(pending_tasks)}, locale={normalized_locale}"
    )

    async def _generate_for_single_task(task: QuestionTask) -> None:
        if getattr(task, "retrieval_query", None):
            return
        try:
            task.retrieval_query = await generate_retrieval_query_from_description_and_prompt(
                description=task.description,
                prompt=task.prompt or "",
                model_id=model_id,
                locale=normalized_locale,
            )
        except Exception as e:
            logger.warning(
                f"问题 {task.id} 生成检索 query 失败，将使用 description 作为检索 query。错误: {e}"
            )
            task.retrieval_query = task.description

    await asyncio.gather(*[_generate_for_single_task(t) for t in pending_tasks])


# 简单回答判定关键词：questionTask.prompt 中包含任一词组时触发简短作答
_SIMPLE_ANSWER_KEYWORDS = ("简单回答", "简要回答")


def judge_simple_question(prompt: Optional[str] = None) -> bool:
    """
    判断问题是否需要简短作答（词组/短句）。
    规则：questionTask 的 prompt 参数中包含「请简单回答」「请简要回答」等关键词组时触发。
    """
    if prompt is None or not str(prompt).strip():
        return False
    text = str(prompt).strip()
    return any(kw in text for kw in _SIMPLE_ANSWER_KEYWORDS)
