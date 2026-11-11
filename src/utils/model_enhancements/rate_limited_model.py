from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import time
import uuid

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.tools import BaseTool
from loguru import logger
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential

from .llm_metrics import MetricsCallbackHandler, get_current_metrics_context, get_current_llm_call_id
from .resource_limiter import ConcurrencyLimiter, RateLimiter
import asyncio
import threading

def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否应该重试，避免对逻辑错误无限重试"""
    error_str = str(exc).lower()
    error_type = type(exc).__name__

    # 不重试的错误类型：参数错误、类型错误、值错误等编程错误
    non_retryable = (TypeError, ValueError, AttributeError, NotImplementedError)
    if isinstance(exc, non_retryable):
        return False

    # 429 限流 / 503 服务不可用 / 连接错误 → 重试
    if any(code in error_str for code in ("429", "503", "rate_limit", "connection")):
        return True

    # 其他网络/服务端错误 → 重试
    retryable_types = ("ratelimiterror", "serviceunavailable", "connectionerror",
                       "timeout", "apierror", "openai")
    if any(t in error_type.lower() for t in retryable_types):
        return True

    return False


class RateLimitedChatModel(BaseChatModel):
    """带并发控制、限流和重试功能的 ChatModel 包装器

    设计要点:
    1. 继承 BaseChatModel，保持与 LangChain 接口兼容
    2. 集成分布式并发信号量（concurrency 控制）
    3. 集成 RPS/RPM 分布式限流（基于 limits 库）
    4. 集成 tenacity 指数退避重试（仅重试可恢复错误）
    5. 通过 agenerate 的 callbacks 参数注入 MetricsCallbackHandler
    """
    _internal_model: Any
    _model_key: str
    _provider_name: str
    _concurrency: Optional[int]
    _rps: Optional[int]
    _rpm: Optional[int]
    _max_retries: int
    _retry_initial_delay: float
    _max_wait: float

    def __init__(
        self,
        model: BaseChatModel,
        model_key: str,
        provider_name: str,
        concurrency: Optional[int] = None,
        rps: Optional[int] = None,
        rpm: Optional[int] = None,
        max_retries: int = 5,
        max_wait: float = 600.0,
        retry_initial_delay: float = 20.0,
    ):
        super().__init__()
        object.__setattr__(self, "_internal_model", model)
        object.__setattr__(self, "_model_key", model_key)
        object.__setattr__(self, "_provider_name", provider_name)
        object.__setattr__(self, "_concurrency", concurrency)
        object.__setattr__(self, "_rps", rps)
        object.__setattr__(self, "_rpm", rpm)
        object.__setattr__(self, "_max_retries", max_retries)
        object.__setattr__(self, "_max_wait", max_wait)
        object.__setattr__(self, "_retry_initial_delay", retry_initial_delay)

    @property
    def _model(self) -> BaseChatModel:
        return self._internal_model

    @property
    def _llm_type(self) -> str:
        return getattr(self._model, "_llm_type", "chat_model")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        # BaseChatModel.dict() 将此作为 property 访问，必须是 @property 返回 dict
        params = getattr(self._model, "_identifying_params", {})
        if callable(params):
            params = params()
        return params if isinstance(params, dict) else {}

    def __getattr__(self, name: str) -> Any:
        """转发底层模型的公共属性访问（跳过已覆盖的方法）"""
        if name.startswith("_") or name in {"generate", "agenerate", "ainvoke", "invoke"}:
            raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")
        return getattr(self._model, name)

    async def ainvoke(
        self,
        input: Any,
        config=None,
        **kwargs: Any,
    ) -> Any:
        """异步调用：如果 _model 是 Runnable，直接调用其 ainvoke"""
        # 判断：如果 _model 不是 BaseChatModel 实例（如 bind_tools/with_structured_output 后的 RunnableBinding）
        # 则直接走 ainvoke 路径，否则走父类的 ainvoke -> agenerate 路径
        if not isinstance(self._model, BaseChatModel) and hasattr(self._model, "ainvoke"):
            # 在 callbacks 层注入 MetricsCallbackHandler
            metrics_callback = MetricsCallbackHandler(
                model_key=self._model_key,
                provider=self._provider_name,
            )

            # 处理 config/callbacks 参数，注入 metrics callback
            if config is None:
                config = {"callbacks": [metrics_callback]}
            elif isinstance(config, dict):
                callbacks = config.get("callbacks")
                if callbacks is None:
                    config["callbacks"] = [metrics_callback]
                elif hasattr(callbacks, "handlers"):  # CallbackManager 对象
                    callbacks.handlers.append(metrics_callback)
                    if hasattr(callbacks, "inheritable_handlers"):
                        callbacks.inheritable_handlers.append(metrics_callback)
                elif isinstance(callbacks, list):
                    config["callbacks"] = list(callbacks) + [metrics_callback]
                else:  # 单个 handler 对象
                    config["callbacks"] = [callbacks, metrics_callback]

            # 我们需要通过限流和重试来调用
            @retry(
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential(
                    multiplier=1,
                    min=self._retry_initial_delay,
                    max=300,
                ),
                retry=retry_if_exception(_is_retryable),
                before_sleep=self._before_retry_sleep,
                reraise=True,
            )
            async def call_with_retry():
                async with await self._acquire_all_limits():
                    return await self._model.ainvoke(input, config=config, **kwargs)

            return await call_with_retry()

        # 否则走父类的 ainvoke（会调用我们的 agenerate）
        return await super().ainvoke(input, config=config, **kwargs)

    def invoke(
        self,
        input: Any,
        config=None,
        **kwargs: Any,
    ) -> Any:
        """同步调用：智能处理已运行事件循环（避免 FastAPI 嵌套循环冲突）"""
        try:
            # 检测是否已有事件循环在运行（FastAPI 环境）
            asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行的循环 → 正常使用 run_until_complete
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self.ainvoke(input, config=config, **kwargs))
            finally:
                loop.close()
        else:
            # 已有循环在运行 → 用新线程隔离运行异步代码
            # 注意：ContextVar（如 MetricsContext）跨线程会丢失，这是预期的隔离行为
            result: Any = None
            error: Optional[Exception] = None
            done = threading.Event()

            def run_async():
                nonlocal result, error
                new_loop: Optional[asyncio.AbstractEventLoop] = None
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(
                        self.ainvoke(input, config=config, **kwargs)
                    )
                except Exception as e:
                    error = e
                finally:
                    if new_loop is not None:
                        new_loop.close()  # 显式关闭循环，避免资源泄漏
                    done.set()

            thread = threading.Thread(target=run_async, daemon=True)
            thread.start()
            done.wait()
            thread.join()
            if error:
                raise error
            return result

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> "RateLimitedChatModel":
        """先让底层模型绑定 tools，然后重新包装成 RateLimitedChatModel，保留限流能力"""
        bound_model = self._model.bind_tools(tools, **kwargs)
        return RateLimitedChatModel(
            model=bound_model,
            model_key=self._model_key,
            provider_name=self._provider_name,
            concurrency=self._concurrency,
            rps=self._rps,
            rpm=self._rpm,
            max_retries=self._max_retries,
            retry_initial_delay=self._retry_initial_delay,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "RateLimitedChatModel":
        """先让底层模型配置结构化输出，然后重新包装成 RateLimitedChatModel，保留限流能力"""
        bound_model = self._model.with_structured_output(schema, **kwargs)
        return RateLimitedChatModel(
            model=bound_model,
            model_key=self._model_key,
            provider_name=self._provider_name,
            concurrency=self._concurrency,
            rps=self._rps,
            rpm=self._rpm,
            max_retries=self._max_retries,
            retry_initial_delay=self._retry_initial_delay,
        )

    async def _acquire_all_limits(self):
        """返回组合限制器的 async context manager"""
        limiters = []

        logger.debug(f"acquire_all_limits: model_key={self._model_key}, "
                    f"provider={self._provider_name}, "
                    f"concurrency={self._concurrency}, "
                    f"rps={self._rps}, "
                    f"rpm={self._rpm}")
        if self._concurrency is not None:
            limiters.append(ConcurrencyLimiter(
                resource_key=self._provider_name,
                max_concurrent=self._concurrency,
                max_wait=self._max_wait,
            ))

        if self._rps is not None or self._rpm is not None:
            limiters.append(RateLimiter(
                resource_key=self._provider_name,
                per_second=self._rps,
                per_minute=self._rpm,
                max_wait=self._max_wait,
            ))

        model_key = self._model_key
        provider_name = self._provider_name

        class CombinedLimiters:
            def __init__(self, limiters):
                self.limiters = limiters
                self._start_time = None

            async def __aenter__(self):
                self._start_time = time.time()
                for limiter in self.limiters:
                    await limiter.__aenter__()
                # 记录排队等待时间到 metrics
                queue_wait_ms = (time.time() - self._start_time) * 1000
                ctx = get_current_metrics_context()
                if ctx:
                    ctx.track_queue_wait(model_key, provider_name, queue_wait_ms)
                return self

            async def __aexit__(self, exc_type, exc, tb):
                for limiter in reversed(self.limiters):
                    await limiter.__aexit__(exc_type, exc, tb)

        return CombinedLimiters(limiters)

    def _before_retry_sleep(self, retry_state: RetryCallState) -> None:
        """重试前回调，记录重试次数"""
        run_id = get_current_llm_call_id()
        if run_id:
            ctx = get_current_metrics_context()
            if ctx:
                # attempt_number 从 1 开始，首次调用是 attempt 1，第一次重试是 attempt 2
                # 所以重试次数 = attempt_number - 1
                ctx.track_retry(run_id, retry_state.attempt_number - 1)

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks=None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """覆盖 agenerate，在 callbacks 层注入 MetricsCallbackHandler，
        并在此处做限流和重试。支持两种模式：
        1. _model 是 BaseChatModel -> 调用 _model.agenerate
        2. _model 是 Runnable (如 with_structured_output 后) -> 转为调用 batch
        """
        metrics_callback = MetricsCallbackHandler(
            model_key=self._model_key,
            provider=self._provider_name,
        )
        if callbacks is None:
            callbacks = [metrics_callback]
        elif hasattr(callbacks, "handlers"):  # 是 CallbackManager 对象，直接追加 handler
            callbacks.handlers.append(metrics_callback)
            if hasattr(callbacks, "inheritable_handlers"):
                callbacks.inheritable_handlers.append(metrics_callback)
        elif isinstance(callbacks, list):
            callbacks = list(callbacks) + [metrics_callback]
        else:
            # 单个 handler 对象
            callbacks = [callbacks, metrics_callback]

        @retry(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(
                multiplier=1,
                min=self._retry_initial_delay,
                max=300,
            ),
            retry=retry_if_exception(_is_retryable),
            before_sleep=self._before_retry_sleep,
            reraise=True,
        )
        async def call_with_retry():
            async with await self._acquire_all_limits():
                # 检查 _model 是否是 BaseChatModel 实例
                if isinstance(self._model, BaseChatModel):
                    return await self._model.agenerate(
                        messages,
                        stop=stop,
                        callbacks=callbacks,
                        tags=tags,
                        metadata=metadata,
                        run_name=run_name,
                        run_id=run_id,
                        **kwargs,
                    )
                else:
                    # _model 是 Runnable，使用 abatch 调用
                    # 将 messages 列表转换为 batch 输入
                    return await self._model.abatch(
                        messages,
                        config={
                            "callbacks": callbacks,
                            "tags": tags,
                            "metadata": metadata,
                            "run_name": run_name,
                        },
                        **kwargs,
                    )

        return await call_with_retry()

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """BaseChatModel 要求实现的抽象方法，直接转发给底层模型。
        限流/重试/metrics 已在 agenerate 层处理，此处不重复。
        """
        return await self._model._agenerate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    # def generate(
    #     self,
    #     messages: List[List[BaseMessage]],
    #     stop: Optional[List[str]] = None,
    #     callbacks=None,
    #     *,
    #     tags: Optional[List[str]] = None,
    #     metadata: Optional[Dict[str, Any]] = None,
    #     run_name: Optional[str] = None,
    #     **kwargs: Any,
    # ) -> LLMResult:
    #     """同步调用：限流、重试、metrics 全部支持（内部调用异步 agenerate）"""
    #     import asyncio

    #     try:
    #         loop = asyncio.get_event_loop()
    #     except RuntimeError:
    #         loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(loop)

    #     return loop.run_until_complete(
    #         self.agenerate(
    #             messages,
    #             stop=stop,
    #             callbacks=callbacks,
    #             tags=tags,
    #             metadata=metadata,
    #             run_name=run_name,
    #             **kwargs,
    #         )
    #     )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """BaseChatModel 要求实现的抽象方法，直接转发给底层模型"""
        return self._model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
