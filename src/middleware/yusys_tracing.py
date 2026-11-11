"""yusys_tracing 独立 TracerProvider（Agent 专用）。

设计目标：
- 只把 yusys_ai 中 `src.agents.*` 装饰器产生的 span 发送到 yusys_tracing。
- **绝不**调用 `trace.set_tracer_provider`，保留 Langfuse 全局 Provider 不变。
- **绝不**让 yusys_tracing 的 span 继承 Langfuse 当前 span 作为父节点：
  - 没有 yusys 父 span 时，显式使用 `INVALID_SPAN` 上下文创建全新 root trace。
  - 有 yusys 父 span 时，才作为子 span 挂到 yusys trace 下。
- 这样两个后端的数据在 trace_id / parent 关系上完全隔离，互不串流。
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_provider = None
_tracer = None
_enabled = False
_default_agent_name = "yusys-ai"

# yusys agent tracing 上下文（只作用于 yusys_tracing，不影响 Langfuse）
_agent_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "yusys_agent_name", default=None
)
_user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "yusys_user_id", default=None
)
_parent_span_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "yusys_parent_span", default=None
)


def setup_yusys_tracing(tracing_config: Any):
    """初始化 yusys_tracing 独立 Provider。

    调用时机：在 setup_otel_langfuse 之后。函数只创建 provider 并持有，
    不修改全局 tracer provider。
    """
    global _provider, _tracer, _enabled, _default_agent_name

    if not getattr(tracing_config, "enabled", False):
        logger.info("yusys_tracing agent tracer 未启用（enabled=false）")
        return None

    if getattr(tracing_config, "backend", "yusys") != "yusys":
        logger.info("yusys_tracing agent tracer 未启用（backend=%s）", getattr(tracing_config, "backend", ""))
        return None

    endpoint = (getattr(tracing_config, "otlp_endpoint", "") or "").strip()
    if not endpoint:
        logger.warning("yusys_tracing agent tracer 未启用（otlp_endpoint 为空）")
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning("OpenTelemetry SDK 不可用，yusys_tracing agent tracer 未启用: %s", exc)
        return None

    _default_agent_name = getattr(tracing_config, "agent_name", "yusys-ai") or "yusys-ai"

    resource = Resource.create(
        {
            "service.name": getattr(tracing_config, "service_name", "yusys-ai") or "yusys-ai",
            "project.id": getattr(tracing_config, "project_id", "yusys-ai") or "yusys-ai",
            "deployment.environment": getattr(tracing_config, "environment", "") or "development",
        }
    )

    normalized_endpoint = endpoint if endpoint.endswith("/v1/traces") else endpoint.rstrip("/") + "/v1/traces"
    exporter = OTLPSpanExporter(endpoint=normalized_endpoint)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(exporter, max_export_batch_size=64, schedule_delay_millis=5000)
    )

    _provider = provider
    # 关键：不使用 trace.set_tracer_provider(provider)，避免污染 Langfuse 全局 Provider
    _tracer = provider.get_tracer("src.agent.tracing")
    _enabled = True
    logger.info(
        "yusys_tracing agent tracer 初始化完成: endpoint=%s project_id=%s agent=%s",
        normalized_endpoint,
        getattr(tracing_config, "project_id", "yusys-ai"),
        _default_agent_name,
    )
    return provider


def is_enabled() -> bool:
    return _enabled and _tracer is not None


def get_agent_tracer():
    return _tracer if _enabled else None


def set_agent_context(agent_name: Optional[str], user_id: Optional[str] = None):
    """设置 yusys span 使用的 agent_name / user_id，返回 reset token。"""
    name_token = _agent_name_var.set(agent_name)
    user_token = _user_id_var.set(user_id)
    return name_token, user_token


def reset_agent_context(tokens) -> None:
    name_token, user_token = tokens
    _agent_name_var.reset(name_token)
    _user_id_var.reset(user_token)


def _current_agent_name() -> str:
    return _agent_name_var.get() or _default_agent_name


def _json_truncate(value: Any, max_chars: int = 2000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...(truncated,{len(text)})"


def _extract_usage_metadata(result: Any) -> tuple[int, int]:
    """尽力从 LangChain/OpenAI 返回对象中提取 token 用量。"""
    input_tokens = 0
    output_tokens = 0
    try:
        usage = getattr(result, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
    except Exception:
        pass
    return input_tokens, output_tokens


def _extract_model(result: Any) -> str:
    try:
        metadata = getattr(result, "response_metadata", None)
        if isinstance(metadata, dict):
            return str(metadata.get("model_name") or metadata.get("model") or "")
    except Exception:
        pass
    return ""


def _start_yusys_span(
    name: str,
    *,
    operation: str,
    parent_span: Any,
    model: Optional[str] = None,
):
    """创建 yusys span 的 contextmanager。"""
    from opentelemetry import trace as otel_trace

    tracer = get_agent_tracer()
    if tracer is None:
        return None

    attrs: dict[str, Any] = {
        "gen_ai.agent.name": _current_agent_name(),
        "gen_ai.operation.name": operation,
    }
    user_id = _user_id_var.get()
    if user_id:
        attrs["user.id"] = user_id
    if model:
        attrs["gen_ai.request.model"] = model

    invalid_span = getattr(otel_trace, "INVALID_SPAN", None)
    if parent_span is not None:
        parent_context = otel_trace.set_span_in_context(parent_span)
    elif invalid_span is not None:
        # 显式创建全新 trace，避免继承 Langfuse 当前 span
        parent_context = otel_trace.set_span_in_context(invalid_span)
    else:
        parent_context = None

    return tracer.start_as_current_span(
        name,
        context=parent_context,
        attributes=attrs,
    )


def _start_span_safely(name: str, *, operation: str, parent_span: Any, model: Optional[str] = None):
    """创建 yusys span，任何失败只降级 tracing，不影响业务。"""
    try:
        return _start_yusys_span(name, operation=operation, parent_span=parent_span, model=model)
    except Exception as exc:
        logger.warning("yusys_tracing span '%s' 创建失败，已跳过: %s", name, exc)
        return None


def _agent_span_decorator(
    func: Callable,
    *,
    name: Optional[str],
    operation: str,
    capture_input: bool,
    capture_output: bool,
    model: Optional[str] = None,
):
    """给函数包一层 yusys span。"""
    resolved_name = name or f"{func.__module__}.{func.__qualname__}"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_enabled():
            return func(*args, **kwargs)

        parent_span = _parent_span_var.get()
        span_cm = _start_span_safely(
            resolved_name,
            operation=operation,
            parent_span=parent_span,
            model=model,
        )
        if span_cm is None:
            return func(*args, **kwargs)

        with span_cm as span:
            parent_token = _parent_span_var.set(span)
            try:
                if capture_input:
                    span.set_attribute("gen_ai.input.messages", _json_truncate({"args": args, "kwargs": kwargs}))
                result = func(*args, **kwargs)
                if capture_output:
                    span.set_attribute("gen_ai.output.messages", _json_truncate(result))
                if operation == "chat":
                    in_tokens, out_tokens = _extract_usage_metadata(result)
                    if in_tokens or out_tokens:
                        span.set_attribute("gen_ai.usage.input_tokens", in_tokens)
                        span.set_attribute("gen_ai.usage.output_tokens", out_tokens)
                    if model is None:
                        extracted_model = _extract_model(result)
                        if extracted_model:
                            span.set_attribute("gen_ai.request.model", extracted_model)
                return result
            except Exception as exc:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute("error.message", str(exc))
                raise
            finally:
                _parent_span_var.reset(parent_token)

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        if not is_enabled():
            return await func(*args, **kwargs)

        parent_span = _parent_span_var.get()
        span_cm = _start_span_safely(
            resolved_name,
            operation=operation,
            parent_span=parent_span,
            model=model,
        )
        if span_cm is None:
            return await func(*args, **kwargs)

        with span_cm as span:
            parent_token = _parent_span_var.set(span)
            try:
                if capture_input:
                    span.set_attribute("gen_ai.input.messages", _json_truncate({"args": args, "kwargs": kwargs}))
                result = await func(*args, **kwargs)
                if capture_output:
                    span.set_attribute("gen_ai.output.messages", _json_truncate(result))
                if operation == "chat":
                    in_tokens, out_tokens = _extract_usage_metadata(result)
                    if in_tokens or out_tokens:
                        span.set_attribute("gen_ai.usage.input_tokens", in_tokens)
                        span.set_attribute("gen_ai.usage.output_tokens", out_tokens)
                    if model is None:
                        extracted_model = _extract_model(result)
                        if extracted_model:
                            span.set_attribute("gen_ai.request.model", extracted_model)
                return result
            except Exception as exc:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute("error.message", str(exc))
                raise
            finally:
                _parent_span_var.reset(parent_token)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper


def agent_span(name: Optional[str] = None, *, operation: str = "workflow", capture_input: bool = True, capture_output: bool = True):
    """创建 yusys span 的装饰器。"""
    def decorator(func: Callable):
        return _agent_span_decorator(
            func,
            name=name,
            operation=operation,
            capture_input=capture_input,
            capture_output=capture_output,
        )
    return decorator


def agent_llm_call(name: Optional[str] = None, *, model: Optional[str] = None, capture_input: bool = True, capture_output: bool = True):
    """创建 yusys LLM span 的装饰器。"""
    def decorator(func: Callable):
        return _agent_span_decorator(
            func,
            name=name,
            operation="chat",
            capture_input=capture_input,
            capture_output=capture_output,
            model=model,
        )
    return decorator


def agent_workflow(name: Optional[str] = None, *, agent_name: Optional[str] = None, user_id: Optional[str] = None, capture_input: bool = True, capture_output: bool = True):
    """创建 yusys Agent 工作流根 span 的装饰器。"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_enabled():
                return func(*args, **kwargs)
            tokens = set_agent_context(agent_name, user_id)
            try:
                return agent_span(
                    name=name,
                    operation="invoke_agent",
                    capture_input=capture_input,
                    capture_output=capture_output,
                )(func)(*args, **kwargs)
            finally:
                reset_agent_context(tokens)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not is_enabled():
                return await func(*args, **kwargs)
            tokens = set_agent_context(agent_name, user_id)
            try:
                decorated = agent_span(
                    name=name,
                    operation="invoke_agent",
                    capture_input=capture_input,
                    capture_output=capture_output,
                )(func)
                return await decorated(*args, **kwargs)
            finally:
                reset_agent_context(tokens)

        return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
    return decorator
