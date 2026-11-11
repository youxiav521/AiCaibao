"""
统一的 Langfuse 追踪装饰器模块，在配置不完整时自动禁用所有追踪功能
集成了 observe 和 trace_span 功能，对外提供统一的 API
"""
import functools
import inspect
from typing import Any, Callable, Optional
from src.base.langfuse_runtime import get_langfuse_runtime
from src.utils.langfuse_tracing import trace_agent_workflow as _original_trace_agent_workflow
from langfuse import observe as _langfuse_observe

# 检查 Langfuse 配置是否完整
_runtime = get_langfuse_runtime()
_langfuse_available = _runtime.enabled

USE_NATIVE_OBSERVE = False


def observe(
    name: Optional[str] = None,
    as_type: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    **kwargs: Any,
) -> Callable:
    """
    统一的 observe 装饰器，当 Langfuse 配置不完整时直接返回原函数，不进行追踪
    """
    # 如果 Langfuse 不可用，返回一个空装饰器
    if not _langfuse_available:
        def decorator(func: Callable) -> Callable:
            return func
        return decorator
    
    # 否则使用真正的 Langfuse observe 装饰器
    return _langfuse_observe(
        name=name,
        as_type=as_type,
        capture_input=capture_input,
        capture_output=capture_output,
        **kwargs
    )


def trace_span(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    **kwargs: Any,
) -> Callable:
    """
    trace_span 装饰器，兼容原有 API，底层统一使用 observe
    """
    if USE_NATIVE_OBSERVE:
        # 使用原生 observe 实现
        return observe(
            name=name,
            as_type="span",
            capture_input=capture_input,
            capture_output=capture_output,
            **kwargs
        )
    else:
        # 保持原有自定义 trace_span 的实现
        def decorator(func: Callable) -> Callable:
            from src.utils.langfuse_tracing import trace_span as _original_trace_span
            return _original_trace_span(
                name=name,
                capture_input=capture_input,
                capture_output=capture_output,
            )(func)
        return decorator


# 为了兼容原有代码，也提供 trace_llm_call 和 trace_agent_workflow
def trace_llm_call(
    name: Optional[str] = None,
    model: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    **kwargs: Any,
) -> Callable:
    """
    trace_llm_call 装饰器，兼容原有 API
    """
    if USE_NATIVE_OBSERVE:
        # 使用原生 observe 实现
        return observe(
            name=name,
            as_type="generation",
            model=model,
            capture_input=capture_input,
            capture_output=capture_output,
            **kwargs
        )
    else:
        # 保持原有自定义实现
        def decorator(func: Callable) -> Callable:
            from src.utils.langfuse_tracing import trace_llm_call as _original_trace_llm_call
            return _original_trace_llm_call(
                name=name,
                model=model,
                capture_input=capture_input,
                capture_output=capture_output,
            )(func)
        return decorator


def trace_agent_workflow(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    **kwargs: Any,
) -> Callable:
    """
    trace_agent_workflow 装饰器，兼容原有 API
    """
    def decorator(func: Callable) -> Callable:
        return _original_trace_agent_workflow(
            name=name,
            capture_input=capture_input,
            capture_output=capture_output,
        )(func)
    return decorator


def trace_event(
    event_name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    level: str = "DEFAULT",
    **kwargs: Any,
) -> Callable:
    """
    trace_event 装饰器，兼容原有 API
    """
    def decorator(func: Callable) -> Callable:
        from src.utils.langfuse_tracing import trace_event as _original_trace_event
        return _original_trace_event(
            event_name=event_name,
            capture_input=capture_input,
            capture_output=capture_output,
            level=level,
        )(func)
    return decorator

