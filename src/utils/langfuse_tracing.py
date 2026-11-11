"""
Langfuse Tracing 工具
提供LangGraph Agent的tracing功能，使用Langfuse SDK v3 (OpenTelemetry-based)

 记录函数执行为一次llm调用，自动链接到父跨度
@trace_llm_call(
    name: Optional[str] = None,#记录llm调用名称
    model: Optional[str] = None,#记录llm调用模型名
    capture_input: bool = True,#是否记录输入
    capture_output: bool = True,#是否记录输出
)
 记录函数执行为一个 "子"跨度，自动链接到父跨度
@trace_span(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True
)
 记录函数执行为一个 "根"跨度（一次调用的起点）
@trace_agent_workflow(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True
) 
 记录函数执行为一个 Event。
@trace_event(
    event_name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    level: str = "DEFAULT" #事件级别 (DEBUG, DEFAULT, WARNING, ERROR)
)

"""

import os
import functools
import time
import inspect
import threading
import contextvars
import uuid
import re
import hashlib
import dataclasses
from typing import Any, Dict, Optional, Callable, List
from contextlib import contextmanager
from loguru import logger
from src.base.langfuse_runtime import get_langfuse_client
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Langfuse未安装，tracing功能将被禁用: {e}")
    Langfuse = type("Langfuse", (), {"__init__": lambda self, *args, **kwargs: None})
    LANGFUSE_AVAILABLE = False


# --- 请求级 Trace 上下文（用于透传 Header trace_id）---
_external_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "external_trace_id", default=None
)
_TRACE_ID_32HEX_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def get_external_trace_id() -> Optional[str]:
    """获取当前上下文中的外部 trace_id（来自请求头）"""
    return _external_trace_id_var.get()


def _is_valid_otel_trace_id(trace_id: Optional[str]) -> bool:
    if not trace_id or not isinstance(trace_id, str):
        return False
    tid = trace_id.strip()
    if not _TRACE_ID_32HEX_RE.fullmatch(tid):
        return False
    return tid.lower() != "0" * 32


def derive_langfuse_trace_id(external_trace_id: Optional[str], client: Optional[Any] = None) -> str:
    """
    将任意外部 trace_id 映射为合法的 OTel trace_id（32位hex）。
    """
    if _is_valid_otel_trace_id(external_trace_id):
        return external_trace_id.strip().lower()

    seed = external_trace_id or uuid.uuid4().hex
    if client is not None and hasattr(client, "create_trace_id"):
        try:
            trace_id = client.create_trace_id(seed=seed)
            if _is_valid_otel_trace_id(trace_id):
                return trace_id.lower()
        except Exception as e:
            logger.warning(f"Langfuse create_trace_id(seed=...) 失败，将退化为 md5 映射: {e}")

    return hashlib.md5(seed.encode("utf-8")).hexdigest().lower()


def set_trace_context(
    external_trace_id: Optional[str],
    langfuse_trace_id: Optional[str] = None,
):
    """设置当前上下文中的 trace_id。返回 token，便于 finally 中 reset。"""
    token = _external_trace_id_var.set(external_trace_id)
    return token


def reset_trace_context(token_external, token_langfuse=None):
    """恢复 trace_id 上下文"""
    _external_trace_id_var.reset(token_external)


def inject_otel_trace_carrier() -> Dict[str, str]:
    """将当前 OTel 上下文注入为可跨进程传递的 W3C trace carrier。"""
    carrier: Dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def attach_otel_trace_context(trace_carrier: Optional[Dict[str, str]]):
    """从 W3C trace carrier 恢复当前 OTel 上下文，返回 detach 用 token。"""
    if not trace_carrier:
        return None
    parent_context = TraceContextTextMapPropagator().extract(carrier=trace_carrier)
    return otel_context.attach(parent_context)


def reset_otel_trace_context(token) -> None:
    """恢复 attach_otel_trace_context 设置的 OTel 上下文。"""
    if token is not None:
        otel_context.detach(token)


def _truncate_text(value: str, max_str_len: int) -> str:
    if len(value) <= max_str_len:
        return value
    return value[:max_str_len] + f"...(truncated,{len(value)})"


def _safe_repr(obj: Any, max_str_len: int) -> str:
    try:
        return _truncate_text(repr(obj), max_str_len)
    except Exception as e:
        return f"<unrepresentable {type(obj).__module__}.{type(obj).__qualname__}: {e}>"


def _qualified_name(obj_or_type: Any) -> str:
    target = obj_or_type if inspect.isclass(obj_or_type) else type(obj_or_type)
    return f"{target.__module__}.{target.__qualname__}"


def _safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except Exception:
        return None


def _serialize_mapping(obj: Dict[Any, Any], *, max_depth: int, max_items: int, max_str_len: int) -> Dict[str, Any]:
    size = _safe_len(obj)
    if max_depth <= 0:
        return {"__type__": type(obj).__name__, "size": size}

    out: Dict[str, Any] = {}
    for i, (k, v) in enumerate(obj.items()):
        if i >= max_items:
            out["__truncated__"] = f"items={size or i}"
            break
        out[_truncate_text(str(k), max_str_len)] = _to_jsonable(
            v,
            max_depth=max_depth - 1,
            max_items=max_items,
            max_str_len=max_str_len,
        )
    if size is not None:
        out["__size__"] = size
    return out


def _serialize_sequence(obj: Any, *, max_depth: int, max_items: int, max_str_len: int) -> Dict[str, Any]:
    size = _safe_len(obj)
    if max_depth <= 0:
        return {"__type__": type(obj).__name__, "size": size}

    items: List[Any] = []
    for i, item in enumerate(obj):
        if i >= max_items:
            items.append(f"...truncated({size or i})")
            break
        items.append(
            _to_jsonable(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_str_len=max_str_len,
            )
        )
    return {"__type__": type(obj).__name__, "items": items, "size": size}


def _serialize_message_like(obj: Any, *, max_depth: int, max_items: int, max_str_len: int) -> Dict[str, Any]:
    preview: Dict[str, Any] = {
        "__type__": _qualified_name(obj),
        "message_type": getattr(obj, "type", None),
    }
    if hasattr(obj, "name"):
        preview["name"] = getattr(obj, "name", None)
    if hasattr(obj, "content"):
        preview["content"] = _to_jsonable(
            getattr(obj, "content", None),
            max_depth=max_depth - 1,
            max_items=max_items,
            max_str_len=max_str_len,
        )
    tool_calls = getattr(obj, "tool_calls", None)
    if tool_calls is not None:
        preview["tool_calls_count"] = _safe_len(tool_calls)
    return preview


def _serialize_pydantic_model(obj: Any, *, max_depth: int, max_items: int, max_str_len: int) -> Dict[str, Any]:
    preview: Dict[str, Any] = {"__pydantic_model__": _qualified_name(obj)}
    if max_depth <= 0:
        return preview

    field_names: List[str] = []
    model_fields = getattr(type(obj), "model_fields", None)
    if isinstance(model_fields, dict):
        field_names = list(model_fields.keys())
    else:
        legacy_fields = getattr(obj, "__fields__", None)
        if isinstance(legacy_fields, dict):
            field_names = list(legacy_fields.keys())

    for i, field_name in enumerate(field_names):
        if i >= max_items:
            preview["__truncated__"] = f"fields={len(field_names)}"
            break
        try:
            value = getattr(obj, field_name)
        except Exception as e:
            preview[field_name] = f"<unavailable:{e}>"
            continue
        preview[field_name] = _to_jsonable(
            value,
            max_depth=max_depth - 1,
            max_items=max_items,
            max_str_len=max_str_len,
        )

    if len(field_names) > max_items:
        preview["__field_count__"] = len(field_names)
    return preview


def _serialize_dataclass(obj: Any, *, max_depth: int, max_items: int, max_str_len: int) -> Dict[str, Any]:
    preview: Dict[str, Any] = {"__dataclass__": _qualified_name(obj)}
    if max_depth <= 0:
        return preview

    fields = dataclasses.fields(obj)
    for i, field in enumerate(fields):
        if i >= max_items:
            preview["__truncated__"] = f"fields={len(fields)}"
            break
        preview[field.name] = _to_jsonable(
            getattr(obj, field.name),
            max_depth=max_depth - 1,
            max_items=max_items,
            max_str_len=max_str_len,
        )

    if len(fields) > max_items:
        preview["__field_count__"] = len(fields)
    return preview


def _to_jsonable(obj: Any, *, max_depth: int = 3, max_items: int = 20, max_str_len: int = 2000) -> Any:
    """将对象转换为可 JSON 序列化的轻量预览结构。"""
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        return _truncate_text(obj, max_str_len)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        raw = bytes(obj)
        return _truncate_text(raw.decode("utf-8", errors="replace"), max_str_len)
    if isinstance(obj, BaseException):
        return {
            "__exception__": _qualified_name(obj),
            "message": _truncate_text(str(obj), max_str_len),
        }
    if inspect.isclass(obj):
        return {"__class__": _qualified_name(obj)}
    if inspect.ismodule(obj):
        return {"__module__": getattr(obj, "__name__", _safe_repr(obj, max_str_len))}
    if inspect.isroutine(obj):
        return {"__callable__": _qualified_name(obj)}
    if hasattr(obj, "type") and hasattr(obj, "content"):
        return _serialize_message_like(obj, max_depth=max_depth, max_items=max_items, max_str_len=max_str_len)
    if isinstance(obj, dict):
        return _serialize_mapping(obj, max_depth=max_depth, max_items=max_items, max_str_len=max_str_len)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return _serialize_sequence(obj, max_depth=max_depth, max_items=max_items, max_str_len=max_str_len)

    try:
        from pydantic import BaseModel
        if isinstance(obj, BaseModel):
            return _serialize_pydantic_model(obj, max_depth=max_depth, max_items=max_items, max_str_len=max_str_len)
    except ImportError:
        pass

    if dataclasses.is_dataclass(obj) and not inspect.isclass(obj):
        return _serialize_dataclass(obj, max_depth=max_depth, max_items=max_items, max_str_len=max_str_len)

    return {
        "__type__": _qualified_name(obj),
        "repr": _safe_repr(obj, max_str_len),
    }


def _safe_jsonable(obj: Any) -> Any:
    """确保 tracing 的输入输出序列化失败时不会影响主流程。"""
    try:
        return _to_jsonable(obj)
    except Exception as e:
        logger.warning("Langfuse tracing 序列化失败，已降级为简要描述: {}", e)
        return {
            "__serialization_error__": str(e),
            "__type__": _qualified_name(obj),
        }


class LangfuseTracer:
    """Langfuse Tracing管理器 - 使用SDK v3 API"""
    
    def __init__(self, session_id: Optional[str] = None):
        try:
            self.client = get_langfuse_client()
            self.enabled = LANGFUSE_AVAILABLE and self.client is not None
            self.session_id = session_id or f"session-{int(time.time() * 1000)}"

            if self.enabled and hasattr(self.client, "auth_check"):
                try:
                    if not self.client.auth_check():
                        self.enabled = False
                        logger.warning("Langfuse auth_check 失败，tracing功能将被禁用")
                except Exception as e:
                    self.enabled = False
                    logger.error(f"Langfuse auth_check 异常，tracing功能将被禁用: {e}")

            if self.enabled:
                logger.info(f"Langfuse Tracer 初始化成功, Session ID: {self.session_id}")
            else:
                logger.warning("Langfuse Tracing 已禁用 (配置不完整、鉴权失败或库未安装)")

        except Exception as e:
            logger.error(f"创建Langfuse客户端失败，tracing功能将被禁用: {e}")
            self.client = None
            self.enabled = False
    
    def is_enabled(self) -> bool:
        return self.enabled

    @contextmanager
    def trace(self,
              name: str,
              user_id: Optional[str] = None,
              session_id: Optional[str] = None,
              trace_context: Optional[Dict[str, Any]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              input: Optional[Any] = None,
              tags: Optional[List[str]] = None,
              **kwargs):
        """创建一个 trace (根 span) 的上下文管理器"""
        if not self.is_enabled():
            yield None
            return

        trace_metadata = dict(metadata or {})
        effective_session_id = session_id or self.session_id
        trace_metadata.setdefault("session_id", effective_session_id)
        if user_id:
            trace_metadata.setdefault("user_id", user_id)
        external_trace_id = get_external_trace_id()
        if external_trace_id:
            trace_metadata.setdefault("external_trace_id", external_trace_id)

        span_kwargs: Dict[str, Any] = {
            "name": name,
            "input": _safe_jsonable(input if input is not None else trace_metadata),
            "metadata": _safe_jsonable(trace_metadata),
        }
        if trace_context is not None:
            span_kwargs["trace_context"] = trace_context

        trace_cm = None
        try:
            trace_cm = self.client.start_as_current_span(**span_kwargs)
            span = trace_cm.__enter__()
        except Exception as e:
            logger.error(f"Langfuse trace '{name}' 创建失败，已跳过 tracing: {e}")
            yield None
            return

        try:
            try:
                self.client.update_current_trace(
                    name=name,
                    user_id=user_id,
                    session_id=effective_session_id,
                    metadata=_safe_jsonable(trace_metadata),
                    tags=tags,
                )
            except Exception:
                pass
            logger.info(f"开始 Langfuse trace: {name}")
            yield span
        except BaseException as e:
            try:
                trace_cm.__exit__(type(e), e, e.__traceback__)
            except Exception as exit_error:
                logger.warning(
                    "Langfuse trace '{}' 退出失败，已忽略并保留原始异常: {}",
                    name,
                    exit_error,
                )
            raise
        else:
            try:
                trace_cm.__exit__(None, None, None)
            except Exception as exit_error:
                logger.warning(
                    "Langfuse trace '{}' 退出失败，已忽略: {}",
                    name,
                    exit_error,
                )
    
    def shutdown(self):
        """安全关闭 Langfuse 客户端"""
        if self.is_enabled():
            self.client.flush()
            logger.info("Langfuse client flushed.")


def _get_trace_context() -> Optional[Dict[str, Any]]:
    """获取当前的 trace context"""
    if otel_trace.get_current_span().is_recording():
        return None

    external_id = get_external_trace_id()
    if external_id:
        return {"trace_id": derive_langfuse_trace_id(external_id)}
    return None


@contextmanager
def _safe_observation_context(factory: Callable[[], Any], *, resolved_name: str):
    """安全创建 Langfuse observation，失败时只降级 tracing，不影响业务。"""
    observation_cm = None
    try:
        observation_cm = factory()
        observation = observation_cm.__enter__()
    except Exception as e:
        logger.warning("Langfuse observation '{}' 创建失败，已跳过 tracing: {}", resolved_name, e)
        yield None
        return

    try:
        yield observation
    except BaseException as e:
        try:
            observation_cm.__exit__(type(e), e, e.__traceback__)
        except Exception as exit_error:
            logger.warning(
                "Langfuse observation '{}' 退出失败，已忽略并保留原始异常: {}",
                resolved_name,
                exit_error,
            )
        raise
    else:
        try:
            observation_cm.__exit__(None, None, None)
        except Exception as exit_error:
            logger.warning(
                "Langfuse observation '{}' 退出失败，已忽略: {}",
                resolved_name,
                exit_error,
            )


def _safe_observation_update(observation: Any, *, resolved_name: str, **kwargs):
    if not observation:
        return
    try:
        observation.update(**kwargs)
    except Exception as e:
        logger.warning("Langfuse observation '{}' 更新失败，已忽略: {}", resolved_name, e)


def _safe_create_event(client: Any, *, resolved_name: str, **kwargs):
    try:
        client.create_event(**kwargs)
    except Exception as e:
        logger.warning("Langfuse event '{}' 创建失败，已忽略: {}", resolved_name, e)


def trace_llm_call(
    name: Optional[str] = None,
    model: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """装饰器：用于追踪一个LLM调用函数或方法"""

    def _start_generation_cm(tracer_client: Any, *, resolved_name: str, input_data: Any, metadata: Any):
        """兼容不同版本 Langfuse SDK"""
        kw = {}
        trace_ctx = _get_trace_context()
        if trace_ctx:
            kw["trace_context"] = trace_ctx

        if hasattr(tracer_client, "start_as_current_observation"):
            return tracer_client.start_as_current_observation(
                as_type="generation", name=resolved_name, model=model,
                input=input_data, metadata=metadata, **kw
            )
        if hasattr(tracer_client, "start_as_current_generation"):
            return tracer_client.start_as_current_generation(
                name=resolved_name, model=model, input=input_data, metadata=metadata, **kw
            )
        return tracer_client.start_as_current_span(
            name=resolved_name, input=input_data, metadata=metadata, **kw
        )

    def decorator(func: Callable) -> Callable:
        resolved_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            if not tracer or not tracer.is_enabled():
                return func(*args, **kwargs)

            external_trace_id = get_external_trace_id()
            input_data = _safe_jsonable({"args": args, "kwargs": kwargs}) if capture_input else None
            base_metadata: Dict[str, Any] = {"function": resolved_name}
            if external_trace_id:
                base_metadata["external_trace_id"] = external_trace_id

            with _safe_observation_context(
                lambda: _start_generation_cm(
                    tracer.client,
                    resolved_name=resolved_name,
                    input_data=input_data,
                    metadata=base_metadata,
                ),
                resolved_name=resolved_name,
            ) as generation:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    _safe_observation_update(
                        generation,
                        resolved_name=resolved_name,
                        output=_safe_jsonable({"error": str(e)}) if capture_output else None,
                        metadata={**base_metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                                  "exception_type": type(e).__name__},
                        level="ERROR",
                        status_message=str(e),
                    )
                    raise

                _safe_observation_update(
                    generation,
                    resolved_name=resolved_name,
                    output=_safe_jsonable(result) if capture_output else None,
                    metadata={**base_metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000},
                )
                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            if not tracer or not tracer.is_enabled():
                return await func(*args, **kwargs)
            
            external_trace_id = get_external_trace_id()
            input_data = _safe_jsonable({"args": args, "kwargs": kwargs}) if capture_input else None
            base_metadata: Dict[str, Any] = {"function": resolved_name, "async": True}
            if external_trace_id:
                base_metadata["external_trace_id"] = external_trace_id

            with _safe_observation_context(
                lambda: _start_generation_cm(
                    tracer.client,
                    resolved_name=resolved_name,
                    input_data=input_data,
                    metadata=base_metadata,
                ),
                resolved_name=resolved_name,
            ) as generation:
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    _safe_observation_update(
                        generation,
                        resolved_name=resolved_name,
                        output=_safe_jsonable({"error": str(e)}) if capture_output else None,
                        metadata={**base_metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                                  "exception_type": type(e).__name__},
                        level="ERROR",
                        status_message=str(e),
                    )
                    raise

                _safe_observation_update(
                    generation,
                    resolved_name=resolved_name,
                    output=_safe_jsonable(result) if capture_output else None,
                    metadata={**base_metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000},
                )
                return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def trace_span(name: Optional[str] = None, capture_input: bool = True, capture_output: bool = True):
    """装饰器：将一个函数或方法追踪为一个 Span"""
    def decorator(func: Callable) -> Callable:
        resolved_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            if not tracer or not tracer.is_enabled():
                return func(*args, **kwargs)

            external_trace_id = get_external_trace_id()
            input_data = _safe_jsonable({"args": args, "kwargs": kwargs}) if capture_input else None
            base_metadata: Dict[str, Any] = {"function": resolved_name}
            if external_trace_id:
                base_metadata["external_trace_id"] = external_trace_id

            span_kwargs = {"name": resolved_name, "input": input_data, "metadata": base_metadata}
            trace_ctx = _get_trace_context()
            if trace_ctx:
                span_kwargs["trace_context"] = trace_ctx

            with _safe_observation_context(
                lambda: tracer.client.start_as_current_span(**span_kwargs),
                resolved_name=resolved_name,
            ) as span:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    _safe_observation_update(
                        span,
                        resolved_name=resolved_name,
                        output=_safe_jsonable({"error": str(e)}) if capture_output else None,
                        metadata={**base_metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                                  "exception_type": type(e).__name__},
                        level="ERROR",
                        status_message=str(e),
                    )
                    raise

                _safe_observation_update(
                    span,
                    resolved_name=resolved_name,
                    output=_safe_jsonable(result) if capture_output else None,
                    metadata={**base_metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000},
                )
                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            if not tracer or not tracer.is_enabled():
                return await func(*args, **kwargs)

            external_trace_id = get_external_trace_id()
            input_data = _safe_jsonable({"args": args, "kwargs": kwargs}) if capture_input else None
            base_metadata: Dict[str, Any] = {"function": resolved_name, "async": True}
            if external_trace_id:
                base_metadata["external_trace_id"] = external_trace_id

            span_kwargs = {"name": resolved_name, "input": input_data, "metadata": base_metadata}
            trace_ctx = _get_trace_context()
            if trace_ctx:
                span_kwargs["trace_context"] = trace_ctx

            with _safe_observation_context(
                lambda: tracer.client.start_as_current_span(**span_kwargs),
                resolved_name=resolved_name,
            ) as span:
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    _safe_observation_update(
                        span,
                        resolved_name=resolved_name,
                        output=_safe_jsonable({"error": str(e)}) if capture_output else None,
                        metadata={**base_metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                                  "exception_type": type(e).__name__},
                        level="ERROR",
                        status_message=str(e),
                    )
                    raise

                _safe_observation_update(
                    span,
                    resolved_name=resolved_name,
                    output=_safe_jsonable(result) if capture_output else None,
                    metadata={**base_metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000},
                )
                return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def trace_event(
    event_name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    level: str = "DEFAULT"
):
    """装饰器：记录函数执行为一个 Event"""
    def decorator(func: Callable) -> Callable:
        resolved_name = event_name or f"{func.__module__}.{func.__qualname__}"

        def _prepare_input(args, kwargs):
            if len(args) > 1:
                return {"input": args[1], "kwargs": kwargs}
            return {"args": args, "kwargs": kwargs}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            if not tracer or not tracer.is_enabled():
                return func(*args, **kwargs)

            input_data = _prepare_input(args, kwargs) if capture_input else None
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                _safe_create_event(
                    tracer.client,
                    resolved_name=resolved_name,
                    name=f"{resolved_name}_error",
                    input=_safe_jsonable(input_data),
                    output=_safe_jsonable({"error": str(e)}) if capture_output else None,
                    metadata={"execution_time_ms": (time.perf_counter() - start_time) * 1000,
                              "error_type": type(e).__name__, "function_name": func.__name__},
                    level="ERROR",
                )
                raise

            _safe_create_event(
                tracer.client,
                resolved_name=resolved_name,
                name=resolved_name,
                input=_safe_jsonable(input_data) if capture_input else None,
                output=_safe_jsonable(result) if capture_output else None,
                metadata={"execution_time_ms": (time.perf_counter() - start_time) * 1000,
                          "function_name": func.__name__},
                level=level,
            )
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            if not tracer or not tracer.is_enabled():
                return await func(*args, **kwargs)

            input_data = _prepare_input(args, kwargs) if capture_input else None
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                _safe_create_event(
                    tracer.client,
                    resolved_name=resolved_name,
                    name=f"{resolved_name}_error",
                    input=_safe_jsonable(input_data),
                    output=_safe_jsonable({"error": str(e)}) if capture_output else None,
                    metadata={"execution_time_ms": (time.perf_counter() - start_time) * 1000,
                              "error_type": type(e).__name__, "async": True, "function_name": func.__name__},
                    level="ERROR",
                )
                raise

            _safe_create_event(
                tracer.client,
                resolved_name=resolved_name,
                name=resolved_name,
                input=_safe_jsonable(input_data) if capture_input else None,
                output=_safe_jsonable(result) if capture_output else None,
                metadata={"execution_time_ms": (time.perf_counter() - start_time) * 1000,
                          "function_name": func.__name__, "async": True},
                level=level,
            )
            return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def trace_agent_workflow(name: Optional[str] = None, capture_input: bool = True, capture_output: bool = True):
    """专门用于追踪 Agent 工作流的装饰器"""
    def decorator(func: Callable) -> Callable:
        resolved_name = name or f"agent_workflow_{func.__name__}"

        def _prepare_input(args, kwargs):
            if len(args) > 1:
                return {"input": args[1], "kwargs": kwargs}
            return {"args": args, "kwargs": kwargs}

        def _prepare_output(result):
            if hasattr(result, "success"):
                return {
                    "success": getattr(result, "success", None),
                    "total_nodes": getattr(result, "total_nodes", None),
                    "processing_time": getattr(result, "processing_time", None),
                    "message": getattr(result, "message", None),
                }
            return result

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            if not tracer or not tracer.is_enabled():
                return func(*args, **kwargs)

            metadata = {"agent_type": "LangGraphAnalysisAgent", "workflow_type": "policy_analysis", "timestamp": time.time()}
            trace_ctx = _get_trace_context()

            with tracer.trace(name=resolved_name, metadata=metadata, trace_context=trace_ctx) as trace_span:
                logger.info(f"开始 Agent 工作流追踪: {resolved_name}")
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    _safe_observation_update(
                        trace_span,
                        resolved_name=resolved_name,
                        output=_safe_jsonable({"error": str(e)}) if capture_output else None,
                        metadata={**metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                                  "exception_type": type(e).__name__},
                        level="ERROR",
                        status_message=str(e),
                    )
                    raise

                if capture_output:
                    _safe_observation_update(
                        trace_span,
                        resolved_name=resolved_name,
                        output=_safe_jsonable(_prepare_output(result)),
                        metadata={**metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000},
                    )

                logger.info(f"完成 Agent 工作流追踪: {resolved_name}")
                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            if not tracer or not tracer.is_enabled():
                return await func(*args, **kwargs)

            metadata = {"agent_type": "LangGraphAnalysisAgent", "workflow_type": "policy_analysis", "timestamp": time.time()}
            trace_ctx = _get_trace_context()

            with tracer.trace(name=resolved_name, metadata=metadata, trace_context=trace_ctx) as trace_span:
                logger.info(f"开始异步 Agent 工作流追踪: {resolved_name}")
                start_time = time.perf_counter()
                
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    _safe_observation_update(
                        trace_span,
                        resolved_name=resolved_name,
                        output=_safe_jsonable({"error": str(e)}) if capture_output else None,
                        metadata={**metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                                  "exception_type": type(e).__name__, "async": True},
                        level="ERROR",
                        status_message=str(e),
                    )
                    raise

                if capture_output:
                    _safe_observation_update(
                        trace_span,
                        resolved_name=resolved_name,
                        output=_safe_jsonable(_prepare_output(result)),
                        metadata={**metadata, "execution_time_ms": (time.perf_counter() - start_time) * 1000, "async": True},
                    )

                logger.info(f"完成异步 Agent 工作流追踪: {resolved_name}")
                return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# yusys_tracing 装饰器组合
# 只把 `src.agents.*` 模块中通过 trace_span / trace_llm_call 装饰的函数，
# 额外以独立 Provider 的 span 发送到 yusys_tracing。
# Langfuse 原有逻辑完全保留；两者 Provider 隔离，互不串流。
# ---------------------------------------------------------------------------

_langfuse_trace_llm_call = trace_llm_call
_langfuse_trace_span = trace_span
_langfuse_trace_agent_workflow = trace_agent_workflow


def _should_send_to_yusys(func: Callable) -> bool:
    module = getattr(func, "__module__", "") or ""
    return module.startswith("src.agents") or module.startswith("agents.")


def trace_llm_call(
    name: Optional[str] = None,
    model: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """LLM 调用装饰器：Langfuse 照旧；`src.agents.*` 内的函数同时发 yusys_tracing。"""
    def decorator(func: Callable):
        base_func = func
        if _should_send_to_yusys(func):
            from src.middleware.yusys_tracing import agent_llm_call

            base_func = agent_llm_call(
                name=name,
                model=model,
                capture_input=capture_input,
                capture_output=capture_output,
            )(func)
        return _langfuse_trace_llm_call(
            name=name,
            model=model,
            capture_input=capture_input,
            capture_output=capture_output,
        )(base_func)
    return decorator


def trace_span(name: Optional[str] = None, capture_input: bool = True, capture_output: bool = True):
    """Span 装饰器：Langfuse 照旧；`src.agents.*` 内的函数同时发 yusys_tracing。"""
    def decorator(func: Callable):
        base_func = func
        if _should_send_to_yusys(func):
            from src.middleware.yusys_tracing import agent_span

            base_func = agent_span(
                name=name,
                operation="workflow",
                capture_input=capture_input,
                capture_output=capture_output,
            )(func)
        return _langfuse_trace_span(
            name=name,
            capture_input=capture_input,
            capture_output=capture_output,
        )(base_func)
    return decorator


def trace_agent_workflow(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """Agent 工作流装饰器：Langfuse 照旧；同时创建 yusys_tracing 的 invoke_agent 根 span。"""
    def decorator(func: Callable):
        base_func = func
        if _should_send_to_yusys(func):
            from src.middleware.yusys_tracing import agent_workflow

            base_func = agent_workflow(
                name=name,
                capture_input=capture_input,
                capture_output=capture_output,
            )(func)
        return _langfuse_trace_agent_workflow(
            name=name,
            capture_input=capture_input,
            capture_output=capture_output,
        )(base_func)
    return decorator


# --- 全局 Tracer 实例管理 (线程安全) ---
_global_tracer: Optional[LangfuseTracer] = None
_global_tracer_lock = threading.Lock()


def get_global_tracer() -> Optional[LangfuseTracer]:
    """获取全局tracer实例 (线程安全)"""
    global _global_tracer
    if _global_tracer is None:
        with _global_tracer_lock:
            if _global_tracer is None:
                _global_tracer = LangfuseTracer()
    return _global_tracer


def set_global_tracer(tracer: LangfuseTracer):
    """设置全局tracer实例"""
    global _global_tracer
    with _global_tracer_lock:
        _global_tracer = tracer


def log_milestone_event(event_name: str, data: Optional[Dict[str, Any]] = None, level: str = "DEFAULT"):
    """记录里程碑事件的便捷函数"""
    tracer = get_global_tracer()
    if tracer and tracer.is_enabled():
        try:
            _safe_create_event(
                tracer.client,
                resolved_name=event_name,
                name=event_name,
                input=_safe_jsonable(data),
                metadata={"timestamp": time.time(), "event_type": "milestone"},
                level=level
            )
            logger.debug(f"Milestone event logged: {event_name}")
        except Exception as e:
            logger.error(f"Failed to log milestone event '{event_name}': {e}")


def log_performance_event(
    event_name: str,
    duration_ms: float,
    event_info:Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None
):
    """记录性能事件的便捷函数"""
    tracer = get_global_tracer()
    if tracer and tracer.is_enabled():
        try:
            # 构建性能数据
            performance_data = {
                "duration_ms": duration_ms,
                "metrics": metrics or {}
            }
            
            # 构建元数据
            metadata = {
                "timestamp": time.time(),
                "event_type": "performance",
                "type": "performance",
                "duration_ms": duration_ms
            }
            
            # 添加调用上下文
            try:
                frame = inspect.currentframe().f_back
                if frame:
                    metadata["caller"] = f"{frame.f_code.co_filename}:{frame.f_lineno}"
            except Exception:
                pass
            
            _safe_create_event(
                tracer.client,
                resolved_name=event_name,
                name=event_name,
                input=_safe_jsonable(event_info),
                output=_safe_jsonable(performance_data),
                metadata=metadata,
                level="DEFAULT"
            )
            logger.debug(f"Performance event logged: {event_name} (duration: {duration_ms}ms)")
        except Exception as e:
            logger.error(f"Failed to log performance event '{event_name}': {e}")


def shutdown_langfuse():
    """在程序退出前调用，确保所有追踪数据被发送"""
    tracer = get_global_tracer()
    if tracer and tracer.is_enabled():
        tracer.shutdown()
