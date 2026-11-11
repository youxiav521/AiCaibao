"""OpenTelemetry + Langfuse 自动仪器化集成"""
import logging
from typing import Any, Dict, Optional

# 抑制 OTel SDK 因第三方 instrumentor 格式问题产生的大量 WARNING
# (如 "Description ... ignored. Use either `Status` or `(StatusCode, Description)`")
logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.ERROR)

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.base.langfuse_runtime import bootstrap_langfuse
from src.utils import json_utils

logger = logging.getLogger(__name__)
_ORIGINAL_REQUEST_JSON = Request.json
_REQUEST_JSON_PATCHED = False


def _parse_content_length(value: Optional[str]) -> Optional[int]:
    """解析 Content-Length，异常时返回 None。"""
    if not value:
        return None
    try:
        length = int(value)
    except (TypeError, ValueError):
        return None
    return length if length >= 0 else None


def _supports_body_preview(content_type: Optional[str]) -> bool:
    """仅对文本类和 JSON body 做预览，避免对二进制内容做无意义解码。"""
    normalized = (content_type or "").lower()
    if not normalized:
        return False
    if json_utils.is_json_content_type(normalized):
        return True
    return any(
        marker in normalized
        for marker in (
            "application/x-www-form-urlencoded",
            "application/xml",
            "text/",
        )
    )


def _truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[:max_length] + f"...(truncated,{len(value)})"


def _preview_body(body: bytes, content_type: Optional[str], max_body_size: int) -> Dict[str, Any]:
    """生成轻量 body 预览，只处理已知的小文本内容。"""
    preview: Dict[str, Any] = {}
    if len(body) > max_body_size:
        preview["body"] = body[:max_body_size].decode("utf-8", errors="ignore")
        preview["body_truncated"] = True
        return preview

    if json_utils.is_json_content_type(content_type):
        try:
            preview["body"] = json_utils.loads(body)
            return preview
        except Exception:
            pass

    text = _truncate_text(body.decode("utf-8", errors="ignore"), max_body_size)
    preview["body"] = text
    if len(body) > max_body_size:
        preview["body_truncated"] = True
    return preview


def _cache_request_json(request: Request, body: bytes, content_type: Optional[str]) -> None:
    """预解析并缓存 JSON 请求体，供后续 FastAPI/Pydantic 复用。"""
    if not body:
        return
    if content_type and not json_utils.is_json_content_type(content_type):
        return
    try:
        parsed = json_utils.loads(body)
    except Exception:
        return
    request._json = parsed
    request.scope["_cached_request_json"] = parsed


def _patch_starlette_request_json() -> None:
    global _REQUEST_JSON_PATCHED
    if _REQUEST_JSON_PATCHED:
        return

    async def _request_json_with_cache(self: Request):
        if not hasattr(self, "_json"):
            if "_cached_request_json" in self.scope:
                self._json = self.scope["_cached_request_json"]
            else:
                return await _ORIGINAL_REQUEST_JSON(self)
        return self._json

    Request.json = _request_json_with_cache
    _REQUEST_JSON_PATCHED = True


class OtelLangfuseMiddleware(BaseHTTPMiddleware):
    """在 OTEL span 上添加 Langfuse input/output 属性"""
    
    MAX_BODY_SIZE = 4096
    # 跳过响应体读取的路径（这些路径需要快速响应）
    # SKIP_RESPONSE_BODY_PATHS = ["/api/template/parse","/api/template/sse"]
    SKIP_RESPONSE_BODY_PATHS=[]
    
    async def dispatch(self, request: Request, call_next):
        input_data = {
            "method": request.method,
            "path": str(request.url.path),
            "query": str(request.url.query) if request.url.query else None,
        }

        request_content_type = request.headers.get("content-type")
        request_content_length = _parse_content_length(request.headers.get("content-length"))
        if request_content_type:
            input_data["content_type"] = request_content_type
        if request_content_length is not None:
            input_data["content_length"] = request_content_length

        should_cache_request_json = not request_content_type or json_utils.is_json_content_type(request_content_type)
        request_body_bytes = b""
        if should_cache_request_json:
            request_body_bytes = await request.body()
            if request_body_bytes:
                _cache_request_json(request, request_body_bytes, request_content_type)

        should_preview_request_body = (
            request_content_length is not None
            and request_content_length <= self.MAX_BODY_SIZE
            and _supports_body_preview(request_content_type)
        )
        if should_preview_request_body:
            if request_body_bytes:
                input_data.update(_preview_body(request_body_bytes, request_content_type, self.MAX_BODY_SIZE))
        elif request_content_length:
            input_data["body_omitted"] = "request_body_too_large_or_unsupported"
        elif request_body_bytes:
            input_data["body_omitted"] = "request_body_preview_skipped"
        
        response = await call_next(request)
        span = trace.get_current_span()
        
        output_data = {"status_code": response.status_code}
        response_content_type = response.headers.get("content-type")
        response_content_length = _parse_content_length(response.headers.get("content-length"))
        if response_content_type:
            output_data["content_type"] = response_content_type
        if response_content_length is not None:
            output_data["content_length"] = response_content_length
        
        # 检查是否需要跳过响应体读取（对于需要快速响应的接口）
        skip_response_body = any(
            request.url.path.startswith(path) for path in self.SKIP_RESPONSE_BODY_PATHS
        )
        
        if skip_response_body:
            output_data["body_omitted"] = "path_skipped"
        else:
            response_body = getattr(response, "body", None)
            if isinstance(response_body, str):
                response_body = response_body.encode("utf-8")
            elif isinstance(response_body, memoryview):
                response_body = response_body.tobytes()
            elif isinstance(response_body, bytearray):
                response_body = bytes(response_body)

            can_preview_response_body = (
                isinstance(response_body, bytes)
                and len(response_body) <= self.MAX_BODY_SIZE
                and _supports_body_preview(response_content_type)
            )
            if can_preview_response_body:
                output_data.update(_preview_body(response_body, response_content_type, self.MAX_BODY_SIZE))
            elif isinstance(response_body, bytes):
                output_data["body_omitted"] = "response_body_too_large_or_unsupported"
            elif hasattr(response, "body_iterator"):
                output_data["body_omitted"] = "response_body_streaming_or_unavailable"
            elif response_content_length:
                output_data["body_omitted"] = "response_body_too_large_or_streaming"
        
        if span.is_recording():
            input_json = json_utils.dumps(input_data, default=str)
            output_json = json_utils.dumps(output_data, default=str)
            
            span.set_attribute("langfuse.observation.input", input_json)
            span.set_attribute("langfuse.observation.output", output_json)
            span.set_attribute("gen_ai.prompt", input_json)
            span.set_attribute("gen_ai.completion", output_json)
            span.set_attribute("input.value", input_json)
            span.set_attribute("output.value", output_json)
        
        return response


def setup_otel_langfuse(
    service_name: str = "yusys-ai",
    enabled: Optional[bool] = None,
    LANGFUSE_API_BASE: Optional[str] = None,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    environment: Optional[str] = None,
) -> Optional[TracerProvider]:
    """初始化共享 Langfuse runtime，并启用 OTel exporter。"""
    runtime = bootstrap_langfuse(
        service_name=service_name,
        enabled=enabled,
        host=LANGFUSE_API_BASE,
        public_key=public_key,
        secret_key=secret_key,
        environment=environment,
    )
    return runtime.otel_provider


def instrument_fastapi(app):
    """仪器化 FastAPI 应用"""
    FastAPIInstrumentor.instrument_app(app)
    _patch_starlette_request_json()

    @app.middleware("http")
    async def cache_json_request_body(request: Request, call_next):
        content_type = request.headers.get("content-type")
        if not content_type or json_utils.is_json_content_type(content_type):
            body = await request.body()
            if body:
                try:
                    request._json = json_utils.loads(body)
                except Exception:
                    pass
        return await call_next(request)

    # 仅捕捉 接口函数 的Input 、Output ，其他自定义函数想步骤Input、Output请添加注解 
    app.add_middleware(OtelLangfuseMiddleware)
    logger.info("OTEL: FastAPI 仪器化完成")


def get_tracer(name: str = __name__):
    """获取 tracer 实例"""
    return trace.get_tracer(name)
