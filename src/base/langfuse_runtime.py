from __future__ import annotations

import base64
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def _parse_timeout() -> Optional[int]:
    value = os.getenv("LANGFUSE_TIMEOUT")
    if not value:
        return None
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        logger.warning("LANGFUSE_TIMEOUT 非法，已忽略该配置: %s", value)
        return None
    return timeout if timeout > 0 else None


def _mask_public_key(value: Optional[str]) -> str:
    if not value:
        return "***"
    return value[:10] + "..." if len(value) > 10 else "***"


class LangfuseRuntime:
    """共享 Langfuse/OTel 运行时。

    统一负责：

    - 解析 Langfuse 环境变量
    - 创建并缓存 Langfuse SDK client
    - 初始化并缓存 OTel -> Langfuse exporter
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        host: Optional[str] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> None:
        self.requested_enabled = bool(enabled)
        self.host = _first_non_empty(
            host,
            os.getenv("LANGFUSE_API_BASE"),
        )
        self.public_key = _first_non_empty(
            public_key,
            os.getenv("LANGFUSE_PUBLIC_KEY"),
        )
        self.secret_key = _first_non_empty(
            secret_key,
            os.getenv("LANGFUSE_SECRET_KEY"),
        )
        self.environment = environment or os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development")
        self.timeout = _parse_timeout()
        self.client = None
        self.otel_provider = None
        self._lock = threading.Lock()
        self._missing_config_logged = False
        self._langfuse_unavailable_logged = False
        self._otel_unavailable_logged = False

    def _is_config_complete(self) -> bool:
        return bool(self.host and self.public_key and self.secret_key)

    def _log_missing_config_once(self) -> None:
        if self._missing_config_logged:
            return
        logger.warning("Langfuse 配置不完整，已跳过初始化")
        self._missing_config_logged = True

    @property
    def enabled(self) -> bool:
        return self.requested_enabled and self._is_config_complete()

    def ensure_client(self):
        if not self.requested_enabled:
            return None
        if not self._is_config_complete():
            self._log_missing_config_once()
            return None

        with self._lock:
            if self.client is not None:
                return self.client
            try:
                from langfuse import Langfuse
            except ImportError as exc:
                if not self._langfuse_unavailable_logged:
                    logger.warning("Langfuse 未安装，SDK client 初始化跳过: %s", exc)
                    self._langfuse_unavailable_logged = True
                return None

            self.client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                base_url=self.host,
                timeout=self.timeout,
            )
            logger.info(
                "Langfuse runtime 初始化完成: host=%s, public_key=%s",
                self.host,
                _mask_public_key(self.public_key),
            )
            return self.client

    def create_callback_handler(self):
        if not self.requested_enabled:
            return None
        if not self._is_config_complete():
            self._log_missing_config_once()
            return None
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:
            if not self._langfuse_unavailable_logged:
                logger.warning("Langfuse 未安装，CallbackHandler 初始化跳过: %s", exc)
                self._langfuse_unavailable_logged = True
            return None
        return CallbackHandler(public_key=self.public_key)

    def setup_otel(self, *, service_name: str = "yusys-ai"):
        if not self.requested_enabled:
            return None
        if not self._is_config_complete():
            self._log_missing_config_once()
            return None

        with self._lock:
            if self.otel_provider is not None:
                return self.otel_provider

            try:
                from opentelemetry import propagate, trace
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
                from opentelemetry.instrumentation.langchain import LangchainInstrumentor
                from opentelemetry.instrumentation.milvus import MilvusInstrumentor
                from opentelemetry.instrumentation.requests import RequestsInstrumentor
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
            except ImportError as exc:
                if not self._otel_unavailable_logged:
                    logger.warning("OpenTelemetry 依赖不可用，OTel 初始化跳过: %s", exc)
                    self._otel_unavailable_logged = True
                return None

            propagate.set_global_textmap(TraceContextTextMapPropagator())

            resource = Resource.create(
                {
                    "service.name": service_name,
                    "deployment.environment": self.environment,
                }
            )
            provider = TracerProvider(resource=resource)
            auth_header = "Basic " + base64.b64encode(
                f"{self.public_key}:{self.secret_key}".encode("utf-8")
            ).decode("ascii")
            endpoint = f"{self.host.rstrip('/')}/api/public/otel/v1/traces"

            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers={
                    "Authorization": auth_header,
                    "x-langfuse-sdk-name": "python-otel",
                    "x-langfuse-public-key": self.public_key,
                },
                timeout=self.timeout,
            )
            processor = BatchSpanProcessor(
                exporter,
                max_export_batch_size=64,
                schedule_delay_millis=5000,
            )
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

            excluded_urls = self.host.rstrip("/")
            HTTPXClientInstrumentor().instrument()
            RequestsInstrumentor().instrument(excluded_urls=excluded_urls)
            LangchainInstrumentor().instrument()
            MilvusInstrumentor().instrument()

            self.otel_provider = provider
            logger.info("OTel Langfuse 初始化完成: endpoint=%s, env=%s", endpoint, self.environment)
            return self.otel_provider


_runtime_instance: Optional[LangfuseRuntime] = None
_runtime_lock = threading.Lock()


def get_langfuse_runtime(
    *,
    enabled: Optional[bool] = None,
    host: Optional[str] = None,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    environment: Optional[str] = None,
    reset: bool = False,
) -> LangfuseRuntime:
    global _runtime_instance

    with _runtime_lock:
        if reset or _runtime_instance is None:
            _runtime_instance = LangfuseRuntime(
                enabled=True if enabled is None else enabled,
                host=host,
                public_key=public_key,
                secret_key=secret_key,
                environment=environment,
            )
        return _runtime_instance


def reset_langfuse_runtime() -> None:
    global _runtime_instance
    with _runtime_lock:
        _runtime_instance = None


def get_langfuse_client(*, enabled: Optional[bool] = None):
    runtime = get_langfuse_runtime(enabled=enabled)
    return runtime.ensure_client()


def bootstrap_langfuse(
    *,
    service_name: str = "yusys-ai",
    enabled: Optional[bool] = None,
    host: Optional[str] = None,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    environment: Optional[str] = None,
) -> LangfuseRuntime:
    runtime = get_langfuse_runtime(
        enabled=enabled,
        host=host,
        public_key=public_key,
        secret_key=secret_key,
        environment=environment,
    )
    runtime.ensure_client()
    runtime.setup_otel(service_name=service_name)
    return runtime
