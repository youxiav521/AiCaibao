from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from .common import cpython_default_max_workers
from .default_executor import ObservedThreadPoolExecutor
from .event_loop import EventLoopStallMonitor
from .metrics import NullRuntimeMetrics, RuntimeMetrics, RuntimePrometheusMetrics
from .tracing import RuntimeEventTracer


@dataclass
class RuntimeMonitoringManager:
    """Runtime 监控管理器。

    该类是启动层唯一需要感知的入口，负责把分散的 runtime 监控子系统组装
    起来，并在应用生命周期中统一启动/停止。
    """

    app_config: Any
    tracer: Any = None
    metrics: Optional[RuntimeMetrics] = None

    def __post_init__(self) -> None:
        """初始化 manager 的内部依赖，但不立即启动监控。"""
        self._event_tracer = RuntimeEventTracer(self.tracer)
        self._metrics = self.metrics
        self._event_loop_monitor: Optional[EventLoopStallMonitor] = None
        self._default_executor: Optional[ObservedThreadPoolExecutor] = None
        self._previous_default_executor: Optional[ThreadPoolExecutor] = None
        self._installed_default_executor = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def default_executor(self) -> Optional[ObservedThreadPoolExecutor]:
        return self._default_executor

    def _is_enabled(self) -> bool:
        """判断 runtime 监控总开关是否开启。"""
        monitor = getattr(self.app_config, "monitor", None)
        return bool(monitor and getattr(monitor, "enabled", False))

    def _event_loop_config(self) -> Any:
        """获取 event loop monitor 配置对象。"""
        return getattr(getattr(self.app_config, "monitor", None), "event_loop_monitor", None)

    def _default_executor_config(self) -> Any:
        """获取默认线程池 monitor 配置对象。"""
        return getattr(getattr(self.app_config, "monitor", None), "default_executor_monitor", None)

    def _resolved_metrics(self) -> RuntimeMetrics:
        """解析本次运行实际使用的 metrics 实现。

        解析顺序：

        1. 如果构造时显式注入 metrics，则直接使用。
        2. 如果 Prometheus 已启用，则使用全局单例的 Prometheus 实现。
        3. 否则退回空实现，保证监控状态机仍可独立运行。
        """
        if self._metrics is not None:
            return self._metrics
        prometheus_enabled = bool(getattr(getattr(self.app_config, "prometheus", None), "enabled", False))
        if prometheus_enabled:
            self._metrics = RuntimePrometheusMetrics.get_instance()
        else:
            self._metrics = NullRuntimeMetrics()
        return self._metrics

    async def start(self) -> None:
        """启动 runtime 监控。

        启动顺序上，优先安装默认 executor，再启动 event loop monitor。
        这样后续在应用启动阶段发生的 `asyncio.to_thread()` / `run_in_executor(None, ...)`
        已经能被默认 executor 监控覆盖到。
        """
        if not self._is_enabled():
            return
        if self._loop is not None:
            return

        loop = asyncio.get_running_loop()
        self._loop = loop
        metrics = self._resolved_metrics()

        try:
            default_executor_config = self._default_executor_config()
            if default_executor_config and getattr(default_executor_config, "enabled", False):
                # 与 CPython 默认行为保持一致，除非配置里明确覆盖 max_workers。
                max_workers = getattr(default_executor_config, "max_workers", None) or cpython_default_max_workers()
                self._previous_default_executor = getattr(loop, "_default_executor", None)
                self._default_executor = ObservedThreadPoolExecutor(
                    max_workers=max_workers,
                    config=default_executor_config,
                    event_tracer=self._event_tracer,
                    metrics=metrics,
                )
                loop.set_default_executor(self._default_executor)
                self._installed_default_executor = True

            event_loop_config = self._event_loop_config()
            if event_loop_config and getattr(event_loop_config, "enabled", False):
                self._event_loop_monitor = EventLoopStallMonitor(
                    loop,
                    event_loop_config,
                    event_tracer=self._event_tracer,
                    metrics=metrics,
                )
                self._event_loop_monitor.start()
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """停止 runtime 监控，并按照依赖顺序清理资源。"""
        if self._event_loop_monitor is not None:
            await self._event_loop_monitor.stop()
            self._event_loop_monitor = None
        if self._loop is not None and self._default_executor is not None:
            # 这里不要调用 loop.shutdown_default_executor():
            # 1. 它会无限等待默认线程池里的任务完成，容易把服务停机卡死；
            # 2. 它会让同一 event loop 后续的 asyncio.to_thread()/run_in_executor(None, ...)
            #    直接抛 RuntimeError，影响其他 shutdown 清理逻辑。
            current_default_executor = getattr(self._loop, "_default_executor", None)
            if self._installed_default_executor and current_default_executor is self._default_executor:
                if self._previous_default_executor is None:
                    self._loop._default_executor = None
                else:
                    self._loop.set_default_executor(self._previous_default_executor)
            self._default_executor.shutdown(wait=False, cancel_futures=False)
            self._default_executor = None
            self._previous_default_executor = None
            self._installed_default_executor = False
        self._loop = None
