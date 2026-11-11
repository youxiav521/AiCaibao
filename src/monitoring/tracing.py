from __future__ import annotations

import os

from opentelemetry import trace


class RuntimeEventTracer:
    """负责发出 runtime 级异常事件 span。

    这里不接管现有 exporter，也不自建新的 provider，而是直接复用当前
    全局 `TracerProvider`。这样可以沿着现有 OTel -> Langfuse 链路上报，
    且不会影响已有的 HTTP / OpenAI / LangChain 等自动埋点。
    """

    def __init__(self, tracer=None) -> None:
        """初始化 runtime tracer。

        Args:
            tracer: 可选的外部 tracer，测试中通常会注入 fake tracer；
                生产环境为空时会从全局 OTel 获取默认 tracer。
        """
        self._tracer = tracer or trace.get_tracer("src.monitoring.runtime")

    def start_event_loop_stall_span(
        self,
        *,
        start_time_ns: int,
        threshold_ms: float,
        lag_ms: float,
        thread_id: int,
        stack: list[str],
    ):
        """创建 event loop stall span。

        Args:
            start_time_ns: stall 发生起点的墙上时间戳（纳秒）。
            threshold_ms: 判定 stall 的阈值。
            lag_ms: 检测到的当前滞后值。
            thread_id: 被判定发生 stall 的 event loop 线程 ID。
            stack: event loop 线程现场堆栈。

        Returns:
            Span: 尚未结束的 runtime span，由调用方在恢复时结束。
        """
        return self._tracer.start_span(
            "runtime.event_loop.stall",
            start_time=start_time_ns,
            attributes={
                "runtime.monitor.type": "event_loop",
                "runtime.monitor.threshold_ms": float(threshold_ms),
                "runtime.event_loop.detected_lag_ms": float(lag_ms),
                "process.pid": os.getpid(),
                "thread.id": int(thread_id),
                "code.stacktrace": "".join(stack),
            },
        )

    def start_default_executor_saturation_span(
        self,
        *,
        start_time_ns: int,
        max_workers: int,
        active_workers: int,
        queued_tasks: int,
        threshold_ms: float,
    ):
        """创建默认线程池饱和 span。

        Args:
            start_time_ns: 饱和开始的墙上时间戳（纳秒）。
            max_workers: 线程池容量上限。
            active_workers: 当前活跃线程数。
            queued_tasks: 当前排队任务数。
            threshold_ms: 配置中的饱和阈值字段，当前主要用于外显配置。

        Returns:
            Span: 尚未结束的 runtime span，由调用方在退出饱和时结束。
        """
        return self._tracer.start_span(
            "runtime.default_executor.saturated",
            start_time=start_time_ns,
            attributes={
                "runtime.monitor.type": "default_executor",
                "runtime.monitor.threshold_ms": float(threshold_ms),
                "runtime.default_executor.max_workers": int(max_workers),
                "runtime.default_executor.active_workers": int(active_workers),
                "runtime.default_executor.queued_tasks": int(queued_tasks),
                "process.pid": os.getpid(),
            },
        )
