from __future__ import annotations

import threading
from typing import Optional, Protocol

try:
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover
    Counter = Gauge = Histogram = None  # type: ignore[assignment]


class RuntimeMetrics(Protocol):
    """Runtime 监控指标抽象协议。

    监控实现只依赖这组方法，而不直接依赖 Prometheus。这样可以让：

    - 生产环境接入 Prometheus 指标；
    - 测试环境注入 fake metrics；
    - 将来如果要切换指标后端，也不需要改监控状态机代码。
    """
    def set_event_loop_lag(self, value: float) -> None: ...

    def inc_event_loop_stall(self) -> None: ...

    def observe_event_loop_stall_duration(self, duration_seconds: float) -> None: ...

    def set_default_executor_active(self, value: int) -> None: ...

    def set_default_executor_queued(self, value: int) -> None: ...

    def inc_default_executor_submitted(self) -> None: ...

    def inc_default_executor_completed(self) -> None: ...

    def inc_default_executor_saturation(self) -> None: ...

    def observe_default_executor_saturation_duration(self, duration_seconds: float) -> None: ...


class NullRuntimeMetrics:
    """空实现指标接收器。

    当 Prometheus 未启用、或测试不关心真实指标时，监控代码仍然会照常
    执行状态机，只是所有指标写入都变成 no-op。这样可以避免在监控逻辑
    中到处写 `if metrics is enabled` 之类的分支。
    """
    def set_event_loop_lag(self, value: float) -> None:
        return None

    def inc_event_loop_stall(self) -> None:
        return None

    def observe_event_loop_stall_duration(self, duration_seconds: float) -> None:
        return None

    def set_default_executor_active(self, value: int) -> None:
        return None

    def set_default_executor_queued(self, value: int) -> None:
        return None

    def inc_default_executor_submitted(self) -> None:
        return None

    def inc_default_executor_completed(self) -> None:
        return None

    def inc_default_executor_saturation(self) -> None:
        return None

    def observe_default_executor_saturation_duration(self, duration_seconds: float) -> None:
        return None


class RuntimePrometheusMetrics:
    """Prometheus 指标实现。

    该类统一持有 runtime 监控相关的 Counter / Gauge / Histogram，并通过
    单例模式避免重复注册同名指标。否则在测试重跑、模块重复导入或多次
    初始化 manager 时，容易触发 Prometheus 指标重复注册错误。
    """
    _instance: Optional["RuntimePrometheusMetrics"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        if Counter is None or Gauge is None or Histogram is None:  # pragma: no cover
            raise RuntimeError("prometheus_client is unavailable")
        # 事件循环类指标：
        # - total / duration_seconds 用于统计 stall 次数和持续时间分布
        # - lag_seconds 用于暴露当前观察到的瞬时 loop 滞后
        self.event_loop_stall_total = Counter(
            "yusys_event_loop_stall_total",
            "Total number of event loop stall episodes.",
        )
        self.event_loop_stall_duration_seconds = Histogram(
            "yusys_event_loop_stall_duration_seconds",
            "Duration of event loop stalls in seconds.",
            buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0, 30.0),
        )
        self.event_loop_lag_seconds = Gauge(
            "yusys_event_loop_lag_seconds",
            "Current observed event loop lag in seconds.",
        )
        # 默认线程池类指标：
        # - active / queued 是当前状态
        # - submitted / completed 是累计吞吐
        # - saturation_* 用于统计“线程池打满且存在排队”的持续时间
        self.default_executor_active_workers = Gauge(
            "yusys_default_executor_active_workers",
            "Number of active workers in the loop default executor.",
        )
        self.default_executor_queued_tasks = Gauge(
            "yusys_default_executor_queued_tasks",
            "Number of queued tasks in the loop default executor.",
        )
        self.default_executor_submitted_total = Counter(
            "yusys_default_executor_submitted_total",
            "Total tasks submitted to the loop default executor.",
        )
        self.default_executor_completed_total = Counter(
            "yusys_default_executor_completed_total",
            "Total tasks completed by the loop default executor.",
        )
        self.default_executor_saturation_total = Counter(
            "yusys_default_executor_saturation_total",
            "Total saturation episodes for the loop default executor.",
        )
        self.default_executor_saturation_duration_seconds = Histogram(
            "yusys_default_executor_saturation_duration_seconds",
            "Duration of default executor saturation episodes in seconds.",
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0, 30.0),
        )

    @classmethod
    def get_instance(cls) -> RuntimeMetrics:
        """获取全局唯一的 Prometheus runtime metrics 实例。

        Returns:
            RuntimeMetrics: 如果 Prometheus 可用则返回真实实现，否则返回空实现。
        """
        if Counter is None or Gauge is None or Histogram is None:  # pragma: no cover
            return NullRuntimeMetrics()
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def set_event_loop_lag(self, value: float) -> None:
        self.event_loop_lag_seconds.set(max(value, 0.0))

    def inc_event_loop_stall(self) -> None:
        self.event_loop_stall_total.inc()

    def observe_event_loop_stall_duration(self, duration_seconds: float) -> None:
        self.event_loop_stall_duration_seconds.observe(max(duration_seconds, 0.0))

    def set_default_executor_active(self, value: int) -> None:
        self.default_executor_active_workers.set(max(value, 0))

    def set_default_executor_queued(self, value: int) -> None:
        self.default_executor_queued_tasks.set(max(value, 0))

    def inc_default_executor_submitted(self) -> None:
        self.default_executor_submitted_total.inc()

    def inc_default_executor_completed(self) -> None:
        self.default_executor_completed_total.inc()

    def inc_default_executor_saturation(self) -> None:
        self.default_executor_saturation_total.inc()

    def observe_default_executor_saturation_duration(self, duration_seconds: float) -> None:
        self.default_executor_saturation_duration_seconds.observe(max(duration_seconds, 0.0))
