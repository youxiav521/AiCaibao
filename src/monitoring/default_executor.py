from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from .metrics import RuntimeMetrics
from .tracing import RuntimeEventTracer


class ObservedThreadPoolExecutor(ThreadPoolExecutor):
    """可观测的默认线程池执行器。

    该执行器的目标不是修改任务行为，而是在 `submit -> 开始执行 -> 执行完成`
    这三个确定边界维护精确状态，从而回答两个问题：

    1. 当前默认线程池有没有被打满。
    2. 被打满时持续了多久、峰值排队有多深。

    “线程池是否满”的严格定义为：

    - `active_workers == max_workers`
    - 且 `queued_tasks > 0`

    只有满足这两个条件时，才认为进入了 saturation。
    """

    def __init__(
        self,
        *,
        max_workers: int,
        config: Any,
        event_tracer: RuntimeEventTracer,
        metrics: RuntimeMetrics,
        thread_name_prefix: str = "runtime-default-executor",
    ) -> None:
        """初始化可观测线程池。

        Args:
            max_workers: 线程池容量上限。
            config: 配置对象，需提供 `emit_queue_metrics`、`saturation_threshold_ms`。
            event_tracer: 用于发出 saturation span。
            metrics: 指标接收器。
            thread_name_prefix: 工作线程名前缀。
        """
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._event_tracer = event_tracer
        self._metrics = metrics
        self._lock = threading.Lock()
        self._submitted = 0
        self._queued = 0
        self._active = 0
        self._completed = 0
        self._peak_queue_depth = 0
        self._peak_active_workers = 0
        self._saturated = False
        self._saturation_span = None
        self._saturation_started_wall_ns: Optional[int] = None
        self._saturation_started_monotonic_ns: Optional[int] = None
        self._saturation_initial_active_workers = 0
        self._saturation_initial_queued_tasks = 0
        self._emit_queue_metrics = bool(getattr(config, "emit_queue_metrics", True))
        self._saturation_threshold_ms = float(getattr(config, "saturation_threshold_ms", 0))
        self._sync_metrics_locked()

    @property
    def submitted_count(self) -> int:
        with self._lock:
            return self._submitted

    @property
    def queued_count(self) -> int:
        with self._lock:
            return self._queued

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active

    @property
    def completed_count(self) -> int:
        with self._lock:
            return self._completed

    def _sync_metrics_locked(self) -> None:
        """把当前内存态计数同步到 metrics。

        该方法要求调用方已经持有 `_lock`，因此命名中显式带上 `_locked`，
        提醒维护者不要在无锁场景下调用，避免读写撕裂。
        """
        self._metrics.set_default_executor_active(self._active)
        if self._emit_queue_metrics:
            self._metrics.set_default_executor_queued(self._queued)

    def _sync_saturation_locked(self, *, now_wall_ns: int, now_monotonic_ns: int) -> None:
        """推进默认线程池“饱和状态机”。

        这是整个模块里最关键的状态机逻辑，职责有三件事：

        1. 判断当前是否满足饱和定义；
        2. 在“未饱和 -> 饱和”边界上创建一次 span；
        3. 在“饱和 -> 未饱和”边界上结束该 span，并记录持续时间/峰值。

        注意这里不是简单看 `queued > 0`：
        如果还有空闲 worker，即使有排队，也不算“线程池已满”，因为系统仍有
        继续消化任务的容量。
        """
        saturated_now = self._active == self._max_workers and self._queued > 0
        if saturated_now:
            if not self._saturated:
                self._saturated = True
                self._saturation_started_wall_ns = now_wall_ns
                self._saturation_started_monotonic_ns = now_monotonic_ns
                self._saturation_initial_active_workers = self._active
                self._saturation_initial_queued_tasks = self._queued
                self._peak_queue_depth = self._queued
                self._peak_active_workers = self._active
            else:
                # 处于饱和期内时，持续刷新峰值，以便 span 结束时写入完整现场。
                self._peak_queue_depth = max(self._peak_queue_depth, self._queued)
                self._peak_active_workers = max(self._peak_active_workers, self._active)
            self._ensure_saturation_span_locked(now_monotonic_ns=now_monotonic_ns)
            return

        if self._saturated:
            self._finish_saturation_locked(
                end_wall_ns=now_wall_ns,
                end_monotonic_ns=now_monotonic_ns,
            )

    def _ensure_saturation_span_locked(self, *, now_monotonic_ns: int) -> None:
        """在达到阈值后创建 saturation span。"""
        if self._saturation_span is not None or self._saturation_started_monotonic_ns is None:
            return
        duration_ms = max(
            now_monotonic_ns - self._saturation_started_monotonic_ns,
            0,
        ) / 1_000_000.0
        if duration_ms < self._saturation_threshold_ms:
            return
        self._saturation_span = self._event_tracer.start_default_executor_saturation_span(
            start_time_ns=self._saturation_started_wall_ns or time.time_ns(),
            max_workers=self._max_workers,
            active_workers=self._saturation_initial_active_workers,
            queued_tasks=self._saturation_initial_queued_tasks,
            threshold_ms=self._saturation_threshold_ms,
        )
        self._metrics.inc_default_executor_saturation()

    def _finish_saturation_locked(self, *, end_wall_ns: int, end_monotonic_ns: int) -> None:
        """结束一次 saturation，并在超过阈值时补齐 span。"""
        if not self._saturated or self._saturation_started_monotonic_ns is None:
            return
        duration_seconds = max(
            end_monotonic_ns - self._saturation_started_monotonic_ns,
            0,
        ) / 1_000_000_000.0
        if self._saturation_span is None:
            self._ensure_saturation_span_locked(now_monotonic_ns=end_monotonic_ns)
        if self._saturation_span is not None:
            self._saturation_span.set_attribute(
                "runtime.default_executor.duration_ms",
                duration_seconds * 1000.0,
            )
            self._saturation_span.set_attribute(
                "runtime.default_executor.peak_queue_depth",
                int(self._peak_queue_depth),
            )
            self._saturation_span.set_attribute(
                "runtime.default_executor.peak_active_workers",
                int(self._peak_active_workers),
            )
            self._saturation_span.end(end_time=end_wall_ns)
            self._metrics.observe_default_executor_saturation_duration(duration_seconds)
        self._saturation_span = None
        self._saturated = False
        self._saturation_started_wall_ns = None
        self._saturation_started_monotonic_ns = None
        self._saturation_initial_active_workers = 0
        self._saturation_initial_queued_tasks = 0
        self._peak_queue_depth = 0
        self._peak_active_workers = 0

    def submit(self, fn: Callable[..., Any], /, *args, **kwargs):  # type: ignore[override]
        """提交任务，并在三个状态边界更新计数。

        状态推进顺序：

        - 提交瞬间：`submitted += 1`, `queued += 1`
        - 真正开始执行：`queued -= 1`, `active += 1`
        - 执行结束：`active -= 1`, `completed += 1`

        之所以要把“提交”和“开始执行”分开，是因为线程池排队问题本质上发生在
        这两者之间。如果只在执行开始时计数，就无法区分“线程池已满”和“任务
        只是尚未被观察到”。
        """
        submit_wall_ns = time.time_ns()
        submit_monotonic_ns = time.monotonic_ns()
        with self._lock:
            self._submitted += 1
            self._queued += 1
            self._metrics.inc_default_executor_submitted()
            self._sync_metrics_locked()
            self._sync_saturation_locked(
                now_wall_ns=submit_wall_ns,
                now_monotonic_ns=submit_monotonic_ns,
            )

        def wrapped():
            # wrapped 在工作线程中执行。真正开始执行时，才说明一个排队槽位被消费。
            start_wall_ns = time.time_ns()
            start_monotonic_ns = time.monotonic_ns()
            with self._lock:
                self._queued -= 1
                self._active += 1
                self._sync_metrics_locked()
                self._sync_saturation_locked(
                    now_wall_ns=start_wall_ns,
                    now_monotonic_ns=start_monotonic_ns,
                )
            try:
                return fn(*args, **kwargs)
            finally:
                finish_wall_ns = time.time_ns()
                finish_monotonic_ns = time.monotonic_ns()
                with self._lock:
                    self._active -= 1
                    self._completed += 1
                    self._metrics.inc_default_executor_completed()
                    self._sync_metrics_locked()
                    self._sync_saturation_locked(
                        now_wall_ns=finish_wall_ns,
                        now_monotonic_ns=finish_monotonic_ns,
                    )

        try:
            return super().submit(wrapped)
        except Exception:
            # 如果底层 submit 失败，前面做过的计数需要回滚，否则会把队列状态污染掉。
            with self._lock:
                self._submitted -= 1
                self._queued -= 1
                self._sync_metrics_locked()
                self._sync_saturation_locked(
                    now_wall_ns=time.time_ns(),
                    now_monotonic_ns=time.monotonic_ns(),
                )
            raise

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:  # type: ignore[override]
        """关闭线程池，并补齐可能仍处于打开状态的 saturation span。"""
        with self._lock:
            if self._saturated:
                self._finish_saturation_locked(
                    end_wall_ns=time.time_ns(),
                    end_monotonic_ns=time.monotonic_ns(),
                )
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
