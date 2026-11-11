from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Optional

from .common import capture_thread_stack
from .metrics import RuntimeMetrics
from .tracing import RuntimeEventTracer


class EventLoopStallMonitor:
    """事件循环阻塞监控器。

    设计思路：

    - 在 event loop 所在线程内起一个轻量心跳协程，周期性更新时间戳。
    - 在独立 watchdog 线程里比较“当前时间 - 最近一次心跳时间”。
    - 如果差值超过阈值，说明 event loop 在这段时间内没有推进，视为 stall。

    这种方案的优点是：

    - 不依赖 `asyncio` debug 模式；
    - 不需要修改业务调用点；
    - 判断条件是单调时钟差值，定义明确且稳定。
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        config: Any,
        *,
        event_tracer: RuntimeEventTracer,
        metrics: RuntimeMetrics,
    ) -> None:
        """初始化事件循环阻塞监控器。

        Args:
            loop: 被监控的事件循环。
            config: 配置对象，需提供 `poll_ms`、`threshold_ms`、`stack_depth`。
            event_tracer: 用于发出异常事件 span。
            metrics: 指标接收器。
        """
        self._loop = loop
        self._event_tracer = event_tracer
        self._metrics = metrics
        self._poll_interval_s = max(float(getattr(config, "poll_ms", 50)) / 1000.0, 0.01)
        self._threshold_ms = float(getattr(config, "threshold_ms", 200))
        self._stack_depth = max(int(getattr(config, "stack_depth", 25)), 1)
        self._heartbeat_lock = threading.Lock()
        self._last_heartbeat_monotonic_ns = time.monotonic_ns()
        self._last_heartbeat_wall_ns = time.time_ns()
        self._fallback_thread_id = threading.main_thread().ident or threading.get_ident()
        self._loop_thread_id: Optional[int] = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._heartbeat_task: Optional[asyncio.Task[Any]] = None
        self._stall_span = None
        self._stall_started_monotonic_ns: Optional[int] = None

    async def _heartbeat(self) -> None:
        """在 loop 线程中持续更新心跳时间。

        这是整个 stall 检测的“基准信号”。只要 loop 正常推进，这个协程就会
        周期性运行并刷新两个时间戳：

        - monotonic: 用于严谨计算 lag / duration
        - wall time: 用于给 OTel span 提供真实起止时间
        """
        while not self._watchdog_stop.is_set():
            now_mono = time.monotonic_ns()
            now_wall = time.time_ns()
            with self._heartbeat_lock:
                self._last_heartbeat_monotonic_ns = now_mono
                self._last_heartbeat_wall_ns = now_wall
            await asyncio.sleep(self._poll_interval_s)

    def _watchdog_main(self) -> None:
        """在独立线程中判断 event loop 是否卡顿。

        关键逻辑：

        - watchdog 自己绝不能跑在被监控的 event loop 线程里，否则 loop 一旦
          阻塞，watchdog 也会一起停掉，监控失效。
        - 这里使用单调时钟差值作为判定标准，避免系统时间跳变影响结果。
        - 只有在“未处于 stall 中”且 lag 首次超过阈值时，才创建一条 span；
          这样一次连续 stall 只会产生一个事件，而不是每次轮询都重复打点。
        """
        while not self._watchdog_stop.is_set():
            time.sleep(self._poll_interval_s)
            with self._heartbeat_lock:
                heartbeat_mono_ns = self._last_heartbeat_monotonic_ns
                heartbeat_wall_ns = self._last_heartbeat_wall_ns
            now_mono_ns = time.monotonic_ns()
            lag_ns = max(now_mono_ns - heartbeat_mono_ns, 0)
            lag_ms = lag_ns / 1_000_000.0
            self._metrics.set_event_loop_lag(lag_ns / 1_000_000_000.0)

            if self._stall_span is None and lag_ms >= self._threshold_ms:
                # 首次跨过阈值：记录现场堆栈，并创建一个“打开状态”的异常 span。
                # 后续只要还没恢复，就维持这一个 span，不重复创建。
                thread_id = self._loop_thread_id or self._fallback_thread_id
                stack = capture_thread_stack(thread_id, limit=self._stack_depth)
                self._stall_span = self._event_tracer.start_event_loop_stall_span(
                    start_time_ns=heartbeat_wall_ns,
                    threshold_ms=self._threshold_ms,
                    lag_ms=lag_ms,
                    thread_id=thread_id,
                    stack=stack,
                )
                self._stall_started_monotonic_ns = heartbeat_mono_ns
                self._metrics.inc_event_loop_stall()
                continue

            if self._stall_span is not None and lag_ms < self._threshold_ms:
                # 从 stall 状态恢复：补齐持续时间并结束 span。
                self._finish_stall(end_wall_ns=time.time_ns(), end_monotonic_ns=now_mono_ns)

    def _finish_stall(self, *, end_wall_ns: int, end_monotonic_ns: int) -> None:
        """结束一次 stall，并回写持续时间指标。

        Args:
            end_wall_ns: 结束时的墙上时间戳（纳秒）。
            end_monotonic_ns: 结束时的单调时间戳（纳秒）。
        """
        if self._stall_span is None or self._stall_started_monotonic_ns is None:
            return
        duration_seconds = max(end_monotonic_ns - self._stall_started_monotonic_ns, 0) / 1_000_000_000.0
        self._stall_span.set_attribute("runtime.event_loop.duration_ms", duration_seconds * 1000.0)
        self._stall_span.end(end_time=end_wall_ns)
        self._metrics.observe_event_loop_stall_duration(duration_seconds)
        self._stall_span = None
        self._stall_started_monotonic_ns = None

    def start(self) -> None:
        """启动监控。

        该方法会同时启动：

        - loop 内的心跳协程
        - 独立 watchdog 线程
        """
        if self._heartbeat_task is not None:
            return
        self._loop_thread_id = threading.get_ident()
        self._heartbeat_task = self._loop.create_task(self._heartbeat(), name="runtime-event-loop-heartbeat")
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_main,
            name="runtime-event-loop-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    async def stop(self) -> None:
        """停止监控，并在必要时关闭悬挂中的 stall span。"""
        self._watchdog_stop.set()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None
        if self._stall_span is not None:
            self._finish_stall(end_wall_ns=time.time_ns(), end_monotonic_ns=time.monotonic_ns())
