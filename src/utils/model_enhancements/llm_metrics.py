from contextvars import ContextVar
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# ============ 事件类型定义 ============
class EventType(str, Enum):
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    LLM_ERROR = "llm_error"
    QUEUE_WAIT = "queue_wait"      # 排队等待事件


# ============ 当前 LLM 调用标识（用于关联排队/重试事件） ============
_CURRENT_LLM_CALL_ID: ContextVar[Optional[str]] = ContextVar(
    "current_llm_call_id", default=None
)


def set_current_llm_call_id(call_id: Optional[str]) -> None:
    """设置当前 LLM 调用标识"""
    _CURRENT_LLM_CALL_ID.set(call_id)


def get_current_llm_call_id() -> Optional[str]:
    """获取当前 LLM 调用标识"""
    return _CURRENT_LLM_CALL_ID.get()


@dataclass
class LlmEvent:
    """LLM 调用事件"""
    event_type: EventType
    model_key: str
    provider: str
    timestamp: float
    duration_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    queue_wait_ms: Optional[float] = None      # 排队等待时间(ms)
    retry_count: Optional[int] = None          # 重试次数


# ============ Metrics 计算结果 ============
@dataclass
class ModelKeyMetrics:
    """按 model_key 分组的指标"""
    call_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_queue_wait_ms: float = 0.0           # 总排队等待时间
    total_retry_count: int = 0                 # 总重试次数

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / max(self.call_count, 1)

    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.call_count, 1)

    @property
    def avg_queue_wait_ms(self) -> float:      # 平均排队等待时间
        return self.total_queue_wait_ms / max(self.call_count, 1)

    @property
    def avg_retry_count(self) -> float:         # 平均重试次数
        return self.total_retry_count / max(self.call_count, 1)


@dataclass
class MetricsResult:
    """Metrics 计算结果，支持格式化输出"""
    total_calls: int
    total_errors: int
    total_duration_ms: float
    total_tokens_in: int
    total_tokens_out: int
    total_queue_wait_ms: float
    total_retry_count: int
    avg_duration_ms: float
    avg_queue_wait_ms: float
    avg_retry_count: float
    overall_error_rate: float
    by_model_key: Dict[str, ModelKeyMetrics]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict"""
        return {
            "summary": {
                "total_calls": self.total_calls,
                "total_errors": self.total_errors,
                "error_rate": f"{self.overall_error_rate:.2%}",
                "total_duration_ms": round(self.total_duration_ms, 2),
                "avg_duration_ms": round(self.avg_duration_ms, 2),
                "total_queue_wait_ms": round(self.total_queue_wait_ms, 2),
                "avg_queue_wait_ms": round(self.avg_queue_wait_ms, 2),
                "total_retry_count": self.total_retry_count,
                "avg_retry_count": round(self.avg_retry_count, 2),
                "total_tokens_in_k": round(self.total_tokens_in / 1000, 2),
                "total_tokens_out_k": round(self.total_tokens_out / 1000, 2),
            },
            "by_model_key": {
                mk: {
                    "call_count": m.call_count,
                    "error_count": m.error_count,
                    "error_rate": f"{m.error_rate:.2%}",
                    "total_duration_ms": round(m.total_duration_ms, 2),
                    "avg_duration_ms": round(m.avg_duration_ms, 2),
                    "total_queue_wait_ms": round(m.total_queue_wait_ms, 2),
                    "avg_queue_wait_ms": round(m.avg_queue_wait_ms, 2),
                    "total_retry_count": m.total_retry_count,
                    "avg_retry_count": round(m.avg_retry_count, 2),
                    "tokens_in_k": round(m.total_tokens_in / 1000, 2),
                    "tokens_out_k": round(m.total_tokens_out / 1000, 2),
                    "duration_percent": f"{m.total_duration_ms / max(self.total_duration_ms, 1):.2%}",
                }
                for mk, m in self.by_model_key.items()
            }
        }

    def pretty_print(self) -> str:
        """友好打印输出"""
        # 辅助函数：格式化耗时显示，超过1500ms用s单位
        def format_duration(ms: float) -> str:
            if ms >= 1500:
                return f"{ms/1000:.2f}s"
            return f"{ms:.2f}ms"

        lines = ["=" * 60, "📊 LLM 调用统计报告", "-" * 60]

        # 整体统计
        lines.extend([
            f"📈 整体统计:",
            f"  总调用次数: {self.total_calls}",
            f"  错误次数: {self.total_errors} ({self.overall_error_rate:.2%})",
            f"  总耗时: {format_duration(self.total_duration_ms)}",
            f"  平均耗时: {format_duration(self.avg_duration_ms)}",
            f"  平均排队: {format_duration(self.avg_queue_wait_ms)}",
            f"  平均重试: {self.avg_retry_count:.2f} 次",
            f"  总输入 Token: {self.total_tokens_in / 1000:.2f}K",
            f"  总输出 Token: {self.total_tokens_out / 1000:.2f}K",
            "",
            f"📋 按 Model Key 分组:"
        ])

        # 分组统计
        for model_key, m in self.by_model_key.items():
            duration_pct = m.total_duration_ms / max(self.total_duration_ms, 1)
            lines.extend([
                f"  [{model_key}]:",
                f"    调用: {m.call_count} 次 | 错误: {m.error_count} ({m.error_rate:.2%})",
                f"    耗时: {format_duration(m.total_duration_ms)} ({duration_pct:.2%}) | 平均: {format_duration(m.avg_duration_ms)}",
                f"    排队: 平均 {format_duration(m.avg_queue_wait_ms)} | 重试: 平均 {m.avg_retry_count:.2f} 次",
                f"    Token: 输入 {m.total_tokens_in / 1000:.2f}K | 输出 {m.total_tokens_out / 1000:.2f}K"
            ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def pretty_print_compact(self) -> str:
        """紧凑格式打印输出，适合日志查看"""
        if self.total_calls == 0:
            return "LLM 调用统计: 无调用"

        total_duration_s = self.total_duration_ms / 1000

        # 辅助函数：格式化耗时显示，超过1500ms用s单位
        def format_duration(ms: float) -> str:
            if ms >= 1500:
                return f"{ms/1000:>5.1f}s"
            return f"{ms:>5.0f}ms"

        # 先收集所有数据行
        data_rows = []
        for model_key, m in self.by_model_key.items():
            if ':' in model_key:
                provider, model_name = model_key.split(':', 1)
            else:
                provider = "unknown"
                model_name = model_key

            provider_short = provider[:9]
            model_short = model_name[:16]
            token_in = f"{m.total_tokens_in/1000:.1f}K"
            token_out = f"{m.total_tokens_out/1000:.1f}K"
            token_str = f"{token_in:>6}/{token_out:<6}"

            data_rows.append(
                f" {provider_short:>9} | {model_short:<16} | {m.call_count:>6} | {format_duration(m.avg_duration_ms)} | "
                f"{format_duration(m.avg_queue_wait_ms)} | {m.avg_retry_count:>5.2f} | {token_str:^16}"
            )

        # 计算最长行来决定分隔线长度
        header_row = " Provider |    Model Name    | 请求数 | 平均耗时 | 排队时间 | 重试  |     Token      "
        max_length = max(len(header_row), max((len(row) for row in data_rows), default=0))
        max_length = max(max_length, 60)  # 最小宽度60

        # 生成分隔线
        separator = "─" * max_length
        title = f"LLM 调用统计 (总耗时: {total_duration_s:.1f}s)"
        title_pad = "═" * max((max_length - len(title) - 4) // 2, 1)
        title_line = f"{title_pad} {title} {title_pad}"

        # 汇总行
        model_count = len(self.by_model_key)
        total_token_in = f"{self.total_tokens_in/1000:.1f}K"
        total_token_out = f"{self.total_tokens_out/1000:.1f}K"
        total_token_str = f"{total_token_in:>6}/{total_token_out:<6}"
        summary_row = (
            f" 汇总 ({model_count:>2}个) |                  | {self.total_calls:>6} | {format_duration(self.avg_duration_ms)} | "
            f"{format_duration(self.avg_queue_wait_ms)} | {self.avg_retry_count:>5.2f} | {total_token_str:^16}"
        )

        # 组合所有行
        lines = [
            title_line,
            header_row,
            separator,
        ]
        lines.extend(data_rows)
        lines.extend([
            separator,
            summary_row,
        ])

        return "\n".join(lines)


# ============ Metrics 上下文管理器 ============
class MetricsContext:
    """LLM 调用统计上下文"""

    def __init__(self):
        self._lock = threading.Lock()
        self._events: List[LlmEvent] = []
        self._pending_calls: Dict[str, LlmEvent] = {}  # call_id -> start event
        self._call_queue_wait: Dict[str, float] = {}   # call_id -> queue_wait_ms
        self._call_retry_count: Dict[str, int] = {}    # call_id -> retry_count
        self._pending_queue_wait: Optional[float] = None  # 暂存还没有 call_id 的排队时间

    def add(self, event: LlmEvent) -> None:
        """添加事件"""
        with self._lock:
            self._events.append(event)

    def track_start(self, call_id: str, model_key: str, provider: str) -> None:
        """跟踪调用开始"""
        event = LlmEvent(
            event_type=EventType.LLM_START,
            model_key=model_key,
            provider=provider,
            timestamp=time.time(),
        )
        with self._lock:
            self._pending_calls[call_id] = event
            # 如果有暂存的排队时间，关联到这个 call_id
            if self._pending_queue_wait is not None:
                self._call_queue_wait[call_id] = self._pending_queue_wait
                self._pending_queue_wait = None

    def track_queue_wait(self, _model_key: str, _provider: str, queue_wait_ms: float) -> None:
        """记录排队等待时间 - 与当前 LLM 调用关联"""
        call_id = get_current_llm_call_id()
        with self._lock:
            if call_id:
                self._call_queue_wait[call_id] = queue_wait_ms
            else:
                # 暂时没有 call_id，暂存排队时间，等下一个 track_start 时关联
                # （适用于 ainvoke 路径：先获取限制器，再调用 LLM）
                self._pending_queue_wait = queue_wait_ms

    def track_retry(self, call_id: str, retry_count: int) -> None:
        """记录重试次数"""
        with self._lock:
            self._call_retry_count[call_id] = retry_count

    def track_end(self, call_id: str, tokens_in: int, tokens_out: int) -> None:
        """跟踪调用结束"""
        with self._lock:
            start_event = self._pending_calls.pop(call_id, None)
            queue_wait_ms = self._call_queue_wait.pop(call_id, None)
            retry_count = self._call_retry_count.pop(call_id, None)
            if start_event:
                duration = (time.time() - start_event.timestamp) * 1000
                self._events.append(LlmEvent(
                    event_type=EventType.LLM_END,
                    model_key=start_event.model_key,
                    provider=start_event.provider,
                    timestamp=time.time(),
                    duration_ms=duration,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    queue_wait_ms=queue_wait_ms,
                    retry_count=retry_count,
                ))

    def track_error(self, call_id: str, error_type: str, error_message: str) -> None:
        """跟踪调用错误"""
        with self._lock:
            start_event = self._pending_calls.pop(call_id, None)
            queue_wait_ms = self._call_queue_wait.pop(call_id, None)
            retry_count = self._call_retry_count.pop(call_id, None)
            if start_event:
                duration = (time.time() - start_event.timestamp) * 1000
                self._events.append(LlmEvent(
                    event_type=EventType.LLM_ERROR,
                    model_key=start_event.model_key,
                    provider=start_event.provider,
                    timestamp=time.time(),
                    duration_ms=duration,
                    error_type=error_type,
                    error_message=error_message,
                    queue_wait_ms=queue_wait_ms,
                    retry_count=retry_count,
                ))

    @property
    def llm_api_calls(self) -> int:
        """总 LLM 调用次数（完成的）"""
        return sum(
            1 for e in self._events
            if e.event_type in (EventType.LLM_END, EventType.LLM_ERROR)
        )

    @property
    def llm_scheduled_units(self) -> int:
        """总 LLM 计划调用次数（等于完成次数，目前设计不额外计数）"""
        return self.llm_api_calls

    @property
    def total_queue_wait_ms(self) -> float:
        """总排队等待时间（毫秒）"""
        return sum(self._call_queue_wait.values())

    def calculate(self) -> MetricsResult:
        """计算最终 metrics"""
        with self._lock:
            events = list(self._events)

        total_calls = 0
        total_errors = 0
        total_duration_ms = 0.0
        total_tokens_in = 0
        total_tokens_out = 0
        total_queue_wait_ms = 0.0
        total_retry_count = 0

        by_model_key: Dict[str, ModelKeyMetrics] = {}

        for event in events:
            if event.event_type == EventType.LLM_START:
                continue

            # 获取或创建 model_key 统计
            mk_stats = by_model_key.setdefault(
                event.model_key, ModelKeyMetrics()
            )

            total_calls += 1
            mk_stats.call_count += 1

            if event.event_type == EventType.LLM_END:
                total_duration_ms += event.duration_ms or 0
                total_tokens_in += event.tokens_in or 0
                total_tokens_out += event.tokens_out or 0
                mk_stats.total_duration_ms += event.duration_ms or 0
                mk_stats.total_tokens_in += event.tokens_in or 0
                mk_stats.total_tokens_out += event.tokens_out or 0

                # 排队时间统计
                if event.queue_wait_ms is not None:
                    total_queue_wait_ms += event.queue_wait_ms
                    mk_stats.total_queue_wait_ms += event.queue_wait_ms

                # 重试次数统计
                if event.retry_count is not None:
                    total_retry_count += event.retry_count
                    mk_stats.total_retry_count += event.retry_count

            elif event.event_type == EventType.LLM_ERROR:
                total_errors += 1
                mk_stats.error_count += 1

                # 排队时间统计（失败调用也统计）
                if event.queue_wait_ms is not None:
                    total_queue_wait_ms += event.queue_wait_ms
                    mk_stats.total_queue_wait_ms += event.queue_wait_ms

                # 重试次数统计
                if event.retry_count is not None:
                    total_retry_count += event.retry_count
                    mk_stats.total_retry_count += event.retry_count

        avg_duration_ms = total_duration_ms / max(total_calls, 1)
        avg_queue_wait_ms = total_queue_wait_ms / max(total_calls, 1)
        avg_retry_count = total_retry_count / max(total_calls, 1)
        overall_error_rate = total_errors / max(total_calls, 1)

        return MetricsResult(
            total_calls=total_calls,
            total_errors=total_errors,
            total_duration_ms=total_duration_ms,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            total_queue_wait_ms=total_queue_wait_ms,
            total_retry_count=total_retry_count,
            avg_duration_ms=avg_duration_ms,
            avg_queue_wait_ms=avg_queue_wait_ms,
            avg_retry_count=avg_retry_count,
            overall_error_rate=overall_error_rate,
            by_model_key=by_model_key,
        )


# ============ ContextVar 全局上下文 ============
_METRICS_CONTEXT: ContextVar[Optional[MetricsContext]] = ContextVar(
    "metrics_context", default=None
)


@asynccontextmanager
async def metrics_context_session(auto_print: bool = True, compact: bool = False):
    """
    Metrics 上下文会话，用于包裹一次 API 调用或 Worker 任务

    使用方式:
    async with metrics_context_session() as ctx:
        # ... 执行 LLM 调用
        result = ctx.calculate()
        print(result.pretty_print())

    Args:
        auto_print: 是否自动打印统计结果
        compact: 是否使用紧凑输出格式（适合日志）
    """
    ctx = MetricsContext()
    token = _METRICS_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _METRICS_CONTEXT.reset(token)
        if auto_print:
            result = ctx.calculate()
            if result.total_calls > 0:
                if compact:
                    print(result.pretty_print_compact())
                else:
                    print(result.pretty_print())


def get_current_metrics_context() -> Optional[MetricsContext]:
    """获取当前上下文中的 MetricsContext"""
    return _METRICS_CONTEXT.get()


# ============ LangChain Callback Handler ============
class MetricsCallbackHandler(BaseCallbackHandler):
    """LangChain Callback，自动收集 LLM 调用指标"""

    def __init__(self, model_key: str, provider: str):
        super().__init__()
        self.model_key = model_key
        self.provider = provider
        self._current_run_id: Optional[str] = None

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        ctx = get_current_metrics_context()
        if ctx:
            run_id_str = str(run_id)
            self._current_run_id = run_id_str
            set_current_llm_call_id(run_id_str)
            ctx.track_start(run_id_str, self.model_key, self.provider)

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        ctx = get_current_metrics_context()
        if ctx:
            tokens_in = 0
            tokens_out = 0
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)
            ctx.track_end(str(run_id), tokens_in, tokens_out)
            set_current_llm_call_id(None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        ctx = get_current_metrics_context()
        if ctx:
            ctx.track_error(
                str(run_id),
                error_type=type(error).__name__,
                error_message=str(error)[:200],
            )
            set_current_llm_call_id(None)
