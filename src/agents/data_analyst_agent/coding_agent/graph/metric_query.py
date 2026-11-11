"""
Langfuse 指标查询和统计分析模块
默认查询计算智能体的相关错误指标

用于从 Langfuse 平台查询 Agent 执行的性能指标，
并进行统计分析、计算和格式化输出。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import statistics
from langfuse import get_client
from loguru import logger


@dataclass
class MetricEvent:
    """单个指标事件"""
    trace_id: str
    name: str
    total_duration_ms: Optional[float] = None
    model_call_count: Optional[int] = None
    code_error_count: Optional[int] = None
    code_errors: List[str] = field(default_factory=list)
    model_call_times_ms: List[float] = field(default_factory=list)
    window_count: Optional[int] = None
    error: Optional[str] = None
    exception_type: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class MetricStatistics:
    """指标统计结果"""
    total_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    
    # 耗时统计
    total_duration_stats: Optional[Dict[str, float]] = None
    avg_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    
    # 模型调用统计
    total_model_calls: int = 0
    avg_model_calls_per_task: float = 0.0
    max_model_calls: int = 0
    
    # 代码错误统计
    total_code_errors: int = 0
    avg_code_errors_per_task: float = 0.0

    # pass@1: 一次代码生成就成功执行（code_error_count = 0）的任务比例
    pass_count: int = 0
    pass_at_1: float = 0.0
    
    # 窗口数量统计
    total_windows: int = 0
    avg_windows_per_task: float = 0.0
    
    # 模型调用时间统计
    model_call_time_stats: Optional[Dict[str, float]] = None
    
    # 错误详情
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    exception_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class CodeErrorRow:
    """代码错误表格行"""
    trace_id: str
    timestamp: str
    code_error: str
    duration_ms: float
    model_calls: int


class MetricQuery:
    """Langfuse 指标事件查询器（单例模式）"""
    
    DEFAULT_OBSERVATION_NAME = "coding_agent_full_analysis"
    _instance: Optional["MetricQuery"] = None
    
    def __new__(cls, observation_name: Optional[str] = None, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, observation_name: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True
        self.langfuse = get_client()
        self.observation_name = observation_name or self.DEFAULT_OBSERVATION_NAME
    
    def _parse_langfuse_value(self, value):
        """
        解析 Langfuse 序列化的值
        
        Langfuse 存储复杂类型时会序列化为特殊格式:
        - {"__type__": "list", "items": [...], "size": N}
        - {"__type__": "dict", "data": {...}}
        - {"__type__": "str", "value": "..."}
        - {"__type__": "set", "items": [...], "size": N}
        """
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, dict)) and "__type__" not in value:
            return value
        
        if isinstance(value, dict):
            type_ = value.get("__type__")
            if type_ == "list":
                return value.get("items", [])
            elif type_ == "dict":
                return value.get("data", {})
            elif type_ in ("str", "int", "float", "bool"):
                return value.get("value")
            elif type_ == "set":
                return value.get("items", [])
            elif type_ == "none":
                return None
        
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        return value
    
    def _fetch_page(
        self,
        limit: int = 100,
        page: int = 1,
        trace_id: Optional[str] = None,
        level: Optional[str] = None,
        environment: Optional[str] = None,
        from_start_time: Optional[datetime] = None,
        to_start_time: Optional[datetime] = None,
    ) -> List[MetricEvent]:
        """内部方法：查询单页指标事件"""
        params = {
            "limit": limit,
            "type": "EVENT",
            "name": self.observation_name,
            "page": page,
        }
        
        if trace_id:
            params["trace_id"] = trace_id
        if level:
            params["level"] = level
        if environment:
            params["environment"] = environment
        if from_start_time:
            params["from_start_time"] = from_start_time
        if to_start_time:
            params["to_start_time"] = to_start_time
        
        response = self.langfuse.api.observations.get_many(**params)
        events = []
        for item in response.data:
            item_output = item.output if item.output else {}
            item_input = item.input if item.input else {}
            metrics = item_output.get("metrics", {}) if isinstance(item_output, dict) else {}
            
            raw_code_errors = item_input.get("code_errors", []) if isinstance(item_input, dict) else []
            parsed_code_errors = self._parse_langfuse_value(raw_code_errors)
            
            raw_call_times = metrics.get("model_call_times_ms", [])
            parsed_call_times = self._parse_langfuse_value(raw_call_times)
            if not isinstance(parsed_call_times, list):
                parsed_call_times = []
            
            def _to_number(val):
                if val is None:
                    return None
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            event = MetricEvent(
                trace_id=item.trace_id or "",
                name=item.name or "",
                total_duration_ms=_to_number(metrics.get("total_duration_ms")),
                model_call_count=_to_number(metrics.get("model_call_count")),
                code_error_count=_to_number(metrics.get("code_error_count")),
                code_errors=parsed_code_errors,
                model_call_times_ms=[_to_number(t) for t in parsed_call_times if _to_number(t) is not None],
                window_count=_to_number(metrics.get("window_count")),
                error=metrics.get("error"),
                exception_type=metrics.get("exception_type"),
                timestamp=item.start_time,
            )
            events.append(event)
        
        return events
    
    def _compute_stats(self, events: List[MetricEvent]) -> MetricStatistics:
        """内部方法：计算指标统计"""
        if not events:
            return MetricStatistics()
        
        stats = MetricStatistics()
        stats.total_count = len(events)
        
        durations = [e.total_duration_ms for e in events if e.total_duration_ms is not None]
        if durations:
            stats.total_duration_stats = {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "stdev": statistics.stdev(durations) if len(durations) > 1 else 0,
                "min": min(durations),
                "max": max(durations),
            }
            stats.avg_duration_ms = stats.total_duration_stats["mean"]
            stats.min_duration_ms = stats.total_duration_stats["min"]
            stats.max_duration_ms = stats.total_duration_stats["max"]
        
        model_calls = [e.model_call_count for e in events if e.model_call_count is not None]
        if model_calls:
            stats.total_model_calls = sum(model_calls)
            stats.avg_model_calls_per_task = statistics.mean(model_calls)
            stats.max_model_calls = max(model_calls)
        
        code_errors = [e.code_error_count for e in events if e.code_error_count is not None]
        if code_errors:
            stats.total_code_errors = sum(code_errors)
            stats.avg_code_errors_per_task = statistics.mean(code_errors)

        # 计算 pass@1: code_error_count = 0 表示一次就成功执行的任务
        valid_for_pass = [e for e in events if e.code_error_count is not None]
        if valid_for_pass:
            stats.pass_count = sum(1 for e in valid_for_pass if e.code_error_count == 0)
            stats.pass_at_1 = stats.pass_count / len(valid_for_pass)
        
        windows = [e.window_count for e in events if e.window_count is not None]
        if windows:
            stats.total_windows = sum(windows)
            stats.avg_windows_per_task = statistics.mean(windows)
        
        all_call_times = []
        for e in events:
            for t in e.model_call_times_ms:
                if t is not None:
                    all_call_times.append(float(t))
        
        if all_call_times:
            stats.model_call_time_stats = {
                "mean": statistics.mean(all_call_times),
                "median": statistics.median(all_call_times),
                "stdev": statistics.stdev(all_call_times) if len(all_call_times) > 1 else 0,
                "min": min(all_call_times),
                "max": max(all_call_times),
            }
        
        error_events = [e for e in events if e.error or e.exception_type]
        stats.error_count = len(error_events)
        stats.error_rate = stats.error_count / stats.total_count if stats.total_count > 0 else 0
        
        for e in error_events:
            stats.error_details.append({
                "trace_id": e.trace_id,
                "error": e.error,
                "exception_type": e.exception_type,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            })
            if e.exception_type:
                stats.exception_types[e.exception_type] = stats.exception_types.get(e.exception_type, 0) + 1
        
        return stats
    
    def _format_duration(self, ms: float) -> str:
        """内部方法：格式化耗时"""
        if ms >= 1000:
            return f"{ms/1000:.2f}s"
        return f"{ms:.0f}ms"

    def query_all(
        self,
        max_pages: int = 10,
        days: Optional[int] = None,
        **kwargs
    ) -> List[MetricEvent]:
        """批量查询所有符合条件的指标事件（自动翻页）"""
        all_events = []
        
        if days:
            kwargs["from_start_time"] = datetime.now() - timedelta(days=days)
        
        for page in range(1, max_pages + 1):
            events = self._fetch_page(page=page, **kwargs)
            all_events.extend(events)
            if len(events) < 100:
                break
        
        return all_events

    def format_statistics(self, stats: MetricStatistics) -> str:
        """
        格式化统计结果为可读文本
        
        Args:
            stats: 统计结果
            
        Returns:
            str: 格式化的报告文本
        """
        lines = [
            "=" * 60,
            " 性能指标统计报告",
            "=" * 60,
            "",
        ]
        
        lines.append("【基础统计】")
        lines.append(f"  总任务数        {stats.total_count}")
        lines.append(f"  错误任务数      {stats.error_count}")
        lines.append(f"  错误率          {stats.error_rate:.2%}")
        lines.append("")
        
        lines.append("【耗时统计】")
        if stats.total_duration_stats:
            lines.append(f"  平均            {self._format_duration(stats.avg_duration_ms)}")
            lines.append(f"  最短            {self._format_duration(stats.min_duration_ms)}")
            lines.append(f"  最长            {self._format_duration(stats.max_duration_ms)}")
            lines.append(f"  中位            {self._format_duration(stats.total_duration_stats['median'])}")
            lines.append(f"  标准差          {stats.total_duration_stats['stdev']:.0f}ms")
        else:
            lines.append("  无数据")
        lines.append("")
        
        lines.append("【模型调用统计】")
        if stats.total_model_calls > 0:
            lines.append(f"  总调用次数      {stats.total_model_calls}")
            lines.append(f"  平均每次任务    {stats.avg_model_calls_per_task:.1f} 次")
            lines.append(f"  单任务最大      {stats.max_model_calls} 次")
        else:
            lines.append("  无数据")
        lines.append("")
        
        if stats.model_call_time_stats:
            lines.append("【单次模型调用耗时】")
            lines.append(f"  平均            {stats.model_call_time_stats['mean']:.0f}ms")
            lines.append(f"  最长            {stats.model_call_time_stats['max']:.0f}ms")
            lines.append("")
        
        lines.append("【代码错误统计】")
        lines.append(f"  总错误数        {stats.total_code_errors}")
        if stats.avg_code_errors_per_task > 0:
            lines.append(f"  平均每次任务    {stats.avg_code_errors_per_task:.1f} 次")

        lines.append("【代码执行成功率】")
        lines.append(f"  一次成功数      {stats.pass_count}")
        lines.append(f"  pass@1          {stats.pass_at_1:.2%}")
        lines.append("")
        
        if stats.total_windows > 0:
            lines.append("【窗口数量统计】")
            lines.append(f"  总窗口数        {stats.total_windows}")
            lines.append(f"  平均每次任务    {stats.avg_windows_per_task:.1f} 个")
            lines.append("")
        
        if stats.exception_types:
            lines.append("【异常类型分布】")
            for exc_type, count in sorted(stats.exception_types.items(), key=lambda x: -x[1]):
                lines.append(f"  {exc_type}        {count} 次")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)

    def format_code_errors_table(self, errors: List[CodeErrorRow]) -> str:
        """
        格式化代码错误为表格
        
        Args:
            errors: 代码错误列表
            
        Returns:
            str: 表格形式文本
        """
        if not errors:
            return "无代码执行错误"
        
        lines = [
            "=" * 100,
            f" 代码执行错误列表 (共 {len(errors)} 条)",
            "=" * 100,
            f"{'#':<4} {'时间':<16} {'耗时':<10} {'调用次数':<8} {'错误信息':<60}",
            "-" * 100,
        ]
        
        for i, err in enumerate(errors, 1):
            code_error = err.code_error.replace("\n", " ")[:58]
            if len(err.code_error) > 58:
                code_error += "..."
            lines.append(
                f"{i:<4} {err.timestamp:<16} "
                f"{self._format_duration(err.duration_ms):<10} "
                f"{err.model_calls:<8} "
                f"{code_error:<60}"
            )
        
        lines.append("=" * 100)
        return "\n".join(lines)

    def query_metrics(
        self,
        days: Optional[int] = 7,
        limit: int = 100,
        max_pages: int = 10,
        include_statistics: bool = True,
        include_code_errors: bool = True,
    ) -> Dict[str, Any]:
        """
        统一查询指标（单次查询，根据参数决定返回内容）

        Args:
            days: 查询最近 N 天的数据
            limit: 每页数量限制
            max_pages: 最大翻页数
            include_statistics: 是否包含统计结果
            include_code_errors: 是否包含代码错误

        Returns:
            Dict: 包含 statistics 和/或 code_errors 的结果
        """
        logger.info("请求Langfuse | days={}, limit={}, include_statistics={}, include_code_errors={}",
                    days, limit, include_statistics, include_code_errors)

        events = self.query_all(max_pages=max_pages, days=days, limit=limit)
        logger.info("查询到 events 数量: {}", len(events))

        result = {}

        if include_statistics:
            stats = self._compute_stats(events)
            result["statistics"] = stats

        if include_code_errors:
            code_errors = []
            for e in events:
                for error in e.code_errors:
                    code_errors.append(CodeErrorRow(
                        trace_id=e.trace_id,
                        timestamp=e.timestamp.strftime("%Y-%m-%d %H:%M") if e.timestamp else "N/A",
                        code_error=error,
                        duration_ms=e.total_duration_ms or 0,
                        model_calls=e.model_call_count or 0,
                    ))
            result["code_errors"] = code_errors

        return result
