"""
指标监控工具类
用来提交性能指标相关的event到langfuse
提供统一的监控接口，整合指标监控和事件记录功能
"""

import time
import functools
import inspect
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
from loguru import logger
from src.utils.langfuse_tracing import (
    log_milestone_event,
    log_performance_event,
    get_global_tracer
)


class MetricTracker:
    """指标追踪器，用于记录和更新指标"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.perf_counter()
        self.event_info: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
    def set_event_info(self, key: str, value: Any):
        """设置事件信息"""
        self.event_info[key] = value
    
    def get_event_info(self) -> Dict[str, Any]:
        """获取事件信息"""
        return self.event_info
    
    def set_metric(self, name: str, value: Any):
        """设置指标值"""
        self.metrics[name] = value
    
    def get_duration_ms(self) -> float:
        """获取执行时间（毫秒）"""
        return (time.perf_counter() - self.start_time) * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return self.metrics


class Monitoring:
    """监控类，提供统一的监控接口"""
    
    def __init__(self):
        self.tracer = get_global_tracer()
    
    def is_enabled(self) -> bool:
        """检查监控是否启用"""
        return self.tracer and self.tracer.is_enabled()
    

    
    @contextmanager
    def track_metric_context(self, name: str):
        """追踪指标的上下文管理器"""
        if not self.is_enabled():
            yield None
            return
        
        tracker = MetricTracker(name)
        try:
            yield tracker
        finally:
            log_performance_event(
                name,
                duration_ms=tracker.get_duration_ms(),
                event_info=tracker.get_event_info(),
                metrics=tracker.get_metrics()
            )
    
    def log_milestone(self, event_name: str, data: Optional[Dict[str, Any]] = None, level: str = "DEFAULT"):
        """记录里程碑事件"""
        log_milestone_event(event_name, data, level)
    
    def log_performance(self, event_name: str, duration_ms: float, event_info: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, Any]] = None):
        """记录性能事件"""
        log_performance_event(event_name, duration_ms, event_info=event_info, metrics=metrics)
    
    def log_api_request(self, path: str, method: str, status_code: int, duration_ms: float):
        """记录API请求事件"""
        self.log_performance(
            "api_request",
            duration_ms=duration_ms,
            event_info={
                "path": path,
                "method": method,
                "status_code": status_code
            }
        )
    
    def log_external_call(self, service: str, operation: str, duration_ms: float, success: bool):
        """记录外部服务调用事件"""
        self.log_performance(
            "external_call",
            duration_ms=duration_ms,
            event_info={
                "service": service,
                "operation": operation,
                "success": success
            }
        )


# 全局监控实例
_global_monitoring: Optional[Monitoring] = None


def get_monitoring() -> Monitoring:
    """获取全局监控实例"""
    global _global_monitoring
    if _global_monitoring is None:
        _global_monitoring = Monitoring()
    return _global_monitoring


# 使用示例
if __name__ == "__main__":
    monitoring = get_monitoring()

    # 示例：追踪一个数据处理任务
    with monitoring.track_metric_context("data_processing") as tracker:
        # 实际业务逻辑
        result = [i * 2 for i in range(100)]
        if tracker:
            # 模拟设置指标
            tracker.set_metric("record_count", 1000)
            tracker.set_metric("result", result)

