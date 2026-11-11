import sys
import logging
import asyncio
import time
import os
import psutil
from functools import wraps
from loguru import logger
from src.config import Config

_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS}|{level: <5}|"
    "{name}:{function}:{line}| PID:{process}| TID:{thread}| {message}"
)
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)


# ── stdlib logging → loguru 拦截器 ───────────────────────────────────────────

class InterceptHandler(logging.Handler):
    """将 stdlib logging 的所有输出转发到 loguru，统一格式与 sink。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _install_stdlib_bridge() -> None:
    """让所有 stdlib logging 调用透明地流向 loguru（OTel、Langfuse、第三方库）。"""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    # docket 是 fastmcp 内部的任务调度器，每几秒打一次 scheduler 心跳 DEBUG 日志
    # 这些属于正常轮询噪音，不需要暴露到应用日志中
    logging.getLogger("docket").setLevel(logging.INFO)
    # Elasticsearch 客户端默认会输出所有请求的 INFO 日志，过于详细
    # 提高到 WARNING 级别，只在出现问题时输出日志
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)


def is_running_in_gunicorn() -> bool:
    """判断当前进程是否运行在gunicorn环境下。"""
    # 检查环境变量
    if os.getenv("GUNICORN_RUNNING", "false").lower() == "true":
        return True
    
    # 检查父进程名称
    try:
        parent_process = psutil.Process(os.getppid())
        if "gunicorn" in parent_process.name().lower():
            return True
        
        # 检查进程启动参数
        for arg in parent_process.cmdline():
            if "gunicorn" in arg.lower():
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    return False


# ── 主初始化 ──────────────────────────────────────────────────────────────────

def initLogger(appConfig: Config) -> None:
    """应用启动时调用一次，配置所有日志 sink。"""
    logger.remove()
    level = os.getenv("LOGURU_LEVEL", appConfig.logger.level)
    
    # 多进程环境下自动开启enqueue，保证日志写入安全
    enqueue = is_running_in_gunicorn()
    # 也可以通过环境变量强制覆盖
    if "LOGURU_ENQUEUE" in os.environ:
        enqueue = os.getenv("LOGURU_ENQUEUE", "false").lower() == "true"

    # 1. 控制台（过滤 context_text 大文本，避免刷屏）
    logger.add(
        sink=sys.stdout,
        format=_CONSOLE_FORMAT,
        level=os.getenv("LOGURU_CONSOLE_LEVEL", level),
        colorize=True,
        backtrace=True,  # 保留基础栈追踪，便于定位问题
        diagnose=False,  # 关闭详细变量诊断，大幅减少控制台输出
        filter=lambda r: r["extra"].get("request_tag") != "context_text",
        enqueue=enqueue,
    )

    # 2. 全量日志文件（JSON 结构化，便于日志系统查询）
    logger.add(
        os.getenv("LOGURU_FILE_PATH", "logs/app.log"),
        format=_LOG_FORMAT,
        level=os.getenv("LOGURU_FILE_LEVEL", level),
        rotation="00:00",
        retention="5 days",
        encoding="utf-8",
        serialize=True,
        backtrace=True,
        diagnose=True,
        enqueue=enqueue,
    )

    # 3. ERROR+ 专用文件（用于告警/监控，包含 CRITICAL）
    logger.add(
        os.getenv("LOGURU_ERROR_FILE_PATH", "logs/error.log"),
        format=_LOG_FORMAT,
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
        filter=lambda r: r["level"].no >= logger.level("ERROR").no,
        enqueue=enqueue,
    )

    # 4. 性能日志（通过 logger.bind(performance=True) 写入）
    logger.add(
        "logs/performance.log",
        format=_LOG_FORMAT,
        level="INFO",
        rotation="20 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        filter=lambda r: r["extra"].get("performance") is True,
        enqueue=enqueue,
    )

    # 6. 接管 stdlib logging（统一收口 OTel、Langfuse、第三方库日志）
    _install_stdlib_bridge()

    # 7. 未捕获异常兜底
    def _handle_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        if issubclass(exc_type, asyncio.CancelledError):
            logger.warning("异步任务被取消")
            return
        logger.opt(exception=(exc_type, exc_value, exc_tb)).critical("未捕获的异常")

    sys.excepthook = _handle_uncaught

    logger.info("日志系统初始化完成 | level={}", level)


def addContextTextHandler() -> None:
    """context_text 专用处理器（大文本单独归档，不污染控制台）。"""
    # 使用和主初始化相同的enqueue策略
    enqueue = is_running_in_gunicorn()
    if "LOGURU_ENQUEUE" in os.environ:
        enqueue = os.getenv("LOGURU_ENQUEUE", "false").lower() == "true"
        
    logger.add(
        "logs/context_text.log",
        format=_LOG_FORMAT,
        level="INFO",
        rotation="50 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        filter=lambda r: r["extra"].get("request_tag") == "context_text",
        enqueue=enqueue,
    )


# ── 工具：LogContext ───────────────────────────────────────────────────────────

class LogContext:
    """
    结构化日志上下文管理器，自动将 key-value 附加到块内所有日志记录。

    用法：
        with LogContext(user_id=123, request_id="abc") as log:
            log.info("处理中")
    """

    def __init__(self, **kwargs):
        self._ctx = kwargs
        self._bound = logger.bind(**kwargs)

    def __enter__(self):
        return self._bound

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._bound.exception("上下文异常 | ctx={}", self._ctx)
        return False  # 不吞掉异常，继续向上传播


# ── 工具：log_performance 装饰器 ──────────────────────────────────────────────

def log_performance(func_name: str | None = None):
    """
    性能计时装饰器，同步和异步函数均支持。
    日志写入 logs/performance.log。

    用法：
        @log_performance()
        async def my_service():
            ...
    """
    def decorator(func):
        name = func_name or func.__qualname__

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    logger.bind(performance=True).info(
                        "async {} 完成 | elapsed={:.4f}s", name, time.perf_counter() - t0
                    )
                    return result
                except Exception:
                    logger.bind(performance=True).exception(
                        "async {} 失败 | elapsed={:.4f}s", name, time.perf_counter() - t0
                    )
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    logger.bind(performance=True).info(
                        "{} 完成 | elapsed={:.4f}s", name, time.perf_counter() - t0
                    )
                    return result
                except Exception:
                    logger.bind(performance=True).exception(
                        "{} 失败 | elapsed={:.4f}s", name, time.perf_counter() - t0
                    )
                    raise
            return sync_wrapper

    return decorator
