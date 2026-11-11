"""
ARQ Worker 入口 — 独立于 FastAPI 启动

启动方式:
    uv run arq src.worker.WorkerSettings

环境变量:
    WORKER_MAX_JOBS   每个 worker 进程处理多少个任务后退出（默认 100）
                      配合 Docker restart: unless-stopped 实现定期重启，防内存泄露
"""

import faulthandler
import asyncio
import os
import signal
import sys
from functools import wraps

faulthandler.enable()

from dotenv import load_dotenv

load_dotenv()

from src.base.logger_factory import initLogger
from src.config import config

initLogger(config)

from loguru import logger
from arq.worker import func
from arq.connections import RedisSettings

from src.middleware.otel_langfuse import setup_otel_langfuse
from src.base.redis_client import RedisClient

from src.ade.parse.tasks.ade_parse_tasks import process_ade_parse
from src.ade.extract.tasks.ade_extract_tasks import process_ade_extract
from src.ade.jobs.tasks.ade_jobs_tasks import process_ade_jobs

from src.applications.rag.services.summary_index_tasks import (
    process_summary_index_summaries,
    process_image_summaries,
    process_table_enrichment,
)
from src.applications.rag.services.backfill_tasks import process_rag_legacy_backfill
from src.applications.rag.services.vectorstore import (
    close_default_stores,
    ensure_default_stores_initialized,
)
from src.applications.content_generate.tasks import process_content_generate
from src.ade.knowledge_graph.tasks.kg_tasks import create_knowledge_graph, update_knowledge_graph
from src.agents.template.tasks import process_template_parse
from src.utils.model_enhancements.llm_metrics import metrics_context_session


def build_redis_settings() -> RedisSettings:
    if RedisClient.is_sentinel_mode():
        return RedisSettings(
            host=RedisClient.parse_sentinel_hosts(),
            password=config.REDIS_PASSWORD or None,
            database=config.REDIS_DB,
            sentinel=True,
            sentinel_master=config.REDIS_SENTINEL_MASTER_NAME,
        )
    return RedisSettings(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD or None,
        database=config.REDIS_DB,
    )


def with_llm_metrics(task_name: str):
    """
    Worker 任务装饰器：给所有任务添加 LLM 调用统计

    使用方式:
        @with_llm_metrics("task_name")
        async def task_name(ctx, ...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(ctx, *args, **kwargs):
            job_id = ctx.get("job_id")
            async with metrics_context_session(auto_print=True, compact=True) as metrics:
                try:
                    result = await func(ctx, *args, **kwargs)
                    return result
                except asyncio.CancelledError:
                    logger.info("Worker 任务被取消，将由 ARQ 重试 | task_name={} job_id={}", task_name, job_id)
                    raise
                finally:
                    if metrics.llm_api_calls > 0:
                        logger.info(
                            "【LLM调用统计-任务结束】 task_name={} job_id={} "
                            "llm_api_calls={} llm_scheduled_units={} total_queue_wait_ms={:.2f}",
                            task_name, job_id, metrics.llm_api_calls,
                            metrics.llm_scheduled_units, metrics.total_queue_wait_ms,
                        )
        return wrapper
    return decorator

_MAX_JOBS = int(os.getenv("WORKER_MAX_JOBS", "100"))
_jobs_processed = 0


async def on_startup(ctx: dict) -> None:
    logger.info("Worker 启动 | max_jobs={}", _MAX_JOBS)

    # 初始化 OTel / Langfuse 追踪（与 main.py 对齐，使 Worker 内的 LLM 调用也能被追踪）
    setup_otel_langfuse(
        service_name="yusys-ai",
        enabled=bool(getattr(getattr(config, "monitor", None), "enabled", False)),
    )

    # 初始化Redis（两个客户端：str模式 + bytes模式），验证连通性，失败则阻止Worker启动
    logger.info("开始初始化Redis连接")
    await RedisClient.get_client(decode_responses=True)
    await RedisClient.get_client(decode_responses=False)
    logger.info("Redis 初始化完成")

    logger.info("开始初始化 RAG 向量集合")
    await ensure_default_stores_initialized()
    logger.info("RAG 向量集合初始化完成")


async def after_job_end(ctx: dict) -> None:
    """每个任务完成后调用，达到上限后优雅退出触发重启"""
    global _jobs_processed
    _jobs_processed += 1
    remaining = _MAX_JOBS - _jobs_processed
    logger.info(
        "任务统计 | 已处理={} 上限={} 剩余={}",
        _jobs_processed,
        _MAX_JOBS,
        remaining,
    )
    if _jobs_processed >= _MAX_JOBS:
        logger.info("Worker 达到任务上限 {}，准备优雅退出等待重启", _MAX_JOBS)
        os.kill(os.getpid(), signal.SIGTERM)


async def on_shutdown(ctx: dict) -> None:
    """Worker 关闭时清理 Redis 连接"""
    logger.info("Worker 关闭，清理Redis连接")
    await close_default_stores()
    await RedisClient.close()
    logger.info("Redis 连接已清理，Worker 关闭完成")


class WorkerSettings:
    functions = [
        with_llm_metrics("process_ade_parse")(process_ade_parse),
        with_llm_metrics("process_ade_extract")(process_ade_extract),
        with_llm_metrics("process_ade_jobs")(process_ade_jobs),
        with_llm_metrics("process_summary_index_summaries")(process_summary_index_summaries),
        with_llm_metrics("process_image_summaries")(process_image_summaries),
        with_llm_metrics("process_table_enrichment")(process_table_enrichment),
        with_llm_metrics("process_rag_legacy_backfill")(process_rag_legacy_backfill),
        with_llm_metrics("process_content_generate")(process_content_generate),
        with_llm_metrics("process_template_parse")(process_template_parse),
        with_llm_metrics("create_knowledge_graph")(create_knowledge_graph),
        with_llm_metrics("update_knowledge_graph")(update_knowledge_graph),
    ]
    redis_settings = build_redis_settings()
    on_startup = on_startup
    on_shutdown = on_shutdown
    after_job_end = after_job_end
    max_jobs = 4  # 最大并发任务数
    job_timeout = 72000  # 单任务最长运行时间（秒）
    keep_result = 360000  # 任务结果在 Redis 中的保留时间（秒）
    max_tries = 3  # 全局默认最大重试次数


if __name__ == "__main__":
    from arq.worker import run_worker

    # 本地调试入口：IDEA 中直接 Debug 运行 src/worker.py 即可启动 ARQ Worker。
    # 与命令行 `uv run arq src.worker.WorkerSettings` 等价。
    run_worker(WorkerSettings)
