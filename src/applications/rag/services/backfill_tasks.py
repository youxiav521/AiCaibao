from __future__ import annotations

from typing import Any

from loguru import logger

from src.base.task_manager import TaskManager, TaskStatus

from .backfill_service import backfill_legacy_rag_stores


async def process_rag_legacy_backfill(
    ctx: dict,
    old_parent_collection: str,
    old_child_collection: str | None = None,
    old_grandchild_collection: str | None = None,
) -> dict[str, Any]:
    job_id = ctx.get("job_id")
    job_try = int(ctx.get("job_try", 1))
    progress_callback = None
    if job_id:
        async def progress_callback(result: dict[str, Any]) -> None:
            await TaskManager.update(job_id, TaskStatus.RUNNING, result=result, attempts=job_try)

    try:
        if job_id:
            await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)
        logger.info(
            "RAG legacy 回填任务开始 | job_id={} old_parent={} old_child={} old_grandchild={}",
            job_id,
            old_parent_collection,
            old_child_collection,
            old_grandchild_collection,
        )
        result = await backfill_legacy_rag_stores(
            old_parent_collection=old_parent_collection,
            old_child_collection=old_child_collection,
            old_grandchild_collection=old_grandchild_collection,
            progress_callback=progress_callback,
        )
        if job_id:
            await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
        logger.info(
            "RAG legacy 回填任务完成 | job_id={} scanned={} written={} failed={}",
            job_id,
            result.get("scanned"),
            result.get("written"),
            result.get("failed"),
        )
        return result
    except Exception as exc:
        logger.exception("RAG legacy 回填任务失败 | job_id={}", job_id)
        if job_id:
            await TaskManager.update(job_id, TaskStatus.FAILED, error=str(exc), attempts=job_try)
        return {"error": str(exc)}
