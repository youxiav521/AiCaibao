"""
ADE Parse 异步任务

流程：
  1. 从 MinIO 下载原始文件
  2. 调用 AdeParseService → MinerU 解析 → 转 ParseResponse
  3. 将 ParseResponse JSON 上传到 MinIO
  4. 通过 TaskManager 写回 SUCCESS/FAILED，result 中仅保存引用路径（避免 Redis 大对象）
  5. 可选：回调 callback_url 通知调用方

注意：worker 进程独立于 FastAPI，Redis/MinIO 均通过 base 服务单例访问。
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from src.ade.parse.service.ade_parse_cache_store import (
    AdeParseCacheRecord,
    AdeParseCacheStore,
)
from src.ade.parse.service.parse_service import AdeParseService
from src.base.storage_service import storage_service
from src.base.task_manager import TaskManager, TaskStatus

_ADE_RESULT_BUCKET = os.getenv("ADE_PARSE_BUCKET", "ade-parse")


async def process_ade_parse(
    ctx: dict,
    file_path: str,
    filename: str,
    file_id: Optional[str] = None,
    file_hash: Optional[str] = None,
    split_by_page: bool = False,
    callback_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ADE Parse 异步任务入口。

    Args:
        ctx: ARQ 运行时上下文，至少包含 job_id / job_try
        file_path: MinIO 中的文件路径（格式 "{bucket}/{object_name}"）
        filename: 原始文件名（用于 metadata 和 worker 临时文件后缀）
        split_by_page: 是否按页分段输出 splits
        callback_url: 可选的回调 URL；任务完成/失败时 POST JSON

    Returns:
        dict: {
            "parse_response_path": "{bucket}/{object_name}",
            "filename": str,
            "page_count": int,
            "duration_ms": int,
        }
    """
    job_id: str = ctx.get("job_id", "")
    job_try: int = int(ctx.get("job_try", 1))

    logger.info(
        "AdeParse任务开始 | job_id={} file_path={} filename={} split_by_page={} job_try={}",
        job_id, file_path, filename, split_by_page, job_try,
    )

    try:
        if not await TaskManager.try_start(job_id, attempts=job_try):
            logger.info("AdeParse任务已取消或不可领取 | job_id={}", job_id)
            return {"cancelled": True, "job_id": job_id}

        if "/" not in file_path:
            raise ValueError(f"非法的 file_path，应为 '{{bucket}}/{{object_name}}'：{file_path}")
        bucket, object_name = file_path.split("/", 1)
        file_bytes = await storage_service.download(bucket, object_name)

        service = AdeParseService()
        parse_response = await service.parse_document(
            file_bytes=file_bytes,
            filename=filename,
            split_by_page=split_by_page,
            file_id=file_id,
            job_id=job_id,
        )

        date_prefix = datetime.now().strftime("%Y_%m_%d")
        result_object = f"results/{date_prefix}/{job_id}.json"
        result_json_bytes = parse_response.model_dump_json(by_alias=True).encode("utf-8")
        await storage_service.upload(
            _ADE_RESULT_BUCKET,
            result_object,
            result_json_bytes,
            "application/json",
        )

        result: Dict[str, Any] = {
            "parse_response_path": f"{_ADE_RESULT_BUCKET}/{result_object}",
            "filename": filename,
            "page_count": parse_response.metadata.page_count,
            "duration_ms": parse_response.metadata.duration_ms,
        }

        if file_hash:
            now_iso = AdeParseCacheStore.now_iso()
            await AdeParseCacheStore.save(
                AdeParseCacheRecord(
                    file_hash=file_hash,
                    split_by_page=split_by_page,
                    parse_response_path=result["parse_response_path"],
                    file_id=file_id,
                    filename=filename,
                    page_count=parse_response.metadata.page_count,
                    duration_ms=parse_response.metadata.duration_ms,
                    created_at=now_iso,
                    updated_at=now_iso,
                )
            )

        await TaskManager.update(
            job_id, TaskStatus.SUCCESS, result=result, attempts=job_try
        )
        logger.info(
            "AdeParse任务完成 | job_id={} result_path={} pages={} duration_ms={}",
            job_id,
            result["parse_response_path"],
            result["page_count"],
            result["duration_ms"],
        )

        # 回调失败不应影响已经成功的任务状态，这里吞掉所有异常。
        await _notify_callback(callback_url, job_id, "completed", result)

        return result

    except Exception as exc:
        logger.exception("AdeParse任务失败 | job_id={} err={}", job_id, str(exc))
        await TaskManager.update(
            job_id, TaskStatus.FAILED, error=str(exc), attempts=job_try
        )
        await _notify_callback(
            callback_url, job_id, "failed", {"error": str(exc)}
        )
        raise


async def _notify_callback(
    url: Optional[str], job_id: str, status: str, data: Dict[str, Any]
) -> None:
    """
    向 callback_url POST JSON 通知任务结束。

    - url 为空或非法（非 http/https）时，直接跳过，不报错。
    - 网络/HTTP 失败仅记录告警，不向上抛，避免影响主任务状态。
    """
    if not url:
        return

    url_stripped = url.strip()
    if not url_stripped.lower().startswith(("http://", "https://")):
        logger.warning(
            "AdeParse 回调已跳过（非法 URL） | job_id={} status={} url={!r}",
            job_id,
            status,
            url,
        )
        return

    payload = {"jobId": job_id, "status": status, "data": data}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url_stripped, json=payload)
            resp.raise_for_status()
        logger.info(
            "AdeParse 回调成功 | job_id={} status={} url={}",
            job_id,
            status,
            url_stripped,
        )
    except Exception as exc:
        logger.warning(
            "AdeParse 回调失败（已忽略） | job_id={} status={} url={} err={}",
            job_id,
            status,
            url_stripped,
            exc,
        )
