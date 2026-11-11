from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Dict, List
from pathlib import Path

from .vectorstore import VectorStoresService
from ..config.rag_settings import RAG_SETTINGS
from loguru import logger
from src.base.distributed_lock import DistributedLock, LockNotAcquiredError
from src.base.storage_service import storage_service
from ..utils.validators import validate_file_id
from .index import default_index, txt_index, json_index
from .summary_job_state import clear_current_summary_job_id, set_current_summary_job_id
from .parent_summary_cache import refresh_parent_summary_cache
from src.base.task_manager import TaskManager
from src.utils import json_utils

_MINERU_OUTPUT_BUCKET = os.getenv("MINERU_OUTPUT_BUCKET", "mineru-output")

_POST_INDEX_VISIBILITY_TIMEOUT_S = 5.0
_POST_INDEX_VISIBILITY_POLL_INTERVAL_S = 0.1


async def _wait_until_index_visible(file_id: str) -> bool:
    """等待新索引数据对 exists 检查可见，缩小后端读写可见性竞争窗口。"""
    await VectorStoresService.refresh_indices()
    deadline = time.monotonic() + _POST_INDEX_VISIBILITY_TIMEOUT_S
    while True:
        if await VectorStoresService.exists_file_id(file_id):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(_POST_INDEX_VISIBILITY_POLL_INTERVAL_S, remaining))


def _detect_plain_text_strategy(
    items: List[Dict[str, Any]], file_name: str | None
) -> str | None:
    if file_name:
        suffix = Path(file_name).suffix.lower()
        if suffix == ".txt":
            return "txt"
        if suffix in {".json", ".jsonl"}:
            return "json"
    for it in items or []:
        if not isinstance(it, dict):
            continue
        raw = it.get("source_type") or it.get("file_type") or it.get("sourceType") or ""
        st = str(raw).strip().lower()
        if st in {"txt", "text"}:
            return "txt"
        if st in {"json", "jsonl"}:
            return "json"
    return None


async def _rollback_index_write(file_id: str, *, reason: str) -> None:
    try:
        await clear_current_summary_job_id(file_id)
        deleted = await VectorStoresService.delete_by_file_id(file_id)
        if not deleted:
            logger.warning(
                "索引失败后回滚未完全成功: file_id={}, reason={}", file_id, reason
            )
    except Exception as exc:
        logger.warning("索引失败后回滚异常: file_id={}, reason={}, error={}", file_id, reason, exc)


async def _cleanup_children_snapshots_from_minio(file_id: str) -> None:
    """删除指定 file_id 的所有子切片快照，避免 MinIO 存储泄漏。"""
    try:
        prefix = f"{file_id}/children/"
        objects = await storage_service.list_objects(_MINERU_OUTPUT_BUCKET, prefix)
        for object_name in objects:
            try:
                await storage_service.delete(_MINERU_OUTPUT_BUCKET, object_name)
            except Exception:
                logger.warning("删除子切片快照失败: object={}", object_name)
    except Exception:
        logger.exception("清理子切片快照失败: file_id={}", file_id)


async def _upload_children_snapshot_to_minio(
    file_id: str,
    summary_job_id: str,
    child_documents: list,
) -> None:
    """将子切片快照上传到 MinIO，供异步 summary job 按需下载。

    summary_job_id 用于版本隔离：不同索引批次的切片快照互不覆盖。
    """
    if not child_documents:
        return
    try:
        data = json_utils.dumps_bytes(
            [doc.model_dump(mode="json", exclude_none=True) for doc in child_documents],
            ensure_ascii=False,
        )
        object_name = f"{file_id}/children/{summary_job_id}.json"
        await storage_service.upload(_MINERU_OUTPUT_BUCKET, object_name, data, json_utils.JSON_MIME_TYPE)
        logger.info(
            "子切片快照已上传到MinIO: file_id={} bucket={} object={} count={}",
            file_id, _MINERU_OUTPUT_BUCKET, object_name, len(child_documents),
        )
    except Exception:
        logger.exception("上传子切片快照到MinIO失败: file_id={}", file_id)


async def build_index(
    doc_parse_items: List[Dict[str, Any]],
    file_id: str,
    file_name: str,
    folder_id: str | None,
) -> str:
    """在已持有 file_id 占位锁的前提下执行正式索引流程。"""
    if not doc_parse_items:
        raise ValueError("mineru_json 不能为空")

    summary_job_id = uuid.uuid4().hex
    strategy = _detect_plain_text_strategy(doc_parse_items, file_name)

    try:
        if strategy == "txt":
            result = await txt_index.build(file_id, doc_parse_items, folder_id, file_name)
        elif strategy == "json":
            result = await json_index.build(file_id, doc_parse_items, folder_id, file_name)
        else:
            result = await default_index.build(
                file_id=file_id,
                mineru_json=doc_parse_items,
                folder_id=folder_id,
                file_name=file_name,
            )
    except Exception as exc:
        cause = exc.__cause__ or exc.__context__
        logger.exception(
            "索引构建失败 | file_id={}, file_name={}, strategy={}, "
            "error_type={}, error_msg={}, cause_type={}, cause_msg={}",
            file_id,
            file_name,
            strategy,
            type(exc).__name__,
            exc,
            type(cause).__name__ if cause else "None",
            cause if cause else "None",
        )
        await _rollback_index_write(file_id, reason=f"{file_name} 处理失败: {exc}")
        raise RuntimeError(f"{file_name} 处理失败: {exc}") from exc

    (parent_documents, child_documents, grandchild_documents) = result

    if strategy is None:
        try:
            await refresh_parent_summary_cache(file_id, parent_documents)
        except Exception:
            logger.exception("刷新父切片摘要缓存失败: file_id={}", file_id)

        try:
            await set_current_summary_job_id(file_id, summary_job_id)
            await _upload_children_snapshot_to_minio(
                file_id, summary_job_id, child_documents,
            )
            common_kwargs = {
                "file_id": file_id,
                "summary_job_id": summary_job_id,
                "file_name": file_name or "",
                "folder_id": folder_id or "",
            }
            job_meta = {"file_id": file_id, "file_name": file_name or ""}
            jobs = [
                TaskManager.submit_with_job_id(
                    job_id=uuid.uuid4().hex,
                    task_name="process_summary_index_summaries",
                    task_type="summary_index",
                    kwargs=dict(common_kwargs),
                    metadata=job_meta,
                ),
                TaskManager.submit_with_job_id(
                    job_id=uuid.uuid4().hex,
                    task_name="process_image_summaries",
                    task_type="image_summary",
                    kwargs=dict(common_kwargs),
                    metadata=job_meta,
                ),
            ]
            if RAG_SETTINGS.table_semantic_enrichment_enabled:
                jobs.append(
                    TaskManager.submit_with_job_id(
                        job_id=uuid.uuid4().hex,
                        task_name="process_table_enrichment",
                        task_type="table_enrichment",
                        kwargs=dict(common_kwargs),
                        metadata=job_meta,
                    )
                )
            await asyncio.gather(*jobs)
        except Exception:
            logger.exception("提交SummaryIndex摘要任务失败: file_id={}", file_id)

    if not await _wait_until_index_visible(file_id):
        logger.warning(
            "file_id={} 索引写入完成后，在 {} 秒内仍未对 exists 检查可见；"
            "后端若存在更长的 refresh 延迟，仍可能影响后续重复上传判断",
            file_id,
            _POST_INDEX_VISIBILITY_TIMEOUT_S,
        )

    return file_id


async def index_document_items(
    doc_parse_items: List[Dict[str, Any]],
    file_name: str,
    folder_id: str | None,
) -> str:
    """将 converted items 直接执行索引入库。

    Args:
        doc_parse_items: converted_output 生成的 items 列表。
        file_name: 原始文件名。
        folder_id: 文档文件夹 ID。

    Returns:
        实际入库使用的 file_id。

    Raises:
        ValueError: 参数非法或 file_id 非法。
        FileExistsError: 向量库中已存在相同 file_id。
        RuntimeError: 入库失败（已尽量回滚）。
    """
    normalized_file_name = file_name.strip()
    if not normalized_file_name:
        raise ValueError("fileName 不能为空")

    data = doc_parse_items or []

    file_id = next(
        (
            str(item.get("file_id"))
            for item in data
            if isinstance(item, dict) and item.get("file_id") is not None
        ),
        None,
    )
    if not file_id:
        file_id = os.path.splitext(os.path.basename(normalized_file_name))[0]

    try:
        file_id = validate_file_id(file_id)
    except ValueError as exc:
        raise ValueError(f"{normalized_file_name} file_id 非法: {exc}") from exc

    try:
        async with DistributedLock.acquire(
            f"rag:index:{file_id}",
            blocking=True,
            blocking_timeout=RAG_SETTINGS.index_wait_timeout_s,
        ):
            if await VectorStoresService.exists_file_id(file_id):
                return file_id
            return await build_index(
                data,
                file_id,
                normalized_file_name,
                folder_id,
            )
    except LockNotAcquiredError:
        raise FileExistsError(
            f"相同 file_id 正在索引中：{file_id}，等待超时"
        ) from None


async def delete_document_by_file_id(
    file_id: str,
) -> bool:
    normalized_file_id = validate_file_id(file_id)

    try:
        async with DistributedLock.acquire(
            f"rag:index:{normalized_file_id}",
            blocking=True,
            blocking_timeout=RAG_SETTINGS.delete_wait_timeout_s,
        ):
            await clear_current_summary_job_id(normalized_file_id)
            await _cleanup_children_snapshots_from_minio(normalized_file_id)
            deleted = await VectorStoresService.delete_by_file_id(normalized_file_id)
            return deleted
    except TimeoutError:
        return False
    except LockNotAcquiredError:
        return False



