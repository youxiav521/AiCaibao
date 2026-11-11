"""
ADE Jobs 异步任务

流程：
  1. 从 MinIO 下载原始文件
  2. 调用 MinerUParsingService → items + mineru_result
  3. 上传 converted.json、images、images_index.json 到 MinIO
  4. 通过 TaskManager 写回 SUCCESS/FAILED，result 中仅保存引用路径
  5. 更新 JobsCacheStore 状态
  6. 可选：回调 callback_url 通知调用方

注意：
- worker 进程独立于 FastAPI，Redis/MinIO 均通过 base 服务单例访问
- 与 src/ade/parse/tasks/ade_parse_tasks.py 的差异：
  * 不转换为 ParseResponse，直接保存 MinerU items（保持 converted.json 原结构）
  * 单独保存图片到 results/{job_id}/images/，并生成 images_index.json
"""
import asyncio
import mimetypes
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from src.ade.jobs.service.jobs_cache_store import JobsCacheStore
from src.applications.parsing.service.parse_minerU_service import MinerUParsingService
from src.base.storage_service import storage_service
from src.base.task_manager import TaskManager, TaskStatus
from src.utils import json_utils

_ADE_JOBS_BUCKET = os.getenv("ADE_JOBS_BUCKET", "ade-jobs")

# 图片转存并发上限（每张图 = 1 次下载 + 1 次上传 MinIO）
_IMAGE_TRANSFER_CONCURRENCY = int(os.getenv("ADE_JOBS_IMAGE_CONCURRENCY", "8"))


async def process_ade_jobs(
    ctx: dict,
    file_path: str,
    filename: str,
    external_file_id: Optional[str],
    file_hash: Optional[str],
    backend: str,
    ang: str,
    method: str,
    formula_enable: bool,
    table_enable: bool,
    callback_url: Optional[str],
) -> Dict[str, Any]:
    """
    ADE Jobs 异步任务入口。

    Args:
        ctx: ARQ 运行时上下文，至少包含 job_id / job_try
        file_path: MinIO 中的文件路径（格式 "{bucket}/{object_name}"）
        filename: 原始文件名
        external_file_id: Parsing 侧 file_id，原样回传 metadata
        file_hash: 文件 sha256，用于幂等去重与缓存更新
        backend / ang / method / formula_enable / table_enable: 解析参数（保留传给 MinerU）
        callback_url: 可选的回调 URL

    Returns:
        dict: {
            "converted_path": "{bucket}/{object_name}",
            "images_index_path": "{bucket}/{object_name}",
            "metadata": { external_file_id, filename, page_count, duration_ms, version }
        }
    """
    job_id: str = ctx.get("job_id", "")
    job_try: int = int(ctx.get("job_try", 1))

    logger.info(
        "AdeJobs任务开始 | job_id={} file_path={} filename={} job_try={}",
        job_id, file_path, filename, job_try,
    )

    tmp_file: Optional[Path] = None
    try:
        if not await TaskManager.try_start(job_id, attempts=job_try):
            logger.info("AdeJobs任务已取消或不可领取 | job_id={}", job_id)
            return {"cancelled": True, "job_id": job_id}

        # 1. 从 MinIO 下载原始文件
        if "/" not in file_path:
            raise ValueError(
                f"非法的 file_path，应为 '{{bucket}}/{{object_name}}'：{file_path}"
            )
        bucket, object_name = file_path.split("/", 1)
        file_bytes = await storage_service.download(bucket, object_name)

        # 2. 写临时文件（finally 清理）
        tmp_file = await _write_temp_file(file_bytes, filename, job_id)

        # 3. 调用 MinerU 解析得到 items + mineru_result
        start_ms = time.time()
        items, mineru_result = await _parse_to_items(
            str(tmp_file),
            filename=filename,
            job_id=job_id,
        )
        duration_ms = int((time.time() - start_ms) * 1000)

        if not isinstance(items, list):
            items = []

        page_count = _estimate_page_count(items)
        logger.info(
            "AdeJobs任务解析完成 | job_id={} items={} pages={} duration_ms={}",
            job_id, len(items), page_count, duration_ms,
        )

        # 4+5. 并发执行：上传 converted.json（保持 items 原结构，不裁剪字段）
        # 与图片转存（两者相互独立，均为 MinIO 网络 IO）
        converted_object = f"results/{job_id}/converted.json"
        converted_bytes = json_utils.dumps_bytes(items, ensure_ascii=False)
        _, images_list = await asyncio.gather(
            storage_service.upload(
                _ADE_JOBS_BUCKET, converted_object, converted_bytes, "application/json"
            ),
            _upload_images_from_mineru_result(mineru_result, items, job_id),
        )
        converted_path = f"{_ADE_JOBS_BUCKET}/{converted_object}"

        # 6. 上传 images_index.json
        images_index_object = f"results/{job_id}/images_index.json"
        images_index_bytes = json_utils.dumps_bytes(
            {"count": len(images_list), "list": images_list},
            ensure_ascii=False,
        )
        await storage_service.upload(
            _ADE_JOBS_BUCKET, images_index_object, images_index_bytes, "application/json"
        )
        images_index_path = f"{_ADE_JOBS_BUCKET}/{images_index_object}"

        # 7. 组装 result（仅引用路径，不入 Redis 大对象）
        version = os.getenv("ADE_VERSION", "unknown")
        metadata = {
            "external_file_id": external_file_id,
            "filename": filename,
            "page_count": page_count,
            "duration_ms": duration_ms,
            "version": version,
        }
        result: Dict[str, Any] = {
            "converted_path": converted_path,
            "images_index_path": images_index_path,
            "metadata": metadata,
        }

        # 8. 标记成功
        await TaskManager.update(
            job_id, TaskStatus.SUCCESS, result=result, attempts=job_try
        )

        # 9. 更新缓存状态
        if file_hash:
            await JobsCacheStore.update_status(
                file_hash,
                status="completed",
                converted_path=converted_path,
                images_index_path=images_index_path,
                page_count=page_count,
                duration_ms=duration_ms,
            )

        logger.info(
            "AdeJobs任务完成 | job_id={} converted_path={} images={} duration_ms={}",
            job_id, converted_path, len(images_list), duration_ms,
        )

        # 回调失败不应影响已经成功的任务状态
        await _notify_callback(callback_url, job_id, "completed", result)
        return result

    except Exception as exc:
        logger.exception("AdeJobs任务失败 | job_id={} err={}", job_id, str(exc))
        await TaskManager.update(
            job_id, TaskStatus.FAILED, error=str(exc), attempts=job_try
        )
        if file_hash:
            await JobsCacheStore.update_status(file_hash, status="failed")
        await _notify_callback(
            callback_url, job_id, "failed", {"error": str(exc)}
        )
        raise

    finally:
        if tmp_file is not None:
            _cleanup_temp_file(tmp_file)


async def _write_temp_file(file_bytes: bytes, filename: str, job_id: str) -> Path:
    """写入临时文件，供 MinerU 解析使用。"""
    import asyncio

    def _do() -> Path:
        ext = Path(filename).suffix
        fd, path = tempfile.mkstemp(prefix=f"ade_jobs_{job_id}_", suffix=ext)
        try:
            with os.fdopen(fd, "wb") as buffer:
                buffer.write(file_bytes)
        except Exception:
            os.close(fd)
            raise
        return Path(path)

    return await asyncio.to_thread(_do)


def _cleanup_temp_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.warning("AdeJobs | 清理临时文件失败 | path={} err={}", path, e)


async def _parse_to_items(
    file_path: str,
    filename: str,
    job_id: str,
) -> Tuple[List[Dict[str, Any]], Any]:
    """
    调用 MinerUParsingService 解析得到 items 与 mineru_result。

    复用 src/ade/parse/service/parse_service.py 中的调用方式，但不做 ParseResponse 转换。
    """
    import uuid

    # 复用 MinerU 的 file_id 体系（用于 mineru-output 桶中的中间产物）
    file_id = uuid.uuid4().hex
    service = MinerUParsingService()
    result = await service.parse_to_items_and_save_async(
        file_path,
        file_id=file_id,
        classify_tables=True,
        original_filename=filename,
    )
    items = result.get("items") or []
    mineru_result = result.get("mineru_result")
    return items, mineru_result


def _estimate_page_count(items: List[Dict[str, Any]]) -> int:
    """从 items 中估算页数（取 page 字段的最大值）。"""
    max_page = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        page = item.get("page")
        if isinstance(page, (int, float)) and page > max_page:
            max_page = int(page)
    return max_page


async def _upload_images_from_mineru_result(
    mineru_result: Any,
    items: List[Dict[str, Any]],
    job_id: str,
) -> List[Dict[str, Any]]:
    """
    从 items 中收集 img_path，从 mineru_output 桶下载图片并转存到 ade-jobs/results/{job_id}/images/。

    返回 images_index.list 结构：[{name, path, size}, ...]
    """
    # 1. 收集 items 中所有 img_path
    img_paths: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        img_path = item.get("img_path")
        if not img_path or not isinstance(img_path, str):
            continue
        if img_path in seen:
            continue
        seen.add(img_path)
        img_paths.append(img_path)

    if not img_paths:
        return []

    # 2. 反查 file_id（mineru_result 中 results 的 key 即为 file_id）
    file_id = _extract_file_id_from_mineru_result(mineru_result)
    if not file_id:
        logger.warning(
            "AdeJobs | 无法从 mineru_result 反查 file_id，跳过图片转存 | job_id={}",
            job_id,
        )
        return []

    # 3. 并发下载并转存每张图片（单图失败仅告警跳过，不影响其他图片）
    from src.applications.parsing.utils.mineru_storage import (
        MINERU_OUTPUT_BUCKET,
        build_mineru_image_object_name,
    )

    # 限制并发数，避免瞬时打满 MinIO 连接池 / to_thread 线程池
    semaphore = asyncio.Semaphore(_IMAGE_TRANSFER_CONCURRENCY)

    async def _transfer_one(img_path: str) -> Optional[Dict[str, Any]]:
        # img_path 形如 "images/xxx.jpg"，name 取 basename
        name = Path(img_path).name or img_path
        source_object = build_mineru_image_object_name(file_id, img_path)
        async with semaphore:
            try:
                img_bytes = await storage_service.download(
                    MINERU_OUTPUT_BUCKET, source_object
                )
            except Exception as exc:
                logger.warning(
                    "AdeJobs | 下载 MinerU 图片失败 | job_id={} img_path={} err={}",
                    job_id, img_path, exc,
                )
                return None

            target_object = f"results/{job_id}/images/{name}"
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            try:
                await storage_service.upload(
                    _ADE_JOBS_BUCKET, target_object, img_bytes, content_type
                )
            except Exception as exc:
                logger.warning(
                    "AdeJobs | 转存图片失败 | job_id={} name={} err={}",
                    job_id, name, exc,
                )
                return None

        return {"name": name, "path": img_path, "size": len(img_bytes)}

    # gather 保持输入顺序，images_index.list 顺序与串行版本一致
    results = await asyncio.gather(*(_transfer_one(p) for p in img_paths))
    return [item for item in results if item is not None]


def _extract_file_id_from_mineru_result(mineru_result: Any) -> Optional[str]:
    """从 mineru_result.results 的第一个 key 反查 file_id。"""
    if not isinstance(mineru_result, dict):
        return None
    results = mineru_result.get("results")
    if not isinstance(results, dict) or not results:
        return None
    return next(iter(results.keys()))


async def _notify_callback(
    url: Optional[str], job_id: str, status: str, data: Dict[str, Any]
) -> None:
    """
    向 callback_url POST JSON 通知任务结束。

    - url 为空或非法（非 http/https）时，直接跳过，不报错
    - 网络/HTTP 失败仅记录告警，不向上抛，避免影响主任务状态

    回调 payload 格式（与 dev_plan 第 13 节待确认事项 6 对齐）：
        {"job_id": "...", "status": "completed|failed", "result": {...}}
    """
    if not url:
        return

    url_stripped = url.strip()
    if not url_stripped.lower().startswith(("http://", "https://")):
        logger.warning(
            "AdeJobs 回调已跳过（非法 URL） | job_id={} status={} url={!r}",
            job_id, status, url,
        )
        return

    payload = {"job_id": job_id, "status": status, "result": data}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url_stripped, json=payload)
            resp.raise_for_status()
        logger.info(
            "AdeJobs 回调成功 | job_id={} status={} url={}",
            job_id, status, url_stripped,
        )
    except Exception as exc:
        logger.warning(
            "AdeJobs 回调失败（已忽略） | job_id={} status={} url={} err={}",
            job_id, status, url_stripped, exc,
        )
