"""
ADE Jobs 服务层

职责：
  1. submit_job：幂等检查 → 落 MinIO → 写缓存 → 入队 ARQ
  2. get_status：从 TaskManager 读取任务状态并映射为 ADE 对外状态
  3. get_data：从 MinIO 加载 converted.json + images_index.json，组装 content_list/images/metadata
  4. get_image：从 MinIO 下载图片二进制

注意：
- 大对象（converted.json、images）一律落 MinIO，TaskManager.result 中只存引用路径
- 与 src/ade/parse 的 AdeParseService 区分：本服务面向 docs/new.md 规范的对外接口
"""
import asyncio
import mimetypes
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.ade.jobs.schema.jobs_schema import (
    ImageItem,
    ImagesPayload,
    JobDataData,
    JobMetadata,
    JobStatus,
    JobStatusData,
    ContentListPayload,
)
from src.ade.jobs.service.jobs_cache_store import JobsCacheRecord, JobsCacheStore
from src.base.storage_service import (
    StorageError,
    StorageObjectNotFoundError,
    storage_service,
)
from src.base.task_manager import TaskManager, TaskRecord, TaskStatus
from src.utils import json_utils


# ── 状态映射 ──────────────────────────────────────────────────────────────────
# TaskManager 内部状态 → ADE 对外状态（与 src/ade/parse/router/parse_router.py 一致）
_TASK_STATUS_TO_ADE: Dict[TaskStatus, JobStatus] = {
    TaskStatus.PENDING: JobStatus.pending,
    TaskStatus.RUNNING: JobStatus.processing,
    TaskStatus.SUCCESS: JobStatus.completed,
    TaskStatus.FAILED: JobStatus.failed,
    TaskStatus.CANCELLED: JobStatus.cancelled,
}


def _progress_of(status: TaskStatus) -> float:
    """根据 TaskStatus 估算进度。"""
    if status == TaskStatus.SUCCESS:
        return 1.0
    if status == TaskStatus.RUNNING:
        return 0.5
    return 0.0


def _to_iso(ts: Optional[str]) -> str:
    """规范化时间戳为 ISO8601 字符串；None 时返回当前 UTC 时间。"""
    if ts:
        return ts
    return datetime.now(timezone.utc).isoformat()


class AdeJobsService:
    """ADE Jobs 业务编排服务。"""

    # ── 接口 1：提交解析任务 ───────────────────────────────────────────────────

    async def submit_job(
        self,
        file_stream,
        filename: Optional[str],
        content_type: Optional[str],
        file_hash: str,
        external_file_id: Optional[str],
        backend: str,
        ang: str,
        method: str,
        formula_enable: bool,
        table_enable: bool,
        priority: int,
        callback_url: Optional[str],
        bucket: str,
        task_name: str,
        task_type: str,
    ) -> str:
        """
        提交解析任务，返回 job_id。

        流程：
          1. 幂等检查：file_hash 命中且为 completed
             - 若原 job_id 在 TaskManager 中仍为 SUCCESS → 直接返回原 job_id（不覆盖）
             - 若原 job_id 已过期/丢失 → 用 create_finished_with_job_id 重建（保留原 created_at）
          2. 生成 job_id，流式落 MinIO
          3. 写缓存索引（status=pending）
          4. 入队 ARQ 任务
          5. 返回 job_id
        """
        from uuid import uuid4

        # 1. 幂等检查
        cached = await JobsCacheStore.get(file_hash)
        if cached is not None and cached.status == "completed":
            existing = await TaskManager.get(cached.job_id)
            if existing is not None and existing.status == TaskStatus.SUCCESS:
                # 原记录仍存活，直接返回原 job_id，不覆盖 created_at 等时间字段
                logger.info(
                    "AdeJobsService.submit_job | 命中缓存且原任务存活 | job_id={} hash={}",
                    cached.job_id, file_hash,
                )
                return cached.job_id

            # 原记录已过期/丢失，用缓存信息重建已完成记录（不入队）
            # result 中保留原始 created_at，避免对外暴露"刚创建即 completed"的异常时间线
            await TaskManager.create_finished_with_job_id(
                job_id=cached.job_id,
                task_name=task_name,
                task_type=task_type,
                result={
                    "converted_path": cached.converted_path,
                    "images_index_path": cached.images_index_path,
                    "metadata": {
                        "external_file_id": cached.external_file_id,
                        "filename": cached.filename,
                        "page_count": cached.page_count,
                        "duration_ms": cached.duration_ms,
                        "version": _get_ade_version(),
                    },
                },
                metadata={
                    "file_hash": file_hash,
                    "external_file_id": external_file_id,
                    "filename": cached.filename,
                    "cache_hit": True,
                },
                request={
                    "file_hash": file_hash,
                    "external_file_id": external_file_id,
                    "callback_url": callback_url,
                },
            )
            logger.info(
                "AdeJobsService.submit_job | 命中缓存重建完成任务 | job_id={} hash={}",
                cached.job_id, file_hash,
            )
            return cached.job_id

        # 2. 生成 job_id + 流式落 MinIO
        job_id = str(uuid4())
        resolved_filename = filename or f"{job_id}.bin"
        resolved_content_type = content_type or "application/octet-stream"
        size = _get_content_length(file_stream) or 0

        date_prefix = datetime.now().strftime("%Y_%m_%d")
        object_name = f"uploads/{date_prefix}/{job_id}_{resolved_filename}"

        try:
            await storage_service.upload_stream(
                bucket, object_name, file_stream, content_type=resolved_content_type,
            )
        except Exception as exc:
            logger.exception(
                "AdeJobsService.submit_job | 上传 MinIO 失败 | job_id={}", job_id
            )
            raise RuntimeError(f"上传文件失败: {exc}") from exc

        file_path = f"{bucket}/{object_name}"

        # 3. 写缓存索引
        now_iso = JobsCacheStore.now_iso()
        await JobsCacheStore.save(
            JobsCacheRecord(
                file_hash=file_hash,
                job_id=job_id,
                status="pending",
                external_file_id=external_file_id,
                filename=resolved_filename,
                created_at=now_iso,
                updated_at=now_iso,
            )
        )

        # 4. 入队
        kwargs = {
            "file_path": file_path,
            "filename": resolved_filename,
            "external_file_id": external_file_id,
            "file_hash": file_hash,
            "backend": backend,
            "ang": ang,
            "method": method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "callback_url": callback_url,
        }
        metadata = {
            "file_hash": file_hash,
            "external_file_id": external_file_id,
            "filename": resolved_filename,
            "size": size,
            "priority": priority,
        }

        try:
            await TaskManager.submit_with_job_id(
                job_id,
                task_name,
                task_type=task_type,
                kwargs=kwargs,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception(
                "AdeJobsService.submit_job | 入队失败 | job_id={}", job_id
            )
            raise RuntimeError(f"任务入队失败: {exc}") from exc

        logger.info(
            "AdeJobsService.submit_job | 已入队 | job_id={} filename={} size={}B",
            job_id,
            resolved_filename,
            size,
        )
        return job_id

    # ── 接口 2：查询任务状态 ───────────────────────────────────────────────────

    async def get_status(self, job_id: str) -> Optional[JobStatusData]:
        """返回任务状态数据；job_id 不存在返回 None。"""
        record = await TaskManager.get(job_id)
        if record is None:
            return None
        return self._build_status_data(record)

    async def cancel_pending(self, job_id: str) -> Optional[JobStatus]:
        """取消尚未被 ADE Worker 领取的解析任务。"""
        record = await TaskManager.get(job_id)
        if record is None or record.task_type != "ade_jobs":
            return None
        task_status = await TaskManager.cancel_pending(job_id)
        if task_status is None:
            return None
        if task_status == TaskStatus.CANCELLED:
            file_hash = str(record.metadata.get("file_hash") or "")
            if file_hash:
                await JobsCacheStore.update_status(file_hash, "cancelled")
        return _TASK_STATUS_TO_ADE.get(task_status, JobStatus.pending)

    # ── 接口 3：拉取解析结果 ───────────────────────────────────────────────────

    async def get_data(self, job_id: str) -> Tuple[Optional[JobDataData], Optional[JobStatus]]:
        """
        返回 (data, status)：
          - job_id 不存在：(None, None)
          - 未完成：(None, JobStatus)
          - 已完成：(JobDataData, JobStatus.completed)

        异常处理：
          - converted.json 下载失败 / JSON 损坏 / 内容非列表 → 抛 RuntimeError，
            由 router 层转 500。这是为了让 Parsing 侧能感知到结果损坏（而非拿到
            空 content_list 误以为解析成功），符合 docs/new.md 第 4 节
            "content 为空视为解析失败"的契约精神。
        """
        record = await TaskManager.get(job_id)
        if record is None:
            return None, None

        ade_status = _TASK_STATUS_TO_ADE.get(record.status, JobStatus.pending)
        if record.status != TaskStatus.SUCCESS:
            return None, ade_status

        result = record.result if isinstance(record.result, dict) else {}
        converted_path = result.get("converted_path")
        images_index_path = result.get("images_index_path")
        metadata_dict = result.get("metadata") or {}

        if not converted_path:
            logger.warning(
                "AdeJobsService.get_data | 缺少 converted_path | job_id={}", job_id
            )
            raise RuntimeError(f"任务结果损坏：缺少 converted_path (job_id={job_id})")

        # 并发加载 converted.json（保持字段原样，不裁剪）与 images_index.json
        content_items, images_payload = await asyncio.gather(
            self._load_converted_items(converted_path),
            self._load_images_payload(images_index_path),
        )

        metadata = JobMetadata(
            external_file_id=metadata_dict.get("external_file_id"),
            filename=metadata_dict.get("filename") or "",
            page_count=int(metadata_dict.get("page_count", 0) or 0),
            duration_ms=int(metadata_dict.get("duration_ms", 0) or 0),
            version=metadata_dict.get("version") or _get_ade_version(),
        )

        data = JobDataData(
            job_id=job_id,
            content_list=ContentListPayload(content=content_items),
            images=images_payload,
            metadata=metadata,
        )
        return data, ade_status

    # ── 接口 4：拉取图片 ───────────────────────────────────────────────────────

    async def get_image(
        self, job_id: str, image_name: str, bucket: str
    ) -> Tuple[Optional[bytes], Optional[str], Optional[JobStatus]]:
        """
        返回 (image_bytes, content_type, status)：
          - job_id 不存在：(None, None, None)
          - 未完成：(None, None, JobStatus)
          - 已完成但图片不存在：(None, None, JobStatus.completed)
          - 已完成且图片存在：(bytes, content_type, JobStatus.completed)
        """
        record = await TaskManager.get(job_id)
        if record is None:
            return None, None, None

        ade_status = _TASK_STATUS_TO_ADE.get(record.status, JobStatus.pending)
        if record.status != TaskStatus.SUCCESS:
            return None, None, ade_status

        object_name = f"results/{job_id}/images/{image_name}"
        try:
            data = await storage_service.download(bucket, object_name)
        except StorageObjectNotFoundError:
            logger.warning(
                "AdeJobsService.get_image | 图片不存在 | job_id={} image={}",
                job_id,
                image_name,
            )
            return None, None, ade_status
        except StorageError:
            logger.exception(
                "AdeJobsService.get_image | 下载失败 | job_id={} image={}",
                job_id,
                image_name,
            )
            return None, None, ade_status

        content_type = _guess_image_content_type(image_name)
        return data, content_type, ade_status

    # ── 内部工具 ───────────────────────────────────────────────────────────────

    def _build_status_data(self, record: TaskRecord) -> JobStatusData:
        ade_status = _TASK_STATUS_TO_ADE.get(record.status, JobStatus.pending)
        error_message = record.error if record.status == TaskStatus.FAILED else None
        return JobStatusData(
            job_id=record.job_id,
            status=ade_status,
            error_message=error_message,
            progress=_progress_of(record.status),
            created_at=_to_iso(record.created_at),
            updated_at=_to_iso(record.finished_at or record.started_at or record.created_at),
        )

    async def _load_converted_items(self, converted_path: str) -> List[Dict[str, Any]]:
        """从 MinIO 加载 converted.json，返回 item 列表（字段不裁剪）。

        下载失败 / JSON 损坏 / 内容非列表时抛 RuntimeError，由上层转 500，
        避免对外返回空 content_list 造成 Parsing 侧误判。
        """
        if "/" not in converted_path:
            raise RuntimeError(f"非法的 converted_path: {converted_path}")
        bucket, object_name = converted_path.split("/", 1)
        try:
            raw = await storage_service.download(bucket, object_name)
        except StorageObjectNotFoundError:
            raise RuntimeError(f"converted.json 不存在: {converted_path}")
        except StorageError as exc:
            raise RuntimeError(f"下载 converted.json 失败: {converted_path}: {exc}")

        try:
            # orjson 直接解析 bytes，避免大 payload 的一次全量 decode 拷贝
            items = json_utils.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"converted.json JSON 解析失败: {converted_path}: {exc}")

        if not isinstance(items, list):
            raise RuntimeError(
                f"converted.json 内容非列表: {converted_path} (type={type(items).__name__})"
            )
        return items

    async def _load_images_payload(
        self, images_index_path: Optional[str]
    ) -> ImagesPayload:
        """从 MinIO 加载 images_index.json，返回 ImagesPayload。

        下载/解析失败时返回空 payload（仅图片索引缺失，不阻塞 converted.json 返回）。
        """
        if not images_index_path or "/" not in images_index_path:
            return ImagesPayload(count=0, list=[])

        bucket, object_name = images_index_path.split("/", 1)
        try:
            raw = await storage_service.download(bucket, object_name)
        except StorageObjectNotFoundError:
            logger.warning(
                "AdeJobsService._load_images_payload | 对象不存在 | path={}",
                images_index_path,
            )
            return ImagesPayload(count=0, list=[])
        except StorageError:
            logger.exception(
                "AdeJobsService._load_images_payload | 下载失败 | path={}",
                images_index_path,
            )
            return ImagesPayload(count=0, list=[])

        try:
            payload = json_utils.loads(raw)
        except Exception:
            logger.exception(
                "AdeJobsService._load_images_payload | JSON 解析失败 | path={}",
                images_index_path,
            )
            return ImagesPayload(count=0, list=[])

        if not isinstance(payload, dict):
            return ImagesPayload(count=0, list=[])

        raw_list = payload.get("list") or []
        if not isinstance(raw_list, list):
            return ImagesPayload(count=0, list=[])

        items: List[ImageItem] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            # name 是接口 4 下载路径 {image_name} 所需标识，必须存在；
            # 不用 path 兜底，因为 path 含 "images/" 前缀会导致下载路径拼错。
            name = item.get("name")
            if not name:
                continue
            items.append(
                ImageItem(
                    name=str(name),
                    path=str(item.get("path") or name),
                    size=int(item.get("size", 0) or 0),
                )
            )

        count = len(items) if items else int(payload.get("count", 0) or 0)
        return ImagesPayload(count=count, list=items)


def _get_ade_version() -> str:
    return os.getenv("ADE_VERSION", "unknown")


def _guess_image_content_type(image_name: str) -> str:
    """根据扩展名推断图片 MIME 类型。"""
    ct, _ = mimetypes.guess_type(image_name)
    if ct and ct.startswith("image/"):
        return ct
    lower = image_name.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".bmp"):
        return "image/bmp"
    if lower.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def _get_content_length(stream) -> int:
    """尝试从文件流获取大小；无法获取时返回 0。"""
    try:
        cur = stream.tell()
        stream.seek(0, 2)  # seek end
        size = stream.tell()
        stream.seek(cur)   # 还原位置
        return max(0, size)
    except Exception:
        return 0
