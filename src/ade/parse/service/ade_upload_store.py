"""
ADE 上传文件元信息存储（Redis）

- 用于 `/openapi/ade/upload-file` 与 `/openapi/ade/submit-parse-job` 之间
  跨进程共享 fileId → minio 对象信息的映射。
- 仅保存元信息（bucket/object_name/filename/...）；真正的文件内容落在 MinIO。
- TTL 默认 7 天（ADE_UPLOAD_TTL_SECONDS 可覆盖），过期后需重新上传。
"""

import os
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger
from pydantic import BaseModel, Field

from src.base.redis_client import RedisClient

_UPLOAD_KEY_PREFIX = "ade_upload:"
_UPLOAD_TTL_SECONDS = int(os.getenv("ADE_UPLOAD_TTL_SECONDS", str(7 * 24 * 3600)))


class AdeUploadRecord(BaseModel):
    """上传到 MinIO 的文件元信息记录。"""

    file_id: str = Field(..., description="文件 ID（upload-file 返回的 fileId）")
    filename: str = Field(..., description="原始文件名")
    content_type: str = Field("application/octet-stream", description="MIME 类型")
    size: int = Field(0, ge=0, description="文件字节数")
    bucket: str = Field(..., description="MinIO 桶名")
    object_name: str = Field(..., description="MinIO 对象路径（不含桶名）")
    file_hash: Optional[str] = Field(None, description="文件内容 sha256（小写十六进制）")
    created_at: str = Field(..., description="创建时间（ISO8601, UTC）")

    @property
    def file_path(self) -> str:
        """`{bucket}/{object_name}` 形式的路径，直接可传给 worker。"""
        return f"{self.bucket}/{self.object_name}"


class AdeUploadStore:
    """Redis 后端的 ADE 上传元信息存储（懒加载单例客户端）。"""

    _redis: Optional[aioredis.Redis] = None

    @classmethod
    async def _get_redis(cls) -> aioredis.Redis:
        if cls._redis is None:
            cls._redis = await RedisClient.get_client(decode_responses=True)
        return cls._redis

    @classmethod
    def _key(cls, file_id: str) -> str:
        return f"{_UPLOAD_KEY_PREFIX}{file_id}"

    @classmethod
    async def save(cls, record: AdeUploadRecord) -> None:
        r = await cls._get_redis()
        await r.setex(
            cls._key(record.file_id),
            _UPLOAD_TTL_SECONDS,
            record.model_dump_json(),
        )
        logger.info(
            "AdeUploadStore | 保存 | file_id={} bucket={} object={} ttl={}s",
            record.file_id,
            record.bucket,
            record.object_name,
            _UPLOAD_TTL_SECONDS,
        )

    @classmethod
    async def get(cls, file_id: str) -> Optional[AdeUploadRecord]:
        r = await cls._get_redis()
        data = await r.get(cls._key(file_id))
        if data is None:
            return None
        try:
            return AdeUploadRecord.model_validate_json(data)
        except Exception as exc:
            logger.error(
                "AdeUploadStore | 解析失败 | file_id={} err={}", file_id, exc
            )
            return None

    @classmethod
    def now_iso(cls) -> str:
        return datetime.now(timezone.utc).isoformat()
