"""
ADE Jobs 结果去重缓存（Redis）

目标：
- 基于文件内容哈希（sha256）建立稳定映射
- 命中且为成功结果时直接复用已有 job_id 与产物路径，避免重复解析
- 仅缓存"结果引用路径"，不在 Redis 存储大对象

与 src/ade/parse/service/ade_parse_cache_store.py 的差异：
- 不区分 split_by_page 维度（新规范无此参数）
- 缓存记录包含 status 字段，可标记 pending / completed / failed
- TTL 默认 24h（与 docs/new.md 第 4 节"结果保留 ≥24h"对齐）
"""
import os
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger
from pydantic import BaseModel, Field

from src.base.redis_client import RedisClient

_CACHE_KEY_PREFIX = "ade_jobs_cache:"
_CACHE_TTL_SECONDS = int(os.getenv("ADE_JOBS_CACHE_TTL_SECONDS", str(24 * 3600)))


class JobsCacheRecord(BaseModel):
    """文件哈希去重缓存记录。"""

    file_hash: str = Field(..., description="文件内容 sha256（小写十六进制）")
    job_id: str = Field(..., description="关联的任务 ID")
    status: str = Field(
        "pending",
        description="任务状态：pending / processing / completed / failed",
    )
    converted_path: Optional[str] = Field(
        None, description="converted.json 路径，格式 {bucket}/{object_name}"
    )
    images_index_path: Optional[str] = Field(
        None, description="图片索引 JSON 路径，格式 {bucket}/{object_name}"
    )
    external_file_id: Optional[str] = Field(None, description="透传的 Parsing file_id")
    filename: Optional[str] = Field(None, description="原始文件名")
    page_count: int = Field(0, ge=0, description="页数")
    duration_ms: int = Field(0, ge=0, description="最近一次解析耗时")
    created_at: str = Field(..., description="首次写入时间（ISO8601, UTC）")
    updated_at: str = Field(..., description="最近刷新时间（ISO8601, UTC）")


class JobsCacheStore:
    """Redis 后端的 ADE Jobs 哈希去重缓存。"""

    _redis: Optional[aioredis.Redis] = None

    @classmethod
    async def _get_redis(cls) -> aioredis.Redis:
        if cls._redis is None:
            cls._redis = await RedisClient.get_client(decode_responses=True)
        return cls._redis

    @classmethod
    def _key(cls, file_hash: str) -> str:
        return f"{_CACHE_KEY_PREFIX}{file_hash}"

    @classmethod
    def now_iso(cls) -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    async def save(cls, record: JobsCacheRecord) -> None:
        r = await cls._get_redis()
        await r.setex(cls._key(record.file_hash), _CACHE_TTL_SECONDS, record.model_dump_json())
        logger.info(
            "JobsCacheStore | 保存 | hash={} job_id={} status={} ttl={}s",
            record.file_hash,
            record.job_id,
            record.status,
            _CACHE_TTL_SECONDS,
        )

    @classmethod
    async def get(cls, file_hash: str) -> Optional[JobsCacheRecord]:
        r = await cls._get_redis()
        data = await r.get(cls._key(file_hash))
        if data is None:
            return None
        try:
            return JobsCacheRecord.model_validate_json(data)
        except Exception as exc:
            logger.error(
                "JobsCacheStore | 解析失败 | hash={} err={}", file_hash, exc
            )
            return None

    @classmethod
    async def update_status(
        cls,
        file_hash: str,
        status: str,
        converted_path: Optional[str] = None,
        images_index_path: Optional[str] = None,
        page_count: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """更新缓存记录的状态与产物路径。"""
        rec = await cls.get(file_hash)
        if rec is None:
            return
        rec.status = status
        if converted_path is not None:
            rec.converted_path = converted_path
        if images_index_path is not None:
            rec.images_index_path = images_index_path
        if page_count is not None:
            rec.page_count = page_count
        if duration_ms is not None:
            rec.duration_ms = duration_ms
        rec.updated_at = cls.now_iso()
        await cls.save(rec)

    @classmethod
    async def delete(cls, file_hash: str) -> None:
        r = await cls._get_redis()
        await r.delete(cls._key(file_hash))
        logger.info("JobsCacheStore | 删除 | hash={}", file_hash)
