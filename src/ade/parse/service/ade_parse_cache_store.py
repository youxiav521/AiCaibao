"""
ADE Parse 结果去重缓存（Redis）。

目标：
- 基于文件内容哈希（sha256）+ split 维度建立稳定映射
- 命中后直接复用已存在的 parse_response_path，避免重复解析
- 仅缓存“结果引用路径”，不在 Redis 存储大对象，便于运维与清理
"""

import os
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger
from pydantic import BaseModel, Field

from src.base.redis_client import RedisClient

_CACHE_KEY_PREFIX = "ade_parse_cache:"
_CACHE_TTL_SECONDS = int(os.getenv("ADE_PARSE_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))


class AdeParseCacheRecord(BaseModel):
    """文件哈希去重缓存记录。"""

    file_hash: str = Field(..., description="文件内容 sha256（小写十六进制）")
    split_by_page: bool = Field(False, description="是否按页分段")
    parse_response_path: str = Field(..., description="结果路径，格式 {bucket}/{object_name}")
    file_id: Optional[str] = Field(None, description="最近一次关联的 file_id")
    filename: Optional[str] = Field(None, description="原始文件名")
    page_count: int = Field(0, ge=0, description="页数")
    duration_ms: int = Field(0, ge=0, description="最近一次解析耗时")
    created_at: str = Field(..., description="首次写入时间（ISO8601, UTC）")
    updated_at: str = Field(..., description="最近刷新时间（ISO8601, UTC）")
    hit_count: int = Field(0, ge=0, description="命中次数")
    last_hit_at: Optional[str] = Field(None, description="最近命中时间（ISO8601, UTC）")


class AdeParseCacheStore:
    """Redis 后端的 ADE Parse 哈希去重缓存。"""

    _redis: Optional[aioredis.Redis] = None

    @classmethod
    async def _get_redis(cls) -> aioredis.Redis:
        if cls._redis is None:
            cls._redis = await RedisClient.get_client(decode_responses=True)
        return cls._redis

    @classmethod
    def _key(cls, file_hash: str, split_by_page: bool) -> str:
        split_flag = "page" if split_by_page else "full"
        return f"{_CACHE_KEY_PREFIX}{file_hash}:{split_flag}"

    @classmethod
    def now_iso(cls) -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    async def save(cls, record: AdeParseCacheRecord) -> None:
        r = await cls._get_redis()
        key = cls._key(record.file_hash, record.split_by_page)
        await r.setex(key, _CACHE_TTL_SECONDS, record.model_dump_json())
        logger.info(
            "AdeParseCacheStore | 保存 | hash={} split_by_page={} path={} ttl={}s",
            record.file_hash,
            record.split_by_page,
            record.parse_response_path,
            _CACHE_TTL_SECONDS,
        )

    @classmethod
    async def get(cls, file_hash: str, split_by_page: bool) -> Optional[AdeParseCacheRecord]:
        r = await cls._get_redis()
        data = await r.get(cls._key(file_hash, split_by_page))
        if data is None:
            return None
        try:
            return AdeParseCacheRecord.model_validate_json(data)
        except Exception as exc:
            logger.error(
                "AdeParseCacheStore | 解析失败 | hash={} split_by_page={} err={}",
                file_hash,
                split_by_page,
                exc,
            )
            return None

    @classmethod
    async def touch_hit(cls, file_hash: str, split_by_page: bool) -> None:
        rec = await cls.get(file_hash, split_by_page)
        if rec is None:
            return
        rec.hit_count += 1
        rec.last_hit_at = cls.now_iso()
        rec.updated_at = rec.last_hit_at
        await cls.save(rec)

    @classmethod
    async def delete(cls, file_hash: str, split_by_page: bool) -> None:
        r = await cls._get_redis()
        await r.delete(cls._key(file_hash, split_by_page))
        logger.info(
            "AdeParseCacheStore | 删除 | hash={} split_by_page={}",
            file_hash,
            split_by_page,
        )
