from __future__ import annotations

from dataclasses import dataclass
import time
from collections.abc import Sequence

from loguru import logger

from src.applications.rag.adapters.minio_byte_store import AsyncMinioByteStore
from src.applications.rag.config.rag_settings import RAG_SETTINGS
from src.base.storage_service import StorageService, storage_service


@dataclass(frozen=True)
class ByteStoreIOStats:
    get_calls: int = 0
    get_items: int = 0
    get_hits: int = 0
    get_misses: int = 0
    get_ms: float = 0.0
    set_calls: int = 0
    set_items: int = 0
    set_ms: float = 0.0


class MinioByteStore(AsyncMinioByteStore):
    """基于 MinIO 存储 embedding 缓存值的 LangChain ByteStore。"""

    def __init__(
        self,
        *,
        bucket: str | None = None,
        object_prefix: str = "embedding_cache",
        max_concurrency: int | None = None,
        storage: StorageService = storage_service,
    ) -> None:
        concurrency = max_concurrency or RAG_SETTINGS.embedding_cache_lookup_concurrency
        super().__init__(
            bucket=bucket or RAG_SETTINGS.embedding_cache_bucket,
            object_prefix=object_prefix,
            max_concurrency=concurrency,
            storage=storage,
        )
        self._lifecycle_ready = False
        self.reset_io_stats()

    def reset_io_stats(self) -> None:
        # 仅记录本实例生命周期内的缓存读写统计，用于 benchmark 和调试观察。
        self._get_calls = 0
        self._get_items = 0
        self._get_hits = 0
        self._get_misses = 0
        self._get_ms = 0.0
        self._set_calls = 0
        self._set_ms = 0.0
        self._set_items = 0

    def get_io_stats(self) -> ByteStoreIOStats:
        return ByteStoreIOStats(
            get_calls=self._get_calls,
            get_items=self._get_items,
            get_hits=self._get_hits,
            get_misses=self._get_misses,
            get_ms=self._get_ms,
            set_calls=self._set_calls,
            set_ms=self._set_ms,
            set_items=self._set_items,
        )

    async def amget(self, keys: Sequence[str]) -> list[bytes | None]:
        start = time.perf_counter()
        values = await self._amget(keys)
        elapsed_ms = (time.perf_counter() - start) * 1000
        hits = sum(1 for value in values if value is not None)
        self._get_calls += 1
        self._get_items += len(keys)
        self._get_hits += hits
        self._get_misses += len(keys) - hits
        self._get_ms += elapsed_ms
        logger.info("embedding缓存查找 | keys={} hits={} elapsed={:.2f}ms", len(keys), hits, elapsed_ms)
        return values

    async def amset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        start = time.perf_counter()
        await self._amset(key_value_pairs)
        await self._set_lifecycle_once()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._set_calls += 1
        self._set_ms += elapsed_ms
        self._set_items += len(key_value_pairs)
        logger.info("embedding缓存写入 | keys={} elapsed={:.2f}ms", len(key_value_pairs), elapsed_ms)

    async def _set_lifecycle_once(self) -> None:
        if self._lifecycle_ready:
            return
        await self._storage.set_bucket_lifecycle(
            self.bucket,
            expiration_days=RAG_SETTINGS.embedding_cache_ttl_days,
        )
        self._lifecycle_ready = True
