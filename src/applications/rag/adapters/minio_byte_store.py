from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence

from langchain_core.stores import BaseStore
from loguru import logger

from src.base.storage_service import StorageObjectNotFoundError, StorageService, storage_service


class AsyncMinioByteStore(BaseStore[str, bytes]):
    """Shared MinIO-backed async byte store for RAG adapters."""

    def __init__(
        self,
        *,
        bucket: str,
        object_prefix: str = "",
        max_concurrency: int = 1,
        storage: StorageService = storage_service,
    ) -> None:
        self.bucket = bucket
        self.object_prefix = object_prefix.strip("/")
        self.max_concurrency = max(1, max_concurrency)
        self._storage = storage

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        raise NotImplementedError(f"{type(self).__name__} only supports async operations")

    async def amget(self, keys: Sequence[str]) -> list[bytes | None]:
        return await self._amget(keys)

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        raise NotImplementedError(f"{type(self).__name__} only supports async operations")

    async def amset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        await self._amset(key_value_pairs)

    def mdelete(self, keys: Sequence[str]) -> None:
        raise NotImplementedError(f"{type(self).__name__} only supports async operations")

    async def amdelete(self, keys: Sequence[str]) -> None:
        await self._amdelete(keys)

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        raise NotImplementedError(f"{type(self).__name__} only supports async iteration")

    async def ayield_keys(self, *, prefix: str | None = None) -> AsyncIterator[str]:
        object_names = await self._storage.list_objects(
            self.bucket,
            prefix=self._object_prefix(prefix),
            recursive=True,
        )
        for object_name in object_names:
            key = self._key_from_object_name(object_name)
            if key:
                yield key

    async def _amget(self, keys: Sequence[str]) -> list[bytes | None]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _download(key: str) -> bytes | None:
            async with semaphore:
                try:
                    return await self._storage.download(self.bucket, self._object_name(key))
                except StorageObjectNotFoundError:
                    return None

        return await asyncio.gather(*(_download(key) for key in keys))

    async def _amset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _upload(key: str, value: bytes) -> str:
            async with semaphore:
                return await self._storage.upload(
                    self.bucket,
                    self._object_name(key),
                    value,
                    content_type="application/octet-stream",
                )

        await asyncio.gather(*(_upload(key, value) for key, value in key_value_pairs))

    async def _amdelete(self, keys: Sequence[str]) -> None:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _delete(key: str) -> None:
            async with semaphore:
                await self._storage.delete(self.bucket, self._object_name(key))

        results = await asyncio.gather(
            *(_delete(key) for key in keys),
            return_exceptions=True,
        )
        for key, result in zip(keys, results, strict=False):
            if isinstance(result, Exception):
                logger.warning("{} delete failed: key={}", type(self).__name__, key)

    async def clear(self) -> None:
        object_names = await self._storage.list_objects(
            self.bucket,
            prefix=self._object_prefix(None),
            recursive=True,
        )
        if object_names:
            results = await asyncio.gather(
                *(self._storage.delete(self.bucket, name) for name in object_names),
                return_exceptions=True,
            )
            for object_name, result in zip(object_names, results, strict=False):
                if isinstance(result, Exception):
                    logger.warning("{} clear failed: object={}", type(self).__name__, object_name)

    def _object_name(self, key: str) -> str:
        normalized_key = str(key).strip("/")
        if self.object_prefix:
            return f"{self.object_prefix}/{normalized_key}"
        return normalized_key

    def _object_prefix(self, prefix: str | None) -> str:
        clean_prefix = (prefix or "").strip("/")
        if self.object_prefix and clean_prefix:
            return f"{self.object_prefix}/{clean_prefix}"
        if self.object_prefix:
            return f"{self.object_prefix}/"
        return clean_prefix

    def _key_from_object_name(self, object_name: str) -> str:
        prefix = f"{self.object_prefix}/" if self.object_prefix else ""
        if prefix and object_name.startswith(prefix):
            return object_name[len(prefix) :]
        return object_name
