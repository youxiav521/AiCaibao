from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import httpx
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from pydantic import SecretStr

from ...config.rag_settings import RAG_SETTINGS
from .minio_byte_store import MinioByteStore

# embedding SDK 默认超时与连接限制，可通过环境变量覆盖
# EMBEDDING_TIMEOUT: 总超时（秒），EMBEDDING_CONNECT_TIMEOUT: 连接建立超时（秒）
# EMBEDDING_MAX_CONNECTIONS: 最大连接数，EMBEDDING_MAX_KEEPALIVE_CONNECTIONS: 最大保活连接数
_EMBEDDING_DEFAULT_TIMEOUT = httpx.Timeout(
    timeout=float(os.getenv("EMBEDDING_TIMEOUT", "600")),
    connect=float(os.getenv("EMBEDDING_CONNECT_TIMEOUT", "5.0")),
)
_EMBEDDING_DEFAULT_LIMITS = httpx.Limits(
    max_connections=int(os.getenv("EMBEDDING_MAX_CONNECTIONS", "1000")),
    max_keepalive_connections=int(os.getenv("EMBEDDING_MAX_KEEPALIVE_CONNECTIONS", "100")),
)


class OpenAIEmbedder(Embeddings):
    """基于 LangChain 的 OpenAI 兼容 embedding 适配器。"""

    def __init__(self) -> None:
        self.api_key = RAG_SETTINGS.embedding_api_key
        self.base_url = RAG_SETTINGS.embedding_base_url
        self.model = RAG_SETTINGS.embedding_model
        self.batch_size = RAG_SETTINGS.embedding_batch_size
        self.embedding_cache_enabled = RAG_SETTINGS.embedding_cache_enabled
        self.instruct_enabled = RAG_SETTINGS.embedding_instruct_enabled
        self.query_instruction = RAG_SETTINGS.embedding_query_instruction

        self.embedding_cache = MinioByteStore() if self.embedding_cache_enabled else None
        # 注意：httpx.AsyncClient / OpenAIEmbeddings 不能在 __init__ 中创建。
        # openai 库的 AsyncAPIClient.__del__ 会调用
        # asyncio.get_running_loop().create_task(self.aclose())，在 uvicorn --reload
        # 模式下旧 loop 关闭后会抛 RuntimeError: Event loop is closed。
        # 改为首次使用时通过 _ensure_embeddings 懒加载。
        self._sync_http_client: httpx.Client | None = None
        self._async_http_client: httpx.AsyncClient | None = None
        self._underlying_embeddings: OpenAIEmbeddings | None = None
        self._embeddings: Embeddings | None = None

    def _ensure_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._underlying_embeddings = self._build_openai_embeddings()
            self._embeddings = self._build_embeddings(self._underlying_embeddings)
        return self._embeddings

    @property
    def underlying_embeddings(self) -> OpenAIEmbeddings:
        self._ensure_embeddings()
        assert self._underlying_embeddings is not None
        return self._underlying_embeddings

    @property
    def embeddings(self) -> Embeddings:
        return self._ensure_embeddings()

    def _build_openai_embeddings(self) -> OpenAIEmbeddings:
        api_key = SecretStr(self.api_key) if self.api_key else None
        dimensions = RAG_SETTINGS.embedding_dim if RAG_SETTINGS.embedding_dim_enabled else None
        # OpenAIEmbeddings 不直接支持 limits 配置，通过 httpx client 注入
        self._sync_http_client = httpx.Client(
            timeout=_EMBEDDING_DEFAULT_TIMEOUT,
            limits=_EMBEDDING_DEFAULT_LIMITS,
            follow_redirects=True,
        )
        self._async_http_client = httpx.AsyncClient(
            timeout=_EMBEDDING_DEFAULT_TIMEOUT,
            limits=_EMBEDDING_DEFAULT_LIMITS,
            follow_redirects=True,
        )
        return OpenAIEmbeddings(
            model=self.model,
            api_key=api_key,
            base_url=self.base_url,
            dimensions=dimensions,
            chunk_size=self.batch_size,
            check_embedding_ctx_length=False,
            # 显式传入 timeout，确保配置生效
            timeout=_EMBEDDING_DEFAULT_TIMEOUT,
            http_client=self._sync_http_client,
            http_async_client=self._async_http_client,
        )

    def _build_embeddings(self, underlying: Embeddings) -> Embeddings:
        if not self.embedding_cache_enabled or self.embedding_cache is None:
            return underlying

        return CacheBackedEmbeddings.from_bytes_store(
            underlying,
            self.embedding_cache,
            namespace=f"{self._cache_namespace()}/document/",
            batch_size=self.batch_size,
            key_encoder="sha256",
        )

    def _cache_namespace(self) -> str:
        cache_parts = {
            "model": self.model,
            "dimensions": RAG_SETTINGS.embedding_dim if RAG_SETTINGS.embedding_dim_enabled else None,
            "instruct_enabled": self.instruct_enabled,
            "query_instruction": self.query_instruction if self.instruct_enabled else None,
        }
        raw = json.dumps(cache_parts, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.instruct_enabled and self.query_instruction:
            text = f"Instruct:{self.query_instruction}\nQuery:{text}"
        return self.embeddings.embed_query(text)

    async def aembed_texts(
        self,
        texts: list[str],
        *,
        stopwatch_task: Any | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        if stopwatch_task is not None and self.embedding_cache_enabled and self.embedding_cache is not None:
            with stopwatch_task.task("embedding_cache_lookup") as task:
                task.detail(f"{len(texts)} texts")
                self.embedding_cache.reset_io_stats()
                start = time.perf_counter()
                vectors = await self.embeddings.aembed_documents(texts)
                total_ms = (time.perf_counter() - start) * 1000
                cache_stats = self.embedding_cache.get_io_stats()
                embed_ms = max(total_ms - cache_stats.get_ms - cache_stats.set_ms, 0.0)
                task.detail(
                    f"hits={cache_stats.get_hits} misses={cache_stats.get_misses} "
                    f"cache_get_ms={cache_stats.get_ms:.1f} embedding_ms={embed_ms:.1f} "
                    f"cache_set_ms={cache_stats.set_ms:.1f} set_items={cache_stats.set_items}",
                )
                task.detail("langchain CacheBackedEmbeddings")
        else:
            vectors = await self.embeddings.aembed_documents(texts)

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding response size mismatch: model={self.model}, requested={len(texts)}, returned={len(vectors)}",
            )

        logger.info(
            "Embedding 完成: total_texts={}, model={}, cache_enabled={}",
            len(texts),
            self.model,
            self.embedding_cache_enabled,
        )
        return vectors

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.aembed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        if self.instruct_enabled and self.query_instruction:
            text = f"Instruct:{self.query_instruction}\nQuery:{text}"
        return await self.embeddings.aembed_query(text)

    async def aclose(self) -> None:
        # 注入的 http_async_client 即 underlying_embeddings.async_client._client
        # 这里基于本实例持有的 httpx client 引用统一关闭，避免重复关闭
        if self._async_http_client is not None:
            await self._async_http_client.aclose()
        if self._sync_http_client is not None:
            self._sync_http_client.close()
