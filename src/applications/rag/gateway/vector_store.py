from __future__ import annotations

import asyncio
from typing import Any, Optional

from elasticsearch import AsyncElasticsearch, BadRequestError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from langchain_elasticsearch import AsyncDenseVectorStrategy, AsyncElasticsearchStore
from langchain_elasticsearch._utilities import _hits_to_docs_scores
from langchain_milvus import Milvus
from langchain_milvus.function import BM25BuiltInFunction
from loguru import logger

from ..config.rag_settings import RAG_SETTINGS
from ..schemas import SearchType


_ES_FILTER_METADATA_MAPPINGS: dict[str, dict[str, Any]] = {
    field_name: {
        "type": "text",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
    }
    for field_name in ("file_id", "folder_id", "type", "doc_kind")
}


class VectorStoreGateway(LangChainVectorStore):
    """基于 LangChain 的孙切片向量集合网关。"""

    def __init__(
        self,
        collection_name: str,
        embedding: Embeddings,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__()
        self._provider = (provider or RAG_SETTINGS.vector_db_provider).strip().lower()
        self._collection_name = collection_name
        self._embedding = embedding
        self._store: Optional[LangChainVectorStore] = None
        self._store_by_hybrid: dict[bool, LangChainVectorStore] = {}

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def collection_name(self) -> str:
        return self._collection_name

    async def get_store(self) -> LangChainVectorStore:
        return await self._ensure_store()

    async def _ensure_store(self, hybrid_enabled: Optional[bool] = None) -> LangChainVectorStore:
        hybrid = bool(RAG_SETTINGS.hybrid_enabled if hybrid_enabled is None else hybrid_enabled)
        if self._provider in {"es", "elasticsearch"}:
            cached = self._store_by_hybrid.get(hybrid)
            if cached is not None:
                return cached
            self._store = await self._create_es_store(hybrid)
            self._store_by_hybrid[hybrid] = self._store
            return self._store

        if self._store is not None:
            return self._store
        if self._provider == "milvus":
            self._store = self._create_milvus_store()
            return self._store
        raise ValueError(f"不支持的向量数据库: {self._provider}")

    async def _create_es_store(self, hybrid_enabled: bool) -> AsyncElasticsearchStore:
        client_kwargs: dict[str, Any] = {
            "request_timeout": int(RAG_SETTINGS.es_request_timeout_s),
            "retry_on_timeout": bool(RAG_SETTINGS.es_retry_on_timeout),
            "max_retries": int(RAG_SETTINGS.es_bulk_max_retries),
        }
        if RAG_SETTINGS.es_username:
            client_kwargs["basic_auth"] = (RAG_SETTINGS.es_username, RAG_SETTINGS.es_password or "")

        es_client = AsyncElasticsearch(parse_es_hosts(RAG_SETTINGS.es_host), **client_kwargs)
        strategy = AsyncDenseVectorStrategy(
            hybrid=hybrid_enabled,
            rrf=False if hybrid_enabled else None,
            text_field="text",
        )
        store = AsyncElasticsearchStore(
            index_name=self._collection_name,
            embedding=self._embedding,
            client=es_client,
            strategy=strategy,
            vector_query_field="vector",
            query_field="text",
            num_dimensions=int(RAG_SETTINGS.embedding_dim),
            metadata_mappings=_ES_FILTER_METADATA_MAPPINGS,
        )
        logger.info(
            "LangChain ES vectorstore 已创建: index={}, hybrid={}",
            self._collection_name,
            hybrid_enabled,
        )
        return store

    async def ensure_initialized(self) -> None:
        """创建配置的向量集合，并拒绝不兼容的集合结构。"""
        store = await self._ensure_store()
        if self._provider not in {"es", "elasticsearch"}:
            return
        await self._ensure_es_index_initialized(store)

    async def _ensure_es_index_initialized(self, store: AsyncElasticsearchStore) -> None:
        es_store = getattr(store, "_store", None)
        if es_store is None:
            raise RuntimeError("无法获取底层 Elasticsearch store")

        try:
            await es_store._create_index_if_not_exists()
        except BadRequestError as error:
            # API 和 Worker 可能同时启动并发创建同一个索引。
            error_body = error.body if isinstance(error.body, dict) else {}
            error_detail = error_body.get("error", {})
            error_type = error_detail.get("type") if isinstance(error_detail, dict) else None
            if error_type != "resource_already_exists_exception":
                raise

        response = await es_store.client.indices.get_mapping(index=self._collection_name)
        properties = response.get(self._collection_name, {}).get("mappings", {}).get("properties", {})
        vector = properties.get("vector", {})
        expected_dim = int(RAG_SETTINGS.embedding_dim)
        if vector.get("type") != "dense_vector" or int(vector.get("dims") or 0) != expected_dim:
            raise RuntimeError(
                f"Elasticsearch 索引向量字段不匹配: index={self._collection_name}, "
                f"expected_dims={expected_dim}, actual={vector}"
            )
        if properties.get("metadata", {}).get("enabled") is False:
            raise RuntimeError(
                f"Elasticsearch 索引 metadata 不可索引: index={self._collection_name}, enabled=false"
            )

        # 检索策略只声明向量字段，这里补齐 BM25 文本和过滤字段。
        await es_store.client.indices.put_mapping(
            index=self._collection_name,
            properties={
                "text": {"type": "text"},
                "metadata": {
                    "type": "object",
                    "properties": _ES_FILTER_METADATA_MAPPINGS,
                },
            },
        )
        logger.info(
            "Elasticsearch 向量索引初始化完成: index={}, dims={}",
            self._collection_name,
            RAG_SETTINGS.embedding_dim,
        )

    def _create_milvus_store(self) -> Milvus:
        builtin_functions = [BM25BuiltInFunction()] if RAG_SETTINGS.hybrid_enabled else None
        store = Milvus(
            embedding_function=self._embedding,
            collection_name=self._collection_name,
            connection_args={
                "uri": f"http://{RAG_SETTINGS.milvus_host}:{RAG_SETTINGS.milvus_port}",
                "db_name": RAG_SETTINGS.milvus_db_name,
            },
            enable_dynamic_field=True,
            index_params={
                "metric_type": RAG_SETTINGS.milvus_metric,
                "index_type": RAG_SETTINGS.milvus_index,
            },
            builtin_function=builtin_functions,
            timeout=120,
        )
        logger.info("LangChain Milvus vectorstore 已创建: collection={}", self._collection_name)
        return store

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        skipped = len(documents) - len(valid_docs)
        if skipped:
            logger.warning("跳过 {} 个空文本孙切片 Document", skipped)
        if not valid_docs:
            return []
        store = await self._ensure_store()
        return await store.aadd_documents(valid_docs, **kwargs)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[list[dict] | str] = None,
        hybrid_enabled: Optional[bool] = None,
        **kwargs: Any,
    ) -> list[Document]:
        store = await self._ensure_store(hybrid_enabled=hybrid_enabled)
        if self._provider in {"es", "elasticsearch"}:
            kwargs.setdefault("fetch_k", max(int(k), 50))
            return await store.asimilarity_search(query, k=k, filter=filter, **kwargs)
        return await store.asimilarity_search(query, k=k, expr=filter, **kwargs)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[list[dict] | str] = None,
        hybrid_enabled: Optional[bool] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        store = await self._ensure_store(hybrid_enabled=hybrid_enabled)
        if self._provider in {"es", "elasticsearch"}:
            if bool(RAG_SETTINGS.hybrid_enabled if hybrid_enabled is None else hybrid_enabled):
                hits = await store._store.search(  # type: ignore[attr-defined]
                    query=query,
                    k=k,
                    num_candidates=kwargs.pop("fetch_k", max(int(k), 50)),
                    filter=filter,
                    custom_query=kwargs.pop("custom_query", None),
                )
                return _hits_to_docs_scores(
                    hits=hits,
                    content_field="text",
                    fields=["metadata", "text"],
                )
            hits = await store._store.search(  # type: ignore[attr-defined]
                query=query,
                k=k,
                num_candidates=kwargs.pop("num_candidates", max(int(k), 50)),
                filter=filter,
                custom_query=kwargs.pop("custom_query", None),
            )
            return _hits_to_docs_scores(
                hits=hits,
                content_field="text",
                fields=["metadata", "text"],
            )
        return await store.asimilarity_search_with_score(query, k=k, expr=filter, **kwargs)

    async def adelete(
        self,
        *,
        ids: Optional[list[str]] = None,
        filter: Optional[list[dict] | str] = None,
        **kwargs: Any,
    ) -> None:
        store = await self._ensure_store()
        if self._provider in {"es", "elasticsearch"}:
            await store.adelete(ids=ids, filter=filter, **kwargs)
        else:
            await store.adelete(ids=ids, expr=filter, **kwargs)

    async def aexists_by_file_id(self, file_id: str) -> bool:
        if self._provider in {"es", "elasticsearch"}:
            return await self._aexists_by_file_id_es(file_id)
        if self._provider == "milvus":
            return await self._aexists_by_file_id_milvus(file_id)
        raise ValueError(f"不支持的向量数据库: {self._provider}")

    async def _aexists_by_file_id_es(self, file_id: str) -> bool:
        store = await self._ensure_store()
        es_store = getattr(store, "_store", None)
        client = getattr(es_store, "client", None)
        index_name = getattr(es_store, "index_name", self._collection_name)
        if client is None:
            raise RuntimeError("ES vectorstore client 未初始化")
        response = await client.count(
            index=index_name,
            query={"term": {"metadata.file_id.keyword": file_id}},
        )
        return int(response.get("count") or 0) > 0

    async def _aexists_by_file_id_milvus(self, file_id: str) -> bool:
        expr = build_milvus_expr(file_id=file_id)
        if not expr:
            return False
        store = await self._ensure_store()
        client = getattr(store, "aclient", None)
        if client is None:
            raise RuntimeError("Milvus async client 未初始化")
        result = await client.query(
            self._collection_name,
            filter=expr,
            output_fields=["pk"],
            limit=1,
        )
        return bool(result)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        async def _do() -> list[Document]:
            return await self.asimilarity_search(query, k=k, **kwargs)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_do())
        raise RuntimeError("当前处于事件循环中，请使用 asimilarity_search")

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "VectorStoreGateway":
        del texts, embedding, metadatas, ids, kwargs
        raise NotImplementedError("VectorStoreGateway 不支持 from_texts，请使用 aadd_documents 写入")

    async def aclose(self) -> None:
        stores = list(self._store_by_hybrid.values())
        if self._store is not None:
            stores.append(self._store)
        for store in list(dict.fromkeys(stores)):
            close_fn = getattr(store, "aclose", None)
            if callable(close_fn):
                await close_fn()
        self._store = None
        self._store_by_hybrid.clear()


def parse_es_hosts(host: str) -> list[str]:
    return [f"http://{p.strip()}" for p in host.split(",") if p.strip()]


def build_es_filter(
    file_id: Optional[str] = None,
    *,
    file_ids: Optional[list[str]] = None,
    folder_id: Optional[str] = None,
    search_type: Optional[Any] = None,
    doc_kinds: Optional[list[str]] = None,
) -> Optional[list[dict]]:
    filters: list[dict] = []
    normalized_file_ids = _normalize_filter_values(file_ids or ([file_id] if file_id else None))
    if len(normalized_file_ids) == 1:
        filters.append({"term": {"metadata.file_id.keyword": normalized_file_ids[0]}})
    elif normalized_file_ids:
        filters.append({"terms": {"metadata.file_id.keyword": normalized_file_ids}})
    if folder_id:
        filters.append({"term": {"metadata.folder_id.keyword": folder_id}})
    type_values = _normalize_metadata_values(search_type)
    if len(type_values) == 1:
        filters.append({"term": {"metadata.type.keyword": type_values[0]}})
    elif type_values:
        filters.append({"terms": {"metadata.type.keyword": type_values}})
    doc_kind_filter = _build_es_doc_kind_filter(doc_kinds)
    if doc_kind_filter:
        filters.append(doc_kind_filter)
    return filters if filters else None


def build_milvus_expr(
    file_id: Optional[str] = None,
    *,
    file_ids: Optional[list[str]] = None,
    folder_id: Optional[str] = None,
    search_type: Optional[Any] = None,
    doc_kinds: Optional[list[str]] = None,
) -> Optional[str]:
    parts: list[str] = []
    normalized_file_ids = _normalize_filter_values(file_ids or ([file_id] if file_id else None))
    if normalized_file_ids:
        parts.append(_build_milvus_in_expr("file_id", normalized_file_ids))
    if folder_id:
        parts.append(f'folder_id == "{_escape_milvus_string(folder_id)}"')
    type_values = _normalize_metadata_values(search_type)
    if type_values:
        parts.append(_build_milvus_in_expr("type", type_values))
    normalized_doc_kinds = _normalize_filter_values(doc_kinds)
    if normalized_doc_kinds:
        parts.append(_build_milvus_in_expr("doc_kind", normalized_doc_kinds))
    return " && ".join(parts) if parts else None


def _normalize_metadata_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, SearchType):
        value = value.to_metadata_type_filter()
    elif hasattr(value, "to_metadata_type_filter"):
        value = value.to_metadata_type_filter()
    if isinstance(value, (list, tuple, set)):
        return _normalize_filter_values(list(value))
    return _normalize_filter_values([value])


def _normalize_filter_values(values: Optional[list[Any]]) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        item = str(value).strip()
        if item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return normalized


def _build_es_doc_kind_filter(doc_kinds: Optional[list[str]]) -> Optional[dict]:
    normalized = _normalize_filter_values(doc_kinds)
    if not normalized:
        return None
    clauses: list[dict] = []
    values = [item for item in normalized if item]
    if values:
        clauses.append({"terms": {"metadata.doc_kind.keyword": values}})
    if "" in normalized:
        clauses.append({"term": {"metadata.doc_kind.keyword": ""}})
        clauses.append({"bool": {"must_not": [{"exists": {"field": "metadata.doc_kind"}}]}})
    if len(clauses) == 1:
        return clauses[0]
    return {"bool": {"should": clauses, "minimum_should_match": 1}}


def _build_milvus_in_expr(field_name: str, values: list[str]) -> str:
    if len(values) == 1:
        return f'{field_name} == "{_escape_milvus_string(values[0])}"'
    formatted = ", ".join(f'"{_escape_milvus_string(value)}"' for value in values)
    return f"{field_name} in [{formatted}]"


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
