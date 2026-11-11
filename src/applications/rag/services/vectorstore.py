from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

from elastic_transport import ConnectionError as EsConnectionError
from elastic_transport import ConnectionTimeout as EsConnectionTimeout
from elasticsearch.helpers import BulkIndexError, async_bulk
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from src.base.langfuse_observe import observe
from loguru import logger

from ..adapters.minio_byte_store import AsyncMinioByteStore
from ..adapters.embeddings.openai_embedder import OpenAIEmbedder
from ..config.rag_settings import RAG_SETTINGS
from ..gateway import VectorStoreGateway, build_es_filter, build_milvus_expr
from ..schemas import ChildChunk, GrandchildChunk, ParentChunk
from ..utils.validators import validate_file_id
from src.utils import json_utils

_CHILDREN_PREFIX = "pcr/children"
_PARENTS_PREFIX = "pcr/parents"
_INVALID_OBJECT_SEGMENT_CHARS = set('%<>:"/\\|?*')


def _chunk_storage_key(file_id: str, chunk_id: str) -> str:
    normalized_file_id = validate_file_id(file_id)
    normalized_chunk_id = str(chunk_id or "").strip().strip("/")
    if not normalized_chunk_id:
        raise ValueError("chunk_id 不能为空")
    return f"{normalized_file_id}/{_safe_object_segment(normalized_chunk_id)}"


def _normalize_chunk_storage_key(value: str) -> str:
    file_id, separator, chunk_id = str(value or "").partition("/")
    if not separator:
        return str(value or "")
    return _chunk_storage_key(file_id, chunk_id)

# 解决windows下minio对象名不能为特殊符号的问题
def _safe_object_segment(value: str) -> str:
    return "".join(
        f"%{ord(ch):02X}" if ch in _INVALID_OBJECT_SEGMENT_CHARS or ord(ch) < 32 else ch
        for ch in value
    )


def _parent_storage_key(chunk: ParentChunk) -> str:
    return _chunk_storage_key(chunk.file_id, chunk.parent_id)


def _child_storage_key(chunk: ChildChunk | GrandchildChunk) -> str:
    return _chunk_storage_key(chunk.file_id, chunk.child_id)


def _build_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


def _build_reranker() -> Any:
    try:
        from ..adapters.rerankers.openai_reranker import OpenAIReranker

        if RAG_SETTINGS.rerank_enabled:
            return OpenAIReranker()
    except Exception as error:
        logger.warning("Failed to initialize reranker: {}", error)
    return None


def _chunk_to_document(chunk: ChildChunk | ParentChunk) -> tuple[str, Document]:
    metadata = chunk.model_dump(mode="json", exclude_none=True)
    return (
        _child_storage_key(chunk) if isinstance(chunk, ChildChunk) else _parent_storage_key(chunk),
        Document(page_content=chunk.text, metadata=metadata),
    )


def _grandchild_to_document(chunk: GrandchildChunk) -> Document:
    metadata = chunk.model_dump(mode="json", exclude_none=True)
    metadata.pop("text", None)
    return Document(page_content=chunk.text, metadata=metadata)


class PrimaryGrandchildrenVectorStore(LangChainVectorStore):
    """主孙切片集合的 LangChain VectorStore 门面。"""

    def __init__(
        self,
        *,
        stores: "RagStores",
        file_ids: Optional[list[str]] = None,
        folder_id: Optional[str] = None,
        search_type: Optional[str] = None,
        doc_kinds: Optional[list[str]] = None,
        use_hybrid: bool = True,
    ) -> None:
        super().__init__()
        self._stores = stores
        self._file_ids = file_ids
        self._folder_id = folder_id
        self._search_type = search_type
        self._doc_kinds = doc_kinds
        self._use_hybrid = bool(use_hybrid)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[list[dict] | str] = None,
        hybrid_enabled: Optional[bool] = None,
        **kwargs: Any,
    ) -> list[Document]:
        docs_scores = await self.asimilarity_search_with_score(
            query,
            k=k,
            filter=filter,
            hybrid_enabled=hybrid_enabled,
            **kwargs,
        )
        return [doc for doc, _ in docs_scores]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[list[dict] | str] = None,
        hybrid_enabled: Optional[bool] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        use_hybrid = self._use_hybrid if hybrid_enabled is None else bool(hybrid_enabled)
        search_filter = filter
        if search_filter is None:
            search_filter = self._stores._build_primary_search_kwargs(
                top_k=k,
                file_ids=self._file_ids,
                folder_id=self._folder_id,
                search_type=self._search_type,
                doc_kinds=self._doc_kinds,
            ).get("filter")
        docs_scores = await self._stores._grandchildren_store.asimilarity_search_with_score(
            query=query,
            k=int(k),
            filter=search_filter,
            hybrid_enabled=use_hybrid,
            **kwargs,
        )
        docs: list[tuple[Document, float]] = []
        for doc, score in docs_scores:
            metadata = dict(doc.metadata or {})
            metadata["score"] = score
            docs.append((Document(page_content=doc.page_content, metadata=metadata), score))
        return docs

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
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> "PrimaryGrandchildrenVectorStore":
        del texts, embedding, metadatas, kwargs
        raise NotImplementedError("PrimaryGrandchildrenVectorStore 不支持 from_texts")


class PrimaryGrandchildrenRetriever:
    """直接召回孙切片的轻量检索适配器。"""

    def __init__(self, *, vectorstore: PrimaryGrandchildrenVectorStore, top_k: int) -> None:
        self._vectorstore = vectorstore
        self._top_k = int(top_k)

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        docs_scores = await self._vectorstore.asimilarity_search_with_score(query, k=self._top_k)
        return [doc for doc, _ in docs_scores]

    async def ainvoke(self, query: str) -> list[Document]:
        return await self.aget_relevant_documents(query)


class ChildViaGrandchildrenRetriever:
    """先召回孙切片，再按 file_id 目录加载子切片。"""

    def __init__(self, *, stores: "RagStores", vectorstore: PrimaryGrandchildrenVectorStore, top_k: int) -> None:
        self._stores = stores
        self._vectorstore = vectorstore
        self._top_k = int(top_k)

    async def ainvoke(self, query: str) -> list[Document]:
        grandchild_docs = await self._vectorstore.asimilarity_search(query, k=self._top_k)
        child_keys: list[str] = []
        for doc in grandchild_docs:
            metadata = dict(doc.metadata or {})
            file_id = str(metadata.get("file_id") or "").strip()
            child_id = str(metadata.get("child_id") or "").strip()
            if not file_id or not child_id:
                continue
            try:
                child_key = _chunk_storage_key(file_id, child_id)
            except ValueError:
                continue
            if child_key not in child_keys:
                child_keys.append(child_key)

        if not child_keys:
            return []

        child_docs = await self._stores._children_docstore.amget(child_keys)
        return [doc for doc in child_docs if doc is not None]


class RagStores:
    """基于 MinIO 文档存储和 LangChain 向量库的 RAG 存储门面。"""

    def __init__(
        self,
        storage_collection_name: str | None = None,
        langchain_collection_name: str | None = None,
    ) -> None:
        self.storage_collection_name = storage_collection_name or RAG_SETTINGS.rag_storage_collection_name
        self.langchain_collection_name = (
            langchain_collection_name
            or RAG_SETTINGS.vector_langchain_collection
        )

        self.settings = RAG_SETTINGS
        self.embedder = _build_embedder()
        self.reranker = _build_reranker()
        self._store_lock = asyncio.Lock()

        self._children_byte_store = AsyncMinioByteStore(
            bucket=RAG_SETTINGS.pcr_minio_bucket,
            object_prefix=f"{_CHILDREN_PREFIX}/{self.storage_collection_name}",
        )
        self._children_docstore = create_kv_docstore(self._children_byte_store)
        self._parents_byte_store = AsyncMinioByteStore(
            bucket=RAG_SETTINGS.pcr_minio_bucket,
            object_prefix=f"{_PARENTS_PREFIX}/{self.storage_collection_name}",
        )
        self._grandchildren_store = VectorStoreGateway(
            collection_name=self.langchain_collection_name,
            embedding=self.embedder,
        )

    @property
    def is_elasticsearch(self) -> bool:
        provider = self.settings.vector_db_provider
        return provider in {"es", "elasticsearch"}

    def _get_store_lock(self):
        lock = getattr(self, "_store_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            setattr(self, "_store_lock", lock)
        return lock

    async def upsert_children(self, documents: Sequence[ChildChunk]) -> int:
        pairs = [_chunk_to_document(chunk) for chunk in documents if chunk.child_id]
        if pairs:
            await self._children_docstore.amset(pairs)
        return len(pairs)

    async def upsert_parents(self, documents: Sequence[ParentChunk]) -> int:
        pairs = [
            (
                _parent_storage_key(chunk),
                json_utils.dumps_bytes(chunk.model_dump(mode="json", exclude_none=True), ensure_ascii=False),
            )
            for chunk in documents
            if chunk.parent_id
        ]
        if pairs:
            await self._parents_byte_store.amset(pairs)
        return len(pairs)

    async def existing_parent_ids(self, parent_ids: Sequence[str]) -> set[str]:
        ids = [str(item) for item in parent_ids if item]
        values = await self._parents_byte_store.amget([_normalize_chunk_storage_key(item) for item in ids])
        return {item for item, value in zip(ids, values) if value is not None}

    async def existing_child_ids(self, child_ids: Sequence[str]) -> set[str]:
        ids = [str(item) for item in child_ids if item]
        values = await self._children_docstore.amget([_normalize_chunk_storage_key(item) for item in ids])
        return {item for item, value in zip(ids, values) if value is not None}

    async def existing_grandchild_ids(self, grandchild_ids: Sequence[str]) -> set[str]:
        ids = [str(item) for item in grandchild_ids if item]
        if not ids or not self.is_elasticsearch:
            return set()
        store = await self._grandchildren_store.get_store()
        es_store = getattr(store, "_store", None)
        if es_store is None:
            return set()
        response = await es_store.client.mget(index=self.langchain_collection_name, ids=ids)
        return {str(doc.get("_id")) for doc in response.get("docs", []) if doc.get("found")}

    async def upsert_grandchildren(self, documents: Sequence[GrandchildChunk]) -> int:
        valid_chunks = [chunk for chunk in documents if chunk.text and chunk.text.strip()]
        if not valid_chunks:
            return 0
        if self.is_elasticsearch:
            written_with_existing_vectors = await self._try_upsert_es_grandchildren_with_existing_vectors(valid_chunks)
            if written_with_existing_vectors is not None:
                return written_with_existing_vectors
        ids = await self._grandchildren_store.aadd_documents(
            [_grandchild_to_document(chunk) for chunk in valid_chunks],
            ids=[chunk.grandchild_id for chunk in valid_chunks],
        )
        return len(ids)

    async def upsert_grandchild_es_actions(self, actions: Sequence[dict[str, Any]]) -> int:
        if not self.is_elasticsearch:
            raise RuntimeError("upsert_grandchild_es_actions 仅支持 Elasticsearch")
        valid_actions = [dict(action) for action in actions if action.get("_id") and action.get("_source")]
        if not valid_actions:
            return 0

        store = await self._grandchildren_store.get_store()
        es_store = getattr(store, "_store", None)
        if es_store is None:
            raise RuntimeError("无法获取底层 Elasticsearch store")
        await es_store._create_index_if_not_exists()

        requests = [
            {
                "_index": self.langchain_collection_name,
                "_id": action["_id"],
                "_source": dict(action["_source"]),
            }
            for action in valid_actions
        ]
        success, errors = await self._run_es_bulk_with_retry(
            es_store.client,
            requests,
            refresh=bool(RAG_SETTINGS.es_bulk_refresh),
            raise_on_error=False,
            chunk_size=max(1, int(RAG_SETTINGS.es_bulk_chunk_size)),
            max_chunk_bytes=int(RAG_SETTINGS.es_bulk_max_chunk_bytes),
            max_retries=int(RAG_SETTINGS.es_bulk_max_retries),
            initial_backoff=float(RAG_SETTINGS.es_bulk_initial_backoff_s),
            max_backoff=float(RAG_SETTINGS.es_bulk_max_backoff_s),
        )
        if errors:
            logger.warning(
                "直接回填孙切片 ES bulk 部分失败: success={}, failed={}",
                success,
                len(errors),
            )
        return int(success)

    async def delete_by_file_id(self, file_id: str) -> bool:
        normalized_file_id = validate_file_id(file_id)
        success = True
        for operation_name, operation in [
            ("children_minio", self._delete_minio_children_by_file_id),
            ("parents_minio", self._delete_minio_parents_by_file_id),
            ("grandchildren_primary", self._delete_primary_grandchildren_by_file_id),
        ]:
            try:
                await operation(normalized_file_id)
            except Exception:
                logger.warning(
                    "删除新 RAG 存储失败: operation={}, file_id={}",
                    operation_name,
                    normalized_file_id,
                    exc_info=True,
                )
                success = False

        return success

    async def reset_all(self) -> bool:
        try:
            async with self._get_store_lock():
                for operation_name, operation in [
                    ("children_minio", lambda: self._clear_minio_store(self._children_byte_store)),
                    ("parents_minio", lambda: self._clear_minio_store(self._parents_byte_store)),
                    ("grandchildren_primary", self._reset_primary_grandchildren),
                ]:
                    try:
                        await operation()
                    except Exception:
                        logger.warning("重置新 RAG 存储失败: operation={}", operation_name, exc_info=True)
                return True
        except Exception:
            logger.exception("全库重置异常")
            return False

    async def exists_file_id(self, file_id: str) -> bool:
        normalized_file_id = validate_file_id(file_id)
        return await self._grandchildren_store.aexists_by_file_id(normalized_file_id)

    def get_grandchildren_retriever(
        self,
        top_k: int = 8,
        file_ids: Optional[list[str]] = None,
        folder_id: Optional[str] = None,
        search_type: Optional[str] = None,
        doc_kinds: Optional[list[str]] = None,
        use_hybrid: bool = True,
        output_fields: Optional[list[str] | tuple[str, ...]] = None,
    ) -> PrimaryGrandchildrenRetriever:
        del output_fields
        vectorstore = PrimaryGrandchildrenVectorStore(
            stores=self,
            file_ids=file_ids,
            folder_id=folder_id,
            search_type=search_type,
            doc_kinds=doc_kinds,
            use_hybrid=bool(use_hybrid and RAG_SETTINGS.hybrid_enabled),
        )
        return PrimaryGrandchildrenRetriever(
            vectorstore=vectorstore,
            top_k=top_k,
        )

    def get_child_retriever_via_grandchildren(
        self,
        *,
        top_k: int = 8,
        file_ids: Optional[list[str]] = None,
        folder_id: Optional[str] = None,
        search_type: Optional[str] = None,
        doc_kinds: Optional[list[str]] = None,
        use_hybrid: bool = True,
    ) -> ChildViaGrandchildrenRetriever:
        vectorstore = PrimaryGrandchildrenVectorStore(
            stores=self,
            file_ids=file_ids,
            folder_id=folder_id,
            search_type=search_type,
            doc_kinds=doc_kinds,
            use_hybrid=bool(use_hybrid and RAG_SETTINGS.hybrid_enabled),
        )
        return ChildViaGrandchildrenRetriever(
            stores=self,
            vectorstore=vectorstore,
            top_k=int(top_k),
        )

    def get_reranker(self):
        return self.reranker

    async def count_files_by_folder_id(self, folder_id: str) -> tuple[int, list[dict[str, str]]]:
        files_by_id: dict[str, dict[str, str]] = {}
        async for parent_key in self._parents_byte_store.ayield_keys():
            data = (await self._parents_byte_store.amget([parent_key]))[0]
            if not data:
                continue
            try:
                meta = json_utils.loads(data.decode("utf-8"))
            except Exception:
                continue
            if str(meta.get("folder_id") or "") != folder_id:
                continue
            file_id = str(meta.get("file_id") or "").strip()
            if file_id:
                files_by_id[file_id] = {"file_id": file_id, "file_name": str(meta.get("file_name") or "")}

        file_items = [files_by_id[file_id] for file_id in sorted(files_by_id)]
        return len(file_items), file_items

    async def read_parent_summary_hits(self, file_ids: list[str], *, limit: int) -> list[dict[str, Any]]:
        normalized_file_ids = [validate_file_id(file_id) for file_id in file_ids if file_id]
        results: list[dict[str, Any]] = []
        for file_id in normalized_file_ids:
            if len(results) >= limit:
                break
            parent_keys = [
                key async for key in self._parents_byte_store.ayield_keys(prefix=f"{file_id}/")
            ]
            values = await self._parents_byte_store.amget(parent_keys)
            for data in values:
                if not data:
                    continue
                try:
                    meta = json_utils.loads(data.decode("utf-8"))
                except Exception:
                    continue
                if meta.get("file_id") != file_id:
                    continue
                results.append(dict(meta))
                if len(results) >= limit:
                    break

        return results

    async def refresh_indices(self) -> bool:
        if not self.is_elasticsearch:
            return False
        try:
            store = await self._grandchildren_store.get_store()
            client = getattr(store, "client", None)
            if client is not None:
                await client.indices.refresh(index=self.langchain_collection_name)
            return True
        except Exception:
            logger.warning("显式刷新 ES 索引可见性失败", exc_info=True)
            return False

    async def ensure_initialized(self) -> None:
        try:
            await self._grandchildren_store.ensure_initialized()
        except Exception:
            logger.exception(
                "RAG 向量集合初始化失败 | provider={}, host={}, collection={}, embedding_dim={}",
                self.settings.vector_db_provider,
                self.settings.es_host if self.is_elasticsearch else self.settings.milvus_host,
                self.langchain_collection_name,
                self.settings.embedding_dim,
            )
            raise

    async def aclose(self) -> None:
        embedder_close = getattr(self.embedder, "aclose", None)
        if callable(embedder_close):
            await embedder_close()
        reranker_close = getattr(self.reranker, "aclose", None)
        if callable(reranker_close):
            await reranker_close()
        await self._grandchildren_store.aclose()

    def _build_primary_search_kwargs(
        self,
        *,
        top_k: int,
        file_ids: Optional[list[str]] = None,
        folder_id: Optional[str] = None,
        search_type: Optional[str] = None,
        doc_kinds: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"k": int(top_k)}
        if self.is_elasticsearch:
            filters = build_es_filter(
                file_ids=file_ids,
                folder_id=folder_id,
                search_type=search_type,
                doc_kinds=doc_kinds,
            )
        else:
            filters = build_milvus_expr(
                file_ids=file_ids,
                folder_id=folder_id,
                search_type=search_type,
                doc_kinds=doc_kinds,
            )
        if filters:
            kwargs["filter"] = filters
        return kwargs

    async def _delete_minio_children_by_file_id(self, file_id: str) -> None:
        child_keys = [
            key async for key in self._children_byte_store.ayield_keys(prefix=f"{validate_file_id(file_id)}/")
        ]
        if child_keys:
            await self._children_docstore.amdelete(child_keys)

    async def _delete_minio_parents_by_file_id(self, file_id: str) -> None:
        parent_keys = [
            key async for key in self._parents_byte_store.ayield_keys(prefix=f"{validate_file_id(file_id)}/")
        ]
        if parent_keys:
            await self._parents_byte_store.amdelete(parent_keys)

    async def _delete_primary_grandchildren_by_file_id(self, file_id: str) -> None:
        if self.is_elasticsearch:
            store = await self._grandchildren_store.get_store()
            es_store = getattr(store, "_store", None)
            if es_store is None:
                raise RuntimeError("无法获取底层 Elasticsearch store")
            await es_store.client.delete_by_query(
                index=self.langchain_collection_name,
                query={"term": {"metadata.file_id.keyword": file_id}},
                conflicts="proceed",
                refresh=True,
            )
        else:
            await self._grandchildren_store.adelete(filter=build_milvus_expr(file_id=file_id))

    async def _try_upsert_es_grandchildren_with_existing_vectors(
        self,
        chunks: Sequence[GrandchildChunk],
    ) -> int | None:
        vectors = [getattr(chunk, "_existing_vector", None) for chunk in chunks]
        if any(vector is None for vector in vectors):
            return None

        try:
            store = await self._grandchildren_store.get_store()
            es_store = getattr(store, "_store", None)
            if es_store is None:
                return None
            await es_store._create_index_if_not_exists()
            requests = [
                {
                    "_index": self.langchain_collection_name,
                    "_id": chunk.grandchild_id,
                    "_source": {
                        "text": chunk.text,
                        "metadata": _grandchild_to_document(chunk).metadata,
                        "vector": [float(item) for item in vector],
                    },
                }
                for chunk, vector in zip(chunks, vectors)
            ]
            success, errors = await self._run_es_bulk_with_retry(
                es_store.client,
                requests,
                refresh=bool(RAG_SETTINGS.es_bulk_refresh),
                raise_on_error=False,
                chunk_size=max(1, int(RAG_SETTINGS.es_bulk_chunk_size)),
                max_chunk_bytes=int(RAG_SETTINGS.es_bulk_max_chunk_bytes),
                max_retries=int(RAG_SETTINGS.es_bulk_max_retries),
                initial_backoff=float(RAG_SETTINGS.es_bulk_initial_backoff_s),
                max_backoff=float(RAG_SETTINGS.es_bulk_max_backoff_s),
            )
            if errors:
                logger.warning(
                    "复用旧 ES 向量写入新孙切片集合部分失败: success={}, failed={}",
                    success,
                    len(errors),
                )
            return int(success)
        except BulkIndexError:
            logger.exception("复用旧 ES 向量写入新孙切片集合失败")
            raise
        except Exception:
            logger.exception("复用旧 ES 向量写入新孙切片集合失败")
            raise

    async def _run_es_bulk_with_retry(
        self,
        client: Any,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[int, list[Any]]:
        max_retries = max(0, int(RAG_SETTINGS.es_bulk_max_retries))
        initial_backoff = max(0.0, float(RAG_SETTINGS.es_bulk_initial_backoff_s))
        max_backoff = max(initial_backoff, float(RAG_SETTINGS.es_bulk_max_backoff_s))

        for attempt in range(max_retries + 1):
            try:
                return await async_bulk(client, requests, **kwargs)
            except (EsConnectionError, EsConnectionTimeout):
                if attempt >= max_retries:
                    raise
                delay = min(max_backoff, initial_backoff * (2**attempt))
                logger.warning(
                    "ES bulk 写入连接异常，准备重试: attempt={}/{}, delay={}",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    exc_info=True,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("ES bulk 写入重试状态异常")

    async def _clear_minio_store(self, byte_store: AsyncMinioByteStore) -> None:
        keys = [key async for key in byte_store.ayield_keys()]
        if keys:
            await byte_store.amdelete(keys)

    async def _reset_primary_grandchildren(self) -> None:
        if self.is_elasticsearch:
            store = await self._grandchildren_store.get_store()
            await store.adelete(filter={"match_all": {}})
        else:
            store = await self._grandchildren_store.get_store()
            await store.adelete(expr="pk >= ''")


def _get_default_stores() -> RagStores:
    langchain_col = RAG_SETTINGS.vector_langchain_collection
    logger.debug(
        "RAG集合解析: 存储命名空间={}, 新孙集合={}",
        RAG_SETTINGS.rag_storage_collection_name,
        langchain_col,
    )
    return RagStores(
        storage_collection_name=RAG_SETTINGS.rag_storage_collection_name,
        langchain_collection_name=langchain_col,
    )


VectorStoresService = _get_default_stores()
logger.info("向量库存储服务已初始化（LangChain + MinIO）")


async def close_default_stores() -> None:
    await VectorStoresService.aclose()


async def ensure_default_stores_initialized() -> None:
    await VectorStoresService.ensure_initialized()
