from collections.abc import Sequence
from typing import Tuple, Optional
from src.applications.rag.config.rag_settings import RAG_SETTINGS
from src.applications.rag.schemas import (
    ChildChunk,
    GrandchildChunk,
    ParentChunk,
)
from ..vectorstore import VectorStoresService
from loguru import logger


async def _rollback_index_write(file_id: str, *, reason: str) -> None:
    try:
        deleted = await VectorStoresService.delete_by_file_id(file_id)
        if not deleted:
            logger.warning(
                "索引失败后回滚未完全成功: file_id={}, reason={}", file_id, reason
            )
    except Exception as exc:
        logger.warning(
            "索引失败后回滚异常: file_id={}, reason={}, error={}", file_id, reason, exc
        )


async def write_index_payloads(
    file_id: str,
    parents: Optional[Sequence[ParentChunk]] = None,
    children: Optional[Sequence[ChildChunk]] = None,
    grandchildren: Optional[Sequence[GrandchildChunk]] = None,
) -> Tuple[list[ParentChunk], list[ChildChunk], list[GrandchildChunk]]:
    normalized_parents = list(parents or [])
    normalized_children = list(children or [])
    normalized_grandchildren = list(grandchildren or [])

    for doc in normalized_parents:
        doc.doc_kind = "chunk"
    for doc in normalized_children:
        doc.doc_kind = "chunk"
    for doc in normalized_grandchildren:
        doc.doc_kind = doc.doc_kind or "chunk"

    parent_documents = normalized_parents
    child_documents = normalized_children
    grandchild_documents = normalized_grandchildren

    def _write_diagnostics(collection_name: str, expected_count: int) -> str:
        provider = RAG_SETTINGS.vector_db_provider
        base = f"集合={collection_name}, 期望写入={expected_count}, 向量库={provider or '未设置'}"
        if provider not in {"es", "elasticsearch"}:
            return base
        return (
            f"{base}, 超时秒={RAG_SETTINGS.es_request_timeout_s}, "
            f"分块大小={RAG_SETTINGS.es_bulk_chunk_size}, "
            f"分块字节上限={RAG_SETTINGS.es_bulk_max_chunk_bytes}, "
            f"刷新索引={RAG_SETTINGS.es_bulk_refresh}, "
            f"最大重试次数={RAG_SETTINGS.es_bulk_max_retries}"
        )

    if parent_documents:
        upserted_parents = await VectorStoresService.upsert_parents(parent_documents)
        if upserted_parents != len(parent_documents):
            reason = f"父切片入库失败: file_id={file_id}"
            logger.error(
                "父切片写入向量库失败: file_id={}, 实际写入={}, {}, 请查看vectorstore服务日志获取具体错误原因",
                file_id,
                upserted_parents,
                _write_diagnostics(VectorStoresService.storage_collection_name, len(parent_documents)),
            )
            await _rollback_index_write(file_id, reason=reason)
            raise RuntimeError(reason)

    upserted_children = 0
    if child_documents:
        upserted_children = await VectorStoresService.upsert_children(child_documents)
        if upserted_children != len(child_documents):
            reason = f"子切片入库失败: file_id={file_id}"
            logger.error(
                "子切片写入向量库失败: file_id={}, 实际写入={}, {}, 请查看vectorstore服务日志获取具体错误原因",
                file_id,
                upserted_children,
                _write_diagnostics(VectorStoresService.storage_collection_name, len(child_documents)),
            )
            await _rollback_index_write(file_id, reason=reason)
            raise RuntimeError(reason)

    if grandchild_documents:
        upserted_grandchildren = await VectorStoresService.upsert_grandchildren(
            grandchild_documents
        )
        if upserted_grandchildren != len(grandchild_documents):
            reason = f"孙切片入库失败: file_id={file_id}"
            logger.error(
                "孙切片写入向量库失败: file_id={}, 实际写入={}, {}, 请查看vectorstore服务日志获取具体错误原因",
                file_id,
                upserted_grandchildren,
                _write_diagnostics(VectorStoresService.langchain_collection_name, len(grandchild_documents)),
            )
            await _rollback_index_write(file_id, reason=reason)
            raise RuntimeError(reason)
        
    return parent_documents, child_documents, grandchild_documents
