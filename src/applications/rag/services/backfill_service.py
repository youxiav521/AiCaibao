from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Optional

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_scan
from loguru import logger

from src.utils import json_utils

from ..config.rag_settings import RAG_SETTINGS
from ..gateway.vector_store import parse_es_hosts
from ..schemas import ChildChunk, GrandchildChunk, ParentChunk
from .vectorstore import RagStores, VectorStoresService

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

_GRANDCHILD_SOURCE_FIELDS = [
    "text",
    "parent_id",
    "child_id",
    "grandchild_id",
    "file_id",
    "folder_id",
    "file_name",
    "type",
    "doc_kind",
    "source_element_ids",
    "heading_node_ids",
    "element_id",
    "image_storage_path",
    "table_caption_context_element_ids",
    "split_strategy",
    "table_row_start",
    "table_row_end",
    "heading_path",
    "vector",
    "metadata",
]


@dataclass(frozen=True)
class LegacyEsHit:
    text: str
    metadata: dict[str, Any]


async def backfill_legacy_rag_stores(
    *,
    stores: Optional[RagStores] = None,
    old_parent_collection: str,
    old_child_collection: str | None = None,
    old_grandchild_collection: str | None = None,
    progress_callback: Optional[ProgressCallback] = None,
    es_client: AsyncElasticsearch | None = None,
) -> dict[str, Any]:
    if stores is None:
        stores = VectorStoresService

    result: dict[str, Any] = {
        "source": {
            "type": "elasticsearch",
            "host": RAG_SETTINGS.es_host,
            "collections": {
                "parents": old_parent_collection,
                "children": old_child_collection,
                "grandchildren": old_grandchild_collection,
            },
        },
        "collections": {},
    }

    owns_client = es_client is None
    client = es_client or _create_legacy_es_client()
    try:
        collection_results: dict[str, dict[str, Any]] = {}

        if old_grandchild_collection:
            es_action_writer = getattr(stores, "upsert_grandchild_es_actions", None)
            if bool(getattr(stores, "is_elasticsearch", False)) and callable(es_action_writer):
                grandchild_job = _backfill_grandchildren_actions(
                    client=client,
                    collection=old_grandchild_collection,
                    action_writer=es_action_writer,
                    existing_ids_getter=getattr(stores, "existing_grandchild_ids", None),
                    progress_callback=progress_callback,
                    result=result,
                )
            else:
                grandchild_job = _backfill_chunks(
                    client=client,
                    kind="grandchildren",
                    collection=old_grandchild_collection,
                    id_field="grandchild_id",
                    chunk_factory=_hit_to_grandchild_chunk,
                    writer=stores.upsert_grandchildren,
                    existing_ids_getter=getattr(stores, "existing_grandchild_ids", None),
                    progress_callback=progress_callback,
                    result=result,
                )
            collection_results["grandchildren"] = await grandchild_job

        collection_results["parents"] = await _backfill_chunks(
            client=client,
            kind="parents",
            collection=old_parent_collection,
            id_field="parent_id",
            chunk_factory=_hit_to_parent_chunk,
            writer=stores.upsert_parents,
            existing_ids_getter=getattr(stores, "existing_parent_ids", None),
            progress_callback=progress_callback,
            result=result,
        )

        if old_child_collection:
            collection_results["children"] = await _backfill_chunks(
                client=client,
                kind="children",
                collection=old_child_collection,
                id_field="child_id",
                chunk_factory=_hit_to_child_chunk,
                writer=stores.upsert_children,
                existing_ids_getter=getattr(stores, "existing_child_ids", None),
                progress_callback=progress_callback,
                result=result,
            )

        result["collections"].update(collection_results)
    finally:
        if owns_client:
            await client.close()

    result["scanned"] = sum(int(item["scanned"]) for item in result["collections"].values())
    result["written"] = sum(int(item["written"]) for item in result["collections"].values())
    result["skipped"] = sum(int(item["skipped"]) for item in result["collections"].values())
    result["failed"] = sum(int(item["failed"]) for item in result["collections"].values())
    await stores.refresh_indices()
    return result


def _create_legacy_es_client() -> AsyncElasticsearch:
    kwargs: dict[str, Any] = {
        "request_timeout": int(RAG_SETTINGS.es_request_timeout_s),
        "retry_on_timeout": bool(RAG_SETTINGS.es_retry_on_timeout),
        "max_retries": int(RAG_SETTINGS.es_bulk_max_retries),
    }
    if RAG_SETTINGS.es_username:
        kwargs["basic_auth"] = (RAG_SETTINGS.es_username, RAG_SETTINGS.es_password or "")
    return AsyncElasticsearch(parse_es_hosts(RAG_SETTINGS.es_host), **kwargs)


async def _backfill_chunks(
    *,
    client: AsyncElasticsearch,
    kind: str,
    collection: str,
    id_field: str,
    chunk_factory: Callable[[LegacyEsHit], Any],
    writer: Callable[[list[Any]], Awaitable[int]],
    existing_ids_getter: Callable[[list[str]], Awaitable[set[str]]] | None,
    progress_callback: Optional[ProgressCallback],
    result: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"scanned": 0, "written": 0, "skipped": 0, "failed": 0, "failed_ids": []}
    batch: list[Any] = []
    batch_size = max(1, int(RAG_SETTINGS.rag_backfill_batch_size))
    log_context = f"kind={kind}"

    def existing_lookup_id(item: Any) -> str:
        item_id = str(getattr(item, id_field, "") or "").strip()
        if kind in {"parents", "children"}:
            file_id = str(getattr(item, "file_id", "") or "").strip()
            if file_id and item_id:
                return f"{file_id}/{item_id}"
        return item_id

    async def skip_existing(items: list[Any]) -> list[Any]:
        if existing_ids_getter is None:
            return items
        existing_ids = await existing_ids_getter([existing_lookup_id(item) for item in items])
        if not existing_ids:
            return items
        pending = [item for item in items if existing_lookup_id(item) not in existing_ids]
        summary["skipped"] += len(items) - len(pending)
        return pending

    async def report_progress() -> None:
        if progress_callback is not None:
            result["collections"][kind] = dict(summary)
            await progress_callback(dict(result))

    async def flush_pending() -> None:
        pending = await skip_existing(batch)
        if not pending:
            batch.clear()
            await report_progress()
            return
        await _flush_batch(
            writer=writer,
            batch=pending,
            summary=summary,
            id_getter=lambda chunk: str(getattr(chunk, id_field, "") or "").strip(),
            log_context=log_context,
        )
        batch.clear()
        await report_progress()

    async for hit in _iter_legacy_es_hits(
        client=client,
        collection_name=collection,
    ):
        summary["scanned"] += 1
        metadata = hit.metadata or {}
        if not str(metadata.get(id_field) or _legacy_record_id(metadata)).strip():
            summary["skipped"] += 1
            continue
        try:
            batch.append(chunk_factory(hit))
        except Exception:
            logger.exception(
                "旧 ES RAG 回填转换失败: kind={} collection={}",
                kind,
                collection,
            )
            summary["failed"] += 1
            continue
        if len(batch) >= batch_size:
            await flush_pending()

    if batch:
        await flush_pending()
    return summary


async def _iter_legacy_es_hits(
    *,
    client: AsyncElasticsearch,
    collection_name: str,
) -> AsyncIterator[LegacyEsHit]:
    query = {"_source": True, "query": {"match_all": {}}}
    async for raw_hit in async_scan(
        client,
        index=collection_name,
        query=query,
        size=max(1, int(RAG_SETTINGS.rag_backfill_batch_size)),
        request_timeout=int(RAG_SETTINGS.es_request_timeout_s),
    ):
        yield _legacy_hit_from_es(raw_hit)


async def _iter_legacy_es_raw_hits(
    *,
    client: AsyncElasticsearch,
    collection_name: str,
    source_fields: list[str],
) -> AsyncIterator[dict[str, Any]]:
    async for raw_hit in async_scan(
        client,
        index=collection_name,
        query={"query": {"match_all": {}}},
        _source=source_fields,
        size=max(
            max(1, int(RAG_SETTINGS.rag_backfill_batch_size)),
            int(RAG_SETTINGS.es_bulk_chunk_size),
            200,
        ),
        request_timeout=int(RAG_SETTINGS.es_request_timeout_s),
    ):
        yield raw_hit


async def _backfill_grandchildren_actions(
    *,
    client: AsyncElasticsearch,
    collection: str,
    action_writer: Callable[[list[dict[str, Any]]], Awaitable[int]],
    existing_ids_getter: Callable[[list[str]], Awaitable[set[str]]] | None,
    progress_callback: Optional[ProgressCallback],
    result: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"scanned": 0, "written": 0, "skipped": 0, "failed": 0, "failed_ids": []}
    batch: list[dict[str, Any]] = []
    read_size = max(
        max(1, int(RAG_SETTINGS.rag_backfill_batch_size)),
        int(RAG_SETTINGS.es_bulk_chunk_size),
        200,
    )

    async def skip_existing(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if existing_ids_getter is None:
            return actions
        existing_ids = await existing_ids_getter([str(action.get("_id") or "").strip() for action in actions])
        if not existing_ids:
            return actions
        pending = [action for action in actions if str(action.get("_id") or "").strip() not in existing_ids]
        summary["skipped"] += len(actions) - len(pending)
        return pending

    async def report_progress() -> None:
        if progress_callback is not None:
            result["collections"]["grandchildren"] = dict(summary)
            await progress_callback(dict(result))

    async def flush_pending() -> None:
        pending = await skip_existing(batch)
        if not pending:
            batch.clear()
            await report_progress()
            return
        await _flush_batch(
            writer=action_writer,
            batch=pending,
            summary=summary,
            id_getter=lambda action: str(action.get("_id") or "").strip(),
            log_context="kind=grandchildren",
        )
        batch.clear()
        await report_progress()

    async for raw_hit in _iter_legacy_es_raw_hits(
        client=client,
        collection_name=collection,
        source_fields=_GRANDCHILD_SOURCE_FIELDS,
    ):
        summary["scanned"] += 1
        try:
            action = _legacy_grandchild_hit_to_es_action(raw_hit)
        except Exception:
            summary["failed"] += 1
            summary["failed_ids"].append(str(raw_hit.get("_id") or ""))
            logger.exception("旧 ES RAG 孙切片 action 转换失败: collection={}", collection)
            continue
        if action is None:
            summary["skipped"] += 1
            continue
        batch.append(action)
        if len(batch) >= read_size:
            await flush_pending()

    if batch:
        await flush_pending()
    return summary


def _legacy_grandchild_hit_to_es_action(hit: dict[str, Any]) -> dict[str, Any] | None:
    source = hit.get("_source") if isinstance(hit.get("_source"), dict) else {}
    nested_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    metadata = dict(nested_metadata)
    for key, value in source.items():
        if key in {"metadata", "text", "vector"}:
            continue
        metadata[key] = value

    grandchild_id = str(
        metadata.get("grandchild_id")
        or source.get("grandchild_id")
        or hit.get("_id")
        or ""
    ).strip()
    if not grandchild_id:
        return None

    metadata["grandchild_id"] = grandchild_id
    metadata.setdefault("child_id", str(metadata.get("child_id") or ""))
    metadata.setdefault("parent_id", str(metadata.get("parent_id") or ""))
    metadata.setdefault("file_id", str(metadata.get("file_id") or ""))
    metadata["source_element_ids"] = _normalize_list(metadata.get("source_element_ids"))
    metadata["heading_node_ids"] = _normalize_list(metadata.get("heading_node_ids"))
    metadata["table_caption_context_element_ids"] = _normalize_list(
        metadata.get("table_caption_context_element_ids")
    )
    metadata["heading_path"] = _normalize_heading_path(metadata.get("heading_path"))

    vector = source.get("vector") or metadata.get("vector")
    metadata.pop("vector", None)
    if not isinstance(vector, list) or not vector:
        return None

    text = str(source.get("text") or metadata.get("text") or "")
    metadata.pop("text", None)
    return {
        "_id": grandchild_id,
        "_source": {
            "text": text,
            "metadata": metadata,
            "vector": [float(item) for item in vector],
        },
    }

def _legacy_hit_from_es(hit: dict[str, Any]) -> LegacyEsHit:
    source = hit.get("_source") if isinstance(hit.get("_source"), dict) else {}
    nested_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    metadata = dict(nested_metadata)
    for key, value in source.items():
        if key in {"metadata", "text"}:
            continue
        metadata[key] = value
    if hit.get("_id") is not None:
        metadata.setdefault("id", str(hit.get("_id")))
    text = str(source.get("text") or metadata.get("text") or "")
    return LegacyEsHit(text=text, metadata=metadata)


async def _flush_batch(
    *,
    writer: Callable[[list[Any]], Awaitable[int]],
    batch: list[Any],
    summary: dict[str, Any],
    id_getter: Callable[[Any], str],
    log_context: str,
) -> None:
    try:
        written = int(await writer(batch))
    except Exception:
        logger.exception("旧 ES RAG 批量写入失败，降级逐条写入: {}", log_context)
        await _flush_batch_one_by_one(
            writer=writer,
            batch=batch,
            summary=summary,
            id_getter=id_getter,
            log_context=log_context,
        )
        return
    if written == len(batch):
        summary["written"] += written
        return
    logger.warning(
        "旧 ES RAG 批量写入数量不完整，降级逐条确认: {} written={} expected={}",
        log_context,
        written,
        len(batch),
    )
    await _flush_batch_one_by_one(
        writer=writer,
        batch=batch,
        summary=summary,
        id_getter=id_getter,
        log_context=log_context,
    )


async def _flush_batch_one_by_one(
    *,
    writer: Callable[[list[Any]], Awaitable[int]],
    batch: list[Any],
    summary: dict[str, Any],
    id_getter: Callable[[Any], str],
    log_context: str,
) -> None:
    failed_ids = summary["failed_ids"]
    for item in batch:
        item_id = id_getter(item)
        try:
            written = int(await writer([item]))
        except Exception:
            summary["failed"] += 1
            failed_ids.append(item_id)
            logger.exception("旧 ES RAG 切片写入失败，已跳过: {} chunk_id={}", log_context, item_id)
            continue
        if written == 1:
            summary["written"] += 1
            continue
        summary["failed"] += 1
        failed_ids.append(item_id)
        logger.warning(
            "旧 ES RAG 切片写入返回数量异常，已跳过: {} chunk_id={} written={}",
            log_context,
            item_id,
            written,
        )


def _hit_to_parent_chunk(hit: LegacyEsHit) -> ParentChunk:
    metadata = dict(hit.metadata or {})
    return ParentChunk(**_chunk_payload(metadata, hit.text, id_defaults={"parent_id": _legacy_record_id(metadata)}))


def _hit_to_child_chunk(hit: LegacyEsHit) -> ChildChunk:
    metadata = dict(hit.metadata or {})
    payload = _chunk_payload(metadata, hit.text, id_defaults={"child_id": _legacy_record_id(metadata)})
    payload.pop("summary", None)
    return ChildChunk(**payload)


def _hit_to_grandchild_chunk(hit: LegacyEsHit) -> GrandchildChunk:
    metadata = dict(hit.metadata or {})
    payload = _chunk_payload(metadata, hit.text, id_defaults={"grandchild_id": _legacy_record_id(metadata)})
    payload.pop("summary", None)
    chunk = GrandchildChunk(**payload)
    vector = metadata.get("vector")
    if isinstance(vector, list) and vector:
        setattr(chunk, "_existing_vector", vector)
    return chunk


def _chunk_payload(metadata: dict[str, Any], text: str, *, id_defaults: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "text": text,
        "file_id": str(metadata.get("file_id") or ""),
        "folder_id": str(metadata.get("folder_id") or ""),
        "file_name": str(metadata.get("file_name") or ""),
        "type": str(metadata.get("type") or ""),
        "doc_kind": str(metadata.get("doc_kind") or "chunk"),
        "parent_id": str(metadata.get("parent_id") or ""),
        "child_id": str(metadata.get("child_id") or ""),
        "grandchild_id": str(metadata.get("grandchild_id") or ""),
        "summary": str(metadata.get("summary") or ""),
        "source_element_ids": _normalize_list(metadata.get("source_element_ids")),
        "heading_node_ids": _normalize_list(metadata.get("heading_node_ids")),
        "element_id": metadata.get("element_id"),
        "image_storage_path": str(metadata.get("image_storage_path") or ""),
        "table_caption_context_element_ids": _normalize_list(metadata.get("table_caption_context_element_ids")),
        "split_strategy": str(metadata.get("split_strategy") or ""),
        "table_row_start": metadata.get("table_row_start"),
        "table_row_end": metadata.get("table_row_end"),
        "heading_path": _normalize_heading_path(metadata.get("heading_path")),
    }
    for field_name, value in id_defaults.items():
        if not payload.get(field_name):
            payload[field_name] = str(value or "")
    return payload


def _legacy_record_id(metadata: dict[str, Any]) -> str:
    return str(metadata.get("id") or metadata.get("_id") or "").strip()


def _normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item is not None]
    if isinstance(value, (tuple, set)):
        return [item for item in value if item is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json_utils.loads(text)
        except Exception:
            return [text]
        return _normalize_list(parsed)
    return [value]


def _normalize_heading_path(value: Any) -> str:
    if isinstance(value, list):
        return "/".join(str(item).strip() for item in value if str(item or "").strip())
    return str(value or "").strip()
