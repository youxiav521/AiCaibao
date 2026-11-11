from __future__ import annotations

from typing import Any, Literal, Sequence, TYPE_CHECKING

from pydantic import BaseModel, Field


def build_parent_summary_available_cache_key(file_id: str) -> str:
    return f"rag:parent-summary:{file_id}:available"


def build_parent_summary_unavailable_cache_key(file_id: str) -> str:
    return f"rag:parent-summary:{file_id}:unavailable"


class ParentSummaryCacheRecord(BaseModel):
    parent_id: str
    file_id: str
    text: str
    summary: str
    source_element_ids: list[int] = Field(default_factory=list)
    heading_path: str = ""
    heading_node_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentSummaryCachePayload(BaseModel):
    state: Literal["available", "unavailable"]
    file_id: str
    records: list[ParentSummaryCacheRecord] = Field(default_factory=list)

    @classmethod
    def available(
        cls,
        *,
        file_id: str,
        records: list[ParentSummaryCacheRecord],
    ) -> "ParentSummaryCachePayload":
        return cls(state="available", file_id=file_id, records=records)

    @classmethod
    def unavailable(cls, file_id: str) -> "ParentSummaryCachePayload":
        return cls(state="unavailable", file_id=file_id, records=[])


# ---------------------------------------------------------------------------
# Cache IO helpers
# ---------------------------------------------------------------------------

from src.base.cache_manager import CacheManager
from .summary_service_utils import coerce_int

if TYPE_CHECKING:
    from .parent_summary_service import ParentSummaryRecord

AVAILABLE_TTL = 3600
UNAVAILABLE_TTL = 300


async def load_parent_summary_cache(file_id: str) -> ParentSummaryCachePayload | None:
    """Load cached parent summary for a file (tries available then unavailable)."""
    payload = await CacheManager.get(
        build_parent_summary_available_cache_key(file_id),
        model_type=ParentSummaryCachePayload,
    )
    if payload is not None:
        return payload
    return await CacheManager.get(
        build_parent_summary_unavailable_cache_key(file_id),
        model_type=ParentSummaryCachePayload,
    )


def _to_cache_record(record: "ParentSummaryRecord" | Any) -> ParentSummaryCacheRecord:
    metadata = getattr(record, "metadata", None)
    if metadata is None:
        model_dump = getattr(record, "model_dump", None)
        metadata = model_dump() if callable(model_dump) else {}
    return ParentSummaryCacheRecord(
        parent_id=record.parent_id,
        file_id=record.file_id,
        text=record.text,
        summary=record.summary,
        source_element_ids=list(record.source_element_ids),
        heading_path=record.heading_path,
        heading_node_ids=list(record.heading_node_ids),
        metadata=dict(metadata),
    )


async def store_parent_summary_available_cache(
    file_id: str,
    records: Sequence["ParentSummaryRecord" | Any],
) -> None:
    payload = ParentSummaryCachePayload.available(
        file_id=file_id,
        records=[_to_cache_record(record) for record in records],
    )
    await CacheManager.delete(build_parent_summary_unavailable_cache_key(file_id))
    await CacheManager.set(
        build_parent_summary_available_cache_key(file_id),
        payload,
        ttl=AVAILABLE_TTL,
    )


async def store_parent_summary_unavailable_cache(file_id: str) -> None:
    await CacheManager.delete(build_parent_summary_available_cache_key(file_id))
    await CacheManager.set(
        build_parent_summary_unavailable_cache_key(file_id),
        ParentSummaryCachePayload.unavailable(file_id),
        ttl=UNAVAILABLE_TTL,
    )


async def clear_parent_summary_cache(file_id: str) -> None:
    await CacheManager.delete(build_parent_summary_available_cache_key(file_id))
    await CacheManager.delete(build_parent_summary_unavailable_cache_key(file_id))


async def refresh_parent_summary_cache(
    file_id: str,
    parent_chunks: Sequence[Any],
) -> None:
    from src.applications.rag.schemas import ParentChunk

    records: list[ParentSummaryCacheRecord] = []
    for chunk in parent_chunks:
        if not isinstance(chunk, ParentChunk):
            continue
        if chunk.file_id != file_id:
            continue
        records.append(
            ParentSummaryCacheRecord(
                parent_id=chunk.parent_id,
                file_id=chunk.file_id,
                text=chunk.text,
                summary=chunk.summary,
                source_element_ids=[
                    int_value
                    for value in (chunk.source_element_ids or [])
                    if (int_value := coerce_int(value)) is not None
                ],
                heading_path=str(chunk.heading_path or ""),
                heading_node_ids=[
                    int_value
                    for value in (chunk.heading_node_ids or [])
                    if (int_value := coerce_int(value)) is not None
                ],
                metadata=chunk.model_dump(),
            )
        )

    if records:
        await CacheManager.delete(build_parent_summary_unavailable_cache_key(file_id))
        await CacheManager.set(
            build_parent_summary_available_cache_key(file_id),
            ParentSummaryCachePayload.available(file_id=file_id, records=records),
            ttl=AVAILABLE_TTL,
        )
    else:
        await clear_parent_summary_cache(file_id)
