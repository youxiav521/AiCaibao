"""Resolve the document scope for content-generation image questions."""

from __future__ import annotations

from typing import Any, Iterable

from src.agents.content_generate_agent.models import (
    ChunkIndexRetrievalTool,
    QuestionTask,
)


def resolve_allowed_file_ids(
    task: QuestionTask,
    typed_items: Iterable[Any],
) -> list[str]:
    """Return file IDs using the same scope semantics as text and table retrieval.

    An explicit ``dataSources[].fileIds`` is a task-level restriction. Without
    it, all files supplied in ``contentList`` are in scope and ``knowledge``
    adds task-specific files.
    """
    explicit_file_ids = _resolve_explicit_file_ids(task)
    if explicit_file_ids:
        return _stable_unique(explicit_file_ids)

    return _stable_unique(
        [
            *(getattr(item, "file_id", "") for item in typed_items),
            *(task.knowledge or []),
        ]
    )


def _resolve_explicit_file_ids(task: QuestionTask) -> list[str]:
    file_ids: list[str] = []
    for data_source in task.get_resolved_data_sources() or []:
        if isinstance(data_source, ChunkIndexRetrievalTool):
            file_ids.extend(data_source.file_ids or [])
    return file_ids


def _stable_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        file_id = str(value or "").strip()
        if not file_id or file_id in seen:
            continue
        seen.add(file_id)
        normalized.append(file_id)
    return normalized
