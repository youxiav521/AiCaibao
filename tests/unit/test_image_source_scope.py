from __future__ import annotations

from types import SimpleNamespace

from src.agents.content_generate_agent.models import QuestionTask
from src.applications.content_generate.services.image_source_scope import (
    resolve_allowed_file_ids,
)


def _task(
    *,
    file_ids: list[str] | None = None,
    knowledge: list[str] | None = None,
) -> QuestionTask:
    data_sources = None
    if file_ids is not None:
        data_sources = [{"type": "chunk_index_retrieval", "file_ids": file_ids}]
    return QuestionTask(
        id="q-image",
        description="展示股权结构图",
        type="image",
        data_sources=data_sources,
        knowledge=knowledge,
    )


def _item(file_id: str) -> SimpleNamespace:
    return SimpleNamespace(file_id=file_id)


def test_explicit_file_ids_restrict_image_scope() -> None:
    task = _task(file_ids=["file-2", "file-1", "file-2"])

    actual = resolve_allowed_file_ids(task, [_item("file-1"), _item("file-3")])

    assert actual == ["file-2", "file-1"]


def test_default_scope_uses_all_content_and_knowledge() -> None:
    task = _task(knowledge=["file-3", "file-2", " "])

    actual = resolve_allowed_file_ids(task, [_item("file-1"), _item("file-2")])

    assert actual == ["file-1", "file-2", "file-3"]


def test_explicit_scope_does_not_expand_with_knowledge() -> None:
    task = _task(file_ids=["file-2"], knowledge=["file-3"])

    actual = resolve_allowed_file_ids(task, [_item("file-1")])

    assert actual == ["file-2"]
