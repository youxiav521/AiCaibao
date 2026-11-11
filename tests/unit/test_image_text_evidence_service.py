from __future__ import annotations

import asyncio

from src.applications.rag.schemas import BatchQueryItem
from src.applications.content_generate.services import image_text_evidence_service
from src.applications.content_generate.services.image_text_evidence_service import (
    ImageTextEvidenceService,
)


def _result(
    *,
    file_id: str = "file-1",
    element_ids: list[int] | None = None,
    text: str | None = "甲公司持有乙公司100%股权",
    score: int | None = 900,
) -> BatchQueryItem:
    return BatchQueryItem(
        question_id="image-evidence",
        element_ids=[8] if element_ids is None else element_ids,
        file_id=file_id,
        text=text,
        score=score,
    )


def test_search_uses_allowed_file_ids_and_deduplicates(monkeypatch) -> None:
    captured = {}

    async def fake_batch_query(request):
        captured["request"] = request
        return [_result(), _result(), _result(element_ids=[9], text="乙公司持有丙公司50%股权")]

    monkeypatch.setattr(image_text_evidence_service, "run_batch_query", fake_batch_query)

    result = asyncio.run(ImageTextEvidenceService().search("展示乙公司股权图", ["file-1", "file-2"]))

    assert captured["request"].file_ids == ["file-1", "file-2"]
    assert captured["request"].return_child_chunks is True
    assert [(item.file_id, item.element_ids) for item in result] == [
        ("file-1", (8,)),
        ("file-1", (9,)),
    ]


def test_search_drops_empty_text_and_empty_file_id(monkeypatch) -> None:
    async def fake_batch_query(_request):
        return [
            _result(text="  "),
            _result(file_id="", text="有效内容"),
            _result(element_ids=[], text="没有元素"),
        ]

    monkeypatch.setattr(image_text_evidence_service, "run_batch_query", fake_batch_query)

    result = asyncio.run(ImageTextEvidenceService().search("展示乙公司股权图", ["file-1"]))

    assert result == []


def test_search_returns_no_evidence_for_empty_scope() -> None:
    result = asyncio.run(ImageTextEvidenceService().search("展示乙公司股权图", []))

    assert result == []
