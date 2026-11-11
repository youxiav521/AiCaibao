from __future__ import annotations

import asyncio

from src.agents.content_generate_agent.models import QuestionTask
from src.applications.rag.services.image_query_service import ImageSearchCandidate


def _task(
    description: str,
    *,
    file_ids: list[str] | None = None,
    prompt: str | None = None,
    retrieval_query: str | None = None,
) -> QuestionTask:
    data_sources = None
    if file_ids is not None:
        data_sources = [{"type": "chunk_index_retrieval", "file_ids": file_ids}]
    return QuestionTask(
        id="q-image",
        description=description,
        type="image",
        data_sources=data_sources,
        prompt=prompt,
        retrieval_query=retrieval_query,
    )


def _candidate(score: float) -> ImageSearchCandidate:
    return ImageSearchCandidate(
        file_id="file-1",
        element_id=18,
        image_storage_path="file-1/images/18.png",
        score=score,
        summary="上海农商银行股权结构图",
    )


def test_high_score_original_image_becomes_answer_text(monkeypatch):
    from src.applications.content_generate.services.image_answer_service import (
        ImageAnswerService,
        ImageRelevanceDecision,
    )

    service = ImageAnswerService(min_score=0.80)

    async def search(*args, **kwargs):
        return [_candidate(0.91)]

    async def load(*args, **kwargs):
        return "data:image/png;base64,aGVsbG8="

    async def judge(*args, **kwargs):
        return ImageRelevanceDecision(
            selected_element_id=18,
            confidence=0.91,
            reason="主体与图类型均匹配",
        )

    monkeypatch.setattr(service, "_search", search)
    monkeypatch.setattr(service, "_load_data_uri", load)
    monkeypatch.setattr(service, "_judge_candidates", judge)
    answer = asyncio.run(service.answer(_task("展示上海农商银行股权结构图", file_ids=["file-1"]), []))

    assert answer.is_answerable is True
    assert answer.answer_text == "data:image/png;base64,aGVsbG8="
    assert answer.citations[0].file_id == "file-1"
    assert answer.citations[0].element_ids == [18]


def test_unmatched_image_uses_text_evidence_graph(monkeypatch):
    from src.applications.content_generate.services.image_answer_service import (
        ImageAnswerService,
        ImageRelevanceDecision,
    )
    from src.applications.content_generate.services.image_text_evidence_service import TextEvidence
    from src.applications.content_generate.services.relationship_graph_service import RelationshipGraphResult
    from src.agents.content_generate_agent.models import Citation

    service = ImageAnswerService(min_score=0.80)

    async def search(*args, **kwargs):
        return [_candidate(0.64)]

    async def judge(*args, **kwargs):
        return ImageRelevanceDecision(
            selected_element_id=None,
            confidence=0.1,
            reason="无匹配图片",
        )

    async def search_text_evidence(*args, **kwargs):
        return [TextEvidence(
            file_id="file-1",
            element_ids=(8,),
            text="甲公司持有乙公司100%股权。",
            score=900,
        )]

    async def generate_graph(*args, **kwargs):
        return RelationshipGraphResult(
            data_uri="data:image/svg+xml;base64,c3Zn",
            citations=[Citation(id=1, file_id="file-1", element_ids=[8], quote="甲公司持有乙公司100%股权。")],
            mermaid_code="flowchart TB",
        )

    monkeypatch.setattr(service, "_search", search)
    monkeypatch.setattr(service, "_judge_candidates", judge)
    monkeypatch.setattr(service, "_search_text_evidence", search_text_evidence)
    monkeypatch.setattr(service, "_generate_graph", generate_graph)
    answer = asyncio.run(service.answer(_task("生成上海农商银行股权结构图", file_ids=["file-1"]), []))

    assert answer.is_answerable is True
    assert answer.answer_text == "data:image/svg+xml;base64,c3Zn"
    assert answer.citations[0].element_ids == [8]


def test_low_score_non_relation_question_is_unanswerable(monkeypatch):
    from src.applications.content_generate.services.image_answer_service import ImageAnswerService

    service = ImageAnswerService(min_score=0.80)

    async def search(*args, **kwargs):
        return []

    async def no_evidence(*args, **kwargs):
        return []

    monkeypatch.setattr(service, "_search", search)
    monkeypatch.setattr(service, "_search_text_evidence", no_evidence)
    answer = asyncio.run(service.answer(_task("给我一张办公室照片", file_ids=["file-1"]), []))

    assert answer.is_answerable is False
    assert answer.answer_text


def test_unmatched_ownership_chart_without_text_evidence_is_not_generated(monkeypatch):
    from src.applications.content_generate.services.image_answer_service import ImageAnswerService

    service = ImageAnswerService(min_score=0.85)
    generated = {"called": False}

    async def search(*args, **kwargs):
        return []

    async def no_evidence(*args, **kwargs):
        return []

    async def generate_graph(*args, **kwargs):
        generated["called"] = True
        return None

    monkeypatch.setattr(service, "_search", search)
    monkeypatch.setattr(service, "_search_text_evidence", no_evidence)
    monkeypatch.setattr(service, "_generate_graph", generate_graph)

    answer = asyncio.run(service.answer(_task("请展示上海农商银行股权结构图", file_ids=["file-1"]), []))

    assert answer.is_answerable is False
    assert generated["called"] is False


def test_image_answer_never_searches_across_reports(monkeypatch):
    from src.applications.content_generate.services.image_answer_service import ImageAnswerService

    service = ImageAnswerService(min_score=0.80)
    captured = {}

    async def search(*, question, file_ids):
        captured["question"] = question
        captured["file_ids"] = file_ids
        return []

    monkeypatch.setattr(service, "_search", search)
    answer = asyncio.run(service.answer(_task("给我一张办公室照片", file_ids=["file-1"]), []))

    assert answer.is_answerable is False
    assert captured["file_ids"] == ["file-1"]


def test_internal_dependency_id_uses_prompt_as_image_retrieval_query(monkeypatch):
    """下游依赖题的 description 是内部 ID，不能拿它检索报告。"""
    from src.applications.content_generate.services.image_answer_service import ImageAnswerService

    service = ImageAnswerService(min_score=0.80)
    captured = {}

    async def search(*, question, file_ids):
        captured["question"] = question
        captured["file_ids"] = file_ids
        return []

    monkeypatch.setattr(service, "_search", search)
    prompt = "参考上下文回答：比亚迪股份有限公司前十大股东及其持股比例，并以图片形式返回。"
    answer = asyncio.run(
        service.answer(
            _task(
                "q_4#lq_1[0]",
                file_ids=["file-1"],
                prompt=prompt,
            ),
            [],
        )
    )

    assert answer.is_answerable is False
    assert captured["file_ids"] == ["file-1"]
    assert captured["question"] == prompt


def test_explicit_retrieval_query_has_priority_over_dependency_prompt(monkeypatch):
    from src.applications.content_generate.services.image_answer_service import ImageAnswerService

    service = ImageAnswerService(min_score=0.80)
    captured = {}

    async def search(*, question, file_ids):
        captured["question"] = question
        return []

    monkeypatch.setattr(service, "_search", search)
    answer = asyncio.run(
        service.answer(
            _task(
                "q_4#lq_1[0]",
                file_ids=["file-1"],
                prompt="不应使用的上下文",
                retrieval_query="比亚迪股东持股结构图",
            ),
            [],
        )
    )

    assert answer.is_answerable is False
    assert captured["question"] == "比亚迪股东持股结构图"


def test_raw_vector_score_cannot_bypass_relevance_confidence_threshold(monkeypatch):
    """ES 原始排序分再高，也必须通过 0~1 的相关性置信度判定。"""
    from src.applications.content_generate.services.image_answer_service import (
        ImageAnswerService,
        ImageRelevanceDecision,
    )

    service = ImageAnswerService(min_score=0.85)

    async def search(*args, **kwargs):
        return [_candidate(35.07547)]

    async def judge(*args, **kwargs):
        return ImageRelevanceDecision(
            selected_element_id=None,
            confidence=0.02,
            reason="候选图片未出现上海农商银行",
        )

    async def generate(*args, **kwargs):
        return "data:image/svg+xml;base64,c3Zn"

    async def no_evidence(*args, **kwargs):
        return []

    monkeypatch.setattr(service, "_search", search)
    monkeypatch.setattr(service, "_judge_candidates", judge)
    monkeypatch.setattr(service, "_search_text_evidence", no_evidence)

    answer = asyncio.run(service.answer(_task("生成上海农商银行股权结构图", file_ids=["file-1"]), []))

    assert answer.is_answerable is False


def test_original_image_over_size_limit_is_not_returned(monkeypatch):
    from src.applications.content_generate.services.image_answer_service import (
        ImageAnswerService,
        ImageRelevanceDecision,
    )

    service = ImageAnswerService(min_score=0.80, max_data_uri_chars=32)

    async def search(*args, **kwargs):
        return [_candidate(0.91)]

    async def load(*args, **kwargs):
        return "data:image/png;base64," + "a" * 100

    async def judge(*args, **kwargs):
        return ImageRelevanceDecision(selected_element_id=18, confidence=0.91, reason="匹配")

    async def no_evidence(*args, **kwargs):
        return []

    monkeypatch.setattr(service, "_search", search)
    monkeypatch.setattr(service, "_load_data_uri", load)
    monkeypatch.setattr(service, "_judge_candidates", judge)
    monkeypatch.setattr(service, "_search_text_evidence", no_evidence)

    answer = asyncio.run(service.answer(_task("展示上海农商银行股权结构图", file_ids=["file-1"]), []))

    assert answer.is_answerable is False
    assert "图片过大" in answer.answer_text


def test_relevance_judge_prompt_explicitly_requests_json():
    from src.applications.content_generate.services.image_answer_service import ImageAnswerService

    prompt = ImageAnswerService._build_relevance_prompt(
        "请展示上海农商银行股权结构图",
        [_candidate(0.91)],
    )

    assert "JSON" in prompt


def test_generic_description_relevance_prompt_does_not_require_a_named_entity():
    from src.applications.content_generate.services.image_answer_service import ImageAnswerService

    prompt = ImageAnswerService._build_relevance_prompt(
        "企业股权架构说明",
        [_candidate(0.91)],
    )

    assert "when the question names one" in prompt.lower()
    assert "requested entity, image type, and relationship" not in prompt


def test_relevance_decision_allows_missing_optional_reason():
    from src.applications.content_generate.services.image_answer_service import ImageRelevanceDecision

    decision = ImageRelevanceDecision.model_validate(
        {"selected_element_id": None, "confidence": 0.02}
    )

    assert decision.reason == ""


def test_image_event_uses_existing_answer_text_contract():
    from src.agents.content_generate_agent.models import ContentGenerateResponseEvent, FinalAnswer

    event = ContentGenerateResponseEvent(
        question_id="q-image",
        question_text="展示股权结构图",
        question_type="image",
        answer=FinalAnswer(
            is_answerable=True,
            answer_text="data:image/svg+xml;base64,c3Zn",
            citations=[],
        ),
    )
    payload = event.model_dump(by_alias=True)

    assert payload["answer"]["answerText"].startswith("data:image/svg+xml;base64,")
    assert "imageUrl" not in payload["answer"]
    assert "imageBase64" not in payload["answer"]
