"""Original-image retrieval and evidence-backed graph answers for ``image`` tasks."""

from __future__ import annotations

import re
from typing import Any, Iterable

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.content_generate_agent.models import Citation, FinalAnswer, QuestionTask
from src.applications.rag.services.image_query_service import (
    ImageSearchCandidate,
    load_image_as_data_uri,
    search_image_candidates,
)
from .evidence_graph_service import EvidenceGraphResult, EvidenceGraphService
from .image_source_scope import resolve_allowed_file_ids
from .image_text_evidence_service import ImageTextEvidenceService, TextEvidence


DEFAULT_IMAGE_RETRIEVAL_MIN_SCORE = 0.85
DEFAULT_MAX_DATA_URI_CHARS = 7 * 1024 * 1024
_IMAGE_CANDIDATE_LIMIT = 5
_IMAGE_RELEVANCE_MODEL_KEY = "image_summary"
_DEPENDENCY_TASK_ID_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*#[A-Za-z][A-Za-z0-9_]*(?:\[\d+\])?$"
)
_TEMPLATE_IMAGE_DESCRIPTION_PATTERN = re.compile(
    r"^(?:申请人|本公司|企业|公司)?(?:股东|股权)(?:结构|架构)图(?:内容描述)?$"
)
_ORGANIZATION_QUERY_KEYWORDS = (
    "组织架构",
    "组织结构",
    "组织机构",
    "组织图",
    "部门架构",
    "部门设置",
    "治理架构",
    "治理结构",
    "下属企业",
)
_OWNERSHIP_QUERY_KEYWORDS = ("股权", "股东", "持股", "控股", "实际控制人")
_ORGANIZATION_IMAGE_SUMMARY_KEYWORDS = (
    "组织架构",
    "组织结构",
    "治理结构",
    "部门架构",
    "董事会",
    "监事会",
    "总裁",
    "事业部",
    "职能部门",
    "研究院",
    "汇报",
    "下设",
)
_OWNERSHIP_IMAGE_SUMMARY_KEYWORDS = (
    "股权结构",
    "股东结构",
    "持股",
    "持股比例",
    "实际控制人",
    "股权穿透",
)


class ImageRelevanceDecision(BaseModel):
    """The strict relevance decision for a retrieved source image."""

    selected_element_id: int | None = Field(
        default=None,
        description="The matching candidate element ID, or null when there is no match.",
    )
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(default="", description="Short audit reason.")


class ImageAnswerService:
    """Return either an original image or a graph supported by report text evidence."""

    def __init__(
        self,
        min_score: float = DEFAULT_IMAGE_RETRIEVAL_MIN_SCORE,
        max_data_uri_chars: int = DEFAULT_MAX_DATA_URI_CHARS,
    ) -> None:
        if not 0 <= min_score <= 1:
            raise ValueError("图片检索相似度阈值必须在 0 到 1 之间")
        if max_data_uri_chars <= 0:
            raise ValueError("图片 Data URI 长度上限必须大于 0")
        self.min_score = min_score
        self.max_data_uri_chars = max_data_uri_chars

    async def answer(self, task: QuestionTask, typed_items: list[Any]) -> FinalAnswer:
        """Use the text/table retrieval scope for original images and graph evidence."""
        file_ids = resolve_allowed_file_ids(task, typed_items)
        question = self._resolve_retrieval_question(task)
        intent_question = self._resolve_intent_question(task)
        if not file_ids:
            return self._unanswerable("未解析到可检索的资料范围，无法生成图片答案。")

        original_image_too_large = False
        try:
            candidates = await self._search(question=question, file_ids=file_ids)
        except Exception as exc:
            logger.exception("图片题原图检索失败 | question_id={} error={}", task.id, exc)
            candidates = []

        selected = await self._select_original_image(task.id, intent_question, candidates)
        if selected is not None:
            original_answer, original_image_too_large = await self._try_original_image(
                task.id,
                selected,
            )
            if original_answer is not None:
                return original_answer

        try:
            evidence = await self._search_text_evidence(question, file_ids)
            enriched_question = _build_entity_enriched_query(question, evidence)
            if enriched_question:
                logger.info(
                    "图片题使用报告主体扩展原图检索 | question_id={} query={}",
                    task.id,
                    enriched_question,
                )
                enriched_candidates = await self._search(
                    question=enriched_question,
                    file_ids=file_ids,
                )
                enriched_selected = await self._select_original_image(
                    task.id,
                    intent_question,
                    enriched_candidates,
                )
                if enriched_selected is not None:
                    original_answer, too_large = await self._try_original_image(
                        task.id,
                        enriched_selected,
                    )
                    original_image_too_large = original_image_too_large or too_large
                    if original_answer is not None:
                        return original_answer
            graph = await self._generate_graph(intent_question, evidence) if evidence else None
        except Exception as exc:
            logger.exception("图片题文本证据图生成失败 | question_id={} error={}", task.id, exc)
            graph = None

        if graph is not None:
            return FinalAnswer(
                is_answerable=True,
                answer_text=graph.data_uri,
                citations=graph.citations,
            )
        if original_image_too_large:
            return self._unanswerable("命中的原图图片过大，且允许资料范围内未找到可核验的关系信息。")
        return self._unanswerable("允许资料范围内未找到匹配原图或可核验的关系信息。")

    @staticmethod
    def _resolve_retrieval_question(task: QuestionTask) -> str:
        """Choose a retrieval query without dropping useful template prompt context."""
        retrieval_query = (task.retrieval_query or "").strip()
        description = (task.description or "").strip()
        prompt = (task.prompt or "").strip()
        if retrieval_query:
            configured_question = "\n".join(part for part in (description, prompt) if part)
            if (
                _is_organization_question(configured_question)
                and _is_ownership_question(retrieval_query)
                and not _is_organization_question(retrieval_query)
            ):
                logger.warning(
                    "图片题 retrievalQuery 与组织架构配置冲突，回退到配置问题 | retrieval_query={}",
                    retrieval_query,
                )
                return configured_question
            return retrieval_query

        if _DEPENDENCY_TASK_ID_PATTERN.fullmatch(description) and prompt:
            return prompt
        if prompt and (
            _TEMPLATE_IMAGE_DESCRIPTION_PATTERN.fullmatch(description)
            or _is_organization_question(description)
        ):
            return f"{description}\n{prompt}"
        return description

    @staticmethod
    def _resolve_intent_question(task: QuestionTask) -> str:
        """Keep the configured image type stable even when a retrieval query is supplied."""
        parts = [task.description, task.prompt, task.retrieval_query]
        return "\n".join(str(part).strip() for part in parts if str(part or "").strip())

    async def _search(
        self,
        *,
        question: str,
        file_ids: list[str],
    ) -> list[ImageSearchCandidate]:
        return await search_image_candidates(
            question,
            top_k=_IMAGE_CANDIDATE_LIMIT,
            file_ids=file_ids,
        )

    async def _load_data_uri(self, candidate: ImageSearchCandidate) -> str:
        return await load_image_as_data_uri(candidate)

    async def _search_text_evidence(
        self,
        question: str,
        file_ids: list[str],
    ) -> list[TextEvidence]:
        service = ImageTextEvidenceService()
        if _is_organization_question(question):
            return await service.search_many(
                _build_organization_evidence_queries(question),
                file_ids,
            )
        return await service.search(question, file_ids)

    async def _generate_graph(
        self,
        question: str,
        evidence: list[TextEvidence],
    ) -> EvidenceGraphResult | None:
        logger.info("图片题进入证据关系图生成 | question={}", question)
        return await EvidenceGraphService().generate(question, evidence)

    async def _select_original_image(
        self,
        question_id: str,
        question: str,
        candidates: list[ImageSearchCandidate],
    ) -> ImageSearchCandidate | None:
        if not candidates:
            return None
        logger.info(
            "图片题召回候选 | question_id={} candidates={} raw_scores={}",
            question_id,
            [candidate.element_id for candidate in candidates],
            [candidate.score for candidate in candidates],
        )
        try:
            decision = await self._judge_candidates(question, candidates)
        except Exception as exc:
            logger.exception("图片题相关性判定失败，将不返回未经确认的原图 | question_id={} error={}", question_id, exc)
            return None
        selected = self._find_selected_candidate(candidates, decision)
        if selected is not None and not _candidate_matches_question_intent(question, selected):
            logger.warning(
                "图片题候选类型与问题意图不一致，拒绝返回原图 | question_id={} element_id={} summary={}",
                question_id,
                selected.element_id,
                selected.summary,
            )
            return None
        return selected

    async def _judge_candidates(
        self,
        question: str,
        candidates: list[ImageSearchCandidate],
    ) -> ImageRelevanceDecision:
        from src.utils.llm_invoke import ainvoke_llm_with_structured_output

        return await ainvoke_llm_with_structured_output(
            self._build_relevance_prompt(question, candidates),
            model_id=_IMAGE_RELEVANCE_MODEL_KEY,
            response_type=ImageRelevanceDecision,
        )

    @staticmethod
    def _build_relevance_prompt(
        question: str,
        candidates: list[ImageSearchCandidate],
    ) -> str:
        candidate_text = "\n".join(
            f"- element_id={candidate.element_id}; image_summary={candidate.summary}"
            for candidate in candidates
        )
        return f"""Return one valid JSON object only.

You are a strict enterprise-report image retrieval reviewer.
User question: {question}

Candidate image summaries:
{candidate_text}

Select an element_id only when the summary explicitly confirms the requested image type and relationship. When the question names one, the summary must also confirm that entity. For a generic question, use only the allowed report candidates shown above; do not infer facts. If no candidate clearly matches, set selected_element_id to null and confidence below 0.85.
An organization chart is not an ownership/shareholding chart. For an organization
question, reject candidates that only describe shareholders, share percentages,
ownership penetration, or actual controllers.
"""

    def _find_selected_candidate(
        self,
        candidates: list[ImageSearchCandidate],
        decision: ImageRelevanceDecision | None,
    ) -> ImageSearchCandidate | None:
        if decision is not None:
            logger.info(
                "图片题相关性判定结果 | selected_element_id={} confidence={} threshold={} reason={}",
                decision.selected_element_id,
                decision.confidence,
                self.min_score,
                decision.reason,
            )
        if decision is None or decision.confidence < self.min_score or decision.selected_element_id is None:
            return None
        for candidate in candidates:
            if candidate.element_id == decision.selected_element_id:
                logger.info(
                    "图片题相关性判定通过 | element_id={} confidence={} threshold={}",
                    candidate.element_id,
                    decision.confidence,
                    self.min_score,
                )
                return candidate
        logger.warning(
            "图片题相关性判定返回了不在候选集中的 element_id={}，将拒绝返回原图",
            decision.selected_element_id,
        )
        return None

    async def _answer_with_original_image(self, candidate: ImageSearchCandidate) -> FinalAnswer:
        image_data_uri = await self._load_data_uri(candidate)
        if len(image_data_uri) > self.max_data_uri_chars:
            raise ValueError(f"Data URI 长度 {len(image_data_uri)} 超过上限 {self.max_data_uri_chars}")
        return FinalAnswer(
            is_answerable=True,
            answer_text=image_data_uri,
            citations=[
                Citation(
                    id=1,
                    element_ids=[candidate.element_id],
                    file_id=candidate.file_id,
                    quote=candidate.summary or f"图片元素 {candidate.element_id}",
                )
            ],
        )

    async def _try_original_image(
        self,
        question_id: str,
        candidate: ImageSearchCandidate,
    ) -> tuple[FinalAnswer | None, bool]:
        try:
            return await self._answer_with_original_image(candidate), False
        except ValueError as exc:
            logger.warning(
                "图片题原图超过 Data URI 大小限制 | question_id={} file_id={} element_id={} error={}",
                question_id,
                candidate.file_id,
                candidate.element_id,
                exc,
            )
            return None, True
        except Exception as exc:
            logger.exception(
                "图片题原图读取失败，将尝试文本证据生成结构图 | question_id={} file_id={} element_id={} error={}",
                question_id,
                candidate.file_id,
                candidate.element_id,
                exc,
            )
            return None, False

    @staticmethod
    def _unanswerable(message: str) -> FinalAnswer:
        return FinalAnswer(is_answerable=False, answer_text=message, citations=[])


def _is_organization_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question or "")
    return any(keyword in normalized for keyword in _ORGANIZATION_QUERY_KEYWORDS)


def _is_ownership_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question or "")
    return any(keyword in normalized for keyword in _OWNERSHIP_QUERY_KEYWORDS)


def _candidate_matches_question_intent(
    question: str,
    candidate: ImageSearchCandidate,
) -> bool:
    """Apply a deterministic guard after the model's image relevance decision."""
    if not _is_organization_question(question):
        return True
    normalized_summary = re.sub(r"\s+", "", candidate.summary or "")
    has_organization_signal = any(
        keyword in normalized_summary
        for keyword in _ORGANIZATION_IMAGE_SUMMARY_KEYWORDS
    )
    has_ownership_signal = any(
        keyword in normalized_summary
        for keyword in _OWNERSHIP_IMAGE_SUMMARY_KEYWORDS
    )
    return has_organization_signal and not (
        has_ownership_signal and not any(
            keyword in normalized_summary
            for keyword in ("组织架构", "组织结构", "治理结构", "部门架构")
        )
    )


def _build_entity_enriched_query(question: str, evidence: list[TextEvidence]) -> str | None:
    """Use one report subject to improve a generic ownership/organization image query."""
    normalized_question = re.sub(r"\s+", "", question or "")
    if not any(
        keyword in normalized_question
        for keyword in (*_OWNERSHIP_QUERY_KEYWORDS, *_ORGANIZATION_QUERY_KEYWORDS)
    ):
        return None

    entities = {
        entity
        for item in evidence
        if (entity := _report_entity_from_file_name(item.file_name))
    }
    if len(entities) != 1:
        return None
    entity = next(iter(entities))
    if re.sub(r"\s+", "", entity) in normalized_question:
        return None
    return f"{entity} {question}"


def _report_entity_from_file_name(file_name: str) -> str:
    name = str(file_name or "").strip()
    name = re.sub(r"\.(?:pdf|docx?|xlsx?)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"20\d{2}年(?:年度)?报告.*$", "", name).strip()


def _build_organization_evidence_queries(question: str) -> list[str]:
    base = question.strip()
    return [
        f"{base} 股东大会 董事会 监事会 专门委员会 治理结构",
        f"{base} 总裁 副总裁 董事会秘书 高级管理层 汇报关系",
        f"{base} 职能部门 管理部门 研究院 研发体系",
        f"{base} 事业部 业务部门 运营体系",
    ]
