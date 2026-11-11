"""Retrieve text evidence for image-answer relationship graphs."""

from __future__ import annotations

from dataclasses import dataclass

from src.applications.rag.schemas import BatchQueryRequest, QuestionItem
_EVIDENCE_QUERY_ID = "image-evidence"
_EVIDENCE_TOP_K = 8
# Organization charts need broader coverage than a normal image question:
# governance, management, departments, research institutes, and divisions
# are often stored in separate adjacent chunks.
_ORGANIZATION_EVIDENCE_TOP_K = 8
DEFAULT_EVIDENCE_ITEM_CHAR_LIMIT = 6_000
DEFAULT_EVIDENCE_TOTAL_CHAR_LIMIT = 24_000


async def run_batch_query(request: object) -> list[object]:
    """Delay the heavy RAG import while preserving the test seam."""
    from src.applications.rag.services.query_service import (
        run_batch_query as execute_batch_query,
    )

    return await execute_batch_query(request)


@dataclass(frozen=True)
class TextEvidence:
    """A ranked text fragment that can support a graph relationship."""

    file_id: str
    element_ids: tuple[int, ...]
    text: str
    score: int | None
    file_name: str = ""


class ImageTextEvidenceService:
    """Use the existing RAG batch query flow for image-question text evidence."""

    async def search(self, question: str, file_ids: list[str]) -> list[TextEvidence]:
        scoped_file_ids = _stable_file_ids(file_ids)
        if not scoped_file_ids:
            return []

        request = BatchQueryRequest(
            questions=[QuestionItem(question_id=_EVIDENCE_QUERY_ID, question_text=question)],
            top_k=_EVIDENCE_TOP_K,
            file_ids=scoped_file_ids,
            return_child_chunks=True,
            rerank_enabled=True,
            hybrid_enabled=True,
            summary_mode="on",
        )
        results = await run_batch_query(request)
        return _normalize_evidence(results)

    async def search_many(
        self,
        questions: list[str],
        file_ids: list[str],
    ) -> list[TextEvidence]:
        """Retrieve several organization layers and interleave their ranked evidence."""
        scoped_file_ids = _stable_file_ids(file_ids)
        scoped_questions = _stable_questions(questions)
        if not scoped_file_ids or not scoped_questions:
            return []

        question_ids = [
            f"{_EVIDENCE_QUERY_ID}-{index}" for index in range(len(scoped_questions))
        ]
        request = BatchQueryRequest(
            questions=[
                QuestionItem(question_id=question_id, question_text=question)
                for question_id, question in zip(
                    question_ids,
                    scoped_questions,
                    strict=True,
                )
            ],
            top_k=_ORGANIZATION_EVIDENCE_TOP_K,
            file_ids=scoped_file_ids,
            return_child_chunks=True,
            rerank_enabled=True,
            hybrid_enabled=True,
            summary_mode="on",
        )
        results = await run_batch_query(request)
        return _normalize_evidence(_interleave_query_results(results, question_ids))


def build_bounded_evidence_text(
    evidence: list[TextEvidence],
    *,
    item_char_limit: int = DEFAULT_EVIDENCE_ITEM_CHAR_LIMIT,
    total_char_limit: int = DEFAULT_EVIDENCE_TOTAL_CHAR_LIMIT,
) -> str:
    """Format ranked evidence without allowing one request to grow unbounded."""
    if item_char_limit <= 0 or total_char_limit <= 0:
        raise ValueError("证据字符上限必须大于 0")

    parts: list[str] = []
    used_chars = 0
    for item in evidence:
        header = "[file_id={}; file_name={}; element_ids={}]\n".format(
            item.file_id,
            item.file_name or "unknown",
            ",".join(str(element_id) for element_id in item.element_ids),
        )
        separator_size = 1 if parts else 0
        remaining = total_char_limit - used_chars - separator_size - len(header)
        if remaining <= 0:
            break

        text_limit = min(item_char_limit, remaining)
        text = item.text.strip()[:text_limit]
        if not text:
            continue
        part = f"{header}{text}"
        parts.append(part)
        used_chars += separator_size + len(part)

    return "\n".join(parts)


def _normalize_evidence(results: list[object]) -> list[TextEvidence]:
    evidence: list[TextEvidence] = []
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    for result in results:
        metadata = getattr(result, "metadata", {}) or {}
        if str(metadata.get("type") or "").strip().lower() == "image":
            continue
        file_id = str(getattr(result, "file_id", "") or "").strip()
        text = str(getattr(result, "text", "") or "").strip()
        element_ids = _normalize_element_ids(getattr(result, "element_ids", []))
        if not file_id or not text or not element_ids:
            continue
        key = (file_id, element_ids, text)
        if key in seen:
            continue
        seen.add(key)
        score = getattr(result, "score", None)
        evidence.append(
            TextEvidence(
                file_id=file_id,
                element_ids=element_ids,
                text=text,
                score=int(score) if isinstance(score, int) else None,
                file_name=str(metadata.get("file_name") or "").strip(),
            )
        )
    return evidence


def _normalize_element_ids(values: object) -> tuple[int, ...]:
    normalized: set[int] = set()
    for value in values if isinstance(values, list) else []:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            normalized.add(value)
        elif isinstance(value, float) and value.is_integer():
            normalized.add(int(value))
        elif isinstance(value, str) and value.strip().isdigit():
            normalized.add(int(value.strip()))
    return tuple(sorted(normalized))


def _stable_file_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        file_id = str(value or "").strip()
        if not file_id or file_id in seen:
            continue
        seen.add(file_id)
        result.append(file_id)
    return result


def _stable_questions(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        question = str(value or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        result.append(question)
    return result


def _interleave_query_results(results: list[object], question_ids: list[str]) -> list[object]:
    grouped = {question_id: [] for question_id in question_ids}
    ungrouped: list[object] = []
    for result in results:
        question_id = str(getattr(result, "question_id", "") or "")
        if question_id in grouped:
            grouped[question_id].append(result)
        else:
            ungrouped.append(result)

    interleaved: list[object] = []
    max_group_size = max((len(items) for items in grouped.values()), default=0)
    for rank in range(max_group_size):
        for question_id in question_ids:
            items = grouped[question_id]
            if rank < len(items):
                interleaved.append(items[rank])
    interleaved.extend(ungrouped)
    return interleaved
