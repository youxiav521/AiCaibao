from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
from typing import Any, Dict, Iterable, List, Set

from ...utils import keyword_tokens, normalize_whitespace
from .rules import (
    DOC_TYPE_RULES,
    MARRIAGE_HINTS,
    MEETING_HINTS,
    MEDIA_FILENAME_HINTS,
    SPOUSE_HINTS,
)
from .schemas import DocProfile, DocSignal, DocType


class DocumentProfileClassifier:
    def __init__(
        self,
        *,
        top_k: int = 3,
        min_effective_score: float = 2.0,
        relative_keep_ratio: float = 0.6,
    ) -> None:
        self.top_k = top_k
        self.min_effective_score = min_effective_score
        self.relative_keep_ratio = relative_keep_ratio
        self._token_doc_frequency = _build_token_doc_frequency()

    def classify(
        self,
        *,
        file_name: str,
        doc_content: List[Dict[str, Any]],
    ) -> DocProfile:
        normalized_name = normalize_whitespace(file_name or "").lower()
        joined_text = self._collect_text(doc_content)
        lowered_text = joined_text.lower()

        scores: Dict[DocType, float] = defaultdict(float)
        signals: List[DocSignal] = []

        self._apply_media_hints(normalized_name, scores=scores, signals=signals)
        self._apply_keyword_rules(
            normalized_name=normalized_name,
            lowered_text=lowered_text,
            scores=scores,
            signals=signals,
        )
        self._apply_semantic_prototype_rules(
            lowered_text=lowered_text,
            scores=scores,
            signals=signals,
        )
        self._apply_disambiguation_rules(lowered_text=lowered_text, scores=scores, signals=signals)
        self._apply_conflict_penalties(scores=scores, signals=signals)

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ordered or ordered[0][1] < self.min_effective_score:
            return DocProfile(
                primaryType=DocType.UNKNOWN,
                docTypes=[DocType.UNKNOWN],
                confidence=0.2,
                isMedia=False,
                scores={},
                signals=signals[:12],
                rawHints={
                    "fileName": file_name,
                    "textLength": len(joined_text),
                    "reason": "insufficient_signal",
                },
            )

        max_score = ordered[0][1]
        min_keep_score = max(self.min_effective_score, max_score * self.relative_keep_ratio)
        kept = [item for item in ordered if item[1] >= min_keep_score][: self.top_k]
        doc_types = [item[0] for item in kept] or [ordered[0][0]]

        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        gap = max_score - second_score
        confidence = self._estimate_confidence(
            max_score=max_score,
            second_score=second_score,
            kept=kept,
        )

        return DocProfile(
            primaryType=doc_types[0],
            docTypes=doc_types,
            confidence=confidence,
            isMedia=doc_types[0] in {DocType.AUDIO, DocType.VIDEO},
            scores={doc_type.value: round(score, 4) for doc_type, score in kept},
            signals=signals[:24],
            rawHints={
                "fileName": file_name,
                "textLength": len(joined_text),
                "topScore": round(max_score, 4),
                "top2Score": round(second_score, 4),
                "topGap": round(gap, 4),
            },
        )

    def _apply_media_hints(
        self,
        normalized_name: str,
        *,
        scores: Dict[DocType, float],
        signals: List[DocSignal],
    ) -> None:
        for doc_type, suffixes in MEDIA_FILENAME_HINTS.items():
            for suffix in suffixes:
                if normalized_name.endswith(suffix):
                    scores[doc_type] += 8.0
                    signals.append(
                        DocSignal(
                            source="structure",
                            token=suffix,
                            weight=8.0,
                            matchedSpan=normalized_name,
                        )
                    )

    def _apply_keyword_rules(
        self,
        *,
        normalized_name: str,
        lowered_text: str,
        scores: Dict[DocType, float],
        signals: List[DocSignal],
    ) -> None:
        for doc_type, pack in DOC_TYPE_RULES.items():
            strong_hits = 0
            weak_hits = 0

            for token in pack.filename_terms:
                token_norm = token.lower()
                if token_norm in normalized_name:
                    scores[doc_type] += 4.0
                    signals.append(
                        DocSignal(
                            source="filename",
                            token=token,
                            weight=4.0,
                            matchedSpan=normalized_name,
                        )
                    )

            for token in pack.strong_terms:
                token_norm = token.lower()
                if token_norm in lowered_text:
                    scores[doc_type] += 3.0
                    strong_hits += 1
                    signals.append(
                        DocSignal(
                            source="content",
                            token=token,
                            weight=3.0,
                            matchedSpan=token,
                        )
                    )

            for token in pack.weak_terms:
                token_norm = token.lower()
                if token_norm in lowered_text:
                    weak_hits += 1
                    signals.append(
                        DocSignal(
                            source="content",
                            token=token,
                            weight=1.0,
                            matchedSpan=token,
                        )
                    )

            if weak_hits > 0:
                scores[doc_type] += min(weak_hits, 3) * 0.9
            if strong_hits >= 2:
                bonus = min(2.0, (strong_hits - 1) * 0.6)
                scores[doc_type] += bonus
                signals.append(
                    DocSignal(
                        source="rule",
                        token=f"strong_coverage:{doc_type.value}",
                        weight=bonus,
                        matchedSpan=None,
                    )
                )

    def _apply_semantic_prototype_rules(
        self,
        *,
        lowered_text: str,
        scores: Dict[DocType, float],
        signals: List[DocSignal],
    ) -> None:
        text_tokens = set(token for token in keyword_tokens(lowered_text) if len(token) >= 2)
        if not text_tokens:
            return
        for doc_type in DOC_TYPE_RULES:
            proto_tokens = _prototype_tokens(doc_type)
            if not proto_tokens:
                continue
            overlap = text_tokens & proto_tokens
            if not overlap:
                continue
            overlap_weight = 0.0
            for token in overlap:
                token_df = self._token_doc_frequency.get(token, 1)
                overlap_weight += 1.0 / token_df
            normalized_score = overlap_weight / math.sqrt(max(len(proto_tokens), 1))
            semantic_score = min(2.8, normalized_score * 4.0)
            if semantic_score <= 0:
                continue
            scores[doc_type] += semantic_score
            signals.append(
                DocSignal(
                    source="rule",
                    token=f"semantic_proto:{doc_type.value}",
                    weight=round(semantic_score, 4),
                    matchedSpan=None,
                )
            )

    def _apply_disambiguation_rules(
        self,
        *,
        lowered_text: str,
        scores: Dict[DocType, float],
        signals: List[DocSignal],
    ) -> None:
        if any(token in lowered_text for token in SPOUSE_HINTS):
            scores[DocType.SPOUSE_ID] += 2.5
            signals.append(DocSignal(source="rule", token="spouse_boost", weight=2.5, matchedSpan="配偶"))

        if any(token in lowered_text for token in MARRIAGE_HINTS):
            scores[DocType.LEGAL_REP_MARRIAGE_CERT] += 2.5
            signals.append(DocSignal(source="rule", token="marriage_boost", weight=2.5, matchedSpan="婚姻"))

        if any(token in lowered_text for token in MEETING_HINTS):
            scores[DocType.MEETING_MINUTES] += 2.0
            signals.append(DocSignal(source="rule", token="minutes_boost", weight=2.0, matchedSpan="会议纪要"))

        if "付款" in lowered_text or "回单" in lowered_text or "转账" in lowered_text:
            scores[DocType.CONTRACT_AND_PAYMENT] += 1.5
            signals.append(DocSignal(source="rule", token="payment_boost", weight=1.5, matchedSpan="付款/回单"))

    @staticmethod
    def _apply_conflict_penalties(
        *,
        scores: Dict[DocType, float],
        signals: List[DocSignal],
    ) -> None:
        conflict_pairs = (
            (DocType.LEGAL_REP_ID, DocType.BUSINESS_LICENSE),
            (DocType.LEGAL_REP_ID, DocType.CONTRACT_AND_PAYMENT),
            (DocType.BUSINESS_LICENSE, DocType.CONTRACT_AND_PAYMENT),
            (DocType.CREDIT_REPORT, DocType.CONTRACT_AND_PAYMENT),
            (DocType.COLLATERAL_APPRAISAL, DocType.CONTRACT_AND_PAYMENT),
        )
        for left, right in conflict_pairs:
            left_score = scores.get(left, 0.0)
            right_score = scores.get(right, 0.0)
            if left_score < 2.0 or right_score < 2.0:
                continue
            gap = abs(left_score - right_score)
            if gap > 1.2:
                continue
            penalty = 0.8 if gap <= 0.4 else 0.5
            if left_score >= right_score:
                scores[right] = max(0.0, right_score - penalty)
                penalized = right
            else:
                scores[left] = max(0.0, left_score - penalty)
                penalized = left
            signals.append(
                DocSignal(
                    source="rule",
                    token=f"conflict_penalty:{left.value}:{right.value}",
                    weight=-penalty,
                    matchedSpan=penalized.value,
                )
            )

    @staticmethod
    def _estimate_confidence(
        *,
        max_score: float,
        second_score: float,
        kept: List[tuple[DocType, float]],
    ) -> float:
        total_score = sum(max(score, 0.0) for _, score in kept)
        if total_score <= 0:
            return 0.2
        dominance = max_score / total_score
        margin_ratio = 1.0 if max_score <= 0 else max(0.0, (max_score - second_score) / max_score)
        confidence = 0.35 + 0.35 * dominance + 0.3 * margin_ratio
        if second_score >= max_score * 0.9:
            confidence -= 0.08
        return round(min(0.99, max(0.2, confidence)), 4)

    def _collect_text(self, doc_content: Iterable[Dict[str, Any]]) -> str:
        texts: List[str] = []
        for item in doc_content:
            value = self._item_text(item)
            if value:
                texts.append(value)
            if len(texts) >= 120:
                break
        return normalize_whitespace(" ".join(texts))

    @staticmethod
    def _item_text(item: Any) -> str:
        if isinstance(item, dict):
            for key in ("md", "value", "text", "image_caption", "ocr", "content"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return ""
        for key in ("md", "value", "text", "image_caption", "ocr", "content"):
            value = getattr(item, key, None)
            if isinstance(value, str) and value.strip():
                return value
        return ""


def classify_doc_profile(
    *,
    file_name: str,
    doc_content: List[Dict[str, Any]],
) -> DocProfile:
    return DocumentProfileClassifier().classify(file_name=file_name, doc_content=doc_content)


@lru_cache(maxsize=128)
def _prototype_tokens(doc_type: DocType) -> Set[str]:
    pack = DOC_TYPE_RULES.get(doc_type)
    if not pack:
        return set()
    terms = " ".join([*pack.filename_terms, *pack.strong_terms, *pack.weak_terms]).lower()
    return set(token for token in keyword_tokens(terms) if len(token) >= 2)


@lru_cache(maxsize=1)
def _build_token_doc_frequency() -> Dict[str, int]:
    doc_frequency: Dict[str, int] = {}
    for doc_type in DOC_TYPE_RULES:
        tokens = _prototype_tokens(doc_type)
        for token in tokens:
            doc_frequency[token] = doc_frequency.get(token, 0) + 1
    return doc_frequency
