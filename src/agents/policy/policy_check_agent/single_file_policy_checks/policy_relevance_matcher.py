from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

from .policy_relevance_profiles import ControlProfile, DocumentProfile, RelevanceDecision, RequirementProfile


@dataclass
class RetrievalFeatures:
    candidate_document_count: int = 0
    evidence_hit_count: int = 0
    fact_hit_count: int = 0
    strongest_document_score: float = 0.0
    strongest_evidence_score: float = 0.0
    strongest_fact_score: float = 0.0

    @property
    def strength(self) -> float:
        doc = min(1.0, self.strongest_document_score / 8.0)
        evidence = min(1.0, self.strongest_evidence_score / 5.0)
        fact = min(1.0, self.strongest_fact_score / 5.0)
        count_signal = 0.0
        if self.fact_hit_count > 0:
            count_signal += 0.25
        if self.evidence_hit_count > 0:
            count_signal += 0.2
        if self.candidate_document_count > 0:
            count_signal += 0.1
        return _clip((0.35 * doc) + (0.35 * evidence) + (0.25 * fact) + count_signal)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _overlap_score(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    denom = max(len(left), len(right), 1)
    return inter / denom


class PolicyRelevanceMatcher:
    def __init__(
        self,
        *,
        requirement_soft_skip_threshold: float = 0.30,
        control_soft_skip_threshold: float = 0.30,
        review_threshold: float = 0.55,
    ) -> None:
        self.requirement_soft_skip_threshold = requirement_soft_skip_threshold
        self.control_soft_skip_threshold = control_soft_skip_threshold
        self.review_threshold = review_threshold

    def match_requirement(
        self,
        document_profile: DocumentProfile,
        requirement_profile: RequirementProfile,
        *,
        llm_override_used: bool = False,
    ) -> RelevanceDecision:
        feature_scores = self._common_feature_scores(
            document_profile=document_profile,
            target_doc_types=requirement_profile.target_doc_types,
            domain_tags=requirement_profile.domain_tags,
            anchor_terms=requirement_profile.anchor_terms,
            field_terms=requirement_profile.field_terms,
            entity_terms=set(),
            retrieval_features=RetrievalFeatures(),
            collection_heavy=requirement_profile.collection_heavy,
        )
        score = self._final_score(feature_scores)
        confidence = _clip(
            0.45
            + (0.2 * document_profile.doc_type_confidence)
            + (0.2 * requirement_profile.confidence)
            + (0.15 * feature_scores["doc_type_compatibility"])
            + (0.1 * feature_scores["domain_overlap"])
        )
        hard_skip = self._should_hard_skip(
            document_profile=document_profile,
            profile_confidence=requirement_profile.confidence,
            target_doc_types=requirement_profile.target_doc_types,
            feature_scores=feature_scores,
            llm_override_used=llm_override_used,
        )
        if hard_skip:
            return RelevanceDecision(
                decision="hard_skip",
                score=score,
                confidence=confidence,
                reason_code="DOC_TYPE_MISMATCH_HARD",
                reasons=["文档类型高置信不匹配且无锚点/字段重叠。"],
                feature_scores=feature_scores,
            )
        if score < self.requirement_soft_skip_threshold:
            return RelevanceDecision(
                decision="soft_skip",
                score=score,
                confidence=confidence,
                reason_code="LOW_RELEVANCE_SOFT",
                reasons=["要求级别相关性不足，建议跳过。"],
                feature_scores=feature_scores,
            )
        return RelevanceDecision(
            decision="review",
            score=score,
            confidence=confidence,
            reason_code="REQUIREMENT_REVIEW",
            reasons=["要求级别通过初筛，进入规则级判断。"],
            feature_scores=feature_scores,
        )

    def match_control(
        self,
        document_profile: DocumentProfile,
        requirement_profile: RequirementProfile,
        control_profile: ControlProfile,
        retrieval_features: RetrievalFeatures,
        *,
        llm_override_used: bool = False,
    ) -> RelevanceDecision:
        merged_domain = set(requirement_profile.domain_tags) | set(control_profile.domain_tags)
        merged_anchors = set(requirement_profile.anchor_terms) | set(control_profile.anchor_terms)
        merged_fields = set(requirement_profile.field_terms) | set(control_profile.target_fields)
        feature_scores = self._common_feature_scores(
            document_profile=document_profile,
            target_doc_types=control_profile.target_doc_types or requirement_profile.target_doc_types,
            domain_tags=merged_domain,
            anchor_terms=merged_anchors,
            field_terms=merged_fields,
            entity_terms=control_profile.target_entities,
            retrieval_features=retrieval_features,
            collection_heavy=requirement_profile.collection_heavy,
        )
        score = self._final_score(feature_scores)
        confidence = _clip(
            0.4
            + (0.15 * requirement_profile.confidence)
            + (0.2 * control_profile.confidence)
            + (0.1 * document_profile.doc_type_confidence)
            + (0.15 * feature_scores["retrieval_strength"])
        )
        hard_skip = self._should_hard_skip(
            document_profile=document_profile,
            profile_confidence=max(requirement_profile.confidence, control_profile.confidence),
            target_doc_types=control_profile.target_doc_types or requirement_profile.target_doc_types,
            feature_scores=feature_scores,
            llm_override_used=llm_override_used,
        )
        if hard_skip:
            return RelevanceDecision(
                decision="hard_skip",
                score=score,
                confidence=confidence,
                reason_code="DOC_TYPE_MISMATCH_HARD",
                reasons=["规则目标材料与当前文档高置信不匹配。"],
                feature_scores=feature_scores,
            )

        if score < self.control_soft_skip_threshold:
            return RelevanceDecision(
                decision="soft_skip",
                score=score,
                confidence=confidence,
                reason_code="LOW_RELEVANCE_SOFT",
                reasons=["规则相关性不足。"],
                feature_scores=feature_scores,
            )

        if self._is_fast_fail_candidate(
            control_profile=control_profile,
            requirement_profile=requirement_profile,
            feature_scores=feature_scores,
        ):
            return RelevanceDecision(
                decision="fast_fail_candidate",
                score=score,
                confidence=confidence,
                reason_code="MISSING_MATERIAL_FAST_FAIL_CANDIDATE",
                reasons=["规则更像材料存在性检查，且证据可行性与检索强度均偏低。"],
                feature_scores=feature_scores,
            )

        if score < self.review_threshold:
            return RelevanceDecision(
                decision="review",
                score=score,
                confidence=confidence,
                reason_code="GRAY_REVIEW",
                reasons=["灰区样本，建议保守进入审查。"],
                feature_scores=feature_scores,
            )

        return RelevanceDecision(
            decision="review",
            score=score,
            confidence=confidence,
            reason_code="CONTROL_REVIEW",
            reasons=["规则相关性充足。"],
            feature_scores=feature_scores,
        )

    @staticmethod
    def _common_feature_scores(
        *,
        document_profile: DocumentProfile,
        target_doc_types: Set[str],
        domain_tags: Set[str],
        anchor_terms: Set[str],
        field_terms: Set[str],
        entity_terms: Set[str],
        retrieval_features: RetrievalFeatures,
        collection_heavy: bool,
    ) -> Dict[str, float]:
        doc_type_compatibility = 0.55
        if target_doc_types:
            overlap = target_doc_types & document_profile.doc_types
            if overlap:
                doc_type_compatibility = _clip(0.65 + (0.15 * len(overlap)))
            else:
                doc_type_compatibility = 0.0
        domain_overlap = _overlap_score(domain_tags, document_profile.domain_tags)
        anchor_overlap = _overlap_score(anchor_terms, document_profile.anchor_terms)
        field_overlap = _overlap_score(field_terms, document_profile.field_terms)
        entity_overlap = 0.0
        if entity_terms:
            entity_overlap = _overlap_score(entity_terms, document_profile.anchor_terms | document_profile.field_terms)

        evidence_feasibility = 0.5
        if document_profile.evidence_density >= 0.45:
            evidence_feasibility += 0.15
        if document_profile.document_signal_count <= 1:
            evidence_feasibility -= 0.2
        if document_profile.is_business_overview and collection_heavy:
            evidence_feasibility -= 0.3
        evidence_feasibility = _clip(evidence_feasibility)

        retrieval_strength = retrieval_features.strength
        overview_penalty = 1.0 if (document_profile.is_business_overview and collection_heavy) else 0.0
        return {
            "doc_type_compatibility": round(doc_type_compatibility, 4),
            "domain_overlap": round(domain_overlap, 4),
            "anchor_overlap": round(anchor_overlap, 4),
            "field_overlap": round(field_overlap, 4),
            "entity_overlap": round(entity_overlap, 4),
            "evidence_feasibility": round(evidence_feasibility, 4),
            "retrieval_strength": round(retrieval_strength, 4),
            "overview_penalty": round(overview_penalty, 4),
        }

    @staticmethod
    def _final_score(feature_scores: Dict[str, float]) -> float:
        score = (
            0.24 * feature_scores["doc_type_compatibility"]
            + 0.14 * feature_scores["domain_overlap"]
            + 0.14 * feature_scores["anchor_overlap"]
            + 0.12 * feature_scores["field_overlap"]
            + 0.08 * feature_scores["entity_overlap"]
            + 0.14 * feature_scores["evidence_feasibility"]
            + 0.2 * feature_scores["retrieval_strength"]
            - 0.14 * feature_scores["overview_penalty"]
        )
        return round(_clip(score), 4)

    @staticmethod
    def _should_hard_skip(
        *,
        document_profile: DocumentProfile,
        profile_confidence: float,
        target_doc_types: Set[str],
        feature_scores: Dict[str, float],
        llm_override_used: bool,
    ) -> bool:
        if llm_override_used:
            return False
        if document_profile.doc_type_confidence < 0.85:
            return False
        if profile_confidence < 0.8:
            return False
        if not target_doc_types:
            return False
        if target_doc_types & document_profile.doc_types:
            return False
        if feature_scores["anchor_overlap"] > 0:
            return False
        if feature_scores["field_overlap"] > 0:
            return False
        if feature_scores["retrieval_strength"] > 0.08:
            return False
        return True

    @staticmethod
    def _is_fast_fail_candidate(
        *,
        control_profile: ControlProfile,
        requirement_profile: RequirementProfile,
        feature_scores: Dict[str, float],
    ) -> bool:
        if control_profile.check_mode not in {"existence", "validity", "mixed"}:
            return False
        if feature_scores["retrieval_strength"] > 0.18:
            return False
        if feature_scores["anchor_overlap"] > 0.0 or feature_scores["field_overlap"] > 0.0:
            return False
        if feature_scores["evidence_feasibility"] > 0.32:
            return False
        if not requirement_profile.collection_heavy and control_profile.evidence_expectation == "fact_first":
            return False
        return True
