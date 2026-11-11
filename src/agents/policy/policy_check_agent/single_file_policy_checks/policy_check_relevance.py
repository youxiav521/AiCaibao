from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..legacy_schemas import Requirement, ResourceFile
from ..schemas.control_models import ControlRetrievalRequest
from ..services.material_retriever import MaterialRetriever
from .policy_check_builders import control_text, requirement_text
from .policy_check_config import PolicyCheckConfig
from .policy_relevance_llm import (
    llm_adjudicate_gray_zone,
    llm_build_control_profile,
    llm_build_requirement_profile,
    llm_semantic_relevance_gate,
)
from .policy_relevance_matcher import PolicyRelevanceMatcher, RetrievalFeatures
from .policy_relevance_profiles import (
    ControlProfile,
    DocumentProfile,
    RelevanceDecision,
    RequirementProfile,
    build_control_profile,
    build_document_profile,
    build_requirement_profile,
)
from .schemas import ControlRetrievalStats


@dataclass(slots=True)
class ControlRelevanceRecord:
    stats: ControlRetrievalStats
    retrieval: Any
    request: ControlRetrievalRequest
    is_fast_fail_candidate: bool
    reason_code: str
    decision_confidence: float
    feature_scores: Dict[str, float]
    llm_override_used: bool
    execution_gate_passed: bool
    decision_stage: str
    decision_score: float
    is_relevant: bool
    reason: str
    skip_reason: str
    allow_direct_fail: bool
    decision: RelevanceDecision


@dataclass(slots=True)
class RequirementReviewState:
    requirement_profile: RequirementProfile
    control_profiles: Dict[str, ControlProfile]
    enriched_requests: Dict[str, ControlRetrievalRequest]
    requirement_decisions_by_file: Dict[str, RelevanceDecision]
    requirement_llm_override_by_file: Dict[str, bool]
    requirement_hard_skipped_file_ids: Set[str] = field(default_factory=set)


class PolicyCheckRelevance:
    def __init__(
        self,
        *,
        config: PolicyCheckConfig,
        matcher: PolicyRelevanceMatcher,
        llm_gate_cache: Dict[str, RelevanceDecision],
    ) -> None:
        self.config = config
        self.matcher = matcher
        self.llm_gate_cache = llm_gate_cache

    async def build_document_profile(
        self,
        resource_file: ResourceFile,
        *,
        effective_model_id: str,
    ) -> DocumentProfile:
        return await build_document_profile(resource_file, model_id=effective_model_id)

    async def build_requirement_state(
        self,
        *,
        requirement: Requirement,
        retriever: MaterialRetriever,
        document_profiles: Dict[str, DocumentProfile],
        effective_model_id: str,
    ) -> RequirementReviewState:
        control_profiles: Dict[str, ControlProfile] = {}
        enriched_requests: Dict[str, ControlRetrievalRequest] = {}
        for control in requirement.controls:
            base_request = ControlRetrievalRequest(
                control_id=control.controlId,
                requirement_id=requirement.requirementId,
                control_text=control_text(control),
                top_k_documents=3,
                top_k_blocks=5,
                top_k_facts=4,
            )
            enriched = retriever._enrich_control_request(base_request)
            profile = build_control_profile(control, enriched_request=enriched)
            if self.config.enable_relevance_llm_profiles:
                llm_profile = await llm_build_control_profile(
                    control_id=control.controlId,
                    control_text=control_text(control),
                    base_profile=profile,
                    model_id=effective_model_id,
                )
                if llm_profile is not None:
                    profile = llm_profile
            control_profiles[control.controlId] = profile
            enriched_requests[control.controlId] = enriched

        requirement_profile = build_requirement_profile(requirement, list(control_profiles.values()))
        if self.config.enable_relevance_llm_profiles:
            llm_requirement_profile = await llm_build_requirement_profile(
                requirement_id=requirement.requirementId,
                requirement_text=requirement_text(requirement),
                control_profiles=list(control_profiles.values()),
                base_profile=requirement_profile,
                model_id=effective_model_id,
            )
            if llm_requirement_profile is not None:
                requirement_profile = llm_requirement_profile

        decisions_by_file: Dict[str, RelevanceDecision] = {}
        override_by_file: Dict[str, bool] = {}
        hard_skip_file_ids: Set[str] = set()
        for file_id, profile in document_profiles.items():
            decision = self.matcher.match_requirement(
                profile,
                requirement_profile,
                llm_override_used=False,
            )
            decision = await self._maybe_override_gray_decision(
                decision=decision,
                document_profile=profile,
                requirement_profile=requirement_profile,
                control_profile=None,
                model_id=effective_model_id,
            )
            decisions_by_file[file_id] = decision
            override_used = decision.reason_code.endswith("_OVERRIDE")
            override_by_file[file_id] = override_used
            if decision.decision == "hard_skip":
                hard_skip_file_ids.add(file_id)

        return RequirementReviewState(
            requirement_profile=requirement_profile,
            control_profiles=control_profiles,
            enriched_requests=enriched_requests,
            requirement_decisions_by_file=decisions_by_file,
            requirement_llm_override_by_file=override_by_file,
            requirement_hard_skipped_file_ids=hard_skip_file_ids,
        )

    async def evaluate_control_relevance(
        self,
        *,
        requirement: Requirement,
        control: Any,
        retriever: MaterialRetriever,
        document_profile: DocumentProfile,
        relevance_threshold: float,
        requirement_profile: RequirementProfile,
        control_profile: ControlProfile,
        enriched_request: ControlRetrievalRequest,
        effective_model_id: str,
    ) -> ControlRelevanceRecord:
        retrieval = retriever.retrieve_for_control(enriched_request)
        strongest_doc_score = max((hit.score for hit in retrieval.candidate_documents), default=0.0)
        strongest_evidence_score = max((hit.score for hit in retrieval.evidence_hits), default=0.0)
        strongest_fact_score = max((hit.score for hit in retrieval.fact_hits), default=0.0)
        stats = ControlRetrievalStats(
            candidate_document_count=len(retrieval.candidate_documents),
            evidence_hit_count=len(retrieval.evidence_hits),
            fact_hit_count=len(retrieval.fact_hits),
            strongest_document_score=round(strongest_doc_score, 4),
            strongest_evidence_score=round(strongest_evidence_score, 4),
            strongest_fact_score=round(strongest_fact_score, 4),
        )
        retrieval_features = self._build_retrieval_features(stats)
        decision = self.matcher.match_control(
            document_profile=document_profile,
            requirement_profile=requirement_profile,
            control_profile=control_profile,
            retrieval_features=retrieval_features,
            llm_override_used=False,
        )
        llm_override_used = False
        decision = await self._maybe_override_gray_decision(
            decision=decision,
            document_profile=document_profile,
            requirement_profile=requirement_profile,
            control_profile=control_profile,
            model_id=effective_model_id,
        )
        if decision.reason_code.endswith("_OVERRIDE"):
            llm_override_used = True

        if strongest_doc_score >= max(relevance_threshold, 5.5) and decision.decision == "soft_skip":
            feature_scores = dict(decision.feature_scores or {})
            semantic_signal = (
                float(feature_scores.get("anchor_overlap", 0.0)) > 0.0
                or float(feature_scores.get("field_overlap", 0.0)) > 0.0
                or float(feature_scores.get("domain_overlap", 0.0)) >= 0.12
                or float(feature_scores.get("doc_type_compatibility", 0.0)) >= 0.65
            )
            if semantic_signal:
                decision = RelevanceDecision(
                    decision="review",
                    score=max(decision.score, 0.55),
                    confidence=max(decision.confidence, 0.65),
                    reason_code="RETRIEVAL_THRESHOLD_PASS",
                    reasons=[f"候选文档最高分 {strongest_doc_score:.2f} 超阈值且语义信号可用，提升为 review。"],
                    feature_scores=decision.feature_scores,
                )
            else:
                decision = RelevanceDecision(
                    decision="soft_skip",
                    score=decision.score,
                    confidence=decision.confidence,
                    reason_code="RETRIEVAL_ONLY_NO_SEMANTIC",
                    reasons=[f"候选文档最高分 {strongest_doc_score:.2f} 较高，但缺少语义重叠信号，保持跳过。"],
                    feature_scores=decision.feature_scores,
                )
        decision = self._apply_execution_gate(
            decision=decision,
            document_profile=document_profile,
            requirement_profile=requirement_profile,
            control_profile=control_profile,
            retrieval_features=retrieval_features,
        )
        gated_decision, gate_called, gate_overridden = await self._maybe_apply_llm_semantic_gate(
            decision=decision,
            document_profile=document_profile,
            requirement_profile=requirement_profile,
            control_profile=control_profile,
            model_id=effective_model_id,
        )
        if gate_overridden:
            decision = gated_decision
            llm_override_used = True
        elif gate_called:
            decision = gated_decision

        is_relevant = decision.decision in {"review", "review_execute", "execute", "fast_fail_candidate"}
        reason_text = "; ".join(decision.reasons) or "规则相关性已评估。"
        skip_reason = ""
        if decision.decision in {"hard_skip", "soft_skip"}:
            skip_reason = "当前单文档与该规则相关性不足，已自动跳过。"
        elif decision.decision == "review_hold":
            skip_reason = "规则命中灰区但证据可执行性不足，暂不进入正式检查。"

        allow_direct_fail = (
            self._allow_direct_fail_for_fast_fail(
                document_profile=document_profile,
                requirement_profile=requirement_profile,
                control_profile=control_profile,
                decision=decision,
                llm_override_used=llm_override_used,
            )
            if decision.decision == "fast_fail_candidate"
            else False
        )

        if decision.decision in {"hard_skip", "soft_skip", "review_hold"}:
            is_relevant = False
        return ControlRelevanceRecord(
            stats=stats,
            retrieval=retrieval,
            request=enriched_request,
            is_fast_fail_candidate=decision.decision == "fast_fail_candidate",
            reason_code=decision.reason_code,
            decision_confidence=decision.confidence,
            feature_scores=decision.feature_scores,
            llm_override_used=llm_override_used,
            execution_gate_passed=decision.decision in {"review_execute", "execute", "fast_fail_candidate"},
            decision_stage=decision.decision,
            decision_score=decision.score,
            is_relevant=is_relevant,
            reason=reason_text,
            skip_reason=skip_reason,
            allow_direct_fail=allow_direct_fail,
            decision=decision,
        )

    def prune_controls_for_execution(
        self,
        *,
        relevant_controls: List[Any],
        relevance_by_control: Dict[str, ControlRelevanceRecord],
    ) -> Tuple[List[Any], Set[str]]:
        if not relevant_controls:
            return [], set()
        keep_ids: Set[str] = set()
        dropped_ids: Set[str] = set()
        review_execute_candidates: List[Tuple[Tuple[float, float, float, float], str]] = []
        for control in relevant_controls:
            control_id = control.controlId
            meta = relevance_by_control.get(control_id)
            if meta is None:
                continue
            stage = str(meta.decision_stage or "")
            if stage in {"fast_fail_candidate", "execute"} or stage != "review_execute":
                keep_ids.add(control_id)
                continue

            feature_scores = dict(meta.feature_scores or {})
            decision_score = float(meta.decision_score)
            retrieval_strength = float(feature_scores.get("retrieval_strength", 0.0))
            anchor_overlap = float(feature_scores.get("anchor_overlap", 0.0))
            field_overlap = float(feature_scores.get("field_overlap", 0.0))
            entity_overlap = float(feature_scores.get("entity_overlap", 0.0))
            domain_overlap = float(feature_scores.get("domain_overlap", 0.0))
            doc_type_compatibility = float(feature_scores.get("doc_type_compatibility", 0.0))
            confidence = float(meta.decision_confidence)
            semantic_overlap = (
                anchor_overlap > 0.0
                or field_overlap > 0.0
                or entity_overlap > 0.0
                or domain_overlap >= 0.12
                or doc_type_compatibility >= 0.65
            )

            if (
                decision_score < self.config.review_execute_score_floor
                and retrieval_strength < self.config.review_execute_retrieval_floor
                and not semantic_overlap
            ):
                dropped_ids.add(control_id)
                continue
            if not semantic_overlap and retrieval_strength < 0.95:
                dropped_ids.add(control_id)
                continue

            rank = (
                decision_score,
                retrieval_strength,
                anchor_overlap + field_overlap + entity_overlap + domain_overlap + doc_type_compatibility,
                confidence,
            )
            review_execute_candidates.append((rank, control_id))

        if self.config.max_review_execute_controls_per_requirement > 0:
            review_execute_candidates.sort(reverse=True)
            allowed = {
                control_id
                for _, control_id in review_execute_candidates[: self.config.max_review_execute_controls_per_requirement]
            }
            for _, control_id in review_execute_candidates:
                if control_id in allowed:
                    keep_ids.add(control_id)
                else:
                    dropped_ids.add(control_id)
        else:
            keep_ids.update(control_id for _, control_id in review_execute_candidates)

        kept_controls = [control for control in relevant_controls if control.controlId in keep_ids]
        return kept_controls, dropped_ids

    def _build_retrieval_features(self, stats: ControlRetrievalStats) -> RetrievalFeatures:
        return RetrievalFeatures(
            candidate_document_count=stats.candidate_document_count,
            evidence_hit_count=stats.evidence_hit_count,
            fact_hit_count=stats.fact_hit_count,
            strongest_document_score=stats.strongest_document_score,
            strongest_evidence_score=stats.strongest_evidence_score,
            strongest_fact_score=stats.strongest_fact_score,
        )

    def _apply_execution_gate(
        self,
        *,
        decision: RelevanceDecision,
        document_profile: DocumentProfile,
        requirement_profile: RequirementProfile,
        control_profile: ControlProfile,
        retrieval_features: RetrievalFeatures,
    ) -> RelevanceDecision:
        if decision.decision in {"hard_skip", "soft_skip", "fast_fail_candidate"}:
            return decision
        features = decision.feature_scores
        retrieval_strength = float(features.get("retrieval_strength", 0.0))
        anchor_overlap = float(features.get("anchor_overlap", 0.0))
        field_overlap = float(features.get("field_overlap", 0.0))
        entity_overlap = float(features.get("entity_overlap", 0.0))
        domain_overlap = float(features.get("domain_overlap", 0.0))
        doc_type_compatibility = float(features.get("doc_type_compatibility", 0.0))
        evidence_feasibility = float(features.get("evidence_feasibility", 0.0))
        has_semantic_overlap = (
            anchor_overlap > 0.0
            or field_overlap > 0.0
            or entity_overlap > 0.0
            or domain_overlap >= 0.12
            or doc_type_compatibility >= 0.65
        )
        semantic_gate_passed = (
            doc_type_compatibility >= 0.75
            and domain_overlap >= 0.5
            and evidence_feasibility >= 0.45
            and retrieval_strength >= max(self.config.execution_retrieval_floor * 0.8, 0.18)
        )
        gate_passed = (
            retrieval_strength >= self.config.execution_retrieval_floor
            or (anchor_overlap > 0 and field_overlap > 0)
            or retrieval_features.fact_hit_count > 0
            or semantic_gate_passed
        )
        if gate_passed and not has_semantic_overlap and retrieval_features.fact_hit_count <= 0:
            gate_passed = False
        if document_profile.is_business_overview and requirement_profile.collection_heavy and retrieval_strength < 0.35:
            gate_passed = False
        if gate_passed:
            return RelevanceDecision(
                decision="review_execute",
                score=decision.score,
                confidence=decision.confidence,
                reason_code=f"{decision.reason_code}_EXEC",
                reasons=[*decision.reasons, "执行门通过，进入正式检查。"],
                feature_scores=decision.feature_scores,
            )
        return RelevanceDecision(
            decision="review_hold",
            score=decision.score,
            confidence=decision.confidence,
            reason_code=f"{decision.reason_code}_HOLD",
            reasons=[*decision.reasons, "执行门未通过，暂不进入正式检查。"],
            feature_scores=decision.feature_scores,
        )

    async def _maybe_override_gray_decision(
        self,
        *,
        decision: RelevanceDecision,
        document_profile: DocumentProfile,
        requirement_profile: RequirementProfile,
        control_profile: Optional[ControlProfile],
        model_id: str,
    ) -> RelevanceDecision:
        if not self.config.enable_relevance_llm_adjudication:
            return decision
        if not (self.config.gray_zone_lower <= decision.score < self.config.gray_zone_upper):
            return decision
        adjudication = await llm_adjudicate_gray_zone(
            model_id=model_id,
            document_profile=document_profile,
            requirement_profile=requirement_profile,
            control_profile=control_profile,
            feature_scores=decision.feature_scores,
            base_decision=decision.decision,
            base_score=decision.score,
        )
        if adjudication is None:
            return decision
        override_decision = (adjudication.override_decision or "no_override").strip()
        allowed = {
            "hard_skip",
            "soft_skip",
            "review",
            "review_execute",
            "review_hold",
            "fast_fail_candidate",
            "no_override",
        }
        if override_decision not in allowed or override_decision == "no_override":
            return decision
        if float(adjudication.override_confidence or 0.0) < 0.65:
            return decision
        reasons = list(decision.reasons)
        if adjudication.override_reason:
            reasons.append(f"LLM仲裁: {adjudication.override_reason}")
        return RelevanceDecision(
            decision=override_decision,
            score=decision.score,
            confidence=max(decision.confidence, float(adjudication.override_confidence or 0.0)),
            reason_code=f"{decision.reason_code}_OVERRIDE",
            reasons=reasons,
            feature_scores=decision.feature_scores,
        )

    async def _maybe_apply_llm_semantic_gate(
        self,
        *,
        decision: RelevanceDecision,
        document_profile: DocumentProfile,
        requirement_profile: RequirementProfile,
        control_profile: Optional[ControlProfile],
        model_id: str,
    ) -> tuple[RelevanceDecision, bool, bool]:
        if not self.config.enable_relevance_llm_adjudication:
            return decision, False, False
        if not self._should_trigger_llm_semantic_gate(decision=decision, control_profile=control_profile):
            return decision, False, False

        cache_key = self._build_llm_gate_cache_key(
            model_id=model_id,
            document_profile=document_profile,
            requirement_profile=requirement_profile,
            control_profile=control_profile,
            decision=decision,
        )
        if cache_key in self.llm_gate_cache:
            cached = self.llm_gate_cache[cache_key]
            changed = cached.decision != decision.decision or cached.reason_code != decision.reason_code
            return cached, True, changed

        review_execute_needs_confirm = str(decision.decision or "") == "review_execute"
        gate_called = False
        semantic_verdict = await llm_semantic_relevance_gate(
            model_id=model_id,
            document_profile=document_profile,
            requirement_profile=requirement_profile,
            control_profile=control_profile,
            feature_scores=decision.feature_scores,
            base_decision=decision.decision,
            base_score=decision.score,
        )
        if semantic_verdict is not None:
            gate_called = True
            verdict_conf = float(semantic_verdict.confidence or 0.0)
            reasons = list(decision.reasons)
            if semantic_verdict.reason:
                reasons.append(f"LLM语义门控: {semantic_verdict.reason}")
            if (not bool(semantic_verdict.is_related)) and verdict_conf >= self.config.llm_veto_confidence:
                gated = RelevanceDecision(
                    decision="hard_skip" if decision.decision == "hard_skip" else "soft_skip",
                    score=min(decision.score, 0.34),
                    confidence=max(decision.confidence, verdict_conf),
                    reason_code=f"{decision.reason_code}_LLM_GATE_VETO",
                    reasons=reasons,
                    feature_scores=decision.feature_scores,
                )
                self._cache_llm_gate_decision(cache_key, gated)
                return gated, True, True
            if review_execute_needs_confirm:
                if (not bool(semantic_verdict.should_execute)) or (
                    verdict_conf < self.config.llm_review_execute_confirm_confidence
                ):
                    gated = RelevanceDecision(
                        decision="review_hold",
                        score=decision.score,
                        confidence=max(decision.confidence, verdict_conf),
                        reason_code=f"{decision.reason_code}_LLM_GATE_NO_CONFIRM",
                        reasons=[*reasons, "弱语义样本未获得LLM执行确认，降级为 review_hold。"],
                        feature_scores=decision.feature_scores,
                    )
                    self._cache_llm_gate_decision(cache_key, gated)
                    return gated, True, True
                gated = RelevanceDecision(
                    decision=decision.decision,
                    score=decision.score,
                    confidence=max(decision.confidence, verdict_conf),
                    reason_code=f"{decision.reason_code}_LLM_GATE_CONFIRMED",
                    reasons=[*reasons, "弱语义样本已获得LLM执行确认。"],
                    feature_scores=decision.feature_scores,
                )
                self._cache_llm_gate_decision(cache_key, gated)
                return gated, True, True
            if (
                bool(semantic_verdict.is_related)
                and bool(semantic_verdict.should_execute)
                and verdict_conf >= self.config.llm_promote_confidence
                and decision.decision in {"review_hold", "soft_skip", "review"}
            ):
                gated = RelevanceDecision(
                    decision="review_execute",
                    score=max(decision.score, 0.58),
                    confidence=max(decision.confidence, verdict_conf),
                    reason_code=f"{decision.reason_code}_LLM_GATE_PROMOTE",
                    reasons=reasons,
                    feature_scores=decision.feature_scores,
                )
                self._cache_llm_gate_decision(cache_key, gated)
                return gated, True, True

        if review_execute_needs_confirm:
            gated = RelevanceDecision(
                decision="review_hold",
                score=decision.score,
                confidence=decision.confidence,
                reason_code=f"{decision.reason_code}_LLM_GATE_MISSING_CONFIRM",
                reasons=[*decision.reasons, "弱语义样本缺少LLM确认，降级为 review_hold。"],
                feature_scores=decision.feature_scores,
            )
            self._cache_llm_gate_decision(cache_key, gated)
            return gated, True, True

        adjudication = await llm_adjudicate_gray_zone(
            model_id=model_id,
            document_profile=document_profile,
            requirement_profile=requirement_profile,
            control_profile=control_profile,
            feature_scores=decision.feature_scores,
            base_decision=decision.decision,
            base_score=decision.score,
        )
        if adjudication is None:
            return decision, gate_called, False
        gate_called = True
        override_decision = str(adjudication.override_decision or "no_override").strip()
        override_conf = float(adjudication.override_confidence or 0.0)
        if override_decision == "no_override":
            return decision, gate_called, False
        reasons = list(decision.reasons)
        if adjudication.override_reason:
            reasons.append(f"LLM语义门控: {adjudication.override_reason}")
        if override_decision in {"soft_skip", "hard_skip"} and override_conf >= self.config.llm_veto_confidence:
            gated = RelevanceDecision(
                decision="soft_skip" if override_decision == "soft_skip" else "hard_skip",
                score=decision.score,
                confidence=max(decision.confidence, override_conf),
                reason_code=f"{decision.reason_code}_LLM_GATE_VETO",
                reasons=reasons,
                feature_scores=decision.feature_scores,
            )
            self._cache_llm_gate_decision(cache_key, gated)
            return gated, True, True
        if override_decision in {"review_execute", "review"} and override_conf >= self.config.llm_promote_confidence:
            target_decision = "review_execute" if override_decision == "review_execute" else "review"
            gated = RelevanceDecision(
                decision=target_decision,
                score=max(decision.score, 0.56),
                confidence=max(decision.confidence, override_conf),
                reason_code=f"{decision.reason_code}_LLM_GATE_PROMOTE",
                reasons=reasons,
                feature_scores=decision.feature_scores,
            )
            self._cache_llm_gate_decision(cache_key, gated)
            return gated, True, True
        if override_decision == "review_hold" and override_conf >= self.config.llm_veto_confidence:
            gated = RelevanceDecision(
                decision="review_hold",
                score=decision.score,
                confidence=max(decision.confidence, override_conf),
                reason_code=f"{decision.reason_code}_LLM_GATE_HOLD",
                reasons=reasons,
                feature_scores=decision.feature_scores,
            )
            self._cache_llm_gate_decision(cache_key, gated)
            return gated, True, True
        return decision, gate_called, False

    def _should_trigger_llm_semantic_gate(
        self,
        *,
        decision: RelevanceDecision,
        control_profile: Optional[ControlProfile],
    ) -> bool:
        if control_profile is None:
            return False
        stage = str(decision.decision or "")
        if stage in {"review_execute", "review", "fast_fail_candidate"}:
            return True
        features = dict(decision.feature_scores or {})
        retrieval_strength = float(features.get("retrieval_strength", 0.0))
        anchor_overlap = float(features.get("anchor_overlap", 0.0))
        field_overlap = float(features.get("field_overlap", 0.0))
        domain_overlap = float(features.get("domain_overlap", 0.0))
        doc_type_compatibility = float(features.get("doc_type_compatibility", 0.0))
        has_semantic_signal = (
            anchor_overlap > 0.0
            or field_overlap > 0.0
            or domain_overlap >= 0.08
            or doc_type_compatibility >= 0.5
        )
        if stage == "soft_skip":
            return retrieval_strength >= 0.45 or has_semantic_signal
        if stage == "review_hold":
            return retrieval_strength >= 0.18 or has_semantic_signal
        return False

    @staticmethod
    def _build_llm_gate_cache_key(
        *,
        model_id: str,
        document_profile: DocumentProfile,
        requirement_profile: RequirementProfile,
        control_profile: Optional[ControlProfile],
        decision: RelevanceDecision,
    ) -> str:
        payload = {
            "model_id": model_id,
            "doc_types": sorted(document_profile.doc_types),
            "doc_conf": round(float(document_profile.doc_type_confidence), 4),
            "doc_anchor": sorted(document_profile.anchor_terms)[:24],
            "doc_fields": sorted(document_profile.field_terms)[:24],
            "req_id": requirement_profile.requirement_id,
            "req_doc_types": sorted(requirement_profile.target_doc_types),
            "req_domain": sorted(requirement_profile.domain_tags),
            "req_anchor": sorted(requirement_profile.anchor_terms)[:24],
            "req_fields": sorted(requirement_profile.field_terms)[:24],
            "control_id": control_profile.control_id if control_profile else "",
            "control_doc_types": sorted(control_profile.target_doc_types) if control_profile else [],
            "control_domain": sorted(control_profile.domain_tags) if control_profile else [],
            "control_anchor": sorted(control_profile.anchor_terms)[:24] if control_profile else [],
            "base_decision": decision.decision,
            "base_score": round(float(decision.score), 4),
            "feature_scores": {k: round(float(v), 4) for k, v in sorted((decision.feature_scores or {}).items())},
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _cache_llm_gate_decision(self, key: str, decision: RelevanceDecision) -> None:
        if len(self.llm_gate_cache) >= 4000:
            self.llm_gate_cache.clear()
        self.llm_gate_cache[key] = decision

    @staticmethod
    def _allow_direct_fail_for_fast_fail(
        *,
        document_profile: DocumentProfile,
        requirement_profile: RequirementProfile,
        control_profile: ControlProfile,
        decision: RelevanceDecision,
        llm_override_used: bool,
    ) -> bool:
        if llm_override_used:
            return False
        if control_profile.check_mode not in {"existence", "validity"}:
            return False
        if control_profile.strictness != "hard":
            return False
        if document_profile.doc_type_confidence < 0.85:
            return False
        if max(requirement_profile.confidence, control_profile.confidence) < 0.8:
            return False
        if decision.feature_scores.get("retrieval_strength", 1.0) > 0.08:
            return False
        if decision.feature_scores.get("anchor_overlap", 1.0) > 0.0:
            return False
        if decision.feature_scores.get("field_overlap", 1.0) > 0.0:
            return False
        return True
