from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.content_generate_agent.utils import ainvoke_llm_with_structured_output

from .policy_relevance_profiles import ControlProfile, DocumentProfile, RequirementProfile


class LlmRequirementProfilePayload(BaseModel):
    target_doc_types: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)
    intent: str = "mixed"
    collection_heavy: bool = False
    anchor_terms: List[str] = Field(default_factory=list)
    field_terms: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class LlmControlProfilePayload(BaseModel):
    target_doc_types: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)
    check_mode: str = "mixed"
    target_fields: List[str] = Field(default_factory=list)
    target_entities: List[str] = Field(default_factory=list)
    anchor_terms: List[str] = Field(default_factory=list)
    evidence_expectation: str = "mixed"
    strictness: str = "soft"
    confidence: float = 0.0


class LlmRelevanceAdjudication(BaseModel):
    override_decision: str = "no_override"
    override_reason: str = ""
    override_confidence: float = 0.0


class LlmSemanticGateVerdict(BaseModel):
    is_related: bool = True
    should_execute: bool = False
    confidence: float = 0.0
    reason: str = ""


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


async def llm_build_requirement_profile(
    *,
    requirement_id: str,
    requirement_text: str,
    control_profiles: List[ControlProfile],
    base_profile: RequirementProfile,
    model_id: str,
) -> Optional[RequirementProfile]:
    prompt = (
        "你是合规匹配画像助手。请基于 requirement 与 controls 输出结构化画像。\n"
        "必须只返回一个合法的 json 对象，不要输出解释文本。\n"
        "json 字段: target_doc_types, domain_tags, intent, collection_heavy, anchor_terms, field_terms, confidence。\n\n"
        f"[requirement_id]\n{requirement_id}\n\n"
        f"[requirement_text]\n{requirement_text}\n\n"
        f"[base_profile]\n{json.dumps(base_profile.__dict__, ensure_ascii=False, default=list, indent=2)}\n\n"
        f"[control_profiles]\n{json.dumps([profile.__dict__ for profile in control_profiles], ensure_ascii=False, default=list, indent=2)}"
    )
    try:
        payload = await ainvoke_llm_with_structured_output(
            prompt=prompt,
            model_id=model_id,
            response_type=LlmRequirementProfilePayload,
        )
    except Exception:
        return None
    return RequirementProfile(
        requirement_id=requirement_id,
        target_doc_types=set(payload.target_doc_types) or set(base_profile.target_doc_types),
        domain_tags=set(payload.domain_tags) or set(base_profile.domain_tags),
        intent=payload.intent or base_profile.intent,
        collection_heavy=bool(payload.collection_heavy),
        anchor_terms=set(payload.anchor_terms) or set(base_profile.anchor_terms),
        field_terms=set(payload.field_terms) or set(base_profile.field_terms),
        confidence=_clip(max(base_profile.confidence, payload.confidence)),
    )


async def llm_build_control_profile(
    *,
    control_id: str,
    control_text: str,
    base_profile: ControlProfile,
    model_id: str,
) -> Optional[ControlProfile]:
    prompt = (
        "你是合规匹配画像助手。请基于 control 输出结构化画像。\n"
        "必须只返回一个合法的 json 对象，不要输出解释文本。\n"
        "json 字段: target_doc_types, domain_tags, check_mode, target_fields, target_entities, anchor_terms, "
        "evidence_expectation, strictness, confidence。\n\n"
        f"[control_id]\n{control_id}\n\n"
        f"[control_text]\n{control_text}\n\n"
        f"[base_profile]\n{json.dumps(base_profile.__dict__, ensure_ascii=False, default=list, indent=2)}"
    )
    try:
        payload = await ainvoke_llm_with_structured_output(
            prompt=prompt,
            model_id=model_id,
            response_type=LlmControlProfilePayload,
        )
    except Exception:
        return None
    return ControlProfile(
        control_id=control_id,
        target_doc_types=set(payload.target_doc_types) or set(base_profile.target_doc_types),
        domain_tags=set(payload.domain_tags) or set(base_profile.domain_tags),
        check_mode=payload.check_mode or base_profile.check_mode,
        target_fields=set(payload.target_fields) or set(base_profile.target_fields),
        target_entities=set(payload.target_entities) or set(base_profile.target_entities),
        anchor_terms=set(payload.anchor_terms) or set(base_profile.anchor_terms),
        evidence_expectation=payload.evidence_expectation or base_profile.evidence_expectation,
        strictness=payload.strictness or base_profile.strictness,
        confidence=_clip(max(base_profile.confidence, payload.confidence)),
    )


async def llm_adjudicate_gray_zone(
    *,
    model_id: str,
    document_profile: DocumentProfile,
    requirement_profile: RequirementProfile,
    control_profile: Optional[ControlProfile],
    feature_scores: Dict[str, float],
    base_decision: str,
    base_score: float,
) -> Optional[LlmRelevanceAdjudication]:
    prompt_payload: Dict[str, Any] = {
        "document_profile": {
            "doc_types": sorted(document_profile.doc_types),
            "doc_type_confidence": document_profile.doc_type_confidence,
            "domain_tags": sorted(document_profile.domain_tags),
            "anchor_terms": sorted(document_profile.anchor_terms)[:20],
            "field_terms": sorted(document_profile.field_terms),
            "is_business_overview": document_profile.is_business_overview,
            "evidence_density": document_profile.evidence_density,
            "document_signal_count": document_profile.document_signal_count,
            "sampled_text_preview": document_profile.sampled_text[:800],
        },
        "requirement_profile": {
            "requirement_id": requirement_profile.requirement_id,
            "target_doc_types": sorted(requirement_profile.target_doc_types),
            "domain_tags": sorted(requirement_profile.domain_tags),
            "intent": requirement_profile.intent,
            "collection_heavy": requirement_profile.collection_heavy,
            "anchor_terms": sorted(requirement_profile.anchor_terms),
            "field_terms": sorted(requirement_profile.field_terms),
            "confidence": requirement_profile.confidence,
        },
        "control_profile": (
            {
                "control_id": control_profile.control_id,
                "target_doc_types": sorted(control_profile.target_doc_types),
                "domain_tags": sorted(control_profile.domain_tags),
                "check_mode": control_profile.check_mode,
                "target_fields": sorted(control_profile.target_fields),
                "target_entities": sorted(control_profile.target_entities),
                "anchor_terms": sorted(control_profile.anchor_terms),
                "evidence_expectation": control_profile.evidence_expectation,
                "strictness": control_profile.strictness,
                "confidence": control_profile.confidence,
            }
            if control_profile is not None
            else None
        ),
        "feature_scores": feature_scores,
        "base_decision": base_decision,
        "base_score": base_score,
    }
    prompt = (
        "你是合规匹配灰区仲裁器，只在证据充分时覆盖原始决策。\n"
        "必须只返回一个合法的 json 对象，不要输出解释文本。\n"
        "json 字段: override_decision(可选 hard_skip/soft_skip/review/review_execute/review_hold/fast_fail_candidate/no_override), "
        "override_reason, override_confidence(0~1)。\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )
    try:
        return await ainvoke_llm_with_structured_output(
            prompt=prompt,
            model_id=model_id,
            response_type=LlmRelevanceAdjudication,
        )
    except Exception:
        return None


async def llm_semantic_relevance_gate(
    *,
    model_id: str,
    document_profile: DocumentProfile,
    requirement_profile: RequirementProfile,
    control_profile: Optional[ControlProfile],
    feature_scores: Dict[str, float],
    base_decision: str,
    base_score: float,
) -> Optional[LlmSemanticGateVerdict]:
    prompt_payload: Dict[str, Any] = {
        "document_profile": {
            "doc_types": sorted(document_profile.doc_types),
            "doc_type_confidence": document_profile.doc_type_confidence,
            "domain_tags": sorted(document_profile.domain_tags),
            "anchor_terms": sorted(document_profile.anchor_terms)[:20],
            "field_terms": sorted(document_profile.field_terms)[:20],
            "sampled_text_preview": document_profile.sampled_text[:1000],
        },
        "requirement_profile": {
            "requirement_id": requirement_profile.requirement_id,
            "target_doc_types": sorted(requirement_profile.target_doc_types),
            "domain_tags": sorted(requirement_profile.domain_tags),
            "intent": requirement_profile.intent,
            "anchor_terms": sorted(requirement_profile.anchor_terms),
            "field_terms": sorted(requirement_profile.field_terms),
            "confidence": requirement_profile.confidence,
        },
        "control_profile": (
            {
                "control_id": control_profile.control_id,
                "target_doc_types": sorted(control_profile.target_doc_types),
                "domain_tags": sorted(control_profile.domain_tags),
                "check_mode": control_profile.check_mode,
                "target_fields": sorted(control_profile.target_fields),
                "target_entities": sorted(control_profile.target_entities),
                "anchor_terms": sorted(control_profile.anchor_terms),
                "confidence": control_profile.confidence,
            }
            if control_profile is not None
            else None
        ),
        "feature_scores": feature_scores,
        "base_decision": base_decision,
        "base_score": base_score,
    }
    prompt = (
        "你是合规规则-文件语义门控器，请做二分类判定。\n"
        "必须只返回一个合法的 json 对象，不要输出解释文本。\n"
        "若规则与当前文件语义明显不相关，is_related=false；若相关且建议继续执行 should_execute=true。\n"
        "json 字段: is_related(bool), should_execute(bool), confidence(0~1), reason。\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )
    try:
        return await ainvoke_llm_with_structured_output(
            prompt=prompt,
            model_id=model_id,
            response_type=LlmSemanticGateVerdict,
        )
    except Exception:
        return None
