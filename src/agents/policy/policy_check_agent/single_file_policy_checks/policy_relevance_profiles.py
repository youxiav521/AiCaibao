from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from ..legacy_schemas import Requirement, ResourceFile, item_content_extract
from ..schemas.control_models import ControlRetrievalRequest
from ..shared.doc_profile import (
    classify_doc_profile_with_llm_fallback,
    collect_policy_doc_type_anchor_terms,
    map_doc_types_to_policy_doc_types,
)
from ..utils import keyword_tokens, normalize_whitespace


DOCUMENT_SIGNAL_TERMS = (
    "合同",
    "协议",
    "申请书",
    "决议",
    "授权",
    "身份证",
    "身份证明",
    "营业执照",
    "报告",
    "发票",
    "登记",
    "保单",
    "章程",
    "凭证",
    "证明",
    "通知书",
    "批复",
    "清单",
    "回单",
)

BUSINESS_OVERVIEW_MARKERS = (
    "主营业务",
    "主要业务",
    "公司从事",
    "报告期内",
    "业务介绍",
    "行业情况",
)

DOMAIN_TAG_RULES: Dict[str, tuple[str, ...]] = {
    "identity": ("身份证", "身份证明", "法定代表人", "自然人"),
    "corporate": ("营业执照", "统一社会信用代码", "章程", "董事会", "工商"),
    "credit": ("授信", "贷款", "借款", "还款", "额度"),
    "collateral": ("担保", "抵押", "质押", "他项权证", "不动产"),
    "transaction": ("回单", "流水", "发票", "支付", "交易"),
    "compliance": ("审查", "合规", "核验", "真实性", "完整性"),
}

FIELD_TERM_RULES: Dict[str, tuple[str, ...]] = {
    "person_name": ("姓名", "法人", "法定代表人"),
    "id_card_number": ("身份证号", "公民身份号码"),
    "company_name": ("公司名称", "企业名称"),
    "uscc": ("统一社会信用代码", "社会信用代码"),
    "effective_date": ("生效日期", "签署日期", "起始日期", "到期"),
    "amount": ("金额", "授信额度", "贷款金额"),
    "rate": ("利率", "费率"),
    "term_limit": ("期限", "有效期"),
    "property_owner": ("产权人", "所有权人"),
    "vehicle_registration_number": ("车牌号", "登记编号"),
}

ENTITY_TERM_RULES: Dict[str, tuple[str, ...]] = {
    "borrower": ("借款人", "贷款人"),
    "guarantor": ("担保人", "保证人"),
    "legal_representative": ("法定代表人", "法人"),
    "shareholder": ("股东", "出资人"),
}

CHECK_MODE_RULES: Dict[str, tuple[str, ...]] = {
    "existence": ("是否存在", "是否提供", "提交", "具备"),
    "consistency": ("一致", "匹配", "相符", "对应"),
    "validity": ("有效", "过期", "真实", "完整", "清晰"),
    "relation": ("关系", "关联", "一致性", "对应关系"),
    "temporal": ("期限", "日期", "时点", "期间"),
}

INTENT_RULES: Dict[str, tuple[str, ...]] = {
    "collect": ("收集", "提交", "提供", "齐备", "材料"),
    "verify": ("核验", "核查", "审查", "确认"),
    "consistency": ("一致", "匹配", "相符"),
    "qualification": ("资质", "准入", "资格", "限制"),
    "temporal": ("时效", "期限", "日期", "期间"),
}


@dataclass
class DocumentProfile:
    doc_types: Set[str]
    doc_type_confidence: float
    domain_tags: Set[str]
    anchor_terms: Set[str]
    field_terms: Set[str]
    is_business_overview: bool
    evidence_density: float
    document_signal_count: int
    sampled_text: str


@dataclass
class RequirementProfile:
    requirement_id: str
    target_doc_types: Set[str]
    domain_tags: Set[str]
    intent: str
    collection_heavy: bool
    anchor_terms: Set[str]
    field_terms: Set[str]
    confidence: float


@dataclass
class ControlProfile:
    control_id: str
    target_doc_types: Set[str]
    domain_tags: Set[str]
    check_mode: str
    target_fields: Set[str]
    target_entities: Set[str]
    anchor_terms: Set[str]
    evidence_expectation: str
    strictness: str
    confidence: float


@dataclass
class RelevanceDecision:
    decision: str
    score: float
    confidence: float
    reason_code: str
    reasons: List[str]
    feature_scores: Dict[str, float]


def _safe_set(values: Iterable[str]) -> Set[str]:
    return {str(item).strip() for item in values if str(item).strip()}


def _extract_terms_by_rules(text: str, rules: Dict[str, tuple[str, ...]]) -> Set[str]:
    terms: Set[str] = set()
    for key, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            terms.add(key)
    return terms


def _extract_anchor_terms(text: str) -> Set[str]:
    anchors = set()
    for term in DOCUMENT_SIGNAL_TERMS:
        if term in text:
            anchors.add(term)
    return anchors


def _detect_intent(text: str) -> str:
    scored: List[tuple[str, int]] = []
    for intent, keywords in INTENT_RULES.items():
        score = sum(1 for keyword in keywords if keyword in text)
        scored.append((intent, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored or scored[0][1] <= 0:
        return "mixed"
    if len(scored) >= 2 and scored[0][1] == scored[1][1]:
        return "mixed"
    return scored[0][0]


def _detect_check_mode(text: str) -> str:
    scored: List[tuple[str, int]] = []
    for mode, keywords in CHECK_MODE_RULES.items():
        score = sum(1 for keyword in keywords if keyword in text)
        scored.append((mode, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored or scored[0][1] <= 0:
        return "mixed"
    if len(scored) >= 2 and scored[0][1] == scored[1][1]:
        return "mixed"
    return scored[0][0]


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _sample_document_text(resource_file: ResourceFile, sample_limit: int = 120) -> str:
    parts: List[str] = [resource_file.fileName or ""]
    for item in (resource_file.content or [])[:sample_limit]:
        text = item_content_extract(item)
        if text:
            parts.append(text)
    return normalize_whitespace(" ".join(parts))


async def build_document_profile(
    resource_file: ResourceFile,
    *,
    model_id: str,
    sample_limit: int = 120,
) -> DocumentProfile:
    sampled_text = _sample_document_text(resource_file, sample_limit=sample_limit)
    doc_profile = await classify_doc_profile_with_llm_fallback(
        file_name=resource_file.fileName or "",
        doc_content=resource_file.content or [],
        model_id=model_id,
        enable_llm_fallback=True,
    )
    mapped_doc_types = _safe_set(map_doc_types_to_policy_doc_types(doc_profile.doc_types))
    profile_anchors = _safe_set(collect_policy_doc_type_anchor_terms(mapped_doc_types))
    text_anchors = _extract_anchor_terms(sampled_text)
    field_terms = _extract_terms_by_rules(sampled_text, FIELD_TERM_RULES)
    domain_tags = _extract_terms_by_rules(sampled_text, DOMAIN_TAG_RULES)
    document_signal_count = sum(1 for marker in DOCUMENT_SIGNAL_TERMS if marker in sampled_text)
    evidence_density = _clip(document_signal_count / max(6.0, len(keyword_tokens(sampled_text)) / 18.0))
    return DocumentProfile(
        doc_types=mapped_doc_types,
        doc_type_confidence=_clip(float(doc_profile.confidence)),
        domain_tags=domain_tags,
        anchor_terms=profile_anchors | text_anchors,
        field_terms=field_terms,
        is_business_overview=any(marker in sampled_text for marker in BUSINESS_OVERVIEW_MARKERS),
        evidence_density=evidence_density,
        document_signal_count=document_signal_count,
        sampled_text=sampled_text,
    )


def _control_text(control: Any) -> str:
    return normalize_whitespace(
        " ".join(
            [
                str(getattr(control, "controlType", "") or ""),
                str(getattr(control, "risk", "") or ""),
                str(getattr(control, "logicDescription", "") or ""),
                str(getattr(control, "sourceText", "") or ""),
            ]
        )
    )


def build_control_profile(
    control: Any,
    *,
    enriched_request: Optional[ControlRetrievalRequest] = None,
) -> ControlProfile:
    text = _control_text(control)
    target_doc_types = set()
    domain_tags = _extract_terms_by_rules(text, DOMAIN_TAG_RULES)
    if enriched_request is not None:
        target_doc_types.update(_safe_set(enriched_request.preferred_doc_types))
        domain_tags.update(_safe_set(enriched_request.preferred_tags))
    target_fields = _extract_terms_by_rules(text, FIELD_TERM_RULES)
    if enriched_request is not None:
        target_fields.update(_safe_set(enriched_request.fact_fields))
    target_entities = _extract_terms_by_rules(text, ENTITY_TERM_RULES)
    anchor_terms = _extract_anchor_terms(text)
    check_mode = _detect_check_mode(text)

    if check_mode == "existence":
        evidence_expectation = "doc_first"
    elif check_mode in {"consistency", "relation"}:
        evidence_expectation = "fact_first"
    elif check_mode in {"validity", "temporal"}:
        evidence_expectation = "evidence_first"
    else:
        evidence_expectation = "mixed"

    risk = str(getattr(control, "risk", "") or "").upper()
    strictness = "hard" if risk in {"HIGH", "CRITICAL"} else "soft"
    confidence = 0.45
    if target_doc_types:
        confidence += 0.2
    if anchor_terms:
        confidence += 0.15
    if target_fields or target_entities:
        confidence += 0.1
    if check_mode != "mixed":
        confidence += 0.1
    return ControlProfile(
        control_id=str(getattr(control, "controlId", "") or ""),
        target_doc_types=target_doc_types,
        domain_tags=domain_tags,
        check_mode=check_mode,
        target_fields=target_fields,
        target_entities=target_entities,
        anchor_terms=anchor_terms,
        evidence_expectation=evidence_expectation,
        strictness=strictness,
        confidence=_clip(confidence),
    )


def build_requirement_profile(requirement: Requirement, control_profiles: List[ControlProfile]) -> RequirementProfile:
    header_text = normalize_whitespace(
        " ".join(
            [
                requirement.title or "",
                requirement.description or "",
                requirement.sourceText or "",
            ]
        )
    )
    control_text = normalize_whitespace(" ".join(_control_text(control) for control in requirement.controls))
    merged_text = normalize_whitespace(f"{header_text} {control_text}")
    target_doc_types = set().union(*(profile.target_doc_types for profile in control_profiles)) if control_profiles else set()
    domain_tags = set().union(*(profile.domain_tags for profile in control_profiles)) if control_profiles else set()
    domain_tags.update(_extract_terms_by_rules(merged_text, DOMAIN_TAG_RULES))
    anchor_terms = set().union(*(profile.anchor_terms for profile in control_profiles)) if control_profiles else set()
    anchor_terms.update(_extract_anchor_terms(merged_text))
    field_terms = set().union(*(profile.target_fields for profile in control_profiles)) if control_profiles else set()
    field_terms.update(_extract_terms_by_rules(merged_text, FIELD_TERM_RULES))
    intent = _detect_intent(merged_text)
    collection_markers = ("收集", "提交", "提供", "清单", "齐备", "资料", "材料", "是否存在")
    collection_heavy = (
        intent == "collect"
        or sum(1 for marker in collection_markers if marker in merged_text) >= 2
        or (len(anchor_terms) >= 3 and "collect" in {intent, "mixed"})
    )
    confidence = 0.4
    if target_doc_types:
        confidence += 0.2
    if domain_tags:
        confidence += 0.12
    if anchor_terms:
        confidence += 0.12
    if field_terms:
        confidence += 0.08
    if intent != "mixed":
        confidence += 0.08
    return RequirementProfile(
        requirement_id=requirement.requirementId,
        target_doc_types=target_doc_types,
        domain_tags=domain_tags,
        intent=intent,
        collection_heavy=collection_heavy,
        anchor_terms=anchor_terms,
        field_terms=field_terms,
        confidence=_clip(confidence),
    )
