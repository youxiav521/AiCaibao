import pytest

from src.agents.policy.policy_check_agent.single_file_policy_checks.policy_relevance_eval import (
    build_minimal_offline_eval_set,
    evaluate_relevance_metrics,
)
from src.agents.policy.policy_check_agent.single_file_policy_checks.policy_relevance_matcher import (
    PolicyRelevanceMatcher,
    RetrievalFeatures,
)
from src.agents.policy.policy_check_agent.single_file_policy_checks.policy_relevance_profiles import (
    ControlProfile,
    DocumentProfile,
    RequirementProfile,
    build_control_profile,
    build_requirement_profile,
)
from src.agents.policy.policy_check_agent.legacy_schemas import Requirement, SourceLocation
from src.agents.policy.policy_check_agent.single_file_policy_checks.policy_relevance_llm import LlmRelevanceAdjudication
from src.agents.policy.policy_check_agent.single_file_policy_checks.service import SingleFilePolicyCheckService


def _doc_profile(**kwargs) -> DocumentProfile:
    payload = {
        "doc_types": {"identity_document"},
        "doc_type_confidence": 0.92,
        "domain_tags": {"identity"},
        "anchor_terms": {"身份证", "身份证明"},
        "field_terms": {"id_card_number", "person_name"},
        "is_business_overview": False,
        "evidence_density": 0.72,
        "document_signal_count": 4,
        "sampled_text": "身份证 姓名 公民身份号码",
    }
    payload.update(kwargs)
    return DocumentProfile(**payload)


def _requirement_profile(**kwargs) -> RequirementProfile:
    payload = {
        "requirement_id": "req_1",
        "target_doc_types": {"identity_document"},
        "domain_tags": {"identity"},
        "intent": "collect",
        "collection_heavy": True,
        "anchor_terms": {"身份证"},
        "field_terms": {"id_card_number"},
        "confidence": 0.86,
    }
    payload.update(kwargs)
    return RequirementProfile(**payload)


def _control_profile(**kwargs) -> ControlProfile:
    payload = {
        "control_id": "ctrl_1",
        "target_doc_types": {"identity_document"},
        "domain_tags": {"identity"},
        "check_mode": "existence",
        "target_fields": {"id_card_number"},
        "target_entities": {"legal_representative"},
        "anchor_terms": {"身份证"},
        "evidence_expectation": "doc_first",
        "strictness": "hard",
        "confidence": 0.9,
    }
    payload.update(kwargs)
    return ControlProfile(**payload)


def test_matcher_hard_skip_requires_strict_conditions() -> None:
    matcher = PolicyRelevanceMatcher()
    decision = matcher.match_control(
        _doc_profile(doc_types={"business_license"}, anchor_terms=set(), field_terms=set(), domain_tags={"corporate"}),
        _requirement_profile(target_doc_types={"identity_document"}, anchor_terms=set(), field_terms=set(), confidence=0.85),
        _control_profile(target_doc_types={"identity_document"}, anchor_terms=set(), target_fields=set(), confidence=0.85),
        RetrievalFeatures(),
    )
    assert decision.decision == "hard_skip"
    assert decision.reason_code == "DOC_TYPE_MISMATCH_HARD"


def test_profile_builder_requirement_is_aggregated() -> None:
    control_1 = type("ControlObj", (), {})()
    control_1.controlId = "c1"
    control_1.controlType = "document_check"
    control_1.risk = "HIGH"
    control_1.logicDescription = "核查身份证号码是否存在"
    control_1.sourceText = "身份证 公民身份号码"
    control_2 = type("ControlObj", (), {})()
    control_2.controlId = "c2"
    control_2.controlType = "document_check"
    control_2.risk = "MEDIUM"
    control_2.logicDescription = "核查营业执照统一社会信用代码"
    control_2.sourceText = "营业执照 统一社会信用代码"
    c1 = build_control_profile(control_1)
    c2 = build_control_profile(control_2)
    requirement = Requirement(
        requirementId="req_profile",
        title="基础材料审查",
        description="收集并核验关键主体材料",
        sourceText="收集身份证与营业执照",
        sourceLocation=SourceLocation(fileId="f1", elementIds=[1]),
        status="active",
        controls=[
            {
                "controlId": "c1",
                "controlType": "document_check",
                "risk": "HIGH",
                "logicDescription": "核查身份证号码是否存在",
                "sourceText": "身份证 公民身份号码",
                "sourceLocation": {"fileId": "f1", "elementIds": [2]},
            },
            {
                "controlId": "c2",
                "controlType": "document_check",
                "risk": "MEDIUM",
                "logicDescription": "核查营业执照统一社会信用代码",
                "sourceText": "营业执照 统一社会信用代码",
                "sourceLocation": {"fileId": "f1", "elementIds": [3]},
            },
        ],
    )
    profile = build_requirement_profile(requirement, [c1, c2])
    assert profile.requirement_id == "req_profile"
    assert profile.collection_heavy is True
    assert len(profile.anchor_terms) >= 2


def test_matcher_fast_fail_candidate_does_not_imply_fail() -> None:
    matcher = PolicyRelevanceMatcher()
    decision = matcher.match_control(
        _doc_profile(is_business_overview=True, document_signal_count=0, evidence_density=0.1, anchor_terms=set(), field_terms=set()),
        _requirement_profile(collection_heavy=True, anchor_terms={"合同"}, field_terms={"amount"}),
        _control_profile(check_mode="existence", evidence_expectation="doc_first", anchor_terms={"合同"}, target_fields={"amount"}),
        RetrievalFeatures(),
    )
    assert decision.decision in {"soft_skip", "fast_fail_candidate", "review"}
    assert decision.reason_code in {"LOW_RELEVANCE_SOFT", "MISSING_MATERIAL_FAST_FAIL_CANDIDATE", "GRAY_REVIEW", "CONTROL_REVIEW"}


@pytest.mark.asyncio
async def test_gray_zone_llm_override(monkeypatch) -> None:
    service = SingleFilePolicyCheckService(enable_relevance_llm_adjudication=True, gray_zone_lower=0.35, gray_zone_upper=0.8)

    async def _fake_llm(**kwargs):
        return LlmRelevanceAdjudication(
            override_decision="review",
            override_reason="灰区提升",
            override_confidence=0.91,
        )

    monkeypatch.setattr(
        "src.agents.policy.policy_check_agent.single_file_policy_checks.service.llm_adjudicate_gray_zone",
        _fake_llm,
    )
    base = matcher_decision = PolicyRelevanceMatcher().match_requirement(
        _doc_profile(doc_type_confidence=0.8, domain_tags={"identity"}, anchor_terms={"身份证"}, field_terms={"id_card_number"}),
        _requirement_profile(confidence=0.7, target_doc_types={"identity_document"}, domain_tags={"identity"}),
    )
    decision = await service._maybe_override_gray_decision(
        decision=base,
        document_profile=_doc_profile(),
        requirement_profile=_requirement_profile(),
        control_profile=None,
        model_id="model_bailian_policy_check",
    )
    assert decision.decision in {"review", matcher_decision.decision}


def test_offline_eval_metrics_schema() -> None:
    metrics = evaluate_relevance_metrics(build_minimal_offline_eval_set())
    assert "hard_skip_false_kill_rate" in metrics
    assert "review_recall" in metrics
    assert "gray_zone_accuracy_gain" in metrics
