import asyncio
import json
from pathlib import Path

from src.agents.policy.policy_check_agent.single_file_completeness_checks.schemas import (
    DocumentTypeClassification,
    SectionExistenceJudgment,
    SectionGroupClassification,
    SingleFileCompletenessCheckRequest,
)
from src.agents.policy.policy_check_agent.single_file_completeness_checks.service import (
    SingleFileCompletenessCheckService,
)


def _build_text_item(line: str, element_id: int, file_id: str = "complete_doc_003") -> dict:
    top = 10 + (element_id - 1) * 40
    return {
        "page": 1,
        "type": "text",
        "bBox": {"l": 10, "t": top, "r": 500, "b": top + 30, "coord_origin": "TOPLEFT"},
        "value": line,
        "md": line,
        "file_id": file_id,
        "element_id": element_id,
    }


def test_complete_sample_detects_all_required_sections() -> None:
    service = SingleFileCompletenessCheckService(enable_llm=False)
    doc = json.loads(
        Path(r"D:\dev\yusys_ai\storage\policy_check\completeness_samples\sample_complete_doc.json").read_text(
            encoding="utf-8"
        )
    )
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-001",
        fileId="complete_doc_001",
        fileName="sample_complete_doc.json",
        docContent=doc,
        requiredSections=["目的", "适用范围"],
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.overall_status == "pass"
    assert result.statistics["pass"] == 2
    assert result.statistics["fail"] == 0
    assert result.items[0].attributes["satisfactionMode"] == "section"


def test_completeness_request_accepts_legacy_callback_field_name() -> None:
    request = SingleFileCompletenessCheckRequest.model_validate(
        {
            "checkReportId": "COMP-UT-CALLBACK-001",
            "fileId": "comp_cb_001",
            "fileName": "comp_cb_001.json",
            "docContent": [_build_text_item("测试", 1, file_id="comp_cb_001")],
            "callBackURL": "http://127.0.0.1:12223/callback",
        }
    )
    assert request.callback_url == "http://127.0.0.1:12223/callback"


def test_missing_scope_sample_reports_fail() -> None:
    service = SingleFileCompletenessCheckService(enable_llm=False)
    doc = json.loads(
        Path(r"D:\dev\yusys_ai\storage\policy_check\completeness_samples\sample_missing_scope_doc.json").read_text(
            encoding="utf-8"
        )
    )
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-002",
        fileId="complete_doc_002",
        fileName="sample_missing_scope_doc.json",
        docContent=doc,
        requiredSections=["目的", "适用范围"],
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.overall_status == "fail"
    failed_sections = {item.section_name for item in result.items if item.status == "fail"}
    assert "适用范围" in failed_sections


def test_content_match_can_cover_scope_without_explicit_title() -> None:
    service = SingleFileCompletenessCheckService(enable_llm=False)
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-003",
        fileId="complete_doc_003",
        fileName="inline_scope_doc.json",
        docContent=[
            _build_text_item("第一章 目的", 1),
            _build_text_item("为规范授信审批流程，制定本制度。", 2),
            _build_text_item("本制度适用于分行授信审批及贷后管理工作。", 3),
        ],
        requiredSections=["目的", "适用范围"],
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.overall_status == "pass"
    scope_items = [item for item in result.items if item.section_name == "适用范围"]
    assert scope_items
    assert scope_items[0].attributes["matchedBy"] == "content"
    assert scope_items[0].attributes["satisfactionMode"] == "element"


def test_credit_policy_profile_expands_required_sections() -> None:
    service = SingleFileCompletenessCheckService(enable_llm=False)
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-004",
        fileId="complete_doc_004",
        fileName="credit_policy_doc.json",
        documentType="credit_policy",
        docContent=[
            _build_text_item("第一章 目的", 1, file_id="complete_doc_004"),
            _build_text_item("第二章 适用范围", 2, file_id="complete_doc_004"),
        ],
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.item_count == 8
    failed_sections = {item.section_name for item in result.items if item.status == "fail"}
    assert "申请材料" in failed_sections
    assert "风险控制" in failed_sections


def test_loan_contract_profile_uses_contract_sections() -> None:
    service = SingleFileCompletenessCheckService(enable_llm=False)
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-005",
        fileId="complete_doc_005",
        fileName="loan_contract_doc.json",
        documentType="loan_contract",
        docContent=[
            _build_text_item("第一条 合同当事人", 1, file_id="complete_doc_005"),
            _build_text_item("借款人、贷款人双方一致同意签署本合同。", 2, file_id="complete_doc_005"),
            _build_text_item("第二条 金额与期限", 3, file_id="complete_doc_005"),
        ],
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    section_names = {item.section_name for item in result.items}
    assert "合同当事人" in section_names
    assert "违约责任" in section_names


def test_llm_profile_classification_selects_known_profile() -> None:
    async def fake_profile_classifier(prompt: str, model_id: str) -> DocumentTypeClassification:
        return DocumentTypeClassification(profile_id="loan_contract", confidence=0.91, reason="contains contract terms")

    async def fake_group_classifier(prompt: str, model_id: str) -> SectionGroupClassification:
        return SectionGroupClassification(group_ids=[], confidence=0.0, reason="unused")

    async def fake_section_judge(prompt: str, model_id: str) -> SectionExistenceJudgment:
        return SectionExistenceJudgment(
            exists=False,
            status="uncertain",
            matchedSectionName="",
            reasoning="unused",
            confidence=0.0,
            evidenceSegmentIds=[],
        )

    service = SingleFileCompletenessCheckService(
        enable_llm=True,
        profile_classifier_func=fake_profile_classifier,
        group_classifier_func=fake_group_classifier,
        section_judge_func=fake_section_judge,
    )
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-006",
        fileId="complete_doc_006",
        fileName="llm_profile_doc.json",
        docContent=[
            _build_text_item("第一条 合同当事人", 1, file_id="complete_doc_006"),
            _build_text_item("第二条 金额与期限", 2, file_id="complete_doc_006"),
        ],
        enableLlm=True,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.item_count == 6
    section_names = {item.section_name for item in result.items}
    assert "合同当事人" in section_names
    assert "争议解决" in section_names


def test_llm_unknown_profile_falls_back_to_section_groups() -> None:
    async def fake_profile_classifier(prompt: str, model_id: str) -> DocumentTypeClassification:
        return DocumentTypeClassification(profile_id="unknown", confidence=0.32, reason="mixed structure")

    async def fake_group_classifier(prompt: str, model_id: str) -> SectionGroupClassification:
        return SectionGroupClassification(group_ids=["governance_core", "procedure_core"], confidence=0.78, reason="policy like")

    async def fake_section_judge(prompt: str, model_id: str) -> SectionExistenceJudgment:
        return SectionExistenceJudgment(
            exists=False,
            status="uncertain",
            matchedSectionName="",
            reasoning="unused",
            confidence=0.0,
            evidenceSegmentIds=[],
        )

    service = SingleFileCompletenessCheckService(
        enable_llm=True,
        profile_classifier_func=fake_profile_classifier,
        group_classifier_func=fake_group_classifier,
        section_judge_func=fake_section_judge,
    )
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-007",
        fileId="complete_doc_007",
        fileName="llm_group_doc.json",
        docContent=[
            _build_text_item("第一章 目的", 1, file_id="complete_doc_007"),
        ],
        enableLlm=True,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.item_count == 7
    section_names = {item.section_name for item in result.items}
    assert "附则" in section_names
    assert "申请材料" in section_names


def test_llm_section_fallback_can_upgrade_missing_section_to_pass() -> None:
    async def fake_profile_classifier(prompt: str, model_id: str) -> DocumentTypeClassification:
        return DocumentTypeClassification(profile_id="bank_policy_basic", confidence=0.92, reason="policy")

    async def fake_group_classifier(prompt: str, model_id: str) -> SectionGroupClassification:
        return SectionGroupClassification(group_ids=[], confidence=0.0, reason="unused")

    async def fake_section_judge(prompt: str, model_id: str) -> SectionExistenceJudgment:
        return SectionExistenceJudgment(
            exists=True,
            status="pass",
            matchedSectionName="适用对象与范围",
            reasoning="semantic match found",
            confidence=0.86,
            evidenceSegmentIds=["cleaned_2"],
        )

    service = SingleFileCompletenessCheckService(
        enable_llm=True,
        profile_classifier_func=fake_profile_classifier,
        group_classifier_func=fake_group_classifier,
        section_judge_func=fake_section_judge,
    )
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-008",
        fileId="complete_doc_008",
        fileName="semantic_scope_doc.json",
        docContent=[
            _build_text_item("第一章 目的", 1, file_id="complete_doc_008"),
            _build_text_item("本制度面向分行、支行及相关审批岗位执行。", 2, file_id="complete_doc_008"),
        ],
        requiredSections=["目的", "适用范围"],
        enableLlm=True,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.overall_status == "pass"
    scope_items = [item for item in result.items if item.section_name == "适用范围"]
    assert scope_items
    assert scope_items[0].attributes["matchedBy"] == "llm_semantic"
    assert scope_items[0].attributes["satisfactionMode"] == "element"
    assert scope_items[0].status == "pass"


def test_llm_section_fallback_can_return_uncertain() -> None:
    async def fake_profile_classifier(prompt: str, model_id: str) -> DocumentTypeClassification:
        return DocumentTypeClassification(profile_id="bank_policy_basic", confidence=0.92, reason="policy")

    async def fake_group_classifier(prompt: str, model_id: str) -> SectionGroupClassification:
        return SectionGroupClassification(group_ids=[], confidence=0.0, reason="unused")

    async def fake_section_judge(prompt: str, model_id: str) -> SectionExistenceJudgment:
        return SectionExistenceJudgment(
            exists=False,
            status="uncertain",
            matchedSectionName="",
            reasoning="insufficient context",
            confidence=0.41,
            evidenceSegmentIds=["cleaned_1"],
        )

    service = SingleFileCompletenessCheckService(
        enable_llm=True,
        profile_classifier_func=fake_profile_classifier,
        group_classifier_func=fake_group_classifier,
        section_judge_func=fake_section_judge,
    )
    request = SingleFileCompletenessCheckRequest(
        checkReportId="COMP-UT-009",
        fileId="complete_doc_009",
        fileName="uncertain_scope_doc.json",
        docContent=[
            _build_text_item("第一章 目的", 1, file_id="complete_doc_009"),
        ],
        requiredSections=["目的", "适用范围"],
        enableLlm=True,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.overall_status == "uncertain"
    scope_items = [item for item in result.items if item.section_name == "适用范围"]
    assert scope_items
    assert scope_items[0].attributes["matchedBy"] == "llm_uncertain"
    assert scope_items[0].status == "uncertain"
