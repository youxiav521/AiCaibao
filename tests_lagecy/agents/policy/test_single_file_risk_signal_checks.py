import asyncio
import json
from pathlib import Path

from src.agents.policy.policy_check_agent.single_file_risk_signal_checks.schemas import (
    RiskSignalJudgment,
    SingleFileRiskSignalCheckRequest,
)
from src.agents.policy.policy_check_agent.single_file_risk_signal_checks.service import (
    SingleFileRiskSignalCheckService,
)


def _build_text_item(line: str, element_id: int, file_id: str = "risk_doc_001") -> dict:
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


def test_rule_candidates_cover_four_risk_patterns() -> None:
    service = SingleFileRiskSignalCheckService(enable_llm=False)
    request = SingleFileRiskSignalCheckRequest(
        checkReportId="RISK-UT-001",
        fileId="risk_doc_001",
        fileName="risk_doc.json",
        docContent=[
            _build_text_item("本协议最终解释权归我方所有。", 1),
            _build_text_item("如遇特殊情况，甲方可视情况决定是否履行通知义务。", 2),
            _build_text_item("在任何情况下，本公司均不承担任何责任。", 3),
            _build_text_item("相关事项原则上应在三个工作日内完成。", 4),
        ],
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    issue_types = {issue.issue_type for issue in result.issues}
    assert result.issue_count == 4
    assert "UNILATERAL_INTERPRETATION" in issue_types
    assert "LIABILITY_DISCLAIMER" in issue_types
    assert "VAGUE_EXPRESSION" in issue_types
    assert [issue.evidence[0].source_location.element_ids for issue in result.issues] == [[1], [2], [3], [4]]


def test_risk_request_accepts_legacy_callback_field_name() -> None:
    request = SingleFileRiskSignalCheckRequest.model_validate(
        {
            "checkReportId": "RISK-UT-CALLBACK-001",
            "fileId": "risk_cb_001",
            "fileName": "risk_cb_001.json",
            "docContent": [_build_text_item("测试", 1, file_id="risk_cb_001")],
            "callBackURL": "http://127.0.0.1:12223/callback",
        }
    )
    assert request.callback_url == "http://127.0.0.1:12223/callback"


def test_vague_expression_without_action_context_is_filtered() -> None:
    service = SingleFileRiskSignalCheckService(enable_llm=False)
    request = SingleFileRiskSignalCheckRequest(
        checkReportId="RISK-UT-002",
        fileId="risk_doc_002",
        fileName="risk_doc_2.json",
        docContent=[
            _build_text_item("公司经营情况原则上保持稳定。", 1, file_id="risk_doc_002"),
        ],
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.issue_count == 0
    assert result.overall_status == "pass"


def test_sample_fixture_runs_and_returns_expected_issue_count() -> None:
    service = SingleFileRiskSignalCheckService(enable_llm=False)
    doc = json.loads(
        Path(r"D:\dev\yusys_ai\storage\policy_check\risk_signal_samples\sample_risky_doc.json").read_text(
            encoding="utf-8"
        )
    )
    request = SingleFileRiskSignalCheckRequest(
        checkReportId="RISK-UT-003",
        fileId="risk_doc_001",
        fileName="sample_risky_doc.json",
        docContent=doc,
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.issue_count == 4
    assert result.statistics["high"] == 2
    assert result.statistics["medium"] == 2


def test_safe_sample_fixture_returns_no_issue() -> None:
    service = SingleFileRiskSignalCheckService(enable_llm=False)
    doc = json.loads(
        Path(r"D:\dev\yusys_ai\storage\policy_check\risk_signal_samples\sample_safe_doc.json").read_text(
            encoding="utf-8"
        )
    )
    request = SingleFileRiskSignalCheckRequest(
        checkReportId="RISK-UT-004",
        fileId="risk_doc_safe_001",
        fileName="sample_safe_doc.json",
        docContent=doc,
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.issue_count == 0
    assert result.overall_status == "pass"


def test_extreme_sample_fixture_hits_extreme_and_unilateral_patterns() -> None:
    service = SingleFileRiskSignalCheckService(enable_llm=False)
    doc = json.loads(
        Path(r"D:\dev\yusys_ai\storage\policy_check\risk_signal_samples\sample_extreme_doc.json").read_text(
            encoding="utf-8"
        )
    )
    request = SingleFileRiskSignalCheckRequest(
        checkReportId="RISK-UT-005",
        fileId="risk_doc_extreme_001",
        fileName="sample_extreme_doc.json",
        docContent=doc,
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    issue_types = {issue.issue_type for issue in result.issues}
    assert result.issue_count == 2
    assert "EXTREME_EXPRESSION" in issue_types
    assert "VAGUE_EXPRESSION" in issue_types or "UNILATERAL_INTERPRETATION" in issue_types


def test_vague_expression_with_process_context_is_kept() -> None:
    service = SingleFileRiskSignalCheckService(enable_llm=False)
    request = SingleFileRiskSignalCheckRequest(
        checkReportId="RISK-UT-006",
        fileId="risk_doc_003",
        fileName="risk_doc_3.json",
        docContent=[
            _build_text_item("审批事项原则上应在五个工作日内完成。", 1, file_id="risk_doc_003"),
        ],
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.issue_count == 1
    assert result.issues[0].issue_type == "VAGUE_EXPRESSION"


def test_bank_safe_sample_fixture_returns_no_issue() -> None:
    service = SingleFileRiskSignalCheckService(enable_llm=False)
    doc = json.loads(
        Path(r"D:\dev\yusys_ai\storage\policy_check\risk_signal_samples\sample_bank_safe_doc.json").read_text(
            encoding="utf-8"
        )
    )
    request = SingleFileRiskSignalCheckRequest(
        checkReportId="RISK-UT-007",
        fileId="risk_doc_bank_safe_001",
        fileName="sample_bank_safe_doc.json",
        docContent=doc,
        enableLlm=False,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.issue_count == 0
    assert result.overall_status == "pass"


def test_llm_judgment_normalizes_uppercase_and_confirmed_status() -> None:
    judgment = RiskSignalJudgment(
        is_risk=True,
        issue_type=None,
        severity="HIGH",
        status="confirmed",
        summary="存在单方解释权风险",
        reasoning="条款授予单方最终解释权。",
        confidence=0.96,
        suggestion="删除单方解释权表述。",
    )

    assert judgment.issue_type == "VAGUE_EXPRESSION"
    assert judgment.severity == "high"
    assert judgment.status == "fail"


def test_mock_llm_judgment_maps_to_final_issue() -> None:
    async def fake_invoke_with_retry(prompts: list[str]) -> list[RiskSignalJudgment]:
        return [
            RiskSignalJudgment(
                is_risk=True,
                issue_type="UNILATERAL_INTERPRETATION",
                severity="HIGH",
                status="confirmed",
                summary="存在单方解释权风险",
                reasoning="条款授予单方最终解释权。",
                confidence=0.97,
                suggestion="删除单方解释权表述。",
            )
        ]

    service = SingleFileRiskSignalCheckService(enable_llm=True)
    service._invoke_with_retry = fake_invoke_with_retry  # type: ignore[method-assign]
    request = SingleFileRiskSignalCheckRequest(
        checkReportId="RISK-UT-008",
        fileId="risk_doc_004",
        fileName="risk_doc_4.json",
        docContent=[
            _build_text_item("本协议最终解释权归我方所有。", 1, file_id="risk_doc_004"),
        ],
        enableLlm=True,
    )

    result = asyncio.run(service.analyze_request(request))

    assert result.issue_count == 1
    assert result.overall_status == "fail"
    assert result.issues[0].issue_type == "UNILATERAL_INTERPRETATION"
    assert result.issues[0].status == "fail"
    assert result.issues[0].severity == "high"
