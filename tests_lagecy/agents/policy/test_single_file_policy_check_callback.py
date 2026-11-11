from src.agents.policy.policy_check_agent.single_file_policy_checks.router import (
    _build_audited_results,
    _build_completed_payload,
)
from src.agents.policy.policy_check_agent.single_file_policy_checks.schemas import (
    SingleFilePolicyCheckRequest,
    SingleFilePolicyCheckResult,
    SingleFilePolicyControlResult,
    SingleFilePolicyRequirementResult,
)


def _frameworks() -> list[dict]:
    return [
        {
            "frameworkId": "FW-001",
            "fileId": "policy_001",
            "name": "测试政策",
            "requirements": [
                {
                    "requirementId": "REQ-001",
                    "title": "公司章程检查",
                    "description": "检查公司章程",
                    "sourceText": "需要检查公司章程",
                    "sourceLocation": {"fileId": "policy_001", "elementIds": [1]},
                    "status": "active",
                    "controls": [
                        {
                            "controlId": "CTRL-001",
                            "controlType": "document_check",
                            "risk": "HIGH",
                            "logicDescription": "检查对象为公司章程，检查动作是对比工商备案档案",
                            "sourceText": "公司章程应与工商备案一致",
                            "sourceLocation": {"fileId": "policy_001", "elementIds": [2]},
                        },
                        {
                            "controlId": "CTRL-002",
                            "controlType": "document_check",
                            "risk": "LOW",
                            "logicDescription": "检查营业执照信息",
                            "sourceText": "营业执照 统一社会信用代码",
                            "sourceLocation": {"fileId": "policy_001", "elementIds": [3]},
                        },
                    ],
                }
            ],
        }
    ]


def test_build_audited_results_maps_java_fields() -> None:
    request = SingleFilePolicyCheckRequest(checkReportId="REPT-001", frameworks=_frameworks(), docContent=[{"value": "x"}])
    result = SingleFilePolicyCheckResult(
        document_id="doc_001",
        file_id="file_001",
        file_name="sample.json",
        overall_status="fail",
        summary="summary",
        statistics={},
        requirements=[
            SingleFilePolicyRequirementResult(
                framework_id="FW-001",
                framework_name="测试政策",
                requirement_id="REQ-001",
                requirement_title="公司章程检查",
                result="fail",
                reason="reason",
                control_results=[],
            )
        ],
        matched_controls=[
            SingleFilePolicyControlResult(
                requirement_id="REQ-001",
                control_id="CTRL-001",
                result="fail",
                reason="未找到公司章程相关证据",
                evidence=[
                    {
                        "evidenceId": "ev_1",
                        "fileId": "file_001",
                        "elementIds": [10, 11],
                        "summary": "未找到公司章程相关证据",
                        "quotedText": "未找到公司章程相关证据",
                    }
                ],
                is_relevant=True,
                relevance_reason="Specific required document was not found in the current single file; fast-path review can be used.",
                retrieval_stats={},
            )
        ],
        skipped_controls=[
            SingleFilePolicyControlResult(
                requirement_id="REQ-001",
                control_id="CTRL-002",
                result="skipped",
                reason="No clear relevance to the current single file. Skipped automatically.",
                evidence=[],
                is_relevant=False,
                relevance_reason="No strong relevance signal was retrieved.",
                retrieval_stats={},
            )
        ],
    )

    callback_results = _build_audited_results(request=request, result=result)

    assert len(callback_results) == 2
    assert callback_results[0].category == "COMPLIANCE"
    assert callback_results[0].control_result.status == "fail"
    assert callback_results[0].control_result.level == "HIGH"
    assert callback_results[0].rule.framework_id == "FW-001"
    assert callback_results[0].rule.requirement_id == "REQ-001"
    assert callback_results[0].rule.control_id == "CTRL-001"
    assert callback_results[0].evidence[0].sequence_id == 1
    assert callback_results[1].control_result.status == "skipped"
    assert callback_results[1].control_result.level == "NONE"


def test_build_completed_payload_uses_control_statistics() -> None:
    result = SingleFilePolicyCheckResult(
        document_id="doc_001",
        file_id="file_001",
        file_name="sample.json",
        overall_status="fail",
        summary="review completed",
        statistics={
            "totalRequirements": 3,
            "totalControls": 8,
            "passedControls": 5,
            "failedControls": 2,
            "uncertainControls": 1,
            "skippedControls": 0,
        },
    )

    payload = _build_completed_payload(result, 12.34)

    assert payload.total_requirements == 3
    assert payload.total_controls == 8
    assert payload.passed == 5
    assert payload.failed == 2
    assert payload.uncertain == 1
    assert payload.duration_seconds == 12.34
    assert payload.pass_rate == "62.5%"
