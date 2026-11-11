import asyncio
import json

from fastapi import BackgroundTasks

from src.agents.policy.policy_check_agent.single_file_consistency_checks.schemas import (
    CandidateExtractionResult,
    CandidateJudgment,
    ConsistencyIssue,
    DocumentSegment,
    LlmCandidateOutline,
    SingleDocConsistencyCheckRequest,
)
from src.agents.policy.policy_check_agent.router import submit_single_doc_consistency_job
from src.agents.policy.policy_check_agent.single_file_consistency_checks.service import (
    SingleDocConsistencyService,
)


def _build_doc_content(lines: list[str], file_id: str = "file_consistency_001") -> list[dict]:
    content = []
    top = 0
    for index, line in enumerate(lines, start=1):
        content.append(
            {
                "page": 1,
                "type": "text",
                "bBox": {"l": 0, "t": top, "r": 100, "b": top + 20, "coord_origin": "TOPLEFT"},
                "value": line,
                "md": line,
                "file_id": file_id,
                "element_id": index,
            }
        )
        top += 24
    return content


def test_build_resource_file_from_doc_content() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    request = SingleDocConsistencyCheckRequest(
        fileId="file_a",
        fileName="sample.json",
        docContent=_build_doc_content(["第1条 总则", "借款金额：100万元"], file_id="file_a"),
        enableLlm=False,
    )

    resource_file = service._build_resource_file(request)

    assert resource_file.fileId == "file_a"
    assert resource_file.fileName == "sample.json"
    assert len(resource_file.content) == 2


def test_candidate_extraction_result_accepts_top_level_empty_list() -> None:
    result = CandidateExtractionResult.model_validate([])
    assert result.candidates == []


def test_consistency_request_accepts_legacy_callback_field_name() -> None:
    request = SingleDocConsistencyCheckRequest.model_validate(
        {
            "checkReportId": "CONS-UT-CALLBACK-001",
            "docContent": _build_doc_content(["测试"], file_id="cons_cb_001"),
            "callBackURL": "http://127.0.0.1:12223/callback",
        }
    )
    assert request.callback_url == "http://127.0.0.1:12223/callback"


def test_extract_term_fact_and_reference_candidates() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    request = SingleDocConsistencyCheckRequest(
        fileId="file_b",
        fileName="term_doc.json",
        docContent=_build_doc_content(
            [
                "第1条 总则",
                "本办法所称借款人，是指向本行申请授信的企业。",
                "借款人应当提交申请材料。",
                "借款金额：100万元。",
                "借款金额：120万元。",
                "详见附件2。",
            ],
            file_id="file_b",
        ),
        enableLlm=False,
    )
    resource_file = service._build_resource_file(request)
    repository_result = asyncio.run(service.analyze_resource_file(resource_file, enable_llm=False))

    issue_types = {issue.issue_type for issue in repository_result.issues}

    assert "FACT_VALUE_MISMATCH" in issue_types
    assert repository_result.issue_count >= 1


def test_merge_issues_deduplicates_same_reference_problem() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    issue_a = ConsistencyIssue.model_validate(
        {
            "issue_id": "issue_1",
            "issue_type": "CROSS_REFERENCE_ERROR",
            "severity": "high",
            "status": "inconsistent",
            "summary": "引用缺失",
            "reasoning": "附件不存在",
            "confidence": 0.9,
            "evidence": [],
            "attributes": {"referenceLabel": "附件2"},
        }
    )
    issue_b = ConsistencyIssue.model_validate(
        {
            "issue_id": "issue_2",
            "issue_type": "CROSS_REFERENCE_ERROR",
            "severity": "high",
            "status": "inconsistent",
            "summary": "重复引用缺失",
            "reasoning": "附件不存在",
            "confidence": 0.8,
            "evidence": [],
            "attributes": {"referenceLabel": "附件2"},
        }
    )

    merged = service._merge_issues([issue_a, issue_b])

    assert len(merged) == 1
    assert merged[0].summary == "引用缺失"


def test_integration_with_mock_llm_judgment() -> None:
    async def fake_judge(prompts: list[str], model_id: str) -> list[CandidateJudgment]:
        results = []
        for prompt in prompts:
            if "TERM_USE_MISMATCH" in prompt:
                results.append(
                    CandidateJudgment(
                        comparable=True,
                        status="inconsistent",
                        issue_type="TERM_USE_MISMATCH",
                        severity="medium",
                        summary="术语定义与使用不一致",
                        reasoning="定义为借款人，但后文用作担保责任主体。",
                        confidence=0.91,
                        suggestion="统一术语指代。",
                    )
                )
            else:
                results.append(
                    CandidateJudgment(
                        comparable=True,
                        status="inconsistent",
                        issue_type="FACT_VALUE_MISMATCH",
                        severity="high",
                        summary="金额前后不一致",
                        reasoning="同一借款金额字段出现两个数值。",
                        confidence=0.95,
                        suggestion="核对金额字段。",
                    )
                )
        return results

    service = SingleDocConsistencyService(enable_llm=True, judge_batch_func=fake_judge)
    request = SingleDocConsistencyCheckRequest(
        fileId="file_c",
        fileName="judge_doc.json",
        docContent=_build_doc_content(
            [
                "第1条 总则",
                "本办法所称借款人，是指向本行申请授信的企业。",
                "担保人应当提交借款人材料。",
                "借款金额：100万元。",
                "借款金额：120万元。",
            ],
            file_id="file_c",
        ),
    )

    result = asyncio.run(service.analyze_request(request))

    issue_types = {issue.issue_type for issue in result.issues}
    assert "TERM_USE_MISMATCH" in issue_types or "FACT_VALUE_MISMATCH" in issue_types
    assert result.overall_status == "inconsistent"
    assert result.issue_count >= 1


def test_llm_candidate_extraction_complements_rule_candidates() -> None:
    async def fake_candidate_batch(prompts: list[str], model_id: str) -> list[CandidateExtractionResult]:
        assert prompts
        payload = json.loads(prompts[0].split("[segments]\n", 1)[1])
        left_segment_id = payload["segments"][0]["segmentId"]
        right_segment_id = payload["segments"][1]["segmentId"]
        return [
            CandidateExtractionResult(
                candidates=[
                    LlmCandidateOutline(
                        candidate_type="fact_value",
                        issue_type_hint="FACT_VALUE_MISMATCH",
                        subject="借款金额",
                        left_segment_id=left_segment_id,
                        right_segment_id=right_segment_id,
                        left_summary="借款金额为100万元",
                        right_summary="借款金额为120万元",
                    )
                ]
            )
        ]

    async def fake_judge(prompts: list[str], model_id: str) -> list[CandidateJudgment]:
        assert prompts
        return [
            CandidateJudgment(
                comparable=True,
                status="inconsistent",
                issue_type="FACT_VALUE_MISMATCH",
                severity="high",
                summary="借款金额前后不一致",
                reasoning="同一字段在两个片段中出现不同金额。",
                confidence=0.93,
                suggestion="统一借款金额表述。",
            )
        ]

    service = SingleDocConsistencyService(
        enable_llm=True,
        candidate_batch_func=fake_candidate_batch,
        judge_batch_func=fake_judge,
    )
    request = SingleDocConsistencyCheckRequest(
        fileId="file_llm",
        fileName="llm_doc.json",
        docContent=_build_doc_content(["占位片段"], file_id="file_llm"),
    )
    service._build_segments = lambda resource_file, store, cleaned_items=None: [  # type: ignore[method-assign]
        DocumentSegment(segment_id="seg_a", file_id="file_llm", text="片段A：本次业务金额按照审批意见执行。"),
        DocumentSegment(segment_id="seg_b", file_id="file_llm", text="片段B：最终放款金额以合同为准。"),
    ]

    async def run() -> None:
        result = await service.analyze_request(request)
        assert result.overall_status == "inconsistent"
        assert result.issue_count == 1
        assert result.issues[0].issue_type == "FACT_VALUE_MISMATCH"

    asyncio.run(run())


def test_submit_single_doc_consistency_job_accepts_inline_doc_content() -> None:
    request = SingleDocConsistencyCheckRequest(
        fileId="file_router",
        fileName="router_doc.json",
        docContent=_build_doc_content(["第1条 总则", "借款金额：100万元。"], file_id="file_router"),
        enableLlm=False,
    )

    result = submit_single_doc_consistency_job(BackgroundTasks(), request)

    assert result["success"] is True
    assert result["message"] == "request accepted"
    assert result["jobId"]


def test_short_document_uses_full_document_candidate_prompt() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    segments = [
        DocumentSegment(segment_id=f"seg_{idx}", file_id="file_prompt", text=f"第{idx}段内容")
        for idx in range(1, 4)
    ]

    prompts = service._build_candidate_extraction_prompts(segments)

    assert len(prompts) == 1
    payload = json.loads(prompts[0].split("[segments]\n", 1)[1])
    assert len(payload["segments"]) == 3


def test_build_segments_supplements_store_with_cleaned_items() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    resource_file = service._build_resource_file(
        SingleDocConsistencyCheckRequest(
            fileId="file_segments",
            fileName="segments.json",
            docContent=_build_doc_content(
                [
                    "第一条 总则",
                    "借款金额为100万元",
                ],
                file_id="file_segments",
            ),
            enableLlm=False,
        )
    )

    class EmptyStore:
        evidence_blocks = []
        facts = []

    segments = service._build_segments(resource_file, EmptyStore(), service.cleaner.clean_file(resource_file))

    assert len(segments) >= 2
    assert any("总则" in segment.text for segment in segments)
    assert any("100万元" in segment.text for segment in segments)
