from src.agents.policy.policy_check_agent.single_file_consistency_checks.schemas import (
    CandidateJudgment,
    ConsistencyIssue,
    ConsistencyEvidence,
    ConsistencySourceLocation,
    CrossReference,
    DocumentSegment,
)
from src.agents.policy.policy_check_agent.single_file_consistency_checks.service import (
    SingleDocConsistencyService,
)


def test_role_party_fact_from_material_list_table_is_filtered() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    segment = DocumentSegment(
        segment_id="seg_table",
        file_id="file_table",
        text="担保人法定代表人身份证明 | | 担保法律文书 | 3 | 担保人法人授权委托书",
        block_type="table",
    )

    should_skip = service._should_skip_fact_mention(
        segment,
        "role_party.担保人",
        "法定代表人身份证明 | | 担保法律文书 | 3 | 担保人法人授权委托书",
        "担保人法定代表人身份证明 | | 担保法律文书 | 3 | 担保人法人授权委托书",
    )

    assert should_skip is True


def test_candidate_judgment_confidence_has_default() -> None:
    judgment = CandidateJudgment(
        comparable=True,
        status="possibly_inconsistent",
        issue_type="FACT_VALUE_MISMATCH",
        severity="medium",
        summary="待复核",
        reasoning="模型未返回 confidence。",
    )

    assert judgment.confidence == 0.5


def test_contract_identifier_instruction_text_is_filtered() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    segment = DocumentSegment(
        segment_id="seg_contract",
        file_id="file_contract",
        text="合同编号应与主合同一致",
        block_type="table",
    )

    should_skip = service._should_skip_fact_mention(
        segment,
        "contract_identifier.合同编号",
        "应与主合同一致",
        "合同编号应与主合同一致",
    )

    assert should_skip is True


def test_retryable_502_error_retries_then_succeeds() -> None:
    service = SingleDocConsistencyService(enable_llm=True, llm_retry_attempts=3, llm_retry_initial_delay=0.01)
    attempts = {"count": 0}

    async def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("Error code: 502")
        return "ok"

    async def run() -> None:
        result = await service._call_llm_batch_with_retries(  # type: ignore[attr-defined]
            flaky_call,
            stage="unit_test",
            file_id="file_retry",
        )
        assert result == "ok"

    import asyncio

    asyncio.run(run())
    assert attempts["count"] == 3


def test_missing_clause_reference_is_not_reported_without_local_clause_anchors() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    reference = CrossReference(
        reference_id="ref_1",
        reference_type="clause",
        reference_label="第130条",
        segment_id="seg_1",
        evidence=ConsistencyEvidence(
            role="reference",
            source_text="见第130条",
            source_location=ConsistencySourceLocation(file_id="file_ref"),
        ),
    )

    issues = service._check_missing_references("file_ref", [reference], {})

    assert issues == []


def test_missing_clause_reference_is_reported_when_document_has_clause_anchor_system() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    reference = CrossReference(
        reference_id="ref_2",
        reference_type="clause",
        reference_label="第130条",
        segment_id="seg_1",
        evidence=ConsistencyEvidence(
            role="reference",
            source_text="见第130条",
            source_location=ConsistencySourceLocation(
                file_id="file_ref",
                segment_id="seg_1",
                page_no=1,
                block_type="text",
            ),
        ),
    )

    issues = service._check_missing_references("file_ref", [reference], {"第1条": "第1条"})

    assert len(issues) == 1
    assert issues[0].issue_type == "CROSS_REFERENCE_ERROR"


def test_missing_appendix_reference_is_reported_when_document_has_appendix_anchor_system() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    reference = CrossReference(
        reference_id="ref_3",
        reference_type="appendix",
        reference_label="附件2",
        segment_id="seg_2",
        evidence=ConsistencyEvidence(
            role="reference",
            source_text="详见附件2",
            source_location=ConsistencySourceLocation(
                file_id="file_ref",
                segment_id="seg_2",
                page_no=2,
                block_type="text",
            ),
        ),
    )

    issues = service._check_missing_references("file_ref", [reference], {"附件1": "附件1"})

    assert len(issues) == 1
    assert issues[0].issue_type == "CROSS_REFERENCE_ERROR"


def test_merge_issues_prefers_single_issue_for_same_evidence_pair() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    evidence_a = ConsistencyEvidence(
        role="mention_a",
        source_text="A",
        source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_a", block_type="table"),
    )
    evidence_b = ConsistencyEvidence(
        role="mention_b",
        source_text="B",
        source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_b", block_type="table"),
    )
    issue_1 = ConsistencyIssue(
        issue_id="issue_1",
        issue_type="FACT_VALUE_MISMATCH",
        severity="high",
        status="inconsistent",
        summary="直接冲突",
        reasoning="两段存在直接矛盾。",
        confidence=0.95,
        evidence=[evidence_a, evidence_b],
        attributes={},
    )
    issue_2 = ConsistencyIssue(
        issue_id="issue_2",
        issue_type="TERM_USE_MISMATCH",
        severity="medium",
        status="possibly_inconsistent",
        summary="命名可能不统一",
        reasoning="两段术语可能存在歧义。",
        confidence=0.75,
        evidence=[evidence_a, evidence_b],
        attributes={},
    )

    merged = service._merge_issues([issue_1, issue_2])

    assert len(merged) == 1
    assert merged[0].issue_type == "FACT_VALUE_MISMATCH"


def test_cross_reference_error_without_reference_label_is_filtered() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    issue = ConsistencyIssue(
        issue_id="issue_x",
        issue_type="CROSS_REFERENCE_ERROR",
        severity="medium",
        status="possibly_inconsistent",
        summary="可能存在引用问题",
        reasoning="两段表格名称不统一。",
        confidence=0.75,
        evidence=[
            ConsistencyEvidence(
                role="mention_a",
                source_text="A",
                source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_a", block_type="table"),
            )
        ],
        attributes={},
    )

    assert service._should_emit_issue(issue) is False


def test_exception_scope_ambiguity_issue_is_filtered() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    issue = ConsistencyIssue(
        issue_id="issue_scope",
        issue_type="CLAUSE_LOGIC_CONFLICT",
        severity="medium",
        status="possibly_inconsistent",
        summary="两处规则因适用范围可能不同而存在潜在冲突",
        reasoning="一处为一般规则，另一处可能属于特定例外情形，适用范围未明确排除。",
        confidence=0.75,
        evidence=[
            ConsistencyEvidence(
                role="mention_a",
                source_text="A",
                source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_a", block_type="table"),
            ),
            ConsistencyEvidence(
                role="mention_b",
                source_text="B",
                source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_b", block_type="table"),
            ),
        ],
        attributes={},
    )

    assert service._should_emit_issue(issue) is False


def test_possible_term_or_role_issue_is_filtered_more_aggressively() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    issue = ConsistencyIssue(
        issue_id="issue_term",
        issue_type="TERM_USE_MISMATCH",
        severity="medium",
        status="possibly_inconsistent",
        summary="术语命名可能不统一",
        reasoning="两处表述存在潜在歧义，但未见明确互斥或直接矛盾。",
        confidence=0.88,
        evidence=[
            ConsistencyEvidence(
                role="mention_a",
                source_text="A",
                source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_a", block_type="text"),
            ),
            ConsistencyEvidence(
                role="mention_b",
                source_text="B",
                source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_b", block_type="text"),
            ),
        ],
        attributes={},
    )

    assert service._should_emit_issue(issue) is False


def test_insufficient_information_issue_is_not_emitted() -> None:
    service = SingleDocConsistencyService(enable_llm=False)
    issue = ConsistencyIssue(
        issue_id="issue_insufficient",
        issue_type="FACT_VALUE_MISMATCH",
        severity="medium",
        status="insufficient_information",
        summary="需要结合上下文进一步确认",
        reasoning="当前仅看到信息颗粒度差异，尚不足以判断前后冲突。",
        confidence=0.78,
        evidence=[
            ConsistencyEvidence(
                role="mention_a",
                source_text="A",
                source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_a", block_type="text"),
            ),
            ConsistencyEvidence(
                role="mention_b",
                source_text="B",
                source_location=ConsistencySourceLocation(file_id="file_x", segment_id="seg_b", block_type="table"),
            ),
        ],
        attributes={},
    )

    assert service._should_emit_issue(issue) is False


def test_judgment_batch_failure_falls_back_to_single_item_judgment() -> None:
    service = SingleDocConsistencyService(enable_llm=True)

    async def failing_batch(prompts: list[str], model_id: str):
        raise RuntimeError("ValidationError: reasoning Field required")

    async def single_success(prompt: str):
        return CandidateJudgment(
            comparable=True,
            status="inconsistent",
            issue_type="FACT_VALUE_MISMATCH",
            severity="medium",
            summary="单条重试成功",
            reasoning="补齐字段后成功解析。",
            confidence=0.81,
        )

    service._judge_batch_func = failing_batch  # type: ignore[method-assign]
    service._default_judge_single = single_success  # type: ignore[method-assign]
    service._uses_default_judge_batch = True

    async def run() -> None:
        judgments = await service._judge_candidates_one_by_one("file_retry", ["prompt_a", "prompt_b"])
        assert len(judgments) == 2
        assert all(item.issue_type == "FACT_VALUE_MISMATCH" for item in judgments)

    import asyncio

    asyncio.run(run())
