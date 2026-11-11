from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


JudgeResult = Literal[
    "consistent",
    "inconsistent",
    "possibly_inconsistent",
    "insufficient_information",
]
IssueSeverity = Literal["high", "medium", "low"]


class SingleDocConsistencyCheckRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    check_report_id: str = Field(serialization_alias="checkReportId")
    input_path: Optional[str] = None
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    doc_content: Optional[List[Dict[str, Any]]] = None
    callback_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("callbackUrl", "callBackURL", "callback_url"),
        serialization_alias="callbackUrl",
    )
    model_id: str = "model_bailian_policy_check"
    enable_llm: bool = True
    max_candidates: int = Field(default=24, ge=1, le=64)
    max_judge_concurrency: int = Field(default=12, ge=1, le=16)
    llm_retry_attempts: int = Field(default=3, ge=1, le=6)
    check_types: List[str] = Field(
        default_factory=lambda: [
            "term_definition_consistency",
            "fact_value_consistency",
            "cross_reference_consistency",
        ]
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConsistencySourceLocation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_id: str = Field(serialization_alias="fileId")
    element_ids: List[int] = Field(default_factory=list, serialization_alias="elementIds")
    page_no: Optional[int] = Field(default=None, serialization_alias="pageNo")
    segment_id: Optional[str] = Field(default=None, serialization_alias="segmentId")
    block_type: Optional[str] = Field(default=None, serialization_alias="blockType")
    anchor_label: Optional[str] = Field(default=None, serialization_alias="anchorLabel")


class ConsistencyEvidence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str
    source_text: str = Field(serialization_alias="sourceText")
    source_location: ConsistencySourceLocation = Field(serialization_alias="sourceLocation")


class ConsistencyIssue(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    issue_id: str = Field(serialization_alias="issueId")
    issue_type: str = Field(serialization_alias="issueType")
    severity: IssueSeverity
    status: JudgeResult
    summary: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[ConsistencyEvidence] = Field(default_factory=list)
    suggestion: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SingleDocConsistencyCheckResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(serialization_alias="documentId")
    file_id: str = Field(serialization_alias="fileId")
    file_name: str = Field(serialization_alias="fileName")
    overall_status: JudgeResult = Field(serialization_alias="overallStatus")
    summary: str
    issue_count: int = Field(serialization_alias="issueCount")
    issues: List[ConsistencyIssue] = Field(default_factory=list)
    statistics: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSegment(BaseModel):
    segment_id: str
    file_id: str
    text: str
    page_no: Optional[int] = None
    element_ids: List[int] = Field(default_factory=list)
    block_type: str = "paragraph"
    anchor_label: Optional[str] = None


class TermDefinition(BaseModel):
    term_id: str
    term: str
    normalized_term: str
    definition: str
    aliases: List[str] = Field(default_factory=list)
    segment_id: str
    evidence: ConsistencyEvidence


class FactMention(BaseModel):
    fact_id: str
    slot_key: str
    slot_label: str
    value: str
    normalized_value: str
    segment_id: str
    evidence: ConsistencyEvidence


class CrossReference(BaseModel):
    reference_id: str
    reference_type: str
    reference_label: str
    segment_id: str
    evidence: ConsistencyEvidence


class ConsistencyCandidate(BaseModel):
    candidate_id: str
    candidate_type: str
    subject: str
    normalized_key: str
    issue_type_hint: str
    left_evidence: ConsistencyEvidence
    right_evidence: Optional[ConsistencyEvidence] = None
    left_summary: str = ""
    right_summary: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)


class LlmCandidateOutline(BaseModel):
    candidate_type: str
    issue_type_hint: str
    subject: str
    left_segment_id: str
    right_segment_id: Optional[str] = None
    left_summary: Optional[str] = ""
    right_summary: Optional[str] = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)


class CandidateExtractionResult(BaseModel):
    candidates: List[LlmCandidateOutline] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_list(cls, data: Any) -> Any:
        # Some models return a top-level array (e.g. []) instead of {"candidates": [...]}
        # when no candidates are found; normalize it to keep the pipeline resilient.
        if isinstance(data, list):
            return {"candidates": data}
        if data is None:
            return {"candidates": []}
        return data


class CandidateJudgment(BaseModel):
    comparable: bool = True
    status: JudgeResult = "insufficient_information"
    issue_type: str = "INSUFFICIENT_INFORMATION"
    severity: IssueSeverity = "low"
    summary: str = ""
    reasoning: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggestion: Optional[str] = None
