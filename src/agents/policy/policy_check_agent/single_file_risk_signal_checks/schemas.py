from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


IssueSeverity = Literal["high", "medium", "low"]
IssueStatus = Literal["pass", "fail", "uncertain"]


class SingleFileRiskSignalCheckRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    check_report_id: str = Field(serialization_alias="checkReportId")
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    doc_content: List[Dict[str, Any]] = Field(serialization_alias="docContent")
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


class RiskSignalSourceLocation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_id: str = Field(serialization_alias="fileId")
    element_ids: List[int] = Field(default_factory=list, serialization_alias="elementIds")
    page_no: Optional[int] = Field(default=None, serialization_alias="pageNo")
    segment_id: Optional[str] = Field(default=None, serialization_alias="segmentId")
    block_type: Optional[str] = Field(default=None, serialization_alias="blockType")


class RiskSignalEvidence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str
    source_text: str = Field(serialization_alias="sourceText")
    source_location: RiskSignalSourceLocation = Field(serialization_alias="sourceLocation")


class RiskSignalIssue(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    issue_id: str = Field(serialization_alias="issueId")
    issue_type: str = Field(serialization_alias="issueType")
    severity: IssueSeverity
    status: IssueStatus
    summary: str
    reasoning: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: List[RiskSignalEvidence] = Field(default_factory=list)
    suggestion: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SingleFileRiskSignalCheckResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(serialization_alias="documentId")
    file_id: str = Field(serialization_alias="fileId")
    file_name: str = Field(serialization_alias="fileName")
    overall_status: IssueStatus = Field(serialization_alias="overallStatus")
    summary: str
    issue_count: int = Field(serialization_alias="issueCount")
    issues: List[RiskSignalIssue] = Field(default_factory=list)
    statistics: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSegment(BaseModel):
    segment_id: str
    file_id: str
    text: str
    page_no: Optional[int] = None
    element_ids: List[int] = Field(default_factory=list)
    block_type: str = "paragraph"


class RiskSignalCandidate(BaseModel):
    candidate_id: str
    issue_type_hint: str
    severity_hint: IssueSeverity
    text: str
    segment_id: str
    evidence: RiskSignalEvidence
    attributes: Dict[str, Any] = Field(default_factory=dict)


class RiskSignalJudgment(BaseModel):
    is_risk: bool = False
    issue_type: str = "VAGUE_EXPRESSION"
    severity: IssueSeverity = "low"
    status: IssueStatus = "uncertain"
    summary: str = ""
    reasoning: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggestion: Optional[str] = None

    @field_validator("is_risk", mode="before")
    @classmethod
    def _normalize_is_risk(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1", "risk", "risky", "存在风险", "有风险"}:
                return True
            if normalized in {"false", "no", "n", "0", "safe", "no_risk", "无风险", "不存在风险"}:
                return False
        return value

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, value: Any) -> Any:
        if value is None:
            return "low"
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "high": "high",
                "medium": "medium",
                "med": "medium",
                "middle": "medium",
                "low": "low",
                "critical": "high",
                "severe": "high",
                "mild": "low",
                "严重": "high",
                "中": "medium",
                "中等": "medium",
                "一般": "medium",
                "轻微": "low",
                "低": "low",
            }
            if normalized in mapping:
                return mapping[normalized]
        return value

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> Any:
        if value is None:
            return "uncertain"
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "confirmed": "fail",
                "certain": "fail",
                "risk": "fail",
                "high_risk": "fail",
                "violation": "fail",
                "warning": "uncertain",
                "uncertain": "uncertain",
                "unknown": "uncertain",
                "review": "uncertain",
                "clear": "pass",
                "ok": "pass",
                "safe": "pass",
                "no_risk": "pass",
                "fail": "fail",
                "pass": "pass",
            }
            return mapping.get(normalized, normalized)
        return value

    @field_validator("issue_type", mode="before")
    @classmethod
    def _normalize_issue_type(cls, value: Any) -> Any:
        if value is None:
            return "VAGUE_EXPRESSION"
        if isinstance(value, str):
            normalized = value.strip().upper()
            mapping = {
                "模糊表述": "VAGUE_EXPRESSION",
                "模糊用语": "VAGUE_EXPRESSION",
                "极端表达": "EXTREME_EXPRESSION",
                "免责条款": "LIABILITY_DISCLAIMER",
                "免责": "LIABILITY_DISCLAIMER",
                "单方解释权": "UNILATERAL_INTERPRETATION",
            }
            return mapping.get(value.strip(), normalized)
        return value

    @field_validator("summary", "reasoning", mode="before")
    @classmethod
    def _normalize_text_fields(cls, value: Any) -> Any:
        if value is None:
            return ""
        return value
