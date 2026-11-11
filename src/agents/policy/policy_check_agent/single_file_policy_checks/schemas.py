from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..legacy_schemas import EvidenceItem, Framework, PolicyCheckStatus


class SingleFilePolicyInputFile(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    file_id: str = Field(serialization_alias="fileId")
    file_name: str = Field(serialization_alias="fileName")
    doc_content: List[Dict[str, Any]] = Field(default_factory=list, serialization_alias="docContent")


class SingleFilePolicyCheckRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    check_report_id: str = Field(serialization_alias="checkReportId")
    callback_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("callbackUrl", "callBackURL", "callback_url"),
        serialization_alias="callbackUrl",
    )
    files: List[SingleFilePolicyInputFile] = Field(default_factory=list)
    frameworks: List[Framework] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_id: str = "model_bailian_policy_check"
    max_concurrency: int = Field(default=12, ge=1, le=16)


class ControlRetrievalStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    candidate_document_count: int = Field(default=0, serialization_alias="candidateDocumentCount")
    evidence_hit_count: int = Field(default=0, serialization_alias="evidenceHitCount")
    fact_hit_count: int = Field(default=0, serialization_alias="factHitCount")
    strongest_document_score: float = Field(default=0.0, serialization_alias="strongestDocumentScore")
    strongest_evidence_score: float = Field(default=0.0, serialization_alias="strongestEvidenceScore")
    strongest_fact_score: float = Field(default=0.0, serialization_alias="strongestFactScore")


class SingleFilePolicyControlResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    requirement_id: str = Field(serialization_alias="requirementId")
    control_id: str = Field(serialization_alias="controlId")
    result: PolicyCheckStatus
    reason: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    degraded: bool = False
    is_relevant: bool = Field(serialization_alias="isRelevant")
    relevance_reason: str = Field(serialization_alias="relevanceReason")
    retrieval_stats: ControlRetrievalStats = Field(serialization_alias="retrievalStats")
    reason_code: str = Field(default="", serialization_alias="reasonCode")
    decision_stage: str = Field(default="", serialization_alias="decisionStage")
    decision_confidence: float = Field(default=0.0, serialization_alias="decisionConfidence")
    feature_scores: Dict[str, float] = Field(default_factory=dict, serialization_alias="featureScores")
    llm_override_used: bool = Field(default=False, serialization_alias="llmOverrideUsed")
    execution_gate_passed: bool = Field(default=False, serialization_alias="executionGatePassed")
    fail_suppressed: bool = Field(default=False, serialization_alias="failSuppressed")


class SingleFilePolicyRequirementResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    framework_id: str = Field(serialization_alias="frameworkId")
    framework_name: str = Field(default="", serialization_alias="frameworkName")
    requirement_id: str = Field(serialization_alias="requirementId")
    requirement_title: str = Field(serialization_alias="requirementTitle")
    result: PolicyCheckStatus
    reason: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    control_results: List[SingleFilePolicyControlResult] = Field(default_factory=list, serialization_alias="controlResults")
    matched_control_count: int = Field(default=0, serialization_alias="matchedControlCount")
    skipped_control_count: int = Field(default=0, serialization_alias="skippedControlCount")
    degraded: bool = False


class SingleFilePolicyCheckResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(serialization_alias="documentId")
    file_id: str = Field(serialization_alias="fileId")
    file_name: str = Field(serialization_alias="fileName")
    overall_status: PolicyCheckStatus = Field(serialization_alias="overallStatus")
    summary: str
    statistics: Dict[str, int] = Field(default_factory=dict)
    requirements: List[SingleFilePolicyRequirementResult] = Field(default_factory=list)
    matched_controls: List[SingleFilePolicyControlResult] = Field(default_factory=list, serialization_alias="matchedControls")
    skipped_controls: List[SingleFilePolicyControlResult] = Field(default_factory=list, serialization_alias="skippedControls")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SingleFilePolicyCallbackEvidence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sequence_id: int = Field(serialization_alias="sequenceId")
    file_id: str = Field(serialization_alias="fileId")
    element_ids: List[int] = Field(default_factory=list, serialization_alias="elementIds")
    page_no: Optional[int] = Field(default=None, serialization_alias="pageNo")
    recommendation: str = ""
    quote: str = ""


class SingleFilePolicyCallbackRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    framework_id: str = Field(default="", serialization_alias="frameworkId")
    requirement_id: str = Field(default="", serialization_alias="requirementId")
    control_id: str = Field(default="", serialization_alias="controlId")


class SingleFilePolicyCallbackControlResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: PolicyCheckStatus
    level: str = "NONE"
    title: str = ""
    detail: str = ""
    suggestion: str = ""
    degraded: bool = False


class SingleFilePolicyCallbackResultItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    scope: str = "file"
    category: str = "COMPLIANCE"
    file_id: str = Field(default="", serialization_alias="fileId")
    doc_file_name: str = Field(default="", serialization_alias="docFileName")
    control_result: SingleFilePolicyCallbackControlResult = Field(serialization_alias="controlResult")
    rule: SingleFilePolicyCallbackRule
    evidence: List[SingleFilePolicyCallbackEvidence] = Field(default_factory=list)


class SingleFilePolicyAuditedPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_id: str = Field(serialization_alias="fileId")
    doc_file_name: str = Field(serialization_alias="docFileName")
    file_status: str = Field(default="F", serialization_alias="fileStatus")
    results: List[SingleFilePolicyCallbackResultItem] = Field(default_factory=list)


class SingleFilePolicyCompletedPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_requirements: int = Field(default=0, serialization_alias="totalRequirements")
    total_controls: int = Field(default=0, serialization_alias="totalControls")
    passed: int = 0
    failed: int = 0
    uncertain: int = 0
    skipped: int = 0
    duration_seconds: float = Field(default=0.0, serialization_alias="durationSeconds")
    pass_rate: str = Field(default="0.0%", serialization_alias="passRate")
    description: str = ""
