from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


CheckStatus = Literal["pass", "fail", "uncertain"]
IssueSeverity = Literal["high", "medium", "low"]


class SingleFileCompletenessCheckRequest(BaseModel):
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
    document_type: Optional[str] = Field(default=None, serialization_alias="documentType")
    model_id: str = "model_bailian_policy_check"
    enable_llm: bool = True
    llm_retry_attempts: int = Field(default=2, ge=1, le=6)
    required_sections: List[str] = Field(
        default_factory=list,
        serialization_alias="requiredSections",
    )


class CompletenessSourceLocation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_id: str = Field(serialization_alias="fileId")
    element_ids: List[int] = Field(default_factory=list, serialization_alias="elementIds")
    page_no: Optional[int] = Field(default=None, serialization_alias="pageNo")
    segment_id: Optional[str] = Field(default=None, serialization_alias="segmentId")
    block_type: Optional[str] = Field(default=None, serialization_alias="blockType")


class CompletenessEvidence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str
    source_text: str = Field(serialization_alias="sourceText")
    source_location: CompletenessSourceLocation = Field(serialization_alias="sourceLocation")


class CompletenessCheckItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    item_id: str = Field(serialization_alias="itemId")
    check_point: str = Field(serialization_alias="checkPoint")
    section_name: str = Field(serialization_alias="sectionName")
    status: CheckStatus
    severity: IssueSeverity
    summary: str
    reasoning: str
    evidence: List[CompletenessEvidence] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SingleFileCompletenessCheckResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(serialization_alias="documentId")
    file_id: str = Field(serialization_alias="fileId")
    file_name: str = Field(serialization_alias="fileName")
    overall_status: CheckStatus = Field(serialization_alias="overallStatus")
    summary: str
    item_count: int = Field(serialization_alias="itemCount")
    items: List[CompletenessCheckItem] = Field(default_factory=list)
    statistics: Dict[str, int] = Field(default_factory=dict)


class DocumentSegment(BaseModel):
    segment_id: str
    file_id: str
    text: str
    page_no: Optional[int] = None
    element_ids: List[int] = Field(default_factory=list)
    block_type: str = "paragraph"


class DocumentTypeClassification(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    profile_id: str = Field(alias="profileId", serialization_alias="profileId")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class SectionGroupClassification(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    group_ids: List[str] = Field(default_factory=list, alias="groupIds", serialization_alias="groupIds")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class SectionExistenceJudgment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exists: bool = False
    status: CheckStatus = "uncertain"
    matched_section_name: Optional[str] = Field(
        default="",
        alias="matchedSectionName",
        serialization_alias="matchedSectionName",
    )
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_segment_ids: List[str] = Field(
        default_factory=list,
        alias="evidenceSegmentIds",
        serialization_alias="evidenceSegmentIds",
    )
