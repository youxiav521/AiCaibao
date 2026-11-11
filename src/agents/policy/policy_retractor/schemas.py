from enum import Enum
from typing import Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ChildMetadata(BaseModel):
    """Metadata fields for one source child node."""
    page: Optional[int] = None
    order: Optional[int] = None
    type: Optional[str] = None
    heading_path: List[str] = Field(default_factory=list)
    table_caption: Optional[str] = None
    hasHeader: Optional[bool] = None
    rows: List[Any] = Field(default_factory=list)
    raw_md: Optional[str] = None
    html: Optional[str] = None


class ChildNode(BaseModel):
    """Generic source child node used by upstream pipelines."""
    child_id: str
    text: Optional[str] = None
    metadata: ChildMetadata = Field(default_factory=ChildMetadata)


class NormalizedChunk(BaseModel):
    """Normalized input chunk used as the extractor's atomic context unit."""
    chunk_id: str
    source_child_id: str
    page: Optional[int] = None
    order: int = 0
    chunk_type: str
    heading_path: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    text: str = ""
    table_caption: Optional[str] = None
    table_rows: List[Any] = Field(default_factory=list)
    raw_md: Optional[str] = None
    html: Optional[str] = None


class PolicyDocument(BaseModel):
    """Normalized policy document plus ordered chunks."""
    document_id: str
    file_name: str
    source_path: str
    chunks: List[NormalizedChunk] = Field(default_factory=list)


class ControlType(str, Enum):
    """Supported control categories for downstream check execution."""
    QUALIFICATION_CHECK = "qualification_check"
    DOCUMENT_CHECK = "document_check"
    QUANTITATIVE_CHECK = "quantitative_check"
    FUND_USE_CHECK = "fund_use_check"
    COLLATERAL_CHECK = "collateral_check"
    PROHIBITED_ITEM_CHECK = "prohibited_item_check"
    MONITORING_CHECK = "monitoring_check"


class ReviewStatus(str, Enum):
    """Review lifecycle status shared by framework/requirement/control."""
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class RiskLevel(str, Enum):
    """Risk level classification for one control."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SourceLocation(BaseModel):
    """Source locator mapping extracted items back to OCR element ids."""
    model_config = ConfigDict(populate_by_name=True)

    file_id: str = Field(
        default="",
        serialization_alias="fileId",
        validation_alias=AliasChoices("fileId", "file_id", "document_id"),
    )
    element_ids: List[int] = Field(
        default_factory=list,
        serialization_alias="elementIds",
        validation_alias=AliasChoices("elementIds", "elementId", "element_id", "chunkId", "chunk_id"),
    )
    page_no: Optional[int] = Field(
        default=None,
        serialization_alias="pageNo",
        validation_alias=AliasChoices("pageNo", "page_no", "page"),
    )

    @field_validator("element_ids", mode="before")
    @classmethod
    def _normalize_element_ids(cls, value: Any) -> List[int]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            normalized: List[int] = []
            for item in value:
                if isinstance(item, int):
                    normalized.append(item)
                elif isinstance(item, str) and item.isdigit():
                    normalized.append(int(item))
            return normalized
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            tokens = [token.strip() for token in value.split(",")]
            return [int(token) for token in tokens if token.isdigit()]
        return []

    @field_validator("page_no", mode="before")
    @classmethod
    def _normalize_page_no(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None


class Control(BaseModel):
    """Normalized control item extracted from policy text."""
    model_config = ConfigDict(populate_by_name=True)

    control_id: str = Field(serialization_alias="controlId")
    requirement_id: str = Field(default="", serialization_alias="requirementId")
    status: ReviewStatus = Field(default=ReviewStatus.PENDING_REVIEW, serialization_alias="status")
    content: str = Field(
        default="",
        serialization_alias="content",
        validation_alias=AliasChoices("content", "statement", "title"),
    )
    control_type: ControlType = Field(serialization_alias="controlType")
    risk: RiskLevel = Field(default=RiskLevel.MEDIUM, serialization_alias="risk")
    source_text: str = Field(
        default="",
        serialization_alias="sourceText",
        validation_alias=AliasChoices("sourceText", "source_text"),
    )
    logic_description: str = Field(
        default="",
        serialization_alias="logicDescription",
        validation_alias=AliasChoices("logicDescription", "logic_description"),
    )
    source_location: SourceLocation = Field(default_factory=SourceLocation, serialization_alias="sourceLocation")


class Requirement(BaseModel):
    """Normalized requirement item grouping one or more controls."""
    model_config = ConfigDict(populate_by_name=True)

    requirement_id: str = Field(serialization_alias="requirementId")
    framework_id: str = Field(default="", serialization_alias="frameworkId")
    title: str
    description: str
    source_text: str = Field(
        default="",
        serialization_alias="sourceText",
        validation_alias=AliasChoices("sourceText", "source_text"),
    )
    status: ReviewStatus = Field(default=ReviewStatus.PENDING_REVIEW, serialization_alias="status")
    source_location: SourceLocation = Field(default_factory=SourceLocation, serialization_alias="sourceLocation")
    controls: List[Control] = Field(default_factory=list)


class RequirementDraft(BaseModel):
    """Lightweight intermediate requirement schema."""
    title: str
    description: str
    controls: List[Control] = Field(default_factory=list)


class SectionExtractionResult(BaseModel):
    """Extraction result for one section window."""
    section_title: str = ""
    section_summary: str = ""
    requirements: List[Requirement] = Field(default_factory=list)


class FrameworkDraft(BaseModel):
    """Lightweight intermediate framework schema."""
    title: str = Field(validation_alias=AliasChoices("title", "framework_title"))
    summary: str = Field(validation_alias=AliasChoices("summary", "framework_summary"))


class Framework(BaseModel):
    """Final framework output consumed by review and storage flows."""
    model_config = ConfigDict(populate_by_name=True)

    framework_id: str = Field(serialization_alias="frameworkId")
    file_id: str = Field(
        default="",
        serialization_alias="fileId",
        validation_alias=AliasChoices("file_id", "fileId", "document_id"),
    )
    name: str = Field(default="", serialization_alias="name")
    status: ReviewStatus = Field(default=ReviewStatus.PENDING_REVIEW, serialization_alias="status")
    description: str = Field(default="", serialization_alias="description")
    requirements: List[Requirement] = Field(default_factory=list)
