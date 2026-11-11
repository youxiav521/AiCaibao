"""
Policy check data models and compatibility helpers.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from src.agents.content_generate_agent.input_models import (
    ExcelTableItem,
    ImageItem,
    Item,
    ListItem,
    PageNumberItem,
    TableItem,
    TextItem,
)
from loguru import logger



PolicyCheckStatus = Literal["pass", "fail", "uncertain", "skipped"]


class SourceLocation(BaseModel):
    """Source location in the policy framework document."""

    fileId: str
    elementIds: List[int] = Field(default_factory=list)
    pageNo: Optional[int] = None


class Control(BaseModel):
    """One control under a requirement."""

    controlId: str
    controlType: str
    risk: str
    content: str = ""
    logicDescription: str
    sourceText: str
    sourceLocation: SourceLocation


class Requirement(BaseModel):
    """One requirement under a framework."""

    requirementId: str
    title: str
    description: str
    sourceText: str
    sourceLocation: SourceLocation
    status: str
    controls: List[Control] = Field(default_factory=list)


class Framework(BaseModel):
    """Policy framework."""

    frameworkId: str
    fileId: str
    name: str
    description: str
    requirements: List[Requirement] = Field(default_factory=list)


class ResourceFile(BaseModel):
    """Parsed resource file."""

    fileName: str
    fileId: str
    originalFileName: Optional[str] = None
    docType: Optional[str] = None
    content: List[Item] = Field(default_factory=list)


class ResourceRepository(BaseModel):
    """Repository of review materials."""

    id: str
    title: str
    description: Optional[str] = None
    files: List[ResourceFile] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Evidence item used in control/requirement results."""

    evidenceId: str
    fileId: str
    elementIds: List[int] = Field(default_factory=list)
    pageNo: Optional[int] = None
    summary: Optional[str] = None
    quotedText: Optional[str] = None


class ControlCheckResult(BaseModel):
    """Control-level check result."""

    requirementId: str
    controlId: str
    result: PolicyCheckStatus
    reason: str
    evidence: List[EvidenceItem] = Field(default_factory=list)


class CheckResult(BaseModel):
    """Requirement-level check result."""

    requirementId: str
    result: PolicyCheckStatus
    reason: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    checkingProcess: Optional[Dict[str, Any]] = None


class PolicyCheckRequest(BaseModel):
    """Request payload for the policy check SSE endpoint."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    resource_repository: ResourceRepository
    frameworks: List[Framework]
    model_id: str = "model_bailian_policy_check"
    max_concurrency: int = 5
    checkReportId: Optional[str] = None
    callbackUrl: Optional[str] = None


def item_content_extract(item: Item) -> Optional[str]:
    """
    Extract readable text from one parsed content item.
    """

    if isinstance(item, TextItem):
        value = item.md
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("TextItem.md is empty and was skipped.")
        return None

    if isinstance(item, TableItem):
        value = item.md
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("TableItem.md is empty and was skipped.")
        return None

    if isinstance(item, ListItem):
        value = item.md
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("ListItem.md is empty and was skipped.")
        return None

    if isinstance(item, ExcelTableItem):
        value = item.md
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("ExcelTableItem.md is empty and was skipped.")
        return None

    if isinstance(item, ImageItem):
        value = item.image_caption if item.image_caption else "No parsed image caption."
        if isinstance(value, str) and value.strip():
            return f"Image caption: {value}"
        logger.warning("ImageItem.image_caption is empty and was skipped.")
        return None

    if isinstance(item, PageNumberItem):
        return None

    logger.warning("Unsupported item type for text extraction: {}", getattr(item, "type", "unknown"))
    return None
