from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


DocClass = Literal[
    "business_license",
    "identity_document",
    "bank_statement",
    "credit_report",
    "credit_application",
    "due_diligence_report",
    "compliance_material",
    "transaction_voucher",
    "collateral_document",
    "contract",
    "invoice",
    "approval_form",
    "authorization_letter",
    "financial_statement",
    "board_resolution",
    "articles_of_association",
    "other",
]


class EvidenceLocator(BaseModel):
    fileId: str
    fileName: str
    pageNo: Optional[int] = None
    elementId: Optional[int] = None
    chunkId: Optional[str] = None
    bbox: Optional[List[float]] = None


class DocumentClassification(BaseModel):
    docId: str
    fileId: str
    fileName: str
    docClass: DocClass
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    matchedRules: List[str] = Field(default_factory=list)


class CanonicalChunk(BaseModel):
    chunkId: str
    pageNo: int
    elementId: Optional[int] = None
    chunkType: Literal["text", "table", "list", "image_caption", "metadata"]
    text: str
    bbox: Optional[List[float]] = None
    tokensApprox: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CanonicalTableRow(BaseModel):
    rowIndex: int
    values: List[str] = Field(default_factory=list)


class CanonicalTable(BaseModel):
    tableId: str
    pageNo: int
    elementId: Optional[int] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[CanonicalTableRow] = Field(default_factory=list)
    rawMarkdown: Optional[str] = None


class CanonicalFieldValue(BaseModel):
    entityType: Literal["company", "person", "account", "document", "unknown"] = "unknown"
    canonicalField: str
    value: str
    normalizedValue: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[EvidenceLocator] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)


class CanonicalPage(BaseModel):
    pageNo: int
    chunks: List[CanonicalChunk] = Field(default_factory=list)
    tables: List[CanonicalTable] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CanonicalDocument(BaseModel):
    docId: str
    fileId: str
    fileName: str
    originalFileName: Optional[str] = None
    docClass: DocClass = "other"
    classification: Optional[DocumentClassification] = None
    pages: List[CanonicalPage] = Field(default_factory=list)
    canonicalFields: List[CanonicalFieldValue] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceIndex(BaseModel):
    docsByClass: Dict[str, List[str]] = Field(default_factory=dict)
    docsByEntity: Dict[str, List[str]] = Field(default_factory=dict)
    docsByField: Dict[str, List[str]] = Field(default_factory=dict)
    fieldValues: Dict[str, List[CanonicalFieldValue]] = Field(default_factory=dict)
    chunkTextIndex: Dict[str, List[EvidenceLocator]] = Field(default_factory=dict)
    docSummaries: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class RetrievalPlan(BaseModel):
    controlText: str
    docClasses: List[DocClass] = Field(default_factory=list)
    canonicalFields: List[str] = Field(default_factory=list)
    keywordQueries: List[str] = Field(default_factory=list)
    entityHints: List[str] = Field(default_factory=list)
    reasoning: str = ""


class MaterialUnderstandingResult(BaseModel):
    repositoryId: str
    documents: List[CanonicalDocument] = Field(default_factory=list)
    evidenceIndex: EvidenceIndex = Field(default_factory=EvidenceIndex)
