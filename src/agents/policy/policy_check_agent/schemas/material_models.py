from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentManifest(BaseModel):
    doc_id: str
    file_id: str
    file_name: str
    doc_type: Optional[str] = None
    page_count: int = 0
    has_table: bool = False
    has_excel_table: bool = False
    candidate_tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceBlock(BaseModel):
    evidence_id: str
    doc_id: str
    file_id: str
    file_name: str
    page: Optional[int] = None
    block_type: str
    source_type: str
    text: str
    source_element_ids: List[int] = Field(default_factory=list)
    bbox: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FactRecord(BaseModel):
    fact_id: str
    doc_id: str
    evidence_id: str
    fact_type: str
    field_name: str
    field_value: str
    normalized_value: Optional[str] = None
    confidence: Optional[float] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MaterialStore(BaseModel):
    case_id: str
    manifests: List[DocumentManifest] = Field(default_factory=list)
    evidence_blocks: List[EvidenceBlock] = Field(default_factory=list)
    facts: List[FactRecord] = Field(default_factory=list)
