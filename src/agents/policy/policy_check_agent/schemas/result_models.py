from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from .material_models import DocumentManifest, EvidenceBlock, FactRecord


class DocumentHit(BaseModel):
    manifest: DocumentManifest
    score: float
    reasons: List[str] = Field(default_factory=list)


class EvidenceHit(BaseModel):
    block: EvidenceBlock
    score: float
    reasons: List[str] = Field(default_factory=list)


class FactHit(BaseModel):
    fact: FactRecord
    score: float
    reasons: List[str] = Field(default_factory=list)


class ControlRetrievalResult(BaseModel):
    control_id: str = ""
    control_text: str
    candidate_documents: List[DocumentHit] = Field(default_factory=list)
    evidence_hits: List[EvidenceHit] = Field(default_factory=list)
    fact_hits: List[FactHit] = Field(default_factory=list)
    debug: Dict[str, object] = Field(default_factory=dict)
