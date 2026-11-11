from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class DocType(str, Enum):
    BUSINESS_LICENSE = "BUSINESS_LICENSE"
    LEGAL_REP_ID = "LEGAL_REP_ID"
    LEGAL_REP_MARRIAGE_CERT = "LEGAL_REP_MARRIAGE_CERT"
    SPOUSE_ID = "SPOUSE_ID"
    LEGAL_REP_CERT = "LEGAL_REP_CERT"
    ARTICLES_OF_ASSOCIATION = "ARTICLES_OF_ASSOCIATION"
    SIGNATURE_SAMPLE = "SIGNATURE_SAMPLE"
    CREDIT_REPORT = "CREDIT_REPORT"
    COLLATERAL_APPRAISAL = "COLLATERAL_APPRAISAL"
    CAPITAL_VERIFICATION_REPORT = "CAPITAL_VERIFICATION_REPORT"
    CONTRACT_AND_PAYMENT = "CONTRACT_AND_PAYMENT"
    OWNERSHIP_STRUCTURE = "OWNERSHIP_STRUCTURE"
    AUDIO = "AUDIO"
    MEETING_MINUTES = "MEETING_MINUTES"
    VIDEO = "VIDEO"
    TAX_REG_CERT = "TAX_REG_CERT"
    UNKNOWN = "UNKNOWN"


class DocSignal(BaseModel):
    source: Literal["filename", "content", "structure", "rule"]
    token: str
    weight: float
    matched_span: str | None = Field(default=None, alias="matchedSpan")


class DocProfile(BaseModel):
    primary_type: DocType = Field(alias="primaryType")
    doc_types: List[DocType] = Field(default_factory=list, alias="docTypes")
    confidence: float = 0.0
    is_media: bool = Field(default=False, alias="isMedia")
    scores: Dict[str, float] = Field(default_factory=dict)
    signals: List[DocSignal] = Field(default_factory=list)
    raw_hints: Dict[str, Any] = Field(default_factory=dict, alias="rawHints")

