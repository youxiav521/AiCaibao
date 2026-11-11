from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ControlRetrievalRequest(BaseModel):
    control_id: str = ""
    control_text: str
    preferred_doc_types: List[str] = Field(default_factory=list)
    preferred_tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    fact_fields: List[str] = Field(default_factory=list)
    top_k_documents: int = 5
    top_k_blocks: int = 8
    top_k_facts: int = 8
    requirement_id: Optional[str] = None
