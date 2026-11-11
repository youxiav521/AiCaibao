from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict

from ..legacy_schemas import ResourceRepository
from ..services.material_retriever import MaterialRetriever
from .policy_relevance_profiles import DocumentProfile


@dataclass(slots=True)
class PolicyCheckRuntimeContext:
    model_id: str
    max_concurrency: int
    relevance_threshold: float
    repository: ResourceRepository
    retriever: MaterialRetriever
    execution_semaphore: asyncio.Semaphore
    document_profiles: Dict[str, DocumentProfile] = field(default_factory=dict)
