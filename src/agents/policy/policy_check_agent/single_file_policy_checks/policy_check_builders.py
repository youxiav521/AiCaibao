from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from ..legacy_schemas import EvidenceItem, Requirement, ResourceFile, ResourceRepository
from ..utils import stable_id
from .schemas import SingleFilePolicyInputFile


def build_resource_file_from_input(input_file: SingleFilePolicyInputFile) -> ResourceFile:
    if not input_file.doc_content:
        raise ValueError("docContent is required")
    file_id = input_file.file_id or build_default_file_id(
        input_file.file_name,
        input_file.doc_content,
    )
    file_name = input_file.file_name or f"{file_id}.json"
    return ResourceFile.model_validate(
        {
            "fileId": file_id,
            "fileName": file_name,
            "content": input_file.doc_content,
        }
    )


def build_default_file_id(file_name: Optional[str], doc_content: List[Dict[str, Any]]) -> str:
    canonical = json.dumps(doc_content or [], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    name = (file_name or "").strip().lower()
    return stable_id("file", name, digest)


def build_single_file_repository(resource_file: ResourceFile) -> ResourceRepository:
    return ResourceRepository(
        id=stable_id("repo", resource_file.fileId, resource_file.fileName),
        title=resource_file.fileName,
        description="single_file_policy_check",
        files=[resource_file],
    )


def build_multi_file_repository(resource_files: List[ResourceFile]) -> ResourceRepository:
    file_fingerprint = [f"{item.fileId}:{item.fileName}" for item in resource_files]
    return ResourceRepository(
        id=stable_id("repo", "hybrid", *file_fingerprint),
        title="single_file_policy_check_hybrid",
        description="single_file_policy_check_hybrid",
        files=resource_files,
    )


def validate_max_concurrency(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"max_concurrency must be an integer >= 1, got: {value!r}")
    if value < 1:
        raise ValueError(f"max_concurrency must be >= 1, got: {value}")
    return value


def control_text(control: Any) -> str:
    return " ".join(
        str(value or "")
        for value in [
            getattr(control, "controlType", ""),
            getattr(control, "risk", ""),
            getattr(control, "logicDescription", ""),
            getattr(control, "sourceText", ""),
        ]
    )


def requirement_text(requirement: Requirement) -> str:
    return " ".join(
        [
            requirement.title or "",
            requirement.description or "",
            requirement.sourceText or "",
            *[
                " ".join(
                    [
                        getattr(control, "controlType", "") or "",
                        getattr(control, "logicDescription", "") or "",
                        getattr(control, "sourceText", "") or "",
                    ]
                )
                for control in requirement.controls
            ],
        ]
    )


def normalize_evidence(items: List[Dict[str, Any]]) -> List[EvidenceItem]:
    normalized: List[EvidenceItem] = []
    for item in items or []:
        normalized.append(EvidenceItem.model_validate(item))
    return normalized
