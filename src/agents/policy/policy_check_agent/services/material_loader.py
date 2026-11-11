from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import ResourceFile, ResourceRepository
from ..schemas.material_models import MaterialStore
from ..utils import stable_id
from .fact_extractor import FactExtractor
from .material_cleaner import MaterialCleaner
from .material_indexer import MaterialIndexer
from .material_normalizer import MaterialNormalizer
from .table_parser import TableParser


class MaterialLoader:
    def load_combined_documents(self, source: str | Path | list[dict[str, Any]] | ResourceRepository) -> ResourceRepository:
        if isinstance(source, ResourceRepository):
            return source
        if isinstance(source, (str, Path)):
            raw = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            raw = source
        if not isinstance(raw, list):
            raise ValueError("combined_documents input must be a list of file objects")
        files = [ResourceFile.model_validate(item) for item in raw]
        return ResourceRepository(
            id=stable_id("repo", len(files), *(file.fileId for file in files[:5])),
            title="combined_documents",
            description="material store source",
            files=files,
        )


def build_material_store(
    raw_documents: str | Path | list[dict[str, Any]] | ResourceRepository,
    case_id: str | None = None,
) -> MaterialStore:
    repository = MaterialLoader().load_combined_documents(raw_documents)
    cleaner = MaterialCleaner()
    normalizer = MaterialNormalizer()
    table_parser = TableParser()
    fact_extractor = FactExtractor()
    indexer = MaterialIndexer()

    manifests = []
    evidence_blocks = []
    facts = []
    for resource_file in repository.files:
        cleaned_items = cleaner.clean_file(resource_file)
        manifest, normalized_blocks = normalizer.normalize_file(resource_file, cleaned_items)
        table_blocks, table_facts = table_parser.parse_file(resource_file, manifest, cleaned_items)
        fact_records = fact_extractor.extract_for_document(resource_file, manifest, normalized_blocks + table_blocks)
        manifests.append(manifest)
        evidence_blocks.extend(normalized_blocks)
        evidence_blocks.extend(table_blocks)
        facts.extend(table_facts)
        facts.extend(fact_records)

    store = MaterialStore(
        case_id=case_id or repository.id,
        manifests=manifests,
        evidence_blocks=evidence_blocks,
        facts=facts,
    )
    return indexer.attach_indexes(store)
