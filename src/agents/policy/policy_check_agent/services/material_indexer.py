from __future__ import annotations

from typing import Dict, List

from ..schemas.material_models import MaterialStore
from ..utils import keyword_tokens


class MaterialIndexer:
    def attach_indexes(self, store: MaterialStore) -> MaterialStore:
        manifest_index = {manifest.doc_id: manifest.model_dump() for manifest in store.manifests}
        block_keyword_index: Dict[str, List[str]] = {}
        fact_index: Dict[str, List[str]] = {}

        for block in store.evidence_blocks:
            for token in keyword_tokens(f"{block.file_name} {block.text}"):
                block_keyword_index.setdefault(token, []).append(block.evidence_id)

        for fact in store.facts:
            fact_index.setdefault(fact.field_name, []).append(fact.fact_id)
            for token in keyword_tokens(f"{fact.field_name} {fact.field_value} {fact.normalized_value or ''}"):
                fact_index.setdefault(token, []).append(fact.fact_id)

        setattr(store, "_manifest_index", manifest_index)
        setattr(store, "_block_keyword_index", block_keyword_index)
        setattr(store, "_fact_index", fact_index)
        return store
