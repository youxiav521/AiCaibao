from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.applications.parsing.service.parse_json_service import JsonParsingService
from src.applications.rag.chunking.plain_text import (
    make_chunks_for_json_items,
    build_grandchildren_from_children_text,
)


def _write_tmp_json(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.mark.unit
def test_json_parsing_and_hierarchical_chunking(tmp_path: Path) -> None:
    payload = {
        "meta": {"source": "unit-test", "desc": "y" * 400},
        "users": [{"id": i, "name": f"User{i}", "bio": "x" * 80} for i in range(120)],
    }
    json_path = _write_tmp_json(tmp_path, payload)

    items = JsonParsingService().parse_to_items(str(json_path))
    assert items, "should produce json elements"

    # simulate element_id injection (normally done in MinerUParsingService before indexing)
    for idx, it in enumerate(items):
        it["element_id"] = idx + 1

    parents, children = make_chunks_for_json_items(
        items,
        file_id="file_1",
        parent_chunk_size=240,
        child_chunk_size=80,
        orphan_min=40,
        folder_id=None,
        file_name="sample.json",
    )

    assert parents, "should build parent chunks"
    assert children, "should build child chunks"

    # parent/child chunks should be valid JSON objects (or wrapped root dict)
    for p in parents:
        assert isinstance(p.get("text"), str) and p["text"].strip()
        assert isinstance(json.loads(p["text"]), dict)

    for c in children:
        assert isinstance(c.get("text"), str) and c["text"].strip()
        assert isinstance(json.loads(c["text"]), dict)

    grandchildren = build_grandchildren_from_children_text(
        children,
        file_id="file_1",
        chunk_size=60,
        chunk_overlap=0,
        strategy="json",
        folder_id=None,
        file_name="sample.json",
    )

    assert grandchildren, "should build json grandchildren"
    assert any(":" in g.get("text", "") for g in grandchildren)
    assert all(
        g.get("metadata", {}).get("chunk_strategy") == "json_flatten_grandchild" for g in grandchildren
    )

