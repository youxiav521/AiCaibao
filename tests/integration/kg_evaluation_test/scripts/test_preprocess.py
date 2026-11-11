from pathlib import Path
from unittest.mock import patch

from tests.integration.kg_evaluation_test.scripts.preprocess import clean_graph, render_table


def graph(references: list[dict]) -> dict:
    return {
        "files": [],
        "entities": [
            {
                "name": "实体",
                "type": "类型",
                "type_en": "TYPE",
                "attributes": {},
                "references": references,
            }
        ],
        "relations": [
            {
                "from_entity": "实体",
                "type": "关系",
                "type_en": "RELATION",
                "to_entity": "另一实体",
                "references": [],
            }
        ],
    }


def test_aggregates_highest_valid_confidence_and_renders_column() -> None:
    payload = clean_graph(
        graph(
            [
                {"confidence": "Low"},
                {"confidence": "unknown"},
                {"confidence": "High"},
            ]
        ),
        Path("graph.json"),
    )

    assert payload["entities"][0]["confidence"] == "High"
    assert payload["relations"][0]["confidence"] is None
    headers = [line for line in render_table(payload).splitlines() if "confidence" in line]
    assert len(headers) == 2
    assert all(header.rstrip().endswith("confidence") for header in headers)


def test_preserves_legacy_output_without_reference_confidence() -> None:
    with patch(
        "tests.integration.kg_evaluation_test.scripts.preprocess.logger.warning"
    ) as warning:
        payload = clean_graph(graph([{"source": "原文"}]), Path("graph.json"))
        table = render_table(payload)

    assert "confidence" not in payload["entities"][0]
    assert "confidence" not in payload["relations"][0]
    assert "confidence" not in table
    warning.assert_not_called()
