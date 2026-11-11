from __future__ import annotations

import pytest

from src.applications.rag.services.table_semantic_enrichment import build_table_semantic_nodes
 

@pytest.mark.unit
def test_build_table_semantic_nodes_creates_table_and_caption_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure this unit test never hits the LLM (repo .env may enable overwrite).
    monkeypatch.setenv("TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_ENABLED", "false")
    monkeypatch.setenv("TABLE_LLM_CATION", "false")

    mineru_items = [
        {
            "type": "table",
            "element_id": 10,
            "page": 1,
            "md": "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
            "rows": [["A", "B"], ["1", "2"], ["3", "4"]],
            "hasHeader": True,
            "headerCells": [
                [
                    {"text": "A", "colstart": 0, "colspan": 1, "rowstart": 0, "rowspan": 1},
                    {"text": "B", "colstart": 1, "colspan": 1, "rowstart": 0, "rowspan": 1},
                ]
            ],
            "table_caption": ["收入表"],
            "table_footnote": ["单位：万元"],
        }
    ]
    children = [
        {
            "text": "dummy child",
            "metadata": {
                "child_id": "c1",
                "source_element_ids": [10],
                "heading_path": "财务/收入",
                "heading_node_ids": [1, 2],
            },
        }
    ]

    table_children, table_grandchildren = build_table_semantic_nodes(
        mineru_items,
        file_id="file_1",
        children=children,
        folder_id=None,
        file_name="demo.pdf",
    )

    assert len(table_children) == 1
    assert len(table_grandchildren) == 1

    table_doc = table_children[0]
    assert table_doc["text"]
    assert table_doc["metadata"]["type"] == "table"
    assert table_doc["metadata"]["doc_kind"] == "table_node"
    assert table_doc["metadata"]["source_element_ids"] == [10]
    assert table_doc["metadata"]["heading_path"] == "财务/收入"

    summary_doc = table_grandchildren[0]
    assert summary_doc["text"]
    assert summary_doc["metadata"]["type"] == "table"
    assert summary_doc["metadata"]["chunk_strategy"] == "table_caption"
    assert summary_doc["metadata"]["representation"] == "caption"
    assert summary_doc["metadata"]["child_id"] == table_doc["metadata"]["child_id"]
    assert "收入表" in summary_doc["text"]
    assert summary_doc["metadata"]["table_caption_source"] == "mineru"


@pytest.mark.unit
def test_table_caption_overwrite_existing_respects_runtime_env_change(monkeypatch: pytest.MonkeyPatch) -> None:
    # Warm config cache with overwrite disabled (explicit env should win over .env).
    monkeypatch.setenv("TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_ENABLED", "true")
    monkeypatch.setenv("TABLE_LLM_CATION", "false")
    from src.applications.rag.config.rag_settings import RAG_SETTINGS

    assert RAG_SETTINGS.table_semantic_enrichment_llm_caption_overwrite_existing is False

    # Flip overwrite to true without restarting the process.
    monkeypatch.setenv("TABLE_LLM_CATION", "true")

    items = [
        {"type": "text", "element_id": 1, "md": "（一）主要会计数据和财务指标"},
        {
            "type": "table",
            "element_id": 2,
            "page": 1,
            "md": "| A | B |\n|---|---|\n| 1 | 2 |",
            "rows": [["A", "B"], ["1", "2"]],
            "hasHeader": True,
            "headerCells": [
                [
                    {"text": "A", "colstart": 0, "colspan": 1, "rowstart": 0, "rowspan": 1},
                    {"text": "B", "colstart": 1, "colspan": 1, "rowstart": 0, "rowspan": 1},
                ]
            ],
            "table_caption": ["原始标题"],
        },
    ]
    children = [
        {
            "text": "dummy child",
            "metadata": {
                "child_id": "c1",
                "source_element_ids": [1, 2],
                "heading_path": "一、主要财务数据",
                "heading_node_ids": [1],
            },
        }
    ]

    def stub_caption_generator(prompt: str) -> str:
        assert "Table content" in prompt
        return "LLM生成的表格标题"

    table_children, table_grandchildren = build_table_semantic_nodes(
        items,
        file_id="file_1",
        children=children,
        caption_generator=stub_caption_generator,
    )

    assert table_children[0]["metadata"]["table_caption"] == "LLM生成的表格标题"
    assert table_children[0]["metadata"]["table_caption_source"] == "llm"
    assert table_children[0]["metadata"]["table_caption_context_element_ids"] == ["1"]
    assert table_grandchildren[0]["metadata"]["table_caption_source"] == "llm"
