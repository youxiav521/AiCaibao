from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.policy.policy_retractor.agent import PolicyRetractorAgent
from src.agents.policy.policy_retractor.extractor import ExtractorConfig, PolicyExtractor
from src.agents.policy.policy_retractor.preprocess import PolicyDocumentLoader
from src.agents.policy.policy_retractor.schemas import ControlType, NormalizedChunk, PolicyDocument, RiskLevel


@pytest.mark.unit
def test_agent_passes_configured_source_level(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_load(self, input_path: str | Path, source_level: str = "auto") -> PolicyDocument:
        captured["source_level"] = source_level
        return PolicyDocument(
            document_id="doc-1",
            file_name="doc.json",
            source_path=str(input_path),
            source_level=source_level,
            chunks=[],
        )

    def fake_extract(self, document: PolicyDocument) -> PolicyDocument:
        return document

    monkeypatch.setattr(PolicyDocumentLoader, "load", fake_load)
    monkeypatch.setattr(PolicyExtractor, "extract", fake_extract)

    agent = PolicyRetractorAgent(config=ExtractorConfig(source_level="children", enable_llm=False))
    result = agent.run("sample-input")

    assert captured["source_level"] == "children"
    assert result.source_level == "children"


@pytest.mark.unit
def test_extractor_does_not_create_model_when_llm_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_create_model(*args, **kwargs):
        raise AssertionError("model provider should not be called when enable_llm is False")

    monkeypatch.setattr("src.agents.policy.policy_retractor.extractor.ModelFactory.get_model", fail_create_model)

    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))

    assert extractor.model is None


@pytest.mark.unit
def test_parse_json_payload_supports_array_fenced_block() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))

    payload = extractor._parse_json_payload(
        """```json
        [
          {"title": "授信准入", "description": "客户需满足准入条件", "controls": []}
        ]
        ```"""
    )

    assert payload["requirements"][0]["title"] == "授信准入"


@pytest.mark.unit
def test_normalize_match_text_strips_common_punctuation() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))

    assert extractor._normalize_match_text(" （客户应提交材料。 ） ") == "客户应提交材料"


@pytest.mark.unit
def test_clean_text_removes_bom_and_heading_marker() -> None:
    loader = PolicyDocumentLoader()

    cleaned = loader._clean_text("\ufeff[H2] 审批条件\n\n正文内容\n\n\n")

    assert cleaned == "正文内容"


@pytest.mark.unit
def test_build_control_from_payload_accepts_camel_case_fields() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))

    control = extractor._build_control_from_payload(
        {
            "content": "客户经理应核验收入证明",
            "controlType": "document",
            "risk": "HIGH",
            "sourceText": "客户经理应核验收入证明",
            "logicDescription": "核验申请材料是否齐全",
            "evidenceText": "收入证明复印件",
        }
    )

    assert control.content == "客户经理应核验收入证明"
    assert control.control_type == ControlType.DOCUMENT
    assert control.risk == RiskLevel.HIGH
    assert control.evidence_text == "收入证明复印件"


@pytest.mark.unit
def test_group_chunks_splits_long_section_without_losing_tail() -> None:
    extractor = PolicyExtractor(
        config=ExtractorConfig(enable_llm=False, max_chars_per_section=10, section_overlap_chunks=1)
    )
    chunks = [
        NormalizedChunk(
            chunk_id="chunk-1",
            source_child_id="chunk-1",
            page=1,
            order=1,
            chunk_type="text",
            heading_path=["第一章", "准入条件"],
            text="12345678",
        ),
        NormalizedChunk(
            chunk_id="chunk-2",
            source_child_id="chunk-2",
            page=1,
            order=2,
            chunk_type="text",
            heading_path=["第一章", "准入条件"],
            text="abcdefgh",
        ),
        NormalizedChunk(
            chunk_id="chunk-3",
            source_child_id="chunk-3",
            page=1,
            order=3,
            chunk_type="text",
            heading_path=["第一章", "准入条件"],
            text="ABCDEFGH",
        ),
    ]

    groups = extractor._group_chunks(chunks)

    assert len(groups) == 3
    assert [chunk.chunk_id for chunk in groups[0]] == ["chunk-1"]
    assert [chunk.chunk_id for chunk in groups[1]] == ["chunk-1", "chunk-2"]
    assert [chunk.chunk_id for chunk in groups[2]] == ["chunk-2", "chunk-3"]


@pytest.mark.unit
def test_group_chunks_skips_heading_only_sections_by_default() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))
    chunks = [
        NormalizedChunk(
            chunk_id="heading-1",
            source_child_id="heading-1",
            page=1,
            order=1,
            chunk_type="heading",
            heading_path=["附件"],
            text="附件",
        ),
        NormalizedChunk(
            chunk_id="text-1",
            source_child_id="text-1",
            page=1,
            order=2,
            chunk_type="text",
            heading_path=["正文"],
            text="第一条 借款人应符合准入条件。",
        ),
    ]

    groups = extractor._group_chunks(chunks)

    assert len(groups) == 1
    assert [chunk.chunk_id for chunk in groups[0]] == ["text-1"]


@pytest.mark.unit
def test_group_chunks_merges_small_adjacent_sections_under_same_root() -> None:
    extractor = PolicyExtractor(
        config=ExtractorConfig(enable_llm=False, min_section_chars=20, merge_small_sections=True)
    )
    chunks = [
        NormalizedChunk(
            chunk_id="chunk-1",
            source_child_id="chunk-1",
            page=1,
            order=1,
            chunk_type="text",
            heading_path=["管理办法", "第一节"],
            text="第一条 目的",
        ),
        NormalizedChunk(
            chunk_id="chunk-2",
            source_child_id="chunk-2",
            page=1,
            order=2,
            chunk_type="text",
            heading_path=["管理办法", "第二节"],
            text="第二条 定义",
        ),
        NormalizedChunk(
            chunk_id="chunk-3",
            source_child_id="chunk-3",
            page=1,
            order=3,
            chunk_type="text",
            heading_path=["银企合作协议", "第一节"],
            text="第三条 合作要求",
        ),
    ]

    groups = extractor._group_chunks(chunks)

    assert len(groups) == 2
    assert [chunk.chunk_id for chunk in groups[0]] == ["chunk-1", "chunk-2"]
    assert [chunk.chunk_id for chunk in groups[1]] == ["chunk-3"]


@pytest.mark.unit
def test_locate_source_location_matches_reference_text() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))
    chunks = [
        NormalizedChunk(
            chunk_id="chunk-1",
            source_child_id="chunk-1",
            page=1,
            order=1,
            chunk_type="text",
            heading_path=["第一章", "准入条件"],
            text="客户应提供营业执照和近一年财务报表。",
        ),
        NormalizedChunk(
            chunk_id="chunk-2",
            source_child_id="chunk-2",
            page=2,
            order=2,
            chunk_type="text",
            heading_path=["第一章", "准入条件"],
            text="客户经理负责尽调并留存影像资料。",
        ),
    ]

    result = extractor._locate_source_location(chunks, "客户应提供营业执照和近一年财务报表。")

    assert result.chunk_ids == ["chunk-1"]
    assert result.page_numbers == [1]
