from __future__ import annotations

import asyncio
import re

from src.agents.policy.policy_retractor.extractor import ExtractorConfig, PolicyExtractor
from src.agents.policy.policy_retractor.preprocess import PolicyDocumentLoader
from src.agents.policy.policy_retractor.schemas import (
    Control,
    ControlType,
    NormalizedChunk,
    PolicyDocument,
    Requirement,
    RiskLevel,
    SourceLocation,
)


def test_loader_prefers_upstream_file_id_from_payload() -> None:
    loader = PolicyDocumentLoader()

    payload = [
        {
            "metadata": {
                "fileId": "FILE-123",
            }
        }
    ]

    assert loader._extract_document_id(payload) == "FILE-123"


def test_loader_accepts_java_payload_shape() -> None:
    loader = PolicyDocumentLoader()

    document = loader.load_payload(
        {
            "fileId": "FILE-789",
            "docContent": [
                {
                    "page": 1,
                    "type": "text",
                    "value": "测试正文",
                    "md": "测试正文",
                    "element_id": 1,
                    "file_id": "FILE-789",
                }
            ],
        }
    )

    assert document.document_id == "FILE-789"
    assert document.file_name == "java_payload.json"
    assert len(document.chunks) == 1
    assert document.chunks[0].source_child_id == "1"


def test_build_framework_uses_fw_file_id_as_framework_id() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))
    document = PolicyDocument(
        document_id="FILE-456",
        file_name="demo.json",
        source_path="demo.json",
        chunks=[],
    )

    framework = extractor._build_framework(
        document=document,
        framework_draft={},
        section_results=[],
    )

    assert framework.framework_id == "FW-FILE-456"
    assert framework.file_id == "FILE-456"


def test_generated_requirement_and_control_ids_match_java_style() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))

    requirement_id = extractor._new_requirement_id()
    control_id = extractor._new_control_id()

    assert requirement_id.startswith("REQ-")
    assert control_id.startswith("RULE-")
    assert re.fullmatch(r"REQ-[0-9a-f]{32}", requirement_id)
    assert re.fullmatch(r"RULE-[0-9a-f]{32}", control_id)


def test_section_extraction_assigns_java_style_ids() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))
    requirement = Requirement(
        requirement_id="",
        title="准入要求",
        description="企业需满足准入要求",
        source_text="企业需满足准入要求",
        source_location=SourceLocation(),
        controls=[
            Control(
                control_id="",
                content="核验营业执照",
                control_type=ControlType.DOCUMENT_CHECK,
                risk=RiskLevel.MEDIUM,
                source_text="核验营业执照",
                logic_description="核验营业执照",
            )
        ],
    )

    result = extractor._finalize_section_requirements(
        section_chunks=[],
        requirement_payloads=[requirement],
    )

    assert len(result) == 1
    assert result[0].requirement_id.startswith("REQ-")
    assert len(result[0].controls) == 1
    assert result[0].controls[0].control_id.startswith("RULE-")
    assert result[0].controls[0].requirement_id == result[0].requirement_id


def test_requirement_callback_emits_per_requirement_with_framework_fields() -> None:
    captured: list[Requirement] = []

    async def on_requirement(requirement: Requirement) -> None:
        captured.append(requirement)

    extractor = PolicyExtractor(
        config=ExtractorConfig(enable_llm=False),
        requirement_callback=on_requirement,
    )
    document = PolicyDocument(
        document_id="FILE-999",
        file_name="demo.json",
        source_path="demo.json",
        chunks=[],
    )
    requirement = Requirement(
        requirement_id="REQ-1",
        title="测试要求",
        description="测试要求描述",
        source_text="测试要求原文",
        source_location=SourceLocation(),
        controls=[
            Control(
                control_id="RULE-1",
                requirement_id="REQ-1",
                content="测试控制",
                control_type=ControlType.DOCUMENT_CHECK,
                risk=RiskLevel.MEDIUM,
                source_text="测试控制原文",
                logic_description="测试控制逻辑",
                source_location=SourceLocation(),
            )
        ],
    )

    normalized = extractor._attach_framework_fields(document, requirement)
    asyncio.run(extractor._emit_requirement_callbacks([normalized]))

    assert len(captured) == 1
    assert captured[0].framework_id == "FW-FILE-999"
    assert captured[0].source_location.file_id == "FILE-999"
    assert captured[0].controls[0].source_location.file_id == "FILE-999"


def test_control_trace_does_not_fallback_to_requirement_when_control_text_is_unmatched() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))
    section_chunks = [
        NormalizedChunk(
            chunk_id="chunk-1",
            source_child_id="101",
            page=1,
            order=1,
            chunk_type="text",
            heading_path=["section"],
            text="客户应提交营业执照。",
        ),
        NormalizedChunk(
            chunk_id="chunk-2",
            source_child_id="102",
            page=1,
            order=2,
            chunk_type="text",
            heading_path=["section"],
            text="客户应提交近一年财务报表。",
        ),
    ]
    requirement = Requirement(
        requirement_id="",
        title="资料要求",
        description="客户应提交营业执照。",
        source_text="客户应提交营业执照。",
        source_location=SourceLocation(),
        controls=[
            Control(
                control_id="",
                content="核查抵押登记证明。",
                control_type=ControlType.DOCUMENT_CHECK,
                risk=RiskLevel.MEDIUM,
                source_text="核查抵押登记证明。",
                logic_description="核查抵押登记证明。",
            )
        ],
    )

    result = extractor._finalize_section_requirements(section_chunks, [requirement])[0]

    assert result.source_location.element_ids == [101]
    assert result.controls[0].source_location.element_ids == []


def test_locate_source_location_returns_empty_when_reference_is_unmatched() -> None:
    extractor = PolicyExtractor(config=ExtractorConfig(enable_llm=False))
    chunks = [
        NormalizedChunk(
            chunk_id="chunk-1",
            source_child_id="201",
            page=3,
            order=1,
            chunk_type="text",
            heading_path=["section"],
            text="申请材料应由法人签章。",
        )
    ]

    result = extractor._locate_source_location(chunks, "完全不相关的条文")

    assert result.element_ids == []
    assert result.page_no is None
