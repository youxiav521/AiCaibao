from __future__ import annotations

import asyncio

from src.applications.content_generate.services.image_text_evidence_service import (
    TextEvidence,
)
from src.applications.content_generate.services.relationship_graph_service import (
    RelationshipFact,
    RelationshipGraphExtraction,
    RelationshipGraphService,
)
import src.applications.content_generate.services.relationship_graph_service as graph_module


def _evidence() -> TextEvidence:
    return TextEvidence(
        file_id="file-1",
        element_ids=(8,),
        text="甲公司持有乙公司100%股权。",
        score=900,
    )


def _valid_extraction() -> RelationshipGraphExtraction:
    return RelationshipGraphExtraction(
        focus_entity="乙公司",
        relations=[
            RelationshipFact(
                source="甲公司",
                target="乙公司",
                relation="持股",
                percentage="100%",
                evidence_file_id="file-1",
                evidence_element_ids=[8],
            )
        ],
    )


def test_valid_relation_builds_svg_and_citations(monkeypatch) -> None:
    service = RelationshipGraphService()

    async def extract(*_args, **_kwargs):
        return _valid_extraction()

    async def render(_mermaid_code: str) -> str:
        return "data:image/svg+xml;base64,c3Zn"

    monkeypatch.setattr(service, "_extract", extract)
    monkeypatch.setattr(service, "_render", render)

    result = asyncio.run(service.generate("请展示乙公司股权结构图", [_evidence()]))

    assert result is not None
    assert result.data_uri == "data:image/svg+xml;base64,c3Zn"
    assert result.citations[0].file_id == "file-1"
    assert result.citations[0].element_ids == [8]
    assert result.citations[0].quote == "甲公司持有乙公司100%股权。"


def test_generic_image_description_uses_evidence_backed_focus_entity(monkeypatch) -> None:
    """图片题可用 fileIds 限定资料范围，不要求 description 写出企业名。"""
    service = RelationshipGraphService()

    async def extract(*_args, **_kwargs):
        return _valid_extraction()

    async def render(_mermaid_code: str) -> str:
        return "data:image/svg+xml;base64,c3Zn"

    monkeypatch.setattr(service, "_extract", extract)
    monkeypatch.setattr(service, "_render", render)

    result = asyncio.run(service.generate("企业股权架构说明", [_evidence()]))

    assert result is not None
    assert result.data_uri == "data:image/svg+xml;base64,c3Zn"
    assert result.citations[0].file_id == "file-1"


def test_relation_with_unknown_evidence_is_rejected(monkeypatch) -> None:
    service = RelationshipGraphService()
    invalid = _valid_extraction()
    invalid.relations[0].evidence_element_ids = [999]

    async def extract(*_args, **_kwargs):
        return invalid

    monkeypatch.setattr(service, "_extract", extract)

    result = asyncio.run(service.generate("请展示乙公司股权结构图", [_evidence()]))

    assert result is None


def test_relation_not_supported_by_evidence_text_is_rejected(monkeypatch) -> None:
    service = RelationshipGraphService()
    invalid = _valid_extraction()
    invalid.relations[0].source = "丙公司"

    async def extract(*_args, **_kwargs):
        return invalid

    monkeypatch.setattr(service, "_extract", extract)

    result = asyncio.run(service.generate("请展示乙公司股权结构图", [_evidence()]))

    assert result is None


def test_extraction_has_a_bounded_timeout_and_no_outer_retry(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def invoke(*_args, **kwargs):
        captured.update(kwargs)
        return _valid_extraction()

    monkeypatch.setattr(graph_module, "ainvoke_llm_with_structured_output", invoke)

    result = asyncio.run(
        RelationshipGraphService()._extract("Please show Beta ownership", [_evidence()])
    )

    assert result == _valid_extraction()
    assert captured["timeout_s"] == 45
    assert captured["max_retries"] == 0


def test_report_company_pronoun_is_resolved_only_from_matching_file_name(monkeypatch) -> None:
    service = RelationshipGraphService()
    evidence = TextEvidence(
        file_id="file-1",
        element_ids=(8,),
        text="\u4e0a\u6d77\u56fd\u9645\u96c6\u56e2\u6709\u9650\u516c\u53f8\u6301\u6709\u672c\u516c\u53f8 0.71% \u80a1\u4efd",
        score=900,
        file_name="\u4e0a\u6d77\u519c\u5546\u94f6\u884c\u80a1\u4efd\u6709\u9650\u516c\u53f82025\u5e74\u5e74\u5ea6\u62a5\u544a.pdf",
    )
    extraction = RelationshipGraphExtraction(
        focus_entity="\u4e0a\u6d77\u519c\u6751\u5546\u4e1a\u94f6\u884c",
        relations=[
            RelationshipFact(
                source="\u4e0a\u6d77\u56fd\u9645\u96c6\u56e2\u6709\u9650\u516c\u53f8",
                target="\u4e0a\u6d77\u519c\u6751\u5546\u4e1a\u94f6\u884c",
                relation="\u6301\u6709\u80a1\u4efd",
                percentage="0.71%",
                evidence_file_id="file-1",
                evidence_element_ids=[8],
            )
        ],
    )

    async def extract(*_args, **_kwargs):
        return extraction

    async def render(_mermaid_code: str) -> str:
        return "data:image/svg+xml;base64,c3Zn"

    monkeypatch.setattr(service, "_extract", extract)
    monkeypatch.setattr(service, "_render", render)

    result = asyncio.run(
        service.generate("\u8bf7\u5c55\u793a\u4e0a\u6d77\u519c\u6751\u5546\u4e1a\u94f6\u884c\u80a1\u6743\u7ed3\u6784\u56fe", [evidence])
    )

    assert result is not None
    assert result.citations[0].file_id == "file-1"


def test_invalid_first_extraction_gets_one_evidence_bound_repair(monkeypatch) -> None:
    service = RelationshipGraphService()
    calls: list[bool] = []
    invalid = RelationshipGraphExtraction(
        focus_entity="\u4e0d\u5728\u95ee\u9898\u91cc\u7684\u5b9e\u4f53",
        relations=[],
    )

    async def extract(*_args, repair: bool = False, **_kwargs):
        calls.append(repair)
        return _valid_extraction() if repair else invalid

    async def render(_mermaid_code: str) -> str:
        return "data:image/svg+xml;base64,c3Zn"

    monkeypatch.setattr(service, "_extract", extract)
    monkeypatch.setattr(service, "_render", render)

    result = asyncio.run(service.generate("\u8bf7\u5c55\u793a\u4e59\u516c\u53f8\u80a1\u6743\u7ed3\u6784\u56fe", [_evidence()]))

    assert result is not None
    assert calls == [False, True]


def test_explicit_holding_sentence_is_used_when_model_extractions_are_invalid(monkeypatch) -> None:
    service = RelationshipGraphService()
    evidence = TextEvidence(
        file_id="file-1",
        element_ids=(8,),
        text="\u4e0a\u6d77\u56fd\u9645\u96c6\u56e2\u6709\u9650\u516c\u53f8\u6301\u6709\u672c\u516c\u53f8 68,638,200 \u80a1\uff0c\u5360\u672c\u516c\u53f8\u603b\u80a1\u672c 0.71%\u3002",
        score=900,
        file_name="\u4e0a\u6d77\u519c\u5546\u94f6\u884c\u80a1\u4efd\u6709\u9650\u516c\u53f82025\u5e74\u5e74\u5ea6\u62a5\u544a.pdf",
    )

    async def extract(*_args, **_kwargs):
        return RelationshipGraphExtraction(focus_entity="\u65e0\u6548\u5b9e\u4f53", relations=[])

    async def render(_mermaid_code: str) -> str:
        return "data:image/svg+xml;base64,c3Zn"

    monkeypatch.setattr(service, "_extract", extract)
    monkeypatch.setattr(service, "_render", render)

    result = asyncio.run(
        service.generate("\u8bf7\u751f\u6210\u4e0a\u6d77\u56fd\u9645\u96c6\u56e2\u6301\u6709\u4e0a\u6d77\u519c\u6751\u5546\u4e1a\u94f6\u884c\u7684\u80a1\u6743\u7ed3\u6784\u56fe", [evidence])
    )

    assert result is not None
    assert "0.71%" in result.mermaid_code


def test_generic_ownership_description_uses_explicit_report_holding(monkeypatch) -> None:
    service = RelationshipGraphService()
    evidence = TextEvidence(
        file_id="file-1",
        element_ids=(8,),
        text="上海国际集团有限公司持有本公司 68,638,200 股，占本公司总股本 0.71%。",
        score=900,
        file_name="上海农商银行股份有限公司2025年年度报告.pdf",
    )

    async def extract(*_args, **_kwargs):
        return RelationshipGraphExtraction(focus_entity="无效实体", relations=[])

    async def render(_mermaid_code: str) -> str:
        return "data:image/svg+xml;base64,c3Zn"

    monkeypatch.setattr(service, "_extract", extract)
    monkeypatch.setattr(service, "_render", render)

    result = asyncio.run(service.generate("企业股权架构说明", [evidence]))

    assert result is not None
    assert "0.71%" in result.mermaid_code


def test_connected_ownership_chain_keeps_intermediate_relationships(monkeypatch) -> None:
    service = RelationshipGraphService()
    evidence = TextEvidence(
        file_id="file-1",
        element_ids=(8, 9),
        text="甲公司持有乙公司100%股权；乙公司持有丙公司50%股权。",
        score=900,
    )
    extraction = RelationshipGraphExtraction(
        focus_entity="丙公司",
        relations=[
            RelationshipFact(
                source="甲公司", target="乙公司", relation="持股", percentage="100%",
                evidence_file_id="file-1", evidence_element_ids=[8, 9],
            ),
            RelationshipFact(
                source="乙公司", target="丙公司", relation="持股", percentage="50%",
                evidence_file_id="file-1", evidence_element_ids=[8, 9],
            ),
        ],
    )

    async def extract(*_args, **_kwargs):
        return extraction

    async def render(_mermaid_code: str) -> str:
        return "data:image/svg+xml;base64,c3Zn"

    monkeypatch.setattr(service, "_extract", extract)
    monkeypatch.setattr(service, "_render", render)

    result = asyncio.run(service.generate("请展示丙公司股权结构图", [evidence]))

    assert result is not None
    assert "甲公司" in result.mermaid_code
    assert "乙公司" in result.mermaid_code
    assert "丙公司" in result.mermaid_code


def test_partial_evidence_element_ids_still_create_a_traceable_citation(monkeypatch) -> None:
    service = RelationshipGraphService()
    evidence = TextEvidence(
        file_id="file-1",
        element_ids=(8, 9),
        text="甲公司持有乙公司100%股权。",
        score=900,
    )
    extraction = RelationshipGraphExtraction(
        focus_entity="乙公司",
        relations=[
            RelationshipFact(
                source="甲公司", target="乙公司", relation="持股", percentage="100%",
                evidence_file_id="file-1", evidence_element_ids=[8],
            )
        ],
    )

    async def extract(*_args, **_kwargs):
        return extraction

    async def render(_mermaid_code: str) -> str:
        return "data:image/svg+xml;base64,c3Zn"

    monkeypatch.setattr(service, "_extract", extract)
    monkeypatch.setattr(service, "_render", render)

    result = asyncio.run(service.generate("请展示乙公司股权结构图", [evidence]))

    assert result is not None
    assert len(result.citations) == 1
    assert result.citations[0].element_ids == [8, 9]
