"""Generate a cited graph model from RAG evidence and render it with Graphviz."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from src.agents.content_generate_agent.models import Citation
from src.applications.equity_diagram.services.graphviz_render_service import (
    GraphEdge,
    GraphGroup,
    GraphNode,
    build_dot_source,
    render_dot_to_svg,
    svg_to_data_uri,
)

from .image_text_evidence_service import TextEvidence, build_bounded_evidence_text


_MODEL_KEY = "image_summary"
_LLM_TIMEOUT_SECONDS = 120
# Organization charts can contain dozens of nodes and cited edges. 6144
# tokens is too small for a complete structured JSON response and can leave
# the response truncated before the closing brace.
_LLM_MAX_OUTPUT_TOKENS = 16_384
_MAX_NODES = 80
_MAX_EDGES = 100
_MAX_GROUPS = 16
_SAFE_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
_PERCENTAGE_PATTERN = re.compile(r"(?:100|\d{1,2})(?:\.\d+)?%")
_OWNERSHIP_INTENT_KEYWORDS = ("股权", "股东", "持股", "控股", "实际控制", "穿透")
_ORGANIZATION_INTENT_KEYWORDS = (
    "组织架构",
    "组织结构",
    "组织机构",
    "组织图",
    "治理架构",
    "治理结构",
    "部门架构",
    "汇报关系",
)


class EvidenceGraphNode(BaseModel):
    id: str = Field(pattern=_SAFE_ID_PATTERN)
    label: str = Field(min_length=1, max_length=160)
    # The model may use more specific labels (for example ``committee`` or
    # ``division``). Keep the wire format permissive and normalize those
    # labels to the small set used by the renderer.
    kind: str = Field(default="other", min_length=1, max_length=40)
    group_id: str | None = Field(default=None, pattern=_SAFE_ID_PATTERN)

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> str:
        normalized = str(value or "other").strip().lower()
        aliases = {
            "body": "governance",
            "committee": "governance",
            "board": "governance",
            "executive": "management",
            "research": "department",
            "division": "business",
            "company": "entity",
            "individual": "person",
        }
        return aliases.get(normalized, normalized if normalized in {
            "entity", "person", "governance", "management", "department",
            "business", "target", "other",
        } else "other")


class EvidenceGraphEdge(BaseModel):
    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    target_id: str = Field(pattern=_SAFE_ID_PATTERN)
    evidence_file_id: str = Field(min_length=1)
    evidence_element_ids: list[int] = Field(min_length=1, max_length=20)
    evidence_quote: str = Field(min_length=1, max_length=2_000)
    evidence_relation_text: str = Field(min_length=1, max_length=300)


class EvidenceGraphGroup(BaseModel):
    id: str = Field(pattern=_SAFE_ID_PATTERN)
    label: str = Field(min_length=1, max_length=160)


class EvidenceGraphExtraction(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    diagram_type: Literal["ownership", "organization", "relationship", "other"]
    layout: Literal["TB", "LR"] = "TB"
    nodes: list[EvidenceGraphNode] = Field(default_factory=list, max_length=_MAX_NODES)
    edges: list[EvidenceGraphEdge] = Field(default_factory=list, max_length=_MAX_EDGES)
    groups: list[EvidenceGraphGroup] = Field(default_factory=list, max_length=_MAX_GROUPS)

    @model_validator(mode="after")
    def validate_references(self) -> "EvidenceGraphExtraction":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("节点 ID 不能重复")
        group_ids = [group.id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("分组 ID 不能重复")
        known_nodes = set(node_ids)
        known_groups = set(group_ids)
        for node in self.nodes:
            if node.group_id and node.group_id not in known_groups:
                raise ValueError(f"节点 {node.id} 引用了不存在的分组")
        for edge in self.edges:
            if edge.source_id not in known_nodes or edge.target_id not in known_nodes:
                raise ValueError("关系边引用了不存在的节点")
            if edge.source_id == edge.target_id:
                raise ValueError("关系边不能指向自身")
        return self


@dataclass(frozen=True)
class EvidenceGraphResult:
    data_uri: str
    citations: list[Citation]
    dot_source: str


class EvidenceGraphService:
    """Use the LLM for semantics and Graphviz only for deterministic layout."""

    async def generate(
        self,
        question: str,
        evidence: list[TextEvidence],
    ) -> EvidenceGraphResult | None:
        if not question.strip() or not evidence:
            return None

        extraction = await self._extract(question, evidence)
        validated = self._validate_extraction(question, extraction, evidence)
        if validated is None:
            logger.warning("关系图首次抽取没有通过证据校验，进行一次纠正 | question={}", question)
            extraction = await self._extract(question, evidence, repair=True)
            validated = self._validate_extraction(question, extraction, evidence)
        if validated is None:
            return None

        graph_nodes, graph_edges, graph_groups, valid_edges = validated
        dot_source = build_dot_source(
            title=extraction.title,
            layout=extraction.layout,
            nodes=graph_nodes,
            edges=graph_edges,
            groups=graph_groups,
        )
        svg = await render_dot_to_svg(dot_source)
        return EvidenceGraphResult(
            data_uri=svg_to_data_uri(svg),
            citations=self._build_citations(valid_edges),
            dot_source=dot_source,
        )

    @staticmethod
    async def _extract(
        question: str,
        evidence: list[TextEvidence],
        *,
        repair: bool = False,
    ) -> EvidenceGraphExtraction:
        from src.utils.llm_invoke import ainvoke_llm_with_structured_output

        evidence_text = build_bounded_evidence_text(evidence)
        repair_instruction = ""
        if repair:
            repair_instruction = """
The previous graph was rejected because its edges did not carry exact supporting
quotes or referenced unsupported entities. Rebuild it conservatively. Copy each
evidence_quote verbatim from one evidence record and keep only supported edges.
"""
        prompt = f"""Return one valid JSON object only.

You convert allowed enterprise-document evidence into a complete relationship
graph. The user may request an ownership chart, shareholder chart, organization
chart, governance chart, reporting hierarchy, control chart, or another factual
relationship diagram.

User request:
{question}

Allowed RAG evidence:
{evidence_text}

Output fields:
- title: concise diagram title.
- diagram_type: ownership, organization, relationship, or other.
- layout: TB for layered hierarchies, or LR for long relationship chains.
- groups: optional logical layers such as governance, management, departments,
  research institutes, business divisions, or ownership branches.
- nodes: id, label, kind, and optional group_id. IDs must start with an ASCII
  letter and contain only ASCII letters, digits, and underscores.
- edges: source_id, target_id, evidence_file_id, evidence_element_ids,
  evidence_quote, and evidence_relation_text.

Rules:
1. Use only entities and relationships explicitly supported by the evidence.
2. Determine the requested diagram type from the user request. Never turn an
   organization question into an ownership graph, or the reverse.
3. Include all important supported layers, not just one local branch. Keep the
   result below {_MAX_NODES} nodes and {_MAX_EDGES} edges.
4. Every edge must cite exactly one printed file_id and one or more printed
   element_ids. evidence_quote must be copied verbatim from that same record and
   must directly support both endpoints and their relationship. Keep each quote
   to the shortest complete supporting sentence or table row; do not repeat the
   whole evidence record.
5. evidence_relation_text must be the shortest verbatim substring of
   evidence_quote that expresses the edge relationship, such as “持有其36%股份”,
   “向董事会汇报”, or “董事会下设”. Never paraphrase this field.
6. A company's full name may represent “本公司” only when the evidence file_name
   clearly identifies that company. Do not infer a company from a generic query.
7. Preserve exact percentages, names and reporting directions. Do not complete
   missing facts from general knowledge.
8. If no relationship is supported, return empty nodes, edges, and groups.
9. Do not output Mermaid, DOT, Markdown, explanations, HTML, or external facts.
{repair_instruction}
"""
        logger.info(
            "证据关系图调用模型 | model_key={} evidence_count={} evidence_chars={} "
            "prompt_chars={} repair={}",
            _MODEL_KEY,
            len(evidence),
            len(evidence_text),
            len(prompt),
            repair,
        )
        return await ainvoke_llm_with_structured_output(
            prompt,
            model_id=_MODEL_KEY,
            response_type=EvidenceGraphExtraction,
            timeout_s=_LLM_TIMEOUT_SECONDS,
            max_retries=0,
            max_tokens=_LLM_MAX_OUTPUT_TOKENS,
        )

    @staticmethod
    def _validate_extraction(
        question: str,
        extraction: EvidenceGraphExtraction,
        evidence: list[TextEvidence],
    ) -> tuple[
        list[GraphNode],
        list[GraphEdge],
        list[GraphGroup],
        list[EvidenceGraphEdge],
    ] | None:
        if not _diagram_type_matches_question(question, extraction.diagram_type):
            logger.warning(
                "关系图类型与用户问题不一致 | question={} diagram_type={}",
                question,
                extraction.diagram_type,
            )
            return None
        node_by_id = {node.id: node for node in extraction.nodes}
        render_edges: list[GraphEdge] = []
        cited_edges: list[EvidenceGraphEdge] = []
        seen_edges: set[tuple[str, str, str, str, tuple[int, ...]]] = set()
        for edge in extraction.edges:
            source = node_by_id.get(edge.source_id)
            target = node_by_id.get(edge.target_id)
            if source is None or target is None or not edge.evidence_relation_text.strip():
                logger.warning(
                    "Invalid graph edge structure; skip rendering | source_id={} target_id={}",
                    edge.source_id,
                    edge.target_id,
                )
                continue
            item = _find_supporting_evidence(edge, source, target, evidence)
            verified = item is not None
            if not verified:
                logger.warning(
                    "关系边未通过证据校验 | source={} target={} relation={} file_id={} element_ids={}",
                    source.label,
                    target.label,
                    edge.evidence_relation_text,
                    edge.evidence_file_id,
                    edge.evidence_element_ids,
                )
            key = (
                edge.source_id,
                edge.target_id,
                edge.evidence_relation_text.strip(),
                edge.evidence_file_id.strip(),
                tuple(sorted(set(edge.evidence_element_ids))),
            )
            if key in seen_edges:
                continue
            seen_edges.add(key)
            label = edge.evidence_relation_text.strip()
            if not verified:
                label = f"{label} [unverified]"
            render_edges.append(
                GraphEdge(source_id=edge.source_id, target_id=edge.target_id, label=label)
            )
            if verified:
                cited_edges.append(edge)

        if not render_edges:
            return None

        logger.info(
            "关系图边处理完成 | total={} rendered={} cited={} unverified={}",
            len(extraction.edges),
            len(render_edges),
            len(cited_edges),
            len(render_edges) - len(cited_edges),
        )

        used_node_ids = {
            node_id
            for edge in render_edges
            for node_id in (edge.source_id, edge.target_id)
        }
        valid_nodes = [node for node in extraction.nodes if node.id in used_node_ids]
        used_group_ids = {node.group_id for node in valid_nodes if node.group_id}
        valid_groups = [group for group in extraction.groups if group.id in used_group_ids]
        return (
            [
                GraphNode(
                    id=node.id,
                    label=node.label.strip(),
                    kind=node.kind,
                    group_id=node.group_id,
                )
                for node in valid_nodes
            ],
            render_edges,
            [GraphGroup(id=group.id, label=group.label.strip()) for group in valid_groups],
            cited_edges,
        )

    @staticmethod
    def _build_citations(edges: list[EvidenceGraphEdge]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, tuple[int, ...], str]] = set()
        for edge in edges:
            file_id = edge.evidence_file_id.strip()
            element_ids = tuple(sorted(set(edge.evidence_element_ids)))
            quote = edge.evidence_quote.strip()
            key = (file_id, element_ids, quote)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    id=len(citations) + 1,
                    file_id=file_id,
                    element_ids=list(element_ids),
                    quote=quote,
                )
            )
        return citations


def _find_supporting_evidence(
    edge: EvidenceGraphEdge,
    source: EvidenceGraphNode,
    target: EvidenceGraphNode,
    evidence: list[TextEvidence],
) -> TextEvidence | None:
    file_id = edge.evidence_file_id.strip()
    element_ids = set(edge.evidence_element_ids)
    quote = edge.evidence_quote.strip()
    normalized_quote = _normalize_text(quote)
    normalized_relation_text = _normalize_text(edge.evidence_relation_text)
    if (
        not file_id
        or not element_ids
        or not normalized_quote
        or not normalized_relation_text
    ):
        return None

    # RAG may split one source sentence/table across adjacent child chunks.
    # Validate against the combined evidence from the same file instead of
    # requiring every endpoint and relation phrase in one individual chunk.
    file_items = [item for item in evidence if item.file_id == file_id]
    if not file_items:
        return None
    available_element_ids = {
        element_id for item in file_items for element_id in item.element_ids
    }
    if not element_ids.issubset(available_element_ids):
        return None

    combined_text = "\n".join(item.text for item in file_items)
    normalized_combined_text = _normalize_text(combined_text)
    if normalized_quote not in normalized_combined_text:
        return None
    if normalized_relation_text not in normalized_combined_text:
        return None
    file_name = next((item.file_name for item in file_items if item.file_name), "")
    if not _label_supported(source.label, combined_text, file_name):
        return None
    if not _label_supported(target.label, combined_text, file_name):
        return None
    if any(
        percentage not in normalized_combined_text
        for percentage in _PERCENTAGE_PATTERN.findall(normalized_relation_text)
    ):
        return None

    for item in file_items:
        if element_ids.intersection(item.element_ids):
            return item
    return None


def _label_supported(label: str, quote: str, file_name: str) -> bool:
    normalized_label = _normalize_text(label)
    normalized_quote = _normalize_text(quote)
    if normalized_label and normalized_label in normalized_quote:
        return True
    if "本公司" not in normalized_quote and "公司" not in normalized_quote:
        return False
    return bool(_company_aliases(label) & _company_aliases(file_name))


def _company_aliases(value: str) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()
    normalized = re.sub(r"\.(?:pdf|docx?|xlsx?)$", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"20\d{2}年(?:年度)?报告.*$", "", normalized)
    aliases = {normalized}
    for suffix in ("股份有限公司", "有限责任公司", "有限公司"):
        if normalized.endswith(suffix):
            aliases.add(normalized[: -len(suffix)])
    return {alias for alias in aliases if len(alias) >= 2}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).replace("％", "%")


def _diagram_type_matches_question(question: str, diagram_type: str) -> bool:
    normalized = _normalize_text(question)
    ownership_text = normalized.replace("股东大会", "")
    asks_ownership = any(
        keyword in ownership_text for keyword in _OWNERSHIP_INTENT_KEYWORDS
    )
    asks_organization = any(
        keyword in normalized for keyword in _ORGANIZATION_INTENT_KEYWORDS
    )
    if asks_organization and not asks_ownership:
        return diagram_type == "organization"
    if asks_ownership and not asks_organization:
        return diagram_type == "ownership"
    return True
