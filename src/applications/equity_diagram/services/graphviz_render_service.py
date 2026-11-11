"""Render validated graph data to SVG with the Graphviz ``dot`` executable."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shutil
from dataclasses import dataclass


MAX_DOT_BYTES = 128 * 1024
MAX_SVG_BYTES = 4 * 1024 * 1024
DEFAULT_RENDER_TIMEOUT_SECONDS = 30.0
DEFAULT_GRAPHVIZ_FONT_NAME = "Noto Sans CJK SC"
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class GraphvizRenderError(RuntimeError):
    """Graphviz rejected the graph or produced an invalid SVG."""


class GraphvizUnavailableError(GraphvizRenderError):
    """The Graphviz ``dot`` executable is not installed or configured."""


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    kind: str = "other"
    group_id: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    label: str


@dataclass(frozen=True)
class GraphGroup:
    id: str
    label: str


def build_dot_source(
    *,
    title: str,
    layout: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    groups: list[GraphGroup] | None = None,
) -> str:
    """Convert a bounded graph model to deterministic DOT source."""
    if not nodes or not edges:
        raise ValueError("关系图至少需要一个节点和一条关系")

    direction = layout.strip().upper()
    if direction not in {"TB", "LR"}:
        raise ValueError("Graphviz 布局仅支持 TB 或 LR")

    node_by_id: dict[str, GraphNode] = {}
    for node in nodes:
        _validate_id(node.id, "节点")
        if node.id in node_by_id:
            raise ValueError(f"节点 ID 重复: {node.id}")
        if not node.label.strip():
            raise ValueError(f"节点名称不能为空: {node.id}")
        node_by_id[node.id] = node

    group_by_id: dict[str, GraphGroup] = {}
    for group in groups or []:
        _validate_id(group.id, "分组")
        if group.id in group_by_id:
            raise ValueError(f"分组 ID 重复: {group.id}")
        if not group.label.strip():
            raise ValueError(f"分组名称不能为空: {group.id}")
        group_by_id[group.id] = group

    for node in nodes:
        if node.group_id and node.group_id not in group_by_id:
            raise ValueError(f"节点 {node.id} 引用了不存在的分组 {node.group_id}")
    for edge in edges:
        if edge.source_id not in node_by_id or edge.target_id not in node_by_id:
            raise ValueError("关系边引用了不存在的节点")
        if edge.source_id == edge.target_id:
            raise ValueError("不允许节点指向自身")

    lines = [
        "digraph EvidenceGraph {",
        "  graph [",
        f'    rankdir="{direction}",',
        '    bgcolor="transparent",',
        '    pad="0.35",',
        '    nodesep="0.45",',
        '    ranksep="0.75",',
        '    splines="polyline",',
        '    outputorder="edgesfirst",',
        f'    fontname="{_dot_text(_graphviz_font_name())}",',
        f'    label="{_dot_text(title)}",',
        '    labelloc="t",',
        '    fontsize="20"',
        "  ];",
        '  node [shape="box", style="rounded,filled", color="#2563EB", '
        'fillcolor="#EFF6FF", fontcolor="#0F172A", penwidth="1.4", '
        f'fontname="{_dot_text(_graphviz_font_name())}", fontsize="13", '
        'margin="0.16,0.10"];',
        '  edge [color="#475569", fontcolor="#334155", penwidth="1.2", '
            f'arrowsize="0.75", fontname="{_dot_text(_graphviz_font_name())}", '
        'fontsize="11"];',
    ]

    grouped_node_ids: set[str] = set()
    for group in groups or []:
        group_nodes = [node for node in nodes if node.group_id == group.id]
        if not group_nodes:
            continue
        lines.extend(
            [
                f"  subgraph cluster_{group.id} {{",
                f'    label="{_dot_text(group.label)}";',
                '    color="#CBD5E1";',
                '    fillcolor="#F8FAFC";',
                '    style="rounded,filled";',
                '    penwidth="1.0";',
            ]
        )
        for node in group_nodes:
            lines.append(f"    {_node_statement(node)}")
            grouped_node_ids.add(node.id)
        lines.append("  }")

    for node in nodes:
        if node.id not in grouped_node_ids:
            lines.append(f"  {_node_statement(node)}")

    for edge in edges:
        label = _dot_text(_wrap_label(edge.label, 18))
        lines.append(
            f'  {edge.source_id} -> {edge.target_id} [label="{label}"];'
        )
    lines.append("}")

    source = "\n".join(lines)
    if len(source.encode("utf-8")) > MAX_DOT_BYTES:
        raise ValueError(f"DOT 内容不能超过 {MAX_DOT_BYTES} 字节")
    return source


async def render_dot_to_svg(
    dot_source: str,
    *,
    executable: str | None = None,
    timeout_seconds: float | None = None,
) -> bytes:
    """Run Graphviz without a shell and return a bounded SVG document."""
    source = str(dot_source or "").strip()
    if not source:
        raise ValueError("DOT 内容不能为空")
    if len(source.encode("utf-8")) > MAX_DOT_BYTES:
        raise ValueError(f"DOT 内容不能超过 {MAX_DOT_BYTES} 字节")

    configured_path = (
        executable or os.getenv("GRAPHVIZ_DOT_PATH", "dot")
    ).strip()
    resolved_path = shutil.which(configured_path)
    if not resolved_path:
        raise GraphvizUnavailableError(
            "未找到 Graphviz dot，请安装 Graphviz 或配置 GRAPHVIZ_DOT_PATH"
        )

    process = await asyncio.create_subprocess_exec(
        resolved_path,
        "-Tsvg",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(source.encode("utf-8")),
            timeout=timeout_seconds or DEFAULT_RENDER_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise GraphvizRenderError("Graphviz 渲染超时") from exc

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[:2_000]
        raise GraphvizRenderError(
            f"Graphviz 渲染失败: returncode={process.returncode}, detail={detail}"
        )
    if not stdout or len(stdout) > MAX_SVG_BYTES or b"<svg" not in stdout[:1_024].lower():
        raise GraphvizRenderError("Graphviz 返回的 SVG 为空、过大或格式错误")
    return stdout


def svg_to_data_uri(svg_bytes: bytes) -> str:
    if not svg_bytes:
        raise ValueError("SVG 内容不能为空")
    if len(svg_bytes) > MAX_SVG_BYTES:
        raise ValueError(f"SVG 不能超过 {MAX_SVG_BYTES} 字节")
    return "data:image/svg+xml;base64," + base64.b64encode(svg_bytes).decode("ascii")


def _validate_id(value: str, kind: str) -> None:
    if not _SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{kind} ID 只能使用字母、数字和下划线，且必须以字母开头: {value}")


def _node_statement(node: GraphNode) -> str:
    label = _dot_text(_wrap_label(node.label, 16))
    fill, border = _node_colors(node.kind)
    return (
        f'{node.id} [label="{label}", fillcolor="{fill}", '
        f'color="{border}"];'
    )


def _graphviz_font_name() -> str:
    return os.getenv("GRAPHVIZ_FONT_NAME", DEFAULT_GRAPHVIZ_FONT_NAME).strip() or DEFAULT_GRAPHVIZ_FONT_NAME


def _node_colors(kind: str) -> tuple[str, str]:
    return {
        "entity": ("#FFF7ED", "#EA580C"),
        "person": ("#F5F3FF", "#7C3AED"),
        "governance": ("#ECFDF5", "#059669"),
        "management": ("#EFF6FF", "#2563EB"),
        "department": ("#F8FAFC", "#64748B"),
        "business": ("#FFFBEB", "#D97706"),
        "target": ("#F0FDF4", "#16A34A"),
    }.get(kind.strip().lower(), ("#F8FAFC", "#64748B"))


def _dot_text(value: str) -> str:
    normalized = _CONTROL_CHARACTER_PATTERN.sub("", str(value or "")).strip()
    return (
        normalized.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "")
        .replace("\n", "\\n")
    )


def _wrap_label(value: str, width: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= width:
        return text
    return "\n".join(text[index : index + width] for index in range(0, len(text), width))
