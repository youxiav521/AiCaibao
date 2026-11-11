"""NebulaGraph schema derived from the knowledge-graph type definitions."""

from __future__ import annotations

import hashlib
import re

from src.ade.knowledge_graph.client.schema.builder import SchemaBuilder
from src.ade.knowledge_graph.client.schema.definitions import RELATION_TYPES

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

VERTEX_PROPERTIES = (
    "folder_key",
    "folder_id",
    "name",
    "attributes_json",
    "references_json",
)
EDGE_PROPERTIES = ("attributes_json", "references_json")
SNAPSHOT_TAG = "kg_snapshot"
SNAPSHOT_PROPERTIES = (
    "folder_id",
    "graph_id",
    "snapshot_version",
    "ready",
)


def validate_identifier(value: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid NebulaGraph identifier: {value!r}")
    return value


def _database_identifier(value: str) -> str:
    return validate_identifier(value.lower())


ENTITY_TAG_BY_TYPE = {
    type_name: _database_identifier(type_en)
    for type_name, type_en in SchemaBuilder.entity_type_map().items()
}
RELATION_EDGE_BY_TYPE = {
    type_name: _database_identifier(type_en)
    for type_name, type_en in SchemaBuilder.relation_type_map().items()
}
ENTITY_TYPE_BY_TAG = {tag: type_name for type_name, tag in ENTITY_TAG_BY_TYPE.items()}
RELATION_TYPE_BY_EDGE = {
    edge: type_name for type_name, edge in RELATION_EDGE_BY_TYPE.items()
}
ENTITY_TYPE_EN_BY_TYPE = SchemaBuilder.entity_type_map()
RELATION_TYPE_EN_BY_TYPE = SchemaBuilder.relation_type_map()
BIDIRECTIONAL_RELATION_TYPES = frozenset(
    definition.name for definition in RELATION_TYPES if definition.is_bidirectional
)

if len(ENTITY_TYPE_BY_TAG) != len(ENTITY_TAG_BY_TYPE):
    raise ValueError("Duplicate NebulaGraph entity Tag identifier")
if len(RELATION_TYPE_BY_EDGE) != len(RELATION_EDGE_BY_TYPE):
    raise ValueError("Duplicate NebulaGraph Edge Type identifier")


def folder_key(folder_id: str) -> str:
    return hashlib.sha256(folder_id.encode()).hexdigest()


def schema_statements(space: str = "knowledge_graph") -> tuple[str, ...]:
    space = validate_identifier(space)
    statements = [f"USE `{space}`"]

    vertex_fields = (
        "`folder_key` fixed_string(64) NOT NULL",
        "`folder_id` string NOT NULL",
        "`name` string NOT NULL",
        '`attributes_json` string NOT NULL DEFAULT "{}"',
        '`references_json` string NOT NULL DEFAULT "[]"',
    )
    edge_fields = (
        '`attributes_json` string NOT NULL DEFAULT "{}"',
        '`references_json` string NOT NULL DEFAULT "[]"',
    )
    snapshot_fields = (
        "`folder_id` string NOT NULL",
        "`graph_id` string NOT NULL",
        "`snapshot_version` string NOT NULL",
        "`ready` bool NOT NULL DEFAULT false",
    )

    statements.append(
        f"CREATE TAG IF NOT EXISTS `{SNAPSHOT_TAG}` "
        f"({', '.join(snapshot_fields)})"
    )
    statements.extend(
        f"CREATE TAG IF NOT EXISTS `{tag}` ({', '.join(vertex_fields)})"
        for tag in ENTITY_TYPE_BY_TAG
    )
    statements.extend(
        f"CREATE EDGE IF NOT EXISTS `{edge}` ({', '.join(edge_fields)})"
        for edge in RELATION_TYPE_BY_EDGE
    )
    statements.extend(
        f"CREATE TAG INDEX IF NOT EXISTS `i_{tag}_folder_key` "
        f"ON `{tag}` (`folder_key`)"
        for tag in ENTITY_TYPE_BY_TAG
    )
    return tuple(statements)


def render_schema(space: str = "knowledge_graph") -> str:
    return ";\n".join(schema_statements(space)) + ";\n"


def render_schema_for_prompt() -> str:
    """Render the physical Nebula schema and edge directions for NL-to-nGQL."""
    lines = [
        "顶点 Tag（业务属性仅存在 attributes_json，由应用回填，不在查询中直接使用）：",
        "| Tag | 中文类型 | 可查询属性 |",
        "|---|---|---|",
    ]
    for type_name, tag in ENTITY_TAG_BY_TYPE.items():
        lines.append(
            f"| {tag} | {type_name} | folder_key, folder_id, name |"
        )

    lines.extend(
        [
            "",
            "边类型（方向固定为 source → target；双向边物理上只存一个方向）：",
            "| Edge | 中文关系 | source Tag | target Tag | 查询方向 |",
            "|---|---|---|---|---|",
        ]
    )
    for definition in RELATION_TYPES:
        source_tags = "/".join(
            ENTITY_TAG_BY_TYPE[type_name]
            for type_name in definition.source_types
        )
        target_tags = "/".join(
            ENTITY_TAG_BY_TYPE[type_name]
            for type_name in definition.target_types
        )
        direction = "BIDIRECT" if definition.is_bidirectional else "source → target"
        lines.append(
            f"| {RELATION_EDGE_BY_TYPE[definition.name]} | {definition.name} | "
            f"{source_tags} | {target_tags} | {direction} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_schema(), end="")
