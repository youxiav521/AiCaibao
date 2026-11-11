from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.ade.knowledge_graph.nebula.client import NebulaClient
from src.ade.knowledge_graph.nebula.repository import NebulaGraphRepository
from src.ade.knowledge_graph.nebula.schema import (
    EDGE_PROPERTIES,
    ENTITY_TAG_BY_TYPE,
    RELATION_EDGE_BY_TYPE,
    SNAPSHOT_TAG,
    VERTEX_PROPERTIES,
    folder_key,
    render_schema,
)
from src.ade.knowledge_graph.schema.kg_schema import (
    Entity,
    KnowledgeGraph,
    Reference,
    Relation,
    entity_id,
    relation_id,
)
from src.ade.knowledge_graph.service.entity_vector_store import EntityVectorStore
from src.ade.knowledge_graph.service.graph_storage_service import (
    GraphStorageService,
)
from src.base.storage_service import storage_service
from src.utils.json_utils import loads as json_loads

PROJECT_ROOT = Path(__file__).parents[4]


def _graph() -> KnowledgeGraph:
    folder_id = "folder-nebula-unit"
    person_id = entity_id(folder_id, "张三", "自然人")
    enterprise_id = entity_id(folder_id, "示例公司", "企业主体")
    reference = Reference(
        source="张三持有示例公司股份",
        element_id=1,
        file_id="file-1",
        source_matched=True,
    )
    return KnowledgeGraph(
        id="job-123",
        folderId=folder_id,
        folderName="测试项目",
        entities=[
            Entity(
                id=person_id,
                name="张三",
                type="自然人",
                type_en="PERSON",
                attributes={"职位": "董事"},
                references=[reference],
                degree=9,
                embed=[0.1, 0.2],
            ),
            Entity(
                id=enterprise_id,
                name="示例公司",
                type="企业主体",
                type_en="ENTERPRISE",
                attributes={},
                references=[],
                degree=4,
            ),
        ],
        relations=[
            Relation(
                id=relation_id(person_id, enterprise_id, "股东"),
                type="股东",
                type_en="SHAREHOLDER_OF",
                from_entity="张三",
                from_entity_id=person_id,
                to_entity="示例公司",
                to_entity_id=enterprise_id,
                attributes={"持股比例": ["10%"]},
                references=[reference],
                degree=13,
                embed=[0.3],
            )
        ],
    )


def test_schema_is_generated_from_semantic_types() -> None:
    assert tuple(ENTITY_TAG_BY_TYPE.values()) == (
        "enterprise",
        "person",
        "mortgage_asset",
        "pledge_asset",
        "creditor",
        "risk_event",
        "project",
        "unknown",
    )
    assert len(RELATION_EDGE_BY_TYPE) == 19
    assert RELATION_EDGE_BY_TYPE["法定代表人"] == "legal_representative"
    assert VERTEX_PROPERTIES == (
        "folder_key",
        "folder_id",
        "name",
        "attributes_json",
        "references_json",
    )
    assert EDGE_PROPERTIES == ("attributes_json", "references_json")

    ddl = render_schema()
    assert ddl == (
        PROJECT_ROOT / "docker/local/nebula/schema.ngql"
    ).read_text(encoding="utf-8")
    for omitted_property in (
        "degree",
        "embed_json",
        "entity_id",
        "relation_id",
        "type_en",
    ):
        assert omitted_property not in ddl
    assert f"CREATE TAG IF NOT EXISTS `{SNAPSHOT_TAG}`" in ddl
    assert "`graph_id` string NOT NULL" in ddl
    assert ddl.count("CREATE TAG INDEX") == 8
    assert "CREATE EDGE INDEX" not in ddl


def test_database_projection_omits_business_only_fields() -> None:
    graph = _graph()

    vertices, edges = NebulaGraphRepository._prepare(graph)

    assert {vertex.folder_key for vertex in vertices} == {
        folder_key(graph.folderId)
    }
    assert {vertex.tag for vertex in vertices} == {"person", "enterprise"}
    assert json_loads(vertices[0].attributes_json) == {"职位": "董事"}
    assert len(json_loads(vertices[0].references_json)) == 1
    assert len(edges) == 1
    assert edges[0].edge_type == "shareholder_of"
    assert json_loads(edges[0].attributes_json) == {"持股比例": ["10%"]}

    assert "degree" in Entity.model_fields
    assert "degree" in Relation.model_fields
    assert graph.entities[0].degree == 9
    assert graph.relations[0].degree == 13


def test_unknown_stub_entity_uses_unknown_tag() -> None:
    graph = _graph()
    stub = Entity(
        id=entity_id(graph.folderId, "未解析主体", "未知"),
        name="未解析主体",
        type="未知",
        type_en="UNKNOWN",
        references=[],
    )
    graph.entities.append(stub)

    vertices, _ = NebulaGraphRepository._prepare(graph)

    assert next(vertex for vertex in vertices if vertex.vid == stub.id).tag == "unknown"


@pytest.mark.asyncio
async def test_minio_keeps_degree_while_nebula_projection_omits_it(
    monkeypatch,
) -> None:
    graph = _graph()
    uploaded: list[bytes] = []

    async def upload(_bucket, _name, data, _content_type) -> None:
        uploaded.append(data)

    replace = AsyncMock()
    monkeypatch.setattr(storage_service, "upload", upload)
    monkeypatch.setattr(EntityVectorStore, "rebuild", AsyncMock())
    monkeypatch.setattr(NebulaClient, "is_configured", lambda: True)
    monkeypatch.setattr(NebulaGraphRepository, "replace", replace)

    await GraphStorageService._save_locked(graph)

    saved = json_loads(uploaded[0])
    assert saved["id"] == "job-123"
    assert saved["entities"][0]["degree"] == 9
    assert saved["relations"][0]["degree"] == 13
    replace.assert_awaited_once_with(graph, acquire_lock=False)


@pytest.mark.asyncio
async def test_nebula_failure_does_not_fail_authoritative_save(
    monkeypatch,
) -> None:
    graph = _graph()

    upload = AsyncMock()
    monkeypatch.setattr(storage_service, "upload", upload)
    monkeypatch.setattr(EntityVectorStore, "rebuild", AsyncMock())
    monkeypatch.setattr(NebulaClient, "is_configured", lambda: True)
    monkeypatch.setattr(
        NebulaGraphRepository,
        "replace",
        AsyncMock(side_effect=RuntimeError("nebula unavailable")),
    )

    assert await GraphStorageService._save_locked(graph) == graph.id
    upload.assert_awaited_once()
