import asyncio
from pathlib import Path

import pytest

from src.ade.knowledge_graph.nebula.client import NebulaClient
from src.ade.knowledge_graph.nebula.repository import NebulaGraphRepository
from src.ade.knowledge_graph.nebula.schema import (
    BIDIRECTIONAL_RELATION_TYPES,
    ENTITY_TAG_BY_TYPE,
    RELATION_EDGE_BY_TYPE,
)
from src.ade.knowledge_graph.schema.kg_schema import (
    Entity,
    KnowledgeGraph,
    entity_id,
    relation_id,
)
from src.base.redis_client import RedisClient
from src.utils.json_utils import load as json_load

SAMPLE_PATH = (
    Path(__file__).parent
    / "kg_evaluation_test/scripts/preprocess_results/12.json"
)


def _current_ids(graph: KnowledgeGraph, folder_id: str) -> KnowledgeGraph:
    graph = graph.model_copy(deep=True)
    graph.id = "nebula-integration-job"
    graph.folderId = folder_id

    old_to_new: dict[str, str] = {}
    for entity in graph.entities:
        old_to_new[entity.id] = entity_id(folder_id, entity.name, entity.type)
        entity.id = old_to_new[entity.id]

    for relation in graph.relations:
        relation.from_entity_id = old_to_new[relation.from_entity_id]
        relation.to_entity_id = old_to_new[relation.to_entity_id]
        id_source, id_target = relation.from_entity_id, relation.to_entity_id
        if relation.type in BIDIRECTIONAL_RELATION_TYPES:
            id_source, id_target = sorted((id_source, id_target))
        relation.id = relation_id(id_source, id_target, relation.type)
    return graph


def _show(statement: str) -> list[dict]:
    with NebulaClient.session() as session:
        return NebulaClient.execute(session, statement)


def _first_column(rows: list[dict]) -> set[str]:
    return {str(next(iter(row.values()))) for row in rows}


@pytest.mark.integration
async def test_nebula_schema_and_folder_snapshot_lifecycle(
    load_env, monkeypatch
) -> None:
    monkeypatch.setenv("NEBULA_HOST", "127.0.0.1")
    monkeypatch.setenv("NEBULA_PORT", "9669")
    monkeypatch.setenv("NEBULA_USER", "root")
    monkeypatch.setenv("NEBULA_PASSWORD", "nebula")
    monkeypatch.setenv("NEBULA_SPACE", "knowledge_graph")
    folder_id = "nebula-integration-folder-12"

    await RedisClient.close()
    NebulaClient.close()
    try:
        await NebulaGraphRepository.initialize_schema()
        await NebulaGraphRepository.initialize_schema()

        tags, edges, tag_indexes, edge_indexes = await asyncio.gather(
            asyncio.to_thread(_show, "SHOW TAGS"),
            asyncio.to_thread(_show, "SHOW EDGES"),
            asyncio.to_thread(_show, "SHOW TAG INDEXES"),
            asyncio.to_thread(_show, "SHOW EDGE INDEXES"),
        )
        assert _first_column(tags) == set(ENTITY_TAG_BY_TYPE.values())
        assert _first_column(edges) == set(RELATION_EDGE_BY_TYPE.values())
        assert _first_column(tag_indexes) == {
            f"i_{tag}_folder_key" for tag in ENTITY_TAG_BY_TYPE.values()
        }
        assert edge_indexes == []

        graph = _current_ids(
            KnowledgeGraph.model_validate(json_load(SAMPLE_PATH)),
            folder_id,
        )
        assert len(graph.entities) == 271
        assert len(graph.relations) == 201

        await NebulaGraphRepository.replace(graph)
        restored = await NebulaGraphRepository.get(folder_id)
        assert restored is not None
        assert restored.totalEntities == 271
        assert restored.totalRelations == 201

        expected_entities = {entity.id: entity for entity in graph.entities}
        for entity in restored.entities:
            expected = expected_entities[entity.id]
            assert (
                entity.name,
                entity.type,
                entity.type_en,
                entity.attributes,
                entity.references,
            ) == (
                expected.name,
                expected.type,
                expected.type_en,
                expected.attributes,
                expected.references,
            )

        with_stub = graph.model_copy(deep=True)
        stub = Entity(
            id=entity_id(folder_id, "未解析主体", "未知"),
            name="未解析主体",
            type="未知",
            type_en="UNKNOWN",
            references=[],
        )
        stub_relation = with_stub.relations[0].model_copy(deep=True)
        stub_relation.from_entity = stub.name
        stub_relation.from_entity_id = stub.id
        id_source, id_target = stub.id, stub_relation.to_entity_id
        if stub_relation.type in BIDIRECTIONAL_RELATION_TYPES:
            id_source, id_target = sorted((id_source, id_target))
        stub_relation.id = relation_id(
            id_source,
            id_target,
            stub_relation.type,
        )
        with_stub.entities.append(stub)
        with_stub.relations.append(stub_relation)

        await NebulaGraphRepository.replace(with_stub)
        restored_with_stub = await NebulaGraphRepository.get(folder_id)
        assert restored_with_stub is not None
        assert restored_with_stub.totalEntities == 272
        assert restored_with_stub.totalRelations == 202
        assert next(
            entity for entity in restored_with_stub.entities if entity.id == stub.id
        ).type == "未知"

        selected_ids = {
            graph.relations[0].from_entity_id,
            graph.relations[0].to_entity_id,
            *(entity.id for entity in graph.entities[:28]),
        }
        smaller = graph.model_copy(deep=True)
        smaller.entities = [
            entity for entity in smaller.entities if entity.id in selected_ids
        ]
        smaller.relations = [
            relation
            for relation in smaller.relations
            if relation.from_entity_id in selected_ids
            and relation.to_entity_id in selected_ids
        ]
        assert 0 < len(smaller.relations) < len(graph.relations)

        await NebulaGraphRepository.replace(smaller)
        replaced = await NebulaGraphRepository.get(folder_id)
        assert replaced is not None
        assert {entity.id for entity in replaced.entities} == selected_ids
        assert {relation.id for relation in replaced.relations} == {
            relation.id for relation in smaller.relations
        }
    finally:
        await NebulaGraphRepository.delete(folder_id)
        assert await NebulaGraphRepository.get(folder_id) is None
        NebulaClient.close()
        await RedisClient.close()
