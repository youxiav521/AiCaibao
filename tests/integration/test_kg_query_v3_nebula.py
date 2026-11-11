from unittest.mock import AsyncMock

import pytest

from src.ade.knowledge_graph.nebula.client import NebulaClient
from src.ade.knowledge_graph.nebula.repository import NebulaGraphRepository
from src.ade.knowledge_graph.nebula.schema import folder_key
from src.ade.knowledge_graph.query.ngql_guard import compile_statement
from src.ade.knowledge_graph.query.query_service_v3 import QueryServiceV3
from src.ade.knowledge_graph.schema.kg_schema import (
    Entity,
    KnowledgeGraph,
    Reference,
    Relation,
    entity_id,
    relation_id,
)
from src.ade.knowledge_graph.service.entity_vector_store import (
    AnchorHit,
    EntityVectorStore,
)
from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService


def _graph(folder_id: str) -> tuple[KnowledgeGraph, dict[str, str]]:
    ids = {
        "company": entity_id(folder_id, "比亚迪股份有限公司", "企业主体"),
        "representative": entity_id(folder_id, "王传福", "自然人"),
        "relative": entity_id(folder_id, "亲属甲", "自然人"),
    }
    entities = [
        Entity(
            id=ids["company"],
            name="比亚迪股份有限公司",
            type="企业主体",
            type_en="ENTERPRISE",
            attributes={"注册资本": "30亿元"},
            references=[],
        ),
        Entity(
            id=ids["representative"],
            name="王传福",
            type="自然人",
            type_en="PERSON",
            references=[],
        ),
        Entity(
            id=ids["relative"],
            name="亲属甲",
            type="自然人",
            type_en="PERSON",
            references=[],
        ),
    ]
    relations = [
        Relation(
            id=relation_id(
                ids["representative"],
                ids["company"],
                "法定代表人",
            ),
            type="法定代表人",
            type_en="LEGAL_REPRESENTATIVE",
            from_entity="王传福",
            from_entity_id=ids["representative"],
            to_entity="比亚迪股份有限公司",
            to_entity_id=ids["company"],
            references=[],
        ),
        Relation(
            id=relation_id(
                *sorted((ids["representative"], ids["relative"])),
                "亲属关系",
            ),
            type="亲属关系",
            type_en="KINSHIP_OF",
            from_entity="王传福",
            from_entity_id=ids["representative"],
            to_entity="亲属甲",
            to_entity_id=ids["relative"],
            references=[],
        ),
        Relation(
            id=relation_id(
                ids["representative"],
                ids["company"],
                "高管任职",
            ),
            type="高管任职",
            type_en="EXECUTIVE_OF",
            from_entity="王传福",
            from_entity_id=ids["representative"],
            to_entity="比亚迪股份有限公司",
            to_entity_id=ids["company"],
            references=[
                Reference(
                    source="王传福担任比亚迪股份有限公司高管。",
                    file_id="association-source",
                )
            ],
        ),
    ]
    return (
        KnowledgeGraph(
            id="query-v3-integration",
            folderId=folder_id,
            folderName="V3集成测试",
            entities=entities,
            relations=relations,
        ),
        ids,
    )


@pytest.mark.integration
async def test_v3_reverse_bidirect_aggregate_and_hydration(
    load_env,
    monkeypatch,
) -> None:
    monkeypatch.setenv("NEBULA_HOST", "127.0.0.1")
    monkeypatch.setenv("NEBULA_PORT", "9669")
    monkeypatch.setenv("NEBULA_USER", "root")
    monkeypatch.setenv("NEBULA_PASSWORD", "nebula")
    monkeypatch.setenv("NEBULA_QUERY_USER", "kg_query")
    monkeypatch.setenv("NEBULA_QUERY_PASSWORD", "nebula_query")
    monkeypatch.setenv("NEBULA_SPACE", "knowledge_graph")
    folder_id = "kg-query-v3-integration"
    other_folder_id = "kg-query-v3-other"
    graph, ids = _graph(folder_id)
    other_id = entity_id(other_folder_id, "其他企业", "企业主体")
    other = KnowledgeGraph(
        id="query-v3-other",
        folderId=other_folder_id,
        folderName="其他项目",
        entities=[
            Entity(
                id=other_id,
                name="其他企业",
                type="企业主体",
                type_en="ENTERPRISE",
                references=[],
            )
        ],
    )

    NebulaClient.close()
    await NebulaGraphRepository.initialize_schema()
    with NebulaClient.session() as session:
        NebulaClient.execute(
            session,
            "CREATE USER IF NOT EXISTS kg_query WITH PASSWORD 'nebula_query'",
        )
        NebulaClient.execute(
            session,
            "GRANT ROLE GUEST ON `knowledge_graph` TO kg_query",
        )
    try:
        await NebulaGraphRepository.replace(graph)
        await NebulaGraphRepository.replace(other)

        baseline = compile_statement(
            """
GO FROM __ANCHOR_0_VIDS__ OVER legal_representative REVERSELY
YIELD src(edge) AS legal_rep,
      src(edge) AS `_vid_legal_rep`,
      src(edge) AS `_edge1_src`, dst(edge) AS `_edge1_dst`,
      type(edge) AS `_edge1_type`, rank(edge) AS `_edge1_rank`
| GO FROM $-.legal_rep OVER kinship_of BIDIRECT
YIELD $-.`_edge1_src` AS `_edge1_src`, $-.`_edge1_dst` AS `_edge1_dst`,
      $-.`_edge1_type` AS `_edge1_type`, $-.`_edge1_rank` AS `_edge1_rank`,
      src(edge) AS `_edge2_src`, dst(edge) AS `_edge2_dst`,
      type(edge) AS `_edge2_type`, rank(edge) AS `_edge2_rank`,
      properties($$).name AS name
| LIMIT 20
""",
            {0: [ids["company"]]},
        )
        rows = await NebulaGraphRepository.execute_read(
            folder_id,
            baseline,
            {"folder_key": folder_key(folder_id)},
        )
        entities, relations, business = await QueryServiceV3._hydrate(
            folder_id,
            rows,
            {0: [ids["company"]]},
        )
        assert business == [{"name": "亲属甲"}]
        assert {entity.id for entity in entities} == set(ids.values())
        assert {relation.type for relation in relations} == {
            "法定代表人",
            "亲属关系",
        }

        property_rows = await NebulaGraphRepository.execute_read(
            folder_id,
            compile_statement(
                "FETCH PROP ON enterprise __ANCHOR_0_VIDS__ "
                "YIELD id(vertex) AS `_vid` | LIMIT 20",
                {0: [ids["company"]]},
            ),
            {"folder_key": folder_key(folder_id)},
        )
        assert property_rows == [{"_vid": ids["company"]}]

        aggregate_rows = await NebulaGraphRepository.execute_read(
            folder_id,
            compile_statement(
                "LOOKUP ON enterprise "
                "WHERE enterprise.folder_key == $folder_key "
                "YIELD id(vertex) AS `_vid` | YIELD COUNT(*) AS n",
                {},
            ),
            {"folder_key": folder_key(folder_id)},
        )
        assert aggregate_rows == [{"n": 1}]

        association_statement = QueryServiceV3._association_statement(
            pairwise=True,
            enterprise_target=False,
        )
        association_rows = await NebulaGraphRepository.execute_read(
            folder_id,
            compile_statement(
                association_statement,
                {0: [ids["company"]], 1: [ids["relative"]]},
                max_hops=4,
            ),
            {"folder_key": folder_key(folder_id)},
        )
        association_entities, association_relations, association_business = (
            await QueryServiceV3._hydrate(
                folder_id,
                association_rows,
                {},
            )
        )
        assert {entity.id for entity in association_entities} == set(ids.values())
        assert {relation.type for relation in association_relations} == {
            "高管任职",
            "亲属关系",
        }
        executive = next(
            relation
            for relation in association_relations
            if relation.type == "高管任职"
        )
        assert executive.references[0].file_id == "association-source"
        assert association_business[0]["association_path_hops"] == 2

        for source, enterprise_target, expected in (
            (
                ids["company"],
                False,
                {ids["representative"], ids["relative"]},
            ),
            (ids["relative"], True, {ids["company"]}),
        ):
            listing_rows = await NebulaGraphRepository.execute_read(
                folder_id,
                compile_statement(
                    QueryServiceV3._association_statement(
                        pairwise=False,
                        enterprise_target=enterprise_target,
                    ),
                    {0: [source]},
                    max_hops=4,
                ),
                {"folder_key": folder_key(folder_id)},
            )
            assert {row["_vid_associated"] for row in listing_rows} == expected
    finally:
        await NebulaGraphRepository.delete(folder_id)
        await NebulaGraphRepository.delete(other_folder_id)
        NebulaClient.close()


@pytest.mark.integration
async def test_v3_llm_baseline_query(load_env, monkeypatch) -> None:
    monkeypatch.setenv("NEBULA_HOST", "127.0.0.1")
    monkeypatch.setenv("NEBULA_PORT", "9669")
    monkeypatch.setenv("NEBULA_USER", "root")
    monkeypatch.setenv("NEBULA_PASSWORD", "nebula")
    monkeypatch.setenv("NEBULA_QUERY_USER", "kg_query")
    monkeypatch.setenv("NEBULA_QUERY_PASSWORD", "nebula_query")
    monkeypatch.setenv("NEBULA_SPACE", "knowledge_graph")
    folder_id = "kg-query-v3-llm-integration"
    graph, ids = _graph(folder_id)
    graph.snapshot_version = "llm-snapshot-1"
    company_hit = AnchorHit(
        vid=ids["company"],
        name="比亚迪股份有限公司",
        type="企业主体",
        score=1.0,
        match_type="substring",
        snapshot_version=graph.snapshot_version,
    )
    relative_hit = AnchorHit(
        vid=ids["relative"],
        name="亲属甲",
        type="自然人",
        score=1.0,
        match_type="substring",
        snapshot_version=graph.snapshot_version,
    )
    monkeypatch.setattr(
        GraphStorageService,
        "get",
        AsyncMock(return_value=graph),
    )
    monkeypatch.setattr(
        EntityVectorStore,
        "metadata",
        AsyncMock(
            return_value={
                "snapshot_version": graph.snapshot_version,
            }
        ),
    )
    monkeypatch.setattr(
        EntityVectorStore,
        "search",
        AsyncMock(
            side_effect=lambda _folder_id, mention, _top_k, **_kwargs: (
                [relative_hit] if mention == "亲属甲" else [company_hit]
            )
        ),
    )

    NebulaClient.close()
    await NebulaGraphRepository.initialize_schema()
    with NebulaClient.session() as session:
        NebulaClient.execute(
            session,
            "CREATE USER IF NOT EXISTS kg_query WITH PASSWORD 'nebula_query'",
        )
        NebulaClient.execute(
            session,
            "GRANT ROLE GUEST ON `knowledge_graph` TO kg_query",
        )
    try:
        await NebulaGraphRepository.replace(graph)
        await NebulaGraphRepository.mark_snapshot(
            folder_id,
            graph.id,
            graph.snapshot_version,
            ready=True,
        )

        result = await QueryServiceV3.query(
            folder_id,
            "比亚迪的法定代表人的亲属有哪些？",
        )

        assert "亲属甲" in result.answer
        assert {entity.id for entity in result.entities} == set(ids.values())
        assert {relation.type for relation in result.relations} == {
            "法定代表人",
            "亲属关系",
        }

        association_result = await QueryServiceV3.query(
            folder_id,
            "比亚迪股份有限公司与亲属甲是否存在关联关系？",
        )

        assert "存在关联关系" in association_result.answer
        assert (
            "比亚迪股份有限公司—[高管任职]—王传福—[亲属关系]—亲属甲"
            in association_result.answer
        )
        assert {entity.id for entity in association_result.entities} == set(
            ids.values()
        )
        assert {relation.type for relation in association_result.relations} == {
            "高管任职",
            "亲属关系",
        }

        self_result = await QueryServiceV3.query(
            folder_id,
            "比亚迪股份有限公司与比亚迪股份有限公司是否存在关联关系？",
        )

        assert "未发现比亚迪股份有限公司与比亚迪股份有限公司存在关联关系" in (
            self_result.answer
        )
        assert self_result.entities == []
        assert self_result.relations == []
    finally:
        await NebulaGraphRepository.delete(folder_id)
        await NebulaGraphRepository.delete_snapshot(folder_id)
        NebulaClient.close()
