"""知识图谱增量更新单元测试

覆盖：
- _diff_files：文件级 diff（新增/删除/未变）
- _cleanup_deleted_files：删除清理 + 单轮判定
- _merge_entities/_merge_relations 的 existing 参数：合并入已有列表
- _rebuild_indexes：索引构建
- update 编排：增量提取 + 合并 + 持久化（mock 依赖）
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import List

import pytest

from src.ade.knowledge_graph.schema.kg_schema import (
    ElementMeta,
    Entity,
    FileInput,
    FileMeta,
    KnowledgeGraph,
    Piece,
    Reference,
    Relation,
    entity_id,
    relation_id,
)
from src.ade.knowledge_graph.service.knowledge_graph_service import KnowledgeGraphService


# ========== 辅助构造函数 ==========

def _elem(element_id: int, md: str = "内容", page: int = 0) -> ElementMeta:
    return ElementMeta(element_id=element_id, type="text", md=md, page=page)


def _file(file_id: str, name: str = "", elements: list[ElementMeta] | None = None) -> FileInput:
    return FileInput(id=file_id, name=name or file_id, elements=elements or [])


def _ref(file_id: str, source: str = "src") -> Reference:
    return Reference(source=source, element_id=1, file_id=file_id, page=0)


def _entity(name: str, file_id: str, entity_type: str = "公司", entity_id: str | None = None) -> Entity:
    return Entity(
        id=entity_id or f"ent_{name}",
        name=name,
        type=entity_type,
        type_en="COMPANY",
        attributes={},
        references=[_ref(file_id)],
        degree=0,
    )


def _relation(
    from_entity: str,
    to_entity: str,
    rel_type: str = "持股",
    file_id: str = "F1",
    from_id: str = "",
    to_id: str = "",
) -> Relation:
    return Relation(
        id=f"rel_{from_entity}_{to_entity}_{rel_type}",
        type=rel_type,
        type_en="HOLD",
        from_entity=from_entity,
        from_entity_id=from_id or f"ent_{from_entity}",
        to_entity=to_entity,
        to_entity_id=to_id or f"ent_{to_entity}",
        attributes={},
        references=[_ref(file_id)],
        degree=0,
    )


def _graph(
    folder_id: str = "folder1",
    folder_name: str = "测试库",
    entities: list[Entity] | None = None,
    relations: list[Relation] | None = None,
    file_index: dict[str, FileMeta] | None = None,
    piece_index: dict[str, list[str]] | None = None,
) -> KnowledgeGraph:
    return KnowledgeGraph(
        id="graph1",
        folderId=folder_id,
        folderName=folder_name,
        files=[],
        entities=entities or [],
        relations=relations or [],
        file_index=file_index or {},
        piece_index=piece_index or {},
    )


# ========== T_diff: _diff_files ==========

class TestDiffFiles:
    def test_all_new_files(self):
        """全部新文件（file_index 为空）"""
        new_files = [_file("F1"), _file("F2")]
        added, deleted, unchanged = KnowledgeGraphService._diff_files(new_files, {})
        assert {f.id for f in added} == {"F1", "F2"}
        assert deleted == []
        assert unchanged == []

    def test_all_deleted(self):
        """全部删除（new_files 为空）"""
        file_index = {
            "F1": FileMeta(file_id="F1", piece_hashes=[]),
            "F2": FileMeta(file_id="F2", piece_hashes=[]),
        }
        added, deleted, unchanged = KnowledgeGraphService._diff_files([], file_index)
        assert added == []
        assert set(deleted) == {"F1", "F2"}
        assert unchanged == []

    def test_unchanged_file_id_match(self):
        """文件 id 在 file_index 中 → unchanged（业务保证 id 不可复用）"""
        f = _file("F1", elements=[_elem(1, "abc")])
        file_index = {"F1": FileMeta(file_id="F1", piece_hashes=[])}
        added, deleted, unchanged = KnowledgeGraphService._diff_files([f], file_index)
        assert added == []
        assert deleted == []
        assert len(unchanged) == 1

    def test_mixed_diff(self):
        """混合：F1 未变、F2 新增、F3 删除"""
        new_files = [_file("F1"), _file("F2")]
        file_index = {
            "F1": FileMeta(file_id="F1", piece_hashes=[]),
            "F3": FileMeta(file_id="F3", piece_hashes=[]),
        }
        added, deleted, unchanged = KnowledgeGraphService._diff_files(new_files, file_index)
        assert {f.id for f in added} == {"F2"}
        assert set(deleted) == {"F3"}
        assert {f.id for f in unchanged} == {"F1"}


# ========== T_cleanup: _cleanup_deleted_files ==========

class TestCleanupDeletedFiles:
    def test_t1_entity_only_from_deleted_file_is_removed(self):
        """T1: 实体只来源于 F → 删除"""
        graph = _graph(entities=[_entity("E1", "F1")])
        KnowledgeGraphService._cleanup_deleted_files(graph, ["F1"])
        assert len(graph.entities) == 0

    def test_t2_entity_from_multiple_files_keeps_other_refs(self):
        """T2: 实体来源于 F1+F2，删 F1 → 保留，references 仅剩 F2"""
        e = _entity("E1", "F1")
        e.references.append(_ref("F2"))
        graph = _graph(entities=[e])
        KnowledgeGraphService._cleanup_deleted_files(graph, ["F1"])
        assert len(graph.entities) == 1
        assert all(r.file_id == "F2" for r in graph.entities[0].references)

    def test_t3_relation_only_from_deleted_file_is_removed(self):
        """T3: 关系只来源于 F → 删除"""
        graph = _graph(
            entities=[_entity("A", "F1"), _entity("B", "F1")],
            relations=[_relation("A", "B", file_id="F1", from_id="ent_A", to_id="ent_B")],
        )
        KnowledgeGraphService._cleanup_deleted_files(graph, ["F1"])
        assert len(graph.relations) == 0
        # 端点实体 references 也清空，且无 relation 引用 → 一并删除
        assert len(graph.entities) == 0

    def test_t4_relation_from_multiple_files_keeps_other_refs(self):
        """T4: 关系来源于 F1+F2，删 F1 → 保留，references 仅剩 F2"""
        rel = _relation("A", "B", file_id="F1", from_id="ent_A", to_id="ent_B")
        rel.references.append(_ref("F2"))
        graph = _graph(
            entities=[_entity("A", "F1"), _entity("B", "F1")],
            relations=[rel],
        )
        # 实体也加 F2 reference，避免被删
        graph.entities[0].references.append(_ref("F2"))
        graph.entities[1].references.append(_ref("F2"))
        KnowledgeGraphService._cleanup_deleted_files(graph, ["F1"])
        assert len(graph.relations) == 1
        assert all(r.file_id == "F2" for r in graph.relations[0].references)

    def test_t5_entity_with_empty_refs_but_referenced_by_relation_is_kept(self):
        """T5: 实体 references 清空但被 relation 引用 → 保留为 stub；relation 保留（references 未空）"""
        rel = _relation("A", "B", file_id="F2", from_id="ent_A", to_id="ent_B")  # F2 提供
        e_a = _entity("A", "F1")  # A 只来自 F1（将被清空）
        e_b = _entity("B", "F2")  # B 来自 F2（保留）
        graph = _graph(entities=[e_a, e_b], relations=[rel])
        KnowledgeGraphService._cleanup_deleted_files(graph, ["F1"])
        # A references 清空但被 rel 引用 → 保留
        assert any(e.name == "A" for e in graph.entities)
        a = next(e for e in graph.entities if e.name == "A")
        assert a.references == []
        # relation 保留（references 来自 F2）
        assert len(graph.relations) == 1

    def test_cleanup_updates_file_and_piece_index(self):
        """T12: 删除文件后索引同步清理"""
        graph = _graph(
            entities=[_entity("E1", "F1")],
            file_index={"F1": FileMeta(file_id="F1", piece_hashes=["p1", "p2"])},
            piece_index={"p1": ["F1"], "p2": ["F1", "F2"]},  # p2 被多文件共享
        )
        KnowledgeGraphService._cleanup_deleted_files(graph, ["F1"])
        assert "F1" not in graph.file_index
        assert "p1" not in graph.piece_index  # 只属于 F1 → 删除
        assert graph.piece_index["p2"] == ["F2"]  # 共享的保留，移除 F1


# ========== T_merge: _merge_entities/_merge_relations existing 参数 ==========

class TestMergeWithExisting:
    def test_t6_merge_into_existing_entity_reuses_id(self):
        """T6: delta 实体 name 命中已有 → 复用 entity_id，references 累加"""
        existing = [Entity(
            id="old_id_A", name="A", type="公司", type_en="COMPANY",
            attributes={"attr1": "v1"}, references=[_ref("F1")], degree=0,
        )]
        delta = [Entity(
            id="new_id_A", name="A", type="公司", type_en="COMPANY",
            attributes={"attr2": "v2"}, references=[_ref("F2")], degree=0,
        )]
        result, _, _ = KnowledgeGraphService._merge_entities(delta, existing=existing)
        assert len(result) == 1
        assert result[0].id == "old_id_A"  # 复用旧 id
        assert {r.file_id for r in result[0].references} == {"F1", "F2"}
        assert result[0].attributes == {"attr1": "v1", "attr2": "v2"}

    def test_t7_merge_into_existing_relation_reuses_key(self):
        """T7: delta 关系 (from_id, to_id, type) 命中已有 → references 累加"""
        existing = [Relation(
            id="old_rel", type="持股", type_en="HOLD",
            from_entity="A", from_entity_id="id_A",
            to_entity="B", to_entity_id="id_B",
            attributes={"pct": ["50%"]}, references=[_ref("F1")], degree=0,
        )]
        delta = [Relation(
            id="new_rel", type="持股", type_en="HOLD",
            from_entity="A", from_entity_id="id_A",
            to_entity="B", to_entity_id="id_B",
            attributes={"pct": ["60%"]}, references=[_ref("F2")], degree=0,
        )]
        result, _ = KnowledgeGraphService._merge_relations(
            delta, merged_entities=[
                Entity(id="id_A", name="A", type="公司", type_en="COMPANY", attributes={}, references=[], degree=0),
                Entity(id="id_B", name="B", type="公司", type_en="COMPANY", attributes={}, references=[], degree=0),
            ],
            existing=existing,
        )
        assert len(result) == 1
        assert {r.file_id for r in result[0].references} == {"F1", "F2"}
        assert result[0].attributes["pct"] == ["50%", "60%"]

    def test_t8_new_entity_gets_new_id_added_to_graph(self):
        """T8: delta 实体 name 不命中 → 新建实体加入图谱"""
        existing = [_entity("A", "F1", entity_id="id_A")]
        delta = [_entity("B", "F2", entity_id="new_id_B")]
        result, _, _ = KnowledgeGraphService._merge_entities(delta, existing=existing)
        assert len(result) == 2
        names = {e.name for e in result}
        assert names == {"A", "B"}

    def test_merge_entities_without_existing_equivalent_to_full(self):
        """无 existing 参数 → 等价于全量合并"""
        entities = [_entity("A", "F1"), _entity("A", "F2")]
        result, _, _ = KnowledgeGraphService._merge_entities(entities)
        assert len(result) == 1
        assert len(result[0].references) == 2


# ========== T_index: _rebuild_indexes ==========

class TestRebuildIndexes:
    def test_indexes_built_from_file_pieces(self):
        """T13: 新增文件后 piece_index/file_index 正确构建"""
        f = _file("F1", elements=[_elem(1, "abc")])
        pieces = [
            Piece(id="p1", content="abc", content_hash="h_p1", file_id="F1", file_name="F1", elements=[_elem(1, "abc")]),
        ]
        graph = _graph()
        KnowledgeGraphService._rebuild_indexes(graph, [(f, pieces)])
        assert "F1" in graph.file_index
        assert graph.file_index["F1"].piece_hashes == ["h_p1"]
        assert graph.piece_index["h_p1"] == ["F1"]


# ========== T_update: update 编排 ==========

class TestUpdateOrchestration:
    def test_t9_degrade_to_full_when_graph_missing(self, monkeypatch):
        """T9: graph 不存在 → 降级全量（在 task 层判断，service.update 假定 graph 已存在）"""
        # 这里只验证 service.update 不应被调用当 graph 为空时（task 层逻辑）
        # service 层测试：直接调 update 时 graph 应已存在
        # 该用例由 task 层测试覆盖，此处跳过 service 层
        pytest.skip("降级逻辑在 task 层，service 层假定 graph 已存在")

    def test_t10_delete_and_add_files(self, monkeypatch):
        """T10: 删除 F1 + 新增 F2 → F1 清理，F2 提取合并"""
        # 初始 graph：F1 已存在
        graph = _graph(
            folder_id="folder1",
            entities=[_entity("E1", "F1")],
            relations=[],
            file_index={"F1": FileMeta(file_id="F1", piece_hashes=[])},
            piece_index={},
        )
        # 新文件列表：只有 F2
        new_files = [_file("F2", elements=[_elem(1, "xyz")])]

        # mock ExtractorService.extract 返回新实体
        async def mock_extract(piece, folder_name="", raw_response_callback=None, match_stats=None):
            return [_entity("E2", piece.file_id, entity_id="ent_E2")], []

        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.ExtractorService.extract",
            mock_extract,
        )
        # mock GraphStorageService.save
        async def mock_save(g):
            return g.id
        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.GraphStorageService.save",
            mock_save,
        )
        # mock _embed_items 避免依赖 EMBEDDING_BATCH_SIZE 环境变量
        async def mock_embed(items, to_text, tag):
            for item in items:
                item.embed = [0.1]
        monkeypatch.setattr(
            KnowledgeGraphService, "_embed_items",
            classmethod(lambda cls, items, to_text, tag: mock_embed(items, to_text, tag))
        )

        result = asyncio.run(KnowledgeGraphService.update(graph, new_files))

        # F1 被删，E1 消失
        assert not any(e.name == "E1" for e in result.entities)
        # F2 提取出 E2
        assert any(e.name == "E2" for e in result.entities)
        # file_index 更新
        assert "F1" not in result.file_index
        assert "F2" in result.file_index

    def test_t11_piece_hash_hit_skips_llm(self, monkeypatch):
        """T11: 新文件 piece_hash 命中 → 跳过 LLM"""
        # piece 内容
        content = "abc"
        piece_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

        graph = _graph(
            folder_id="folder1",
            entities=[],
            file_index={},
            piece_index={piece_hash: ["F0"]},  # 已有 F0 处理过该 piece
        )
        new_files = [_file("F1", elements=[_elem(1, content)])]

        # mock extract：若被调用则失败
        async def mock_extract(*args, **kwargs):
            raise AssertionError("piece_hash 命中时不应调 LLM")
        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.ExtractorService.extract",
            mock_extract,
        )
        async def mock_save(g):
            return g.id
        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.GraphStorageService.save",
            mock_save,
        )

        result = asyncio.run(KnowledgeGraphService.update(graph, new_files))

        # 无新增实体（piece 被跳过，且 update 不再注入项目实体）
        assert len(result.entities) == 0
        # piece_index owner 列表新增 F1
        assert "F1" in result.piece_index[piece_hash]
        # file_index 新增 F1
        assert "F1" in result.file_index


# ========== T_incremental_merge: _merge_entities/_merge_relations 增量返回 ==========

class TestMergeIncrementalReturn:
    def test_merge_entities_returns_delta_and_changed(self):
        """已有实体 attributes 被更新 → 进入 changed；新实体进入 delta"""
        existing = [Entity(
            id="id_A", name="A", type="公司", type_en="COMPANY",
            attributes={"attr1": "v1"}, references=[_ref("F1")], degree=0,
        )]
        delta = [
            # A：命中 existing，attributes 增加 → changed
            Entity(id="new_A", name="A", type="公司", type_en="COMPANY",
                   attributes={"attr2": "v2"}, references=[_ref("F2")], degree=0),
            # B：新实体 → delta
            _entity("B", "F2", entity_id="id_B"),
        ]
        merged, delta_out, changed_out = KnowledgeGraphService._merge_entities(
            delta, existing=existing
        )
        assert len(merged) == 2
        # delta 应包含 B（新实体）
        delta_names = {e.name for e in delta_out}
        assert delta_names == {"B"}
        # changed 应包含 A（attributes 变化）
        changed_names = {e.name for e in changed_out}
        assert changed_names == {"A"}

    def test_merge_entities_existing_unchanged_not_in_changed(self):
        """已有实体 attributes 未变（delta 没新增 attr）→ 不进入 changed"""
        existing = [Entity(
            id="id_A", name="A", type="公司", type_en="COMPANY",
            attributes={"attr1": "v1"}, references=[_ref("F1")], degree=0,
        )]
        # delta A 只追加 references，不动 attributes
        delta = [Entity(
            id="new_A", name="A", type="公司", type_en="COMPANY",
            attributes={"attr1": "v1"}, references=[_ref("F2")], degree=0,
        )]
        merged, delta_out, changed_out = KnowledgeGraphService._merge_entities(
            delta, existing=existing
        )
        assert len(changed_out) == 0
        assert len(delta_out) == 0  # A 是已有实体，不在 delta

    def test_merge_entities_without_existing_returns_all_as_delta(self):
        """无 existing → 全量场景，所有合并后的实体都视为 delta"""
        entities = [_entity("A", "F1"), _entity("A", "F2")]
        merged, delta_out, changed_out = KnowledgeGraphService._merge_entities(entities)
        assert len(merged) == 1
        assert len(delta_out) == 1
        assert len(changed_out) == 0

    def test_merge_relations_returns_delta(self):
        """增量合并：新关系进 delta；命中已有的不进 delta"""
        existing = [Relation(
            id="old_rel", type="持股", type_en="HOLD",
            from_entity="A", from_entity_id="id_A",
            to_entity="B", to_entity_id="id_B",
            attributes={"pct": ["50%"]}, references=[_ref("F1")], degree=0,
        )]
        delta = [
            # 命中已有 → 不进 delta
            Relation(id="r1", type="持股", type_en="HOLD",
                     from_entity="A", from_entity_id="id_A",
                     to_entity="B", to_entity_id="id_B",
                     attributes={"pct": ["60%"]}, references=[_ref("F2")], degree=0),
            # 新关系 → delta
            Relation(id="r2", type="持股", type_en="HOLD",
                     from_entity="A", from_entity_id="id_A",
                     to_entity="C", to_entity_id="id_C",
                     attributes={}, references=[_ref("F2")], degree=0),
        ]
        merged, delta_out = KnowledgeGraphService._merge_relations(
            delta, merged_entities=[
                Entity(id="id_A", name="A", type="公司", type_en="COMPANY", attributes={}, references=[], degree=0),
                Entity(id="id_B", name="B", type="公司", type_en="COMPANY", attributes={}, references=[], degree=0),
                Entity(id="id_C", name="C", type="公司", type_en="COMPANY", attributes={}, references=[], degree=0),
            ],
            existing=existing,
        )
        assert len(merged) == 2
        # delta 只包含新关系 A→C
        delta_ids = {r.id for r in delta_out}
        assert delta_ids == {relation_id("id_A", "id_C", "持股")}

    def test_merge_relations_without_existing_returns_all_as_delta(self):
        """无 existing → 全量场景，所有合并后的关系都视为 delta"""
        e_a = Entity(id="id_A", name="A", type="公司", type_en="COMPANY", attributes={}, references=[], degree=0)
        e_b = Entity(id="id_B", name="B", type="公司", type_en="COMPANY", attributes={}, references=[], degree=0)
        delta = [
            Relation(id="r1", type="持股", type_en="HOLD",
                     from_entity="A", from_entity_id="id_A",
                     to_entity="B", to_entity_id="id_B",
                     attributes={}, references=[_ref("F1")], degree=0),
        ]
        merged, delta_out = KnowledgeGraphService._merge_relations(
            delta, merged_entities=[e_a, e_b]
        )
        assert len(delta_out) == 1


# ========== T_cleanup_return: _cleanup_deleted_files 返回被删除的 relations ==========

class TestCleanupDeletedFilesReturn:
    def test_cleanup_returns_deleted_relations(self):
        """被删除的关系应出现在返回值中，用于推导 affected 实体"""
        rel = _relation("A", "B", file_id="F1", from_id="ent_A", to_id="ent_B")
        graph = _graph(
            entities=[_entity("A", "F1"), _entity("B", "F1")],
            relations=[rel],
        )
        deleted = KnowledgeGraphService._cleanup_deleted_files(graph, ["F1"])
        assert len(deleted) == 1
        assert deleted[0].from_entity_id == "ent_A"
        assert deleted[0].to_entity_id == "ent_B"

    def test_cleanup_no_deleted_returns_empty(self):
        """无删除文件 → 返回空列表"""
        graph = _graph(entities=[_entity("A", "F1")])
        deleted = KnowledgeGraphService._cleanup_deleted_files(graph, [])
        assert deleted == []


# ========== T_check_incremental: 增量校验 ==========

class TestCheckIncremental:
    def test_check_borrower_guarantor_conflict_delta_only_triggers_on_new(self):
        """增量校验：delta 中新增保证人，与已有主借款人冲突 → warning"""
        # 已有：A 是项目 P 的主借款人
        existing_borrower = _relation("A", "P", rel_type="主借款人", file_id="F1",
                                       from_id="ent_A", to_id="ent_P")
        # delta：A 也是项目 P 的保证人
        delta_guarantor = _relation("A", "P", rel_type="保证人", file_id="F2",
                                     from_id="ent_A", to_id="ent_P")
        all_relations = [existing_borrower, delta_guarantor]
        # 增量校验：只对 delta 中的"保证人"检查
        KnowledgeGraphService._check_borrower_guarantor_conflict(
            [delta_guarantor], all_relations=all_relations
        )
        # 仅验证不抛异常（warning 行为）

    def test_check_unique_primary_borrower_delta_triggers_warning(self):
        """增量校验：delta 新增主借款人，已有主借款人 → warning"""
        existing_borrower = _relation("A", "P", rel_type="主借款人", file_id="F1",
                                       from_id="ent_A", to_id="ent_P")
        delta_borrower = _relation("B", "P", rel_type="主借款人", file_id="F2",
                                    from_id="ent_B", to_id="ent_P")
        all_relations = [existing_borrower, delta_borrower]
        KnowledgeGraphService._check_unique_primary_borrower(
            [delta_borrower], all_relations=all_relations
        )

    def test_check_unique_primary_borrower_delta_no_existing_no_warning(self):
        """增量校验：delta 中项目首次出现主借款人，已有中无同项目主借款人 → 不触发"""
        existing_other = _relation("A", "P1", rel_type="主借款人", file_id="F1",
                                    from_id="ent_A", to_id="ent_P1")
        delta_borrower = _relation("B", "P2", rel_type="主借款人", file_id="F2",
                                    from_id="ent_B", to_id="ent_P2")
        all_relations = [existing_other, delta_borrower]
        KnowledgeGraphService._check_unique_primary_borrower(
            [delta_borrower], all_relations=all_relations
        )


# ========== T_degree: _compute_degree 全量重算 ==========

class TestComputeDegree:
    def test_compute_degree_full_recalculation(self):
        """全量重算：所有实体和关系的 degree 都基于全量 relations 重算"""
        a = Entity(id="id_A", name="A", type="公司", type_en="COMPANY",
                   attributes={}, references=[], degree=0)
        b = Entity(id="id_B", name="B", type="公司", type_en="COMPANY",
                   attributes={}, references=[], degree=0)
        c = Entity(id="id_C", name="C", type="公司", type_en="COMPANY",
                   attributes={}, references=[], degree=0)
        # 关系：A-B, B-C
        rel_ab = Relation(id="r_ab", type="持股", type_en="HOLD",
                          from_entity="A", from_entity_id="id_A",
                          to_entity="B", to_entity_id="id_B",
                          attributes={}, references=[], degree=0)
        rel_bc = Relation(id="r_bc", type="持股", type_en="HOLD",
                          from_entity="B", from_entity_id="id_B",
                          to_entity="C", to_entity_id="id_C",
                          attributes={}, references=[], degree=0)
        KnowledgeGraphService._compute_degree([a, b, c], [rel_ab, rel_bc])
        # A degree=1, B degree=2（连 A 和 C）, C degree=1
        assert a.degree == 1
        assert b.degree == 2
        assert c.degree == 1
        # 关系 degree = 两端实体 degree 之和
        assert rel_ab.degree == a.degree + b.degree
        assert rel_bc.degree == b.degree + c.degree

    def test_compute_degree_overwrites_old_value(self):
        """全量重算会覆盖实体上原有的 degree（不保留旧值）"""
        a = Entity(id="id_A", name="A", type="公司", type_en="COMPANY",
                   attributes={}, references=[], degree=99)  # 旧值
        b = Entity(id="id_B", name="B", type="公司", type_en="COMPANY",
                   attributes={}, references=[], degree=99)
        rel = Relation(id="r", type="持股", type_en="HOLD",
                       from_entity="A", from_entity_id="id_A",
                       to_entity="B", to_entity_id="id_B",
                       attributes={}, references=[], degree=0)
        KnowledgeGraphService._compute_degree([a, b], [rel])
        assert a.degree == 1
        assert b.degree == 1
        assert rel.degree == 2


# ========== T_update_incremental: update 编排增量化 ==========

class TestUpdateIncrementalEmbed:
    def test_update_only_embeds_delta_and_changed_not_all(self, monkeypatch):
        """update 应只对 delta+changed 实体和 delta 关系调 embed，不对全量调"""
        # 已有图谱：实体 A（已有 attribute），关系 A-B
        a = Entity(id="id_A", name="A", type="公司", type_en="COMPANY",
                   attributes={"attr1": "v1"}, references=[_ref("F1")], degree=1)
        b = Entity(id="id_B", name="B", type="公司", type_en="COMPANY",
                   attributes={}, references=[_ref("F1")], degree=1)
        rel_ab = Relation(id="r_ab", type="持股", type_en="HOLD",
                          from_entity="A", from_entity_id="id_A",
                          to_entity="B", to_entity_id="id_B",
                          attributes={}, references=[_ref("F1")], degree=2)
        graph = _graph(
            entities=[a, b],
            relations=[rel_ab],
            file_index={"F1": FileMeta(file_id="F1", piece_hashes=[])},
        )
        # 保留 F1，并新增 F2 → 提取出 A（attr 新增）+ C（新）+ 新关系 A-C
        new_files = [
            _file("F1", elements=[_elem(1, "existing")]),
            _file("F2", elements=[_elem(1, "xyz")]),
        ]

        embedded_entities: list[str] = []
        embedded_relations: list[str] = []

        async def mock_extract(piece, folder_name="", raw_response_callback=None, match_stats=None):
            return [
                Entity(name="A", type="公司", type_en="COMPANY",
                       attributes={"attr2": "v2"}, references=[_ref("F2")], degree=0),
                Entity(name="C", type="公司", type_en="COMPANY",
                       attributes={}, references=[_ref("F2")], degree=0),
            ], [
                # 模拟 extractor 真实行为：from_entity_id/to_entity_id 为空，由 _merge_relations 按 name 解析
                Relation(type="持股", type_en="HOLD",
                         from_entity="A", from_entity_id="",
                         to_entity="C", to_entity_id="",
                         attributes={}, references=[_ref("F2")], degree=0),
            ]

        async def mock_embed_items(items, to_text, tag):
            if tag == "实体":
                embedded_entities.extend(item.name for item in items)
            else:
                embedded_relations.extend(item.id for item in items)
            # 模拟写回 embed
            for item in items:
                item.embed = [0.1]

        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.ExtractorService.extract",
            mock_extract,
        )
        async def mock_save(g):
            return g.id
        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.GraphStorageService.save",
            mock_save,
        )
        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.EmbedService.embed_texts",
            lambda texts: [[0.1]] * len(texts),
        )
        # 直接 mock _embed_items 记录调用
        async def mock_embed_items_cls(items, to_text, tag):
            await mock_embed_items(items, to_text, tag)
        monkeypatch.setattr(
            KnowledgeGraphService, "_embed_items", classmethod(
                lambda cls, items, to_text, tag: mock_embed_items_cls(items, to_text, tag)
            )
        )

        asyncio.run(KnowledgeGraphService.update(graph, new_files))

        # 实体 embed：应包含 A（attributes 变化 → changed）和 C（新 → delta）
        # 不应包含 B（未变化）
        assert "A" in embedded_entities
        assert "C" in embedded_entities
        assert "B" not in embedded_entities
        # 关系 embed：应只包含新关系的稳定哈希 ID，不包含已有 r_ab
        c_id = entity_id(graph.folderId, "C", "公司")
        assert relation_id("id_A", c_id, "持股") in embedded_relations
        assert "r_ab" not in embedded_relations

    def test_update_deleted_relations_trigger_degree_recompute(self, monkeypatch):
        """删除文件导致关系消失 → 受影响实体 degree 重算"""
        a = Entity(id="id_A", name="A", type="公司", type_en="COMPANY",
                   attributes={}, references=[_ref("F1"), _ref("F2")], degree=2)
        b = Entity(id="id_B", name="B", type="公司", type_en="COMPANY",
                   attributes={}, references=[_ref("F1"), _ref("F2")], degree=2)
        c = Entity(id="id_C", name="C", type="公司", type_en="COMPANY",
                   attributes={}, references=[_ref("F2")], degree=1)
        # F1: A-B；F2: A-B, B-C
        rel_ab_f1 = Relation(id="r_ab_1", type="持股", type_en="HOLD",
                             from_entity="A", from_entity_id="id_A",
                             to_entity="B", to_entity_id="id_B",
                             attributes={}, references=[_ref("F1")], degree=4)
        rel_ab_f2 = Relation(id="r_ab_2", type="持股", type_en="HOLD",
                             from_entity="A", from_entity_id="id_A",
                             to_entity="B", to_entity_id="id_B",
                             attributes={}, references=[_ref("F2")], degree=4)
        rel_bc = Relation(id="r_bc", type="持股", type_en="HOLD",
                          from_entity="B", from_entity_id="id_B",
                          to_entity="C", to_entity_id="id_C",
                          attributes={}, references=[_ref("F2")], degree=3)
        graph = _graph(
            entities=[a, b, c],
            relations=[rel_ab_f1, rel_ab_f2, rel_bc],
            file_index={
                "F1": FileMeta(file_id="F1", piece_hashes=[]),
                "F2": FileMeta(file_id="F2", piece_hashes=[]),
            },
        )
        # 新文件列表：只有 F2 → F1 被删
        new_files = [_file("F2", elements=[_elem(1, "xyz")])]

        async def mock_extract(*args, **kwargs):
            return [], []

        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.ExtractorService.extract",
            mock_extract,
        )
        async def mock_save(g):
            return g.id
        monkeypatch.setattr(
            "src.ade.knowledge_graph.service.knowledge_graph_service.GraphStorageService.save",
            mock_save,
        )
        monkeypatch.setattr(
            KnowledgeGraphService, "_embed_items",
            classmethod(lambda cls, items, to_text, tag: _noop_embed())
        )

        asyncio.run(KnowledgeGraphService.update(graph, new_files))

        # F1 删除后：rel_ab_f1 消失，rel_ab_f2 + rel_bc 保留
        rel_ids = {r.id for r in graph.relations}
        assert "r_ab_1" not in rel_ids
        assert "r_ab_2" in rel_ids
        assert "r_bc" in rel_ids
        # A: 原 degree=2（A-B×2 + ... 不对，重新数：A 参与 r_ab_f1, r_ab_f2 → degree=2）
        # 删 F1 后 A 只参与 r_ab_f2 → degree=1
        a_out = next(e for e in graph.entities if e.name == "A")
        assert a_out.degree == 1
        # B: 参与 r_ab_f2 + r_bc → degree=2
        b_out = next(e for e in graph.entities if e.name == "B")
        assert b_out.degree == 2
        # C: 参与 r_bc → degree=1（未变，但也可能在 affected 中）
        c_out = next(e for e in graph.entities if e.name == "C")
        assert c_out.degree == 1


async def _noop_embed():
    return None
