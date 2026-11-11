import asyncio
import re
import os
import numpy as np
from typing import Any, Callable, List, Optional
from collections import defaultdict
from loguru import logger

from src.ade.knowledge_graph.schema.kg_schema import (
    FileInput,
    FileMeta,
    Entity,
    Relation,
    Piece,
    KnowledgeGraph,
    entity_id,
    relation_id,
)
from src.ade.knowledge_graph.client.schema.extraction_schema import ENTITY_TYPE_MAP
from src.ade.knowledge_graph.client.schema.definitions import RELATION_TYPES
from src.ade.knowledge_graph.service.splitting_service import SplittingService
from src.ade.knowledge_graph.service.extractor_service import (
    ExtractorService,
    FatalKGExtractionError,
    KGExtractionOutputTruncatedError,
    create_match_stats,
)
from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService
from src.ade.knowledge_graph.service.embed_service import EmbedService
from src.utils import StopWatch
from rapidfuzz.distance import Levenshtein

_WILDCARD = "*"
_MAX_OUTPUT_SPLIT_DEPTH = 4
_RELATION_SCHEMA = {relation.name: relation for relation in RELATION_TYPES}
_ENTERPRISE_TYPE = "企业主体"
_ENTERPRISE_SUFFIX_FAMILIES = (
    ("有限公司", "有限责任公司"),
    ("股份公司", "股份有限公司"),
)
_ENTERPRISE_NAME_MARKERS = ("公司", "企业")
_ABBREVIATION_ATTRIBUTE = "简称"
_ABBREVIATION_SEPARATOR = "、"
_UNKNOWN_TYPE = "未知"

# 后处理反向补全：SUPPLIER_OF → CUSTOMER_OF（不进 LLM prompt，仅由后处理派生）
_CUSTOMER_RELATION_TYPE = "客户关系"
_SUPPLIER_RELATION_TYPE = "供应商关系"


def _merge_relation_content(existing: Relation, incoming: Relation) -> None:
    existing.references.extend(incoming.references)
    for attribute_name, values in incoming.attributes.items():
        existing_values = existing.attributes.setdefault(attribute_name, [])
        existing_values.extend(value for value in values if value not in existing_values)


def _group_entities_by_name(entities: List[Entity]) -> dict[str, list[Entity]]:
    grouped: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        grouped[entity.name].append(entity)
    return grouped


def _resolve_entity_candidate(
    candidates: list[Entity],
    allowed_types: list[str],
) -> tuple[Entity | None, str]:
    known_candidates = [
        candidate for candidate in candidates if candidate.type != _UNKNOWN_TYPE
    ]
    compatible = [
        candidate
        for candidate in known_candidates
        if candidate.type in allowed_types
    ]
    compatible_types = {candidate.type for candidate in compatible}

    if len(compatible_types) > 1:
        return None, "ambiguous_endpoint_type"
    if compatible:
        return compatible[0], "resolved"
    if not candidates:
        return None, "missing"
    if not known_candidates:
        return candidates[0], "unknown"
    return None, "type_mismatch"


class KnowledgeGraphService:
    """知识图谱生成服务"""

    @classmethod
    async def generate(
        cls,
        files: List[FileInput],
        folder_id: str,
        folder_name: str,
        graph_id: str,
        raw_response_callback: Callable[[Piece, str], None] | None = None,
    ) -> KnowledgeGraph:
        """生成知识图谱

        Args:
            files: 文件列表
            folder_id: 知识库ID
            folder_name: 知识库名称
            graph_id: 图谱ID
            raw_response_callback: LLM诊断包络回调（JSON 字符串）

        Returns:
            KnowledgeGraph: 生成的知识图谱
        """
        project_name = folder_name
        # 🔍 调试日志：记录 generate 接收到的文件详情
        logger.info("KnowledgeGraphService.generate 开始 | graph_id={} folder_id={} files_count={}",
                    graph_id, folder_id, len(files))
        for f in files:
            logger.info("  - 文件: id={} name={} elements_count={}", f.id, f.name, len(f.elements))

        sw = StopWatch("generate_knowledge_graph")
        with sw.task("生成知识图谱") as root:
            all_entities: List[Entity] = []
            all_relations: List[Relation] = []
            total_pieces = 0
            with root.task("文件切片"):
                file_pieces: list[tuple[FileInput, List[Piece]]] = []
                for f in files:
                    pieces = SplittingService.split(f)
                    file_pieces.append((f, pieces))
                    total_pieces += len(pieces)
                    logger.info("KnowledgeGraphService | 文件切片完成 | file_id={} pieces={}", f.id, len(pieces))

            # 🔍 调试日志：记录总切片数
            logger.info("KnowledgeGraphService | 切片总数: {}", total_pieces)

            with root.task("LLM提取"):
                match_stats = create_match_stats()
                sem = asyncio.Semaphore(3)

                async def process_one(piece: Piece, split_depth: int = 0):
                    try:
                        async with sem:
                            result = await ExtractorService.extract(
                                piece,
                                folder_name=folder_name,
                                project_name=project_name,
                                raw_response_callback=raw_response_callback,
                                match_stats=match_stats,
                            )
                    except KGExtractionOutputTruncatedError:
                        pass
                    except FatalKGExtractionError:
                        logger.exception(
                            "KnowledgeGraphService | piece结构化输出重试耗尽，跳过 | piece_id={} file_name={}",
                            piece.id,
                            piece.file_name,
                        )
                        return [], []
                    except Exception:
                        logger.exception(
                            "KnowledgeGraphService | piece提取失败，跳过 | piece_id={} file_name={}",
                            piece.id,
                            piece.file_name,
                        )
                        return [], []
                    else:
                        return result

                    if split_depth >= _MAX_OUTPUT_SPLIT_DEPTH:
                        logger.warning(
                            "KnowledgeGraphService | 输出截断且已达动态拆分深度上限，跳过 | "
                            "piece_id={} file_name={} split_depth={}",
                            piece.id,
                            piece.file_name,
                            split_depth,
                        )
                        return [], []

                    child_pieces = SplittingService.split_piece(piece)
                    if not child_pieces:
                        logger.warning(
                            "KnowledgeGraphService | 输出截断且piece无法继续拆分，跳过 | "
                            "piece_id={} file_name={} split_depth={} elements={}",
                            piece.id,
                            piece.file_name,
                            split_depth,
                            len(piece.elements),
                        )
                        return [], []

                    logger.warning(
                        "KnowledgeGraphService | 输出截断，动态二分piece | "
                        "parent_piece_id={} split_depth={} child_ids={} "
                        "child_chars={} child_elements={}",
                        piece.id,
                        split_depth + 1,
                        [child.id for child in child_pieces],
                        [len(child.content) for child in child_pieces],
                        [len(child.elements) for child in child_pieces],
                    )
                    child_results = await asyncio.gather(
                        *(
                            process_one(child, split_depth + 1)
                            for child in child_pieces
                        )
                    )
                    entities = [
                        entity
                        for child_entities, _ in child_results
                        for entity in child_entities
                    ]
                    relations = [
                        relation
                        for _, child_relations in child_results
                        for relation in child_relations
                    ]
                    return entities, relations

                tasks = [
                    asyncio.create_task(process_one(piece))
                    for _, pieces in file_pieces
                    for piece in pieces
                ]
                results = await asyncio.gather(*tasks)

                for entities, relations in results:
                    all_entities.extend(entities)
                    all_relations.extend(relations)

                # 🔍 调试日志：记录LLM提取结果
                logger.info("KnowledgeGraphService | LLM提取完成 | 提取实体数={} 提取关系数={}",
                            len(all_entities), len(all_relations))

                if match_stats["total"] > 0:
                    logger.info(
                        "KnowledgeGraphService | 溯源匹配统计 | "
                        "总尝试={} 匹配成功={}({:.1f}%) 文本失配={} element不存在={} 空输入={}",
                        match_stats["total"],
                        match_stats["matched"],
                        match_stats["matched"] / match_stats["total"] * 100,
                        match_stats["source_mismatch"],
                        match_stats["element_not_found"],
                        match_stats["empty_source"],
                    )

            # with root.task("LLM提取"):
            #     for f, pieces in file_pieces:
            #         for piece in pieces:
            #             try:
            #                 entities, relations = await ExtractorService.extract(piece, folder_name=folder_name)
            #                 all_entities.extend(entities)
            #                 all_relations.extend(relations)
            #             except Exception:
            #                 logger.exception("KnowledgeGraphService | piece提取失败，跳过 | piece_id={} | file_name={}", piece.id, piece.file_name)
            
            
            # 注入项目实体（使prompt中关系表指向的目标实体在图谱中有对应的节点）
            project_entity = Entity(
                name=project_name,
                type="项目",
                type_en="PROJECT",
                attributes={},
                references=[],
                degree=0,
            )
            all_entities.append(project_entity)

            with root.task("实体去重与关系合并"):
                merged_entities, _, _ = cls._merge_entities(all_entities)
                cls._normalize_enterprise_abbreviations(
                    merged_entities, all_relations
                )
                merged_entities, _, _ = cls._merge_entities(merged_entities)
                # 实体名此时已定型，ID 用去重键哈希生成，保证重跑幂等
                for entity in merged_entities:
                    entity.id = entity_id(folder_id, entity.name, entity.type)
                filtered_relations = cls._filter_relations_by_schema(
                    all_relations, merged_entities, project_name
                )
                # 反向派生客户关系后与原关系一起合并，同头尾同类型由 _merge_relations 去重
                derived_relations = cls._derive_customer_relations(filtered_relations)
                merged_relations, _ = cls._merge_relations(
                    filtered_relations + derived_relations, merged_entities
                )

            with root.task("关系校验"):
                cls._check_borrower_guarantor_conflict(merged_relations)
                cls._check_unique_primary_borrower(merged_relations)

            with root.task("计算degree"):
                cls._compute_degree(merged_entities, merged_relations)

            with root.task("实体向量化"):
                await cls._embed_items(merged_entities, EmbedService.entity_to_text, "实体")

            with root.task("关系向量化"):
                await cls._embed_items(merged_relations, EmbedService.relation_to_text, "关系")

            with root.task("构建并保存图谱"):
                graph = KnowledgeGraph(
                    id=graph_id,
                    folderId=folder_id,
                    folderName=folder_name,
                    files=files,
                    entities=merged_entities,
                    relations=merged_relations,
                )
                cls._rebuild_indexes(graph, file_pieces)
                await GraphStorageService.save(graph)

        sw.print()
        logger.info("KnowledgeGraphService | 图谱生成完成 | graph_id={} entities={} relations={}", graph_id, len(merged_entities), len(merged_relations))
        return graph

    # ========== 增量更新相关方法 ==========

    @classmethod
    def _rebuild_indexes(
        cls,
        graph: KnowledgeGraph,
        file_pieces: list[tuple[FileInput, List[Piece]]],
    ) -> None:
        """根据 file_pieces 重建 graph.file_index 和 graph.piece_index"""
        graph.file_index.clear()
        graph.piece_index.clear()
        for f, pieces in file_pieces:
            piece_hashes = [p.content_hash for p in pieces]
            graph.file_index[f.id] = FileMeta(
                file_id=f.id,
                piece_hashes=piece_hashes,
            )
            for p in pieces:
                graph.piece_index.setdefault(p.content_hash, []).append(f.id)

    @classmethod
    def _diff_files(
        cls,
        new_files: List[FileInput],
        file_index: dict[str, FileMeta],
    ) -> tuple[list[FileInput], list[str], list[FileInput]]:
        """文件级 diff（基于 file_id 判别，业务保证 file_id 全局唯一且不可复用）

        Returns:
            (added, deleted_file_ids, unchanged)
            - added: 新增文件（file_id 不在 file_index 中）
            - deleted_file_ids: 删除文件的 file_id 列表（file_index 有但 new_files 没有）
            - unchanged: 未变文件（file_id 都在）
        """
        new_file_map = {f.id: f for f in new_files}
        new_file_ids = set(new_file_map.keys())
        old_file_ids = set(file_index.keys())

        added = [f for fid, f in new_file_map.items() if fid not in old_file_ids]
        deleted_file_ids = list(old_file_ids - new_file_ids)
        unchanged = [f for fid, f in new_file_map.items() if fid in old_file_ids]

        return added, deleted_file_ids, unchanged

    @classmethod
    def _cleanup_deleted_files(
        cls,
        graph: KnowledgeGraph,
        deleted_file_ids: List[str],
    ) -> List[Relation]:
        """清理被删除文件的 references，并单轮判定删除空实体/空关系

        单轮逻辑：
        1. 按 file_id 过滤 entities/relations 的 references
        2. 删除 references 全空的关系
        3. 删除 references 全空 且 未被任何剩余 relation 引用的实体

        Returns:
            deleted_relations: 因清理被移除的关系列表（用于推导 degree 重算的受影响实体）
        """
        if not deleted_file_ids:
            return []

        deleted_set = set(deleted_file_ids)

        # 1. 清理 references
        for entity in graph.entities:
            entity.references = [r for r in entity.references if r.file_id not in deleted_set]
        for relation in graph.relations:
            relation.references = [r for r in relation.references if r.file_id not in deleted_set]

        # 2. 删除空关系
        before_rel = len(graph.relations)
        deleted_relations = [r for r in graph.relations if not r.references]
        graph.relations = [r for r in graph.relations if r.references]

        # 3. 删除空实体（references 全空 且 未被任何 relation 引用）
        used_entity_ids = {r.from_entity_id for r in graph.relations} | {r.to_entity_id for r in graph.relations}
        before_ent = len(graph.entities)
        graph.entities = [
            e for e in graph.entities
            if e.references or e.id in used_entity_ids
        ]

        # 4. 清理 file_index / piece_index
        for fid in deleted_set:
            meta = graph.file_index.pop(fid, None)
            if meta is None:
                continue
            for piece_hash in meta.piece_hashes:
                owners = graph.piece_index.get(piece_hash)
                if not owners:
                    continue
                owners = [x for x in owners if x != fid]
                if owners:
                    graph.piece_index[piece_hash] = owners
                else:
                    graph.piece_index.pop(piece_hash, None)

        logger.info(
            "KnowledgeGraphService | 删除文件清理完成 | deleted_files={} entities: {}→{} relations: {}→{}",
            len(deleted_set), before_ent, len(graph.entities), before_rel, len(graph.relations),
        )
        return deleted_relations

    @classmethod
    async def update(
        cls,
        graph: KnowledgeGraph,
        new_files: List[FileInput],
        raw_response_callback: Callable[[Piece, str], None] | None = None,
    ) -> KnowledgeGraph:
        """增量更新已有图谱

        流程：
        1. 文件级 diff（新增/删除/未变，无"修改"分支，修改视为先删后增）
        2. 删除文件 → 按 file_id 清理 references + 单轮判定删除
        3. 新增文件 → 切片，piece_hash 命中 piece_index 则跳过，否则 LLM 提取
        4. delta 实体/关系合并入已有列表（复用 _merge_entities/_merge_relations 的 existing 参数）
        5. 更新 file_index / piece_index
        6. 持久化（finally 块保证保存）

        Args:
            graph: 已有图谱（会被原地修改）
            new_files: 全量最新文件列表
            raw_response_callback: LLM诊断包络回调
        """
        sw = StopWatch("update_knowledge_graph")
        with sw.task("增量更新图谱") as root:
            added_files, deleted_file_ids, _ = cls._diff_files(new_files, graph.file_index)
            logger.info(
                "KnowledgeGraphService.update | diff完成 | added={} deleted={}",
                len(added_files), len(deleted_file_ids),
            )

            # 同步 graph.files 为最新全量
            graph.files = new_files

            with root.task("清理删除文件"):
                cls._cleanup_deleted_files(graph, deleted_file_ids)

            with root.task("增量提取与合并"):
                # 切片 + piece_hash 去重
                new_pieces: List[Piece] = []
                added_file_pieces: list[tuple[FileInput, List[Piece]]] = []
                for f in added_files:
                    pieces = SplittingService.split(f)
                    added_file_pieces.append((f, pieces))
                    for p in pieces:
                        if p.content_hash in graph.piece_index:
                            # 命中：piece 已被其他文件处理过，复用其提取结果（已在 graph 中）
                            # 仅更新 piece_index 的 owner 列表
                            graph.piece_index[p.content_hash].append(f.id)
                        else:
                            new_pieces.append(p)
                            graph.piece_index[p.content_hash] = [f.id]

                logger.info(
                    "KnowledgeGraphService.update | piece去重 | 新增piece={} 命中跳过={}",
                    len(new_pieces),
                    sum(len(ps) for _, ps in added_file_pieces) - len(new_pieces),
                )

                # 增量 embed 收集：delta（新项）+ changed（attributes 变化的已有项）
                embed_entities: List[Entity] = []
                rel_delta: List[Relation] = []

                if new_pieces:
                    match_stats = create_match_stats()
                    sem = asyncio.Semaphore(3)

                    async def process_one(piece: Piece):
                        async with sem:
                            try:
                                return await ExtractorService.extract(
                                    piece,
                                    folder_name=graph.folderName,
                                    raw_response_callback=raw_response_callback,
                                    match_stats=match_stats,
                                )
                            except FatalKGExtractionError:
                                logger.exception(
                                    "KnowledgeGraphService.update | piece提取重试耗尽，跳过 | piece_id={} file_name={}",
                                    piece.id, piece.file_name,
                                )
                                return [], []
                            except Exception:
                                logger.exception(
                                    "KnowledgeGraphService.update | piece提取失败，跳过 | piece_id={} file_name={}",
                                    piece.id, piece.file_name,
                                )
                                return [], []

                    tasks = [asyncio.create_task(process_one(p)) for p in new_pieces]
                    results = await asyncio.gather(*tasks)

                    delta_entities: List[Entity] = []
                    delta_relations: List[Relation] = []
                    for entities, relations in results:
                        delta_entities.extend(entities)
                        delta_relations.extend(relations)

                    # 合并入已有列表（复用 entity_id / relation key）
                    graph.entities, ent_delta, ent_changed = cls._merge_entities(
                        delta_entities, existing=graph.entities
                    )
                    for entity in graph.entities:
                        if not entity.id:
                            entity.id = entity_id(
                                graph.folderId, entity.name, entity.type
                            )
                    # 反向派生客户关系（仅针对 delta 中的供应商关系，已有的在 existing 中复用）
                    derived_relations = cls._derive_customer_relations(delta_relations)
                    graph.relations, rel_delta = cls._merge_relations(
                        delta_relations + derived_relations,
                        graph.entities,
                        existing=graph.relations,
                    )

                    # 收集需要重新向量化的实体（delta + changed）
                    embed_entities.extend(ent_delta)
                    embed_entities.extend(ent_changed)

                    if match_stats["total"] > 0:
                        logger.info(
                            "KnowledgeGraphService.update | 溯源匹配统计 | "
                            "总尝试={} 匹配成功={}({:.1f}%) 文本失配={} element不存在={} 空输入={}",
                            match_stats["total"],
                            match_stats["matched"],
                            match_stats["matched"] / match_stats["total"] * 100,
                            match_stats["source_mismatch"],
                            match_stats["element_not_found"],
                            match_stats["empty_source"],
                        )

                # 关系校验（仅 warning，增量触发）
                if rel_delta:
                    cls._check_borrower_guarantor_conflict(
                        rel_delta, all_relations=graph.relations
                    )
                    cls._check_unique_primary_borrower(
                        rel_delta, all_relations=graph.relations
                    )

                # degree 全量重算
                cls._compute_degree(graph.entities, graph.relations)

                # 增量向量化：仅对 delta + changed 调 embed
                if embed_entities:
                    await cls._embed_items(embed_entities, EmbedService.entity_to_text, "实体")
                if rel_delta:
                    await cls._embed_items(rel_delta, EmbedService.relation_to_text, "关系")

            # 更新 file_index（新增/修改的文件）
            with root.task("更新索引并持久化"):
                for f, pieces in added_file_pieces:
                    graph.file_index[f.id] = FileMeta(
                        file_id=f.id,
                        piece_hashes=[p.content_hash for p in pieces],
                    )

                try:
                    await GraphStorageService.save(graph)
                except Exception:
                    logger.exception("KnowledgeGraphService.update | 持久化失败 | graph_id={}", graph.id)
                    raise

        sw.print()
        logger.info(
            "KnowledgeGraphService.update | 增量更新完成 | graph_id={} entities={} relations={}",
            graph.id, len(graph.entities), len(graph.relations),
        )
        return graph

    @classmethod
    def _merge_entities(
        cls,
        entities: List[Entity],
        existing: Optional[List[Entity]] = None,
    ) -> tuple[List[Entity], List[Entity], List[Entity]]:
        """合并名称和类型均相同的实体

        Args:
            entities: 待合并的实体列表（来自本次提取）
            existing: 已有实体列表（增量场景）。若提供，则将 entities 合并入 existing，
                     复用 existing 中的 entity_id（按 name+type 匹配），references/attributes 累加。
                     若为 None，则等价于全量合并。

        Returns:
            (merged, delta, changed)
            - merged: 合并后的全量实体列表
            - delta: 本次新增的实体（仅增量场景非空，全量场景等于 merged）
            - changed: attributes 被更新的已有实体（embed 文本会变，需重新向量化）
        """
        entity_map: dict[tuple[str, str], Entity] = {}
        changed_keys: set[tuple[str, str]] = set()

        # 增量场景：先用 existing 初始化 entity_map，保留原 id
        if existing:
            for entity in existing:
                key = (entity.name, entity.type)
                entity_map[key] = entity.model_copy(deep=True)

        for entity in entities:
            key = (entity.name, entity.type)
            if key not in entity_map:
                entity_map[key] = entity.model_copy(deep=True)
            else:
                existing_entity = entity_map[key]
                existing_entity.references.extend(entity.references)
                attrs_changed = any(
                    existing_entity.attributes.get(k) != v
                    for k, v in entity.attributes.items()
                )
                existing_entity.attributes.update(entity.attributes)
                if attrs_changed:
                    changed_keys.add(key)

        for entity in entity_map.values():
            mapped_type_en = ENTITY_TYPE_MAP.get(entity.type)
            if mapped_type_en is not None:
                entity.type_en = mapped_type_en
            elif entity.type_en == entity.type:
                logger.warning(
                    "KnowledgeGraphService | 合并后未知实体类型 | type={}",
                    entity.type,
                )

        merged_list = list(entity_map.values())

        if existing is None:
            # 全量场景：所有合并后的实体都视为 delta
            return merged_list, list(merged_list), []
        else:
            # 增量场景：delta = existing 中没有的；changed = attributes 变化的已有实体
            existing_keys = {(e.name, e.type) for e in existing}
            delta_list = [e for e in merged_list if (e.name, e.type) not in existing_keys]
            changed_list = [e for e in merged_list if (e.name, e.type) in changed_keys]
            return merged_list, delta_list, changed_list

    @classmethod
    def _normalize_enterprise_abbreviations(
        cls, entities: List[Entity], relations: List[Relation]
    ) -> None:
        """将企业简称和等价组织形式归一为实体列表中已有的最长名称。"""
        enterprise_names = {
            entity.name for entity in entities if entity.type == _ENTERPRISE_TYPE
        }
        name_remap: dict[str, str] = {}

        # 即使裸简称不存在，也归一同一组织形式的两种法定名称。
        for short_suffix, long_suffix in _ENTERPRISE_SUFFIX_FAMILIES:
            for long_name in enterprise_names:
                if not long_name.endswith(long_suffix):
                    continue
                stem = long_name[: -len(long_suffix)]
                short_name = f"{stem}{short_suffix}"
                if short_name not in enterprise_names:
                    continue
                # “股份有限公司”优先解释为更长的股份公司后缀。
                if (
                    short_suffix == "有限公司"
                    and short_name.endswith("股份有限公司")
                ):
                    continue
                name_remap[short_name] = long_name

        candidates = [
            entity.name
            for entity in entities
            if entity.type == _ENTERPRISE_TYPE
            and not any(marker in entity.name for marker in _ENTERPRISE_NAME_MARKERS)
        ]
        aliases_by_canonical: dict[str, list[str]] = {}
        conflict_count = 0

        for abbreviation in candidates:
            matched_families: list[tuple[str, list[str]]] = []
            for short_suffix, long_suffix in _ENTERPRISE_SUFFIX_FAMILIES:
                short_name = f"{abbreviation}{short_suffix}"
                long_name = f"{abbreviation}{long_suffix}"
                hits: list[str] = []
                if short_name in enterprise_names and not (
                    short_suffix == "有限公司"
                    and short_name.endswith("股份有限公司")
                ):
                    hits.append(short_name)
                if long_name in enterprise_names:
                    hits.append(long_name)
                if not hits:
                    continue

                canonical_name = long_name if long_name in hits else short_name
                for hit in hits:
                    if hit != canonical_name:
                        name_remap[hit] = canonical_name
                matched_families.append((canonical_name, hits))

            if len(matched_families) == 1:
                canonical_name = matched_families[0][0]
                name_remap[abbreviation] = canonical_name
                aliases_by_canonical.setdefault(canonical_name, []).append(abbreviation)
            elif len(matched_families) > 1:
                conflict_count += 1
                logger.warning(
                    "KnowledgeGraphService | 企业简称跨类型冲突，保留简称 | "
                    "abbreviation={} targets={}",
                    abbreviation,
                    [canonical for canonical, _ in matched_families],
                )

        entities_by_canonical: dict[str, list[Entity]] = defaultdict(list)
        for entity in entities:
            if entity.type != _ENTERPRISE_TYPE:
                continue
            canonical_name = name_remap.get(entity.name, entity.name)
            entities_by_canonical[canonical_name].append(entity)

        for canonical_name, generated_aliases in aliases_by_canonical.items():
            aliases: list[str] = []
            canonical_entities = entities_by_canonical[canonical_name]
            for entity in canonical_entities:
                existing = entity.attributes.get(_ABBREVIATION_ATTRIBUTE)
                if isinstance(existing, str):
                    for alias in existing.split(_ABBREVIATION_SEPARATOR):
                        alias = alias.strip()
                        if alias and alias not in aliases:
                            aliases.append(alias)
            for alias in generated_aliases:
                if alias not in aliases:
                    aliases.append(alias)

            abbreviation_value = _ABBREVIATION_SEPARATOR.join(aliases)
            for entity in canonical_entities:
                entity.attributes[_ABBREVIATION_ATTRIBUTE] = abbreviation_value

        for entity in entities:
            if entity.type == _ENTERPRISE_TYPE:
                entity.name = name_remap.get(entity.name, entity.name)
        for relation in relations:
            relation_schema = _RELATION_SCHEMA.get(relation.type)
            if (
                relation_schema is not None
                and _ENTERPRISE_TYPE in relation_schema.source_types
            ):
                relation.from_entity = name_remap.get(
                    relation.from_entity, relation.from_entity
                )
            if (
                relation_schema is not None
                and _ENTERPRISE_TYPE in relation_schema.target_types
            ):
                relation.to_entity = name_remap.get(
                    relation.to_entity, relation.to_entity
                )

        logger.info(
            "KnowledgeGraphService | 企业简称归一完成 | "
            "candidates={} remapped_names={} conflicts={}",
            len(candidates),
            len(name_remap),
            conflict_count,
        )

    @classmethod
    def _filter_relations_by_schema(
        cls,
        relations: List[Relation],
        merged_entities: List[Entity],
        project_name: str,
    ) -> List[Relation]:
        """删除关系类型或已知端点类型不符合 schema 的关系。"""
        entities_by_name = _group_entities_by_name(merged_entities)
        filtered: List[Relation] = []
        valid_count = 0
        unresolved_count = 0
        deleted_count = 0

        for relation in relations:
            relation.from_entity_id = ""
            relation.to_entity_id = ""
            relation_schema = _RELATION_SCHEMA.get(relation.type)
            reason = None
            reason_endpoint = None
            from_entity = None
            to_entity = None
            from_status = "missing"
            to_status = "missing"

            if relation_schema is None:
                reason = "unknown_relation_type"
            else:
                from_entity, from_status = _resolve_entity_candidate(
                    entities_by_name.get(relation.from_entity, []),
                    relation_schema.source_types,
                )
                if from_status == "ambiguous_endpoint_type":
                    reason = from_status
                    reason_endpoint = "from"
                elif from_status == "type_mismatch":
                    reason = "from_type_mismatch"
                    reason_endpoint = "from"

            if relation_schema is not None and reason is None:
                if "项目" in relation_schema.target_types:
                    if relation.to_entity != project_name:
                        reason = "invalid_project_target"
                        reason_endpoint = "to"
                    else:
                        to_entity, to_status = _resolve_entity_candidate(
                            entities_by_name.get(relation.to_entity, []),
                            ["项目"],
                        )
                        if to_status != "resolved":
                            reason = "invalid_project_target"
                            reason_endpoint = "to"
                else:
                    to_entity, to_status = _resolve_entity_candidate(
                        entities_by_name.get(relation.to_entity, []),
                        relation_schema.target_types,
                    )
                    if to_status == "ambiguous_endpoint_type":
                        reason = to_status
                        reason_endpoint = "to"
                    elif to_status == "type_mismatch":
                        reason = "to_type_mismatch"
                        reason_endpoint = "to"

            if reason is not None:
                deleted_count += 1
                logger.warning(
                    "KnowledgeGraphService | 删除不符合schema的关系 | reason={} type={} "
                    "endpoint={} from_entity={} from_types={} to_entity={} to_types={}",
                    reason,
                    relation.type,
                    reason_endpoint,
                    relation.from_entity,
                    [
                        entity.type
                        for entity in entities_by_name.get(relation.from_entity, [])
                    ],
                    relation.to_entity,
                    [
                        entity.type
                        for entity in entities_by_name.get(relation.to_entity, [])
                    ],
                )
                continue

            if from_entity is not None:
                relation.from_entity_id = from_entity.id
            if to_entity is not None:
                relation.to_entity_id = to_entity.id
            filtered.append(relation)
            if from_status in {"missing", "unknown"} or to_status in {
                "missing",
                "unknown",
            }:
                unresolved_count += 1
                logger.warning(
                    "KnowledgeGraphService | 保留端点类型未决关系 | type={} "
                    "from_entity={} from_status={} to_entity={} to_status={}",
                    relation.type,
                    relation.from_entity,
                    from_status,
                    relation.to_entity,
                    to_status,
                )
            else:
                valid_count += 1

        logger.info(
            "KnowledgeGraphService | 关系schema过滤完成 | total={} valid={} unresolved={} deleted={}",
            len(relations),
            valid_count,
            unresolved_count,
            deleted_count,
        )
        return filtered

    @classmethod
    def _merge_relations(
        cls,
        relations: List[Relation],
        merged_entities: List[Entity],
        existing: Optional[List[Relation]] = None,
    ) -> tuple[List[Relation], List[Relation]]:
        """按已解析的实体ID合并关系；端点无法解析的关系将被跳过。

        Args:
            relations: 待合并的关系列表（来自本次提取，from_entity_id/to_entity_id 可能为空）
            merged_entities: 已合并的实体列表，用于 name → id 映射
            existing: 已有关系列表（增量场景）。若提供，则将 relations 合并入 existing，
                     复用 existing 中 (from_id, to_id, type) 匹配的关系，references/attributes 累加。
                     若为 None，则等价于全量合并。

        Returns:
            (merged, delta)
            - merged: 合并后的全量关系列表
            - delta: 本次新增的关系（关系 embed 文本不依赖 references/attributes，已存在关系无需重算）
        """
        entities_by_id = {entity.id: entity for entity in merged_entities}
        entities_by_name = _group_entities_by_name(merged_entities)

        def resolve_existing_endpoint(
            endpoint_id: str,
            endpoint_name: str,
        ) -> tuple[str | None, str]:
            if endpoint_id:
                if endpoint_id in entities_by_id:
                    return endpoint_id, "resolved"
                return None, "invalid_id"

            candidates = entities_by_name.get(endpoint_name, [])
            if len(candidates) == 1:
                return candidates[0].id, "resolved"
            if len(candidates) > 1:
                return None, "ambiguous_name"
            return None, "missing"

        # 增量场景：用 existing 初始化 merged，保证已有关系被复用
        merged: dict[tuple[str, str, str], Relation] = {}
        existing_keys: set[tuple[str, str, str]] = set()
        if existing:
            for rel in existing:
                key = (rel.from_entity_id, rel.to_entity_id, rel.type)
                merged[key] = rel.model_copy(deep=True)
                existing_keys.add(key)

        delta_list: List[Relation] = []

        for rel in relations:
            from_id, from_status = resolve_existing_endpoint(
                rel.from_entity_id, rel.from_entity
            )
            to_id, to_status = resolve_existing_endpoint(
                rel.to_entity_id, rel.to_entity
            )

            if from_id is None or to_id is None:
                logger.warning(
                    "KnowledgeGraphService | 跳过端点未解析关系 | type={} "
                    "from_entity={} from_status={} to_entity={} to_status={}",
                    rel.type,
                    rel.from_entity,
                    from_status,
                    rel.to_entity,
                    to_status,
                )
                continue

            rel.from_entity_id = from_id
            rel.to_entity_id = to_id

            relation_schema = _RELATION_SCHEMA.get(rel.type)
            if relation_schema is not None and relation_schema.is_bidirectional:
                key = (min(from_id, to_id), max(from_id, to_id), rel.type)
            else:
                key = (from_id, to_id, rel.type)
            rel.id = relation_id(*key)
            if key in merged:
                _merge_relation_content(merged[key], rel)
            else:
                merged[key] = rel.model_copy(deep=True)
                delta_list.append(merged[key])

        return list(merged.values()), delta_list

    @classmethod
    def _derive_customer_relations(cls, relations: List[Relation]) -> List[Relation]:
        """根据供应商关系反向派生客户关系（CUSTOMER_OF）。

        对每条 SUPPLIER_OF 关系 A→B，派生一条 CUSTOMER_OF 关系 B→A，
        references/attributes 复用原关系（深拷贝）。同头尾同类型的去重交给后续 _merge_relations。
        """
        derived: List[Relation] = []
        for rel in relations:
            if rel.type != _SUPPLIER_RELATION_TYPE:
                continue
            if not rel.from_entity_id or not rel.to_entity_id:
                continue
            customer_rel = Relation(
                type=_CUSTOMER_RELATION_TYPE,
                type_en="CUSTOMER_OF",
                from_entity=rel.to_entity,
                from_entity_id=rel.to_entity_id,
                to_entity=rel.from_entity,
                to_entity_id=rel.from_entity_id,
                attributes={k: list(v) for k, v in rel.attributes.items()},
                references=[ref.model_copy(deep=True) for ref in rel.references],
                degree=rel.degree,
            )
            derived.append(customer_rel)
        if derived:
            logger.info(
                "KnowledgeGraphService | 反向派生客户关系 | supplier={} derived={}",
                sum(1 for r in relations if r.type == _SUPPLIER_RELATION_TYPE),
                len(derived),
            )
        return derived

    @classmethod
    def _check_borrower_guarantor_conflict(
        cls,
        relations: List[Relation],
        all_relations: Optional[List[Relation]] = None,
    ) -> None:
        """校验同一实体不能同时是同一项目的主借款人和保证人，违反时仅 warning。

        增量模式：传入 all_relations 时，只对 delta 中出现的主借款人/保证人做冲突检查，
        避免全量扫描。冲突只可能在 delta 与 all（含 delta）之间发生。
        """
        scan_relations = all_relations if all_relations is not None else relations
        # 全量场景下 relations 即 scan_relations；增量场景下 scan_relations 应为全量
        borrower_pairs: dict[tuple[str, str], str] = {}
        guarantor_pairs: dict[tuple[str, str], str] = {}
        for rel in scan_relations:
            if rel.type == "主借款人":
                key = (rel.from_entity_id, rel.to_entity_id)
                if key not in borrower_pairs:
                    borrower_pairs[key] = rel.from_entity
            elif rel.type == "保证人":
                key = (rel.from_entity_id, rel.to_entity_id)
                if key not in guarantor_pairs:
                    guarantor_pairs[key] = rel.from_entity

        # 增量模式：只对 delta 中出现的关系触发 warning
        delta_borrower_keys = set()
        delta_guarantor_keys = set()
        if all_relations is not None:
            for rel in relations:
                if rel.type == "主借款人":
                    delta_borrower_keys.add((rel.from_entity_id, rel.to_entity_id))
                elif rel.type == "保证人":
                    delta_guarantor_keys.add((rel.from_entity_id, rel.to_entity_id))

        for key, borrower_name in borrower_pairs.items():
            if key in guarantor_pairs:
                # 增量模式：至少一端在 delta 中才 warning
                if all_relations is not None and key not in delta_borrower_keys and key not in delta_guarantor_keys:
                    continue
                logger.warning(
                    "KnowledgeGraphService | 借款人保证人冲突 | "
                    "entity={} to_entity_id={}",
                    borrower_name,
                    key[1],
                )

    @classmethod
    def _check_unique_primary_borrower(
        cls,
        relations: List[Relation],
        all_relations: Optional[List[Relation]] = None,
    ) -> None:
        """校验同一项目主借款人唯一，违反时仅 warning。

        增量模式：传入 all_relations 时，只对 delta 中出现的主借款人对应项目做检查。
        """
        scan_relations = all_relations if all_relations is not None else relations
        project_borrowers: dict[str, list[str]] = defaultdict(list)
        for rel in scan_relations:
            if rel.type == "主借款人":
                project_borrowers[rel.to_entity_id].append(rel.from_entity)

        # 增量模式：只对 delta 中出现的项目触发 warning
        delta_projects: set[str] = set()
        if all_relations is not None:
            for rel in relations:
                if rel.type == "主借款人":
                    delta_projects.add(rel.to_entity_id)

        for to_entity_id, borrower_names in project_borrowers.items():
            if len(borrower_names) > 1:
                if all_relations is not None and to_entity_id not in delta_projects:
                    continue
                logger.warning(
                    "KnowledgeGraphService | 主借款人不唯一 | "
                    "to_entity_id={} count={} borrowers={}",
                    to_entity_id,
                    len(borrower_names),
                    borrower_names,
                )

    @classmethod
    async def _embed_items(
        cls,
        items: List,
        to_text: Callable[[Any], str],
        tag: str,
    ) -> None:
        """对实体或关系批量向量化，结果就地写回 items[i].embed。

        Args:
            items: 实体或关系列表
            to_text: EmbedService.entity_to_text / relation_to_text
            tag: 日志标签，"实体" 或 "关系"
        """
        if not items:
            return
        batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE"))
        texts = [to_text(item) for item in items]

        async def embed_one_batch(batch, offset):
            try:
                vectors = await EmbedService.embed_texts(batch)
                for j, v in enumerate(vectors):
                    items[offset + j].embed = v
            except Exception:
                logger.exception(
                    "EmbedService | {}向量化批次失败 | offset={} size={}",
                    tag, offset, len(batch),
                )

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        await asyncio.gather(*[embed_one_batch(b, i * batch_size) for i, b in enumerate(batches)])

    @classmethod
    def _compute_degree(
        cls,
        entities: List[Entity],
        relations: List[Relation],
    ) -> None:
        """计算实体的degree（连接关系数量）和关系的degree（两端实体degree之和）"""
        degree_map: dict[str, int] = {entity.id: 0 for entity in entities}
        for rel in relations:
            degree_map[rel.from_entity_id] = degree_map.get(rel.from_entity_id, 0) + 1
            degree_map[rel.to_entity_id] = degree_map.get(rel.to_entity_id, 0) + 1
        for entity in entities:
            entity.degree = degree_map.get(entity.id, 0)
        for rel in relations:
            rel.degree = degree_map.get(rel.from_entity_id, 0) + degree_map.get(rel.to_entity_id, 0)
