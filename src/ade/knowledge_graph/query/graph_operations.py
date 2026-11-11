"""知识图谱查询 Phase 2：无状态图操作。"""

from collections import defaultdict, deque
from typing import Optional

import numpy as np
from loguru import logger

from src.ade.knowledge_graph.schema.kg_schema import Entity, Relation


def entity_lookup(
    query_vec: np.ndarray,
    entities: list[Entity],
    target_entity_types: Optional[list[str]] = None,
) -> list[Entity]:
    """
    向量搜索实体，可按类型过滤。
    返回按相似度降序排列的实体列表（80% 截止）。
    """
    candidates = entities
    if target_entity_types:
        type_set = set(target_entity_types)
        candidates = [e for e in entities if e.type in type_set]

    if not candidates:
        return []

    scored = []
    for entity in candidates:
        if entity.embed is None:
            continue
        emb = np.array(entity.embed)
        score = float(np.dot(query_vec, emb))
        scored.append((score, entity))

    scored.sort(key=lambda x: x[0], reverse=True)

    result = []
    first_score = scored[0][0] if scored else 0
    for score, entity in scored:
        if score < first_score * 0.8:
            break
        result.append(entity)

    logger.info(
        "GraphOps | 实体向量搜索完成 | total={}, matched={}",
        len(candidates),
        len(result),
    )
    return result


def relation_search(
    query_vec: np.ndarray,
    relations: list[Relation],
    target_relation_types: Optional[list[str]] = None,
) -> list[Relation]:
    """
    向量搜索关系，可按类型过滤。
    返回按相似度降序排列的关系列表（80% 截止）。
    """
    candidates = relations
    if target_relation_types:
        type_set = set(target_relation_types)
        candidates = [r for r in relations if r.type in type_set]

    if not candidates:
        return []

    scored = []
    for relation in candidates:
        if relation.embed is None:
            continue
        emb = np.array(relation.embed)
        score = float(np.dot(query_vec, emb))
        scored.append((score, relation))

    scored.sort(key=lambda x: x[0], reverse=True)

    result = []
    first_score = scored[0][0] if scored else 0
    for score, relation in scored:
        if score < first_score * 0.8:
            break
        result.append(relation)

    logger.info(
        "GraphOps | 关系向量搜索完成 | total={}, matched={}",
        len(candidates),
        len(result),
    )
    return result


def attribute_extract(
    entity: Entity,
    requested_attributes: list[str],
) -> dict[str, str]:
    """
    从实体属性中精准提取指定属性。
    返回 {属性名: 属性值}，未找到的属性不出现在结果中。
    """
    if not requested_attributes or not entity.attributes:
        return {}

    result = {}
    attrs_lower = {k.lower(): v for k, v in entity.attributes.items()}
    for attr in requested_attributes:
        if attr.lower() in attrs_lower:
            result[attr] = attrs_lower[attr.lower()]
    return result


def graph_traverse(
    start_entities: list[Entity],
    all_relations: list[Relation],
    target_relation_types: Optional[list[str]] = None,
    max_hops: int = 2,
) -> list[list[tuple[str, str, str]]]:
    """
    BFS 多跳图遍历。
    从起始实体出发，沿目标关系类型遍历 max_hops 层。
    返回路径列表，每条路径为 [(entity_name, relation_type, entity_name), ...]。
    遍历是双向的（from_entity 和 to_entity 方向均可走）。
    """
    if not start_entities or max_hops < 1:
        return []

    type_set = set(target_relation_types) if target_relation_types else None

    # 构建双向邻接表
    adj: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in all_relations:
        if type_set and r.type not in type_set:
            continue
        adj[r.from_entity].append((r.type, r.to_entity, r.id))
        adj[r.to_entity].append((r.type, r.from_entity, r.id))

    all_paths: list[list[tuple[str, str, str]]] = []
    seen_path_sigs: set[tuple] = set()

    for start in start_entities:
        queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque()
        queue.append((start.name, []))  # (current_entity_name, path_so_far)

        for hop in range(max_hops):
            for _ in range(len(queue)):
                current, path = queue.popleft()
                for rel_type, neighbor, rel_id in adj.get(current, []):
                    new_path = path + [(current, rel_type, neighbor)]
                    sig = tuple((p[0], p[1], p[2]) for p in new_path)
                    if sig not in seen_path_sigs:
                        seen_path_sigs.add(sig)
                        all_paths.append(new_path)
                        queue.append((neighbor, new_path))

    logger.info(
        "GraphOps | 图遍历完成 | start_entities={}, hops={}, paths={}",
        len(start_entities),
        max_hops,
        len(all_paths),
    )
    return all_paths


def aggregate_count_by_type(entities: list[Entity]) -> dict[str, int]:
    """按实体类型统计数量"""
    counts: dict[str, int] = defaultdict(int)
    for e in entities:
        counts[e.type] += 1
    return dict(counts)


def expand_relations(
    matched_entities: list[Entity],
    all_relations: list[Relation],
    target_relation_types: Optional[list[str]] = None,
) -> list[Relation]:
    """
    获取与已匹配实体关联的所有关系（一层）。
    按实体顺序遍历，支持关系类型过滤。
    """
    type_set = set(target_relation_types) if target_relation_types else None
    seen_ids: set[str] = set()
    result: list[Relation] = []

    for entity in matched_entities:
        for r in all_relations:
            if r.id in seen_ids:
                continue
            if r.from_entity != entity.name and r.to_entity != entity.name:
                continue
            if type_set and r.type not in type_set:
                continue
            seen_ids.add(r.id)
            result.append(r)

    logger.info(
        "GraphOps | 关系扩展完成 | entities={}, expanded_relations={}",
        len(matched_entities),
        len(result),
    )
    return result


def merge_relations_unique(
    vec_relations: list[Relation],
    expanded_relations: list[Relation],
) -> list[Relation]:
    """合并向量搜索和扩展的关系，去重，向量搜索结果在前"""
    seen_ids: set[str] = set()
    result: list[Relation] = []
    for r in vec_relations:
        if r.id not in seen_ids:
            seen_ids.add(r.id)
            result.append(r)
    for r in expanded_relations:
        if r.id not in seen_ids:
            seen_ids.add(r.id)
            result.append(r)
    return result


def collect_entities_from_paths(
    paths: list[list[tuple[str, str, str]]],
    all_entities: list[Entity],
) -> list[Entity]:
    """从遍历路径中收集涉及的实体对象"""
    name_to_entity = {e.name: e for e in all_entities}
    seen_names: set[str] = set()
    result: list[Entity] = []
    for path in paths:
        for src, _, dst in path:
            for name in (src, dst):
                if name not in seen_names and name in name_to_entity:
                    seen_names.add(name)
                    result.append(name_to_entity[name])
    return result
