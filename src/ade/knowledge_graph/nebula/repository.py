"""Folder-scoped NebulaGraph query replica."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TypeVar

from loguru import logger

from src.ade.knowledge_graph.nebula.client import NebulaClient, NebulaGraphError
from src.ade.knowledge_graph.nebula.schema import (
    BIDIRECTIONAL_RELATION_TYPES,
    EDGE_PROPERTIES,
    ENTITY_TAG_BY_TYPE,
    ENTITY_TYPE_BY_TAG,
    ENTITY_TYPE_EN_BY_TYPE,
    RELATION_EDGE_BY_TYPE,
    RELATION_TYPE_BY_EDGE,
    RELATION_TYPE_EN_BY_TYPE,
    SNAPSHOT_PROPERTIES,
    SNAPSHOT_TAG,
    VERTEX_PROPERTIES,
    folder_key,
    schema_statements,
)
from src.ade.knowledge_graph.schema.kg_schema import (
    Entity,
    GraphData,
    KnowledgeGraph,
    Relation,
    entity_id,
    relation_id,
)
from src.base.distributed_lock import DistributedLock
from src.utils.json_utils import dumps as json_dumps, loads as json_loads

_BATCH_SIZE = 100
_T = TypeVar("_T")
_VID_PATTERN = re.compile(r"^ent_[0-9a-f]{16}$")
EdgeKey = tuple[str, str, str, int]


@dataclass(frozen=True)
class _Vertex:
    vid: str
    tag: str
    folder_key: str
    folder_id: str
    name: str
    attributes_json: str
    references_json: str


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    edge_type: str
    attributes_json: str
    references_json: str


@dataclass(frozen=True)
class KGSnapshot:
    folder_id: str
    graph_id: str
    snapshot_version: str
    ready: bool


def _batches(values: Sequence[_T]) -> Iterable[Sequence[_T]]:
    for offset in range(0, len(values), _BATCH_SIZE):
        yield values[offset : offset + _BATCH_SIZE]


def _vid_literal(vid: str) -> str:
    return json_dumps(vid)


def _snapshot_vid(folder_id: str) -> str:
    digest = hashlib.blake2b(folder_id.encode(), digest_size=8).hexdigest()
    return f"snapshot_{digest}"


class NebulaGraphRepository:
    """Persists only the fields needed by the NebulaGraph query replica."""

    @classmethod
    async def initialize_schema(cls) -> None:
        await asyncio.to_thread(cls._initialize_schema_sync)

    @classmethod
    async def replace(
        cls,
        graph: KnowledgeGraph,
        *,
        acquire_lock: bool = True,
    ) -> None:
        vertices, edges = cls._prepare(graph)
        if acquire_lock:
            async with DistributedLock.acquire(f"kg:{graph.folderId}"):
                await asyncio.to_thread(
                    cls._replace_sync, graph.folderId, vertices, edges
                )
            return
        await asyncio.to_thread(cls._replace_sync, graph.folderId, vertices, edges)

    @classmethod
    async def delete(
        cls,
        folder_id: str,
        *,
        acquire_lock: bool = True,
    ) -> None:
        if acquire_lock:
            async with DistributedLock.acquire(f"kg:{folder_id}"):
                await asyncio.to_thread(cls._delete_sync, folder_id)
            return
        await asyncio.to_thread(cls._delete_sync, folder_id)

    @classmethod
    async def get(cls, folder_id: str) -> GraphData | None:
        async with DistributedLock.acquire(f"kg:{folder_id}"):
            return await asyncio.to_thread(cls._get_sync, folder_id)

    @classmethod
    async def mark_snapshot(
        cls,
        folder_id: str,
        graph_id: str,
        snapshot_version: str,
        *,
        ready: bool,
    ) -> None:
        await asyncio.to_thread(
            cls._mark_snapshot_sync,
            folder_id,
            graph_id,
            snapshot_version,
            ready,
        )

    @classmethod
    async def get_snapshot(cls, folder_id: str) -> KGSnapshot | None:
        return await cls._run_query(cls._get_snapshot_sync, folder_id)

    @classmethod
    async def delete_snapshot(cls, folder_id: str) -> None:
        await asyncio.to_thread(cls._delete_snapshot_sync, folder_id)

    @classmethod
    async def execute_read(
        cls,
        folder_id: str,
        statement: str,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows = await cls._run_query(cls._execute_read_sync, statement, parameters)
        logger.info(
            "NebulaGraphRepository | 只读查询完成 | folder_id={} rows={}",
            folder_id,
            len(rows),
        )
        return rows

    @classmethod
    async def fetch_entities(
        cls,
        folder_id: str,
        vids: Sequence[str],
    ) -> list[Entity]:
        unique_vids = sorted(set(vids))
        if not unique_vids:
            return []
        if any(not _VID_PATTERN.fullmatch(vid) for vid in unique_vids):
            raise NebulaGraphError("Invalid entity VID in hydration request")
        entities = await cls._run_query(
            cls._fetch_entities_sync,
            folder_id,
            unique_vids,
        )
        return sorted(entities.values(), key=lambda entity: entity.id)

    @classmethod
    async def fetch_relations(
        cls,
        folder_id: str,
        edge_keys: Sequence[EdgeKey],
    ) -> list[Relation]:
        unique_keys = sorted(set(edge_keys))
        if not unique_keys:
            return []
        return await cls._run_query(
            cls._fetch_relations_sync,
            folder_id,
            unique_keys,
        )

    @classmethod
    async def _run_query(cls, function, *args):
        timeout_s = NebulaClient._timeout_ms() / 1000
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(function, *args),
                timeout=timeout_s,
            )
        except TimeoutError as error:
            raise NebulaGraphError(
                f"NebulaGraph query exceeded {NebulaClient._timeout_ms()} ms"
            ) from error

    @classmethod
    def _prepare(
        cls, graph: KnowledgeGraph
    ) -> tuple[list[_Vertex], list[_Edge]]:
        key = folder_key(graph.folderId)
        vertices: list[_Vertex] = []
        entities_by_id: dict[str, Entity] = {}

        for entity in graph.entities:
            tag = ENTITY_TAG_BY_TYPE.get(entity.type)
            expected_type_en = ENTITY_TYPE_EN_BY_TYPE.get(entity.type)
            if tag is None or expected_type_en is None:
                raise ValueError(f"Unsupported entity type: {entity.type!r}")
            if entity.type_en != expected_type_en:
                raise ValueError(
                    f"Entity type mapping mismatch: {entity.type!r}/{entity.type_en!r}"
                )
            expected_id = entity_id(graph.folderId, entity.name, entity.type)
            if entity.id != expected_id:
                raise ValueError(
                    f"Entity ID does not match current hash algorithm: {entity.id!r}"
                )
            if entity.id in entities_by_id:
                raise ValueError(f"Duplicate entity ID: {entity.id!r}")

            entities_by_id[entity.id] = entity
            vertices.append(
                _Vertex(
                    vid=entity.id,
                    tag=tag,
                    folder_key=key,
                    folder_id=graph.folderId,
                    name=entity.name,
                    attributes_json=json_dumps(entity.attributes, sort_keys=True),
                    references_json=json_dumps(
                        [
                            reference.model_dump(mode="json")
                            for reference in entity.references
                        ],
                        sort_keys=True,
                    ),
                )
            )

        edges: list[_Edge] = []
        edge_keys: set[tuple[str, str, str]] = set()
        for relation in graph.relations:
            edge_type = RELATION_EDGE_BY_TYPE.get(relation.type)
            expected_type_en = RELATION_TYPE_EN_BY_TYPE.get(relation.type)
            if edge_type is None or expected_type_en is None:
                raise ValueError(f"Unsupported relation type: {relation.type!r}")
            if relation.type_en != expected_type_en:
                raise ValueError(
                    f"Relation type mapping mismatch: "
                    f"{relation.type!r}/{relation.type_en!r}"
                )

            source = entities_by_id.get(relation.from_entity_id)
            target = entities_by_id.get(relation.to_entity_id)
            if source is None or target is None:
                raise ValueError(f"Relation endpoint is missing: {relation.id!r}")
            if (
                relation.from_entity != source.name
                or relation.to_entity != target.name
            ):
                raise ValueError(f"Relation endpoint name mismatch: {relation.id!r}")

            id_source, id_target = relation.from_entity_id, relation.to_entity_id
            if relation.type in BIDIRECTIONAL_RELATION_TYPES:
                id_source, id_target = sorted((id_source, id_target))
            expected_id = relation_id(id_source, id_target, relation.type)
            if relation.id != expected_id:
                raise ValueError(
                    f"Relation ID does not match current hash algorithm: {relation.id!r}"
                )

            edge_key = (
                relation.from_entity_id,
                relation.to_entity_id,
                edge_type,
            )
            if edge_key in edge_keys:
                raise ValueError(f"Duplicate NebulaGraph edge: {edge_key!r}")
            edge_keys.add(edge_key)
            edges.append(
                _Edge(
                    source=relation.from_entity_id,
                    target=relation.to_entity_id,
                    edge_type=edge_type,
                    attributes_json=json_dumps(
                        relation.attributes, sort_keys=True
                    ),
                    references_json=json_dumps(
                        [
                            reference.model_dump(mode="json")
                            for reference in relation.references
                        ],
                        sort_keys=True,
                    ),
                )
            )

        return vertices, edges

    @classmethod
    def _initialize_schema_sync(cls) -> None:
        with NebulaClient.session() as session:
            for statement in schema_statements(NebulaClient.space())[1:]:
                NebulaClient.execute(session, statement)

    @classmethod
    def _mark_snapshot_sync(
        cls,
        folder_id: str,
        graph_id: str,
        snapshot_version: str,
        ready: bool,
    ) -> None:
        columns = ", ".join(f"`{name}`" for name in SNAPSHOT_PROPERTIES)
        values = ", ".join(f"${name}" for name in SNAPSHOT_PROPERTIES)
        parameters = {
            "folder_id": folder_id,
            "graph_id": graph_id,
            "snapshot_version": snapshot_version,
            "ready": ready,
        }
        with NebulaClient.session() as session:
            NebulaClient.execute(
                session,
                f"INSERT VERTEX `{SNAPSHOT_TAG}` ({columns}) VALUES "
                f"{_vid_literal(_snapshot_vid(folder_id))}:({values})",
                parameters,
            )

    @classmethod
    def _get_snapshot_sync(cls, folder_id: str) -> KGSnapshot | None:
        with NebulaClient.query_session() as session:
            rows = NebulaClient.execute(
                session,
                f"FETCH PROP ON `{SNAPSHOT_TAG}` "
                f"{_vid_literal(_snapshot_vid(folder_id))} "
                "YIELD properties(vertex) AS properties",
            )
        if not rows:
            return None
        properties = rows[0]["properties"]
        if properties["folder_id"] != folder_id:
            raise NebulaGraphError("Snapshot folder mismatch")
        return KGSnapshot(
            folder_id=properties["folder_id"],
            graph_id=properties["graph_id"],
            snapshot_version=properties["snapshot_version"],
            ready=bool(properties["ready"]),
        )

    @classmethod
    def _delete_snapshot_sync(cls, folder_id: str) -> None:
        with NebulaClient.session() as session:
            NebulaClient.execute(
                session,
                f"DELETE VERTEX {_vid_literal(_snapshot_vid(folder_id))}",
            )

    @classmethod
    def _execute_read_sync(
        cls,
        statement: str,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        with NebulaClient.query_session() as session:
            scoped_parameters = parameters if "$folder_key" in statement else None
            return NebulaClient.execute(session, statement, scoped_parameters)

    @classmethod
    def _replace_sync(
        cls,
        folder_id: str,
        vertices: list[_Vertex],
        edges: list[_Edge],
    ) -> None:
        with NebulaClient.session() as session:
            cls._delete_sync(folder_id, session=session)
            cls._insert_vertices(session, vertices)
            cls._insert_edges(session, edges)
        logger.info(
            "NebulaGraphRepository | folder快照已替换 | "
            "folder_id={} entities={} relations={}",
            folder_id,
            len(vertices),
            len(edges),
        )

    @classmethod
    def _delete_sync(cls, folder_id: str, *, session: Any | None = None) -> None:
        if session is None:
            with NebulaClient.session() as own_session:
                cls._delete_sync(folder_id, session=own_session)
            return

        vids = cls._find_vids(session, folder_id)
        for batch in _batches(vids):
            NebulaClient.execute(
                session,
                "DELETE VERTEX "
                + ", ".join(_vid_literal(vid) for vid in batch)
                + " WITH EDGE",
            )

    @classmethod
    def _find_vids(cls, session: Any, folder_id: str) -> list[str]:
        key = folder_key(folder_id)
        vids: set[str] = set()
        for tag in ENTITY_TYPE_BY_TAG:
            rows = NebulaClient.execute(
                session,
                f"LOOKUP ON `{tag}` WHERE `{tag}`.`folder_key` == $folder_key "
                "YIELD id(vertex) AS vid",
                {"folder_key": key},
            )
            vids.update(str(row["vid"]) for row in rows)
        return sorted(vids)

    @classmethod
    def _insert_vertices(cls, session: Any, vertices: list[_Vertex]) -> None:
        by_tag: dict[str, list[_Vertex]] = defaultdict(list)
        for vertex in vertices:
            by_tag[vertex.tag].append(vertex)

        columns = ", ".join(f"`{name}`" for name in VERTEX_PROPERTIES)
        for tag, tagged_vertices in by_tag.items():
            for batch in _batches(tagged_vertices):
                parameters: dict[str, str] = {}
                values: list[str] = []
                for index, vertex in enumerate(batch):
                    property_parameters = []
                    for property_name in VERTEX_PROPERTIES:
                        parameter_name = f"{property_name}_{index}"
                        parameters[parameter_name] = getattr(
                            vertex, property_name
                        )
                        property_parameters.append(f"${parameter_name}")
                    values.append(
                        f"{_vid_literal(vertex.vid)}:"
                        f"({', '.join(property_parameters)})"
                    )
                NebulaClient.execute(
                    session,
                    f"INSERT VERTEX `{tag}` ({columns}) VALUES "
                    + ", ".join(values),
                    parameters,
                )

    @classmethod
    def _insert_edges(cls, session: Any, edges: list[_Edge]) -> None:
        by_type: dict[str, list[_Edge]] = defaultdict(list)
        for edge in edges:
            by_type[edge.edge_type].append(edge)

        columns = ", ".join(f"`{name}`" for name in EDGE_PROPERTIES)
        for edge_type, typed_edges in by_type.items():
            for batch in _batches(typed_edges):
                parameters: dict[str, str] = {}
                values: list[str] = []
                for index, edge in enumerate(batch):
                    parameters[f"attributes_json_{index}"] = (
                        edge.attributes_json
                    )
                    parameters[f"references_json_{index}"] = (
                        edge.references_json
                    )
                    values.append(
                        f"{_vid_literal(edge.source)}"
                        f"->{_vid_literal(edge.target)}@0:"
                        f"($attributes_json_{index}, $references_json_{index})"
                    )
                NebulaClient.execute(
                    session,
                    f"INSERT EDGE `{edge_type}` ({columns}) VALUES "
                    + ", ".join(values),
                    parameters,
                )

    @classmethod
    def _fetch_entities_sync(
        cls,
        folder_id: str,
        vids: Sequence[str],
    ) -> dict[str, Entity]:
        with NebulaClient.query_session() as session:
            return cls._fetch_entities_with_session(session, folder_id, vids)

    @classmethod
    def _fetch_entities_with_session(
        cls,
        session: Any,
        folder_id: str,
        vids: Sequence[str],
    ) -> dict[str, Entity]:
        requested = set(vids)
        entities: dict[str, Entity] = {}
        vid_list = ", ".join(_vid_literal(vid) for vid in vids)
        for tag, type_name in ENTITY_TYPE_BY_TAG.items():
            rows = NebulaClient.execute(
                session,
                f"FETCH PROP ON `{tag}` {vid_list} "
                "YIELD id(vertex) AS vid, properties(vertex) AS properties",
            )
            for row in rows:
                vid = str(row["vid"])
                properties = row["properties"]
                if not isinstance(properties, dict) or "folder_id" not in properties:
                    continue
                if vid not in requested or properties["folder_id"] != folder_id:
                    raise NebulaGraphError(
                        f"Hydrated vertex is outside the requested folder: {vid!r}"
                    )
                if vid in entities:
                    raise NebulaGraphError(
                        f"Vertex has multiple semantic Tags: {vid!r}"
                    )
                attributes = json_loads(properties["attributes_json"])
                references = json_loads(properties["references_json"])
                if not isinstance(attributes, dict) or not isinstance(
                    references, list
                ):
                    raise NebulaGraphError(
                        f"Invalid vertex JSON properties: {vid!r}"
                    )
                entities[vid] = Entity(
                    id=vid,
                    name=properties["name"],
                    type=type_name,
                    type_en=ENTITY_TYPE_EN_BY_TYPE[type_name],
                    attributes=attributes,
                    references=references,
                )
        missing = requested - entities.keys()
        if missing:
            raise NebulaGraphError(
                f"Hydration could not resolve VIDs in current folder: {sorted(missing)!r}"
            )
        return entities

    @classmethod
    def _fetch_relations_sync(
        cls,
        folder_id: str,
        edge_keys: Sequence[EdgeKey],
    ) -> list[Relation]:
        for source, target, edge_type, rank in edge_keys:
            if (
                not _VID_PATTERN.fullmatch(source)
                or not _VID_PATTERN.fullmatch(target)
                or edge_type not in RELATION_TYPE_BY_EDGE
                or rank < 0
            ):
                raise NebulaGraphError("Invalid edge key in hydration request")

        with NebulaClient.query_session() as session:
            entities = cls._fetch_entities_with_session(
                session,
                folder_id,
                sorted(
                    {
                        endpoint
                        for source, target, _, _ in edge_keys
                        for endpoint in (source, target)
                    }
                ),
            )
            requested = set(edge_keys)
            relations: list[Relation] = []
            found: set[EdgeKey] = set()
            keys_by_type: dict[str, list[EdgeKey]] = defaultdict(list)
            for key in edge_keys:
                keys_by_type[key[2]].append(key)

            for edge_type, typed_keys in keys_by_type.items():
                literals = ", ".join(
                    f"{_vid_literal(source)}->{_vid_literal(target)}@{rank}"
                    for source, target, _, rank in typed_keys
                )
                rows = NebulaClient.execute(
                    session,
                    f"FETCH PROP ON `{edge_type}` {literals} "
                    "YIELD src(edge) AS source, dst(edge) AS target, "
                    "type(edge) AS edge_type, rank(edge) AS rank, "
                    "properties(edge) AS properties",
                )
                for row in rows:
                    source_id = str(row["source"])
                    target_id = str(row["target"])
                    actual_type = str(row["edge_type"])
                    rank = int(row["rank"])
                    key = (source_id, target_id, actual_type, rank)
                    if key not in requested:
                        raise NebulaGraphError(
                            f"Hydrated edge was not requested: {key!r}"
                        )
                    found.add(key)
                    properties = row["properties"]
                    attributes = json_loads(properties["attributes_json"])
                    references = json_loads(properties["references_json"])
                    if not isinstance(attributes, dict) or not isinstance(
                        references, list
                    ):
                        raise NebulaGraphError(
                            f"Invalid edge JSON properties: {key!r}"
                        )

                    type_name = RELATION_TYPE_BY_EDGE[actual_type]
                    id_source, id_target = source_id, target_id
                    if type_name in BIDIRECTIONAL_RELATION_TYPES:
                        id_source, id_target = sorted((source_id, target_id))
                    relations.append(
                        Relation(
                            id=relation_id(id_source, id_target, type_name),
                            type=type_name,
                            type_en=RELATION_TYPE_EN_BY_TYPE[type_name],
                            from_entity=entities[source_id].name,
                            from_entity_id=source_id,
                            to_entity=entities[target_id].name,
                            to_entity_id=target_id,
                            attributes=attributes,
                            references=references,
                        )
                    )
        missing = requested - found
        if missing:
            raise NebulaGraphError(
                f"Hydration could not resolve edge keys: {sorted(missing)!r}"
            )
        return relations

    @classmethod
    def _get_sync(cls, folder_id: str) -> GraphData | None:
        with NebulaClient.session() as session:
            entities = cls._read_entities(session, folder_id)
            if not entities:
                return None
            relations = cls._read_relations(session, entities)
        return GraphData(
            totalEntities=len(entities),
            totalRelations=len(relations),
            entities=sorted(entities.values(), key=lambda entity: entity.id),
            relations=sorted(relations, key=lambda relation: relation.id),
        )

    @classmethod
    def _read_entities(
        cls, session: Any, folder_id: str
    ) -> dict[str, Entity]:
        key = folder_key(folder_id)
        entities: dict[str, Entity] = {}
        for tag, type_name in ENTITY_TYPE_BY_TAG.items():
            rows = NebulaClient.execute(
                session,
                f"LOOKUP ON `{tag}` WHERE `{tag}`.`folder_key` == $folder_key "
                f"YIELD id(vertex) AS vid, `{tag}`.`folder_id` AS folder_id, "
                f"`{tag}`.`name` AS name, "
                f"`{tag}`.`attributes_json` AS attributes_json, "
                f"`{tag}`.`references_json` AS references_json",
                {"folder_key": key},
            )
            for row in rows:
                vid = str(row["vid"])
                if vid in entities:
                    raise NebulaGraphError(
                        f"Vertex has multiple semantic Tags: {vid!r}"
                    )
                if row["folder_id"] != folder_id:
                    raise NebulaGraphError(f"Vertex folder mismatch: {vid!r}")
                attributes = json_loads(row["attributes_json"])
                references = json_loads(row["references_json"])
                if not isinstance(attributes, dict) or not isinstance(
                    references, list
                ):
                    raise NebulaGraphError(
                        f"Invalid vertex JSON properties: {vid!r}"
                    )
                entities[vid] = Entity(
                    id=vid,
                    name=row["name"],
                    type=type_name,
                    type_en=ENTITY_TYPE_EN_BY_TYPE[type_name],
                    attributes=attributes,
                    references=references,
                )
        return entities

    @classmethod
    def _read_relations(
        cls, session: Any, entities: dict[str, Entity]
    ) -> list[Relation]:
        relations: list[Relation] = []
        vids = sorted(entities)
        for batch in _batches(vids):
            rows = NebulaClient.execute(
                session,
                "GO FROM "
                + ", ".join(_vid_literal(vid) for vid in batch)
                + " OVER * "
                "YIELD src(edge) AS source, dst(edge) AS target, "
                "type(edge) AS edge_type, properties(edge) AS properties",
            )
            for row in rows:
                source_id = str(row["source"])
                target_id = str(row["target"])
                edge_type = row["edge_type"]
                type_name = RELATION_TYPE_BY_EDGE.get(edge_type)
                source = entities.get(source_id)
                target = entities.get(target_id)
                if type_name is None or source is None or target is None:
                    raise NebulaGraphError(
                        "NebulaGraph edge has unknown type or cross-folder endpoint"
                    )

                properties = row["properties"]
                attributes = json_loads(properties["attributes_json"])
                references = json_loads(properties["references_json"])
                if not isinstance(attributes, dict) or not isinstance(
                    references, list
                ):
                    raise NebulaGraphError(
                        f"Invalid edge JSON properties: {edge_type!r}"
                    )

                id_source, id_target = source_id, target_id
                if type_name in BIDIRECTIONAL_RELATION_TYPES:
                    id_source, id_target = sorted((id_source, id_target))
                relations.append(
                    Relation(
                        id=relation_id(id_source, id_target, type_name),
                        type=type_name,
                        type_en=RELATION_TYPE_EN_BY_TYPE[type_name],
                        from_entity=source.name,
                        from_entity_id=source_id,
                        to_entity=target.name,
                        to_entity_id=target_id,
                        attributes=attributes,
                        references=references,
                    )
                )
        return relations
