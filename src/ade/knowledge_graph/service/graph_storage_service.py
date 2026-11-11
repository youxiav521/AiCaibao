import uuid

from src.base.distributed_lock import DistributedLock
from src.utils.json_utils import dumps_bytes as json_dumps_bytes, loads as json_loads
from src.base.storage_service import StorageObjectNotFoundError, storage_service
from src.ade.knowledge_graph.nebula.client import NebulaClient
from src.ade.knowledge_graph.nebula.repository import NebulaGraphRepository
from src.ade.knowledge_graph.schema.kg_schema import KnowledgeGraph, Entity, Relation
from src.ade.knowledge_graph.service.entity_vector_store import EntityVectorStore
from loguru import logger
from typing import Optional

_KG_BUCKET = "ddis"


class GraphStorageService:
    """知识图谱存储服务，基于MinIO"""

    @classmethod
    def _object_name(cls, folder_id: str) -> str:
        return f"survey/{folder_id}/knowledge_graph.json"

    @classmethod
    async def save(cls, graph: KnowledgeGraph) -> str:
        async with DistributedLock.acquire(f"kg:{graph.folderId}"):
            return await cls._save_locked(graph)

    @classmethod
    async def _save_locked(cls, graph: KnowledgeGraph) -> str:
        graph.snapshot_version = uuid.uuid4().hex
        # MinIO 是权威快照，必须先落地；Nebula 只是由它派生的查询副本
        # 持久化时排除向量字段，向量由 EntityVectorStore 承载
        data = json_dumps_bytes(
            graph.model_dump(
                by_alias=True,
                exclude={
                    "entities": {"__all__": {"embed"}},
                    "relations": {"__all__": {"embed"}},
                },
            )
        )
        await storage_service.upload(
            _KG_BUCKET,
            cls._object_name(graph.folderId),
            data,
            "application/json",
        )
        await EntityVectorStore.rebuild(graph)
        if NebulaClient.is_configured():
            try:
                await NebulaGraphRepository.mark_snapshot(
                    graph.folderId,
                    graph.id,
                    graph.snapshot_version,
                    ready=False,
                )
                await NebulaGraphRepository.replace(graph, acquire_lock=False)
                await NebulaGraphRepository.mark_snapshot(
                    graph.folderId,
                    graph.id,
                    graph.snapshot_version,
                    ready=True,
                )
            except Exception:
                # 副本落后于权威快照，查询侧靠 snapshot_version 比对识别并拒绝服务
                logger.exception(
                    "GraphStorageService | Nebula 查询副本重建失败 | folder_id={}",
                    graph.folderId,
                )
        logger.info(
            "GraphStorageService | 图谱已保存 | folder_id={} entities={} relations={} snapshot_version={}",
            graph.folderId,
            len(graph.entities),
            len(graph.relations),
            graph.snapshot_version,
        )
        return graph.id

    @classmethod
    async def get(cls, folder_id: str) -> Optional[KnowledgeGraph]:
        try:
            data = await storage_service.download(_KG_BUCKET, cls._object_name(folder_id))
        except StorageObjectNotFoundError:
            return None
        return KnowledgeGraph.model_validate(json_loads(data))

    @classmethod
    async def delete(cls, folder_id: str) -> bool:
        async with DistributedLock.acquire(f"kg:{folder_id}"):
            graph = await cls.get(folder_id)
            exists = graph is not None
            if NebulaClient.is_configured():
                await NebulaGraphRepository.mark_snapshot(
                    folder_id,
                    graph.id if graph else "",
                    graph.snapshot_version if graph else "",
                    ready=False,
                )
                await NebulaGraphRepository.delete(folder_id, acquire_lock=False)
            if exists:
                await storage_service.delete(_KG_BUCKET, cls._object_name(folder_id))
            await EntityVectorStore.drop(folder_id)
            if NebulaClient.is_configured():
                await NebulaGraphRepository.delete_snapshot(folder_id)
            if not exists:
                return False
            logger.info("GraphStorageService | 图谱已删除 | folder_id={}", folder_id)
            return True

    @classmethod
    async def exists(cls, folder_id: str) -> bool:
        return await storage_service.exists(_KG_BUCKET, cls._object_name(folder_id))

    @classmethod
    async def update_entity(cls, folder_id: str, entity_id: str, update_data: dict) -> Optional[Entity]:
        async with DistributedLock.acquire(f"kg:{folder_id}"):
            graph = await cls.get(folder_id)
            if graph is None:
                return None

            for i, entity in enumerate(graph.entities):
                if entity.id == entity_id:
                    for field, value in update_data.items():
                        if value is not None and hasattr(entity, field):
                            setattr(entity, field, value)
                    graph.entities[i] = entity
                    await cls._save_locked(graph)
                    logger.info("GraphStorageService | 实体已更新 | folder_id={} entity_id={}", folder_id, entity_id)
                    return entity

            return None

    @classmethod
    async def update_relation(cls, folder_id: str, relation_id: str, update_data: dict) -> Optional[Relation]:
        async with DistributedLock.acquire(f"kg:{folder_id}"):
            graph = await cls.get(folder_id)
            if graph is None:
                return None

            for i, relation in enumerate(graph.relations):
                if relation.id == relation_id:
                    for field, value in update_data.items():
                        if value is not None and hasattr(relation, field):
                            setattr(relation, field, value)
                    graph.relations[i] = relation
                    await cls._save_locked(graph)
                    logger.info("GraphStorageService | 关系已更新 | folder_id={} relation_id={}", folder_id, relation_id)
                    return relation

            return None
