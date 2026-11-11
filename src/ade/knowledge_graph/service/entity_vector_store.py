"""Folder-scoped entity vector index backed by MinIO and cached in Redis."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from src.ade.knowledge_graph.schema.kg_schema import Entity, KnowledgeGraph
from src.ade.knowledge_graph.service.embed_service import EmbedService
from src.base.distributed_lock import DistributedLock
from src.base.redis_client import RedisClient
from src.base.storage_service import (
    StorageObjectNotFoundError,
    storage_service,
)
from src.utils.json_utils import dumps_bytes as json_dumps_bytes, loads as json_loads

_KG_BUCKET = "ddis"
_DTYPE = "float32"
_INDEX_VERSION = 2
_ALIAS_ATTRIBUTES = ("简称", "曾用名")
# 企业名去掉称呼前缀后只剩下这些法定后缀的，才是该称呼对应的主体公司：
# “比亚迪”+“股份有限公司”是，“比亚迪”+“慈善基金会”不是。
_LEGAL_SUFFIXES = frozenset(
    {
        "公司",
        "集团",
        "有限公司",
        "股份公司",
        "集团有限公司",
        "股份有限公司",
        "有限责任公司",
        "集团股份有限公司",
    }
)
# 下标即 _match_tier 的层级值。
_MATCH_TYPES = ("exact", "alias", "legal_suffix", "prefix", "substring")


@dataclass(frozen=True)
class AnchorHit:
    vid: str
    name: str
    type: str
    score: float
    match_type: str
    snapshot_version: str


@dataclass(frozen=True)
class _VectorIndex:
    meta: dict[str, Any]
    matrix: np.ndarray


class EntityVectorStore:
    """MinIO-persisted, Redis-cached entity vectors for one folder."""

    @classmethod
    def _meta_key(cls, folder_id: str) -> str:
        return f"kg:vec:{folder_id}:meta"

    @classmethod
    def _matrix_key(cls, folder_id: str) -> str:
        return f"kg:vec:{folder_id}:mat"

    @classmethod
    def _object_name(cls, folder_id: str) -> str:
        return f"survey/{folder_id}/knowledge_graph_vectors.npz"

    @classmethod
    async def rebuild(cls, graph: KnowledgeGraph) -> None:
        snapshot_version = graph.snapshot_version or graph.id
        graph.snapshot_version = snapshot_version
        missing = [
            entity
            for entity in graph.entities
            if not entity.embed
        ]
        if missing:
            batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "128"))
            if batch_size <= 0:
                raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
            texts = [EmbedService.entity_to_text(entity) for entity in missing]
            vectors: list[list[float]] = []
            for offset in range(0, len(texts), batch_size):
                vectors.extend(
                    await EmbedService.embed_texts(texts[offset : offset + batch_size])
                )
            for entity, vector in zip(missing, vectors, strict=True):
                entity.embed = vector

        matrix = cls._normalized_matrix(
            [entity.embed for entity in graph.entities]
        )
        meta = {
            "graph_id": graph.id,
            "index_version": _INDEX_VERSION,
            "snapshot_version": snapshot_version,
            "embedding_model": EmbedService.MODEL or "",
            "dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "row_count": int(matrix.shape[0]),
            "dtype": _DTYPE,
            "items": [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "degree": entity.degree,
                    "aliases": cls._aliases(entity),
                }
                for entity in graph.entities
            ],
        }
        payload = cls._pack(meta, matrix)
        await storage_service.upload(
            _KG_BUCKET,
            cls._object_name(graph.folderId),
            payload,
            "application/octet-stream",
        )
        await cls._cache_best_effort(graph.folderId, meta, matrix)

    @classmethod
    async def search(
        cls,
        folder_id: str,
        mention: str,
        top_k: int,
        entity_types: list[str] | None = None,
    ) -> list[AnchorHit]:
        if top_k <= 0:
            return []
        mention = mention.strip()
        if not mention:
            return []

        index = await cls._load(folder_id)
        allowed_types = set(entity_types) if entity_types else None
        candidates = [
            (row, item)
            for row, item in enumerate(index.meta["items"])
            # 项目节点由 _resolve_anchors 的“本项目”通路绑定，不参与具名实体匹配：
            # 项目名含授信企业名（“礼光信贷尽调项目”），否则会抢掉企业锚点。
            if item["type"] != "项目"
            and (allowed_types is None or item["type"] in allowed_types)
        ]
        mention_folded = mention.casefold()

        matches = [
            (tier, item)
            for _, item in candidates
            if (tier := cls._match_tier(mention_folded, item)) is not None
        ]
        if matches:
            matches.sort(
                key=lambda pair: (
                    pair[0],
                    -int(pair[1].get("degree", 0)),
                    len(pair[1]["name"]),
                    pair[1]["name"],
                )
            )
            return cls._hits(index, matches[:top_k])

        query_vectors = await EmbedService.embed_texts([mention])
        query = np.asarray(query_vectors[0], dtype=np.float32)
        if query.ndim != 1 or query.size != index.meta["dim"]:
            raise ValueError("Mention vector dimension does not match entity index")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query /= norm
        threshold = float(os.environ.get("KG_ANCHOR_MIN_SCORE", "0.5"))
        scored = [
            (float(index.matrix[row] @ query), row, item)
            for row, item in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            AnchorHit(
                vid=item["id"],
                name=item["name"],
                type=item["type"],
                score=score,
                match_type="vector",
                snapshot_version=index.meta["snapshot_version"],
            )
            for score, _, item in scored[:top_k]
            if score >= threshold
        ]

    @classmethod
    async def metadata(cls, folder_id: str) -> dict[str, Any]:
        return dict((await cls._load(folder_id)).meta)

    @classmethod
    async def invalidate_cache(cls, folder_id: str) -> None:
        try:
            redis = await RedisClient.get_client(decode_responses=False)
            await redis.delete(cls._meta_key(folder_id), cls._matrix_key(folder_id))
        except Exception:
            logger.exception(
                "EntityVectorStore | Redis缓存失效失败 | folder_id={}",
                folder_id,
            )

    @classmethod
    async def drop(cls, folder_id: str) -> None:
        await storage_service.delete(
            _KG_BUCKET,
            cls._object_name(folder_id),
        )
        await cls.invalidate_cache(folder_id)

    @classmethod
    def _normalized_matrix(
        cls,
        vectors: list[list[float] | None],
    ) -> np.ndarray:
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        if any(vector is None or len(vector) == 0 for vector in vectors):
            raise ValueError("Every entity must have a non-empty embedding")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Entity embeddings must form a finite 2D matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Entity embeddings must not contain zero vectors")
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)

    @classmethod
    def _pack(cls, meta: dict[str, Any], matrix: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            meta=np.array(json_dumps_bytes(meta).decode("utf-8")),
            matrix=matrix,
        )
        return buffer.getvalue()

    @classmethod
    def _unpack(cls, payload: bytes) -> _VectorIndex:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            meta = json_loads(str(archive["meta"].item()))
            matrix = np.asarray(archive["matrix"], dtype=np.float32)
        return cls._validate_index(meta, matrix)

    @classmethod
    def _validate_index(
        cls,
        meta: dict[str, Any],
        matrix: np.ndarray,
    ) -> _VectorIndex:
        required = {
            "graph_id",
            "snapshot_version",
            "embedding_model",
            "dim",
            "row_count",
            "dtype",
            "items",
        }
        if not isinstance(meta, dict) or not required.issubset(meta):
            raise ValueError("Entity vector metadata is incomplete")
        if meta.get("index_version") != _INDEX_VERSION:
            raise ValueError("Entity vector index format is stale")
        if meta["dtype"] != _DTYPE:
            raise ValueError("Entity vector dtype is unsupported")
        if matrix.ndim != 2 or matrix.shape != (
            meta["row_count"],
            meta["dim"],
        ):
            raise ValueError("Entity vector matrix shape does not match metadata")
        if len(meta["items"]) != meta["row_count"]:
            raise ValueError("Entity vector item count does not match metadata")
        expected_dim = int(os.environ.get("EMBEDDING_DIM", "0"))
        if expected_dim and meta["dim"] != expected_dim:
            raise ValueError("Entity vector dimension does not match configuration")
        if meta["embedding_model"] != (EmbedService.MODEL or ""):
            raise ValueError("Entity vector embedding model is stale")
        return _VectorIndex(meta=meta, matrix=np.ascontiguousarray(matrix))

    @classmethod
    async def _load(cls, folder_id: str) -> _VectorIndex:
        cached = await cls._load_cache(folder_id)
        if cached is not None:
            return cached

        try:
            payload = await storage_service.download(
                _KG_BUCKET,
                cls._object_name(folder_id),
            )
        except StorageObjectNotFoundError:
            return await cls._rebuild_missing(folder_id)

        try:
            index = cls._unpack(payload)
        except ValueError as error:
            if "is stale" not in str(error) and "configuration" not in str(error):
                raise
            return await cls._rebuild_missing(folder_id)
        await cls._cache_best_effort(folder_id, index.meta, index.matrix)
        return index

    @classmethod
    async def _load_cache(cls, folder_id: str) -> _VectorIndex | None:
        try:
            redis = await RedisClient.get_client(decode_responses=False)
            pipeline = redis.pipeline()
            pipeline.get(cls._meta_key(folder_id))
            pipeline.get(cls._matrix_key(folder_id))
            meta_raw, matrix_raw = await pipeline.execute()
            if meta_raw is None or matrix_raw is None:
                return None
            meta = json_loads(meta_raw)
            matrix = np.frombuffer(matrix_raw, dtype=np.float32).reshape(
                meta["row_count"],
                meta["dim"],
            )
            return cls._validate_index(meta, matrix)
        except Exception:
            logger.exception(
                "EntityVectorStore | Redis缓存无效，回源MinIO | folder_id={}",
                folder_id,
            )
            await cls.invalidate_cache(folder_id)
            return None

    @classmethod
    async def _cache_best_effort(
        cls,
        folder_id: str,
        meta: dict[str, Any],
        matrix: np.ndarray,
    ) -> None:
        try:
            redis = await RedisClient.get_client(decode_responses=False)
            pipeline = redis.pipeline()
            pipeline.set(cls._meta_key(folder_id), json_dumps_bytes(meta))
            pipeline.set(cls._matrix_key(folder_id), matrix.tobytes())
            await pipeline.execute()
        except Exception:
            logger.exception(
                "EntityVectorStore | Redis缓存写入失败 | folder_id={}",
                folder_id,
            )

    @classmethod
    async def _rebuild_missing(cls, folder_id: str) -> _VectorIndex:
        async with DistributedLock.acquire(f"kg:vec-rebuild:{folder_id}"):
            try:
                payload = await storage_service.download(
                    _KG_BUCKET,
                    cls._object_name(folder_id),
                )
            except StorageObjectNotFoundError:
                payload = None
            if payload is not None:
                try:
                    return cls._unpack(payload)
                except ValueError:
                    pass

            from src.ade.knowledge_graph.service.graph_storage_service import (
                GraphStorageService,
            )

            graph = await GraphStorageService.get(folder_id)
            if graph is None:
                raise FileNotFoundError(
                    f"Knowledge graph does not exist for folder {folder_id!r}"
                )
            await cls.rebuild(graph)
            return cls._unpack(
                await storage_service.download(
                    _KG_BUCKET,
                    cls._object_name(folder_id),
                )
            )

    @staticmethod
    def _aliases(entity: Entity) -> list[str]:
        """实体的简称、曾用名，作为锚点精确匹配的补充名称。"""
        aliases: list[str] = []
        for attribute in _ALIAS_ATTRIBUTES:
            value = entity.attributes.get(attribute)
            if isinstance(value, str) and value.strip():
                aliases.append(value.strip())
        return aliases

    @staticmethod
    def _match_tier(mention_folded: str, item: dict[str, Any]) -> int | None:
        """称呼与实体的规则匹配层级，越小越可信；None 表示未命中。

        分层是为了消解简称歧义：“比亚迪”应命中“比亚迪股份有限公司”，
        而不是“比亚迪慈善基金会”“上海比亚迪有限公司”这类关联企业。
        """
        name = item["name"].casefold()
        if name == mention_folded:
            return 0
        if any(
            alias.casefold() == mention_folded
            for alias in item.get("aliases", ())
        ):
            return 1
        if name.startswith(mention_folded):
            remainder = name[len(mention_folded) :]
            return 2 if remainder in _LEGAL_SUFFIXES else 3
        if mention_folded in name:
            return 4
        return None

    @classmethod
    def _hits(
        cls,
        index: _VectorIndex,
        matches: list[tuple[int, dict[str, Any]]],
    ) -> list[AnchorHit]:
        return [
            AnchorHit(
                vid=item["id"],
                name=item["name"],
                type=item["type"],
                score=1.0,
                match_type=_MATCH_TYPES[tier],
                snapshot_version=index.meta["snapshot_version"],
            )
            for tier, item in matches
        ]
