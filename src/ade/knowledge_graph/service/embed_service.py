"""知识图谱实体向量化服务

将实体（名称 + 类型 + 属性）拼成文本，调用外部 Embedding API 生成向量。

依赖环境变量（来自 .env）：
    EMBEDDING_API_BASE    Embedding 服务地址
    EMBEDDING_API_KEY     API Key
    EMBEDDING_MODEL       模型名
"""
import os
import hashlib
from typing import List
from loguru import logger
from openai import AsyncOpenAI

from src.ade.knowledge_graph.schema.kg_schema import Entity, Relation
from src.base.cache_manager import CacheManager


class EmbedService:
    """实体和关系向量化服务"""

    EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY")
    EMBEDDING_API_BASE = os.environ.get("EMBEDDING_API_BASE")
    MODEL = os.environ.get("EMBEDDING_MODEL")

    @classmethod
    def entity_to_text(cls, entity: Entity) -> str:
        """将实体拼接为用于向量化的文本"""
        parts = [entity.name, entity.type]
        if entity.attributes:
            attr_str = " ".join(f"{k}={v}" for k, v in entity.attributes.items())
            parts.append(attr_str)
        return " ".join(parts)

    @classmethod
    def relation_to_text(cls, relation: Relation) -> str:
        """将关系拼接为用于向量化的文本：type\tto_entity\n,from_entity"""
        parts = [f"{relation.type}\t{relation.to_entity}\n,{relation.from_entity}"]
        return "".join(parts)

    @classmethod
    def _get_cache_key(cls, text: str, dimension: int | None = None) -> str:
        """生成文本的缓存 Key"""
        # 使用 MD5 哈希文本，避免 Key 过长
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"kg:embed:{cls.MODEL}:{dimension or 'native'}:{text_hash}"
    
    @classmethod
    async def embed_texts(cls, texts: List[str]) -> List[List[float]]:
        """调用 Embedding API 对一批文本生成向量，支持缓存"""
        if not texts:
            return []
        dimension = int(os.environ.get("EMBEDDING_DIM", "0")) or None
        
        result_vectors = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        # 1. 先尝试从缓存获取
        cache_hits = 0
        for idx, text in enumerate(texts):
            cache_key = cls._get_cache_key(text, dimension)
            cached_vector = await CacheManager.get(cache_key)
            if cached_vector is not None:
                result_vectors[idx] = cached_vector
                cache_hits += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(idx)
        
        # 2. 对缓存未命中的调用 API
        if texts_to_embed:
            client = None

            client = AsyncOpenAI(
                api_key=cls.EMBEDDING_API_KEY,
                base_url=cls.EMBEDDING_API_BASE,
            )
            resp = await client.embeddings.create(
                model=cls.MODEL,
                input=texts_to_embed,
                dimensions=dimension,
            )
            new_vectors = [d.embedding for d in resp.data]
            
            # 3. 回填结果和批量缓存
            cache_items = {}
            for idx, text, vector in zip(indices_to_embed, texts_to_embed, new_vectors):
                result_vectors[idx] = vector
                cache_key = cls._get_cache_key(text, dimension)
                cache_items[cache_key] = vector
            
            # 批量缓存 24 小时
            await CacheManager.set_many(cache_items, ttl=86400)

        
        logger.info(
            "EmbedService | 向量化完成 | total={}, cached={}, model={}",
            len(texts), cache_hits, cls.MODEL,
        )
        
        return result_vectors
