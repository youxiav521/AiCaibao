"""
CacheManager — 分布式缓存管理器

基于Redis实现，支持Pydantic模型自动序列化/反序列化，统一缓存key命名规范，
简化缓存操作，避免业务代码重复编写序列化逻辑。

特性:
  - 自动处理Pydantic模型的序列化/反序列化
  - 支持任意JSON可序列化类型
  - 统一缓存key前缀，避免命名冲突
  - 支持空值缓存（防止缓存穿透）
  - 内置缓存穿透防护

用法:
    from src.base.cache_manager import CacheManager
    from pydantic import BaseModel
    
    class User(BaseModel):
        id: int
        name: str
    
    # 缓存Pydantic模型
    user = User(id=1, name="test")
    await CacheManager.set("user:1", user, ttl=3600)
    
    # 获取并自动反序列化为模型
    cached_user = await CacheManager.get("user:1", model_type=User)
    
    # 缓存普通类型
    await CacheManager.set("config:site_name", "My Site", ttl=86400)
    site_name = await CacheManager.get("config:site_name")
    
    # 删除缓存
    await CacheManager.delete("user:1")
    
    # 检查缓存是否存在
    exists = await CacheManager.exists("user:1")
"""

import json
from typing import Any, Optional, Type
from loguru import logger
from pydantic import BaseModel, ValidationError

from src.base.redis_client import RedisClient
from src.utils.json_utils import dumps_bytes


class CacheManager:
    """分布式缓存管理器，基于Redis实现"""
    
    _KEY_PREFIX = "cache:"  # 统一缓存key前缀
    _NULL_VALUE = "__NULL__"  # 空值标记，用于防止缓存穿透
    
    @classmethod
    async def get(
        cls,
        key: str,
        model_type: Optional[Type[BaseModel]] = None,
        default: Any = None
    ) -> Any:
        """
        获取缓存内容。
        
        Args:
            key: 缓存键（自动添加cache:前缀）
            model_type: Pydantic模型类型，若指定则自动反序列化为该模型实例
            default: 缓存不存在时返回的默认值
        
        Returns:
            缓存内容，若不存在则返回default
        """
        redis = await RedisClient.get_client()
        full_key = f"{cls._KEY_PREFIX}{key}"
        
        try:
            data = await redis.get(full_key)
            if data is None:
                return default
            
            # 处理空值标记
            if data == cls._NULL_VALUE:
                return None
            
            # 如果指定了模型类型，反序列化为模型
            if model_type is not None:
                try:
                    return model_type.model_validate_json(data)
                except ValidationError as e:
                    logger.error("缓存反序列化为模型失败 | key={} model={} error={}", key, model_type.__name__, e)
                    return default
            
            # 普通JSON反序列化
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                logger.error("缓存JSON解析失败 | key={} error={}", key, e)
                return default
                
        except Exception as e:
            logger.exception("获取缓存失败 | key={}", key)
            return default
    
    @classmethod
    async def set(
        cls,
        key: str,
        value: Any,
        ttl: int = 3600,
        cache_null: bool = False
    ) -> None:
        """
        设置缓存内容。
        
        Args:
            key: 缓存键（自动添加cache:前缀）
            value: 要缓存的值，支持Pydantic模型和任意JSON可序列化类型
            ttl: 缓存过期时间（秒），默认1小时
            cache_null: 是否缓存None值，用于防止缓存穿透
        """
        redis = await RedisClient.get_client()
        full_key = f"{cls._KEY_PREFIX}{key}"
        
        try:
            # 处理None值
            if value is None:
                if cache_null:
                    await redis.setex(full_key, ttl, cls._NULL_VALUE)
                    logger.debug("已缓存空值 | key={} ttl={}s", key, ttl)
                return
            
            # 序列化处理
            if isinstance(value, BaseModel):
                data = value.model_dump_json()
            else:
                data = json.dumps(value, ensure_ascii=False, default=str)
            
            await redis.setex(full_key, ttl, data)
            logger.debug("已设置缓存 | key={} ttl={}s", key, ttl)
            
        except Exception as e:
            logger.exception("设置缓存失败 | key={}", key)
    
    @classmethod
    async def delete(cls, key: str) -> None:
        """
        删除缓存。
        
        Args:
            key: 缓存键（自动添加cache:前缀）
        """
        redis = await RedisClient.get_client()
        full_key = f"{cls._KEY_PREFIX}{key}"
        
        try:
            await redis.delete(full_key)
            logger.debug("已删除缓存 | key={}", key)
        except Exception as e:
            logger.exception("删除缓存失败 | key={}", key)
    
    @classmethod
    async def exists(cls, key: str) -> bool:
        """
        检查缓存是否存在。
        
        Args:
            key: 缓存键（自动添加cache:前缀）
        
        Returns:
            True: 缓存存在，False: 缓存不存在
        """
        redis = await RedisClient.get_client()
        full_key = f"{cls._KEY_PREFIX}{key}"
        
        try:
            exists = await redis.exists(full_key)
            return exists > 0
        except Exception as e:
            logger.exception("检查缓存存在性失败 | key={}", key)
            return False
    
    @classmethod
    async def get_ttl(cls, key: str) -> int:
        """
        获取缓存剩余过期时间。
        
        Args:
            key: 缓存键（自动添加cache:前缀）
        
        Returns:
            剩余过期时间（秒），-1表示永不过期，-2表示键不存在
        """
        redis = await RedisClient.get_client()
        full_key = f"{cls._KEY_PREFIX}{key}"
        
        try:
            ttl = await redis.ttl(full_key)
            return ttl
        except Exception as e:
            logger.exception("获取缓存TTL失败 | key={}", key)
            return -2
    
    @classmethod
    async def incr(cls, key: str, amount: int = 1, ttl: int = 3600) -> int:
        """
        原子递增计数器。
        
        Args:
            key: 缓存键（自动添加cache:前缀）
            amount: 递增步长
            ttl: 初始过期时间（仅当key不存在时生效）
        
        Returns:
            递增后的值
        """
        redis = await RedisClient.get_client()
        full_key = f"{cls._KEY_PREFIX}{key}"
        
        try:
            value = await redis.incrby(full_key, amount)
            # 如果是新创建的key，设置过期时间
            if value == amount:
                await redis.expire(full_key, ttl)
            return value
        except Exception as e:
            logger.exception("计数器递增失败 | key={} amount={}", key, amount)
            return 0
            
    @classmethod
    async def set_many(
        cls,
        items: dict[str, Any],
        ttl: int = 3600,
        cache_null: bool = False
    ) -> None:
        """
        批量设置缓存内容。
        
        Args:
            items: 键值对字典，key为缓存键（自动添加cache:前缀），value为要缓存的值
            ttl: 缓存过期时间（秒），默认1小时
            cache_null: 是否缓存None值，用于防止缓存穿透
        """
        if not items:
            return
        
        redis = await RedisClient.get_client()
        pipe = redis.pipeline()
        
        try:
            for key, value in items.items():
                full_key = f"{cls._KEY_PREFIX}{key}"
                
                # 处理None值
                if value is None:
                    if cache_null:
                        pipe.setex(full_key, ttl, cls._NULL_VALUE)
                    continue
                
                # 序列化处理
                if isinstance(value, BaseModel):
                    data = value.model_dump_json().encode("utf-8")
                else:
                    data = dumps_bytes(value, default=str)
                
                pipe.setex(full_key, ttl, data)
            
            await pipe.execute()
            logger.debug("已批量设置缓存 | count={} ttl={}s", len(items), ttl)
            
        except Exception as e:
            logger.exception("批量设置缓存失败")

