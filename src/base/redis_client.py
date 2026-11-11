"""
RedisClient — 统一Redis客户端管理

全局单例，统一管理Redis连接池，避免多个模块各自创建连接池导致资源浪费。
支持 decode_responses 参数切换返回类型（bytes/str），满足不同场景需求。

与现有 DistributedLock、TaskManager 的兼容：
- DistributedLock 需要 decode_responses=False（锁token比较需要bytes）
- TaskManager 需要 decode_responses=True（JSON解析需要str）
- 本模块根据参数自动返回对应配置的客户端

用法:
    from src.base.redis_client import RedisClient
    
    # 获取默认客户端（decode_responses=True，返回str）
    redis = await RedisClient.get_client()
    await redis.set("key", "value")
    
    # 获取bytes客户端（用于分布式锁等场景）
    redis_bytes = await RedisClient.get_client(decode_responses=False)
    await redis_bytes.set("binary_key", b"binary_value")
"""

import asyncio
from typing import Optional, Dict
from urllib.parse import quote

from loguru import logger
import redis.asyncio as aioredis
from redis.asyncio.sentinel import Sentinel

from src.config import config


class RedisClient:
    """
    统一Redis客户端管理器，按事件循环隔离的单例模式。

    支持根据 decode_responses 参数维护不同的连接池。
    关键设计：每个事件循环拥有独立的客户端实例，避免 "Future attached to a different loop" 错误。
    适用于 LangGraph、多线程等会创建新事件循环的场景。
    """

    # key: (loop_id, decode_responses), value: Redis 客户端
    _clients: Dict[tuple[int, bool], aioredis.Redis] = {}
    _lock: Optional[asyncio.Lock] = None  # 惰性初始化，避免 Python 3.12+ 无事件循环时创建报错

    @classmethod
    def is_sentinel_mode(cls) -> bool:
        return config.REDIS_MODE.strip().lower() == "sentinel"

    @classmethod
    def parse_sentinel_hosts(cls) -> list[tuple[str, int]]:
        sentinels: list[tuple[str, int]] = []
        for item in config.REDIS_SENTINELS.split(","):
            item = item.strip()
            if not item:
                continue
            host, sep, port = item.rpartition(":")
            if not sep or not host or not port:
                raise ValueError("REDIS_SENTINELS 格式错误，应为 host:port,host:port")
            sentinels.append((host, int(port)))
        if not sentinels:
            raise ValueError("REDIS_SENTINELS 环境变量未配置，请设置 Sentinel 节点列表")
        return sentinels

    @classmethod
    def sentinel_kwargs(cls) -> dict:
        kwargs = {
            "socket_connect_timeout": 5,
            "socket_timeout": 30,
        }
        if config.REDIS_SENTINEL_USERNAME:
            kwargs["username"] = config.REDIS_SENTINEL_USERNAME
        if config.REDIS_SENTINEL_PASSWORD:
            kwargs["password"] = config.REDIS_SENTINEL_PASSWORD
        return kwargs

    @classmethod
    async def get_client(cls, decode_responses: bool = True) -> aioredis.Redis:
        """
        获取Redis客户端实例，懒加载初始化。
        每个事件循环会有独立的客户端实例，自动处理跨 loop 场景。

        Args:
            decode_responses: 是否自动将Redis返回的bytes解码为str
                            True: 返回str类型，适合JSON序列化、普通缓存场景
                            False: 返回bytes类型，适合分布式锁等需要二进制比较的场景

        Returns:
            aioredis.Redis: Redis异步客户端实例
        """
        loop_id = id(asyncio.get_event_loop())
        cache_key = (loop_id, decode_responses)

        if cache_key not in cls._clients:
            if cls._lock is None:
                cls._lock = asyncio.Lock()
            async with cls._lock:
                # 双重检查，防止并发初始化
                if cache_key not in cls._clients:
                    await cls._init_client(cache_key, decode_responses)

        return cls._clients[cache_key]
    
    @classmethod
    async def _init_client(cls, cache_key: tuple[int, bool], decode_responses: bool) -> None:
        """初始化Redis客户端连接池，并验证连通性"""
        loop_id, _ = cache_key
        if cls.is_sentinel_mode():
            client = cls._build_sentinel_client(decode_responses)
            logger.info(
                "初始化Redis Sentinel客户端 | loop_id={} decode_responses={} sentinels={} master={} db={}",
                loop_id,
                decode_responses,
                config.REDIS_SENTINELS,
                config.REDIS_SENTINEL_MASTER_NAME,
                config.REDIS_DB,
            )
        else:
            client = cls._build_standalone_client(decode_responses)
            logger.info(
                "初始化Redis客户端 | loop_id={} decode_responses={} host={}:{} db={}",
                loop_id, decode_responses, config.REDIS_HOST, config.REDIS_PORT, config.REDIS_DB
            )

        await client.ping()
        logger.info("Redis连接验证成功 | loop_id={} decode_responses={}", loop_id, decode_responses)

        cls._clients[cache_key] = client

    @classmethod
    def _build_standalone_client(cls, decode_responses: bool) -> aioredis.Redis:
        pw_part = f":{quote(config.REDIS_PASSWORD, safe='')}@" if config.REDIS_PASSWORD else ""
        url = f"redis://{pw_part}{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}"
        return aioredis.from_url(
            url,
            decode_responses=decode_responses,
            encoding="utf-8",
            socket_connect_timeout=5,
            socket_timeout=30,
            retry_on_timeout=True,
            max_connections=100,
        )

    @classmethod
    def _build_sentinel_client(cls, decode_responses: bool) -> aioredis.Redis:
        sentinel = Sentinel(
            cls.parse_sentinel_hosts(),
            sentinel_kwargs=cls.sentinel_kwargs(),
            db=config.REDIS_DB,
            password=config.REDIS_PASSWORD or None,
            decode_responses=decode_responses,
            encoding="utf-8",
            socket_connect_timeout=5,
            socket_timeout=30,
            retry_on_timeout=True,
            max_connections=100,
        )
        return sentinel.master_for(config.REDIS_SENTINEL_MASTER_NAME)

    @classmethod
    async def close(cls) -> None:
        """关闭当前事件循环下的所有Redis客户端连接，优雅退出"""
        current_loop_id = id(asyncio.get_event_loop())
        keys_to_close = [
            key for key in cls._clients
            if key[0] == current_loop_id
        ]
        for cache_key in keys_to_close:
            client = cls._clients.pop(cache_key)
            loop_id, decode_responses = cache_key
            try:
                await client.aclose()
                logger.info(
                    "Redis客户端已关闭 | loop_id={} decode_responses={}",
                    loop_id, decode_responses
                )
            except Exception:
                logger.exception(
                    "关闭Redis客户端失败 | loop_id={} decode_responses={}",
                    loop_id, decode_responses
                )
