from abc import ABC, abstractmethod
import asyncio
import random
import uuid
from typing import Optional, Dict

from limits import RateLimitItemPerSecond, RateLimitItemPerMinute
from limits.aio.storage import RedisStorage as AsyncRedisStorage
from limits.aio.strategies import MovingWindowRateLimiter as AsyncMovingWindowRateLimiter
from loguru import logger
import time

from src.base.redis_client import RedisClient


# ============ 异常定义 ============
class LimitTimeoutError(Exception):
    """资源限制超时异常"""
    pass


# ============ 抽象基类 ============
class ResourceLimiter(ABC):
    """资源限制器抽象基类

    统一的上下文管理器接口，Concurrency 和 Rate Limiter 使用方式完全一致。
    每个请求应创建独立的实例，不应在多个协程间复用同一实例。
    """

    # 默认等待时间 10 分钟，适应 LLM API 高并发排队场景
    DEFAULT_MAX_WAIT = 600.0

    def __init__(self, resource_key: str, max_wait: float = DEFAULT_MAX_WAIT):
        self.resource_key = resource_key
        self.max_wait = max_wait
        self._acquired = False

    @abstractmethod
    async def _try_acquire(self) -> bool:
        """子类实现：尝试获取资源"""
        pass

    @abstractmethod
    async def _release(self) -> None:
        """子类实现：释放资源（可选）"""
        pass

    async def acquire(self) -> bool:
        """获取资源（带指数退避等待重试）"""
        start = time.time()
        attempt = 0
        while True:
            if await self._try_acquire():
                self._acquired = True
                return True

            elapsed = time.time() - start
            if elapsed >= self.max_wait:
                logger.warning("[Limiter] {} 等待资源超时 | elapsed={:.2f}s", self.resource_key, elapsed)
                raise LimitTimeoutError(
                    f"Timeout waiting for resource: {self.resource_key}"
                )

            base = min(0.05 * (2 ** attempt), 2.0)
            jitter = base * (0.5 + random.random())
            await asyncio.sleep(jitter)
            attempt += 1

    async def release(self) -> None:
        """释放资源"""
        if self._acquired:
            try:
                await self._release()
            except Exception:
                logger.exception("[Limiter] {} 释放资源异常", self.resource_key)
            finally:
                self._acquired = False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()


# ============ 并发限制器 (Concurrency) ============

# Lua 脚本：原子地执行「清过期成员 → 检查数量 → 满足条件才写入」
# 避免 pipeline 非原子导致的 TOCTOU 竞态，确保并发数严格不超过 max_concurrent
_CONCURRENCY_ACQUIRE_SCRIPT = """
local key        = KEYS[1]
local now        = tonumber(ARGV[1])
local max_c      = tonumber(ARGV[2])
local holder_id  = ARGV[3]
local expire_at  = tonumber(ARGV[4])
local ttl        = tonumber(ARGV[5])

-- 1. 清除已过期的持有者
redis.call('ZREMRANGEBYSCORE', key, 0, now)

-- 2. 检查当前持有数量
local count = redis.call('ZCARD', key)
if count >= max_c then
    return 0
end

-- 3. 写入本次持有者，并刷新 key TTL 防止内存泄漏
redis.call('ZADD', key, expire_at, holder_id)
redis.call('EXPIRE', key, ttl)
return 1
"""


class ConcurrencyLimiter(ResourceLimiter):
    """基于 Redis ZSET + Lua 脚本的分布式并发限制器

    控制 {resource_key} 同时活跃的最大请求数。
    使用 Lua 脚本保证「检查-写入」原子性，避免超限竞态。
    """

    def __init__(
        self,
        resource_key: str,
        max_concurrent: int,
        lease_timeout: float = 90.0,
        max_wait: float = ResourceLimiter.DEFAULT_MAX_WAIT,
    ):
        super().__init__(f"concurrency:{resource_key}", max_wait)
        self._max_concurrent = max_concurrent
        self._lease_timeout = lease_timeout
        # key TTL = lease_timeout + 一定余量，保证 key 不会永久占用内存
        self._key_ttl = int(lease_timeout) + 30
        self._holder_id = str(uuid.uuid4())

    async def _get_client(self):
        return await RedisClient.get_client()

    async def _try_acquire(self) -> bool:
        if self._max_concurrent <= 0:
            return True

        redis = await self._get_client()
        now = time.time()

        result = await redis.eval(
            _CONCURRENCY_ACQUIRE_SCRIPT,
            1,  # numkeys
            self.resource_key,
            now,
            self._max_concurrent,
            self._holder_id,
            now + self._lease_timeout,
            self._key_ttl,
        )
        return bool(result)

    async def _release(self) -> None:
        redis = await self._get_client()
        await redis.zrem(self.resource_key, self._holder_id)


# ============ 速率限制器 (RPS/RPM) ============
class RateLimiter(ResourceLimiter):
    """基于 limits 库的分布式速率限制器

    控制 {resource_key} 的 RPS/RPM 请求速率。
    同时配置 RPS 和 RPM 时，优先检查 RPM（更稀缺的资源），
    避免 RPS 配额被"幽灵消耗"。
    """

    # 按事件循环 id 隔离单例，避免跨循环共享连接池导致 RuntimeError
    # 使用 WeakValueDictionary 使事件循环销毁后对应 state 可被 GC 自动回收
    _instances: Dict[int, "_RateLimiterState"] = {}
    _init_locks: Dict[int, asyncio.Lock] = {}

    @classmethod
    def _loop_id(cls) -> int:
        # get_running_loop() 仅在协程内有效，语义精确且兼容 Python 3.12+
        # （get_event_loop() 在 3.12 无运行循环时会 RuntimeError）
        loop = asyncio.get_running_loop()
        return id(loop)

    @classmethod
    async def _get_strategy(cls) -> AsyncMovingWindowRateLimiter:
        """懒加载 limits 实例，按事件循环隔离，复用 RedisClient 的连接池"""
        loop_id = cls._loop_id()

        # 快速路径：当前事件循环已有实例
        # 同时清理已关闭循环的 state 和锁，防止 id 复用后取到脏数据
        stale = [lid for lid, s in cls._instances.items() if s.loop.is_closed()]
        for lid in stale:
            cls._instances.pop(lid, None)
            cls._init_locks.pop(lid, None)

        state = cls._instances.get(loop_id)
        if state is not None:
            return state.strategy

        # 慢路径：初始化，按事件循环隔离锁，避免跨循环使用 Lock 导致 RuntimeError
        if loop_id not in cls._init_locks:
            cls._init_locks[loop_id] = asyncio.Lock()

        async with cls._init_locks[loop_id]:
            # 双重检查
            state = cls._instances.get(loop_id)
            if state is not None:
                return state.strategy

            redis = await RedisClient.get_client()
            pool = redis.connection_pool
            pool_kwargs = pool.connection_kwargs
            host = pool_kwargs.get("host", "localhost")
            port = pool_kwargs.get("port", 6379)
            db = pool_kwargs.get("db", 0)
            storage = AsyncRedisStorage(
                f"async+redis://{host}:{port}/{db}",
                implementation="redispy",
                connection_pool=pool,
                key_prefix="rate_limit:",
            )
            strategy = AsyncMovingWindowRateLimiter(storage)
            loop = asyncio.get_running_loop()
            cls._instances[loop_id] = _RateLimiterState(storage=storage, strategy=strategy, loop=loop)

        return cls._instances[loop_id].strategy

    @classmethod
    def reset(cls):
        """重置单例状态（主要用于测试）"""
        cls._instances.clear()
        cls._init_locks.clear()

    def __init__(
        self,
        resource_key: str,
        per_second: Optional[int] = None,
        per_minute: Optional[int] = None,
        max_wait: float = ResourceLimiter.DEFAULT_MAX_WAIT,
    ):
        super().__init__(f"rate:{resource_key}", max_wait)
        self._per_second = per_second
        self._per_minute = per_minute

    async def _try_acquire(self) -> bool:
        """RPS/RPM 限制

        优先检查 RPM（分钟级配额更稀缺），再检查 RPS，
        减少因 RPM 拒绝时 RPS 配额被幽灵消耗的概率。
        注意：limits 库不支持事务性回滚，此顺序是最优近似方案。
        """
        if self._per_second is None and self._per_minute is None:
            return True

        strategy = await self._get_strategy()

        # 优先检查 RPM（更稀缺），RPM 不足时直接拒绝，RPS 配额不受影响
        if self._per_minute is not None:
            if not await strategy.hit(
                RateLimitItemPerMinute(self._per_minute), self.resource_key
            ):
                return False

        # RPM 通过后再检查 RPS
        if self._per_second is not None:
            if not await strategy.hit(
                RateLimitItemPerSecond(self._per_second), self.resource_key
            ):
                # RPS 拒绝：此次 RPM 已消耗，属于可接受的小误差
                # （limits 库无原子回滚接口，实际影响仅为 RPM 多计 1 次）
                return False

        return True

    async def _release(self) -> None:
        """Rate Limiter 不需要释放资源（由时间窗口自动过期）"""
        pass


class _RateLimiterState:
    """内部数据类，持有单个事件循环的 limits 实例"""
    __slots__ = ("storage", "strategy", "loop")

    def __init__(self, storage: AsyncRedisStorage, strategy: AsyncMovingWindowRateLimiter, loop: asyncio.AbstractEventLoop):
        self.storage = storage
        self.strategy = strategy
        self.loop = loop
