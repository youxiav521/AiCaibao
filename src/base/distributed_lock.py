"""
DistributedLock — 基于 Redis 的分布式锁

底层: redis-py 5.x 原生 asyncio Lock（已是项目依赖，零额外引入）
  - 加锁: SET lock:{name} {token} NX PX {timeout_ms}  （原子操作）
  - 解锁: Lua 脚本先比较 token 再删除                 （防误删他人锁）
  - 续期: Watchdog 后台任务每 timeout/3 秒调用 lock.extend()，防止业务超时导致锁提前释放

锁 Key 格式: lock:{name}
连接: 独立连接池，decode_responses=False（保证 token bytes 比较正确）

用法:
    from src.base.distributed_lock import DistributedLock, LockNotAcquiredError

    # 阻塞模式（等待最多 10 秒）
    async with DistributedLock.acquire("my_resource"):
        # 临界区代码
        ...

    # 非阻塞模式（获取失败立即抛出 LockNotAcquiredError）
    try:
        async with DistributedLock.acquire("my_resource", blocking=False):
            ...
    except LockNotAcquiredError:
        # 处理锁竞争
        ...

    # 自定义参数
    async with DistributedLock.acquire("my_resource", timeout=60, blocking_timeout=5):
        ...
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import redis.asyncio as aioredis
from loguru import logger

from src.base.redis_client import RedisClient

_LOCK_PREFIX = "lock:"
_DEFAULT_TIMEOUT = 30.0            # 锁的初始 TTL（秒），Watchdog 会自动续期
_DEFAULT_BLOCKING_TIMEOUT = 10.0   # 阻塞等待获取锁的最长时间（秒）


async def _watchdog(lock: aioredis.lock.Lock, name: str, timeout: float) -> None:
    """
    Watchdog 后台任务：每 timeout/3 秒自动续期，防止业务代码执行时间超过锁 TTL。

    类似 Redisson 的 watchdog 机制。续期失败时仅记录警告，不中断业务。
    由 acquire() 在获得锁后启动，yield 结束后自动取消。
    """
    interval = timeout / 3
    while True:
        await asyncio.sleep(interval)
        try:
            extended = await lock.extend(timeout, replace_ttl=True)
            if extended:
                logger.debug("分布式锁已续期 | lock={} +{}s", name, timeout)
            else:
                # 锁已不存在（可能已被释放或意外过期）
                logger.warning("分布式锁续期失败，锁已不存在 | lock={}", name)
                break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("分布式锁续期异常 | lock={}", name)
            break


class DistributedLock:
    """
    分布式锁工具类。

    连接管理：使用统一的 RedisClient 管理连接池，Gunicorn fork
    后各 worker 进程独立持有各自连接池，不存在跨进程 socket 共享问题。

    特性:
      - 原子加锁/解锁（SET NX + Lua）
      - Watchdog 自动续期（防止操作耗时超过锁 TTL）
      - 阻塞 / 非阻塞两种模式
    """

    @classmethod
    async def _get_redis(cls) -> aioredis.Redis:
        """返回 Redis 连接池（decode_responses=False）"""
        # decode_responses=False: redis-py Lock 的 token 为 bytes，
        # 若开启 decode_responses=True，GET 返回 str 会导致 Lua 脚本所有权比较失败
        return await RedisClient.get_client(decode_responses=False)

    @classmethod
    @asynccontextmanager
    async def acquire(
        cls,
        name: str,
        timeout: float = _DEFAULT_TIMEOUT,
        blocking: bool = True,
        blocking_timeout: float = _DEFAULT_BLOCKING_TIMEOUT,
    ) -> AsyncGenerator[None, None]:
        """
        获取分布式锁（异步上下文管理器）。

        获取成功后自动启动 Watchdog 后台任务（每 timeout/3 秒续期），
        退出 with 块时 Watchdog 被取消，锁被释放。

        Args:
            name:             锁名称，Redis key 为 lock:{name}
            timeout:          锁的初始 TTL（秒）。Watchdog 会自动续期，
                              因此不需要预估业务最长耗时，保持默认即可。
                              仅在进程崩溃时作为最终兜底（防死锁）。
            blocking:         True = 轮询等待直到获取或超时；False = 立即返回
            blocking_timeout: blocking=True 时，等待获取锁的最长时间（秒）。
                              超时后抛出 LockNotAcquiredError。

        Raises:
            LockNotAcquiredError: blocking=False 且锁已被持有，或等待超时

        示例:
            async with DistributedLock.acquire("report:gen:123"):
                await generate_report(...)   # 即使耗时超过 30s，锁也不会提前释放
        """
        redis = await cls._get_redis()
        lock_key = f"{_LOCK_PREFIX}{name}"
        lock = redis.lock(
            lock_key,
            timeout=timeout,
            blocking_timeout=blocking_timeout,
        )

        acquired = await lock.acquire(blocking=blocking)
        if not acquired:
            raise LockNotAcquiredError(name)

        logger.debug("分布式锁已获取 | lock={} timeout={}s", name, timeout)

        watchdog_task = asyncio.create_task(_watchdog(lock, name, timeout))
        try:
            yield
        finally:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
            try:
                await lock.release()
                logger.debug("分布式锁已释放 | lock={}", name)
            except Exception:
                logger.exception("分布式锁释放失败 | lock={}", name)


class LockNotAcquiredError(Exception):
    """无法获取分布式锁时抛出（锁被占用或等待超时）"""

    def __init__(self, name: str) -> None:
        super().__init__(f"无法获取分布式锁 | lock={name}")
        self.lock_name = name
