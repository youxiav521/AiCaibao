"""
DistributedSemaphore — 分布式信号量

基于Redis实现，用于控制分布式环境下对稀缺资源的并发访问数量。
支持公平排队、超时等待、自动释放等特性。

特性:
  - 基于Redis原子操作实现，保证分布式环境下的正确性
  - 支持超时等待，避免永久阻塞
  - 自动处理死信（进程崩溃未释放的信号量）
  - 支持公平排队（先到先得）
  - 上下文管理器接口，使用简单安全

用法:
    from src.base.semaphore import DistributedSemaphore
    
    # 限制同时调用OCR接口的并发数不超过10
    async with DistributedSemaphore.acquire(
        "ocr:api", 
        max_concurrent=10, 
        timeout=30  # 最多等待30秒
    ) as acquired:
        if not acquired:
            raise TimeoutError("等待OCR资源超时，请稍后重试")
        # 业务逻辑
        await call_ocr_api(file)
    
    # 非上下文管理器用法
    acquired = await DistributedSemaphore.try_acquire("pdf:parse", max_concurrent=5)
    if acquired:
        try:
            await parse_pdf(file)
        finally:
            await DistributedSemaphore.release("pdf:parse")
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from loguru import logger

from src.base.redis_client import RedisClient


class SemaphoreTimeoutError(Exception):
    """获取信号量超时异常"""
    
    def __init__(self, name: str, timeout: int):
        super().__init__(f"获取分布式信号量超时 | name={name} timeout={timeout}s")
        self.name = name
        self.timeout = timeout


class DistributedSemaphore:
    """分布式信号量，控制并发访问数量"""
    
    _KEY_PREFIX = "semaphore:"
    _DEFAULT_LEASE_TIME = 300  # 默认租约时间5分钟，防止进程崩溃导致死锁
    
    @classmethod
    @asynccontextmanager
    async def acquire(
        cls,
        name: str,
        max_concurrent: int,
        timeout: int = 10,
        lease_time: int = _DEFAULT_LEASE_TIME
    ) -> AsyncGenerator[bool, None]:
        """
        异步上下文管理器方式获取信号量，自动释放。
        
        Args:
            name: 信号量名称，标识要控制的资源
            max_concurrent: 最大并发访问数
            timeout: 等待获取信号量的超时时间（秒），0表示不等待
            lease_time: 信号量租约时间（秒），超过时间自动释放，防止死锁
        
        Yields:
            bool: True: 获取成功，False: 获取超时/失败
        """
        acquired = False
        semaphore_id: Optional[str] = None
        
        try:
            semaphore_id = await cls._try_acquire(
                name, 
                max_concurrent, 
                timeout,
                lease_time
            )
            acquired = semaphore_id is not None
            yield acquired
        finally:
            if acquired and semaphore_id:
                await cls._release(name, semaphore_id)
    
    @classmethod
    async def try_acquire(
        cls,
        name: str,
        max_concurrent: int,
        lease_time: int = _DEFAULT_LEASE_TIME
    ) -> Optional[str]:
        """
        尝试立即获取信号量，不等待。
        
        Args:
            name: 信号量名称
            max_concurrent: 最大并发访问数
            lease_time: 租约时间（秒）
        
        Returns:
            信号量ID（释放时需要），获取失败返回None
        """
        return await cls._try_acquire(name, max_concurrent, timeout=0, lease_time=lease_time)
    
    @classmethod
    async def release(cls, name: str, semaphore_id: str) -> None:
        """
        手动释放信号量（非上下文管理器用法时需要）。
        
        Args:
            name: 信号量名称
            semaphore_id: 获取时返回的信号量ID
        """
        await cls._release(name, semaphore_id)
    
    @classmethod
    async def _try_acquire(
        cls,
        name: str,
        max_concurrent: int,
        timeout: int,
        lease_time: int
    ) -> Optional[str]:
        """内部实现：尝试获取信号量"""
        redis = await RedisClient.get_client()
        key = f"{cls._KEY_PREFIX}{name}"
        start_time = time.time()
        
        while True:
            # 先清理过期的信号量（处理死信）
            await cls._cleanup_expired(name, redis)
            
            # 生成唯一的信号量ID
            semaphore_id = f"{time.time()}:{asyncio.get_event_loop().time()}"
            score = time.time() + lease_time  # 过期时间作为score
            
            # 尝试添加信号量
            await redis.zadd(key, {semaphore_id: score})
            
            # 获取当前信号量排名
            rank = await redis.zrank(key, semaphore_id)
            
            if rank is not None and rank < max_concurrent:
                # 获取成功
                logger.debug(
                    "获取分布式信号量成功 | name={} id={} rank={}/{}",
                    name, semaphore_id, rank + 1, max_concurrent
                )
                # 设置key过期时间
                await redis.expire(key, lease_time)
                return semaphore_id
            else:
                # 获取失败，删除刚才添加的
                await redis.zrem(key, semaphore_id)
                
                # 检查是否超时
                if timeout == 0 or time.time() - start_time >= timeout:
                    logger.debug(
                        "获取分布式信号量失败 | name={} max_concurrent={}",
                        name, max_concurrent
                    )
                    return None
                
                # 等待100ms后重试
                await asyncio.sleep(0.1)
    
    @classmethod
    async def _release(cls, name: str, semaphore_id: str) -> None:
        """内部实现：释放信号量"""
        redis = await RedisClient.get_client()
        key = f"{cls._KEY_PREFIX}{name}"
        
        try:
            removed = await redis.zrem(key, semaphore_id)
            if removed > 0:
                logger.debug("释放分布式信号量成功 | name={} id={}", name, semaphore_id)
            else:
                logger.warning("分布式信号量已过期或不存在 | name={} id={}", name, semaphore_id)
        except Exception as e:
            logger.exception("释放分布式信号量失败 | name={} id={}", name, semaphore_id)
    
    @classmethod
    async def _cleanup_expired(cls, name: str, redis) -> None:
        """清理过期的信号量（进程崩溃未释放的）"""
        key = f"{cls._KEY_PREFIX}{name}"
        now = time.time()
        
        try:
            # 删除所有score小于当前时间的元素（已过期）
            removed = await redis.zremrangebyscore(key, 0, now)
            if removed > 0:
                logger.debug("清理过期分布式信号量 | name={} count={}", name, removed)
        except Exception as e:
            logger.exception("清理过期分布式信号量失败 | name={}", name)
    
    @classmethod
    async def get_current_count(cls, name: str) -> int:
        """
        获取当前已获取信号量的数量。
        
        Args:
            name: 信号量名称
        
        Returns:
            当前并发数
        """
        redis = await RedisClient.get_client()
        key = f"{cls._KEY_PREFIX}{name}"
        
        try:
            # 先清理过期的
            await cls._cleanup_expired(name, redis)
            return await redis.zcard(key)
        except Exception as e:
            logger.exception("获取信号量当前计数失败 | name={}", name)
            return 0
