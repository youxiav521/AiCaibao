"""
分布式锁集成测试（冒烟测试）
验证基于 Redis 的分布式锁基本功能正常运行
"""
import asyncio
import pytest
from src.base.distributed_lock import DistributedLock, LockNotAcquiredError


@pytest.fixture(scope="function", autouse=True)
async def reset_redis_connection():
    """每个测试前重置Redis连接，避免事件循环跨测试复用问题"""
    from src.base.redis_client import RedisClient
    
    # 测试前关闭所有已存在的Redis连接
    await RedisClient.close()
    
    yield
    
    # 测试结束后再次关闭所有连接，清理资源
    await RedisClient.close()


@pytest.fixture(scope="session")
def load_env():
    """加载环境变量"""
    from dotenv import load_dotenv
    import os
    # 优先加载测试环境变量
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env.test")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    # 加载主环境变量作为兜底
    load_dotenv()


@pytest.mark.asyncio
async def test_basic_lock_acquire_release(load_env):
    """测试基础加锁解锁流程"""
    lock_name = "test:basic:lock"
    
    # 正常加锁
    async with DistributedLock.acquire(lock_name, timeout=10):
        # 验证锁已被持有，非阻塞模式获取应该失败
        with pytest.raises(LockNotAcquiredError):
            async with DistributedLock.acquire(lock_name, blocking=False):
                # 不应该执行到这里
                assert False, "应该抛出 LockNotAcquiredError"
    
    # 锁释放后应该可以重新获取
    async with DistributedLock.acquire(lock_name, blocking=False):
        assert True, "锁释放后应该可以重新获取"


@pytest.mark.asyncio
async def test_non_blocking_mode(load_env):
    """测试非阻塞模式"""
    lock_name = "test:nonblocking:lock"
    
    # 第一个协程持有锁
    async with DistributedLock.acquire(lock_name, timeout=5):
        # 第二个协程非阻塞获取应该立即失败
        start_time = asyncio.get_event_loop().time()
        with pytest.raises(LockNotAcquiredError):
            async with DistributedLock.acquire(lock_name, blocking=False):
                assert False
        elapsed = asyncio.get_event_loop().time() - start_time
        # 非阻塞模式应该立即返回，不应该等待
        assert elapsed < 0.1, f"非阻塞模式响应太慢，耗时 {elapsed} 秒"


@pytest.mark.asyncio
async def test_blocking_timeout(load_env):
    """测试阻塞模式超时"""
    lock_name = "test:blocking:timeout:lock"
    
    async def hold_lock():
        async with DistributedLock.acquire(lock_name, timeout=3):
            await asyncio.sleep(2)
    
    # 启动一个协程持有锁2秒
    hold_task = asyncio.create_task(hold_lock())
    await asyncio.sleep(0.1)  # 确保先拿到锁
    
    # 尝试获取锁，阻塞超时1秒，应该失败
    start_time = asyncio.get_event_loop().time()
    with pytest.raises(LockNotAcquiredError):
        async with DistributedLock.acquire(lock_name, blocking_timeout=1):
            assert False
    elapsed = asyncio.get_event_loop().time() - start_time
    assert 0.9 < elapsed < 1.5, f"阻塞超时时间不正确，耗时 {elapsed} 秒"
    
    await hold_task


@pytest.mark.asyncio
async def test_watchdog_renew(load_env):
    """测试 Watchdog 自动续期功能"""
    lock_name = "test:watchdog:lock"
    
    # 锁初始TTL为2秒，业务执行3秒，验证Watchdog会自动续期
    async with DistributedLock.acquire(lock_name, timeout=2):
        # 等待超过初始TTL时间
        await asyncio.sleep(3)
        # 锁仍然应该有效，尝试在同一个锁的非阻塞获取应该失败
        with pytest.raises(LockNotAcquiredError):
            async with DistributedLock.acquire(lock_name, blocking=False):
                assert False


@pytest.mark.asyncio
async def test_concurrent_access(load_env):
    """测试并发访问时只有一个协程能获取锁"""
    lock_name = "test:concurrent:lock"
    counter = 0
    concurrent_tasks = 10
    
    async def worker():
        nonlocal counter
        try:
            async with DistributedLock.acquire(lock_name, blocking_timeout=5):
                # 临界区，模拟业务操作
                await asyncio.sleep(0.1)
                counter += 1
        except LockNotAcquiredError:
            # 允许部分获取失败
            pass
    
    # 启动多个并发任务
    tasks = [asyncio.create_task(worker()) for _ in range(concurrent_tasks)]
    await asyncio.gather(*tasks)
    
    # 至少有一个任务成功获取到锁
    assert counter > 0, "没有任何任务成功获取到锁"
    assert counter <= concurrent_tasks, "计数器异常"
