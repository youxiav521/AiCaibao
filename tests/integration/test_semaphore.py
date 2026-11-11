"""
DistributedSemaphore 集成测试（冒烟测试）
验证基于 Redis 的分布式信号量基本功能正常运行
"""
import asyncio
import pytest
from src.base.semaphore import DistributedSemaphore
from src.base.redis_client import RedisClient


@pytest.fixture(scope="function", autouse=True)
async def reset_redis_connection():
    """每个测试前重置Redis连接并清理测试数据"""
    # 重置RedisClient连接
    RedisClient._clients.clear()
    yield
    # 清理所有测试相关的信号量key
    redis = await RedisClient.get_client()
    keys = await redis.keys("semaphore:test:*")
    if keys:
        await redis.delete(*keys)
    # 重置连接
    RedisClient._clients.clear()


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
async def test_basic_acquire_release(load_env):
    """测试基础的获取和释放信号量流程"""
    semaphore_name = "test:basic:semaphore"
    max_concurrent = 2
    
    # 获取第一个信号量
    async with DistributedSemaphore.acquire(semaphore_name, max_concurrent) as acquired:
        assert acquired is True, "应该成功获取信号量"
        
        # 获取第二个信号量
        async with DistributedSemaphore.acquire(semaphore_name, max_concurrent) as acquired2:
            assert acquired2 is True, "应该成功获取第二个信号量"
            
            # 第三个应该获取失败（非阻塞）
            semaphore_id = await DistributedSemaphore.try_acquire(semaphore_name, max_concurrent)
            assert semaphore_id is None, "达到最大并发数时应该获取失败"
    
    # 释放后应该可以重新获取
    async with DistributedSemaphore.acquire(semaphore_name, max_concurrent) as acquired:
        assert acquired is True, "释放后应该可以重新获取信号量"


@pytest.mark.asyncio
async def test_try_acquire_non_blocking(load_env):
    """测试非阻塞模式获取信号量"""
    semaphore_name = "test:nonblocking:semaphore"
    max_concurrent = 1
    
    # 第一个获取成功
    semaphore_id = await DistributedSemaphore.try_acquire(semaphore_name, max_concurrent)
    assert semaphore_id is not None, "应该成功获取信号量"
    
    # 第二个获取应该立即失败
    start_time = asyncio.get_event_loop().time()
    semaphore_id2 = await DistributedSemaphore.try_acquire(semaphore_name, max_concurrent)
    elapsed = asyncio.get_event_loop().time() - start_time
    
    assert semaphore_id2 is None, "非阻塞模式下应该获取失败"
    assert elapsed < 0.1, f"非阻塞模式应该立即返回，耗时 {elapsed} 秒"
    
    # 手动释放
    await DistributedSemaphore.release(semaphore_name, semaphore_id)
    
    # 释放后可以再次获取
    semaphore_id3 = await DistributedSemaphore.try_acquire(semaphore_name, max_concurrent)
    assert semaphore_id3 is not None, "释放后应该可以重新获取"


@pytest.mark.asyncio
async def test_blocking_timeout(load_env):
    """测试阻塞模式超时"""
    semaphore_name = "test:blocking:timeout:semaphore"
    max_concurrent = 1
    
    async def hold_semaphore():
        async with DistributedSemaphore.acquire(semaphore_name, max_concurrent, lease_time=5):
            await asyncio.sleep(2)
    
    # 启动一个协程持有信号量2秒
    hold_task = asyncio.create_task(hold_semaphore())
    await asyncio.sleep(0.1)  # 确保先拿到信号量
    
    # 尝试获取信号量，阻塞超时1秒，应该失败
    start_time = asyncio.get_event_loop().time()
    async with DistributedSemaphore.acquire(
        semaphore_name, max_concurrent, timeout=1
    ) as acquired:
        assert acquired is False, "应该超时获取失败"
    
    elapsed = asyncio.get_event_loop().time() - start_time
    assert 0.9 < elapsed < 1.5, f"阻塞超时时间不正确，耗时 {elapsed} 秒"
    
    await hold_task


@pytest.mark.asyncio
async def test_concurrent_access_control(load_env):
    """测试并发访问控制，确保不超过最大并发数"""
    semaphore_name = "test:concurrent:semaphore"
    max_concurrent = 3
    concurrent_tasks = 10
    active_count = 0
    max_active = 0
    completed_tasks = 0
    
    async def worker():
        nonlocal active_count, max_active, completed_tasks
        async with DistributedSemaphore.acquire(
            semaphore_name, max_concurrent, timeout=5
        ) as acquired:
            if acquired:
                active_count += 1
                if active_count > max_active:
                    max_active = active_count
                # 模拟业务处理
                await asyncio.sleep(0.2)
                active_count -= 1
                completed_tasks += 1
    
    # 启动多个并发任务
    tasks = [asyncio.create_task(worker()) for _ in range(concurrent_tasks)]
    await asyncio.gather(*tasks)
    
    # 验证最大并发数不超过限制
    assert max_active <= max_concurrent, f"最大并发数 {max_active} 超过限制 {max_concurrent}"
    assert max_active > 0, "应该有任务成功获取到信号量"
    # 所有任务都应该完成
    assert completed_tasks == concurrent_tasks, f"应该有 {concurrent_tasks} 个任务完成，实际 {completed_tasks}"


@pytest.mark.asyncio
async def test_lease_time_expiration(load_env):
    """测试租约过期自动释放，防止死锁"""
    semaphore_name = "test:lease:expire:semaphore"
    max_concurrent = 1
    
    # 获取信号量，租约时间1秒
    semaphore_id = await DistributedSemaphore.try_acquire(
        semaphore_name, max_concurrent, lease_time=1
    )
    assert semaphore_id is not None
    
    # 立即尝试获取第二个应该失败
    semaphore_id2 = await DistributedSemaphore.try_acquire(semaphore_name, max_concurrent)
    assert semaphore_id2 is None
    
    # 等待2秒让租约过期
    await asyncio.sleep(2)
    
    # 过期后应该可以获取到
    semaphore_id3 = await DistributedSemaphore.try_acquire(semaphore_name, max_concurrent)
    assert semaphore_id3 is not None, "租约过期后应该可以重新获取信号量"


@pytest.mark.asyncio
async def test_get_current_count(load_env):
    """测试获取当前已使用的信号量数量"""
    semaphore_name = "test:count:semaphore"
    max_concurrent = 3
    
    # 初始应该为0
    count = await DistributedSemaphore.get_current_count(semaphore_name)
    assert count == 0, "初始计数应该为0"
    
    # 获取一个
    async with DistributedSemaphore.acquire(semaphore_name, max_concurrent):
        count = await DistributedSemaphore.get_current_count(semaphore_name)
        assert count == 1, "计数应该为1"
        
        # 获取第二个
        async with DistributedSemaphore.acquire(semaphore_name, max_concurrent):
            count = await DistributedSemaphore.get_current_count(semaphore_name)
            assert count == 2, "计数应该为2"
    
    # 释放后应该回到0
    count = await DistributedSemaphore.get_current_count(semaphore_name)
    assert count == 0, "释放后计数应该回到0"


@pytest.mark.asyncio
async def test_fairness_order(load_env):
    """测试公平性，先到先得"""
    semaphore_name = "test:fairness:semaphore"
    max_concurrent = 1
    execution_order = []
    
    async def worker(task_id):
        async with DistributedSemaphore.acquire(
            semaphore_name, max_concurrent, timeout=10
        ) as acquired:
            if acquired:
                execution_order.append(task_id)
                await asyncio.sleep(0.1)
    
    # 按顺序启动任务
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(worker(i)))
        await asyncio.sleep(0.01)  # 确保任务按顺序发起
    
    await asyncio.gather(*tasks)
    
    # 验证执行顺序与启动顺序一致（公平性）
    assert execution_order == [0, 1, 2, 3, 4], f"执行顺序应该是 [0,1,2,3,4]，实际 {execution_order}"


@pytest.mark.asyncio
async def test_semaphore_with_different_max_values(load_env):
    """测试不同最大并发数的信号量互不影响"""
    # 两个不同的信号量
    sem1_name = "test:sem:1"
    sem2_name = "test:sem:2"
    
    max1 = 2
    max2 = 3
    
    # 获取sem1的2个
    async with DistributedSemaphore.acquire(sem1_name, max1):
        async with DistributedSemaphore.acquire(sem1_name, max1):
            # sem1满了，获取失败
            id1 = await DistributedSemaphore.try_acquire(sem1_name, max1)
            assert id1 is None
            
            # sem2还可以获取3个
            id2 = await DistributedSemaphore.try_acquire(sem2_name, max2)
            assert id2 is not None
