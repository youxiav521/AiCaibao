import asyncio
import os
import time

import pytest
from dotenv import load_dotenv

from src.base.redis_client import RedisClient
from src.utils.model_enhancements.resource_limiter import (
    ConcurrencyLimiter,
    LimitTimeoutError,
    RateLimiter,
)


@pytest.fixture(scope="function", autouse=True)
async def reset_redis_and_limiter():
    RedisClient._clients.clear()
    RedisClient._lock = None
    RateLimiter.reset()
    RateLimiter._init_lock = asyncio.Lock()
    yield
    redis = await RedisClient.get_client()
    concurrency_keys = await redis.keys("concurrency:test:*")
    if concurrency_keys:
        await redis.delete(*concurrency_keys)
    rate_keys = await redis.keys("rate_limit:rate:test:*")
    if rate_keys:
        await redis.delete(*rate_keys)
    RedisClient._clients.clear()
    RedisClient._lock = None
    RateLimiter.reset()
    RateLimiter._init_lock = asyncio.Lock()


@pytest.fixture(scope="session")
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env.test")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    load_dotenv()


# ============ ConcurrencyLimiter ============


class TestConcurrencyLimiter:

    @pytest.mark.asyncio
    async def test_basic_acquire_release(self, load_env):
        limiter = ConcurrencyLimiter("test:basic", max_concurrent=2, max_wait=5)
        await limiter.acquire()
        assert limiter._acquired is True
        await limiter.release()
        assert limiter._acquired is False

    @pytest.mark.asyncio
    async def test_acquire_returns_true(self, load_env):
        limiter = ConcurrencyLimiter("test:acquire:true", max_concurrent=2, max_wait=5)
        result = await limiter.acquire()
        assert result is True
        await limiter.release()

    @pytest.mark.asyncio
    async def test_context_manager(self, load_env):
        async with ConcurrencyLimiter("test:ctx", max_concurrent=2, max_wait=5) as limiter:
            assert limiter._acquired is True
        assert limiter._acquired is False

    @pytest.mark.asyncio
    async def test_cannot_exceed_max_concurrent(self, load_env):
        async with ConcurrencyLimiter("test:exceed", max_concurrent=2, max_wait=5):
            async with ConcurrencyLimiter("test:exceed", max_concurrent=2, max_wait=5):
                with pytest.raises(LimitTimeoutError):
                    await ConcurrencyLimiter("test:exceed", max_concurrent=2, max_wait=0.5).acquire()

    @pytest.mark.asyncio
    async def test_release_when_not_acquired_is_noop(self, load_env):
        limiter = ConcurrencyLimiter("test:noop", max_concurrent=2, max_wait=5)
        await limiter.release()
        assert limiter._acquired is False

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self, load_env):
        limiter = ConcurrencyLimiter("test:exc", max_concurrent=1, max_wait=5)
        with pytest.raises(ValueError):
            async with limiter:
                raise ValueError("simulated")
        assert limiter._acquired is False
        async with ConcurrencyLimiter("test:exc", max_concurrent=1, max_wait=5):
            pass

    @pytest.mark.asyncio
    async def test_concurrent_not_exceeded(self, load_env):
        max_concurrent = 3
        total_tasks = 20
        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def worker():
            nonlocal active_count, max_active
            limiter = ConcurrencyLimiter(
                "test:concurrent", max_concurrent=max_concurrent, max_wait=30, lease_timeout=30
            )
            async with limiter:
                async with lock:
                    active_count += 1
                    max_active = max(max_active, active_count)
                await asyncio.sleep(0.05)
                async with lock:
                    active_count -= 1

        tasks = [asyncio.create_task(worker()) for _ in range(total_tasks)]
        await asyncio.gather(*tasks)

        assert max_active <= max_concurrent, (
            f"max_active={max_active} exceeded max_concurrent={max_concurrent}"
        )
        assert max_active == max_concurrent, (
            f"expected to hit max_concurrent={max_concurrent} but max_active={max_active}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_all_complete(self, load_env):
        max_concurrent = 5
        total_tasks = 50
        completed = 0
        lock = asyncio.Lock()

        async def worker():
            nonlocal completed
            limiter = ConcurrencyLimiter(
                "test:all:complete", max_concurrent=max_concurrent, max_wait=60, lease_timeout=60
            )
            async with limiter:
                await asyncio.sleep(0.01)
                async with lock:
                    completed += 1

        tasks = [asyncio.create_task(worker()) for _ in range(total_tasks)]
        await asyncio.gather(*tasks)

        assert completed == total_tasks, f"only {completed}/{total_tasks} tasks completed"

    @pytest.mark.asyncio
    async def test_release_frees_slot(self, load_env):
        async with ConcurrencyLimiter("test:release", max_concurrent=1, max_wait=5):
            pass
        async with ConcurrencyLimiter("test:release", max_concurrent=1, max_wait=5):
            pass

    @pytest.mark.asyncio
    async def test_lease_expiration(self, load_env):
        limiter = ConcurrencyLimiter("test:lease", max_concurrent=1, lease_timeout=1, max_wait=5)
        await limiter.acquire()

        result = await ConcurrencyLimiter(
            "test:lease", max_concurrent=1, lease_timeout=1, max_wait=0.5
        )._try_acquire()
        assert result is False, "holder not yet expired, should block"

        await asyncio.sleep(1.5)

        result = await ConcurrencyLimiter(
            "test:lease", max_concurrent=1, lease_timeout=10, max_wait=5
        )._try_acquire()
        assert result is True, "should acquire after lease expired"

    @pytest.mark.asyncio
    async def test_unlimited_max_concurrent_zero(self, load_env):
        l1 = ConcurrencyLimiter("test:unlimited", max_concurrent=0, max_wait=1)
        l2 = ConcurrencyLimiter("test:unlimited", max_concurrent=0, max_wait=1)
        assert await l1._try_acquire() is True
        assert await l2._try_acquire() is True


# ============ RateLimiter ============


class TestRateLimiter:

    @pytest.mark.asyncio
    async def test_basic_acquire_no_limits(self, load_env):
        limiter = RateLimiter("test:no:limit", max_wait=1)
        assert await limiter._try_acquire() is True
        assert await limiter._try_acquire() is True

    @pytest.mark.asyncio
    async def test_rps_limit_enforced(self, load_env):
        per_second = 5
        limiter = RateLimiter("test:rps:basic", per_second=per_second)

        success = 0
        for _ in range(per_second + 5):
            if await limiter._try_acquire():
                success += 1

        assert success == per_second, f"expected {per_second} successes, got {success}"

    @pytest.mark.asyncio
    async def test_rpm_limit_enforced(self, load_env):
        per_minute = 5
        limiter = RateLimiter("test:rpm:basic", per_minute=per_minute)

        success = 0
        for _ in range(per_minute + 5):
            if await limiter._try_acquire():
                success += 1

        assert success == per_minute, f"expected {per_minute} successes, got {success}"

    @pytest.mark.asyncio
    async def test_combined_rps_binding(self, load_env):
        limiter = RateLimiter("test:combined:rps", per_second=3, per_minute=10)

        success = 0
        for _ in range(10):
            if await limiter._try_acquire():
                success += 1

        assert success == 3, f"RPS should be the binding constraint, got {success}"

    @pytest.mark.asyncio
    async def test_combined_rpm_binding(self, load_env):
        limiter = RateLimiter("test:combined:rpm", per_second=10, per_minute=3)

        success = 0
        for _ in range(10):
            if await limiter._try_acquire():
                success += 1

        assert success == 3, f"RPM should be the binding constraint, got {success}"

    @pytest.mark.asyncio
    async def test_acquire_returns_true_for_rate_limiter(self, load_env):
        limiter = RateLimiter("test:acquire:true", per_second=10, max_wait=5)
        result = await limiter.acquire()
        assert result is True
        await limiter.release()

    @pytest.mark.asyncio
    async def test_context_manager_for_rate(self, load_env):
        async with RateLimiter("test:rate:ctx", per_second=10, max_wait=5) as limiter:
            assert limiter._acquired is True
        assert limiter._acquired is False

    @pytest.mark.asyncio
    async def test_concurrent_rps_not_exceeded(self, load_env):
        per_second = 25
        total_tasks = 100
        success_times = []
        lock = asyncio.Lock()

        async def worker():
            limiter = RateLimiter("test:rps:concurrent", per_second=per_second, max_wait=60)
            await limiter.acquire()
            async with lock:
                success_times.append(time.monotonic())
            await limiter.release()

        tasks = [asyncio.create_task(worker()) for _ in range(total_tasks)]
        await asyncio.gather(*tasks)

        success_times.sort()

        for i in range(len(success_times)):
            window_end = success_times[i]
            window_start = window_end - 1.0
            count = sum(1 for t in success_times if window_start < t <= window_end)
            assert count <= per_second + 2, (
                f"RPS breached: {count} > {per_second}+2 at t={window_end - success_times[0]:.2f}s"
            )

    @pytest.mark.asyncio
    async def test_concurrent_rpm_not_exceeded(self, load_env):
        per_minute = 30
        total_tasks = 40
        success_times = []
        lock = asyncio.Lock()

        async def worker():
            limiter = RateLimiter("test:rpm:concurrent", per_minute=per_minute, max_wait=120)
            await limiter.acquire()
            async with lock:
                success_times.append(time.monotonic())
            await limiter.release()

        tasks = [asyncio.create_task(worker()) for _ in range(total_tasks)]
        await asyncio.gather(*tasks)

        success_times.sort()

        for i in range(len(success_times)):
            window_end = success_times[i]
            window_start = window_end - 60.0
            count = sum(1 for t in success_times if window_start < t <= window_end)
            assert count <= per_minute + 2, (
                f"RPM breached: {count} > {per_minute}+2 at t={window_end - success_times[0]:.1f}s"
            )

    @pytest.mark.asyncio
    async def test_high_concurrency_rps_accuracy(self, load_env):
        per_second = 50
        total_tasks = 200
        success_times = []
        lock = asyncio.Lock()
        start = time.monotonic()

        async def worker():
            limiter = RateLimiter("test:high:rps", per_second=per_second, max_wait=30)
            await limiter.acquire()
            async with lock:
                success_times.append(time.monotonic())
            await limiter.release()

        tasks = [asyncio.create_task(worker()) for _ in range(total_tasks)]
        await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start
        success_times.sort()

        for i in range(len(success_times)):
            window_end = success_times[i]
            window_start = window_end - 1.0
            count = sum(1 for t in success_times if window_start < t <= window_end)
            assert count <= per_second + 2, (
                f"high concurrency RPS breached: {count} > {per_second}+2"
            )

        expected_min_time = total_tasks / per_second
        assert elapsed >= expected_min_time * 0.85, (
            f"throughput too fast: {elapsed:.1f}s < {expected_min_time:.1f}s"
        )
