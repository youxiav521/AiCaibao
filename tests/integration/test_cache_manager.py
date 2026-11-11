"""
CacheManager 集成测试（冒烟测试）
验证基于 Redis 的分布式缓存基本功能正常运行
"""
import asyncio
import pytest
from pydantic import BaseModel
from src.base.cache_manager import CacheManager
from src.base.redis_client import RedisClient


class TestModel(BaseModel):
    """测试用Pydantic模型"""
    id: int
    name: str
    value: float


@pytest.fixture(scope="function", autouse=True)
async def reset_redis_connection():
    """每个测试前重置Redis连接并清理测试数据"""
    # 重置RedisClient连接
    RedisClient._clients.clear()
    yield
    # 清理所有测试相关的缓存key
    redis = await RedisClient.get_client()
    keys = await redis.keys("cache:test:*")
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
async def test_basic_set_get(load_env):
    """测试基础的设置和获取缓存功能"""
    key = "test:basic:key"
    value = "test_value"
    
    # 设置缓存
    await CacheManager.set(key, value, ttl=60)
    
    # 获取缓存
    result = await CacheManager.get(key)
    assert result == value, f"缓存值不匹配，预期 {value}，实际 {result}"
    
    # 验证key存在
    exists = await CacheManager.exists(key)
    assert exists is True, "缓存key应该存在"


@pytest.mark.asyncio
async def test_get_nonexistent_key(load_env):
    """测试获取不存在的key返回默认值"""
    key = "test:nonexistent:key"
    
    # 不存在的key默认返回None
    result = await CacheManager.get(key)
    assert result is None, "不存在的key应该返回None"
    
    # 自定义默认值
    result = await CacheManager.get(key, default="default_value")
    assert result == "default_value", "自定义默认值不生效"


@pytest.mark.asyncio
async def test_pydantic_model_serialization(load_env):
    """测试Pydantic模型的自动序列化和反序列化"""
    key = "test:model:key"
    model = TestModel(id=1, name="test", value=3.14)
    
    # 存储模型
    await CacheManager.set(key, model, ttl=60)
    
    # 取回并自动反序列化为模型
    result = await CacheManager.get(key, model_type=TestModel)
    assert isinstance(result, TestModel), "返回值应该是TestModel实例"
    assert result.id == model.id
    assert result.name == model.name
    assert result.value == model.value


@pytest.mark.asyncio
async def test_delete_cache(load_env):
    """测试删除缓存功能"""
    key = "test:delete:key"
    value = "to_delete"
    
    # 设置缓存
    await CacheManager.set(key, value, ttl=60)
    assert await CacheManager.exists(key) is True
    
    # 删除缓存
    await CacheManager.delete(key)
    assert await CacheManager.exists(key) is False
    assert await CacheManager.get(key) is None


@pytest.mark.asyncio
async def test_null_value_caching(load_env):
    """测试空值缓存，防止缓存穿透"""
    key = "test:null:key"
    
    # 缓存None值
    await CacheManager.set(key, None, ttl=60, cache_null=True)
    
    # 获取应该返回None而不是默认值
    result = await CacheManager.get(key, default="default")
    assert result is None, "缓存的None值应该返回None"
    assert await CacheManager.exists(key) is True


@pytest.mark.asyncio
async def test_ttl_expiration(load_env):
    """测试缓存TTL过期功能"""
    key = "test:ttl:key"
    value = "expire_test"
    
    # 设置短TTL
    await CacheManager.set(key, value, ttl=1)
    
    # 立即获取应该存在
    assert await CacheManager.get(key) == value
    
    # 等待2秒让缓存过期
    await asyncio.sleep(2)
    
    # 应该获取不到
    assert await CacheManager.get(key) is None
    assert await CacheManager.exists(key) is False


@pytest.mark.asyncio
async def test_counter_incr(load_env):
    """测试原子递增计数器功能"""
    key = "test:counter:key"
    
    # 第一次递增，初始为1
    result = await CacheManager.incr(key, amount=1, ttl=60)
    assert result == 1, "第一次递增应该返回1"
    
    # 第二次递增，返回2
    result = await CacheManager.incr(key, amount=1)
    assert result == 2, "第二次递增应该返回2"
    
    # 递增5
    result = await CacheManager.incr(key, amount=5)
    assert result == 7, "递增5应该返回7"
    
    # 验证值正确
    value = await CacheManager.get(key)
    assert int(value) == 7, "计数器值不正确"


@pytest.mark.asyncio
async def test_different_data_types(load_env):
    """测试不同数据类型的序列化"""
    # 字符串
    await CacheManager.set("test:type:string", "hello", ttl=60)
    assert await CacheManager.get("test:type:string") == "hello"
    
    # 数字
    await CacheManager.set("test:type:int", 123, ttl=60)
    assert await CacheManager.get("test:type:int") == 123
    
    # 列表
    await CacheManager.set("test:type:list", [1, 2, 3], ttl=60)
    assert await CacheManager.get("test:type:list") == [1, 2, 3]
    
    # 字典
    await CacheManager.set("test:type:dict", {"a": 1, "b": 2}, ttl=60)
    assert await CacheManager.get("test:type:dict") == {"a": 1, "b": 2}
    
    # 布尔值
    await CacheManager.set("test:type:bool", True, ttl=60)
    assert await CacheManager.get("test:type:bool") is True
