"""
TaskManager — 任务生命周期管理

职责:
  1. submit(task_name, task_type, kwargs, metadata) → 入队 ARQ + 写 Redis 记录 → 返回 job_id
  2. get(job_id) → 从 Redis 读取完整 TaskRecord
  3. update(job_id, status, ...) → Worker 执行各阶段回调，更新 Redis 记录

数据存储:
  Key: task:{job_id}
  Value: TaskRecord JSON
  TTL: 默认 4 周 (2419200s)，可通过 TASK_RECORD_TTL_SECONDS 环境变量覆盖

索引:
  task_type:{type}:jobs — Sorted Set，按 created_at 分数索引 job_id，支持高效分页

字段规范:
  - task_type: 任务类型，用于分类筛选（默认 "default"）
  - 大内容（如 markdown、文件）应上传至 storage_service，结果中仅保留引用路径, 否则redis会占用大量服务器内存.

使用方式:
  # 提交任务（指定 task_type）
  job_id = await TaskManager.submit("say_hello", task_type="default", kwargs={"message": "hi"})

  # Worker 内 — 标记运行中
  await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=1)

  # Worker 内 — 持久化中间结果（断点续跑）
  await TaskManager.update(job_id, TaskStatus.RUNNING, result=partial_result, attempts=1)

  # Worker 内 — 标记成功（result 中存储引用路径，非大内容）
  await TaskManager.update(job_id, TaskStatus.SUCCESS, result={"markdown_path": "bucket/path/to/file.md"}, attempts=1)

  # 查询结果
  record = await TaskManager.get(job_id)
  # record.status / record.result / record.error / record.attempts

  # 按 task_type 分页获取
  records = await TaskManager.get_all(task_type="mineru", limit=100, offset=0)

断点续跑模式（由任务函数自行管理）:
  async def my_task(ctx, ...):
      job_id = ctx["job_id"]
      job_try = ctx.get("job_try", 1)

      # 重试时读取之前已持久化的中间结果
      prev = (await TaskManager.get(job_id)).result or {}
      await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)

      if "step_a" not in prev:
          prev["step_a"] = do_step_a()
          await TaskManager.update(job_id, TaskStatus.RUNNING, result=prev, attempts=job_try)

      if "step_b" not in prev:
          prev["step_b"] = do_step_b()
          await TaskManager.update(job_id, TaskStatus.RUNNING, result=prev, attempts=job_try)

      await TaskManager.update(job_id, TaskStatus.SUCCESS, result=prev, attempts=job_try)
"""

import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import RedisSettings
from arq.constants import (
    abort_jobs_ss,
    default_queue_name,
    in_progress_key_prefix,
    job_key_prefix,
    result_key_prefix,
    retry_key_prefix,
)
from loguru import logger
from pydantic import BaseModel, Field

from src.config import config
from src.base.redis_client import RedisClient

# 默认过期时间: 4 周
_DEFAULT_TTL_SECONDS: int = int(
    os.getenv("TASK_RECORD_TTL_SECONDS", str(28 * 24 * 3600))
)
_KEY_PREFIX = "task:"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled" #TODO: 待实现


class TaskRecord(BaseModel):
    """完整的任务记录，存储在 Redis 中"""

    job_id: str
    task_name: str
    task_type: str = "default"
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    # 提交时的入参
    request: dict[str, Any] = Field(default_factory=dict)
    # 执行完成后的结果（也用于存储中间状态，支持断点续跑）
    result: Optional[Any] = None
    # 失败时的错误信息
    error: Optional[str] = None
    # 已执行次数（含重试）
    attempts: int = 0
    # 调用方附加的业务元数据
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskManager:
    """
    应用级任务管理器（类方法单例）。

    连接池在首次调用时懒加载，Gunicorn fork 后各 worker 进程独立持有各自连接池，
    不存在跨进程 socket 共享问题。
    """

    _arq_pool = None

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @classmethod
    def _build_redis_settings(cls) -> RedisSettings:
        if RedisClient.is_sentinel_mode():
            return RedisSettings(
                host=RedisClient.parse_sentinel_hosts(),
                password=config.REDIS_PASSWORD or None,
                database=config.REDIS_DB,
                sentinel=True,
                sentinel_master=config.REDIS_SENTINEL_MASTER_NAME,
            )
        return RedisSettings(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD or None,
            database=config.REDIS_DB,
        )

    @classmethod
    async def _get_redis(cls) -> aioredis.Redis:
        """返回 Redis 连接（decode_responses=True）"""
        return await RedisClient.get_client(decode_responses=True)

    @classmethod
    async def _get_arq_pool(cls):
        """返回懒加载的 ARQ 连接池（用于 enqueue_job）"""
        if cls._arq_pool is None:
            cls._arq_pool = await create_pool(cls._build_redis_settings())
        return cls._arq_pool

    @classmethod
    def _key(cls, job_id: str) -> str:
        return f"{_KEY_PREFIX}{job_id}"

    @classmethod
    def _type_index_key(cls, task_type: str) -> str:
        return f"task_type:{task_type}:jobs"

    @classmethod
    async def _add_to_type_index(cls, r: aioredis.Redis, task_type: str, job_id: str, created_at: str) -> None:
        """将 job_id 添加到 task_type 对应的 Sorted Set 中"""
        try:
            score = datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp()
        except Exception:
            score = 0
        await r.zadd(cls._type_index_key(task_type), {job_id: score})

    # ── 公开 API ──────────────────────────────────────────────────────────────

    @classmethod
    async def submit(
        cls,
        task_name: str,
        task_type: str = "default",
        kwargs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        提交任务到 ARQ 队列，同时在 Redis 写入初始 TaskRecord。

        Args:
            task_name: 任务函数名
            task_type: 任务类型（用于分类筛选）
            kwargs: 任务参数
            metadata: 业务元数据

        Returns:
            job_id: ARQ 分配的任务 ID，同时作为 Redis 记录的查询键
        """
        kwargs = kwargs or {}
        metadata = metadata or {}

        pool = await cls._get_arq_pool()
        job = await pool.enqueue_job(task_name, **kwargs)
        if job is None:
            raise RuntimeError(f"任务 {task_name} 入队失败（未知原因）")
        job_id = job.job_id

        record = TaskRecord(
            job_id=job_id,
            task_name=task_name,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            request=kwargs,
            metadata=metadata,
        )

        r = await cls._get_redis()
        await r.setex(cls._key(job_id), _DEFAULT_TTL_SECONDS, record.model_dump_json())
        await cls._add_to_type_index(r, task_type, job_id, record.created_at)

        logger.info(
            "TaskManager | 任务已提交 | job_id={} task={} task_type={} ttl={}s",
            job_id, task_name, task_type, _DEFAULT_TTL_SECONDS,
        )
        return job_id
    @classmethod
    async def submit_with_job_id(

        cls,
        job_id: str,
        task_name: str,
        task_type: str ,
        kwargs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        使用指定的 job_id 提交任务到 ARQ 队列，同时在 Redis 写入初始 TaskRecord。

        Args:
            job_id: 指定的任务 ID
            task_name: 任务函数名
            task_type: 任务类型（用于分类筛选）
            kwargs: 任务参数
            metadata: 业务元数据

        Returns:
            job_id: 指定的任务 ID

        Raises:
            RuntimeError: job_id 冲突导致入队失败
        """
        kwargs = kwargs or {}
        metadata = metadata or {}

        pool = await cls._get_arq_pool()

        r = await cls._get_redis()

        result = await pool.enqueue_job(task_name, _job_id=job_id, **kwargs)
        if result is None:
            logger.error(
                "TaskManager | 任务入队失败 | job_id={} task={} 可能存在残留 key 阻塞",
                job_id, task_name,
            )
            raise RuntimeError(
                f"job_id={job_id} 入队失败，该 ID 可能被残留 key 阻塞，无法重复提交"
            )

        record = TaskRecord(
            job_id=job_id,
            task_name=task_name,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            request=kwargs,
            metadata=metadata,
        )

        await r.setex(cls._key(job_id), _DEFAULT_TTL_SECONDS, record.model_dump_json())
        await cls._add_to_type_index(r, task_type, job_id, record.created_at)

        logger.info(
            "TaskManager | 任务已提交(指定job_id) | job_id={} task={} task_type={} ttl={}s",
            job_id, task_name, task_type, _DEFAULT_TTL_SECONDS,
        )
        return job_id

    @classmethod
    async def create_finished_with_job_id(
        cls,
        job_id: str,
        task_name: str,
        task_type: str ,
        result: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        request: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        直接创建已完成任务记录（不入 ARQ 队列）。

        适用于缓存命中等“无需执行 worker”的场景，避免无意义占用队列与 worker 配额。
        """
        metadata = metadata or {}
        request = request or {}
        now_iso = datetime.now(timezone.utc).isoformat()
        record = TaskRecord(
            job_id=job_id,
            task_name=task_name,
            task_type=task_type,
            status=TaskStatus.SUCCESS,
            created_at=now_iso,
            started_at=now_iso,
            finished_at=now_iso,
            request=request,
            result=result,
            attempts=0,
            metadata=metadata,
        )

        r = await cls._get_redis()
        await r.setex(cls._key(job_id), _DEFAULT_TTL_SECONDS, record.model_dump_json())
        await cls._add_to_type_index(r, task_type, job_id, record.created_at)
        logger.info(
            "TaskManager | 已创建完成任务(无队列) | job_id={} task={} task_type={} ttl={}s",
            job_id,
            task_name,
            task_type,
            _DEFAULT_TTL_SECONDS,
        )
        return job_id

    @classmethod
    async def get(cls, job_id: str) -> Optional[TaskRecord]:
        """
        按 job_id 查询任务记录。

        Returns:
            TaskRecord，或 None（不存在 / 已过期）
        """
        r = await cls._get_redis()
        data = await r.get(cls._key(job_id))
        if data is None:
            return None
        return TaskRecord.model_validate_json(data)

    @classmethod
    async def update(
        cls,
        job_id: str,
        status: TaskStatus,
        result: Any = None,
        error: Optional[str] = None,
        attempts: int = 0,
    ) -> None:
        """
        更新任务状态，由 Worker 在执行各阶段主动调用。

        状态转换:
          RUNNING  → 记录 started_at（仅首次），可携带 result 保存中间检查点
          SUCCESS  → 记录 finished_at + result
          FAILED   → 记录 finished_at + error

        result 字段在任何状态下传入都会被持久化，支持多步骤任务的断点续跑：
          await TaskManager.update(job_id, TaskStatus.RUNNING, result=partial, attempts=n)

        每次更新都会重置 TTL，确保长耗时任务不会在执行期间过期。
        """
        r = await cls._get_redis()
        key = cls._key(job_id)
        data = await r.get(key)

        if data is None:
            logger.warning("TaskManager | 记录不存在，跳过更新 | job_id={}", job_id)
            return

        record = TaskRecord.model_validate_json(data)
        record.status = status
        record.attempts = attempts

        # result 和 error 在任何状态下均持久化（支持中间检查点）
        if result is not None:
            record.result = result
        if error is not None:
            record.error = error

        now = datetime.now(timezone.utc).isoformat()

        if status == TaskStatus.RUNNING:
            # 第一次进入 RUNNING 时设置 started_at
            if record.started_at is None:
                record.started_at = now
        elif status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
            record.finished_at = now

        # 重置 TTL，确保结果不会在返回前过期
        await r.setex(key, _DEFAULT_TTL_SECONDS, record.model_dump_json())

        logger.info(
            "TaskManager | 任务状态更新 | job_id={} status={} attempts={}",
            job_id, status.value, attempts,
        )

    @classmethod
    async def try_start(cls, job_id: str, attempts: int = 1) -> bool:
        """原子地领取待执行任务。

        PENDING/CANCELLED 的竞争由 Redis Lua 一次完成：取消先成功时，本方法
        返回 False，Worker 不得继续调用外部执行器。FAILED 允许 ARQ 重试。
        """
        script = """
        local raw = redis.call('GET', KEYS[1])
        if not raw then return -1 end
        local record = cjson.decode(raw)
        if record.status ~= 'pending' and record.status ~= 'failed' then
            return 0
        end
        record.status = 'running'
        record.attempts = tonumber(ARGV[1])
        if type(record.started_at) ~= 'string' then
            record.started_at = ARGV[2]
        end
        record.finished_at = cjson.null
        record.error = cjson.null
        redis.call('SETEX', KEYS[1], tonumber(ARGV[3]), cjson.encode(record))
        return 1
        """
        r = await cls._get_redis()
        now = datetime.now(timezone.utc).isoformat()
        result = await r.eval(
            script,
            1,
            cls._key(job_id),
            attempts,
            now,
            _DEFAULT_TTL_SECONDS,
        )
        return int(result) == 1

    @classmethod
    async def cancel_pending(cls, job_id: str) -> Optional[TaskStatus]:
        """仅取消尚未被 Worker 领取的任务。

        返回 CANCELLED 表示取消成功；返回其他状态表示任务已开始或已结束；
        返回 None 表示任务不存在。取消记录会保留，便于调用方查询。
        """
        script = """
        local raw = redis.call('GET', KEYS[1])
        if not raw then return -1 end
        local record = cjson.decode(raw)
        if record.status ~= 'pending' then return 0 end
        record.status = 'cancelled'
        record.finished_at = ARGV[1]
        record.error = 'cancelled before execution'
        redis.call('SETEX', KEYS[1], tonumber(ARGV[2]), cjson.encode(record))
        redis.call('ZREM', KEYS[2], ARGV[3])
        redis.call('DEL', KEYS[3], KEYS[4])
        return 1
        """
        r = await cls._get_redis()
        result = await r.eval(
            script,
            4,
            cls._key(job_id),
            default_queue_name,
            job_key_prefix + job_id,
            retry_key_prefix + job_id,
            datetime.now(timezone.utc).isoformat(),
            _DEFAULT_TTL_SECONDS,
            job_id,
        )
        if int(result) == -1:
            return None
        if int(result) == 1:
            logger.info("TaskManager | 待执行任务已取消 | job_id={}", job_id)
            return TaskStatus.CANCELLED
        record = await cls.get(job_id)
        return record.status if record is not None else None

    @classmethod
    async def delete(
        cls,
        job_id: str,
        task_type: Optional[str] = None,
        remove_arq_keys: bool = True,
    ) -> None:
        """
        删除任务记录和可选的 ARQ 去重键。

        Args:
            job_id: 任务 ID
            task_type: 任务类型；为空时尝试从任务记录中读取
            remove_arq_keys: 是否同时删除 ARQ 队列 / 结果 / 去重键
        """
        r = await cls._get_redis()
        key = cls._key(job_id)
        data = await r.get(key)
        resolved_task_type = task_type

        # task_type 索引 key 依赖任务类型；调用方没传时，从现有 TaskRecord 中反查。
        # 即使记录解析失败，也继续删除主记录，避免坏数据阻塞后续同 job_id 提交。
        if data is not None and resolved_task_type is None:
            try:
                resolved_task_type = TaskRecord.model_validate_json(data).task_type
            except Exception:
                logger.exception(
                    "TaskManager | 删除任务前解析记录失败 | job_id={}", job_id
                )

        async with r.pipeline(transaction=True) as pipe:
            delete_keys = [key]
            if remove_arq_keys:
                # ARQ 会用 job/result/retry/in-progress key 判断 _job_id 是否已存在。
                # 删除这些 key 后，指定 job_id 的任务才能被重新提交。
                delete_keys.extend(
                    [
                        job_key_prefix + job_id,
                        result_key_prefix + job_id,
                        retry_key_prefix + job_id,
                        in_progress_key_prefix + job_id,
                    ]
                )
            pipe.delete(*delete_keys)

            if resolved_task_type:
                pipe.zrem(cls._type_index_key(resolved_task_type), job_id)

            if remove_arq_keys:
                # 队列和 abort 集合是 sorted set，需要按 member 移除 job_id。
                pipe.zrem(default_queue_name, job_id)
                pipe.zrem(abort_jobs_ss, job_id)

            await pipe.execute()

        logger.info(
            "TaskManager | 任务记录已删除 | job_id={} task_type={} remove_arq_keys={}",
            job_id, resolved_task_type, remove_arq_keys,
        )

    @classmethod
    async def get_all(cls, task_type: str = "default", limit: int = 100, offset: int = 0) -> list[TaskRecord]:
        """
        获取指定 task_type 的任务记录，按创建时间倒序分页。

        Args:
            task_type: 任务类型筛选
            limit: 返回记录的最大数量
            offset: 偏移量

        Returns:
            任务记录列表
        """
        r = await cls._get_redis()

        job_ids = await r.zrevrange(cls._type_index_key(task_type), offset, offset + limit - 1)

        if not job_ids:
            return []

        records = []
        for job_id in job_ids:
            data = await r.get(cls._key(job_id))
            if data:
                try:
                    record = TaskRecord.model_validate_json(data)
                    records.append(record)
                except Exception as e:
                    logger.error("TaskManager | 解析任务记录失败 | key={} error={}", job_id, e)

        return records
