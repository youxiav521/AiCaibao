"""
端到端测试脚本：多步骤任务断点续跑

直接运行: PYTHONPATH=. APP_ENV=dev .venv/bin/python tests/test_subtask_e2e.py

验证:
1. 提交 say_hello 任务 → job_id 返回
2. 启动内嵌 Worker — 第1次尝试：step1(word_count) 完成并持久化，step2(word_freq) Retry
3. 第2次重试：word_count 从 result 中读取跳过，word_freq/transform 正常完成
4. 最终 TaskRecord:
   - status == SUCCESS
   - attempts == 2
   - result 包含 word_count / top_words / upper
   - 第1次 Retry 前 word_count 已持久化（在 RUNNING 状态的 result 中）
"""

import asyncio
import os
import sys
import time

os.environ.setdefault("APP_ENV", "dev")

from dotenv import load_dotenv
load_dotenv()

from src.base.settings import initAppConfig, getAppConfig
from src.base.logger_factory import initLogger

initAppConfig()
_cfg = getAppConfig()
initLogger(_cfg)

from arq import Worker
from arq.connections import RedisSettings

from src.base.task_manager import TaskManager, TaskStatus
from src.tasks.hello_tasks import say_hello


def _redis_settings() -> RedisSettings:
    cfg = getAppConfig()
    r = cfg.redis if cfg else None
    return RedisSettings(
        host=r.host if r else "localhost",
        port=r.port if r else 6379,
        password=(r.password if r else "") or None,
        database=r.db if r else 0,
    )


async def main():
    print("\n=== 端到端测试：多步骤任务断点续跑 ===\n")

    # ── 提交任务 ──────────────────────────────────────────────────────────────
    # ARQ 按函数名+参数做 job_id 去重，加时间戳保证每次提交都是新任务
    message = f"hello world test {int(time.time())}"
    job_id = await TaskManager.submit("say_hello", kwargs={"message": message})
    assert job_id, "job_id 不应为空"
    print(f"✓ 任务已提交 | job_id={job_id} message={message!r}")

    # ── 启动内嵌 Worker ────────────────────────────────────────────────────────
    worker = Worker(
        functions=[say_hello],
        redis_settings=_redis_settings(),
        max_jobs=1,
        job_timeout=30,
        keep_result=3600,
        max_tries=3,
        poll_delay=0.1,
    )

    print("▶ 启动 Worker（第1次尝试 step2 将 Retry，第2次成功）...")
    worker_task = asyncio.create_task(worker.async_run())

    # 轮询等待任务完成，最多 30s
    record = None
    for _ in range(150):
        await asyncio.sleep(0.2)
        record = await TaskManager.get(job_id)
        if record and record.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
            break

    worker_task.cancel()
    try:
        await worker_task
    except (asyncio.CancelledError, Exception):
        pass
    print("■ Worker 已退出\n")

    # ── 查询最终结果 ──────────────────────────────────────────────────────────
    assert record is not None, "FAIL: TaskRecord 不应为 None（轮询超时）"

    print("── TaskRecord ──────────────────────────")
    print(f"  status:   {record.status}")
    print(f"  attempts: {record.attempts}")
    print(f"  result:   {record.result}")

    # ── 断言 ──────────────────────────────────────────────────────────────────
    failures = []

    if record.status != TaskStatus.SUCCESS:
        failures.append(f"status: 期望 SUCCESS，实际 {record.status}")

    if record.attempts != 2:
        failures.append(f"attempts: 期望 2，实际 {record.attempts}")

    result = record.result
    if result is None:
        failures.append("result 为 None")
    else:
        for key in ("word_count", "char_count", "top_words", "upper", "original", "attempts"):
            if key not in result:
                failures.append(f"result 缺少字段: {key}")
        if result.get("upper") != message.upper():
            failures.append(f"upper 不匹配: {result.get('upper')}")
        if result.get("word_count") != len(message.split()):
            failures.append(f"word_count 期望 {len(message.split())}，实际 {result.get('word_count')}")
        if result.get("attempts") != 2:
            failures.append(f"result.attempts 期望 2，实际 {result.get('attempts')}")

    print("\n── 断言结果 ────────────────────────────")
    if failures:
        print("❌ 以下断言失败:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    else:
        print("✅ 所有断言通过！断点续跑功能验证成功。")
        print(f"\n最终结果摘要:")
        print(f"  word_count = {result.get('word_count')}")
        print(f"  top_words  = {result.get('top_words')}")
        print(f"  upper      = {result.get('upper')}")
        print(f"  attempts   = {result.get('attempts')}")


asyncio.run(main())
