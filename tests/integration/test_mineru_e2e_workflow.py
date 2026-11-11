"""
MinerU 端到端工作流测试
测试完整的任务提交、执行、结果返回流程，需要启动 ARQ Worker
"""
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

POLL_INTERVAL_S = 3
MAX_WAIT_S = 30

# ─── Fixture 辅助函数 ────────────────────────────────────────────────────────────────

@pytest.fixture
def poll_job(max_wait_s=MAX_WAIT_S, poll_interval_s=POLL_INTERVAL_S):
    """返回轮询任务状态的函数，可直接在测试中使用
    
    Usage:
        async def test_something(poll_job, client):
            job_id = "xxx"
            result = await poll_job(client, job_id)
    """
    import asyncio
    import time
    
    async def _poll_job(client, job_id, max_wait=max_wait_s, poll_interval=poll_interval_s):
        deadline = time.monotonic() + max_wait
        last_body: dict | None = None

        while time.monotonic() < deadline:
            poll_resp = await client.get(f"/api/ade/parse/jobs/{job_id}")
            assert poll_resp.status_code == 200
            last_body = poll_resp.json()
            if last_body["status"] in ("success", "failed"):
                break
            await asyncio.sleep(poll_interval)
        
        return last_body
    
    return _poll_job

@pytest.mark.integration
async def test_job_completes_within_60s(client, arq_worker, sample_pdf, poll_job):
    """端到端：Worker 启动后，提交任务并轮询，最多 60s 必须到达 success/failed 终态。"""
    # 1. 提交任务
    resp = await client.post(
        "/api/ade/parse/jobs",
        files={"document": ("test.pdf", sample_pdf, "application/pdf")},
        params={"model": "vlm", "language": "ch"},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    # 2. 轮询直到终态或超时
    last_body = await poll_job(client, job_id)

    assert last_body is not None
    assert last_body["status"] in ("success", "failed"), (
        f"任务 {job_id} 在 30s 内未到达终态，最后状态: {last_body}"
    )
