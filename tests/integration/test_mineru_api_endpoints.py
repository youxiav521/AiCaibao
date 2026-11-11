"""
MinerU API 接口测试
测试 API 接口的基本功能、参数校验、响应结构等，不需要启动 ARQ Worker
"""
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.mark.integration
async def test_create_job(client, sample_pdf):
    """POST /api/ade/parse/jobs 应返回含 job_id 的 200 响应。"""
    resp = await client.post(
        "/api/ade/parse/jobs",
        files={"document": ("test.pdf", sample_pdf, "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["job_id"]  # 非空字符串


@pytest.mark.integration
async def test_get_job_detail_structure(client, sample_pdf):
    """创建任务后立即查询，验证响应字段结构和类型正确。"""
    create_resp = await client.post(
        "/api/ade/parse/jobs",
        files={"document": ("test.pdf", sample_pdf, "application/pdf")},
    )
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]

    detail_resp = await client.get(f"/api/ade/parse/jobs/{job_id}")
    assert detail_resp.status_code == 200

    body = detail_resp.json()
    # 字段存在性
    assert body["job_id"] == job_id
    assert body["status"] in ("pending", "running", "success", "failed")
    assert isinstance(body["received_at"], int) and body["received_at"] > 0
    assert 0.0 <= body["progress"] <= 1.0


@pytest.mark.integration
async def test_get_job_detail_404(client):
    """查询不存在的 job_id 应返回 404。"""
    resp = await client.get("/api/ade/parse/jobs/nonexistent-job-id-000")
    assert resp.status_code == 404
