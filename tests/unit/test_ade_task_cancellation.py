import json
import os
import sys
import types
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("REDIS_HOST", "localhost")

# ADE worker imports the full MinerU stack at module load. Cancellation exits before
# constructing it, so keep this unit test isolated from unrelated LLM configuration.
mineru_module = types.ModuleType(
    "src.applications.parsing.service.parse_minerU_service"
)
mineru_module.MinerUParsingService = type("MinerUParsingService", (), {})
sys.modules[mineru_module.__name__] = mineru_module

from src.ade.jobs.router import jobs_router  # noqa: E402
from src.ade.jobs.schema.jobs_schema import CancelJobResponse, JobStatus  # noqa: E402
from src.ade.jobs.service.jobs_cache_store import JobsCacheStore  # noqa: E402
from src.ade.jobs.service.jobs_service import AdeJobsService  # noqa: E402
from src.ade.jobs.tasks.ade_jobs_tasks import process_ade_jobs  # noqa: E402
from src.base.task_manager import TaskManager, TaskRecord, TaskStatus  # noqa: E402


class FakeRedis:
    def __init__(self, record: TaskRecord):
        self.raw = record.model_dump_json()
        self.queue = {record.job_id}

    async def get(self, key):
        return self.raw

    async def eval(self, script, numkeys, *args):
        record = json.loads(self.raw) if self.raw else None
        if record is None:
            return -1
        if numkeys == 4:
            if record["status"] != "pending":
                return 0
            record["status"] = "cancelled"
            record["finished_at"] = args[4]
            record["error"] = "cancelled before execution"
            self.queue.discard(args[6])
            self.raw = json.dumps(record)
            return 1
        if record["status"] not in {"pending", "failed"}:
            return 0
        record["status"] = "running"
        record["attempts"] = int(args[1])
        record["started_at"] = args[2]
        record["finished_at"] = None
        record["error"] = None
        self.raw = json.dumps(record)
        return 1


def make_record(job_id="job-1"):
    return TaskRecord(
        job_id=job_id,
        task_name="process_ade_jobs",
        task_type="ade_jobs",
        status=TaskStatus.PENDING,
        created_at="2026-08-04T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_cancel_wins_before_worker_claim(monkeypatch):
    redis = FakeRedis(make_record())
    monkeypatch.setattr(TaskManager, "_get_redis", AsyncMock(return_value=redis))

    assert await TaskManager.cancel_pending("job-1") == TaskStatus.CANCELLED
    assert await TaskManager.try_start("job-1") is False
    assert "job-1" not in redis.queue
    assert (await TaskManager.get("job-1")).status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_worker_claim_wins_before_cancel(monkeypatch):
    redis = FakeRedis(make_record())
    monkeypatch.setattr(TaskManager, "_get_redis", AsyncMock(return_value=redis))

    assert await TaskManager.try_start("job-1") is True
    assert await TaskManager.cancel_pending("job-1") == TaskStatus.RUNNING
    assert (await TaskManager.get("job-1")).status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_cancelled_worker_never_downloads_or_calls_mineru(monkeypatch):
    monkeypatch.setattr(TaskManager, "try_start", AsyncMock(return_value=False))

    result = await process_ade_jobs(
        {"job_id": "job-1", "job_try": 1},
        file_path="ade-jobs/uploads/report.pdf",
        filename="report.pdf",
        external_file_id="file-1",
        file_hash="hash",
        backend="pipeline/hybrid-http-client",
        ang="ch",
        method="auto",
        formula_enable=True,
        table_enable=True,
        callback_url=None,
    )

    assert result == {"cancelled": True, "job_id": "job-1"}


@pytest.mark.asyncio
async def test_cancel_endpoint_reports_success_and_conflict(monkeypatch):
    monkeypatch.setattr(
        jobs_router._jobs_service,
        "cancel_pending",
        AsyncMock(return_value=JobStatus.cancelled),
    )
    response = await jobs_router.cancel_job("job-1")
    assert isinstance(response, CancelJobResponse)
    assert response.data.status == JobStatus.cancelled

    jobs_router._jobs_service.cancel_pending = AsyncMock(
        return_value=JobStatus.processing
    )
    conflict = await jobs_router.cancel_job("job-1")
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_cancel_service_marks_hash_cache_cancelled(monkeypatch):
    record = make_record()
    record.metadata = {"file_hash": "hash-1"}
    monkeypatch.setattr(TaskManager, "get", AsyncMock(return_value=record))
    monkeypatch.setattr(
        TaskManager,
        "cancel_pending",
        AsyncMock(return_value=TaskStatus.CANCELLED),
    )
    update_cache = AsyncMock()
    monkeypatch.setattr(JobsCacheStore, "update_status", update_cache)

    assert await AdeJobsService().cancel_pending("job-1") == JobStatus.cancelled
    update_cache.assert_awaited_once_with("hash-1", "cancelled")
