import asyncio
import importlib.util
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CALLBACK_SERVICE_PATH = (
    _PROJECT_ROOT / "src/applications/content_generate/tasks/callback_service.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "callback_service_under_test", _CALLBACK_SERVICE_PATH
)
assert _SPEC is not None
assert _SPEC.loader is not None
callback_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(callback_module)
CallbackService = callback_module.CallbackService


async def _reset_callback_service() -> None:
    tasks = list(CallbackService._consumer_tasks.values())
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    with CallbackService._lock:
        CallbackService._queues.clear()
        CallbackService._consumer_tasks.clear()
        CallbackService._interrupt_events.clear()
        CallbackService._ending_tasks.clear()


@pytest.fixture(autouse=True)
async def clean_callback_service_state() -> None:
    await _reset_callback_service()
    yield
    await _reset_callback_service()


@pytest.mark.asyncio
async def test_cleanup_timeout_does_not_make_consumer_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "TASK-CLEANUP-RACE"
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    wait_for_started = asyncio.Event()
    release_wait_for = asyncio.Event()
    real_wait_for = asyncio.wait_for
    log_records: list[str] = []
    wait_for_calls = 0

    async def fake_send_one(
        cls: type[CallbackService],
        event: str,
        data: dict[str, Any],
        tid: str,
        callback_url: str | None = None,
    ) -> bool:
        send_started.set()
        await release_send.wait()
        return False

    async def block_then_timeout(awaitable: Any, timeout: float) -> Any:
        nonlocal wait_for_calls
        wait_for_calls += 1
        wait_for_started.set()
        await release_wait_for.wait()
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise asyncio.TimeoutError

    sink_id = callback_module.logger.add(
        lambda message: log_records.append(message.record["message"]),
        level="ERROR",
    )
    monkeypatch.setattr(CallbackService, "_send_one", classmethod(fake_send_one))

    try:
        CallbackService.send("message", {"value": 1}, task_id)
        await real_wait_for(send_started.wait(), timeout=1.0)

        monkeypatch.setattr(callback_module.asyncio, "wait_for", block_then_timeout)
        first_end = asyncio.create_task(CallbackService.signal_task_end(task_id))
        await real_wait_for(wait_for_started.wait(), timeout=1.0)

        queue = CallbackService._queues[task_id]
        await CallbackService.signal_task_end(task_id)
        assert queue.qsize() == 1
        assert wait_for_calls == 1

        release_wait_for.set()
        await first_end
    finally:
        release_send.set()
        callback_module.logger.remove(sink_id)

    assert not any("CallbackService consumer异常退出" in msg for msg in log_records)
