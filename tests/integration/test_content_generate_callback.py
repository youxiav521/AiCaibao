"""
内容生成回调接口集成测试（真实 HTTP + 真实数据集）。

保留的测试：
- 1 个连通性冒烟测试
- 4 个按题型划分的 E2E 测试：text_plain、text_custom、table、text_json

Set-Location "e:\yusp-202506-Work\yusys_ai"; .\.venv\Scripts\python.exe -m pytest tests\integration\test_content_generate_callback.py
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest
from tests.fixtures.content_generate.poc1226_notebook_question_tasks import (
    QUESTION_TABLE,
    QUESTION_TEXT_CUSTOM,
    QUESTION_TEXT_JSON,
    QUESTION_TEXT_PLAIN,
)

_project_root = Path(__file__).resolve().parents[2]
_api_base_url = os.environ.get("CONTENT_GENERATE_API_BASE_URL", "http://127.0.0.1:6300")

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture
async def content_generate_api_base_url() -> str:
    def _get_openapi() -> httpx.Response:
        with httpx.Client(base_url=_api_base_url, timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            return client.get("/openapi.json")

    try:
        health = await asyncio.to_thread(_get_openapi)
    except httpx.HTTPError as exc:
        pytest.skip(f"无法连接到 {_api_base_url}，请先启动服务。error={exc}")
    if health.status_code >= 500:
        pytest.skip(f"服务 {_api_base_url} 不可用。openapi status={health.status_code}")
    return _api_base_url


def _callback_handler_factory(
    recorded: List[Dict[str, Any]],
    interrupt_on_start: bool = False,
) -> type[BaseHTTPRequestHandler]:
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
                event = body.get("event")
                data_raw = body.get("data", "{}")
                task_id = body.get("taskId")
                try:
                    data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
                except json.JSONDecodeError:
                    data = data_raw
                recorded.append({"event": event, "data": data, "taskId": task_id})

                interrupt = bool(interrupt_on_start and event == "start")
                resp = {
                    "code": "0",
                    "total": 0,
                    "message": None,
                    "level": "info",
                    "data": {"taskId": task_id, "interrupt": interrupt},
                    "i18nData": None,
                    "extData": None,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception:
                self.send_response(500)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return CallbackHandler


def _start_callback_server(
    interrupt_on_start: bool = False,
) -> Tuple[HTTPServer, int, List[Dict[str, Any]], threading.Thread]:
    recorded: List[Dict[str, Any]] = []
    handler_cls = _callback_handler_factory(recorded, interrupt_on_start=interrupt_on_start)
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    _, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(port), recorded, thread


def _shutdown_callback_server(server: HTTPServer) -> None:
    server.shutdown()
    server.server_close()


async def _wait_for_terminal_callback_event(
    recorded: List[Dict[str, Any]],
    *,
    timeout_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if recorded and recorded[-1]["event"] in ("complete", "error"):
            return str(recorded[-1]["event"])
        await asyncio.sleep(0.2)
    raise AssertionError(
        f"等待 complete/error 超时（{timeout_s}s）| events={[e.get('event') for e in recorded]}"
    )


async def _post_content_generate_callback(
    api_base_url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[httpx.Timeout] = None,
) -> httpx.Response:
    def _post() -> httpx.Response:
        with httpx.Client(
            base_url=api_base_url,
            timeout=timeout or httpx.Timeout(600.0, connect=15.0),
        ) as client:
            return client.post("/api/content_generate/callback", json=payload, headers=headers)

    return await asyncio.to_thread(_post)


@pytest.fixture(scope="module")
def poc1226_content_list() -> List[Dict[str, Any]]:
    merged_path = _project_root / "storage" / "merged_converted_output_poc1226.json"
    if not merged_path.is_file():
        pytest.skip("缺少 storage/merged_converted_output_poc1226.json")
    with merged_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        pytest.skip("merged_converted_output_poc1226.json 为空或格式不正确")
    return data


def _strip_markdown_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\\n".join(lines[1:-1]).strip()
    return s


async def _run_single_question_e2e(
    *,
    api_base_url: str,
    content_list: List[Dict[str, Any]],
    question_task: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 1200.0,
) -> Dict[str, Any]:
    server, port, recorded, _thr = _start_callback_server(interrupt_on_start=False)
    try:
        task_id = f"TEST_{uuid.uuid4().hex[:8]}"
        payload = {
            "contentList": content_list,
            "questionTasks": [question_task],
            "taskId": task_id,
            "callbackUrl": f"http://127.0.0.1:{port}/callback",
        }
        resp = await _post_content_generate_callback(
            api_base_url,
            payload,
            headers=headers,
            timeout=httpx.Timeout(600.0, connect=15.0),
        )
        assert resp.status_code == 200, resp.text
        submit = resp.json()
        assert submit.get("success") is True
        assert submit.get("taskId") == task_id

        terminal = await _wait_for_terminal_callback_event(recorded, timeout_s=timeout_s)
        assert terminal == "complete", f"E2E 未成功完成：last={recorded[-1]!r}"

        events = [e["event"] for e in recorded]
        assert events[0] == "start"
        assert events[-1] == "complete"
        messages = [e for e in recorded if e["event"] == "message"]
        assert len(messages) == 1

        msg = messages[0]["data"] # 单个问题测试所以只应该有一个
        assert msg.get("questionId") == question_task["id"]
        assert msg.get("questionType") == question_task["type"]

        answer = msg.get("answer", {})
        assert isinstance(answer, dict)
        answer_text = str(answer.get("answerText", "")).strip()
        assert answer_text != ""

        complete = recorded[-1]["data"]
        assert complete.get("success") is True
        assert complete.get("completed_questions_counts", 0) >= 1
        return msg
    finally:
        _shutdown_callback_server(server)


@pytest.mark.integration
async def test_content_generate_connectivity(content_generate_api_base_url: str) -> None:
    assert content_generate_api_base_url.startswith("http")


@pytest.mark.integration
async def test_content_generate_e2e_text_plain(
    content_generate_api_base_url: str,
    poc1226_content_list: List[Dict[str, Any]],
) -> None:
    msg = await _run_single_question_e2e(
        api_base_url=content_generate_api_base_url,
        content_list=poc1226_content_list,
        question_task=QUESTION_TEXT_PLAIN,
    )
    answer_text = str(msg["answer"].get("answerText", "")).strip()
    assert "error" not in answer_text.lower()


@pytest.mark.integration
async def test_content_generate_e2e_text_custom(
    content_generate_api_base_url: str,
    poc1226_content_list: List[Dict[str, Any]],
) -> None:
    msg = await _run_single_question_e2e(
        api_base_url=content_generate_api_base_url,
        content_list=poc1226_content_list,
        question_task=QUESTION_TEXT_CUSTOM,
        headers={"Accept-Language": "en_US"},
    )
    answer_text = str(msg["answer"].get("answerText", ""))
    assert "error" not in answer_text.lower()


@pytest.mark.integration
async def test_content_generate_e2e_table(
    content_generate_api_base_url: str,
    poc1226_content_list: List[Dict[str, Any]],
) -> None:
    msg = await _run_single_question_e2e(
        api_base_url=content_generate_api_base_url,
        content_list=poc1226_content_list,
        question_task=QUESTION_TABLE,
    )
    answer_text = str(msg["answer"].get("answerText", "")).strip()
    assert "error" not in answer_text.lower()


@pytest.mark.integration
async def test_content_generate_e2e_text_json(
    content_generate_api_base_url: str,
    poc1226_content_list: List[Dict[str, Any]],
) -> None:
    msg = await _run_single_question_e2e(
        api_base_url=content_generate_api_base_url,
        content_list=poc1226_content_list,
        question_task=QUESTION_TEXT_JSON,
    )
    answer_text = _strip_markdown_json_fence(str(msg["answer"].get("answerText", "")))
    parsed = json.loads(answer_text)
    assert isinstance(parsed, (list, dict))
