from __future__ import annotations

import json

from src.applications.content_generate.routers.content_generate_router import (
    redact_callback_payload_for_log,
)


def test_callback_log_redacts_data_uri_but_keeps_audit_metadata() -> None:
    original_data_uri = "data:image/png;base64," + "a" * 64
    payload = {
        "taskId": "task-1",
        "event": "message",
        "data": json.dumps({"answer": {"answerText": original_data_uri}}),
    }

    redacted = redact_callback_payload_for_log(payload)

    logged_answer = json.loads(redacted["data"])["answer"]["answerText"]
    assert original_data_uri not in json.dumps(redacted, ensure_ascii=False)
    assert logged_answer["redacted"] is True
    assert logged_answer["mime"] == "image/png"
    assert logged_answer["chars"] == len(original_data_uri)
    assert len(logged_answer["sha256_12"]) == 12


def test_callback_log_leaves_normal_text_answer_unchanged() -> None:
    payload = {"data": {"answer": {"answerText": "普通文本"}}}

    redacted = redact_callback_payload_for_log(payload)

    assert redacted == payload
