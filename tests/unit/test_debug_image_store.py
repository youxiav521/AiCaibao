from __future__ import annotations

import json

from src.applications.content_generate.services.debug_image_store import DebugImageStore


def test_message_callback_stores_image_data_uri_by_task_id() -> None:
    store = DebugImageStore(max_entries=2)
    data_uri = "data:image/svg+xml;base64,c3Zn"

    stored = store.store_callback(
        {
            "event": "message",
            "taskId": "debug-image-1",
            "data": json.dumps({"answer": {"answerText": data_uri}}),
        }
    )

    assert stored is True
    assert store.get_image("debug-image-1") == data_uri


def test_non_image_callback_is_not_stored() -> None:
    store = DebugImageStore(max_entries=2)

    stored = store.store_callback(
        {
            "event": "message",
            "taskId": "debug-text-1",
            "data": json.dumps({"answer": {"answerText": "normal answer"}}),
        }
    )

    assert stored is False
    assert store.get_image("debug-text-1") is None
