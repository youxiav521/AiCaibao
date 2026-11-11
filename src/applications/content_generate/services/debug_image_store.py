"""In-memory, local-only storage for viewing image callback results in development."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any


_IMAGE_DATA_URI_PATTERN = re.compile(
    r"^data:image/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+$",
    re.IGNORECASE,
)


class DebugImageStore:
    """Keep a small, expiring set of valid image Data URIs in process memory."""

    def __init__(self, max_entries: int = 3, ttl_seconds: int = 600) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._images: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    def store_callback(self, payload: dict[str, Any]) -> bool:
        task_id = str(payload.get("taskId") or "").strip()
        data_uri = _extract_image_data_uri(payload)
        if not task_id or data_uri is None:
            return False
        now = time.monotonic()
        with self._lock:
            self._prune_expired(now)
            self._images[task_id] = (now, data_uri)
            self._images.move_to_end(task_id)
            while len(self._images) > self._max_entries:
                self._images.popitem(last=False)
        return True

    def get_image(self, task_id: str) -> str | None:
        normalized_task_id = str(task_id or "").strip()
        if not normalized_task_id:
            return None
        now = time.monotonic()
        with self._lock:
            self._prune_expired(now)
            entry = self._images.get(normalized_task_id)
            if entry is None:
                return None
            self._images.move_to_end(normalized_task_id)
            return entry[1]

    def _prune_expired(self, now: float) -> None:
        expired = [
            task_id
            for task_id, (stored_at, _data_uri) in self._images.items()
            if now - stored_at > self._ttl_seconds
        ]
        for task_id in expired:
            self._images.pop(task_id, None)


def _extract_image_data_uri(payload: dict[str, Any]) -> str | None:
    if payload.get("event") != "message":
        return None
    data = payload.get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (TypeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    answer = data.get("answer")
    if not isinstance(answer, dict):
        return None
    answer_text = answer.get("answerText")
    if not isinstance(answer_text, str):
        return None
    return answer_text if _IMAGE_DATA_URI_PATTERN.fullmatch(answer_text) else None
