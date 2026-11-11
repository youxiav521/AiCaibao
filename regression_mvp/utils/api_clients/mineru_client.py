from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class MinerUParseResult:
    file_id: str
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float


@dataclass(frozen=True)
class MinerUResultFetchResult:
    items: list[Any]
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float


class MinerUApiClient:
    def __init__(self, api_config: dict[str, Any]) -> None:
        self.base_url = str(api_config["base_url"]).rstrip("/")
        self.upload_path = str(api_config.get("mineru_upload_path", "/api/parsing/mineru/upload"))
        self.to_json_path = str(api_config.get("mineru_to_json_path", "/api/parsing/mineru/to-json"))
        self.result_path_template = str(api_config["result_path_template"])
        self.timeout_seconds = float(api_config.get("timeout_seconds", 300))
        self.request_timeout_seconds = _optional_timeout(
            api_config.get("mineru_request_timeout_seconds", self.timeout_seconds)
        )
        self.wait_timeout_seconds = float(api_config.get("mineru_wait_timeout_seconds", self.timeout_seconds))
        self.poll_interval_seconds = float(api_config.get("poll_interval_seconds", 5))

    def upload_async(self, input_file: Path) -> MinerUParseResult:
        return self._upload_for_file_id(input_file, self.upload_path, "mineru upload")

    def parse_to_json(self, input_file: Path) -> MinerUParseResult:
        return self._upload_for_file_id(input_file, self.to_json_path, "mineru to-json")

    def _upload_for_file_id(self, input_file: Path, path: str, label: str) -> MinerUParseResult:
        if not input_file.is_file():
            raise FileNotFoundError(f"input file not found: {input_file}")

        with input_file.open("rb") as file_obj:
            files = {"file": (input_file.name, file_obj, "application/octet-stream")}
            start_time = time.monotonic()
            response = httpx.post(
                f"{self.base_url}{path}",
                files=files,
                timeout=self.request_timeout_seconds,
                trust_env=False,
            )
            elapsed_seconds = time.monotonic() - start_time

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"{label} failed: status={response.status_code}, url={response.url}, body={response.text}"
            ) from exc

        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"{label} response is not an object: {body!r}")
        if body.get("success") is not True:
            raise ValueError(f"{label} response indicates failure: {body}")

        file_id = body.get("file_id") or body.get("fileId")
        if not file_id:
            raise ValueError(f"{label} response missing file_id: {body}")

        return MinerUParseResult(
            file_id=str(file_id),
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )

    def get_result(self, file_id: str, *, require_success: bool = True) -> MinerUResultFetchResult | None:
        path = self.result_path_template.format(file_id=file_id)
        start_time = time.monotonic()
        response = httpx.get(
            f"{self.base_url}{path}",
            timeout=self.request_timeout_seconds,
            trust_env=False,
        )
        elapsed_seconds = time.monotonic() - start_time
        response.raise_for_status()
        body = response.json()

        if not isinstance(body, dict):
            raise ValueError(f"mineru result response is not an object: {body!r}")
        if body.get("success") is not True:
            if not require_success:
                return None
            raise ValueError(f"mineru result response indicates failure: {body}")

        payload = body.get("json")
        items = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(items, list):
            raise ValueError(f"mineru result items is not a list: {items!r}")
        return MinerUResultFetchResult(
            items=items,
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )

    def get_result_items(self, file_id: str) -> list[Any]:
        result = self.get_result(file_id)
        if result is None:
            raise ValueError(f"mineru result is not ready: {file_id}")
        return result.items

    def wait_result(self, file_id: str) -> MinerUResultFetchResult:
        deadline = time.monotonic() + self.wait_timeout_seconds
        while time.monotonic() < deadline:
            result = self.get_result(file_id, require_success=False)
            if result is not None:
                return result
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"timeout waiting for mineru result: file_id={file_id}")


def _optional_timeout(value: Any) -> float | None:
    timeout_seconds = float(value)
    if timeout_seconds <= 0:
        return None
    return timeout_seconds
