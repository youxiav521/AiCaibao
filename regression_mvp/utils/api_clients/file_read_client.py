from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class UploadResult:
    file_id: str | None
    items: list[Any] | None
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float


class FileReadApiClient:
    def __init__(self, api_config: dict[str, Any]) -> None:
        self.base_url = str(api_config["base_url"]).rstrip("/")
        self.file_read_path = str(api_config["file_read_path"])
        self.result_path_template = str(api_config["result_path_template"])
        self.timeout_seconds = float(api_config.get("timeout_seconds", 300))
        self.poll_interval_seconds = float(api_config.get("poll_interval_seconds", 5))

    def upload_file(self, input_file: Path, *, folder_id: str | None = None) -> UploadResult:
        if not input_file.is_file():
            raise FileNotFoundError(f"input file not found: {input_file}")

        data = {}
        if folder_id:
            data["folderId"] = folder_id

        with input_file.open("rb") as file_obj:
            files = {"file": (input_file.name, file_obj, "application/octet-stream")}
            start_time = time.monotonic()
            response = httpx.post(
                f"{self.base_url}{self.file_read_path}",
                files=files,
                data=data,
                timeout=self.timeout_seconds,
                trust_env=False,
            )
            elapsed_seconds = time.monotonic() - start_time

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"upload failed: status={response.status_code}, url={response.url}, body={response.text}"
            ) from exc
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"upload response is not an object: {body!r}")
        if body.get("success") is not True:
            raise ValueError(f"upload response indicates failure: {body}")

        items = body.get("convert_json")
        if items is not None and not isinstance(items, list):
            raise ValueError("upload response convert_json is not a list")

        file_id = body.get("file_id") or body.get("fileId")
        return UploadResult(
            file_id=str(file_id) if file_id else None,
            items=items,
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )

    def wait_result_items(self, file_id: str) -> list[Any]:
        deadline = time.monotonic() + self.timeout_seconds
        last_body: Any = None

        while time.monotonic() < deadline:
            path = self.result_path_template.format(file_id=file_id)
            response = httpx.get(
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
                trust_env=False,
            )
            response.raise_for_status()
            last_body = response.json()

            if isinstance(last_body, dict) and last_body.get("success") is True:
                payload = last_body.get("json")
                items = json.loads(payload) if isinstance(payload, str) else payload
                if isinstance(items, list):
                    return items
                raise ValueError(f"result items is not a list: {items!r}")

            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"timeout waiting for file_id={file_id}, last_response={last_body!r}")
