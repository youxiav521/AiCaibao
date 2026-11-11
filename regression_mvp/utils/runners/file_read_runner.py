from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regression_mvp.utils.api_clients.file_read_client import FileReadApiClient
from regression_mvp.utils.item_loader import resolve_actual_items


@dataclass(frozen=True)
class FileReadRunResult:
    file_id: str | None
    actual_items: list[Any]
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float


def run_file_read(
    api_config: dict[str, Any],
    input_file: Path,
    *,
    folder_id: str | None = None,
) -> FileReadRunResult:
    client = FileReadApiClient(api_config)
    upload_result = client.upload_file(input_file, folder_id=folder_id)
    actual_items = resolve_actual_items(client, upload_result)
    return FileReadRunResult(
        file_id=upload_result.file_id,
        actual_items=actual_items,
        raw_response=upload_result.raw_response,
        status_code=upload_result.status_code,
        elapsed_seconds=upload_result.elapsed_seconds,
    )
