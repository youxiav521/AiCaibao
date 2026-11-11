from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regression_mvp.utils.api_clients.mineru_client import MinerUApiClient


@dataclass(frozen=True)
class MinerUUploadAsyncRunResult:
    file_id: str
    actual_items: list[Any]
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float
    result_raw_response: dict[str, Any]
    result_status_code: int
    result_elapsed_seconds: float


def run_mineru_upload_async(api_config: dict[str, Any], input_file: Path) -> MinerUUploadAsyncRunResult:
    client = MinerUApiClient(api_config)
    upload_result = client.upload_async(input_file)
    result_fetch = client.wait_result(upload_result.file_id)
    return MinerUUploadAsyncRunResult(
        file_id=upload_result.file_id,
        actual_items=result_fetch.items,
        raw_response=upload_result.raw_response,
        status_code=upload_result.status_code,
        elapsed_seconds=upload_result.elapsed_seconds,
        result_raw_response=result_fetch.raw_response,
        result_status_code=result_fetch.status_code,
        result_elapsed_seconds=result_fetch.elapsed_seconds,
    )
