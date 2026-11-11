from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regression_mvp.utils.api_clients.ade_jobs_client import AdeJobsApiClient


@dataclass(frozen=True)
class AdeJobsAsyncRunResult:
    job_id: str
    actual_items: list[Any]
    submit_raw_response: dict[str, Any]
    submit_status_code: int
    submit_elapsed_seconds: float
    status_raw_response: dict[str, Any]
    status_status_code: int
    status_elapsed_seconds: float
    wait_elapsed_seconds: float
    data_raw_response: dict[str, Any]
    data_status_code: int
    data_elapsed_seconds: float


def run_ade_jobs_async(api_config: dict[str, Any], input_file: Path) -> AdeJobsAsyncRunResult:
    client = AdeJobsApiClient(api_config)
    submit_result = client.submit(input_file)
    wait_start = time.monotonic()
    status_result = client.wait_until_completed(submit_result.job_id)
    data_result = client.get_data(submit_result.job_id)
    return AdeJobsAsyncRunResult(
        job_id=submit_result.job_id,
        actual_items=data_result.items,
        submit_raw_response=submit_result.raw_response,
        submit_status_code=submit_result.status_code,
        submit_elapsed_seconds=submit_result.elapsed_seconds,
        status_raw_response=status_result.raw_response,
        status_status_code=status_result.status_code,
        status_elapsed_seconds=status_result.elapsed_seconds,
        wait_elapsed_seconds=time.monotonic() - wait_start,
        data_raw_response=data_result.raw_response,
        data_status_code=data_result.status_code,
        data_elapsed_seconds=data_result.elapsed_seconds,
    )
