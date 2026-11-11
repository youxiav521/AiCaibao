from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class AdeJobSubmitResult:
    job_id: str
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float


@dataclass(frozen=True)
class AdeJobStatusResult:
    job_id: str
    status: str
    error_message: str | None
    progress: float | None
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float


@dataclass(frozen=True)
class AdeJobDataResult:
    items: list[Any]
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float


class AdeJobsApiClient:
    """Synchronous client for the ADE submit -> status -> data workflow."""

    def __init__(self, api_config: dict[str, Any]) -> None:
        self.base_url = str(api_config["base_url"]).rstrip("/")
        self.submit_path = str(api_config.get("ade_jobs_submit_path", "/openapi/ade/jobs/submit"))
        self.status_path_template = str(
            api_config.get("ade_jobs_status_path_template", "/openapi/ade/jobs/{job_id}")
        )
        self.data_path_template = str(
            api_config.get("ade_jobs_data_path_template", "/openapi/ade/jobs/{job_id}/data")
        )
        timeout_seconds = float(api_config.get("timeout_seconds", 300))
        self.submit_request_timeout_seconds = _optional_timeout(
            api_config.get("ade_jobs_request_timeout_seconds", timeout_seconds)
        )
        self.status_request_timeout_seconds = _optional_timeout(
            api_config.get("ade_jobs_status_request_timeout_seconds", 30)
        )
        self.wait_timeout_seconds = float(api_config.get("ade_jobs_wait_timeout_seconds", timeout_seconds))
        self.poll_interval_seconds = float(api_config.get("poll_interval_seconds", 5))

    def submit(self, input_file: Path) -> AdeJobSubmitResult:
        if not input_file.is_file():
            raise FileNotFoundError(f"input file not found: {input_file}")

        file_hash = _sha256(input_file)
        with input_file.open("rb") as file_obj:
            start_time = time.monotonic()
            response = httpx.post(
                f"{self.base_url}{self.submit_path}",
                files={"file": (input_file.name, file_obj, "application/octet-stream")},
                data={"file_hash": file_hash},
                timeout=self.submit_request_timeout_seconds,
                trust_env=False,
            )
            elapsed_seconds = time.monotonic() - start_time

        body = _success_body(response, "ADE job submit")
        data = _response_data(body, "ADE job submit")
        job_id = data.get("job_id")
        if not job_id:
            raise ValueError(f"ADE job submit response missing job_id: {body}")
        return AdeJobSubmitResult(str(job_id), body, response.status_code, elapsed_seconds)

    def get_status(self, job_id: str) -> AdeJobStatusResult:
        response, elapsed_seconds = self._get(
            self.status_path_template.format(job_id=job_id), "ADE job status"
        )
        body = _success_body(response, "ADE job status")
        data = _response_data(body, "ADE job status")
        status = str(data.get("status") or "").lower()
        if status not in {"pending", "processing", "completed", "failed"}:
            raise ValueError(f"ADE job status is invalid: {data}")
        progress = data.get("progress")
        return AdeJobStatusResult(
            job_id=str(data.get("job_id") or job_id),
            status=status,
            error_message=_optional_string(data.get("error_message")),
            progress=float(progress) if isinstance(progress, (int, float)) else None,
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )

    def wait_until_completed(self, job_id: str) -> AdeJobStatusResult:
        deadline = time.monotonic() + self.wait_timeout_seconds
        while time.monotonic() < deadline:
            result = self.get_status(job_id)
            if result.status == "completed":
                return result
            if result.status == "failed":
                detail = result.error_message or "no error_message returned"
                raise RuntimeError(f"ADE job failed: job_id={job_id}, error_message={detail}")
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"timeout waiting for ADE job: job_id={job_id}")

    def get_data(self, job_id: str) -> AdeJobDataResult:
        response, elapsed_seconds = self._get(
            self.data_path_template.format(job_id=job_id), "ADE job data"
        )
        body = _success_body(response, "ADE job data")
        data = _response_data(body, "ADE job data")
        content_list = data.get("content_list")
        if not isinstance(content_list, dict):
            raise ValueError(f"ADE job data response missing content_list: {body}")
        items = content_list.get("content")
        if not isinstance(items, list):
            raise ValueError(f"ADE job data response content_list.content is not a list: {items!r}")
        return AdeJobDataResult(items, body, response.status_code, elapsed_seconds)

    def _get(self, path: str, label: str) -> tuple[httpx.Response, float]:
        start_time = time.monotonic()
        response = httpx.get(
            f"{self.base_url}{path}",
            timeout=self.status_request_timeout_seconds,
            trust_env=False,
        )
        return response, time.monotonic() - start_time


def _success_body(response: httpx.Response, label: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"{label} failed: status={response.status_code}, url={response.url}, body={response.text}"
        ) from exc
    try:
        body = response.json()
    except ValueError as exc:
        raise ValueError(f"{label} response is not JSON: {response.text}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"{label} response is not an object: {body!r}")
    if body.get("code") != 200:
        raise ValueError(f"{label} response indicates failure: {body}")
    return body


def _response_data(body: dict[str, Any], label: str) -> dict[str, Any]:
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{label} response data is not an object: {body}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_timeout(value: Any) -> float | None:
    timeout_seconds = float(value)
    return timeout_seconds if timeout_seconds > 0 else None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None
