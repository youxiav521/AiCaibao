from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class SubjectMatchResult:
    raw_response: list[Any]
    status_code: int
    elapsed_seconds: float


class SubjectMatchApiClient:
    def __init__(self, api_config: dict[str, Any]) -> None:
        self.base_url = str(api_config["base_url"]).rstrip("/")
        self.subject_match_path = str(
            api_config.get("subject_match_path", "/api/parsing/subject-match/match")
        )
        self.templates_path = str(
            api_config.get("subject_match_templates_path", "/api/parsing/subject-match/templates")
        )
        self.timeout_seconds = float(api_config.get("timeout_seconds", 300))
        self.use_llm = bool(api_config.get("subject_match_use_llm", False))

    def match(self, request_payload: dict[str, Any]) -> SubjectMatchResult:
        start_time = time.monotonic()
        response = httpx.post(
            f"{self.base_url}{self.subject_match_path}",
            params={"use_llm": str(self.use_llm).lower()},
            json=request_payload,
            timeout=self.timeout_seconds,
            trust_env=False,
        )
        elapsed_seconds = time.monotonic() - start_time

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"subject match failed: status={response.status_code}, url={response.url}, body={response.text}"
            ) from exc

        body = response.json()
        if not isinstance(body, list):
            raise ValueError(f"subject match response is not a list: {body!r}")

        return SubjectMatchResult(
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )

    def get_templates(self) -> list[Any]:
        response = httpx.get(
            f"{self.base_url}{self.templates_path}",
            timeout=self.timeout_seconds,
            trust_env=False,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"subject match templates failed: status={response.status_code}, url={response.url}, body={response.text}"
            ) from exc

        body = response.json()
        if not isinstance(body, list):
            raise ValueError(f"subject match templates response is not a list: {body!r}")
        return body
