from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class FinancialMetadataResult:
    raw_response: dict[str, Any]
    status_code: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FinancialThreeStatementsResult:
    raw_response: list[Any]
    status_code: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FinancialOrgTablesResult:
    raw_response: list[Any]
    status_code: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FinancialMergeTablesResult:
    raw_response: list[Any]
    status_code: int
    elapsed_seconds: float


class FinancialExtractApiClient:
    def __init__(self, api_config: dict[str, Any]) -> None:
        self.base_url = str(api_config["base_url"]).rstrip("/")
        self.metadata_path_template = str(
            api_config.get(
                "financial_metadata_path_template",
                "/api/parsing/extract/financial-metadata/{file_id}",
            )
        )
        self.three_statements_path_template = str(
            api_config.get(
                "financial_three_statements_path_template",
                "/api/parsing/extract/three-statements/{file_id}",
            )
        )
        self.org_tables_path_template = str(
            api_config.get(
                "financial_org_tables_path_template",
                "/api/parsing/extract/org-tables/{file_id}",
            )
        )
        self.merge_tables_path = str(
            api_config.get(
                "financial_merge_tables_path",
                "/api/parsing/extract/merge-tables",
            )
        )
        self.timeout_seconds = float(api_config.get("timeout_seconds", 300))

    def get_financial_metadata(self, file_id: str, *, save: bool = False) -> FinancialMetadataResult:
        path = self.metadata_path_template.format(file_id=file_id)
        start_time = time.monotonic()
        response = httpx.get(
            f"{self.base_url}{path}",
            params={"save": str(save).lower()},
            timeout=self.timeout_seconds,
            trust_env=False,
        )
        elapsed_seconds = time.monotonic() - start_time

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"financial metadata failed: status={response.status_code}, url={response.url}, body={response.text}"
            ) from exc

        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"financial metadata response is not an object: {body!r}")
        if body.get("success") is not True:
            raise ValueError(f"financial metadata response indicates failure: {body}")

        return FinancialMetadataResult(
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )

    def get_three_statements(self, file_id: str) -> FinancialThreeStatementsResult:
        path = self.three_statements_path_template.format(file_id=file_id)
        start_time = time.monotonic()
        response = httpx.get(
            f"{self.base_url}{path}",
            timeout=self.timeout_seconds,
            trust_env=False,
        )
        elapsed_seconds = time.monotonic() - start_time

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"financial three-statements failed: status={response.status_code}, url={response.url}, body={response.text}"
            ) from exc

        body = response.json()
        if not isinstance(body, list):
            raise ValueError(f"financial three-statements response is not a list: {body!r}")

        return FinancialThreeStatementsResult(
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )

    def get_org_tables(self, file_id: str) -> FinancialOrgTablesResult:
        path = self.org_tables_path_template.format(file_id=file_id)
        start_time = time.monotonic()
        response = httpx.get(
            f"{self.base_url}{path}",
            timeout=self.timeout_seconds,
            trust_env=False,
        )
        elapsed_seconds = time.monotonic() - start_time

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"financial org-tables failed: status={response.status_code}, url={response.url}, body={response.text}"
            ) from exc

        body = response.json()
        if not isinstance(body, list):
            raise ValueError(f"financial org-tables response is not a list: {body!r}")

        return FinancialOrgTablesResult(
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )

    def merge_tables(self, file_id: str, merges: list[dict[str, Any]]) -> FinancialMergeTablesResult:
        start_time = time.monotonic()
        response = httpx.post(
            f"{self.base_url}{self.merge_tables_path}",
            json={"fileId": file_id, "merges": merges},
            timeout=self.timeout_seconds,
            trust_env=False,
        )
        elapsed_seconds = time.monotonic() - start_time

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"financial merge-tables failed: status={response.status_code}, url={response.url}, body={response.text}"
            ) from exc

        body = response.json()
        if not isinstance(body, list):
            raise ValueError(f"financial merge-tables response is not a list: {body!r}")

        return FinancialMergeTablesResult(
            raw_response=body,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )
