from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regression_mvp.utils.api_clients.financial_extract_client import FinancialExtractApiClient
from regression_mvp.utils.api_clients.mineru_client import MinerUApiClient


@dataclass(frozen=True)
class FinancialOrgTablesRunResult:
    file_id: str
    actual_response: list[Any]
    mineru_items: list[Any]
    mineru_raw_response: dict[str, Any]
    mineru_status_code: int
    mineru_elapsed_seconds: float
    result_raw_response: dict[str, Any]
    result_status_code: int
    result_elapsed_seconds: float
    org_tables_status_code: int
    org_tables_elapsed_seconds: float


def run_financial_org_tables(
    api_config: dict[str, Any],
    input_file: Path,
) -> FinancialOrgTablesRunResult:
    mineru_client = MinerUApiClient(api_config)
    financial_client = FinancialExtractApiClient(api_config)

    parse_result = mineru_client.parse_to_json(input_file)
    result_fetch = mineru_client.get_result(parse_result.file_id)
    org_tables_result = financial_client.get_org_tables(parse_result.file_id)

    return FinancialOrgTablesRunResult(
        file_id=parse_result.file_id,
        actual_response=org_tables_result.raw_response,
        mineru_items=result_fetch.items,
        mineru_raw_response=parse_result.raw_response,
        mineru_status_code=parse_result.status_code,
        mineru_elapsed_seconds=parse_result.elapsed_seconds,
        result_raw_response=result_fetch.raw_response,
        result_status_code=result_fetch.status_code,
        result_elapsed_seconds=result_fetch.elapsed_seconds,
        org_tables_status_code=org_tables_result.status_code,
        org_tables_elapsed_seconds=org_tables_result.elapsed_seconds,
    )
