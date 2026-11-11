from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regression_mvp.utils.api_clients.financial_extract_client import FinancialExtractApiClient
from regression_mvp.utils.api_clients.mineru_client import MinerUApiClient


@dataclass(frozen=True)
class FinancialMetadataRunResult:
    file_id: str
    actual_response: dict[str, Any]
    mineru_items: list[Any]
    mineru_raw_response: dict[str, Any]
    mineru_status_code: int
    mineru_elapsed_seconds: float
    result_raw_response: dict[str, Any]
    result_status_code: int
    result_elapsed_seconds: float
    metadata_status_code: int
    metadata_elapsed_seconds: float


def run_financial_metadata(api_config: dict[str, Any], input_file: Path) -> FinancialMetadataRunResult:
    mineru_client = MinerUApiClient(api_config)
    financial_client = FinancialExtractApiClient(api_config)

    parse_result = mineru_client.parse_to_json(input_file)
    result_fetch = mineru_client.get_result(parse_result.file_id)
    metadata_result = financial_client.get_financial_metadata(parse_result.file_id, save=False)

    return FinancialMetadataRunResult(
        file_id=parse_result.file_id,
        actual_response=metadata_result.raw_response,
        mineru_items=result_fetch.items,
        mineru_raw_response=parse_result.raw_response,
        mineru_status_code=parse_result.status_code,
        mineru_elapsed_seconds=parse_result.elapsed_seconds,
        result_raw_response=result_fetch.raw_response,
        result_status_code=result_fetch.status_code,
        result_elapsed_seconds=result_fetch.elapsed_seconds,
        metadata_status_code=metadata_result.status_code,
        metadata_elapsed_seconds=metadata_result.elapsed_seconds,
    )
