from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regression_mvp.utils.api_clients.financial_extract_client import FinancialExtractApiClient
from regression_mvp.utils.api_clients.mineru_client import MinerUApiClient


@dataclass(frozen=True)
class FinancialMergeTablesRunResult:
    file_id: str
    actual_response: list[Any]
    mineru_items: list[Any]
    org_tables: list[Any]
    merge_request: dict[str, Any]
    mineru_raw_response: dict[str, Any]
    mineru_status_code: int
    mineru_elapsed_seconds: float
    result_raw_response: dict[str, Any]
    result_status_code: int
    result_elapsed_seconds: float
    org_tables_status_code: int
    org_tables_elapsed_seconds: float
    merge_tables_status_code: int
    merge_tables_elapsed_seconds: float


def run_financial_merge_tables(
    api_config: dict[str, Any],
    input_file: Path,
    *,
    source_indices: list[int] | None = None,
    merge_text: str | None = None,
) -> FinancialMergeTablesRunResult:
    mineru_client = MinerUApiClient(api_config)
    financial_client = FinancialExtractApiClient(api_config)

    parse_result = mineru_client.parse_to_json(input_file)
    result_fetch = mineru_client.get_result(parse_result.file_id)
    org_tables_result = financial_client.get_org_tables(parse_result.file_id)
    merge_request = build_merge_request(
        parse_result.file_id,
        org_tables_result.raw_response,
        source_indices=source_indices,
        merge_text=merge_text,
    )
    merge_result = financial_client.merge_tables(parse_result.file_id, merge_request["merges"])

    return FinancialMergeTablesRunResult(
        file_id=parse_result.file_id,
        actual_response=merge_result.raw_response,
        mineru_items=result_fetch.items,
        org_tables=org_tables_result.raw_response,
        merge_request=merge_request,
        mineru_raw_response=parse_result.raw_response,
        mineru_status_code=parse_result.status_code,
        mineru_elapsed_seconds=parse_result.elapsed_seconds,
        result_raw_response=result_fetch.raw_response,
        result_status_code=result_fetch.status_code,
        result_elapsed_seconds=result_fetch.elapsed_seconds,
        org_tables_status_code=org_tables_result.status_code,
        org_tables_elapsed_seconds=org_tables_result.elapsed_seconds,
        merge_tables_status_code=merge_result.status_code,
        merge_tables_elapsed_seconds=merge_result.elapsed_seconds,
    )


def build_merge_request(
    file_id: str,
    org_tables: list[Any],
    *,
    source_indices: list[int] | None = None,
    merge_text: str | None = None,
) -> dict[str, Any]:
    indices = source_indices or _default_source_indices(org_tables)
    source_ids: list[str] = []
    selected_org_tables: list[dict[str, Any]] = []
    for index in indices:
        if index < 0 or index >= len(org_tables):
            raise IndexError(f"merge source index out of range: {index}, org_tables={len(org_tables)}")
        table = org_tables[index]
        if not isinstance(table, dict):
            raise ValueError(f"org table at index {index} is not an object: {table!r}")
        source_id = table.get("primaryElementId") or table.get("element_id") or table.get("id")
        if not source_id:
            raise ValueError(f"org table at index {index} has no primaryElementId/id: {table!r}")
        selected_org_tables.append(table)
        source_id_text = str(source_id)
        if source_id_text not in source_ids:
            source_ids.append(source_id_text)

    if len(source_ids) < 2:
        raise ValueError(
            "merge-tables requires at least two distinct original sourceIds; "
            f"indices={indices}, sourceIds={source_ids}"
        )

    return {
        "fileId": file_id,
        "merges": [
            {
                "sourceIds": source_ids,
                "text": merge_text or "",
            }
        ],
        "debug": {
            "source_indices": indices,
            "selected_org_tables": [
                {
                    "id": table.get("id"),
                    "primaryElementId": table.get("primaryElementId"),
                    "text": table.get("text"),
                    "pageAt": table.get("pageAt"),
                    "mergeIds": table.get("mergeIds"),
                }
                for table in selected_org_tables
            ],
        },
    }


def _default_source_indices(org_tables: list[Any]) -> list[int]:
    indices: list[int] = []
    seen_source_ids: set[str] = set()
    for index, table in enumerate(org_tables):
        if not isinstance(table, dict):
            continue
        source_id = table.get("primaryElementId") or table.get("element_id") or table.get("id")
        if source_id is None:
            continue
        source_id_text = str(source_id)
        if source_id_text in seen_source_ids:
            continue
        seen_source_ids.add(source_id_text)
        indices.append(index)
        if len(indices) >= 2:
            return indices
    return indices
