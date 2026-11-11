from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regression_mvp.utils.api_clients.financial_extract_client import FinancialExtractApiClient
from regression_mvp.utils.api_clients.mineru_client import MinerUApiClient
from regression_mvp.utils.api_clients.subject_match_client import SubjectMatchApiClient
from regression_mvp.utils.json_io import load_json
from regression_mvp.utils.subject_match_presentation import infer_report_presentations
from regression_mvp.utils.subject_match_request_builder import (
    build_subject_match_request,
    local_template_config_key,
    load_local_report_list,
    load_raw_local_report_list,
)


@dataclass(frozen=True)
class FinancialSubjectMatchRunResult:
    actual_response: list[Any]
    request_payload: dict[str, Any]
    input_mode: str
    file_id: str
    mineru_items: list[Any]
    statement_tables: list[Any]
    mineru_status_code: int | None
    mineru_elapsed_seconds: float
    result_status_code: int | None
    result_elapsed_seconds: float
    three_statements_status_code: int | None
    three_statements_elapsed_seconds: float
    subject_match_status_code: int
    subject_match_elapsed_seconds: float


def run_financial_subject_match(
    api_config: dict[str, Any],
    input_file: Path,
    *,
    root_dir: Path | None = None,
) -> FinancialSubjectMatchRunResult:
    if input_file.suffix.lower() == ".pdf":
        return _run_from_pdf(api_config, input_file, root_dir=root_dir)

    request_payload = load_json(input_file)
    if not isinstance(request_payload, dict):
        raise ValueError(f"subject match input request is not an object: {input_file}")
    client = SubjectMatchApiClient(api_config)
    result = client.match(request_payload)

    return FinancialSubjectMatchRunResult(
        actual_response=result.raw_response,
        request_payload=request_payload,
        input_mode="json",
        file_id=str(request_payload.get("fileId") or ""),
        mineru_items=[],
        statement_tables=[],
        mineru_status_code=None,
        mineru_elapsed_seconds=0,
        result_status_code=None,
        result_elapsed_seconds=0,
        three_statements_status_code=None,
        three_statements_elapsed_seconds=0,
        subject_match_status_code=result.status_code,
        subject_match_elapsed_seconds=result.elapsed_seconds,
    )


def _run_from_pdf(
    api_config: dict[str, Any],
    input_file: Path,
    *,
    root_dir: Path | None,
) -> FinancialSubjectMatchRunResult:
    root_dir = root_dir or Path(__file__).resolve().parents[2]
    mineru_client = MinerUApiClient(api_config)
    financial_client = FinancialExtractApiClient(api_config)
    subject_client = SubjectMatchApiClient(api_config)

    parse_result = mineru_client.parse_to_json(input_file)
    result_fetch = mineru_client.get_result(parse_result.file_id)
    three_statements_result = financial_client.get_three_statements(parse_result.file_id)
    report_list = _load_report_list(
        subject_client,
        root_dir,
        api_config,
        org_tables=three_statements_result.raw_response,
    )
    request_payload = build_subject_match_request(
        file_id=parse_result.file_id,
        report_list=report_list,
        org_tables=three_statements_result.raw_response,
    )
    subject_result = subject_client.match(request_payload)

    return FinancialSubjectMatchRunResult(
        actual_response=subject_result.raw_response,
        request_payload=request_payload,
        input_mode="pdf",
        file_id=parse_result.file_id,
        mineru_items=result_fetch.items,
        statement_tables=three_statements_result.raw_response,
        mineru_status_code=parse_result.status_code,
        mineru_elapsed_seconds=parse_result.elapsed_seconds,
        result_status_code=result_fetch.status_code,
        result_elapsed_seconds=result_fetch.elapsed_seconds,
        three_statements_status_code=three_statements_result.status_code,
        three_statements_elapsed_seconds=three_statements_result.elapsed_seconds,
        subject_match_status_code=subject_result.status_code,
        subject_match_elapsed_seconds=subject_result.elapsed_seconds,
    )


def _load_report_list(
    client: SubjectMatchApiClient,
    root_dir: Path,
    api_config: dict[str, Any],
    *,
    org_tables: list[Any] | None = None,
) -> list[Any]:
    template_source = str(
        api_config.get("subject_match_template_source") or "api"
    ).strip().lower()
    if template_source == "local_file":
        return _load_configured_local_report_list(root_dir, api_config, org_tables or [])
    if template_source != "api":
        raise ValueError(
            "subject_match_template_source must be 'api' or 'local_file': "
            f"{template_source!r}"
        )
    try:
        return client.get_templates()
    except Exception:
        return _load_configured_local_report_list(root_dir, api_config, org_tables or [])


def _load_configured_local_report_list(
    root_dir: Path,
    api_config: dict[str, Any],
    org_tables: list[Any],
) -> list[Any]:
    template_config_key = local_template_config_key(org_tables)

    presentation_by_report_type = None
    if template_config_key == "subject_match_english_template_file":
        raw_report_list = load_raw_local_report_list(
            root_dir,
            api_config,
            template_config_key=template_config_key,
        )
        timeout_value = api_config.get("subject_match_presentation_timeout_seconds", 300)
        timeout_s = float(timeout_value) if timeout_value is not None else None
        presentation_by_report_type = infer_report_presentations(
            raw_report_list,
            org_tables,
            model_id=str(
                api_config.get("subject_match_presentation_model_id")
                or "model_for_table_classify"
            ),
            timeout_s=timeout_s,
        )

    return load_local_report_list(
        root_dir,
        api_config,
        template_config_key=template_config_key,
        presentation_by_report_type=presentation_by_report_type,
    )
