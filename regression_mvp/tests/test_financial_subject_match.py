from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from regression_mvp.utils.case_loader import load_enabled_cases
from regression_mvp.utils.comparator import compare_items
from regression_mvp.utils.json_io import dump_json, load_json
from regression_mvp.utils.metrics import (
    classify_failure,
    diff_metrics,
    file_fingerprint,
    financial_subject_match_metrics,
    result_metrics,
    run_id_timestamp,
    validate_input_fingerprint,
)
from regression_mvp.utils.normalizer import normalize_items
from regression_mvp.utils.reporter import ReportEntry, write_reports
from regression_mvp.utils.runners.financial_subject_match_runner import run_financial_subject_match


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"


def _load_config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


CONFIG = _load_config()
CASES = load_enabled_cases(ROOT_DIR, CONFIG, endpoint_type="financial_subject_match")
if not CASES:
    pytest.skip("no enabled financial_subject_match cases", allow_module_level=True)
REPORT_ENTRIES: list[ReportEntry] = []
RUN_ID = run_id_timestamp()


@pytest.mark.parametrize("case", CASES, ids=[case.case_name for case in CASES])
def test_financial_subject_match_regression(case, request: pytest.FixtureRequest) -> None:
    actual_dir = ROOT_DIR / CONFIG["paths"]["actual_dir"] / "financial_subject_match"
    report_dir = ROOT_DIR / CONFIG["paths"]["report_dir"] / "financial_subject_match"
    actual_path = actual_dir / f"{case.case_name}.actual_response.json"
    request_path = actual_dir / f"{case.case_name}.subject_match_request.json"
    start_time = time.monotonic()
    input_info = file_fingerprint(case.input_file)
    warnings = validate_input_fingerprint(case.input_sha256, input_info["sha256"])

    try:
        fingerprint_errors = [warning for warning in warnings if warning.startswith("input_sha256 mismatch")]
        if fingerprint_errors:
            raise ValueError("; ".join(fingerprint_errors))

        run_result = run_financial_subject_match(CONFIG["api"], case.input_file, root_dir=ROOT_DIR)
        actual_response = run_result.actual_response
        dump_json(actual_path, actual_response)
        dump_json(request_path, run_result.request_payload)

        if request.config.getoption("--update-expected"):
            candidate_path = _candidate_expected_path(case.expected_file)
            dump_json(candidate_path, actual_response)
            warnings.append(
                f"wrote expected candidate to {candidate_path}; review it before replacing expected_response.json"
            )

        expected_response = load_json(case.expected_file)
        normalized_actual = normalize_items(actual_response, CONFIG["compare"])
        normalized_expected = normalize_items(expected_response, CONFIG["compare"])
        diff = compare_items(normalized_expected, normalized_actual)
        diff_stats = diff_metrics(diff.diffs)

        status = "passed" if diff.passed else "failed"
        subject_metrics = financial_subject_match_metrics(actual_response)
        step_summary = {
            "success": True,
            "file_id": run_result.request_payload.get("fileId"),
            "actual_items_path": str(actual_path),
            "subject_match_request_path": str(request_path),
            "input_mode": run_result.input_mode,
            "steps": {
                "financial_subject_match": {
                    "status_code": run_result.subject_match_status_code,
                    "elapsed_seconds": round(run_result.subject_match_elapsed_seconds, 3),
                    "use_llm": bool(CONFIG["api"].get("subject_match_use_llm", False)),
                },
            },
        }
        if run_result.input_mode == "pdf":
            step_summary["steps"].update(
                {
                    "mineru_to_json": {
                        "status_code": run_result.mineru_status_code,
                        "elapsed_seconds": round(run_result.mineru_elapsed_seconds, 3),
                        "file_id": run_result.file_id,
                    },
                    "mineru_result": {
                        "status_code": run_result.result_status_code,
                        "elapsed_seconds": round(run_result.result_elapsed_seconds, 3),
                    },
                    "financial_three_statements": {
                        "status_code": run_result.three_statements_status_code,
                        "elapsed_seconds": round(run_result.three_statements_elapsed_seconds, 3),
                        "statement_table_count": len(run_result.statement_tables),
                    },
                }
            )
        combined_result_metrics = result_metrics(actual_response)
        combined_result_metrics["financial_subject_match"] = subject_metrics

        entry = ReportEntry(
            case_name=case.case_name,
            description=case.description,
            input_file=str(case.input_file),
            status=status,
            expected_items_path=str(case.expected_file),
            actual_items_path=str(actual_path),
            duration_seconds=time.monotonic() - start_time,
            upload_elapsed_seconds=run_result.subject_match_elapsed_seconds,
            http_status_code=run_result.subject_match_status_code,
            failure_category=classify_failure("", diff_stats["total"]),
            diff_summary=diff.summary,
            diff_detail=diff.diffs,
            diff_metrics=diff_stats,
            result_metrics=combined_result_metrics,
            file_info=input_info,
            upload_summary=step_summary,
            warnings=warnings,
            error_message="",
        )
        REPORT_ENTRIES.append(entry)
        write_reports(
            report_dir,
            REPORT_ENTRIES,
            RUN_ID,
            target="POST /api/parsing/subject-match/match",
        )

        assert diff.passed, (
            f"{case.case_name} regression failed: {diff.summary}. "
            f"See {report_dir / 'latest_report.md'}"
        )
    except Exception as exc:
        if any(entry.case_name == case.case_name for entry in REPORT_ENTRIES):
            raise
        error_message = str(exc)
        entry = ReportEntry(
            case_name=case.case_name,
            description=case.description,
            input_file=str(case.input_file),
            status="failed",
            expected_items_path=str(case.expected_file),
            actual_items_path=str(actual_path),
            duration_seconds=time.monotonic() - start_time,
            upload_elapsed_seconds=0,
            http_status_code=None,
            failure_category=classify_failure(error_message, 0),
            diff_summary="case execution failed",
            diff_detail=[],
            diff_metrics=diff_metrics([]),
            result_metrics=result_metrics([]),
            file_info=input_info,
            upload_summary={},
            warnings=warnings,
            error_message=error_message,
        )
        REPORT_ENTRIES.append(entry)
        write_reports(
            report_dir,
            REPORT_ENTRIES,
            RUN_ID,
            target="POST /api/parsing/subject-match/match",
        )
        raise


def _candidate_expected_path(expected_file: Path) -> Path:
    return expected_file.with_name(f"{expected_file.stem}.candidate{expected_file.suffix}")
