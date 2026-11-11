from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from regression_mvp.utils.case_exclusions import filter_compare_exclusions, load_case_compare_exclusions
from regression_mvp.utils.case_loader import load_enabled_cases
from regression_mvp.utils.comparator import compare_items
from regression_mvp.utils.json_io import dump_json, load_json
from regression_mvp.utils.metrics import (
    classify_failure,
    diff_metrics,
    file_fingerprint,
    result_metrics,
    run_id_timestamp,
    upload_summary,
    validate_input_fingerprint,
)
from regression_mvp.utils.mineru_assessment import assess_mineru_regression
from regression_mvp.utils.normalizer import normalize_items
from regression_mvp.utils.reporter import ReportEntry, write_reports
from regression_mvp.utils.runners.ade_jobs_async_runner import run_ade_jobs_async


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"


def _load_config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


CONFIG = _load_config()
CASES = load_enabled_cases(ROOT_DIR, CONFIG, endpoint_type="ade_jobs_async")
REPORT_ENTRIES: list[ReportEntry] = []
RUN_ID = run_id_timestamp()
TARGET = "POST /openapi/ade/jobs/submit -> GET /openapi/ade/jobs/{job_id} -> GET /openapi/ade/jobs/{job_id}/data"


@pytest.mark.parametrize("case", CASES, ids=[case.case_name for case in CASES])
def test_ade_jobs_async_regression(case, request: pytest.FixtureRequest) -> None:
    actual_dir = ROOT_DIR / CONFIG["paths"]["actual_dir"] / "ade_jobs_async"
    report_dir = ROOT_DIR / CONFIG["paths"]["report_dir"] / "ade_jobs_async"
    actual_path = actual_dir / f"{case.case_name}.actual_items.json"
    start_time = time.monotonic()
    input_info = file_fingerprint(case.input_file)
    warnings = validate_input_fingerprint(case.input_sha256, input_info["sha256"])

    try:
        fingerprint_errors = [warning for warning in warnings if warning.startswith("input_sha256 mismatch")]
        if fingerprint_errors:
            raise ValueError("; ".join(fingerprint_errors))

        run_result = run_ade_jobs_async(CONFIG["api"], case.input_file)
        actual_items = run_result.actual_items
        dump_json(actual_path, actual_items)
        if request.config.getoption("--update-expected"):
            candidate_path = _candidate_expected_path(case.expected_file)
            dump_json(candidate_path, actual_items)
            warnings.append(f"wrote expected candidate to {candidate_path}; review it before replacing expected_items.json")

        expected_items = load_json(case.expected_file)
        comparison_expected, comparison_actual, ignored = filter_compare_exclusions(
            expected_items, actual_items, load_case_compare_exclusions(case)
        )
        diff = compare_items(
            normalize_items(comparison_expected, CONFIG["compare"]),
            normalize_items(comparison_actual, CONFIG["compare"]),
        )
        assessment = assess_mineru_regression(
            comparison_expected,
            comparison_actual,
            diff.diffs,
            input_file=case.input_file,
            profile=CONFIG["compare"].get("mineru_assessment"),
        )
        metrics = result_metrics(actual_items)
        metrics["mineru_assessment"] = assessment
        metrics["compare_exclusions"] = ignored
        step_summary = upload_summary(run_result.submit_raw_response, str(actual_path))
        step_summary["steps"] = {
            "submit": {
                "status_code": run_result.submit_status_code,
                "elapsed_seconds": round(run_result.submit_elapsed_seconds, 3),
                "job_id": run_result.job_id,
            },
            "status_wait": {
                "status_code": run_result.status_status_code,
                "elapsed_seconds": round(run_result.wait_elapsed_seconds, 3),
                "status": "completed",
            },
            "data": {
                "status_code": run_result.data_status_code,
                "elapsed_seconds": round(run_result.data_elapsed_seconds, 3),
            },
        }
        entry = ReportEntry(
            case_name=case.case_name,
            description=case.description,
            input_file=str(case.input_file),
            status=str(assessment["status"]),
            expected_items_path=str(case.expected_file),
            actual_items_path=str(actual_path),
            duration_seconds=time.monotonic() - start_time,
            upload_elapsed_seconds=run_result.submit_elapsed_seconds,
            http_status_code=run_result.submit_status_code,
            failure_category=str(assessment["failure_category"]),
            diff_summary=diff.summary,
            diff_detail=diff.diffs,
            diff_metrics=diff_metrics(diff.diffs),
            result_metrics=metrics,
            file_info=input_info,
            upload_summary=step_summary,
            warnings=warnings,
            error_message="",
        )
        REPORT_ENTRIES.append(entry)
        write_reports(report_dir, REPORT_ENTRIES, RUN_ID, target=TARGET)
        assert not assessment["should_fail"], (
            f"{case.case_name} regression {assessment['status']}: {assessment['summary']}. "
            f"See {report_dir / 'latest_report.md'}"
        )
    except Exception as exc:
        if any(entry.case_name == case.case_name for entry in REPORT_ENTRIES):
            raise
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
            failure_category=classify_failure(str(exc), 0),
            diff_summary="case execution failed",
            diff_detail=[],
            diff_metrics=diff_metrics([]),
            result_metrics=result_metrics([]),
            file_info=input_info,
            upload_summary={},
            warnings=warnings,
            error_message=str(exc),
        )
        REPORT_ENTRIES.append(entry)
        write_reports(report_dir, REPORT_ENTRIES, RUN_ID, target=TARGET)
        raise


def _candidate_expected_path(expected_file: Path) -> Path:
    return expected_file.with_name(f"{expected_file.stem}.candidate{expected_file.suffix}")
