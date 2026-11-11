from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any

from regression_mvp.utils.api_clients.file_read_client import FileReadApiClient
from regression_mvp.utils.artifact_renderer import render_content_preview, render_overlay_preview
from regression_mvp.utils.financial_statement_artifacts import (
    write_merge_tables_artifact,
    write_three_statements_artifacts,
)
from regression_mvp.utils.item_loader import resolve_actual_items
from regression_mvp.utils.json_io import dump_json
from regression_mvp.utils.metrics import (
    file_fingerprint,
    financial_metadata_metrics,
    financial_merge_tables_metrics,
    financial_org_tables_metrics,
    financial_subject_match_metrics,
    financial_three_statements_metrics,
    result_metrics,
    upload_summary,
)
from regression_mvp.utils.runners.financial_metadata_runner import run_financial_metadata
from regression_mvp.utils.runners.financial_merge_tables_runner import run_financial_merge_tables
from regression_mvp.utils.runners.financial_org_tables_runner import run_financial_org_tables
from regression_mvp.utils.runners.financial_subject_match_runner import run_financial_subject_match
from regression_mvp.utils.runners.financial_three_statements_runner import run_financial_three_statements
from regression_mvp.utils.runners.mineru_to_json_runner import run_mineru_to_json
from regression_mvp.utils.runners.mineru_upload_async_runner import run_mineru_upload_async
from regression_mvp.utils.runners.ade_jobs_async_runner import run_ade_jobs_async
from regression_mvp.utils.subject_match_artifacts import write_subject_match_table_artifact


def generate_candidate(
    root_dir: Path,
    config: dict[str, Any],
    endpoint: str,
    input_file: Path,
    case_name: str,
    description: str,
) -> Path:
    input_file = input_file.resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"input file not found: {input_file}")

    safe_case_name = _safe_name(case_name)
    candidate_dir = root_dir / "case_candidates" / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_case_name}"
    candidate_dir.mkdir(parents=True, exist_ok=False)

    copied_input = candidate_dir / f"input{input_file.suffix.lower()}"
    shutil.copy2(input_file, copied_input)

    if endpoint == "file_read":
        client = FileReadApiClient(config["api"])
        upload_result = client.upload_file(copied_input)
        preview_items = resolve_actual_items(client, upload_result)
        actual_payload: Any = preview_items
        raw_response = upload_result.raw_response
        summary_response = raw_response
        status_code = upload_result.status_code
        elapsed_seconds = upload_result.elapsed_seconds
        expected_name = "expected_items.candidate.json"
    elif endpoint == "mineru_to_json":
        run_result = run_mineru_to_json(config["api"], copied_input)
        preview_items = run_result.actual_items
        actual_payload = preview_items
        raw_response = run_result.raw_response
        summary_response = raw_response
        status_code = run_result.status_code
        elapsed_seconds = run_result.elapsed_seconds
        expected_name = "expected_items.candidate.json"
    elif endpoint == "mineru_upload_async":
        run_result = run_mineru_upload_async(config["api"], copied_input)
        preview_items = run_result.actual_items
        actual_payload = preview_items
        raw_response = run_result.raw_response
        summary_response = raw_response
        status_code = run_result.status_code
        elapsed_seconds = run_result.elapsed_seconds
        expected_name = "expected_items.candidate.json"
    elif endpoint == "ade_jobs_async":
        run_result = run_ade_jobs_async(config["api"], copied_input)
        preview_items = run_result.actual_items
        actual_payload = preview_items
        raw_response = run_result.data_raw_response
        summary_response = run_result.submit_raw_response
        status_code = run_result.submit_status_code
        elapsed_seconds = run_result.submit_elapsed_seconds
        expected_name = "expected_items.candidate.json"
    elif endpoint == "financial_metadata":
        run_result = run_financial_metadata(config["api"], copied_input)
        preview_items = run_result.mineru_items
        actual_payload = run_result.actual_response
        raw_response = run_result.actual_response
        summary_response = raw_response
        status_code = run_result.metadata_status_code
        elapsed_seconds = run_result.metadata_elapsed_seconds
        expected_name = "expected_response.candidate.json"
    elif endpoint == "financial_three_statements":
        run_result = run_financial_three_statements(config["api"], copied_input)
        preview_items = run_result.mineru_items
        actual_payload = run_result.actual_response
        raw_response = run_result.actual_response
        summary_response = {"three_statements": run_result.actual_response}
        status_code = run_result.three_statements_status_code
        elapsed_seconds = run_result.three_statements_elapsed_seconds
        expected_name = "expected_response.candidate.json"
    elif endpoint == "financial_org_tables":
        run_result = run_financial_org_tables(config["api"], copied_input)
        preview_items = run_result.mineru_items
        actual_payload = run_result.actual_response
        raw_response = run_result.actual_response
        summary_response = {"org_tables": run_result.actual_response}
        status_code = run_result.org_tables_status_code
        elapsed_seconds = run_result.org_tables_elapsed_seconds
        expected_name = "expected_response.candidate.json"
    elif endpoint == "financial_merge_tables":
        run_result = run_financial_merge_tables(config["api"], copied_input)
        preview_items = run_result.mineru_items
        actual_payload = run_result.actual_response
        raw_response = run_result.actual_response
        summary_response = {"merge_tables": run_result.actual_response}
        status_code = run_result.merge_tables_status_code
        elapsed_seconds = run_result.merge_tables_elapsed_seconds
        expected_name = "expected_response.candidate.json"
    elif endpoint == "financial_subject_match":
        run_result = run_financial_subject_match(config["api"], copied_input, root_dir=root_dir)
        preview_items = run_result.actual_response
        actual_payload = run_result.actual_response
        raw_response = run_result.actual_response
        summary_response = {"subject_match": run_result.actual_response}
        status_code = run_result.subject_match_status_code
        elapsed_seconds = run_result.subject_match_elapsed_seconds
        expected_name = "expected_response.candidate.json"
    else:
        raise ValueError(f"unsupported endpoint: {endpoint}")

    actual_items_path = candidate_dir / "actual_items.json"
    actual_response_path = candidate_dir / "actual_response.json"
    org_tables_path = candidate_dir / "org_tables.json"
    subject_match_request_path = candidate_dir / "subject_match_request.json"
    expected_candidate_path = candidate_dir / expected_name
    raw_response_path = candidate_dir / "actual_response.json"
    metrics_path = candidate_dir / "metrics.json"
    preview_md_path = candidate_dir / "preview.md"
    preview_html_path = candidate_dir / "preview.html"

    dump_json(actual_items_path, preview_items)
    dump_json(actual_response_path, actual_payload)
    dump_json(expected_candidate_path, actual_payload)
    dump_json(raw_response_path, raw_response)
    if endpoint == "financial_merge_tables":
        dump_json(org_tables_path, run_result.org_tables)
    if endpoint == "financial_subject_match":
        dump_json(subject_match_request_path, run_result.request_payload)

    file_info = file_fingerprint(copied_input)
    result_stats = result_metrics(preview_items)
    if endpoint == "financial_metadata":
        result_stats["financial_metadata"] = financial_metadata_metrics(actual_payload)
    elif endpoint == "financial_three_statements":
        result_stats["financial_three_statements"] = financial_three_statements_metrics(actual_payload)
        statement_artifacts = write_three_statements_artifacts(
            actual_payload,
            actual_payload,
            candidate_dir,
            safe_case_name,
        )
        result_stats["financial_three_statements_artifacts"] = statement_artifacts
    elif endpoint == "financial_org_tables":
        result_stats["financial_org_tables"] = financial_org_tables_metrics(actual_payload)
    elif endpoint == "financial_merge_tables":
        result_stats["financial_merge_tables"] = financial_merge_tables_metrics(
            actual_payload,
            run_result.merge_request,
        )
        statement_artifacts = write_three_statements_artifacts(
            actual_payload,
            actual_payload,
            candidate_dir,
            safe_case_name,
        )
        statement_artifacts.update(
            write_merge_tables_artifact(
                run_result.org_tables,
                actual_payload,
                run_result.merge_request,
                candidate_dir,
                safe_case_name,
            )
        )
        result_stats["financial_merge_tables_artifacts"] = statement_artifacts
    elif endpoint == "financial_subject_match":
        result_stats["financial_subject_match"] = financial_subject_match_metrics(actual_payload)
        write_subject_match_table_artifact(
            actual_payload,
            actual_payload,
            candidate_dir,
            request_payload=run_result.request_payload,
        )

    metrics = {
        "file": file_info,
        "upload": {
            "status_code": status_code,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "summary": upload_summary(summary_response, str(actual_items_path)),
        },
        "result": result_stats,
    }
    dump_json(metrics_path, metrics)

    render_content_preview(preview_items, preview_md_path, preview_html_path)
    overlay_path = render_overlay_preview(copied_input, preview_items, candidate_dir)

    candidate_yaml = {
        "case_name": safe_case_name,
        "endpoint_type": endpoint,
        "description": description,
        "enabled": "true",
        "input_file": copied_input.name,
        "expected_file": expected_candidate_path.name,
        "input_sha256": file_info["sha256"],
        "actual_items_file": actual_items_path.name,
        "actual_response_file": raw_response_path.name,
        "metrics_file": metrics_path.name,
        "preview_md": preview_md_path.name,
        "preview_html": preview_html_path.name,
        "overlay_html": overlay_path.name if overlay_path else "",
        "review_status": "pending",
    }
    if endpoint == "financial_merge_tables":
        source_indices = run_result.merge_request.get("debug", {}).get("source_indices") or []
        merges = run_result.merge_request.get("merges") or []
        first_merge = merges[0] if merges else {}
        candidate_yaml["merge_source_indices"] = ",".join(str(index) for index in source_indices)
        candidate_yaml["merge_text"] = first_merge.get("text", "")
        candidate_yaml["org_tables_file"] = org_tables_path.name
    _write_candidate_yaml(candidate_dir / "candidate.yaml", candidate_yaml)
    (candidate_dir / "review_notes.md").write_text(
        "# Review Notes\n\n"
        f"- Open `{copied_input.name}` and `overlay.html` side by side to verify coordinate boxes.\n"
        "- Open `preview.html` to verify content and table structure.\n"
        "- Promote this candidate only after manual review.\n",
        encoding="utf-8",
    )
    return candidate_dir


def promote_candidate(root_dir: Path, candidate_dir: Path, *, force: bool = False) -> Path:
    candidate_dir = candidate_dir.resolve()
    candidate_yaml = candidate_dir / "candidate.yaml"
    if not candidate_yaml.is_file():
        raise FileNotFoundError(f"candidate.yaml not found: {candidate_yaml}")

    raw = _read_simple_yaml(candidate_yaml)
    case_name = _safe_name(raw.get("case_name") or candidate_dir.name)
    endpoint_type = raw.get("endpoint_type") or "file_read"
    case_dir = root_dir / "cases" / endpoint_type / case_name
    if case_dir.exists():
        if not force:
            raise FileExistsError(f"case already exists: {case_dir}. Use --force to overwrite it.")
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)

    input_file_name = raw.get("input_file") or "input.pdf"
    source_input = candidate_dir / input_file_name
    source_expected = candidate_dir / (raw.get("expected_file") or "expected_items.candidate.json")
    if not source_input.is_file():
        raise FileNotFoundError(f"candidate input file not found: {source_input}")
    if not source_expected.is_file():
        raise FileNotFoundError(f"candidate expected file not found: {source_expected}")

    target_input = case_dir / source_input.name
    response_endpoints = {
        "financial_metadata",
        "financial_three_statements",
        "financial_org_tables",
        "financial_merge_tables",
        "financial_subject_match",
    }
    target_expected_name = "expected_response.json" if endpoint_type in response_endpoints else "expected_items.json"
    target_expected = case_dir / target_expected_name
    shutil.copy2(source_input, target_input)
    shutil.copy2(source_expected, target_expected)

    case_yaml = {
        "case_name": case_name,
        "endpoint_type": endpoint_type,
        "description": raw.get("description") or "",
        "enabled": "true",
        "input_file": target_input.name,
        "expected_file": target_expected.name,
        "input_sha256": raw.get("input_sha256") or file_fingerprint(target_input)["sha256"],
        "created_from_candidate": str(candidate_dir),
        "review_status": "approved",
    }
    if endpoint_type == "financial_merge_tables":
        case_yaml["merge_source_indices"] = raw.get("merge_source_indices") or "0,2"
        case_yaml["merge_text"] = raw.get("merge_text") or ""
    _write_candidate_yaml(case_dir / "case.yaml", case_yaml)
    return case_dir


def _safe_name(value: str) -> str:
    name = re.sub("[^\\w.\\-\u4e00-\u9fff]+", "_", value.strip())
    return name.strip("._") or "case"


def _write_candidate_yaml(path: Path, values: dict[str, Any]) -> None:
    lines = []
    for key, value in values.items():
        text = "" if value is None else str(value)
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}: "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_simple_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"').strip("'").replace('\\"', '"').replace("\\\\", "\\")
    return result
