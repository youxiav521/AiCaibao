from __future__ import annotations

import html
import json
import os
import re
import shutil
from urllib.parse import quote
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from regression_mvp.utils.artifact_labels import artifact_label
from regression_mvp.utils.case_exclusions import filter_compare_exclusions, load_compare_exclusions
from regression_mvp.utils.comparator import compare_items
from regression_mvp.utils.artifact_renderer import (
    diff_item_indexes,
    render_content_preview,
    render_diff_preview,
    render_overlay_preview,
    render_table_diff_preview,
)
from regression_mvp.utils.diagnostics import diagnose_failure, diagnostic_counts
from regression_mvp.utils.json_io import dump_json
from regression_mvp.utils.metric_display import diff_metric_rows, key_metric_rows
from regression_mvp.utils.metrics import SEVERITY_RANK, build_run_summary, classify_failure, diff_metrics, diff_severity, utc_timestamp
from regression_mvp.utils.mineru_assessment import MINERU_ENDPOINTS, assess_mineru_regression
from regression_mvp.utils.normalizer import normalize_items
from regression_mvp.utils.report_store import store_report_run
from regression_mvp.utils.subject_match_artifacts import write_subject_match_table_artifact


COMPARE_EXCLUSION_ENDPOINTS = MINERU_ENDPOINTS | {"file_read"}


@dataclass(frozen=True)
class ReportEntry:
    case_name: str
    description: str
    input_file: str
    status: str
    expected_items_path: str
    actual_items_path: str
    duration_seconds: float
    upload_elapsed_seconds: float
    http_status_code: int | None
    failure_category: str
    diff_summary: str
    diff_detail: list[dict[str, Any]]
    diff_metrics: dict[str, Any]
    result_metrics: dict[str, Any]
    file_info: dict[str, Any]
    upload_summary: dict[str, Any]
    warnings: list[str]
    error_message: str


def write_reports(
    report_dir: Path,
    entries: list[ReportEntry],
    run_id: str,
    *,
    target: str = "POST /api/file/read",
    generated_at: str | None = None,
    update_store: bool = True,
    rebuild_diffs_only: bool = False,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    run_dir = report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = build_run_summary(entries)
    diagnostics = [_entry_diagnostic(entry) for entry in entries]
    summary["diagnostic_counts"] = diagnostic_counts(diagnostics)
    generated_at = generated_at or utc_timestamp()
    payload = {
        "run": {
            "run_id": run_id,
            "generated_at": generated_at,
            "target": target,
            "report_dir": str(run_dir),
        },
        "summary": summary,
        "cases": [_entry_payload(entry) for entry in entries],
    }

    markdown = _render_markdown(entries, run_id, target, run_dir, generated_at, summary)

    dump_json(report_dir / "latest_report.json", payload)
    dump_json(run_dir / "report.json", payload)
    dump_json(run_dir / "summary.json", summary)
    (report_dir / "latest_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")

    _write_case_artifacts(run_dir, entries, rebuild_diffs_only=rebuild_diffs_only)
    html_report = _render_html(entries, run_id, target, run_dir, generated_at, summary)
    (report_dir / "latest_report.html").write_text(_latest_html_redirect(run_id), encoding="utf-8")
    (run_dir / "report.html").write_text(html_report, encoding="utf-8")
    _write_index(report_dir)
    if update_store:
        _write_report_store_best_effort(report_dir, entries, run_id, target, summary, generated_at)


def write_html_reports_from_existing(report_root: Path) -> int:
    count = 0
    for report_json in _existing_report_jsons(report_root):
        try:
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            run = payload.get("run") or {}
            summary = payload.get("summary") or {}
            cases = payload.get("cases") or []
            entries = [_entry_from_payload(case) for case in cases if isinstance(case, dict)]
            summary = dict(summary)
            summary["diagnostic_counts"] = diagnostic_counts(
                [_entry_diagnostic(entry) for entry in entries]
            )
            run_dir = report_json.parent
            report_dir = report_json.parents[2]
            run_id = str(run.get("run_id") or run_dir.name)
            target = str(run.get("target") or report_dir.name)
            generated_at = str(run.get("generated_at") or "")
            _write_case_artifacts(run_dir, entries)
            html_report = _render_html(entries, run_id, target, run_dir, generated_at, summary)
            (run_dir / "report.html").write_text(html_report, encoding="utf-8")
            count += 1
        except Exception:
            continue

    for latest_json in sorted(report_root.glob("*/latest_report.json")):
        try:
            payload = json.loads(latest_json.read_text(encoding="utf-8"))
            run = payload.get("run") or {}
            run_id = str(run.get("run_id") or "")
            if run_id:
                (latest_json.parent / "latest_report.html").write_text(_latest_html_redirect(run_id), encoding="utf-8")
        except Exception:
            continue
    return count


def rebuild_reports_from_existing(report_root: Path, compare_config: dict[str, Any]) -> dict[str, Any]:
    summary = {"runs": 0, "recompared_cases": 0, "skipped_cases": 0, "failed_runs": []}
    for report_json in _existing_report_jsons(report_root):
        try:
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            run = payload.get("run") or {}
            cases = payload.get("cases") or []
            run_dir = report_json.parent
            report_dir = report_json.parents[2]
            entries: list[ReportEntry] = []
            for case_payload in cases:
                if not isinstance(case_payload, dict):
                    continue
                entry = _entry_from_payload(case_payload)
                refreshed_entry, refreshed = _recompare_existing_entry(entry, run_dir, compare_config, report_dir.name)
                entries.append(refreshed_entry)
                if refreshed:
                    summary["recompared_cases"] += 1
                else:
                    summary["skipped_cases"] += 1

            run_id = str(run.get("run_id") or run_dir.name)
            target = str(run.get("target") or report_dir.name)
            generated_at = str(run.get("generated_at") or "") or None
            write_reports(
                report_dir,
                entries,
                run_id,
                target=target,
                generated_at=generated_at,
                update_store=False,
                rebuild_diffs_only=True,
            )
            summary["runs"] += 1
        except Exception as exc:
            summary["failed_runs"].append({"report": str(report_json), "error": str(exc)})
    if summary["runs"]:
        from regression_mvp.utils.report_store import rebuild_store

        rebuild_store(report_root)
    return summary


def recompare_report_case(
    report_root: Path,
    endpoint: str,
    run_id: str,
    case_name: str,
    compare_config: dict[str, Any],
) -> bool:
    """Refresh one historical case after a reviewed comparison-rule change."""
    report_dir = report_root / Path(endpoint).name
    run_dir = report_dir / "runs" / Path(run_id).name
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"report not found: {endpoint}/{run_id}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    run = payload.get("run") if isinstance(payload, dict) else {}
    cases = payload.get("cases") if isinstance(payload, dict) else []
    if not isinstance(run, dict) or not isinstance(cases, list):
        raise ValueError("report payload is invalid")

    entries: list[ReportEntry] = []
    refreshed = False
    for case_payload in cases:
        if not isinstance(case_payload, dict):
            continue
        entry = _entry_from_payload(case_payload)
        if entry.case_name == case_name:
            entry, changed = _recompare_existing_entry(entry, run_dir, compare_config, endpoint)
            refreshed = refreshed or changed
        entries.append(entry)
    if not refreshed:
        return False

    write_reports(
        report_dir,
        entries,
        str(run.get("run_id") or run_id),
        target=str(run.get("target") or endpoint),
        generated_at=str(run.get("generated_at") or "") or None,
        rebuild_diffs_only=True,
    )
    return True


def _existing_report_jsons(report_root: Path) -> list[Path]:
    report_paths = list(report_root.glob("runs/*/report.json"))
    report_paths.extend(report_root.glob("*/runs/*/report.json"))
    return sorted(set(report_paths))


def _recompare_existing_entry(
    entry: ReportEntry,
    run_dir: Path,
    compare_config: dict[str, Any],
    endpoint: str,
) -> tuple[ReportEntry, bool]:
    if entry.error_message:
        return entry, False
    expected_items = _load_list(Path(entry.expected_items_path))
    actual_items = _load_list(Path(entry.actual_items_path))
    if actual_items is None:
        actual_items = _load_list(run_dir / "cases" / _safe_path_name(entry.case_name) / "actual_items.json")
    if expected_items is None or actual_items is None:
        return entry, False

    exclusions = load_compare_exclusions(Path(entry.expected_items_path).parent)
    comparison_expected, comparison_actual, ignored = filter_compare_exclusions(
        expected_items,
        actual_items,
        exclusions,
    )
    diff = compare_items(
        normalize_items(comparison_expected, compare_config),
        normalize_items(comparison_actual, compare_config),
    )
    stats = diff_metrics(diff.diffs)
    result_metrics = dict(entry.result_metrics)
    result_metrics["compare_exclusions"] = ignored
    status = "passed" if diff.passed else "failed"
    failure_category = classify_failure("", int(stats.get("total", 0)))
    if endpoint in MINERU_ENDPOINTS:
        assessment = assess_mineru_regression(
            comparison_expected,
            comparison_actual,
            diff.diffs,
            input_file=Path(entry.input_file),
            profile=compare_config.get("mineru_assessment"),
        )
        result_metrics["mineru_assessment"] = assessment
        status = str(assessment["status"])
        failure_category = str(assessment["failure_category"])
    return (
        replace(
            entry,
            status=status,
            failure_category=failure_category,
            diff_summary=diff.summary,
            diff_detail=diff.diffs,
            diff_metrics=stats,
            result_metrics=result_metrics,
        ),
        True,
    )


def _render_markdown(
    entries: list[ReportEntry],
    run_id: str,
    target: str,
    run_dir: Path,
    generated_at: str,
    summary: dict[str, Any],
) -> str:
    lines = [
        f"# {target} Regression Report",
        "",
        "## Run Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_at: `{generated_at}`",
        f"- total_cases: {summary['total_cases']}",
        f"- passed_cases: {summary['passed_cases']}",
        f"- failed_cases: {summary['failed_cases']}",
        f"- structural_warning_cases: {summary.get('warning_cases', 0)}",
        f"- content_suspected_cases: {summary.get('suspected_cases', 0)}",
        f"- pass_rate: {summary['pass_rate']:.2%}",
        f"- total_duration_seconds: {summary['total_duration_seconds']}",
        f"- total_diffs: {summary['total_diffs']}",
        f"- archived_report: `reports/runs/{run_id}/report.md`",
        "",
    ]

    if summary["failure_counts"]:
        lines.extend(["### Failure Categories", "", "| category | count |", "| --- | --- |"])
        for category, count in summary["failure_counts"].items():
            lines.append(f"| `{category}` | {count} |")
        lines.append("")

    if summary.get("diagnostic_counts"):
        lines.extend(["### Diagnostic Categories", "", "| category | count |", "| --- | ---: |"])
        for category, count in summary["diagnostic_counts"].items():
            lines.append(f"| `{category}` | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Case Overview",
            "",
            "| case | status | duration_s | upload_s | items | tables | pages | raw_diffs | assessment | failure |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for entry in entries:
        result = entry.result_metrics
        diff = entry.diff_metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{entry.case_name}`",
                    entry.status,
                    str(round(entry.duration_seconds, 3)),
                    str(round(entry.upload_elapsed_seconds, 3)),
                    str(result.get("item_count", 0)),
                    str(result.get("table_count", 0)),
                    str(result.get("page_count", 0)),
                    str(diff.get("total", 0)),
                    _mineru_assessment_markdown(entry),
                    f"`{entry.failure_category}`" if entry.failure_category else "",
                ]
            )
            + " |"
        )
    lines.append("")

    for entry in entries:
        lines.extend(_render_case(entry, run_dir))
    return "\n".join(lines)


def _render_html(
    entries: list[ReportEntry],
    run_id: str,
    target: str,
    run_dir: Path,
    generated_at: str,
    summary: dict[str, Any],
) -> str:
    case_overview = "".join(
        "<tr>"
        f'<td><a href="#{_case_anchor(entry.case_name)}">{_esc(entry.case_name)}</a></td>'
        f"<td>{_status_badge(entry.status)}</td>"
        f"<td>{entry.duration_seconds:.3f}</td>"
        f"<td>{entry.upload_elapsed_seconds:.3f}</td>"
        f"<td>{_esc(entry.result_metrics.get('item_count', 0))}</td>"
        f"<td>{_esc(entry.result_metrics.get('table_count', 0))}</td>"
        f"<td>{_esc(entry.result_metrics.get('page_count', 0))}</td>"
        f"<td>{_esc(entry.diff_metrics.get('total', 0))}</td>"
        f"<td>{_mineru_assessment_overview(entry)}</td>"
        f"<td>{_diagnostic_overview(entry)}</td>"
        f"<td>{_case_preview_links(entry, run_dir)}</td>"
        "</tr>"
        for entry in entries
    )
    case_sections = "\n".join(_render_case_html(entry, run_dir) for entry in entries)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(target)} Regression Report</title>
  <style>
    :root {{ --bg:#f6f7f9; --panel:#fff; --text:#17202a; --muted:#667085; --border:#d9dee7; --accent:#2563eb; --pass:#16794c; --fail:#b42318; --warn:#9a6700; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.55 "Segoe UI","Microsoft YaHei",Arial,sans-serif; }}
    a {{ color:var(--accent); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    header {{ padding:22px 28px; background:#fff; border-bottom:1px solid var(--border); }}
    h1 {{ margin:0 0 8px; font-size:22px; line-height:1.3; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    h3 {{ margin:18px 0 10px; font-size:15px; }}
    main {{ max-width:1280px; margin:0 auto; padding:22px 28px 42px; }}
    .muted {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; margin-bottom:18px; }}
    .metric,.section {{ background:var(--panel); border:1px solid var(--border); border-radius:6px; }}
    .metric {{ padding:14px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric strong {{ display:block; margin-top:4px; font-size:18px; overflow-wrap:anywhere; }}
    .section {{ margin-bottom:18px; padding:18px; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; min-width:760px; border-collapse:collapse; }}
    th,td {{ padding:9px 11px; border-top:1px solid var(--border); text-align:left; vertical-align:top; }}
    th {{ background:#f8fafc; color:#475467; font-size:12px; }}
    .key-metrics-table th:first-child,.key-metrics-table td:first-child {{ width:14rem; min-width:14rem; white-space:nowrap; word-break:keep-all; }}
    .key-metrics-table th:last-child,.key-metrics-table td:last-child {{ overflow-wrap:anywhere; }}
    code,pre {{ font-family:Consolas,"Liberation Mono",monospace; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#101828; color:#f9fafb; padding:12px; border-radius:6px; }}
    .badge {{ display:inline-flex; min-height:22px; align-items:center; padding:2px 8px; border-radius:999px; background:#eef2f6; color:#344054; font-size:12px; font-weight:600; }}
    .passed,.pass,.success {{ background:#e7f6ee; color:var(--pass); }}
    .warning {{ background:#fff6d7; color:var(--warn); }}
    .suspected {{ background:#fff1f0; color:#b42318; }}
    .failed,.fail,.error {{ background:#fde8e7; color:var(--fail); }}
    .blocker,.critical {{ background:#fde8e7; color:var(--fail); }}
    .major {{ background:#fff1f0; color:#b42318; }}
    .minor {{ background:#fff6d7; color:var(--warn); }}
    .info {{ background:#eef2f6; color:var(--muted); }}
    .high {{ background:#fde8e7; color:var(--fail); }}
    .medium {{ background:#fff6d7; color:var(--warn); }}
    .low {{ background:#eef2f6; color:var(--muted); }}
    .links {{ display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start; }}
    .links a {{ display:inline-flex; align-items:center; min-height:26px; padding:3px 9px; border:1px solid var(--border); border-radius:999px; background:#fff; line-height:1.2; white-space:nowrap; }}
    .links a:hover {{ background:#eef4ff; text-decoration:none; }}
    .diagnostic {{ margin:14px 0; padding:14px; border:1px solid var(--border); border-left:4px solid var(--border); border-radius:6px; background:#fbfcfe; }}
    .diagnostic-high {{ border-left-color:var(--fail); }}
    .diagnostic-medium {{ border-left-color:var(--warn); }}
    .diagnostic-low {{ border-left-color:var(--muted); }}
    .diagnostic ul {{ margin:0; padding-left:20px; }}
    @media (max-width:900px) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} main,header {{ padding-left:16px; padding-right:16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{_esc(target)} Regression Report</h1>
    <div class="muted">run_id: <code>{_esc(run_id)}</code> · generated_at: <code>{_esc(generated_at)}</code></div>
  </header>
  <main>
    <section class="grid">
      <div class="metric"><span>用例总数</span><strong>{_esc(summary.get("total_cases", 0))}</strong></div>
      <div class="metric"><span>通过用例</span><strong>{_esc(summary.get("passed_cases", 0))}</strong></div>
      <div class="metric"><span>失败用例</span><strong>{_esc(summary.get("failed_cases", 0))}</strong></div>
      <div class="metric"><span>结构性提示</span><strong>{_esc(summary.get("warning_cases", 0))}</strong></div>
      <div class="metric"><span>内容疑似回归</span><strong>{_esc(summary.get("suspected_cases", 0))}</strong></div>
      <div class="metric"><span>通过率</span><strong>{float(summary.get("pass_rate", 0)):.1%}</strong></div>
      <div class="metric"><span>总耗时（秒）</span><strong>{_esc(summary.get("total_duration_seconds", 0))}</strong></div>
      <div class="metric"><span>原始差异数</span><strong>{_esc(summary.get("total_diffs", 0))}</strong></div>
    </section>
    {_render_diagnostic_summary(summary)}
    <section class="section">
      <h2>用例概览</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>用例</th><th>状态</th><th>耗时（秒）</th><th>上传耗时（秒）</th><th>解析元素</th><th>表格</th><th>页数</th><th>原始差异</th><th>评估结论</th><th>失败分类</th><th>预览</th></tr></thead>
          <tbody>{case_overview}</tbody>
        </table>
      </div>
    </section>
    {case_sections}
  </main>
</body>
</html>
"""


def _render_case_html(entry: ReportEntry, run_dir: Path) -> str:
    endpoint = _endpoint_from_run_dir(run_dir)
    key_result_rows = _metric_rows(key_metric_rows(endpoint, entry.result_metrics))
    result_rows = _dict_rows(entry.result_metrics)
    key_diff_rows = _metric_rows(diff_metric_rows(entry.diff_metrics))
    diff_rows = _dict_rows(entry.diff_metrics)
    links = _case_artifact_links(entry, run_dir)
    diagnostic_html = _render_entry_diagnostic_html(entry)
    assessment_html = _render_mineru_assessment_html(entry, run_dir)
    exclusions_html = _render_compare_exclusions_html(entry)
    alignment_warning = _render_alignment_warning_html(entry.diff_detail)
    diff_samples = ""
    if entry.diff_detail:
        sorted_diffs = _sorted_diffs(entry.diff_detail)
        rows = "".join(
            "<tr>"
            f"<td>{_severity_badge(item)}</td>"
            f"<td><code>{_esc(item.get('type') or '')}</code></td>"
            f"<td><code>{_esc(item.get('path'))}</code></td>"
            f"<td><code>{_esc(_short(item.get('expected')))}</code></td>"
            f"<td><code>{_esc(_short(item.get('actual')))}</code></td>"
            "</tr>"
            for item in sorted_diffs[:50]
        )
        diff_samples = f"""
      <details>
        <summary>高级原始差异明细（{len(sorted_diffs)} 项）</summary>
        <div class="table-wrap"><table>
          <thead><tr><th>severity</th><th>type</th><th>path</th><th>expected</th><th>actual</th></tr></thead>
          <tbody>{rows}</tbody>
        </table></div>
      </details>
"""
    warnings = "".join(f"<li>{_esc(warning)}</li>" for warning in entry.warnings)
    error = f"<h3>错误信息</h3><pre>{_esc(entry.error_message)}</pre>" if entry.error_message else ""
    return f"""
    <section class="section" id="{_case_anchor(entry.case_name)}">
      <h2>{_esc(entry.case_name)} {_status_badge(entry.status)}</h2>
      <p class="muted">{_esc(entry.description or "")}</p>
      <div class="links">{links}</div>
      {diagnostic_html}
      {assessment_html}
      {exclusions_html}
      {alignment_warning}
      <h3>关键指标</h3>
      <div class="table-wrap"><table class="key-metrics-table"><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>{key_result_rows}</tbody></table></div>
      <details>
        <summary>完整指标明细</summary>
        <div class="table-wrap"><table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>{result_rows}</tbody></table></div>
      </details>
      <h3>差异摘要</h3>
      <div class="table-wrap"><table><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>{key_diff_rows}</tbody></table></div>
      <details>
        <summary>完整差异指标明细</summary>
        <div class="table-wrap"><table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>{diff_rows}</tbody></table></div>
      </details>
      {diff_samples}
      {f"<h3>Warnings</h3><ul>{warnings}</ul>" if warnings else ""}
      {error}
    </section>
"""


def _render_compare_exclusions_html(entry: ReportEntry) -> str:
    exclusions = entry.result_metrics.get("compare_exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        return ""
    rows = []
    for rule in exclusions:
        if not isinstance(rule, dict):
            continue
        selector = rule.get("selector") if isinstance(rule.get("selector"), dict) else {}
        scope = "整页" if rule.get("scope") == "page" else "单项"
        page = selector.get("page") or "-"
        item_type = selector.get("type") or "-"
        rows.append(
            "<tr>"
            f"<td>{_esc(rule.get('id') or '-')}</td>"
            f"<td>{_esc(scope)}</td>"
            f"<td>第 {_esc(page)} 页 / {_esc(item_type)}</td>"
            f"<td>{_esc(rule.get('expected_count', 0))} / {_esc(rule.get('actual_count', 0))}</td>"
            f"<td>{_esc(rule.get('reason') or '-')}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
      <details>
        <summary>本次比较已按人工审查排除 {len(rows)} 条规则</summary>
        <div class=\"table-wrap\"><table>
          <thead><tr><th>规则</th><th>范围</th><th>定位</th><th>排除项（预期 / 实际）</th><th>原因</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table></div>
      </details>
"""


def _mineru_assessment(entry: ReportEntry) -> dict[str, Any] | None:
    assessment = entry.result_metrics.get("mineru_assessment")
    return assessment if isinstance(assessment, dict) else None


def _mineru_assessment_overview(entry: ReportEntry) -> str:
    assessment = _mineru_assessment(entry)
    if assessment is None:
        return ""
    content = assessment.get("content") or {}
    return (
        f"{_status_badge(assessment.get('status') or 'unknown')}<br>"
        f"<span class=\"muted\">文本 {_percent(content.get('text_coverage'))} / "
        f"表格 {_percent(content.get('table_text_coverage'))}</span>"
    )


def _render_mineru_assessment_html(entry: ReportEntry, run_dir: Path) -> str:
    assessment = _mineru_assessment(entry)
    if assessment is None:
        return ""

    contract = assessment.get("contract") or {}
    content = assessment.get("content") or {}
    structure = assessment.get("structure") or {}
    status = str(assessment.get("status") or "unknown")
    severity = "high" if status in {"failed", "suspected"} else "medium" if status == "warning" else "low"
    metrics = [
        ("文本覆盖率", _percent(content.get("text_coverage"))),
        ("表格内容覆盖率", _percent(content.get("table_text_coverage"))),
        ("数字覆盖率", _percent(content.get("number_coverage"))),
        ("新增文本占比", _percent(content.get("added_text_ratio"))),
        ("预期 / 实际页数", f"{contract.get('expected_page_count', 0)} / {contract.get('actual_page_count', 0)}"),
        ("预期 / 实际元素数", f"{structure.get('expected_item_count', 0)} / {structure.get('actual_item_count', 0)}"),
    ]
    metric_rows = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>" for label, value in metrics
    )
    issue_rows = _assessment_issue_rows(contract.get("issues"), "基础契约")
    issue_rows += _assessment_issue_rows(content.get("issues"), "内容")
    issue_rows += _assessment_issue_rows(structure.get("warnings"), "结构性提示")
    issues = (
        "<h3>评估发现</h3><ul>" + issue_rows + "</ul>"
        if issue_rows
        else "<p class=\"muted\">未发现基础契约、内容或结构性问题。</p>"
    )

    focus_rows = "".join(
        "<tr>"
        f"<td>{_focus_page_link(entry, run_dir, page.get('page'))}</td>"
        f"<td>{_percent(page.get('text_coverage'))}</td>"
        f"<td>{_percent(page.get('table_text_coverage'))}</td>"
        f"<td>{_percent(page.get('added_text_ratio'))}</td>"
        f"<td>{_esc(page.get('expected_text_chars', 0))} / {_esc(page.get('actual_text_chars', 0))}</td>"
        "</tr>"
        for page in (content.get("focus_pages") or [])
        if isinstance(page, dict)
    )
    focus = ""
    if focus_rows:
        focus = f"""
          <h3>重点复核页</h3>
          <div class="table-wrap"><table>
            <thead><tr><th>页码</th><th>文本覆盖率</th><th>表格覆盖率</th><th>新增文本占比</th><th>预期 / 实际文本字符数</th></tr></thead>
            <tbody>{focus_rows}</tbody>
          </table></div>
        """
    return f"""
      <div class="diagnostic diagnostic-{_esc(severity)}">
        <h3>解析内容质量评估 {_status_badge(status)}</h3>
        <p>{_esc(assessment.get('summary') or '')}</p>
        <div class="table-wrap"><table>
          <thead><tr><th>指标</th><th>值</th></tr></thead>
          <tbody>{metric_rows}</tbody>
        </table></div>
        {issues}
        {focus}
      </div>
"""


def _focus_page_link(entry: ReportEntry, run_dir: Path, page: Any) -> str:
    page_text = str(page or "").strip()
    if not page_text:
        return "-"

    fragment = re.sub(r"[^0-9A-Za-z_-]", "", page_text)
    overlay_path = run_dir / "cases" / _safe_path_name(entry.case_name) / "diff_overlay.html"
    if not fragment or not overlay_path.is_file():
        return _esc(page_text)

    href = f"{_relative_link(overlay_path, run_dir)}#page-{fragment}"
    return f'<a href="{_esc(href)}" target="_blank" rel="noopener">第 {_esc(page_text)} 页</a>'


def _mineru_focus_pages(entry: ReportEntry) -> set[int]:
    assessment = _mineru_assessment(entry)
    if assessment is None:
        return set()
    content = assessment.get("content")
    if not isinstance(content, dict):
        return set()
    pages: set[int] = set()
    for item in content.get("focus_pages") or []:
        if not isinstance(item, dict):
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.add(page)
    return pages


def _assessment_issue_rows(issues: Any, prefix: str) -> str:
    if not isinstance(issues, list):
        return ""
    rows: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        page = f" (page {issue['page']})" if issue.get("page") else ""
        rows.append(f"<li><code>{_esc(prefix)}</code>: {_esc(issue.get('message') or issue.get('code') or '')}{_esc(page)}</li>")
    return "".join(rows)


def _percent(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def _mineru_assessment_markdown(entry: ReportEntry) -> str:
    assessment = _mineru_assessment(entry)
    if assessment is None:
        return ""
    content = assessment.get("content") or {}
    return (
        f"{_status_label(str(assessment.get('status') or ''))}；文本 {_percent(content.get('text_coverage'))}；"
        f"表格 {_percent(content.get('table_text_coverage'))}"
    )


def _dict_rows(value: Any, prefix: str = "") -> str:
    if not isinstance(value, dict):
        return f"<tr><td>{_esc(prefix or 'value')}</td><td>{_esc(value)}</td></tr>"
    rows: list[str] = []
    for key, child in value.items():
        child_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            rows.append(_dict_rows(child, child_key))
        else:
            rows.append(f"<tr><td><code>{_esc(child_key)}</code></td><td>{_esc(child)}</td></tr>")
    return "".join(rows) or "<tr><td colspan=\"2\" class=\"muted\">empty</td></tr>"


def _metric_rows(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return '<tr><td colspan="2" class="muted">empty</td></tr>'
    return "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>"
        for label, value in rows
    )


def _endpoint_from_run_dir(run_dir: Path) -> str:
    try:
        return run_dir.parent.parent.name
    except Exception:
        return ""


def _compare_exclusion_editor_url(entry: ReportEntry, run_dir: Path) -> str | None:
    endpoint = _endpoint_from_run_dir(run_dir)
    if not endpoint or endpoint not in COMPARE_EXCLUSION_ENDPOINTS:
        return None
    case_dir = Path(entry.expected_items_path).parent
    if not (case_dir / "case.yaml").is_file():
        return None
    return (
        f"/cases/{quote(endpoint, safe='')}/{quote(case_dir.name, safe='')}"
        f"/compare-exclusions/new?run_id={quote(run_dir.name, safe='')}"
    )


def _sorted_diffs(diffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        diffs,
        key=lambda diff: (
            SEVERITY_RANK.get(diff_severity(diff), SEVERITY_RANK["major"]),
            str(diff.get("path") or ""),
        ),
    )


def _severity_badge(diff: dict[str, Any]) -> str:
    severity = diff_severity(diff)
    return f'<span class="badge {severity}">{_esc(severity)}</span>'


def _render_alignment_warning_html(diff_detail: list[dict[str, Any]]) -> str:
    warnings = _detect_row_alignment_shifts(diff_detail)
    if not warnings:
        return ""
    cards = []
    for warning in warnings[:3]:
        samples = "".join(
            "<li>"
            f"<code>{_esc(sample['expected_path'])}</code> "
            f"expected <code>{_esc(_short(sample['text'], 80))}</code> "
            f"matches actual at <code>{_esc(sample['actual_path'])}</code>"
            "</li>"
            for sample in warning["samples"][:5]
        )
        cards.append(
            "<div class=\"diagnostic diagnostic-medium\">"
            f"<h3>疑似表格行错位：{_esc(warning['item'])}</h3>"
            f"<p>检测到同一表内有 {warning['count']} 个文本差异符合 "
            f"<strong>actual row = expected row {warning['offset']:+d}</strong> 的模式。"
            "这通常表示多识别/漏识别了一行，后续单元格被顺序比较放大成大量 diff。</p>"
            f"<ul>{samples}</ul>"
            "</div>"
        )
    return "\n".join(cards)


ROW_TEXT_DIFF_RE = re.compile(
    r"(?P<prefix>items\[[^\]]+\]\.rows)\[(?P<row>\d+)\]\[(?P<col>\d+)\]\.text$"
)


def _detect_row_alignment_shifts(diff_detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_texts: dict[str, dict[tuple[int, int], str]] = defaultdict(dict)
    actual_texts: dict[str, dict[tuple[int, int], str]] = defaultdict(dict)
    paths: dict[tuple[str, int, int, str], str] = {}

    for diff in diff_detail:
        match = ROW_TEXT_DIFF_RE.search(str(diff.get("path") or ""))
        if not match:
            continue
        item = match.group("prefix")
        row = int(match.group("row"))
        col = int(match.group("col"))
        expected = _normalized_cell_text(diff.get("expected"))
        actual = _normalized_cell_text(diff.get("actual"))
        if expected:
            expected_texts[item][(row, col)] = expected
            paths[(item, row, col, "expected")] = str(diff.get("path") or "")
        if actual:
            actual_texts[item][(row, col)] = actual
            paths[(item, row, col, "actual")] = str(diff.get("path") or "")

    warnings: list[dict[str, Any]] = []
    for item, expected_by_pos in expected_texts.items():
        actual_by_pos = actual_texts.get(item) or {}
        if len(expected_by_pos) < 4 or len(actual_by_pos) < 4:
            continue

        offset_counts: Counter[int] = Counter()
        offset_samples: dict[int, list[dict[str, Any]]] = defaultdict(list)
        actual_lookup = {
            (col, text): row
            for (row, col), text in actual_by_pos.items()
            if text and len(text) > 1
        }
        for (expected_row, col), text in expected_by_pos.items():
            if not text or len(text) <= 1:
                continue
            actual_row = actual_lookup.get((col, text))
            if actual_row is None:
                continue
            offset = actual_row - expected_row
            if offset == 0 or abs(offset) > 5:
                continue
            offset_counts[offset] += 1
            offset_samples[offset].append(
                {
                    "text": text,
                    "expected_path": paths.get((item, expected_row, col, "expected"), f"{item}[{expected_row}][{col}]"),
                    "actual_path": paths.get((item, actual_row, col, "actual"), f"{item}[{actual_row}][{col}]"),
                }
            )

        if not offset_counts:
            continue
        offset, count = offset_counts.most_common(1)[0]
        if count >= 3 and count >= max(3, int(len(expected_by_pos) * 0.12)):
            warnings.append(
                {
                    "item": item.replace(".rows", ""),
                    "offset": offset,
                    "count": count,
                    "samples": offset_samples[offset],
                }
            )

    return sorted(warnings, key=lambda warning: warning["count"], reverse=True)


def _normalized_cell_text(value: Any) -> str:
    if value in (None, "", "<missing>"):
        return ""
    return " ".join(str(value).split()).strip()


def _case_artifact_links(entry: ReportEntry, run_dir: Path) -> str:
    case_dir = run_dir / "cases" / _safe_path_name(entry.case_name)
    links = [
        ("metrics", case_dir / "metrics.json"),
        ("diff_detail", case_dir / "diff_detail.json"),
        ("actual_items", case_dir / "actual_items.json"),
    ]
    links.extend(_case_preview_targets(case_dir))
    extra_artifacts = entry.upload_summary.get("extra_artifacts")
    if isinstance(extra_artifacts, dict):
        for name, source in extra_artifacts.items():
            links.append((str(name), case_dir / Path(str(source)).name))
    rendered = [
        _artifact_link(_relative_link(path, run_dir), label)
        for label, path in links
        if Path(path).is_file()
    ]
    return "".join(rendered) or '<span class="muted">No archived artifacts</span>'


def _case_preview_links(entry: ReportEntry, run_dir: Path) -> str:
    case_dir = run_dir / "cases" / _safe_path_name(entry.case_name)
    links = _case_preview_targets(case_dir)
    source_href = _source_input_href(entry.input_file, run_dir)
    if not links and source_href is None:
        return '<span class="muted">-</span>'
    rendered: list[str] = []
    if source_href is not None:
        rendered.append(_artifact_link(source_href, "source_input"))
    rendered.extend(
        _artifact_link(_relative_link(path, run_dir), label)
        for label, path in links
    )
    return f'<div class="links">{"".join(rendered)}</div>'


def _source_input_href(input_file: str, run_dir: Path) -> str | None:
    source = Path(input_file)
    if not source.is_file():
        return None
    try:
        return Path(os.path.relpath(source.resolve(), run_dir.resolve())).as_posix()
    except ValueError:
        return None


def _artifact_link(href: str, label: str) -> str:
    return f'<a href="{_esc(_versioned_artifact_href(href))}" target="_blank" rel="noopener">{_esc(artifact_label(label))}</a>'


def _versioned_artifact_href(href: str) -> str:
    if href.lower().endswith((".html", ".htm")):
        return f"{href}?v=20260728-coordinate-inspector"
    return href


def _case_preview_targets(case_dir: Path) -> list[tuple[str, Path]]:
    patterns = [
        ("merge", "*.merge_tables_preview.html"),
        ("subject_match_source_table", "subject_match_source_table.html"),
        ("subject_match_table", "subject_match_table.html"),
        ("grid", "*.statement_grid_preview.html"),
        ("actual_preview", "actual_preview.html"),
        ("actual_overlay", "actual_overlay.html"),
        ("expected_preview", "expected_preview.html"),
        ("expected_overlay", "expected_overlay.html"),
        ("diff_preview", "diff_preview.html"),
        ("diff_overlay", "diff_overlay.html"),
        ("table_diff_preview", "table_diff_preview.html"),
        ("preview", "preview.html"),
        ("overlay", "overlay.html"),
        ("summary", "*.statement_diff_summary.md"),
    ]
    results: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, pattern in patterns:
        for path in sorted(case_dir.glob(pattern)):
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                results.append((label, path))
                seen.add(resolved)
    return results


def _case_anchor(case_name: str) -> str:
    return "case-" + _safe_path_name(case_name)


def _relative_link(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _status_badge(status: str) -> str:
    return f'<span class="badge {_esc(status)}">{_esc(_status_label(status))}</span>'


def _status_label(status: str) -> str:
    labels = {
        "passed": "通过",
        "failed": "失败",
        "warning": "结构性提示",
        "suspected": "疑似回归",
        "mineru_contract_failed": "基础契约失败",
        "mineru_content_suspected": "内容疑似回归",
        "regression_diff": "基线差异",
    }
    return labels.get(status, status)


def _latest_html_redirect(run_id: str) -> str:
    target = f"runs/{run_id}/report.html"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={_esc(target)}">
  <title>Latest Regression Report</title>
</head>
<body>
  <a href="{_esc(target)}">Open latest report</a>
</body>
</html>
"""


def _entry_from_payload(payload: dict[str, Any]) -> ReportEntry:
    fields = ReportEntry.__dataclass_fields__
    values: dict[str, Any] = {}
    for name, field in fields.items():
        if name in payload:
            values[name] = payload[name]
        elif field.type in (dict[str, Any], "dict[str, Any]"):
            values[name] = {}
        elif field.type in (list[str], "list[str]"):
            values[name] = []
        elif field.type in (list[dict[str, Any]], "list[dict[str, Any]]"):
            values[name] = []
        elif field.type in (float, "float"):
            values[name] = 0.0
        else:
            values[name] = ""
    return ReportEntry(**values)


def _entry_diagnostic(entry: ReportEntry):
    """Only surface diagnostics for failures, not tolerated structural drift."""
    if entry.status in {"passed", "warning"} and not entry.error_message:
        return None
    return diagnose_failure(
        error_message=entry.error_message,
        diff_total=int(entry.diff_metrics.get("total", 0)),
        failure_category=entry.failure_category,
    )


def _entry_payload(entry: ReportEntry) -> dict[str, Any]:
    payload = asdict(entry)
    diagnostic = _entry_diagnostic(entry)
    payload["diagnostic"] = diagnostic.to_dict() if diagnostic else None
    return payload


def _diagnostic_overview(entry: ReportEntry) -> str:
    diagnostic = _entry_diagnostic(entry)
    if diagnostic:
        return f"{_status_badge(diagnostic.category)}"
    return _esc(entry.failure_category)


def _render_diagnostic_summary(summary: dict[str, Any]) -> str:
    counts = summary.get("diagnostic_counts") or {}
    if not counts:
        return ""
    rows = "".join(
        f"<tr><td>{_esc(_status_label(str(category)))}</td><td>{_esc(count)}</td></tr>"
        for category, count in counts.items()
    )
    return f"""
    <section class="section">
      <h2>问题分类统计</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>分类</th><th>数量</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>
"""


def _render_entry_diagnostic_html(entry: ReportEntry) -> str:
    diagnostic = _entry_diagnostic(entry)
    if diagnostic is None:
        return ""
    steps = "".join(f"<li>{_esc(step)}</li>" for step in diagnostic.next_steps)
    evidence = ""
    if diagnostic.evidence:
        evidence = f"<h3>证据</h3><pre>{_esc(chr(10).join(diagnostic.evidence))}</pre>"
    return f"""
      <div class="diagnostic diagnostic-{_esc(diagnostic.severity)}">
        <h3>{_esc(diagnostic.title)} {_status_badge(diagnostic.category)}</h3>
        <p>{_esc(diagnostic.reason)}</p>
        <h3>建议下一步</h3>
        <ul>{steps}</ul>
        {evidence}
      </div>
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _write_case_artifacts(
    run_dir: Path,
    entries: list[ReportEntry],
    *,
    rebuild_diffs_only: bool = False,
) -> None:
    cases_dir = run_dir / "cases"
    for entry in entries:
        case_dir = cases_dir / _safe_path_name(entry.case_name)
        case_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_diff_artifacts(case_dir)
        diagnostic = _entry_diagnostic(entry)
        dump_json(case_dir / "metrics.json", {
            "result_metrics": entry.result_metrics,
            "diff_metrics": entry.diff_metrics,
            "diagnostic": diagnostic.to_dict() if diagnostic else None,
            "file_info": entry.file_info,
            "upload_summary": entry.upload_summary,
            "duration_seconds": entry.duration_seconds,
            "upload_elapsed_seconds": entry.upload_elapsed_seconds,
            "http_status_code": entry.http_status_code,
            "failure_category": entry.failure_category,
            "warnings": entry.warnings,
        })
        dump_json(case_dir / "diff_detail.json", entry.diff_detail)
        actual_path = Path(entry.actual_items_path)
        actual_items = _load_list(actual_path)
        if actual_path.is_file():
            shutil.copy2(actual_path, case_dir / "actual_items.json")
        expected_path = Path(entry.expected_items_path)
        expected_items = _load_list(expected_path)
        comparison_expected, comparison_actual, _ = filter_compare_exclusions(
            expected_items,
            actual_items,
            load_compare_exclusions(expected_path.parent),
        )
        if actual_items is not None:
            if not rebuild_diffs_only:
                render_content_preview(
                    actual_items,
                    case_dir / "actual_preview.md",
                    case_dir / "actual_preview.html",
                    title="Actual Parsed Content Preview",
                )
            render_overlay_preview(
                Path(entry.input_file),
                actual_items,
                case_dir,
                output_name="actual_overlay.html",
                title="实际结果坐标预览",
            )
        if expected_items is not None:
            if not rebuild_diffs_only:
                render_content_preview(
                    expected_items,
                    case_dir / "expected_preview.md",
                    case_dir / "expected_preview.html",
                    title="Expected Parsed Content Preview",
                )
            render_overlay_preview(
                Path(entry.input_file),
                expected_items,
                case_dir,
                output_name="expected_overlay.html",
                title="预期答案坐标预览",
            )
        if entry.diff_detail:
            render_diff_preview(comparison_expected, comparison_actual, entry.diff_detail, case_dir / "diff_preview.html")
            render_table_diff_preview(comparison_expected, comparison_actual, entry.diff_detail, case_dir / "table_diff_preview.html")
            focus_pages = _mineru_focus_pages(entry)
            changed_indexes = diff_item_indexes(entry.diff_detail)
            if comparison_actual is not None or comparison_expected is not None:
                render_overlay_preview(
                    Path(entry.input_file),
                    comparison_actual or [],
                    case_dir,
                    output_name="diff_overlay.html",
                    title="差异定位工作台",
                    highlight_indexes=changed_indexes,
                    diff_detail=entry.diff_detail,
                    comparison_items=comparison_expected,
                    extra_pages=focus_pages,
                    include_missing_cell_search=False,
                    ignore_rule_url=_compare_exclusion_editor_url(entry, run_dir),
                )
        if isinstance(entry.result_metrics.get("financial_subject_match"), dict):
            write_subject_match_table_artifact(
                actual_items,
                expected_items,
                case_dir,
                request_payload=_subject_match_request_payload(entry),
            )
        extra_artifacts = entry.upload_summary.get("extra_artifacts")
        if isinstance(extra_artifacts, dict):
            copied: dict[str, str] = {}
            for name, source in extra_artifacts.items():
                source_path = Path(str(source))
                if source_path.is_file():
                    target_path = case_dir / source_path.name
                    shutil.copy2(source_path, target_path)
                    copied[str(name)] = str(target_path)
            if copied:
                dump_json(case_dir / "extra_artifacts.json", copied)


def _remove_stale_diff_artifacts(case_dir: Path) -> None:
    for filename in (
        "diff_preview.html",
        "diff_overlay.html",
        "expected_diff_overlay.html",
        "table_diff_preview.html",
    ):
        (case_dir / filename).unlink(missing_ok=True)


def _load_list(path: Path) -> list[Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, list) else None


def _load_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _subject_match_request_payload(entry: ReportEntry) -> dict[str, Any] | None:
    from_input = _load_object(Path(entry.input_file))
    if from_input:
        return from_input
    request_path = entry.upload_summary.get("subject_match_request_path")
    if request_path:
        return _load_object(Path(str(request_path)))
    return None


def _write_index(report_dir: Path) -> None:
    runs_dir = report_dir / "runs"
    rows: list[tuple[str, dict[str, Any]]] = []
    if runs_dir.exists():
        for summary_path in sorted(runs_dir.glob("*/summary.json"), reverse=True):
            try:
                import json

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append((summary_path.parent.name, summary))

    lines = [
        "# Regression Report Index",
        "",
        "| run_id | pass_rate | passed | failed | duration_s | diffs | report |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run_id, summary in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run_id}`",
                    f"{float(summary.get('pass_rate', 0)):.2%}",
                    str(summary.get("passed_cases", 0)),
                    str(summary.get("failed_cases", 0)),
                    str(summary.get("total_duration_seconds", 0)),
                    str(summary.get("total_diffs", 0)),
                    f"[report](runs/{run_id}/report.md)",
                ]
            )
            + " |"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_case(entry: ReportEntry, run_dir: Path) -> list[str]:
    result = entry.result_metrics
    diff = entry.diff_metrics
    lines = [
        f"## {entry.case_name}",
        "",
        f"- status: {entry.status}",
        f"- description: {entry.description or ''}",
        f"- input_file: `{entry.input_file}`",
        f"- input_sha256: `{entry.file_info.get('sha256', '')}`",
        f"- input_size_bytes: {entry.file_info.get('size_bytes', 0)}",
        f"- expected_items_path: `{entry.expected_items_path}`",
        f"- actual_items_path: `{entry.actual_items_path}`",
        f"- duration_seconds: {round(entry.duration_seconds, 3)}",
        f"- upload_elapsed_seconds: {round(entry.upload_elapsed_seconds, 3)}",
        f"- http_status_code: {entry.http_status_code if entry.http_status_code is not None else ''}",
        f"- failure_category: `{entry.failure_category}`" if entry.failure_category else "- failure_category:",
        f"- diff_summary: {entry.diff_summary}",
        "",
        "### Result Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| item_count | {result.get('item_count', 0)} |",
        f"| page_count | {result.get('page_count', 0)} |",
        f"| text_count | {result.get('text_count', 0)} |",
        f"| table_count | {result.get('table_count', 0)} |",
        f"| image_count | {result.get('image_count', 0)} |",
        f"| empty_text_ratio | {result.get('empty_text_ratio', 0)} |",
        f"| max_table_rows | {result.get('max_table_rows', 0)} |",
        "",
    ]

    assessment = _mineru_assessment(entry)
    if assessment:
        contract = assessment.get("contract") or {}
        content = assessment.get("content") or {}
        structure = assessment.get("structure") or {}
        lines.extend(
            [
                "### MinerU 解析质量评估",
                "",
                f"- 结论: `{_status_label(str(assessment.get('status') or ''))}`",
                f"- 摘要: {assessment.get('summary', '')}",
                f"- 文本覆盖率: {_percent(content.get('text_coverage'))}",
                f"- 表格内容覆盖率: {_percent(content.get('table_text_coverage'))}",
                f"- 数字覆盖率: {_percent(content.get('number_coverage'))}",
                f"- 新增文本占比: {_percent(content.get('added_text_ratio'))}",
                f"- 预期 / 实际页数: {contract.get('expected_page_count', 0)} / {contract.get('actual_page_count', 0)}",
                f"- 预期 / 实际元素数: {structure.get('expected_item_count', 0)} / {structure.get('actual_item_count', 0)}",
                "",
            ]
        )
        findings = [
            *(contract.get("issues") or []),
            *(content.get("issues") or []),
            *(structure.get("warnings") or []),
        ]
        if findings:
            lines.extend(["#### 评估发现", ""])
            for finding in findings:
                if isinstance(finding, dict):
                    page = f" (page {finding['page']})" if finding.get("page") else ""
                    lines.append(f"- `{finding.get('code', '')}`: {finding.get('message', '')}{page}")
            lines.append("")

    if result.get("type_counts"):
        lines.extend(["### Type Counts", "", "| type | count |", "| --- | ---: |"])
        for item_type, count in result["type_counts"].items():
            lines.append(f"| `{item_type}` | {count} |")
        lines.append("")

    financial_metadata = result.get("financial_metadata")
    if isinstance(financial_metadata, dict):
        lines.extend(
            [
                "### Financial Metadata Metrics",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| success | {financial_metadata.get('success', False)} |",
                f"| metadata_field_count | {financial_metadata.get('metadata_field_count', 0)} |",
                f"| metadata_non_empty_field_count | {financial_metadata.get('metadata_non_empty_field_count', 0)} |",
                f"| metadata_source_element_count | {financial_metadata.get('metadata_source_element_count', 0)} |",
                f"| three_statements_count | {financial_metadata.get('three_statements_count', 0)} |",
                f"| change_explanation_count | {financial_metadata.get('change_explanation_count', 0)} |",
                f"| has_company_full_name | {financial_metadata.get('has_company_full_name', False)} |",
                f"| has_current_period | {financial_metadata.get('has_current_period', False)} |",
                f"| has_audit | {financial_metadata.get('has_audit', False)} |",
                "",
            ]
        )
        recognized_fields = financial_metadata.get("recognized_fields") or []
        if recognized_fields:
            lines.extend(["### Recognized Financial Metadata Fields", ""])
            for field in recognized_fields:
                lines.append(f"- `{field}`")
            lines.append("")

        missing_core_fields = financial_metadata.get("missing_core_fields") or []
        if missing_core_fields:
            lines.extend(["### Missing Core Financial Metadata Fields", ""])
            for field in missing_core_fields:
                lines.append(f"- `{field}`")
            lines.append("")

    financial_three_statements = result.get("financial_three_statements")
    if isinstance(financial_three_statements, dict):
        lines.extend(
            [
                "### Financial Three Statements Metrics",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| statement_count | {financial_three_statements.get('statement_count', 0)} |",
                f"| total_rows | {financial_three_statements.get('total_rows', 0)} |",
                f"| total_lines | {financial_three_statements.get('total_lines', 0)} |",
                f"| tables_with_rows | {financial_three_statements.get('tables_with_rows', 0)} |",
                f"| tables_with_lines | {financial_three_statements.get('tables_with_lines', 0)} |",
                f"| max_rows | {financial_three_statements.get('max_rows', 0)} |",
                "",
            ]
        )
        statement_names = financial_three_statements.get("statement_names") or []
        if statement_names:
            lines.extend(["### Financial Statement Names", ""])
            for name in statement_names:
                lines.append(f"- `{name}`")
            lines.append("")

        currency_units = financial_three_statements.get("currency_units") or []
        if currency_units:
            lines.extend(["### Financial Statement Currency Units", ""])
            for unit in currency_units:
                lines.append(f"- `{unit}`")
            lines.append("")

    financial_org_tables = result.get("financial_org_tables")
    if isinstance(financial_org_tables, dict):
        lines.extend(
            [
                "### Financial Org Tables Metrics",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| org_table_count | {financial_org_tables.get('org_table_count', 0)} |",
                f"| financial_statement_table_count | {financial_org_tables.get('financial_statement_table_count', 0)} |",
                f"| total_rows | {financial_org_tables.get('total_rows', 0)} |",
                f"| max_rows | {financial_org_tables.get('max_rows', 0)} |",
                f"| tables_with_bbox | {financial_org_tables.get('tables_with_bbox', 0)} |",
                f"| bbox_coverage_ratio | {financial_org_tables.get('bbox_coverage_ratio', 0)} |",
                "",
            ]
        )
        table_names = financial_org_tables.get("table_names") or []
        if table_names:
            lines.extend(["### Financial Org Table Names", ""])
            for name in table_names:
                lines.append(f"- `{name}`")
            lines.append("")

        pages = financial_org_tables.get("pages") or []
        if pages:
            lines.extend(["### Financial Org Table Pages", ""])
            for page in pages:
                lines.append(f"- `{page}`")
            lines.append("")

    financial_merge_tables = result.get("financial_merge_tables")
    if isinstance(financial_merge_tables, dict):
        lines.extend(
            [
                "### Financial Merge Tables Metrics",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| merge_count | {financial_merge_tables.get('merge_count', 0)} |",
                f"| source_id_count | {financial_merge_tables.get('source_id_count', 0)} |",
                f"| output_table_count | {financial_merge_tables.get('output_table_count', 0)} |",
                f"| output_total_rows | {financial_merge_tables.get('output_total_rows', 0)} |",
                "",
            ]
        )
        source_ids = financial_merge_tables.get("source_ids") or []
        if source_ids:
            lines.extend(["### Merge Source IDs", ""])
            for source_id in source_ids:
                lines.append(f"- `{source_id}`")
            lines.append("")

        output_table_names = financial_merge_tables.get("output_table_names") or []
        if output_table_names:
            lines.extend(["### Merge Output Table Names", ""])
            for name in output_table_names:
                lines.append(f"- `{name}`")
            lines.append("")

    financial_subject_match = result.get("financial_subject_match")
    if isinstance(financial_subject_match, dict):
        lines.extend(
            [
                "### Financial Subject Match Metrics",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| row_count | {financial_subject_match.get('row_count', 0)} |",
                f"| cell_count | {financial_subject_match.get('cell_count', 0)} |",
                f"| matched_subject_count | {financial_subject_match.get('matched_subject_count', 0)} |",
                f"| matched_subject_rate | {financial_subject_match.get('matched_subject_rate', 0)} |",
                f"| empty_subject_count | {financial_subject_match.get('empty_subject_count', 0)} |",
                f"| unique_subject_count | {financial_subject_match.get('unique_subject_count', 0)} |",
                "",
            ]
        )
        matched_subject_ids = financial_subject_match.get("matched_subject_ids") or []
        if matched_subject_ids:
            lines.extend(["### Matched Subject IDs", ""])
            for subject_id in matched_subject_ids:
                lines.append(f"- `{subject_id}`")
            lines.append("")

    if diff.get("total", 0):
        if diff.get("by_severity"):
            lines.extend(["### Diff Severity", "", "| severity | count |", "| --- | ---: |"])
            for severity in SEVERITY_RANK:
                count = diff["by_severity"].get(severity)
                if count:
                    lines.append(f"| `{severity}` | {count} |")
            lines.append("")

        lines.extend(["### Diff Metrics", "", "| kind | count |", "| --- | ---: |"])
        for kind, count in diff.get("by_kind", {}).items():
            lines.append(f"| `{kind}` | {count} |")
        lines.append("")

        alignment_warnings = _detect_row_alignment_shifts(entry.diff_detail)
        if alignment_warnings:
            lines.extend(["### Suspected Row Alignment Shift", ""])
            for warning in alignment_warnings[:3]:
                lines.append(
                    f"- `{warning['item']}`: {warning['count']} text diffs match "
                    f"`actual row = expected row {warning['offset']:+d}`."
                )
                for sample in warning["samples"][:5]:
                    lines.append(
                        f"  - `{_short(sample['text'], 80)}`: "
                        f"`{sample['expected_path']}` -> `{sample['actual_path']}`"
                    )
            lines.append("")

        if diff.get("top_fields"):
            lines.extend(["### Top Diff Fields", "", "| field | count |", "| --- | ---: |"])
            for field, count in diff["top_fields"].items():
                lines.append(f"| `{field}` | {count} |")
            lines.append("")

        lines.extend(
            [
                "### 高级原始差异明细",
                "",
                "| severity | type | path | expected | actual |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in _sorted_diffs(entry.diff_detail)[:50]:
            lines.append(
                f"| `{diff_severity(item)}` | `{item.get('type') or ''}` | `{item.get('path')}` | "
                f"`{_short(item.get('expected'))}` | `{_short(item.get('actual'))}` |"
            )
        lines.append("")

    diagnostic = _entry_diagnostic(entry)
    if diagnostic:
        lines.extend(
            [
                "### Diagnostic",
                "",
                f"- category: `{diagnostic.category}`",
                f"- severity: `{diagnostic.severity}`",
                f"- title: {diagnostic.title}",
                f"- reason: {diagnostic.reason}",
                "",
                "#### Next Steps",
                "",
            ]
        )
        for step in diagnostic.next_steps:
            lines.append(f"- {step}")
        if diagnostic.evidence:
            lines.extend(["", "#### Evidence", ""])
            for item in diagnostic.evidence:
                lines.append(f"- `{_short(item, 180)}`")
        lines.append("")

    if entry.warnings:
        lines.extend(["### Warnings", ""])
        for warning in entry.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    extra_artifacts = entry.upload_summary.get("extra_artifacts")
    if isinstance(extra_artifacts, dict) and extra_artifacts:
        lines.extend(["### Extra Artifacts", ""])
        for name, path in extra_artifacts.items():
            archived_path = run_dir / "cases" / _safe_path_name(entry.case_name) / Path(str(path)).name
            lines.append(f"- {name}: `{archived_path}`")
        lines.append("")

    if entry.error_message:
        lines.extend(["### Error", "", f"```text\n{entry.error_message}\n```", ""])

    return lines


def _short(value: Any, limit: int = 120) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_path_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe.strip("._") or "case"


def _write_report_store_best_effort(
    report_dir: Path,
    entries: list[ReportEntry],
    run_id: str,
    target: str,
    summary: dict[str, Any],
    generated_at: str,
) -> None:
    try:
        store_report_run(report_dir, entries, run_id, target, summary, generated_at=generated_at)
    except Exception as exc:
        warning_path = report_dir / "report_store_warning.txt"
        warning_path.write_text(
            f"failed to write sqlite report store for run_id={run_id}: {exc}\n",
            encoding="utf-8",
        )
