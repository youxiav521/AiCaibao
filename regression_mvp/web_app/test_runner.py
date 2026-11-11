from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

from regression_mvp.utils.diagnostics import diagnose_failure
from regression_mvp.utils.metrics import utc_timestamp
from regression_mvp.utils.report_store import write_dashboard
from regression_mvp.utils.reporter import write_html_reports_from_existing


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT_DIR.parent
LOG_DIR = ROOT_DIR / "reports" / "web_runs"
HISTORY_PATH = LOG_DIR / "index.jsonl"

TEST_TARGETS = {
    "file_read": ROOT_DIR / "tests" / "test_file_read.py",
    "mineru_to_json": ROOT_DIR / "tests" / "test_mineru_to_json.py",
    "mineru_upload_async": ROOT_DIR / "tests" / "test_mineru_upload_async.py",
    "ade_jobs_async": ROOT_DIR / "tests" / "test_ade_jobs_async.py",
    "financial_metadata": ROOT_DIR / "tests" / "test_financial_metadata.py",
    "financial_three_statements": ROOT_DIR / "tests" / "test_financial_three_statements.py",
    "financial_org_tables": ROOT_DIR / "tests" / "test_financial_org_tables.py",
    "financial_merge_tables": ROOT_DIR / "tests" / "test_financial_merge_tables.py",
    "financial_subject_match": ROOT_DIR / "tests" / "test_financial_subject_match.py",
}


@dataclass
class RunState:
    status: str = "idle"
    target: str = ""
    started_at: str = ""
    finished_at: str = ""
    return_code: int | None = None
    log_path: str = ""
    command: list[str] = field(default_factory=list)
    message: str = ""
    selected_targets: list[str] = field(default_factory=list)
    selected_case_name: str = ""
    summary_report_path: str = ""


_lock = threading.Lock()
_state = RunState()
_thread: threading.Thread | None = None


def available_targets() -> dict[str, Path]:
    return TEST_TARGETS.copy()


def start_run(target: str) -> RunState:
    if target == "all":
        return start_targets(list(TEST_TARGETS), label="all")
    return start_targets([target], label=target)


def start_case(endpoint: str, case_name: str) -> RunState:
    if endpoint not in TEST_TARGETS:
        raise ValueError(f"unknown test target: {endpoint}")
    normalized_case_name = str(case_name or "").strip()
    if not normalized_case_name:
        raise ValueError("case name is required")
    return start_targets(
        [endpoint],
        label=f"{endpoint} / {normalized_case_name}",
        case_name=normalized_case_name,
    )


def start_targets(targets: list[str], label: str | None = None, *, case_name: str = "") -> RunState:
    global _state, _thread
    selected_targets = _normalize_targets(targets)
    target_label = label or _target_label(selected_targets)
    with _lock:
        if _state.status == "running":
            return _state
        command = _build_command_for_targets(selected_targets)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        safe_label = _safe_log_label(target_label)
        stamp = utc_timestamp().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        log_path = LOG_DIR / f"{stamp}_{safe_label}.log"
        _state = RunState(
            status="running",
            target=target_label,
            started_at=utc_timestamp(),
            log_path=str(log_path),
            command=command,
            selected_targets=selected_targets,
            selected_case_name=case_name,
            message="测试运行中",
        )
        _thread = threading.Thread(
            target=_run_command,
            args=(command, log_path, selected_targets, _state.started_at, case_name),
            daemon=True,
        )
        _thread.start()
        return _state


def current_status() -> dict[str, Any]:
    with _lock:
        payload = _state.__dict__.copy()
    payload["log_tail"] = read_log_tail(payload.get("log_path") or "")
    payload["diagnostic"] = _diagnostic_for_status(payload)
    payload["is_running"] = payload["status"] == "running"
    return _enrich_run_report_links(payload)


def recent_runs(limit: int = 20) -> list[dict[str, Any]]:
    if not HISTORY_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in HISTORY_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return [_enrich_run_report_links(row) for row in reversed(rows[-limit:])]


def read_log_tail(log_path: str, max_chars: int = 12000) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _build_command(target: str) -> list[str]:
    if target == "all":
        return [sys.executable, "-m", "pytest", *[str(path) for path in TEST_TARGETS.values()]]
    if target not in TEST_TARGETS:
        raise ValueError(f"unknown test target: {target}")
    return [sys.executable, "-m", "pytest", str(TEST_TARGETS[target])]


def _build_command_for_targets(targets: list[str]) -> list[str]:
    if len(targets) == len(TEST_TARGETS):
        return _build_command("all")
    return [sys.executable, "-m", "pytest", *[str(TEST_TARGETS[target]) for target in targets]]


def _normalize_targets(targets: list[str]) -> list[str]:
    selected: list[str] = []
    for target in targets:
        if target not in TEST_TARGETS:
            raise ValueError(f"unknown test target: {target}")
        if target not in selected:
            selected.append(target)
    if not selected:
        raise ValueError("no test target selected")
    return selected


def _target_label(targets: list[str]) -> str:
    if len(targets) == 1:
        return targets[0]
    if len(targets) == len(TEST_TARGETS):
        return "all"
    return "multi:" + ",".join(targets)


def _safe_log_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    return safe.strip("_") or "run"


def _run_command(command: list[str], log_path: Path, selected_targets: list[str], started_at: str, case_name: str = "") -> None:
    return_code = -1
    message = ""
    summary_report_path = ""
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"$ {' '.join(command)}\n\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=_subprocess_env(case_name),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = process.wait()
        _refresh_reports()
        summary_report_path = _write_aggregate_report(selected_targets, started_at)
        message = "测试完成" if return_code == 0 else "测试失败"
    except Exception as exc:
        message = f"测试执行异常: {exc}"
        with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"\n{message}\n")
    finally:
        with _lock:
            _state.status = "success" if return_code == 0 else "failed"
            _state.return_code = return_code
            _state.finished_at = utc_timestamp()
            _state.message = message
            _state.summary_report_path = summary_report_path
            snapshot = _state.__dict__.copy()
        _append_history(snapshot)


def _refresh_reports() -> None:
    report_root = ROOT_DIR / "reports"
    write_dashboard(report_root)
    write_html_reports_from_existing(report_root)


def _diagnostic_for_status(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("status") != "failed":
        return None
    diagnostic = diagnose_failure(
        error_message=str(payload.get("message") or ""),
        log_text=str(payload.get("log_tail") or ""),
    )
    return diagnostic.to_dict() if diagnostic else None


def _append_history(snapshot: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    diagnostic = None
    if snapshot.get("status") == "failed":
        log_tail = read_log_tail(str(snapshot.get("log_path") or ""), max_chars=6000)
        diagnostic = diagnose_failure(
            error_message=str(snapshot.get("message") or ""),
            log_text=log_tail,
        )
    row = {
        "target": snapshot.get("target", ""),
        "status": snapshot.get("status", ""),
        "started_at": snapshot.get("started_at", ""),
        "finished_at": snapshot.get("finished_at", ""),
        "return_code": snapshot.get("return_code"),
        "log_path": snapshot.get("log_path", ""),
        "selected_targets": snapshot.get("selected_targets", []),
        "selected_case_name": snapshot.get("selected_case_name", ""),
        "summary_report_path": snapshot.get("summary_report_path", ""),
        "diagnostic_category": diagnostic.category if diagnostic else "",
        "diagnostic_title": diagnostic.title if diagnostic else "",
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _enrich_run_report_links(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.setdefault("report_html_path", "")
    row.setdefault("report_md_path", "")
    row.setdefault("report_json_path", "")
    row.setdefault("report_detail_url", "")
    row.setdefault("selected_case_name", "")
    if row.get("summary_report_path"):
        return row

    targets = row.get("selected_targets")
    if not isinstance(targets, list) or not targets:
        target = str(row.get("target") or "")
        targets = [target] if target in TEST_TARGETS else []
    if len(targets) != 1:
        return row

    report = _find_single_endpoint_report(
        endpoint=str(targets[0]),
        started_at=str(row.get("started_at") or ""),
        finished_at=str(row.get("finished_at") or ""),
    )
    if not report:
        return row
    row["report_html_path"] = report.get("report_html_path") or ""
    row["report_md_path"] = report.get("report_md_path") or ""
    row["report_json_path"] = report.get("report_json_path") or ""
    row["report_detail_url"] = f"/reports/{report['endpoint']}/{report['run_id']}"
    return row


def _subprocess_env(case_name: str = "") -> dict[str, str]:
    env = os.environ.copy()
    if case_name:
        env["REGRESSION_MVP_CASE_NAME"] = case_name
    else:
        env.pop("REGRESSION_MVP_CASE_NAME", None)
    return env


def _find_single_endpoint_report(endpoint: str, started_at: str, finished_at: str) -> dict[str, Any] | None:
    from regression_mvp.web_app import data as web_data

    candidates = [
        report for report in web_data.recent_runs(limit=1000)
        if report.get("endpoint") == endpoint and _report_in_run_window(report, started_at, finished_at)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: str(item.get("generated_at") or ""), reverse=True)[0]


def _report_in_run_window(report: dict[str, Any], started_at: str, finished_at: str) -> bool:
    generated_at = str(report.get("generated_at") or "")
    if started_at and generated_at < started_at:
        return False
    if finished_at and generated_at > finished_at:
        return False
    return True


def _write_aggregate_report(selected_targets: list[str], started_at: str) -> str:
    if len(selected_targets) <= 1:
        return ""
    from regression_mvp.web_app import data as web_data

    latest_by_endpoint = {row["endpoint"]: row for row in web_data.latest_endpoint_status()}
    rows = []
    for target in selected_targets:
        row = latest_by_endpoint.get(target)
        if row and str(row.get("generated_at") or "") >= started_at:
            rows.append(row)
        else:
            rows.append(_missing_report_row(target))
    if not rows:
        return ""

    total_cases = sum(int(row.get("total_cases") or 0) for row in rows)
    passed_cases = sum(int(row.get("passed_cases") or 0) for row in rows)
    failed_cases = sum(int(row.get("failed_cases") or 0) for row in rows)
    total_diffs = sum(int(row.get("total_diffs") or 0) for row in rows)
    total_duration = sum(float(row.get("total_duration_seconds") or 0) for row in rows)
    failed_endpoints = sum(
        1 for row in rows
        if row.get("status") == "no_report" or int(row.get("failed_cases") or 0) > 0
    )
    pass_rate = (passed_cases / total_cases) if total_cases else 0.0

    stamp = utc_timestamp().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    json_path = LOG_DIR / f"{stamp}_aggregate.json"
    html_path = LOG_DIR / f"{stamp}_aggregate.html"
    payload = {
        "generated_at": utc_timestamp(),
        "started_at": started_at,
        "selected_targets": selected_targets,
        "endpoint_count": len(rows),
        "failed_endpoint_count": failed_endpoints,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": pass_rate,
        "total_diffs": total_diffs,
        "total_duration_seconds": total_duration,
        "endpoints": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_aggregate_html(payload), encoding="utf-8")
    return str(html_path)


def _aggregate_html(payload: dict[str, Any]) -> str:
    rows_html = "\n".join(_aggregate_row_html(row) for row in payload["endpoints"])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>接口汇总报告</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f6f7f9; color: #17202a; font: 14px/1.5 "Segoe UI", "Microsoft YaHei", Arial, sans-serif; }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{ padding: 14px; background: #fff; border: 1px solid #d9dee7; border-radius: 6px; }}
    .card span {{ display: block; color: #6a7380; font-size: 12px; }}
    .card strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dee7; }}
    th, td {{ padding: 10px 12px; border-top: 1px solid #d9dee7; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; color: #475467; font-size: 12px; }}
    .status {{ display: inline-flex; min-height: 22px; align-items: center; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
    .passed {{ background: #e7f6ee; color: #16794c; }}
    .failed {{ background: #fde8e7; color: #b42318; }}
    .no_report {{ background: #fff6d7; color: #9a6700; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>接口汇总报告</h1>
  <div class="grid">
    <div class="card"><span>接口数</span><strong>{payload["endpoint_count"]}</strong></div>
    <div class="card"><span>失败接口</span><strong>{payload["failed_endpoint_count"]}</strong></div>
    <div class="card"><span>用例通过率</span><strong>{payload["pass_rate"] * 100:.1f}%</strong></div>
    <div class="card"><span>失败用例</span><strong>{payload["failed_cases"]}</strong></div>
    <div class="card"><span>差异数</span><strong>{payload["total_diffs"]}</strong></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>接口</th>
        <th>批次</th>
        <th>状态</th>
        <th>通过率</th>
        <th>通过/失败</th>
        <th>差异</th>
        <th>耗时</th>
        <th>报告</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>
"""


def _aggregate_row_html(row: dict[str, Any]) -> str:
    failed_cases = int(row.get("failed_cases") or 0)
    status = str(row.get("status") or ("failed" if failed_cases else "passed"))
    report_links = []
    if row.get("report_html_path"):
        report_links.append(f'<a href="{_file_href(str(row["report_html_path"]))}" target="_blank">HTML</a>')
    return f"""<tr>
  <td>{escape(str(row.get("endpoint") or ""))}</td>
  <td>{escape(str(row.get("run_id") or ""))}</td>
  <td><span class="status {status}">{status}</span></td>
  <td>{float(row.get("pass_rate") or 0) * 100:.1f}%</td>
  <td>{int(row.get("passed_cases") or 0)} / {failed_cases}</td>
  <td>{int(row.get("total_diffs") or 0)}</td>
  <td>{float(row.get("total_duration_seconds") or 0):.2f}s</td>
  <td>{" ".join(report_links)}</td>
</tr>"""


def _missing_report_row(endpoint: str) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "run_id": "-",
        "status": "no_report",
        "total_cases": 0,
        "passed_cases": 0,
        "failed_cases": 0,
        "pass_rate": 0,
        "total_diffs": 0,
        "total_duration_seconds": 0,
        "report_html_path": "",
        "report_md_path": "",
    }


def _file_href(path: str) -> str:
    try:
        rel = Path(path).resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return "#"
    return f"/files/{rel}"
