from __future__ import annotations

import json
import shutil
from pathlib import Path

from regression_mvp.utils.metrics import run_id_timestamp
from regression_mvp.utils.report_store import rebuild_store
from regression_mvp.utils.reporter import write_html_reports_from_existing
from regression_mvp.web_app import test_runner


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT_DIR / "reports"
ARCHIVE_ROOT = REPORT_ROOT / "_archive"


def archive_endpoint_report(endpoint: str, run_id: str) -> Path:
    endpoint_dir = _safe_child(REPORT_ROOT, endpoint)
    run_dir = _safe_child(endpoint_dir / "runs", run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError("report run not found")

    archive_dir = _unique_archive_path(ARCHIVE_ROOT / "runs" / endpoint, run_id)
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(archive_dir))

    _refresh_endpoint_latest(endpoint_dir)
    rebuild_store(REPORT_ROOT)
    write_html_reports_from_existing(REPORT_ROOT)
    return archive_dir


def archive_aggregate_report(summary_report_path: str) -> Path:
    summary_path = Path(summary_report_path).resolve()
    log_dir = test_runner.LOG_DIR.resolve()
    if not _is_under(summary_path, log_dir) or not summary_path.is_file():
        raise FileNotFoundError("aggregate report not found")

    related_paths = _aggregate_related_paths(summary_path)
    archive_dir = _unique_archive_path(ARCHIVE_ROOT / "web_runs", summary_path.stem)
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved_summary = archive_dir / summary_path.name
    for source in related_paths:
        if source.is_file():
            shutil.move(str(source), str(archive_dir / source.name))

    _remove_aggregate_history(summary_path)
    return moved_summary


def _refresh_endpoint_latest(endpoint_dir: Path) -> None:
    latest_files = [
        endpoint_dir / "latest_report.json",
        endpoint_dir / "latest_report.md",
        endpoint_dir / "latest_report.html",
    ]
    remaining_reports = sorted(
        (endpoint_dir / "runs").glob("*/report.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    if not remaining_reports:
        for path in latest_files:
            if path.exists():
                path.unlink()
        return

    latest_json = remaining_reports[0]
    latest_run_id = latest_json.parent.name
    latest_md = latest_json.with_suffix(".md")
    shutil.copy2(latest_json, endpoint_dir / "latest_report.json")
    if latest_md.is_file():
        shutil.copy2(latest_md, endpoint_dir / "latest_report.md")
    (endpoint_dir / "latest_report.html").write_text(_latest_html_redirect(latest_run_id), encoding="utf-8")


def _aggregate_related_paths(summary_path: Path) -> list[Path]:
    paths = [summary_path]
    if summary_path.suffix == ".html":
        json_path = summary_path.with_suffix(".json")
        if json_path.is_file():
            paths.append(json_path)

    row = _aggregate_history_row(summary_path)
    if row:
        log_path = Path(str(row.get("log_path") or "")).resolve()
        if log_path.is_file() and _is_under(log_path, test_runner.LOG_DIR.resolve()):
            paths.append(log_path)
    return list(dict.fromkeys(paths))


def _aggregate_history_row(summary_path: Path) -> dict | None:
    history_path = test_runner.HISTORY_PATH
    if not history_path.is_file():
        return None
    for line in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if Path(str(row.get("summary_report_path") or "")).resolve() == summary_path:
            return row
    return None


def _remove_aggregate_history(summary_path: Path) -> None:
    history_path = test_runner.HISTORY_PATH
    if not history_path.is_file():
        return
    kept_lines: list[str] = []
    for line in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(line)
            continue
        if Path(str(row.get("summary_report_path") or "")).resolve() == summary_path:
            continue
        kept_lines.append(json.dumps(row, ensure_ascii=False))
    history_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")


def _safe_child(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not _is_under(target, root.resolve()):
        raise ValueError("invalid path")
    return target


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _unique_archive_path(root: Path, name: str) -> Path:
    safe_name = _safe_path_name(name)
    stamp = run_id_timestamp()
    candidate = root / f"{safe_name}_{stamp}"
    index = 1
    while candidate.exists():
        candidate = root / f"{safe_name}_{stamp}_{index}"
        index += 1
    return candidate


def _safe_path_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe.strip("._") or "report"


def _latest_html_redirect(run_id: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=runs/{run_id}/report.html">
  <title>Latest Regression Report</title>
</head>
<body>
  <a href="runs/{run_id}/report.html">Open latest report</a>
</body>
</html>
"""
