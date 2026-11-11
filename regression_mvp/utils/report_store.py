from __future__ import annotations

import html
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from regression_mvp.utils.metrics import utc_timestamp


DB_FILENAME = "regression_history.sqlite"


def store_report_run(
    report_dir: Path,
    entries: list[Any],
    run_id: str,
    target: str,
    summary: dict[str, Any],
    generated_at: str | None = None,
) -> None:
    db_path = _db_path(report_dir)
    endpoint = _endpoint_name(report_dir)
    root_report_dir = db_path.parent
    run_dir = report_dir / "runs" / run_id
    generated_at_value = generated_at or utc_timestamp()

    with sqlite3.connect(db_path) as conn:
        _init_schema(conn)
        conn.execute(
            """
            insert or replace into runs (
                run_id, endpoint, target, generated_at, total_cases, passed_cases,
                failed_cases, pass_rate, total_duration_seconds, total_diffs,
                report_md_path, report_json_path, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                endpoint,
                target,
                generated_at_value,
                int(summary.get("total_cases", 0)),
                int(summary.get("passed_cases", 0)),
                int(summary.get("failed_cases", 0)),
                float(summary.get("pass_rate", 0)),
                float(summary.get("total_duration_seconds", 0)),
                int(summary.get("total_diffs", 0)),
                str(run_dir / "report.md"),
                str(run_dir / "report.json"),
                utc_timestamp(),
            ),
        )

        conn.execute("delete from case_runs where run_id = ? and endpoint = ?", (run_id, endpoint))
        conn.execute("delete from case_metrics where run_id = ? and endpoint = ?", (run_id, endpoint))

        for entry in entries:
            case_dir = run_dir / "cases" / _safe_path_name(entry.case_name)
            conn.execute(
                """
                insert into case_runs (
                    run_id, endpoint, case_name, status, duration_seconds,
                    upload_elapsed_seconds, http_status_code, failure_category,
                    diff_total, input_sha256, expected_path, actual_path,
                    metrics_json_path, diff_detail_json_path, extra_artifacts_json_path
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    endpoint,
                    entry.case_name,
                    entry.status,
                    float(entry.duration_seconds),
                    float(entry.upload_elapsed_seconds),
                    entry.http_status_code,
                    entry.failure_category,
                    int(entry.diff_metrics.get("total", 0)),
                    str(entry.file_info.get("sha256", "")),
                    entry.expected_items_path,
                    entry.actual_items_path,
                    str(case_dir / "metrics.json"),
                    str(case_dir / "diff_detail.json"),
                    str(case_dir / "extra_artifacts.json"),
                ),
            )
            _insert_metrics(conn, run_id, endpoint, entry.case_name, "result", entry.result_metrics)
            _insert_metrics(conn, run_id, endpoint, entry.case_name, "diff", entry.diff_metrics)
            _insert_metrics(conn, run_id, endpoint, entry.case_name, "upload", entry.upload_summary)

        conn.commit()

    write_dashboard(root_report_dir)


def write_dashboard(report_root: Path) -> Path:
    db_path = report_root / DB_FILENAME
    dashboard_path = report_root / "dashboard.md"
    dashboard_html_path = report_root / "dashboard.html"
    generated_at = utc_timestamp()
    if not db_path.exists():
        dashboard_path.write_text("# Regression Dashboard\n\nNo SQLite history found.\n", encoding="utf-8")
        dashboard_html_path.write_text(_empty_dashboard_html(), encoding="utf-8")
        return dashboard_html_path

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        latest_rows = conn.execute(
            """
            with ranked as (
                select *, row_number() over (
                    partition by endpoint order by generated_at desc, run_id desc
                ) as rn
                from runs
            )
            select * from ranked where rn = 1 order by endpoint
            """
        ).fetchall()
        recent_runs = conn.execute(
            """
            select * from runs
            order by generated_at desc, run_id desc
            limit 20
            """
        ).fetchall()
        slow_cases = conn.execute(
            """
            select endpoint, case_name, status, duration_seconds, run_id
            from case_runs
            order by duration_seconds desc
            limit 10
            """
        ).fetchall()
        failing_cases = conn.execute(
            """
            select endpoint, case_name, count(*) as fail_count, max(run_id) as last_run_id
            from case_runs
            where status in ('failed', 'suspected')
            group by endpoint, case_name
            order by fail_count desc, last_run_id desc
            limit 10
            """
        ).fetchall()

    lines = [
        "# Regression Dashboard",
        "",
        f"- generated_at: `{generated_at}`",
        f"- sqlite: `{db_path}`",
        "",
        "## Latest Endpoint Status",
        "",
        "| endpoint | run_id | pass_rate | passed | failed | duration_s | diffs | report |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in latest_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['endpoint']}`",
                    f"`{row['run_id']}`",
                    f"{float(row['pass_rate']):.2%}",
                    str(row["passed_cases"]),
                    str(row["failed_cases"]),
                    str(row["total_duration_seconds"]),
                    str(row["total_diffs"]),
                    f"[report]({_rel(row['report_md_path'], report_root)})",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Recent Runs",
            "",
            "| generated_at | endpoint | run_id | pass_rate | duration_s | diffs |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in recent_runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['generated_at']}`",
                    f"`{row['endpoint']}`",
                    f"`{row['run_id']}`",
                    f"{float(row['pass_rate']):.2%}",
                    str(row["total_duration_seconds"]),
                    str(row["total_diffs"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Slowest Case Runs",
            "",
            "| endpoint | case | status | duration_s | run_id |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in slow_cases:
        lines.append(
            f"| `{row['endpoint']}` | `{row['case_name']}` | {row['status']} | {row['duration_seconds']:.3f} | `{row['run_id']}` |"
        )

    lines.extend(
        [
            "",
            "## Frequent Failing Cases",
            "",
            "| endpoint | case | fail_count | last_run_id |",
            "| --- | --- | ---: | --- |",
        ]
    )
    if failing_cases:
        for row in failing_cases:
            lines.append(
                f"| `{row['endpoint']}` | `{row['case_name']}` | {row['fail_count']} | `{row['last_run_id']}` |"
            )
    else:
        lines.append("|  |  | 0 |  |")

    dashboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dashboard_html_path.write_text(
        _render_dashboard_html(
            latest_rows,
            recent_runs,
            slow_cases,
            failing_cases,
            db_path,
            report_root,
            generated_at,
        ),
        encoding="utf-8",
    )
    return dashboard_html_path


def rebuild_store(report_root: Path) -> Path:
    db_path = report_root / DB_FILENAME
    with sqlite3.connect(db_path) as conn:
        _init_schema(conn)
        conn.execute("delete from case_metrics")
        conn.execute("delete from case_runs")
        conn.execute("delete from runs")
        conn.commit()

    report_paths = list(report_root.glob("runs/*/report.json"))
    report_paths.extend(report_root.glob("*/runs/*/report.json"))
    for report_json in sorted(set(report_paths)):
        payload = json.loads(report_json.read_text(encoding="utf-8"))
        run = payload.get("run") or {}
        summary = payload.get("summary") or {}
        cases = payload.get("cases") or []
        report_dir = report_json.parents[2]
        entries = [_StoredEntry(case) for case in cases if isinstance(case, dict)]
        store_report_run(
            report_dir,
            entries,
            str(run.get("run_id") or report_json.parent.name),
            str(run.get("target") or report_dir.name),
            summary,
            generated_at=str(run.get("generated_at") or ""),
        )

    return write_dashboard(report_root)


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists runs (
            run_id text not null,
            endpoint text not null,
            target text not null,
            generated_at text not null,
            total_cases integer not null,
            passed_cases integer not null,
            failed_cases integer not null,
            pass_rate real not null,
            total_duration_seconds real not null,
            total_diffs integer not null,
            report_md_path text not null,
            report_json_path text not null,
            created_at text not null,
            primary key (run_id, endpoint)
        );

        create table if not exists case_runs (
            run_id text not null,
            endpoint text not null,
            case_name text not null,
            status text not null,
            duration_seconds real not null,
            upload_elapsed_seconds real not null,
            http_status_code integer,
            failure_category text,
            diff_total integer not null,
            input_sha256 text,
            expected_path text,
            actual_path text,
            metrics_json_path text,
            diff_detail_json_path text,
            extra_artifacts_json_path text,
            primary key (run_id, endpoint, case_name)
        );

        create table if not exists case_metrics (
            run_id text not null,
            endpoint text not null,
            case_name text not null,
            metric_group text not null,
            metric_key text not null,
            metric_value text,
            primary key (run_id, endpoint, case_name, metric_group, metric_key)
        );
        """
    )


def _insert_metrics(
    conn: sqlite3.Connection,
    run_id: str,
    endpoint: str,
    case_name: str,
    metric_group: str,
    value: Any,
    prefix: str = "",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            _insert_metrics(conn, run_id, endpoint, case_name, metric_group, child, child_key)
        return
    if isinstance(value, list):
        metric_value = json.dumps(value, ensure_ascii=False)
    else:
        metric_value = "" if value is None else str(value)
    conn.execute(
        """
        insert or replace into case_metrics (
            run_id, endpoint, case_name, metric_group, metric_key, metric_value
        ) values (?, ?, ?, ?, ?, ?)
        """,
        (run_id, endpoint, case_name, metric_group, prefix, metric_value),
    )


def _db_path(report_dir: Path) -> Path:
    report_dir = report_dir.resolve()
    if report_dir.name == "reports":
        return report_dir / DB_FILENAME
    return report_dir.parent / DB_FILENAME


def _endpoint_name(report_dir: Path) -> str:
    return report_dir.resolve().name


def _rel(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path


def _render_dashboard_html(
    latest_rows: list[sqlite3.Row],
    recent_runs: list[sqlite3.Row],
    slow_cases: list[sqlite3.Row],
    failing_cases: list[sqlite3.Row],
    db_path: Path,
    report_root: Path,
    generated_at: str,
) -> str:
    latest_html = "".join(
        "<tr>"
        f"<td><code>{_esc(row['endpoint'])}</code></td>"
        f"<td><code>{_esc(row['run_id'])}</code></td>"
        f"<td>{float(row['pass_rate']):.1%}</td>"
        f"<td>{_esc(row['passed_cases'])}</td>"
        f"<td>{_esc(row['failed_cases'])}</td>"
        f"<td>{_esc(row['total_duration_seconds'])}</td>"
        f"<td>{_esc(row['total_diffs'])}</td>"
        f"<td>{_report_link(row, report_root)}</td>"
        "</tr>"
        for row in latest_rows
    )
    recent_html = "".join(
        "<tr>"
        f"<td><code>{_esc(row['generated_at'])}</code></td>"
        f"<td><code>{_esc(row['endpoint'])}</code></td>"
        f"<td><code>{_esc(row['run_id'])}</code></td>"
        f"<td>{float(row['pass_rate']):.1%}</td>"
        f"<td>{_esc(row['total_duration_seconds'])}</td>"
        f"<td>{_esc(row['total_diffs'])}</td>"
        "</tr>"
        for row in recent_runs
    )
    slow_html = "".join(
        "<tr>"
        f"<td><code>{_esc(row['endpoint'])}</code></td>"
        f"<td>{_esc(row['case_name'])}</td>"
        f"<td>{_status_badge(row['status'])}</td>"
        f"<td>{float(row['duration_seconds']):.3f}</td>"
        f"<td><code>{_esc(row['run_id'])}</code></td>"
        "</tr>"
        for row in slow_cases
    )
    if failing_cases:
        failing_html = "".join(
            "<tr>"
            f"<td><code>{_esc(row['endpoint'])}</code></td>"
            f"<td>{_esc(row['case_name'])}</td>"
            f"<td>{_esc(row['fail_count'])}</td>"
            f"<td><code>{_esc(row['last_run_id'])}</code></td>"
            "</tr>"
            for row in failing_cases
        )
    else:
        failing_html = '<tr><td colspan="4" class="muted">暂无失败用例</td></tr>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Regression Dashboard</title>
  <style>
    :root {{ --bg:#f6f7f9; --panel:#fff; --text:#17202a; --muted:#667085; --border:#d9dee7; --accent:#2563eb; --pass:#16794c; --fail:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.55 "Segoe UI","Microsoft YaHei",Arial,sans-serif; }}
    a {{ color:var(--accent); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    header {{ padding:22px 28px; background:#fff; border-bottom:1px solid var(--border); }}
    h1 {{ margin:0 0 8px; font-size:22px; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    main {{ max-width:1280px; margin:0 auto; padding:22px 28px 42px; }}
    .muted {{ color:var(--muted); }}
    .section {{ margin-bottom:18px; padding:18px; background:var(--panel); border:1px solid var(--border); border-radius:6px; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; min-width:760px; border-collapse:collapse; }}
    th,td {{ padding:9px 11px; border-top:1px solid var(--border); text-align:left; vertical-align:top; }}
    th {{ background:#f8fafc; color:#475467; font-size:12px; }}
    code {{ font-family:Consolas,"Liberation Mono",monospace; }}
    .badge {{ display:inline-flex; min-height:22px; align-items:center; padding:2px 8px; border-radius:999px; background:#eef2f6; color:#344054; font-size:12px; font-weight:600; }}
    .passed,.pass,.success {{ background:#e7f6ee; color:var(--pass); }}
    .failed,.fail,.error {{ background:#fde8e7; color:var(--fail); }}
    @media (max-width:900px) {{ main,header {{ padding-left:16px; padding-right:16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Regression Dashboard</h1>
    <div class="muted">generated_at: <code>{_esc(generated_at)}</code> · sqlite: <code>{_esc(db_path)}</code></div>
  </header>
  <main>
    <section class="section">
      <h2>Latest Endpoint Status</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>endpoint</th><th>run_id</th><th>pass_rate</th><th>passed</th><th>failed</th><th>duration_s</th><th>diffs</th><th>report</th></tr></thead>
        <tbody>{latest_html}</tbody>
      </table></div>
    </section>
    <section class="section">
      <h2>Recent Runs</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>generated_at</th><th>endpoint</th><th>run_id</th><th>pass_rate</th><th>duration_s</th><th>diffs</th></tr></thead>
        <tbody>{recent_html}</tbody>
      </table></div>
    </section>
    <section class="section">
      <h2>Slowest Case Runs</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>endpoint</th><th>case</th><th>status</th><th>duration_s</th><th>run_id</th></tr></thead>
        <tbody>{slow_html}</tbody>
      </table></div>
    </section>
    <section class="section">
      <h2>Frequent Failing Cases</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>endpoint</th><th>case</th><th>fail_count</th><th>last_run_id</th></tr></thead>
        <tbody>{failing_html}</tbody>
      </table></div>
    </section>
  </main>
</body>
</html>
"""


def _empty_dashboard_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Regression Dashboard</title></head>
<body><h1>Regression Dashboard</h1><p>No SQLite history found.</p></body>
</html>
"""


def _report_link(row: sqlite3.Row, report_root: Path) -> str:
    md_path = Path(str(row["report_md_path"]))
    html_path = md_path.with_suffix(".html")
    if html_path.is_file():
        return f'<a href="{_esc(_rel(str(html_path), report_root))}">HTML</a>'
    return f'<a href="{_esc(_rel(row["report_md_path"], report_root))}">Markdown</a>'


def _status_badge(status: str) -> str:
    return f'<span class="badge {_esc(status)}">{_esc(status)}</span>'


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _safe_path_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe.strip("._") or "case"


class _StoredEntry:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __getattr__(self, name: str) -> Any:
        if name in self._payload:
            return self._payload[name]
        raise AttributeError(name)

    @property
    def diff_metrics(self) -> dict[str, Any]:
        return self._payload.get("diff_metrics") or {}

    @property
    def result_metrics(self) -> dict[str, Any]:
        return self._payload.get("result_metrics") or {}

    @property
    def file_info(self) -> dict[str, Any]:
        return self._payload.get("file_info") or {}

    @property
    def upload_summary(self) -> dict[str, Any]:
        return self._payload.get("upload_summary") or {}
