from __future__ import annotations

import json
import sqlite3
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT_DIR / "reports"
DB_PATH = REPORT_ROOT / "regression_history.sqlite"


ENDPOINT_DISPLAY = {
    "mineru_upload_async": {
        "group": "MinerU解析",
        "order": 10,
        "method": "POST",
        "path": "/api/parsing/mineru/upload",
        "description": "上传文件并异步触发 MinerU 解析，立即返回可查询 MinIO 解析结果的 file_id",
    },
    "mineru_to_json": {
        "group": "MinerU解析",
        "order": 20,
        "method": "POST",
        "path": "/api/parsing/mineru/to-json",
        "description": "上传文件并同步调用 MinerU，返回 file_id 与解析结果对象信息（推荐）",
    },
    "ade_jobs_async": {
        "group": "ADE 解析",
        "order": 25,
        "method": "POST",
        "path": "/openapi/ade/jobs/submit",
        "description": "提交文件至 ADE 异步解析任务，轮询任务状态并读取最终解析结果",
    },
    "financial_metadata": {
        "group": "财报元数据提取",
        "order": 30,
        "method": "GET",
        "path": "/api/parsing/extract/financial-metadata/{file_id}",
        "description": "基于 file_id 提取财报元数据和财务三大表",
    },
    "financial_three_statements": {
        "group": "财报元数据提取",
        "order": 40,
        "method": "GET",
        "path": "/api/parsing/extract/three-statements/{file_id}",
        "description": "基于 file_id 提取财务三大表",
    },
    "financial_org_tables": {
        "group": "财报元数据提取",
        "order": 50,
        "method": "GET",
        "path": "/api/parsing/extract/org-tables/{file_id}",
        "description": "获取所有原始表格块（未合并）",
    },
    "financial_merge_tables": {
        "group": "财报元数据提取",
        "order": 60,
        "method": "POST",
        "path": "/api/parsing/extract/merge-tables",
        "description": "手动合并表格",
    },
    "financial_subject_match": {
        "group": "财报科目匹配",
        "order": 65,
        "method": "POST",
        "path": "/api/parsing/subject-match/match",
        "description": "基于 reportList 和 tables 为财报表格单元格匹配 subjectId",
    },
    "file_read": {
        "group": "解析给前端展示",
        "order": 70,
        "method": "POST",
        "path": "/api/file/read",
        "description": "上传文件并异步调用 MinerU，生成并保存目标 items（推荐）",
    },
}


def latest_endpoint_status() -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _connect() as conn:
        rows = conn.execute(
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
    return [_with_report_html(_row(row)) for row in rows]


def recent_runs(limit: int = 100) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            select * from runs
            order by generated_at desc, run_id desc
            limit ?
            """,
            (limit,),
        ).fetchall()
    return [_with_report_html(_row(row)) for row in rows]


def get_run(endpoint: str, run_id: str) -> dict[str, Any] | None:
    if not DB_PATH.exists():
        return None
    with _connect() as conn:
        row = conn.execute(
            "select * from runs where endpoint = ? and run_id = ?",
            (endpoint, run_id),
        ).fetchone()
    return _with_report_html(_row(row)) if row else None


def case_runs_for_run(endpoint: str, run_id: str) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            select * from case_runs
            where endpoint = ? and run_id = ?
            order by case_name
            """,
            (endpoint, run_id),
        ).fetchall()
    return [_enrich_case_run(_row(row)) for row in rows]


def all_cases(*, include_preview_artifacts: bool = True) -> list[dict[str, Any]]:
    cases_root = ROOT_DIR / "cases"
    results: list[dict[str, Any]] = []
    if not cases_root.exists():
        return results

    for case_yaml in sorted(cases_root.glob("**/case.yaml")):
        raw = _read_simple_yaml(case_yaml)
        endpoint = raw.get("endpoint_type") or _endpoint_from_path(cases_root, case_yaml)
        case_name = raw.get("case_name") or case_yaml.parent.name
        lifecycle_status = str(raw.get("lifecycle_status") or "active").strip().lower()
        latest = latest_case_run(endpoint, case_name)
        candidate_path = raw.get("created_from_candidate", "")
        results.append(
            {
                "endpoint": endpoint,
                "case_name": case_name,
                "enabled": _truthy(raw.get("enabled", "true")),
                "status": "deleted" if lifecycle_status == "deleted" else ("enabled" if _truthy(raw.get("enabled", "true")) else "disabled"),
                "lifecycle_status": lifecycle_status,
                "deleted": lifecycle_status == "deleted",
                "description": raw.get("description", ""),
                "input_file": _resolve(case_yaml.parent, raw.get("input_file", "")),
                "expected_file": _resolve(case_yaml.parent, raw.get("expected_file", "")),
                "input_sha256": raw.get("input_sha256", ""),
                "review_status": raw.get("review_status", ""),
                "created_from_candidate": candidate_path,
                "case_yaml": str(case_yaml),
                "latest_run": latest,
                "preview_artifacts": _preview_artifacts(candidate_path) if include_preview_artifacts else [],
            }
        )
    return results


def all_candidates() -> list[dict[str, Any]]:
    candidates_root = ROOT_DIR / "case_candidates"
    results: list[dict[str, Any]] = []
    if not candidates_root.exists():
        return results

    promoted_sources = {
        str(Path(case["created_from_candidate"]).resolve())
        for case in all_cases(include_preview_artifacts=False)
        if case.get("created_from_candidate")
    }
    for candidate_yaml in sorted(candidates_root.glob("*/candidate.yaml"), reverse=True):
        candidate_dir = candidate_yaml.parent
        raw = _read_simple_yaml(candidate_yaml)
        artifacts: list[dict[str, str]] = []
        seen: set[str] = set()
        _add_known_artifacts(artifacts, seen, candidate_dir, "candidate")
        results.append(
            {
                "name": candidate_dir.name,
                "path": str(candidate_dir),
                "endpoint": raw.get("endpoint_type", ""),
                "case_name": raw.get("case_name", candidate_dir.name),
                "description": raw.get("description", ""),
                "review_status": raw.get("review_status", ""),
                "input_file": _resolve(candidate_dir, raw.get("input_file", "")),
                "expected_file": _resolve(candidate_dir, raw.get("expected_file", "")),
                "candidate_yaml": str(candidate_yaml),
                "promoted": str(candidate_dir.resolve()) in promoted_sources,
                "preview_artifacts": artifacts,
            }
        )
    return results


def endpoint_case_summary(cases: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {
        endpoint: {
            "endpoint": endpoint,
            "case_count": 0,
            "enabled_count": 0,
            "latest": None,
        }
        for endpoint in ENDPOINT_DISPLAY
    }
    for case in cases if cases is not None else all_cases(include_preview_artifacts=False):
        endpoint = case["endpoint"]
        item = summary.setdefault(
            endpoint,
            {
                "endpoint": endpoint,
                "case_count": 0,
                "enabled_count": 0,
                "latest": None,
            },
        )
        item["case_count"] += 1
        if case["enabled"]:
            item["enabled_count"] += 1

    latest_by_endpoint = {row["endpoint"]: row for row in latest_endpoint_status()}
    for endpoint, item in summary.items():
        item["latest"] = latest_by_endpoint.get(endpoint)

    for row in latest_endpoint_status():
        summary.setdefault(
            row["endpoint"],
            {
                "endpoint": row["endpoint"],
                "case_count": 0,
                "enabled_count": 0,
                "latest": row,
            },
        )
    results = []
    for item in summary.values():
        display = ENDPOINT_DISPLAY.get(
            item["endpoint"],
            {
                "group": "其他接口",
                "order": 999,
                "method": "",
                "path": item["endpoint"],
                "description": "",
            },
        )
        results.append({**item, **display})
    return sorted(results, key=lambda item: (item["order"], item["endpoint"]))


def latest_case_run(endpoint: str, case_name: str) -> dict[str, Any] | None:
    if not DB_PATH.exists():
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            select * from case_runs
            where endpoint = ? and case_name = ?
            order by run_id desc
            limit 1
            """,
            (endpoint, case_name),
        ).fetchone()
    return _row(row) if row else None


def case_run_history(endpoint: str, case_name: str) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            select
                cr.*,
                r.generated_at,
                r.report_md_path,
                r.report_json_path,
                r.total_duration_seconds as run_duration_seconds,
                r.total_cases,
                r.passed_cases,
                r.pass_rate
            from case_runs cr
            left join runs r on r.endpoint = cr.endpoint and r.run_id = cr.run_id
            where cr.endpoint = ? and cr.case_name = ?
            order by coalesce(r.generated_at, cr.run_id) desc, cr.run_id desc
            """,
            (endpoint, case_name),
        ).fetchall()
    return [_enrich_case_history_run(_row(row)) for row in rows]


def load_report_json(endpoint: str, run_id: str) -> dict[str, Any]:
    path = REPORT_ROOT / endpoint / "runs" / run_id / "report.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dashboard_path() -> Path:
    html_path = REPORT_ROOT / "dashboard.html"
    if html_path.is_file():
        return html_path
    return REPORT_ROOT / "dashboard.md"


def report_root() -> Path:
    return REPORT_ROOT


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _with_report_html(row: dict[str, Any]) -> dict[str, Any]:
    md_path = Path(str(row.get("report_md_path") or ""))
    html_path = md_path.with_suffix(".html")
    row["report_html_path"] = str(html_path) if html_path.is_file() else ""
    return row


def _enrich_case_run(case_run: dict[str, Any]) -> dict[str, Any]:
    extra_path = Path(str(case_run.get("extra_artifacts_json_path") or ""))
    if extra_path.is_file():
        try:
            case_run["extra_artifacts"] = json.loads(extra_path.read_text(encoding="utf-8"))
        except Exception:
            case_run["extra_artifacts"] = {}
    else:
        case_run["extra_artifacts"] = {}
    return case_run


def _enrich_case_history_run(case_run: dict[str, Any]) -> dict[str, Any]:
    case_run = _with_report_html(case_run)
    case_dir = REPORT_ROOT / str(case_run["endpoint"]) / "runs" / str(case_run["run_id"]) / "cases" / _safe_path_name(str(case_run["case_name"]))
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    if case_dir.is_dir():
        _add_known_artifacts(artifacts, seen, case_dir, "run")
        _append_artifact(artifacts, seen, "metrics", case_dir / "metrics.json")
        _append_artifact(artifacts, seen, "diff_detail", case_dir / "diff_detail.json")
        _append_artifact(artifacts, seen, "actual_items", case_dir / "actual_items.json")

    extra_path = Path(str(case_run.get("extra_artifacts_json_path") or ""))
    if extra_path.is_file():
        try:
            extra_artifacts = json.loads(extra_path.read_text(encoding="utf-8"))
        except Exception:
            extra_artifacts = {}
        for name, path in extra_artifacts.items():
            _append_artifact(artifacts, seen, str(name), Path(str(path)))

    case_run["artifacts"] = artifacts
    return case_run


def _preview_artifacts(candidate_path: str) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()

    candidate_dir = Path(candidate_path) if candidate_path else None
    if candidate_dir and candidate_dir.is_dir():
        _add_known_artifacts(artifacts, seen, candidate_dir, "candidate")

    return artifacts


def _add_known_artifacts(artifacts: list[dict[str, str]], seen: set[str], root: Path, source: str) -> None:
    patterns = [
        ("actual_preview", "actual_preview.html"),
        ("actual_overlay", "actual_overlay.html"),
        ("expected_preview", "expected_preview.html"),
        ("expected_overlay", "expected_overlay.html"),
        ("diff_preview", "diff_preview.html"),
        ("diff_overlay", "diff_overlay.html"),
        ("table_diff_preview", "table_diff_preview.html"),
        ("subject_match_source_table", "subject_match_source_table.html"),
        ("subject_match_table", "subject_match_table.html"),
        ("subject_match_request", "subject_match_request.json"),
        ("preview", "preview.html"),
        ("overlay", "overlay.html"),
        ("grid", "*.statement_grid_preview.html"),
        ("merge", "*.merge_tables_preview.html"),
    ]
    if source != "candidate":
        patterns.append(("summary", "*.statement_diff_summary.md"))
    if source != "candidate":
        patterns.append(("preview_md", "preview.md"))

    files = sorted((path for path in root.iterdir() if path.is_file()), key=lambda path: path.name)
    for label, pattern in patterns:
        for path in files:
            if not fnmatch(path.name, pattern):
                continue
            _append_artifact(artifacts, seen, f"{source}_{label}", path)

    pages_dir = root / "pages"
    if source != "candidate" and pages_dir.is_dir():
        for index, path in enumerate(sorted(pages_dir.glob("page_*.png"))[:3], start=1):
            _append_artifact(artifacts, seen, f"{source}_page_{index}", path)


def _append_artifact(artifacts: list[dict[str, str]], seen: set[str], label: str, path: Path) -> None:
    if label in {"latest_expected_preview", "latest_expected_overlay"}:
        return
    if label.startswith("latest_page_"):
        return
    if not path.is_file():
        return
    resolved = str(path.absolute())
    if resolved in seen:
        return
    seen.add(resolved)
    artifacts.append({"label": label, "path": resolved})


def _read_simple_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _endpoint_from_path(cases_root: Path, case_yaml: Path) -> str:
    try:
        return case_yaml.parent.relative_to(cases_root).parts[0]
    except Exception:
        return "unknown"


def _resolve(base: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _safe_path_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe.strip("._") or "case"
