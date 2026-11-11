from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from regression_mvp.utils.case_manager import generate_candidate
from regression_mvp.utils.comparator import compare_items
from regression_mvp.utils.json_io import load_json
from regression_mvp.utils.json_io import dump_json
from regression_mvp.utils.metrics import SEVERITY_RANK, diff_metrics, diff_severity, utc_timestamp
from regression_mvp.utils.normalizer import normalize_items
from regression_mvp.storage import minio_sync


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"
CASES_DIR = ROOT_DIR / "cases"
CASE_HISTORY_DIR = ROOT_DIR / "case_history"


@dataclass
class CaseUpdateState:
    status: str = "idle"
    endpoint: str = ""
    case_name: str = ""
    input_file: str = ""
    candidate_dir: str = ""
    backup_dir: str = ""
    message: str = ""
    started_at: str = ""
    finished_at: str = ""
    errors: list[str] = field(default_factory=list)


_lock = threading.Lock()
_state = CaseUpdateState()
_thread: threading.Thread | None = None


def current_status(endpoint: str = "", case_name: str = "") -> dict[str, Any]:
    with _lock:
        payload = _state.__dict__.copy()
    if endpoint and case_name and (payload["endpoint"] != endpoint or payload["case_name"] != case_name):
        payload = CaseUpdateState().__dict__.copy()
    if endpoint and case_name:
        try:
            payload["case_input_file"] = str(case_input_file(endpoint, case_name))
        except Exception:
            payload["case_input_file"] = ""
    payload["update_diff"] = _update_diff_payload(endpoint, case_name, Path(payload["candidate_dir"]), payload["status"])
    payload["candidate_artifacts"] = _candidate_artifacts(Path(payload["candidate_dir"]))
    return payload


def start_update(endpoint: str, case_name: str, input_file: str | None = None) -> CaseUpdateState:
    global _state, _thread
    case_dir = _case_dir(endpoint, case_name)
    input_path = Path(input_file).resolve() if input_file else case_input_file(endpoint, case_name)
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    with _lock:
        if _state.status == "running":
            return _state
        _state = CaseUpdateState(
            status="running",
            endpoint=endpoint,
            case_name=case_name,
            input_file=str(input_path),
            started_at=utc_timestamp(),
            message="答案更新候选生成中",
        )
        _thread = threading.Thread(
            target=_update_worker,
            args=(endpoint, case_name, str(input_path), _case_description(case_dir)),
            daemon=True,
        )
        _thread.start()
        return _state


def apply_update(endpoint: str, case_name: str) -> Path:
    case_dir = _case_dir(endpoint, case_name)
    with _lock:
        state = _state.__dict__.copy()
    if state["endpoint"] != endpoint or state["case_name"] != case_name or state["status"] != "success":
        raise ValueError("no successful update candidate for this case")

    candidate_dir = Path(state["candidate_dir"])
    candidate_yaml = candidate_dir / "candidate.yaml"
    if not candidate_yaml.is_file():
        raise FileNotFoundError(f"candidate.yaml not found: {candidate_yaml}")

    case_yaml = case_dir / "case.yaml"
    case_raw = _read_simple_yaml(case_yaml)
    candidate_raw = _read_simple_yaml(candidate_yaml)
    source_input = candidate_dir / (candidate_raw.get("input_file") or "input.pdf")
    source_expected = candidate_dir / (candidate_raw.get("expected_file") or "expected_items.candidate.json")
    if not source_input.is_file():
        raise FileNotFoundError(f"candidate input file not found: {source_input}")
    if not source_expected.is_file():
        raise FileNotFoundError(f"candidate expected file not found: {source_expected}")

    backup_dir = _next_backup_dir(endpoint, case_name)
    shutil.copytree(case_dir, backup_dir)

    input_name = case_raw.get("input_file") or source_input.name
    expected_name = case_raw.get("expected_file") or _default_expected_name(endpoint)
    shutil.copy2(source_input, case_dir / input_name)
    shutil.copy2(source_expected, case_dir / expected_name)

    case_raw.update(
        {
            "case_name": case_name,
            "endpoint_type": endpoint,
            "input_file": input_name,
            "expected_file": expected_name,
            "input_sha256": candidate_raw.get("input_sha256", ""),
            "baseline_updated_at": utc_timestamp(),
            "baseline_updated_from_candidate": str(candidate_dir),
            "baseline_backup_dir": str(backup_dir),
            "review_status": "approved",
        }
    )
    _write_simple_yaml(case_yaml, case_raw)
    _try_upload_case(case_dir)

    with _lock:
        _state.status = "applied"
        _state.backup_dir = str(backup_dir)
        _state.finished_at = utc_timestamp()
        _state.message = "答案更新已应用"
    return backup_dir


def _update_worker(endpoint: str, case_name: str, input_file: str, description: str) -> None:
    try:
        config = load_json(CONFIG_PATH)
        candidate_dir = generate_candidate(
            ROOT_DIR,
            config,
            endpoint,
            Path(input_file),
            case_name,
            description,
        )
        _try_upload_candidate(candidate_dir)
        with _lock:
            _state.status = "success"
            _state.candidate_dir = str(candidate_dir)
            _state.finished_at = utc_timestamp()
            _state.message = "答案更新候选生成完成，请确认后覆盖原答案"
    except Exception as exc:
        with _lock:
            _state.status = "failed"
            _state.finished_at = utc_timestamp()
            _state.message = "答案更新候选生成失败"
            _state.errors = [str(exc)]


def _case_dir(endpoint: str, case_name: str) -> Path:
    safe_case_name = Path(case_name.strip()).name
    case_dir = (CASES_DIR / endpoint.strip() / safe_case_name).resolve()
    try:
        case_dir.relative_to(CASES_DIR.resolve())
    except ValueError:
        raise ValueError(f"invalid case name: {case_name}")
    if not (case_dir / "case.yaml").is_file():
        raise FileNotFoundError(f"case.yaml not found: {case_dir}")
    return case_dir


def _case_description(case_dir: Path) -> str:
    raw = _read_simple_yaml(case_dir / "case.yaml")
    return raw.get("description", "")


def case_input_file(endpoint: str, case_name: str) -> Path:
    case_dir = _case_dir(endpoint, case_name)
    raw = _read_simple_yaml(case_dir / "case.yaml")
    input_name = raw.get("input_file") or "input.pdf"
    input_path = (case_dir / input_name).resolve()
    try:
        input_path.relative_to(case_dir.resolve())
    except ValueError:
        raise ValueError(f"invalid input file path in case.yaml: {input_name}")
    return input_path


def _next_backup_dir(endpoint: str, case_name: str) -> Path:
    CASE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    base = CASE_HISTORY_DIR / f"{_stamp()}_{endpoint}_{case_name}"
    target = base
    index = 1
    while target.exists():
        index += 1
        target = CASE_HISTORY_DIR / f"{base.name}_{index}"
    return target


def _candidate_artifacts(candidate_dir: Path) -> list[dict[str, str]]:
    if not candidate_dir.is_dir():
        return []
    patterns = [
        ("candidate.yaml", "candidate.yaml"),
        ("input", "input.*"),
        ("expected", "expected_*.candidate.json"),
        ("preview", "preview.html"),
        ("overlay", "overlay.html"),
        ("subject_match_source_table", "subject_match_source_table.html"),
        ("subject_match_table", "subject_match_table.html"),
        ("subject_match_request", "subject_match_request.json"),
        ("preview_md", "preview.md"),
        ("metrics", "metrics.json"),
        ("actual_response", "actual_response.json"),
        ("update_diff_detail", "update_diff_detail.json"),
    ]
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, pattern in patterns:
        for path in sorted(candidate_dir.glob(pattern)):
            resolved = str(path.resolve())
            if path.is_file() and resolved not in seen:
                artifacts.append({"label": label, "path": resolved})
                seen.add(resolved)
    return artifacts


def _update_diff_payload(endpoint: str, case_name: str, candidate_dir: Path, status: str) -> dict[str, Any]:
    if status != "success" or not endpoint or not case_name or not candidate_dir.is_dir():
        return {}
    try:
        case_expected = load_json(case_expected_file(endpoint, case_name))
        candidate_expected = load_json(_candidate_expected_file(candidate_dir))
        compare_config = load_json(CONFIG_PATH).get("compare", {})
        diff = compare_items(
            normalize_items(case_expected, compare_config),
            normalize_items(candidate_expected, compare_config),
        )
        stats = diff_metrics(diff.diffs)
        detail_path = candidate_dir / "update_diff_detail.json"
        dump_json(
            detail_path,
            {
                "summary": diff.summary,
                "passed": diff.passed,
                "metrics": stats,
                "diffs": diff.diffs,
            },
        )
        return {
            "passed": diff.passed,
            "summary": diff.summary,
            "metrics": stats,
            "detail_path": str(detail_path),
            "samples": _diff_samples(diff.diffs),
        }
    except Exception as exc:
        return {
            "passed": False,
            "summary": f"更新前后 diff 生成失败: {exc}",
            "metrics": {},
            "detail_path": "",
            "samples": [],
            "error": str(exc),
        }


def case_expected_file(endpoint: str, case_name: str) -> Path:
    case_dir = _case_dir(endpoint, case_name)
    raw = _read_simple_yaml(case_dir / "case.yaml")
    expected_name = raw.get("expected_file") or _default_expected_name(endpoint)
    expected_path = (case_dir / expected_name).resolve()
    try:
        expected_path.relative_to(case_dir.resolve())
    except ValueError:
        raise ValueError(f"invalid expected file path in case.yaml: {expected_name}")
    if not expected_path.is_file():
        raise FileNotFoundError(f"expected file not found: {expected_path}")
    return expected_path


def _candidate_expected_file(candidate_dir: Path) -> Path:
    raw = _read_simple_yaml(candidate_dir / "candidate.yaml")
    expected_name = raw.get("expected_file") or "expected_items.candidate.json"
    expected_path = (candidate_dir / expected_name).resolve()
    try:
        expected_path.relative_to(candidate_dir.resolve())
    except ValueError:
        raise ValueError(f"invalid candidate expected file path: {expected_name}")
    if not expected_path.is_file():
        raise FileNotFoundError(f"candidate expected file not found: {expected_path}")
    return expected_path


def _diff_samples(diffs: list[dict[str, Any]], limit: int = 30) -> list[dict[str, str]]:
    sorted_diffs = sorted(
        diffs,
        key=lambda item: (SEVERITY_RANK.get(diff_severity(item), SEVERITY_RANK["major"]), str(item.get("path") or "")),
    )
    return [
        {
            "severity": diff_severity(diff),
            "type": str(diff.get("type") or ""),
            "path": str(diff.get("path") or ""),
            "expected": _short_value(diff.get("expected")),
            "actual": _short_value(diff.get("actual")),
        }
        for diff in sorted_diffs[:limit]
    ]


def _short_value(value: Any, limit: int = 180) -> str:
    text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _default_expected_name(endpoint: str) -> str:
    if endpoint in {
        "financial_metadata",
        "financial_three_statements",
        "financial_org_tables",
        "financial_merge_tables",
        "financial_subject_match",
    }:
        return "expected_response.json"
    return "expected_items.json"


def _read_simple_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"').strip("'").replace('\\"', '"').replace("\\\\", "\\")
    return result


def _write_simple_yaml(path: Path, values: dict[str, Any]) -> None:
    lines = []
    for key, value in values.items():
        text = "" if value is None else str(value)
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}: "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stamp() -> str:
    return utc_timestamp().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")


def _try_upload_case(case_dir: Path) -> None:
    try:
        minio_sync.upload_case_dir(case_dir)
    except Exception as exc:
        from loguru import logger

        minio_sync.record_sync_event("auto_upload_case", target=str(case_dir), error=exc)
        logger.warning("Regression MinIO case sync failed | case_dir={} err={}", case_dir, exc)


def _try_upload_candidate(candidate_dir: Path) -> None:
    try:
        minio_sync.upload_candidate_dir(candidate_dir)
    except Exception as exc:
        from loguru import logger

        minio_sync.record_sync_event("auto_upload_candidate", target=str(candidate_dir), error=exc)
        logger.warning("Regression MinIO candidate sync failed | candidate_dir={} err={}", candidate_dir, exc)
