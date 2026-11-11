from __future__ import annotations

import threading
import shutil
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from regression_mvp.utils.case_manager import generate_candidate, promote_candidate
from regression_mvp.utils.diagnostics import diagnose_failure
from regression_mvp.utils.json_io import load_json
from regression_mvp.utils.metrics import utc_timestamp
from regression_mvp.storage import minio_sync


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"
UPLOAD_DIR = ROOT_DIR / "uploads"
CASES_DIR = ROOT_DIR / "cases"
CASE_ARCHIVE_DIR = ROOT_DIR / "case_archive"

ENDPOINTS = {
    "file_read",
    "mineru_to_json",
    "mineru_upload_async",
    "ade_jobs_async",
    "financial_metadata",
    "financial_three_statements",
    "financial_org_tables",
    "financial_merge_tables",
    "financial_subject_match",
}


@dataclass
class CandidateJobState:
    status: str = "idle"
    endpoint: str = ""
    input_file: str = ""
    case_name: str = ""
    description: str = ""
    started_at: str = ""
    finished_at: str = ""
    candidate_dir: str = ""
    message: str = ""
    total: int = 0
    completed: int = 0
    failed: int = 0
    candidate_dirs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


_lock = threading.Lock()
_state = CandidateJobState()
_thread: threading.Thread | None = None


def current_status() -> dict[str, Any]:
    with _lock:
        payload = _state.__dict__.copy()
    diagnostic = diagnose_failure(error_message="\n".join(payload.get("errors") or []))
    payload["diagnostic"] = diagnostic.to_dict() if diagnostic else None
    return payload


def start_generate(endpoint: str, input_file: str, case_name: str, description: str) -> CandidateJobState:
    return start_generate_batch(endpoint, [(input_file, case_name, description)])


def start_generate_batch(endpoint: str, jobs: list[tuple[str, str, str]]) -> CandidateJobState:
    global _state, _thread
    endpoint = endpoint.strip()
    normalized_jobs = [
        (input_file.strip().strip('"'), case_name.strip(), description.strip())
        for input_file, case_name, description in jobs
    ]
    if not normalized_jobs:
        raise ValueError("input file is required")
    for input_file, case_name, _description in normalized_jobs:
        _validate_generate(endpoint, input_file, case_name)

    with _lock:
        if _state.status == "running":
            return _state
        _state = CandidateJobState(
            status="running",
            endpoint=endpoint,
            input_file=normalized_jobs[0][0],
            case_name=normalized_jobs[0][1],
            description=normalized_jobs[0][2],
            started_at=utc_timestamp(),
            message="候选用例生成中",
            total=len(normalized_jobs),
        )
        _thread = threading.Thread(
            target=_generate_batch_worker,
            args=(endpoint, normalized_jobs),
            daemon=True,
        )
        _thread.start()
        return _state


def promote(candidate_name: str, *, force: bool = False) -> Path:
    candidate_dir = _candidate_dir(candidate_name)
    case_dir = promote_candidate(ROOT_DIR, candidate_dir, force=force)
    _try_upload_case(case_dir)
    _try_upload_candidate(candidate_dir)
    return case_dir


def discard_candidate(candidate_name: str) -> Path:
    candidate_dir = _candidate_dir(candidate_name)
    _try_delete_candidate(candidate_dir.name)
    shutil.rmtree(candidate_dir)
    return candidate_dir


def set_case_enabled(endpoint: str, case_name: str, enabled: bool) -> Path:
    case_dir = _case_dir(endpoint, case_name)
    case_yaml = case_dir / "case.yaml"
    lines = case_yaml.read_text(encoding="utf-8").splitlines()
    enabled_line = f"enabled: {'true' if enabled else 'false'}"
    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.strip().startswith("enabled:"):
            updated.append(enabled_line)
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(enabled_line)
    if enabled:
        updated = [
            'lifecycle_status: "active"' if line.strip().startswith("lifecycle_status:") else line
            for line in updated
            if not line.strip().startswith("deleted_at:")
        ]
        if not any(line.strip().startswith("lifecycle_status:") for line in updated):
            updated.append('lifecycle_status: "active"')
    case_yaml.write_text("\n".join(updated) + "\n", encoding="utf-8")
    _try_upload_case(case_dir)
    return case_yaml


def archive_case(endpoint: str, case_name: str) -> Path:
    case_dir = _case_dir(endpoint, case_name)
    _set_case_lifecycle(case_dir, "deleted", enabled=False)
    _try_upload_case(case_dir)
    return case_dir


def restore_case(endpoint: str, case_name: str) -> Path:
    case_dir = _case_dir(endpoint, case_name)
    _set_case_lifecycle(case_dir, "active", enabled=True)
    _try_upload_case(case_dir)
    return case_dir


def hard_delete_case(endpoint: str, case_name: str) -> Path:
    case_dir = _case_dir(endpoint, case_name)
    raw = _read_simple_yaml(case_dir / "case.yaml")
    lifecycle_status = str(raw.get("lifecycle_status") or "active").strip().lower()
    if lifecycle_status != "deleted":
        raise ValueError("only deleted cases can be permanently deleted")
    CASE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_name = f"{utc_timestamp().replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}_{endpoint}_{case_dir.name}_hard_deleted"
    target = (CASE_ARCHIVE_DIR / archive_name).resolve()
    try:
        target.relative_to(CASE_ARCHIVE_DIR.resolve())
    except ValueError:
        raise ValueError(f"invalid archive target: {target}")
    if target.exists():
        raise FileExistsError(f"archive target already exists: {target}")
    shutil.move(str(case_dir), str(target))
    _try_delete_case(endpoint, case_dir.name)
    return target


def case_metadata(endpoint: str, case_name: str) -> dict[str, str]:
    case_dir = _case_dir(endpoint, case_name)
    raw = _read_simple_yaml(case_dir / "case.yaml")
    return {
        "endpoint": endpoint,
        "case_name": raw.get("case_name") or case_dir.name,
        "description": raw.get("description", ""),
        "case_dir": str(case_dir),
    }


def update_case_metadata(endpoint: str, old_case_name: str, new_case_name: str, description: str) -> Path:
    case_dir = _case_dir(endpoint, old_case_name)
    safe_name = _validate_case_name(new_case_name)
    target_dir = case_dir.with_name(safe_name)
    if target_dir != case_dir and target_dir.exists():
        raise FileExistsError(f"case already exists: {safe_name}")

    case_yaml = case_dir / "case.yaml"
    raw = _read_simple_yaml(case_yaml)
    raw["case_name"] = safe_name
    raw["description"] = description.strip()
    _write_simple_yaml(case_yaml, raw)

    if target_dir != case_dir:
        shutil.move(str(case_dir), str(target_dir))
        _try_delete_case(endpoint, case_dir.name)
    _try_upload_case(target_dir)
    return target_dir


def save_uploaded_file(filename: str, source_file: Any) -> Path:
    safe_name = _safe_upload_name(filename)
    if not safe_name:
        raise ValueError("uploaded file name is empty")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{utc_timestamp().replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}_{safe_name}"
    with target.open("wb") as output:
        shutil.copyfileobj(source_file, output)
    return target


def batch_case_name(base_name: str, filename: str, index: int, total: int) -> str:
    prefix = _safe_case_part(base_name) if base_name.strip() else "case"
    stem = _safe_case_part(Path(filename).stem)
    if total <= 1:
        return prefix if base_name.strip() else stem
    return f"{prefix}_{stem}_{index:03d}" if base_name.strip() else f"{stem}_{index:03d}"


def _generate_worker(endpoint: str, input_file: str, case_name: str, description: str) -> None:
    _generate_batch_worker(endpoint, [(input_file, case_name, description)])


def _generate_batch_worker(endpoint: str, jobs: list[tuple[str, str, str]]) -> None:
    candidate_dirs: list[str] = []
    errors: list[str] = []
    config = load_json(CONFIG_PATH)
    for index, (input_file, case_name, description) in enumerate(jobs, start=1):
        with _lock:
            _state.input_file = input_file
            _state.case_name = case_name
            _state.description = description
            _state.message = f"候选用例生成中 ({index}/{len(jobs)}): {case_name}"
        try:
            candidate_dir = generate_candidate(
                ROOT_DIR,
                config,
                endpoint,
                Path(input_file),
                case_name,
                description,
            )
            candidate_dirs.append(str(candidate_dir))
            _try_upload_candidate(candidate_dir)
            with _lock:
                _state.completed += 1
                _state.candidate_dir = str(candidate_dir)
                _state.candidate_dirs = list(candidate_dirs)
        except Exception as exc:
            errors.append(f"{case_name}: {exc}")
            with _lock:
                _state.failed += 1
                _state.errors = list(errors)

    with _lock:
        _state.finished_at = utc_timestamp()
        _state.candidate_dirs = list(candidate_dirs)
        _state.errors = list(errors)
        if errors and candidate_dirs:
            _state.status = "failed"
            _state.message = f"候选用例部分生成失败：成功 {len(candidate_dirs)} 个，失败 {len(errors)} 个"
        elif errors:
            _state.status = "failed"
            _state.message = "候选用例生成失败"
        else:
            _state.status = "success"
            _state.message = f"候选用例生成完成：{len(candidate_dirs)} 个"

def _validate_generate(endpoint: str, input_file: str, case_name: str) -> None:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    if not input_file:
        raise ValueError("input file is required")
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"input file not found: {input_file}")
    if not case_name:
        raise ValueError("case name is required")


def _candidate_dir(candidate_name: str) -> Path:
    safe_name = Path(candidate_name).name
    candidate_dir = (ROOT_DIR / "case_candidates" / safe_name).resolve()
    try:
        candidate_dir.relative_to((ROOT_DIR / "case_candidates").resolve())
    except ValueError:
        raise ValueError(f"invalid candidate name: {candidate_name}")
    if not (candidate_dir / "candidate.yaml").is_file():
        raise FileNotFoundError(f"candidate.yaml not found: {candidate_dir}")
    return candidate_dir


def _case_dir(endpoint: str, case_name: str) -> Path:
    endpoint = endpoint.strip()
    safe_case_name = Path(case_name.strip()).name
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    if not safe_case_name or safe_case_name in {".", ".."}:
        raise ValueError("case name is required")
    cases_root = CASES_DIR.resolve()
    case_dir = (CASES_DIR / endpoint / safe_case_name).resolve()
    try:
        case_dir.relative_to(cases_root)
    except ValueError:
        raise ValueError(f"invalid case name: {case_name}")
    if (case_dir / "case.yaml").is_file():
        return case_dir

    endpoint_dir = (CASES_DIR / endpoint).resolve()
    if endpoint_dir.is_dir():
        for candidate_yaml in endpoint_dir.glob("*/case.yaml"):
            raw = _read_simple_yaml(candidate_yaml)
            if raw.get("case_name") == case_name:
                return candidate_yaml.parent.resolve()
    raise FileNotFoundError(f"case.yaml not found: {case_dir}")


def _validate_case_name(case_name: str) -> str:
    value = case_name.strip()
    if not value:
        raise ValueError("case name is required")
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("case name cannot contain path separators")
    if not re.fullmatch(r"[\w.\-\u4e00-\u9fff]+", value):
        raise ValueError("case name only supports Chinese, letters, numbers, underscore, dash and dot")
    return value


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


def _set_case_lifecycle(case_dir: Path, lifecycle_status: str, *, enabled: bool) -> None:
    case_yaml = case_dir / "case.yaml"
    raw = _read_simple_yaml(case_yaml)
    raw["lifecycle_status"] = lifecycle_status
    raw["enabled"] = "true" if enabled else "false"
    if lifecycle_status == "deleted":
        raw["deleted_at"] = utc_timestamp()
    elif lifecycle_status == "active":
        raw.pop("deleted_at", None)
    _write_simple_yaml(case_yaml, raw)


def _safe_upload_name(filename: str) -> str:
    name = Path(filename).name.strip()
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe.strip("._")


def _safe_case_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe.strip("._") or "case"


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


def _try_delete_case(endpoint: str, case_dir_name: str) -> None:
    try:
        minio_sync.delete_case_prefix(endpoint, case_dir_name)
    except Exception as exc:
        from loguru import logger

        minio_sync.record_sync_event("auto_delete_case", target=f"{endpoint}/{case_dir_name}", error=exc)
        logger.warning("Regression MinIO case delete sync failed | endpoint={} case={} err={}", endpoint, case_dir_name, exc)


def _try_delete_candidate(candidate_name: str) -> None:
    try:
        minio_sync.delete_candidate_prefix(candidate_name)
    except Exception as exc:
        from loguru import logger

        minio_sync.record_sync_event("auto_delete_candidate", target=candidate_name, error=exc)
        logger.warning("Regression MinIO candidate delete sync failed | candidate={} err={}", candidate_name, exc)
