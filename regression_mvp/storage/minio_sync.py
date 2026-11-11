from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from loguru import logger
from minio import Minio
from minio.commonconfig import CopySource
from minio.deleteobjects import DeleteObject
from minio.error import S3Error

from regression_mvp.utils.metrics import utc_timestamp
from src.config import config


ROOT_DIR = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT_DIR / "cases"
CANDIDATES_DIR = ROOT_DIR / "case_candidates"
SYNC_STATE_DIR = ROOT_DIR / ".sync_state"
CONFLICT_DIR = SYNC_STATE_DIR / "conflicts"
SYNC_EVENT_LOG = SYNC_STATE_DIR / "sync_events.jsonl"
SYNC_STATUS_FILE = SYNC_STATE_DIR / "sync_status.json"
MAX_SYNC_EVENTS = 100

_CLIENT: Minio | None = None
_SCAN_LOCK = threading.Lock()


class SyncError(Exception):
    """Base class for regression MinIO sync errors."""


class PathSafetyError(SyncError):
    """A file path is unsafe for case storage."""


class RemoteObjectNotFoundError(SyncError):
    """Remote object or case does not exist."""


class VersionConflictError(SyncError):
    """Local and remote changes need manual review."""

    def __init__(self, message: str, report: ConflictReport | None = None) -> None:
        super().__init__(message)
        self.report = report
        self.report_path = _write_conflict_report(report) if report else None


@dataclass(frozen=True)
class SyncResult:
    uploaded: int = 0
    downloaded: int = 0
    deleted: int = 0
    skipped: int = 0
    conflicts: int = 0
    version: int | None = None
    conflict_report: str = ""
    object_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    sha256: str
    content_type: str
    object_name: str = ""


@dataclass(frozen=True)
class CaseManifest:
    object_type: str
    case_id: str
    endpoint: str
    case_name: str
    version: int
    files: list[FileRecord]
    created_at: str
    updated_at: str

    def file_map(self) -> dict[str, FileRecord]:
        return {record.path: record for record in self.files}


@dataclass(frozen=True)
class ConflictEntry:
    path: str
    base_sha256: str | None
    local_sha256: str | None
    remote_sha256: str | None
    local_status: str
    remote_status: str
    conflict_type: str
    choices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConflictReport:
    object_type: str
    case_id: str
    base_version: int | None
    remote_version: int | None
    generated_at: str
    entries: list[ConflictEntry]


def is_enabled() -> bool:
    value = os.getenv("REGRESSION_MVP_MINIO_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def bucket_name() -> str:
    return os.getenv("REGRESSION_MVP_MINIO_BUCKET", "regression-mvp-dev").strip() or "regression-mvp-dev"


def object_prefix() -> str:
    return os.getenv("REGRESSION_MVP_MINIO_PREFIX", "regression_mvp").strip().strip("/")


def auto_create_bucket() -> bool:
    value = os.getenv("REGRESSION_MVP_MINIO_AUTO_CREATE_BUCKET", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def cleanup_keep_versions() -> int:
    return _env_int("REGRESSION_MVP_MINIO_CLEANUP_KEEP_VERSIONS", 10)


def warn_history_versions() -> int:
    return _env_int("REGRESSION_MVP_MINIO_WARN_HISTORY_VERSIONS", 20)


def warn_history_bytes() -> int:
    return _env_int("REGRESSION_MVP_MINIO_WARN_HISTORY_MB", 1024) * 1024 * 1024


def sync_info() -> dict[str, str | bool]:
    return {
        "enabled": is_enabled(),
        "endpoint": config.minio.endpoint,
        "bucket": bucket_name(),
        "prefix": object_prefix(),
        "secure": config.minio.secure,
        "cleanup_keep_versions": cleanup_keep_versions(),
        "warn_history_versions": warn_history_versions(),
        "warn_history_mb": _env_int("REGRESSION_MVP_MINIO_WARN_HISTORY_MB", 1024),
    }


def management_overview() -> dict[str, Any]:
    local_cases = len(list(CASES_DIR.glob("*/*/case.yaml"))) if CASES_DIR.exists() else 0
    local_candidates = len(list(CANDIDATES_DIR.glob("*/candidate.yaml"))) if CANDIDATES_DIR.exists() else 0
    conflict_reports = list_conflict_reports()
    remote_cases = 0
    remote_candidates = 0
    remote_error = ""
    remote_error_diagnostic: dict[str, str] = {}
    if is_enabled():
        try:
            remote_cases = len(_list_remote_case_ids("case"))
            remote_candidates = len(_list_remote_case_ids("candidate"))
        except Exception as exc:
            remote_error = str(exc)
            remote_error_diagnostic = diagnose_sync_error(exc)
    return {
        "sync_info": sync_info(),
        "local_cases": local_cases,
        "local_candidates": local_candidates,
        "remote_cases": remote_cases,
        "remote_candidates": remote_candidates,
        "remote_error": remote_error,
        "remote_error_diagnostic": remote_error_diagnostic,
        "conflict_count": len(conflict_reports),
        "conflicts": conflict_reports,
        "recent_events": recent_sync_events(),
        "sync_status": load_sync_status(),
    }


def start_background_scan(reason: str = "startup") -> None:
    thread = threading.Thread(target=lambda: scan_sync_status(reason=reason), daemon=True)
    thread.start()


def load_sync_status() -> dict[str, Any]:
    if not SYNC_STATUS_FILE.is_file():
        return {
            "status": "unknown",
            "message": "尚未扫描同步状态",
            "scanned_at": "",
            "reason": "",
            "summary": {},
            "issues": [],
            "error": {},
        }
    try:
        return json.loads(SYNC_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "failed",
            "message": "同步状态文件读取失败",
            "scanned_at": "",
            "reason": "",
            "summary": {},
            "issues": [],
            "error": diagnose_sync_error(exc),
        }


def scan_sync_status(reason: str = "manual") -> dict[str, Any]:
    if not _SCAN_LOCK.acquire(blocking=False):
        return load_sync_status()
    try:
        status = _build_sync_status(reason)
        _write_sync_status(status)
        record_sync_event("scan_status", result={"skipped": 0}, target=reason)
        return status
    except Exception as exc:
        status = {
            "status": "failed",
            "message": diagnose_sync_error(exc)["title"],
            "scanned_at": utc_timestamp(),
            "reason": reason,
            "summary": {},
            "issues": [],
            "error": diagnose_sync_error(exc),
        }
        _write_sync_status(status)
        record_sync_event("scan_status", target=reason, error=exc)
        return status
    finally:
        _SCAN_LOCK.release()


def _build_sync_status(reason: str) -> dict[str, Any]:
    local_case_ids = _list_local_case_ids()
    local_candidate_ids = _list_local_candidate_ids()
    remote_case_ids: list[str] = []
    remote_candidate_ids: list[str] = []
    remote_error: dict[str, str] = {}
    history_summary = _empty_history_summary()
    if is_enabled():
        try:
            remote_case_ids = _list_remote_case_ids("case")
            remote_candidate_ids = _list_remote_case_ids("candidate")
        except Exception as exc:
            remote_error = diagnose_sync_error(exc)

    issues: list[dict[str, Any]] = []
    if not is_enabled():
        issues.append(
            {
                "level": "warning",
                "object_type": "system",
                "case_id": "",
                "issue_type": "minio_disabled",
                "message": "MinIO 同步未启用，只能查看本地状态。",
                "suggestion": "如果需要团队同步，请启用 REGRESSION_MVP_MINIO_ENABLED 并重启 Web。",
            }
        )
    elif remote_error:
        issues.append(
            {
                "level": "error",
                "object_type": "system",
                "case_id": "",
                "issue_type": remote_error.get("category", "remote_error"),
                "message": remote_error.get("title", "远端扫描失败"),
                "suggestion": remote_error.get("suggestion", ""),
            }
        )
    else:
        issues.extend(_scan_object_type_status("case", local_case_ids, remote_case_ids))
        issues.extend(_scan_object_type_status("candidate", local_candidate_ids, remote_candidate_ids))
        history_summary = _remote_history_summary(cleanup_keep_versions())
        history_issue = _history_cleanup_issue(history_summary)
        if history_issue:
            issues.append(history_issue)

    summary = {
        "local_cases": len(local_case_ids),
        "local_candidates": len(local_candidate_ids),
        "remote_cases": len(remote_case_ids),
        "remote_candidates": len(remote_candidate_ids),
        "local_unpublished": sum(1 for item in issues if item["issue_type"] == "local_unpublished"),
        "remote_unpulled": sum(1 for item in issues if item["issue_type"] == "remote_unpulled"),
        "conflict_risk": sum(1 for item in issues if item["issue_type"] == "conflict_risk"),
        "missing_baseline": sum(1 for item in issues if item["issue_type"] == "missing_baseline"),
        "history_delete_versions": history_summary["delete_version_count"],
        "history_delete_objects": history_summary["delete_object_count"],
        "history_delete_bytes": history_summary["delete_bytes"],
        "history_warn_versions": warn_history_versions(),
        "history_warn_bytes": warn_history_bytes(),
        "history_cleanup_keep_versions": cleanup_keep_versions(),
    }
    status_name, message, suggestion = _sync_status_message(issues, remote_error)
    return {
        "status": status_name,
        "message": message,
        "suggestion": suggestion,
        "scanned_at": utc_timestamp(),
        "reason": reason,
        "summary": summary,
        "issues": issues[:100],
        "error": remote_error,
    }


def _scan_object_type_status(object_type: str, local_ids: list[str], remote_ids: list[str]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for case_id in sorted(set(local_ids) | set(remote_ids)):
        local_dir = _local_dir_for_case_id(object_type, case_id)
        local_manifest = _local_manifest_for_dir(object_type, case_id, local_dir)
        remote_manifest = _load_remote_manifest(_client(), bucket_name(), object_type, case_id) if case_id in remote_ids else None
        base_manifest = _load_base_manifest(object_type, case_id)
        issues.extend(_object_sync_issues(object_type, case_id, local_manifest, remote_manifest, base_manifest))
    return issues


def _object_sync_issues(
    object_type: str,
    case_id: str,
    local_manifest: CaseManifest | None,
    remote_manifest: CaseManifest | None,
    base_manifest: CaseManifest | None,
) -> list[dict[str, Any]]:
    if local_manifest is None and remote_manifest is None:
        return []
    if local_manifest is not None and remote_manifest is None:
        return [_sync_issue("info", object_type, case_id, "local_unpublished", "本地存在但远端不存在", "确认后点击“上传本地全部”发布到 MinIO。")]
    if local_manifest is None and remote_manifest is not None:
        return [_sync_issue("info", object_type, case_id, "remote_unpulled", "远端存在但本地不存在", "点击“从 MinIO 刷新全部”拉取到本地。")]
    if local_manifest is None or remote_manifest is None:
        return []

    if _same_manifest_content(local_manifest, remote_manifest):
        if base_manifest is None:
            return [_sync_issue("warning", object_type, case_id, "missing_baseline", "本地和远端内容一致但缺少同步基线", "可先从 MinIO 刷新或上传本地，建立共同基线。")]
        return []
    if base_manifest is None:
        return [_sync_issue("warning", object_type, case_id, "missing_baseline", "本地和远端内容不一致且缺少共同基线", "先人工确认谁是正确版本，再在冲突详情中选择采用远端或强制采用本地。")]

    local_changed = not _same_manifest_content(local_manifest, base_manifest)
    remote_changed = not _same_manifest_content(remote_manifest, base_manifest)
    if local_changed and remote_changed:
        return [_sync_issue("error", object_type, case_id, "conflict_risk", "本地和远端都相对基线发生变化", "不要直接上传或刷新，先查看冲突报告或人工确认处理方向。")]
    if local_changed:
        return [_sync_issue("info", object_type, case_id, "local_unpublished", "本地相对基线有改动", "确认后点击“上传本地全部”发布到 MinIO。")]
    if remote_changed:
        return [_sync_issue("info", object_type, case_id, "remote_unpulled", "远端相对基线有改动", "点击“从 MinIO 刷新全部”拉取远端更新。")]
    return []


def _sync_issue(level: str, object_type: str, case_id: str, issue_type: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "level": level,
        "object_type": object_type,
        "case_id": case_id,
        "issue_type": issue_type,
        "message": message,
        "suggestion": suggestion,
    }


def _empty_history_summary() -> dict[str, int]:
    return {
        "delete_version_count": 0,
        "delete_object_count": 0,
        "delete_bytes": 0,
    }


def _remote_history_summary(keep_versions: int) -> dict[str, int]:
    plan = remote_version_cleanup_plan(keep_versions)
    return {
        "delete_version_count": int(plan["delete_version_count"]),
        "delete_object_count": int(plan["delete_object_count"]),
        "delete_bytes": int(plan["delete_bytes"]),
    }


def _history_cleanup_issue(history_summary: dict[str, int]) -> dict[str, Any] | None:
    version_count = int(history_summary.get("delete_version_count") or 0)
    byte_count = int(history_summary.get("delete_bytes") or 0)
    if version_count <= warn_history_versions() and byte_count <= warn_history_bytes():
        return None
    size_mb = byte_count / 1024 / 1024
    return _sync_issue(
        "warning",
        "system",
        "",
        "history_cleanup_recommended",
        f"MinIO 历史版本缓存偏大：可清理旧版本 {version_count} 个，约 {size_mb:.1f} MB。",
        f"建议在远端历史版本清理区保留最近 {cleanup_keep_versions()} 个版本，先预览再确认清理。",
    )


def _sync_status_message(issues: list[dict[str, Any]], remote_error: dict[str, str]) -> tuple[str, str, str]:
    if remote_error:
        return "failed", remote_error.get("title", "远端扫描失败"), remote_error.get("suggestion", "")
    if any(item["issue_type"] == "minio_disabled" for item in issues):
        return "warning", "MinIO 同步未启用", "当前只完成本地扫描；如需团队同步，请启用 MinIO 后重启 Web。"
    if any(item["issue_type"] == "conflict_risk" for item in issues):
        return "conflict", "发现本地和远端同时修改的对象", "先处理冲突，再执行上传或刷新。"
    if any(item["issue_type"] == "missing_baseline" for item in issues):
        return "warning", "存在缺少共同基线的对象", "人工确认后通过刷新或上传建立基线。"
    if any(item["issue_type"] == "history_cleanup_recommended" for item in issues):
        return "warning", "MinIO 历史版本缓存偏大", "先预览远端历史版本清理计划，确认后再执行清理。"
    if any(item["issue_type"] in {"local_unpublished", "remote_unpulled"} for item in issues):
        return "needs_sync", "发现本地或远端存在未同步对象", "根据问题列表选择上传本地或从 MinIO 刷新。"
    return "ok", "本地和远端看起来一致", "无需操作。"


def _write_sync_status(status: dict[str, Any]) -> None:
    SYNC_STATE_DIR.mkdir(parents=True, exist_ok=True)
    SYNC_STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _local_manifest_for_dir(object_type: str, case_id: str, local_dir: Path) -> CaseManifest | None:
    if not local_dir.is_dir():
        return None
    return _manifest_from_records(
        object_type=object_type,
        case_id=case_id,
        version=0,
        records=list(_scan_local_files(local_dir).values()),
        existing=None,
    )


def _list_local_case_ids() -> list[str]:
    if not CASES_DIR.exists():
        return []
    return sorted(f"{_safe_name(path.parent.parent.name)}/{_safe_name(path.parent.name)}" for path in CASES_DIR.glob("*/*/case.yaml"))


def _list_local_candidate_ids() -> list[str]:
    if not CANDIDATES_DIR.exists():
        return []
    return sorted(_safe_name(path.parent.name) for path in CANDIDATES_DIR.glob("*/candidate.yaml"))


def diagnose_sync_error(error: BaseException | str) -> dict[str, str]:
    message = str(error)
    lowered = message.lower()
    category = "unknown"
    title = "同步失败"
    suggestion = "查看完整错误信息，确认 MinIO 配置、网络和当前操作对象是否正常。"

    if isinstance(error, VersionConflictError) or "conflict" in lowered:
        category = "version_conflict"
        title = "本地和远端同时修改"
        suggestion = "打开冲突详情，选择采用远端、强制采用本地，或确认后标记已处理。"
    elif isinstance(error, PathSafetyError) or "unsafe" in lowered or "invalid" in lowered:
        category = "invalid_path"
        title = "文件路径不合法"
        suggestion = "检查用例名、候选名和文件相对路径，避免绝对路径、冒号、反斜杠或 ../。"
    elif isinstance(error, RemoteObjectNotFoundError) or "not found" in lowered or "nosuchkey" in lowered:
        category = "not_found"
        title = "远端对象不存在"
        suggestion = "先从 MinIO 刷新列表，确认该用例或候选是否已被其他人删除。"
    elif isinstance(error, S3Error):
        code = str(getattr(error, "code", "") or "").lower()
        if code in {"accessdenied", "invalidaccesskeyid", "signaturedoesnotmatch"}:
            category = "permission"
            title = "MinIO 权限或凭证错误"
            suggestion = "检查 .env 中的 MINIO 账号、密钥和 bucket 权限。"
        elif code in {"nosuchbucket"}:
            category = "bucket_missing"
            title = "Bucket 不存在"
            suggestion = "确认 bucket 名称是否正确，或让负责人初始化 bucket。"
        else:
            category = "minio_error"
            title = f"MinIO 返回错误 {getattr(error, 'code', '') or ''}".strip()
            suggestion = "根据 MinIO 错误码排查 bucket、权限、对象路径或服务状态。"
    elif any(token in lowered for token in ("timed out", "timeout", "connecttimeout")):
        category = "timeout"
        title = "连接 MinIO 超时"
        suggestion = "确认公司 VPN、MinIO 地址和网络代理状态；稍后可重试。"
    elif any(token in lowered for token in ("connection refused", "failed to establish", "max retries", "nodename", "name resolution", "no route")):
        category = "network"
        title = "无法连接 MinIO"
        suggestion = "确认 VPN 已连接、MinIO endpoint 可访问，并检查本机代理设置。"
    elif "sha256 mismatch" in lowered or "size mismatch" in lowered:
        category = "checksum"
        title = "文件校验失败"
        suggestion = "重新执行上传或刷新；如果重复出现，检查 MinIO 对象是否被部分写入。"
    elif "disabled" in lowered:
        category = "disabled"
        title = "MinIO 同步未启用"
        suggestion = "如果需要同步，请设置 REGRESSION_MVP_MINIO_ENABLED=true 并重启 Web。"

    return {
        "category": category,
        "title": title,
        "message": message,
        "suggestion": suggestion,
    }


def record_sync_event(
    operation: str,
    *,
    target: str = "",
    result: SyncResult | dict[str, Any] | None = None,
    error: BaseException | str | None = None,
) -> None:
    SYNC_STATE_DIR.mkdir(parents=True, exist_ok=True)
    result_data = _sync_result_dict(result)
    event = {
        "event_id": uuid.uuid4().hex,
        "created_at": utc_timestamp(),
        "operation": operation,
        "target": target,
        "status": "failed" if error else "success",
        "result": result_data,
        "error": diagnose_sync_error(error) if error else {},
    }
    events = recent_sync_events(limit=MAX_SYNC_EVENTS - 1)
    events.insert(0, event)
    SYNC_EVENT_LOG.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in events[:MAX_SYNC_EVENTS]) + "\n",
        encoding="utf-8",
    )


def recent_sync_events(limit: int = 20) -> list[dict[str, Any]]:
    if not SYNC_EVENT_LOG.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in SYNC_EVENT_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event.setdefault("event_id", _event_fallback_id(event))
        events.append(event)
        if len(events) >= limit:
            break
    return events


def sync_event_detail(event_id: str) -> dict[str, Any]:
    for event in recent_sync_events(limit=MAX_SYNC_EVENTS):
        if event.get("event_id") == event_id:
            return event
    raise FileNotFoundError(f"sync event not found: {event_id}")


def _sync_result_dict(result: SyncResult | dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, SyncResult):
        return asdict(result)
    return dict(result)


def _event_fallback_id(event: dict[str, Any]) -> str:
    raw = json.dumps(event, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default
    return value if value > 0 else default


def list_conflict_reports() -> list[dict[str, Any]]:
    if not CONFLICT_DIR.is_dir():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(CONFLICT_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            reports.append(
                {
                    "path": str(path.resolve()),
                    "name": path.name,
                    "object_type": "unknown",
                    "case_id": "",
                    "generated_at": "",
                    "base_version": "",
                    "remote_version": "",
                    "entry_count": 0,
                    "error": str(exc),
                }
            )
            continue
        entries = raw.get("entries") if isinstance(raw.get("entries"), list) else []
        reports.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "object_type": str(raw.get("object_type") or ""),
                "case_id": str(raw.get("case_id") or ""),
                "generated_at": str(raw.get("generated_at") or ""),
                "base_version": raw.get("base_version", ""),
                "remote_version": raw.get("remote_version", ""),
                "entry_count": len(entries),
                "error": "",
            }
        )
    return reports


def conflict_report_detail(report_name: str) -> dict[str, Any]:
    safe_name = Path(report_name).name
    if safe_name != report_name or not safe_name.endswith(".json"):
        raise PathSafetyError(f"invalid conflict report name: {report_name}")
    path = (CONFLICT_DIR / safe_name).resolve()
    try:
        path.relative_to(CONFLICT_DIR.resolve())
    except ValueError:
        raise PathSafetyError(f"invalid conflict report path: {report_name}")
    if not path.is_file():
        raise FileNotFoundError(f"conflict report not found: {report_name}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries") if isinstance(raw.get("entries"), list) else []
    return {
        "path": str(path),
        "name": path.name,
        "object_type": str(raw.get("object_type") or ""),
        "case_id": str(raw.get("case_id") or ""),
        "base_version": raw.get("base_version", ""),
        "remote_version": raw.get("remote_version", ""),
        "generated_at": str(raw.get("generated_at") or ""),
        "entry_count": len(entries),
        "entries": [_conflict_entry_view(entry) for entry in entries],
    }


def _conflict_entry_view(entry: dict[str, Any]) -> dict[str, Any]:
    local_status = str(entry.get("local_status") or "")
    remote_status = str(entry.get("remote_status") or "")
    conflict_type = str(entry.get("conflict_type") or "")
    choices = entry.get("choices") if isinstance(entry.get("choices"), list) else []
    return {
        "path": str(entry.get("path") or ""),
        "base_sha256": _short_sha(entry.get("base_sha256")),
        "local_sha256": _short_sha(entry.get("local_sha256")),
        "remote_sha256": _short_sha(entry.get("remote_sha256")),
        "local_status": local_status,
        "remote_status": remote_status,
        "conflict_type": conflict_type,
        "conflict_label": _conflict_label(conflict_type, local_status, remote_status),
        "severity": _conflict_severity(conflict_type, local_status, remote_status),
        "choices": [str(choice) for choice in choices],
    }


def _short_sha(value: Any) -> str:
    text = "" if value is None else str(value)
    return text[:12] if text else "-"


def _conflict_label(conflict_type: str, local_status: str, remote_status: str) -> str:
    labels = {
        "missing_base_manifest": "缺少共同基线",
        "delete_modify": "一边删除一边修改",
        "both_added_different": "本地和远端新增不同内容",
        "both_changed_different": "本地和远端同时修改",
        "local_added_remote_deleted": "本地新增但远端删除",
        "remote_added_local_deleted": "远端新增但本地删除",
    }
    if conflict_type in labels:
        return labels[conflict_type]
    if local_status == "deleted" and remote_status in {"modified", "added"}:
        return "本地删除但远端有修改"
    if remote_status == "deleted" and local_status in {"modified", "added"}:
        return "远端删除但本地有修改"
    if local_status in {"modified", "added"} and remote_status in {"modified", "added"}:
        return "本地和远端同时修改"
    return conflict_type or "冲突"


def _conflict_severity(conflict_type: str, local_status: str, remote_status: str) -> str:
    if conflict_type == "missing_base_manifest":
        return "high"
    if "deleted" in {local_status, remote_status}:
        return "high"
    if local_status in {"modified", "added"} and remote_status in {"modified", "added"}:
        return "medium"
    return "low"


def refresh_cases_and_candidates() -> SyncResult:
    if not is_enabled():
        return SyncResult(skipped=1)
    downloaded = 0
    skipped = 0
    conflicts = 0
    object_details: list[dict[str, Any]] = []
    for case_id in _list_remote_case_ids("case"):
        endpoint, case_name = case_id.split("/", 1)
        try:
            result = pull_case(endpoint, case_name)
            downloaded += result.downloaded
            object_details.extend(result.object_details)
        except VersionConflictError as exc:
            conflicts += len(exc.report.entries) if exc.report else 1
            object_details.append(_conflict_object_detail("case", case_id, exc.report))
            logger.warning("Regression MinIO case pull conflict | case_id={} report={}", case_id, exc.report_path)
    for candidate_name in _list_remote_case_ids("candidate"):
        try:
            result = pull_candidate(candidate_name)
            downloaded += result.downloaded
            object_details.extend(result.object_details)
        except VersionConflictError as exc:
            conflicts += len(exc.report.entries) if exc.report else 1
            object_details.append(_conflict_object_detail("candidate", candidate_name, exc.report))
            logger.warning("Regression MinIO candidate pull conflict | candidate={} report={}", candidate_name, exc.report_path)
    return SyncResult(downloaded=downloaded, skipped=skipped, conflicts=conflicts, object_details=object_details)


def upload_cases_and_candidates() -> SyncResult:
    if not is_enabled():
        return SyncResult(skipped=1)
    uploaded = 0
    skipped = 0
    conflicts = 0
    conflict_report = ""
    object_details: list[dict[str, Any]] = []
    for case_yaml in sorted(CASES_DIR.glob("*/*/case.yaml")):
        try:
            result = upload_case_dir(case_yaml.parent)
            uploaded += result.uploaded
            skipped += result.skipped
            object_details.extend(result.object_details)
        except VersionConflictError as exc:
            conflicts += len(exc.report.entries) if exc.report else 1
            conflict_report = str(exc.report_path or conflict_report)
            object_details.append(_conflict_object_detail("case", f"{case_yaml.parent.parent.name}/{case_yaml.parent.name}", exc.report))
            logger.warning("Regression MinIO case publish conflict | case_dir={} report={}", case_yaml.parent, exc.report_path)
    for candidate_yaml in sorted(CANDIDATES_DIR.glob("*/candidate.yaml")):
        try:
            result = upload_candidate_dir(candidate_yaml.parent)
            uploaded += result.uploaded
            skipped += result.skipped
            object_details.extend(result.object_details)
        except VersionConflictError as exc:
            conflicts += len(exc.report.entries) if exc.report else 1
            conflict_report = str(exc.report_path or conflict_report)
            object_details.append(_conflict_object_detail("candidate", candidate_yaml.parent.name, exc.report))
            logger.warning("Regression MinIO candidate publish conflict | candidate_dir={} report={}", candidate_yaml.parent, exc.report_path)
    return SyncResult(uploaded=uploaded, skipped=skipped, conflicts=conflicts, conflict_report=conflict_report, object_details=object_details)


def upload_case_dir(case_dir: Path) -> SyncResult:
    if not is_enabled():
        return SyncResult(skipped=1)
    case_dir = case_dir.resolve()
    endpoint = _safe_name(case_dir.parent.name)
    case_name = _safe_name(case_dir.name)
    return publish_workspace("case", f"{endpoint}/{case_name}", case_dir)


def upload_candidate_dir(candidate_dir: Path) -> SyncResult:
    if not is_enabled():
        return SyncResult(skipped=1)
    candidate_dir = candidate_dir.resolve()
    return publish_workspace("candidate", _safe_name(candidate_dir.name), candidate_dir)


def pull_case(endpoint: str, case_name: str) -> SyncResult:
    local_dir = (CASES_DIR / _safe_name(endpoint) / _safe_name(case_name)).resolve()
    return pull_workspace("case", f"{_safe_name(endpoint)}/{_safe_name(case_name)}", local_dir)


def pull_candidate(candidate_name: str) -> SyncResult:
    local_dir = (CANDIDATES_DIR / _safe_name(candidate_name)).resolve()
    return pull_workspace("candidate", _safe_name(candidate_name), local_dir)


def delete_case_prefix(endpoint: str, case_dir_name: str) -> SyncResult:
    if not is_enabled():
        return SyncResult(skipped=1)
    return delete_remote_case("case", f"{_safe_name(endpoint)}/{_safe_name(case_dir_name)}")


def delete_candidate_prefix(candidate_name: str) -> SyncResult:
    if not is_enabled():
        return SyncResult(skipped=1)
    return delete_remote_case("candidate", _safe_name(candidate_name))


def resolve_conflict_report(report_name: str, action: str) -> dict[str, Any]:
    detail = conflict_report_detail(report_name)
    object_type = str(detail["object_type"])
    case_id = str(detail["case_id"])
    if action == "use_remote":
        result = force_pull_workspace(object_type, case_id, _local_dir_for_case_id(object_type, case_id))
    elif action == "force_local":
        result = force_publish_workspace(object_type, case_id, _local_dir_for_case_id(object_type, case_id))
    elif action == "mark_handled":
        result = SyncResult(skipped=1)
    else:
        raise ValueError(f"unsupported conflict action: {action}")
    resolved_path = _archive_conflict_report(Path(str(detail["path"])), action)
    return {
        "action": action,
        "object_type": object_type,
        "case_id": case_id,
        "resolved_path": str(resolved_path),
        "result": asdict(result),
    }


def remote_version_cleanup_plan(keep_versions: int = 10) -> dict[str, Any]:
    keep_versions = max(1, int(keep_versions))
    if not is_enabled():
        raise SyncError("MinIO sync is disabled")
    client = _client()
    bucket = bucket_name()
    _ensure_bucket(client, bucket)
    objects: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    total_bytes = 0
    for object_type in ("case", "candidate"):
        for case_id in _list_remote_case_ids(object_type):
            target = _remote_cleanup_target(client, bucket, object_type, case_id, keep_versions)
            targets.append(target)
            objects.extend(target["delete_objects"])
            total_bytes += int(target["delete_bytes"])
    cleanup_targets = [target for target in targets if target["delete_versions"]]
    return {
        "keep_versions": keep_versions,
        "target_count": len(targets),
        "cleanup_target_count": len(cleanup_targets),
        "delete_version_count": sum(len(target["delete_versions"]) for target in targets),
        "delete_object_count": len(objects),
        "delete_bytes": total_bytes,
        "targets": targets,
        "cleanup_targets": cleanup_targets,
        "objects": objects,
    }


def cleanup_remote_versions(keep_versions: int = 10) -> SyncResult:
    plan = remote_version_cleanup_plan(keep_versions)
    object_names = [str(item["object_name"]) for item in plan["objects"]]
    if not object_names:
        return SyncResult(skipped=1)
    client = _client()
    bucket = bucket_name()
    errors = list(client.remove_objects(bucket, [DeleteObject(name) for name in object_names]))
    if errors:
        first = errors[0]
        raise SyncError(f"cleanup remote versions failed: {first}")
    result = SyncResult(deleted=len(object_names), skipped=max(0, int(plan["delete_version_count"]) - len(object_names)))
    record_sync_event(
        "cleanup_remote_versions",
        target=f"keep={plan['keep_versions']}",
        result=result,
    )
    logger.info(
        "Regression MinIO version cleanup finished | keep={} objects={} bytes={}",
        plan["keep_versions"],
        len(object_names),
        plan["delete_bytes"],
    )
    return result


def local_cleanup_plan(keep_days: int = 30) -> dict[str, Any]:
    keep_days = max(0, int(keep_days))
    cutoff = time.time() - keep_days * 86400
    roots = [
        ("case_history", ROOT_DIR / "case_history"),
        ("case_archive", ROOT_DIR / "case_archive"),
        ("report_archive", ROOT_DIR / "reports" / "_archive"),
        ("resolved_conflicts", CONFLICT_DIR / "resolved"),
    ]
    entries: list[dict[str, Any]] = []
    for label, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.exists():
                continue
            stat = path.stat()
            if stat.st_mtime >= cutoff:
                continue
            entries.append(
                {
                    "group": label,
                    "path": str(path.resolve()),
                    "name": path.name,
                    "is_dir": path.is_dir(),
                    "modified_at": utc_timestamp_from_epoch(stat.st_mtime),
                    "bytes": _path_size(path),
                }
            )
    return {
        "keep_days": keep_days,
        "delete_count": len(entries),
        "delete_bytes": sum(int(item["bytes"]) for item in entries),
        "entries": entries,
    }


def cleanup_local_archives(keep_days: int = 30) -> dict[str, Any]:
    plan = local_cleanup_plan(keep_days)
    deleted = 0
    failed: list[dict[str, str]] = []
    released_bytes = 0
    for entry in plan["entries"]:
        path = Path(str(entry["path"])).resolve()
        try:
            _validate_local_cleanup_target(path)
            if not path.exists():
                continue
            released_bytes += int(entry["bytes"])
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted += 1
        except Exception as exc:
            failed.append({"path": str(path), "error": str(exc)})
    result = {
        "deleted": deleted,
        "failed": len(failed),
        "failed_entries": failed,
        "released_bytes": released_bytes,
        "keep_days": int(plan["keep_days"]),
    }
    record_sync_event(
        "cleanup_local_archives",
        target=f"keep_days={plan['keep_days']}",
        result={"deleted": deleted, "skipped": len(failed), "released_bytes": released_bytes},
    )
    return result


def force_pull_workspace(object_type: str, case_id: str, local_dir: Path) -> SyncResult:
    start = time.perf_counter()
    if not is_enabled():
        raise SyncError("MinIO sync is disabled")
    _validate_case_id(object_type, case_id)
    client = _client()
    bucket = bucket_name()
    _ensure_bucket(client, bucket)
    remote_manifest = _load_remote_manifest(client, bucket, object_type, case_id)
    if remote_manifest is None:
        raise RemoteObjectNotFoundError(f"remote {object_type} not found: {case_id}")

    previous_manifest: CaseManifest | None = None
    if local_dir.exists():
        previous_manifest = _manifest_from_records(
            object_type=object_type,
            case_id=case_id,
            version=0,
            records=list(_scan_local_files(local_dir).values()),
            existing=remote_manifest,
        )
    tmp_parent = local_dir.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.force_pull_", dir=tmp_parent))
    downloaded = 0
    try:
        for record in remote_manifest.files:
            target = _safe_join(tmp_dir, record.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            client.fget_object(bucket, record.object_name, str(target))
            _verify_file(target, record)
            downloaded += 1
        backup_dir = None
        if local_dir.exists():
            backup_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.backup_", dir=tmp_parent))
            shutil.rmtree(backup_dir)
            local_dir.replace(backup_dir)
        tmp_dir.replace(local_dir)
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    _save_base_manifest(object_type, case_id, remote_manifest)
    logger.info(
        "Regression MinIO force pulled | type={} case_id={} version={} files={} cost={:.2f}s",
        object_type,
        case_id,
        remote_manifest.version,
        downloaded,
        time.perf_counter() - start,
    )
    return SyncResult(
        downloaded=downloaded,
        version=remote_manifest.version,
        object_details=[_object_detail("download", object_type, case_id, previous_manifest, remote_manifest, downloaded)],
    )


def force_publish_workspace(object_type: str, case_id: str, local_dir: Path) -> SyncResult:
    start = time.perf_counter()
    if not is_enabled():
        raise SyncError("MinIO sync is disabled")
    if not local_dir.is_dir():
        raise FileNotFoundError(f"directory not found: {local_dir}")
    _validate_case_id(object_type, case_id)
    client = _client()
    bucket = bucket_name()
    _ensure_bucket(client, bucket)

    local_files = _scan_local_files(local_dir)
    local_manifest = _manifest_from_records(
        object_type=object_type,
        case_id=case_id,
        version=0,
        records=list(local_files.values()),
        existing=None,
    )
    remote_manifest = _load_remote_manifest(client, bucket, object_type, case_id)
    previous_manifest = remote_manifest
    desired_records = (
        _reuse_unchanged_remote_objects(local_manifest, remote_manifest)
        if remote_manifest
        else list(local_manifest.files)
    )
    next_version = (remote_manifest.version if remote_manifest else 0) + 1
    staged_records, uploaded_count = _upload_new_records(
        client,
        bucket,
        object_type,
        case_id,
        next_version,
        local_dir,
        desired_records,
    )
    final_manifest = _manifest_from_records(
        object_type=object_type,
        case_id=case_id,
        version=next_version,
        records=staged_records,
        existing=remote_manifest,
    )
    _write_remote_manifest(client, bucket, object_type, case_id, final_manifest)
    _save_base_manifest(object_type, case_id, final_manifest)
    logger.info(
        "Regression MinIO force published | type={} case_id={} version={} files={} uploaded={} cost={:.2f}s",
        object_type,
        case_id,
        next_version,
        len(final_manifest.files),
        uploaded_count,
        time.perf_counter() - start,
    )
    return SyncResult(
        uploaded=uploaded_count,
        skipped=len(final_manifest.files) - uploaded_count,
        version=next_version,
        object_details=[_object_detail("upload", object_type, case_id, previous_manifest, final_manifest, uploaded_count)],
    )


def publish_workspace(object_type: str, case_id: str, local_dir: Path) -> SyncResult:
    start = time.perf_counter()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"directory not found: {local_dir}")
    _validate_case_id(object_type, case_id)

    client = _client()
    bucket = bucket_name()
    _ensure_bucket(client, bucket)

    local_files = _scan_local_files(local_dir)
    local_manifest = _manifest_from_records(
        object_type=object_type,
        case_id=case_id,
        version=0,
        records=list(local_files.values()),
        existing=None,
    )
    remote_manifest = _load_remote_manifest(client, bucket, object_type, case_id)
    previous_manifest = remote_manifest
    base_manifest = _load_base_manifest(object_type, case_id)

    desired_records = _desired_publish_records(object_type, case_id, local_manifest, remote_manifest, base_manifest)
    next_version = (remote_manifest.version if remote_manifest else 0) + 1
    staged_records, uploaded_count = _upload_new_records(
        client,
        bucket,
        object_type,
        case_id,
        next_version,
        local_dir,
        desired_records,
    )
    final_manifest = _manifest_from_records(
        object_type=object_type,
        case_id=case_id,
        version=next_version,
        records=staged_records,
        existing=remote_manifest,
    )
    _write_remote_manifest(client, bucket, object_type, case_id, final_manifest)
    _save_base_manifest(object_type, case_id, final_manifest)
    logger.info(
        "Regression MinIO published | type={} case_id={} version={} files={} uploaded={} cost={:.2f}s",
        object_type,
        case_id,
        next_version,
        len(final_manifest.files),
        uploaded_count,
        time.perf_counter() - start,
    )
    return SyncResult(
        uploaded=uploaded_count,
        skipped=len(final_manifest.files) - uploaded_count,
        version=next_version,
        object_details=[_object_detail("upload", object_type, case_id, previous_manifest, final_manifest, uploaded_count)],
    )


def pull_workspace(object_type: str, case_id: str, local_dir: Path) -> SyncResult:
    start = time.perf_counter()
    _validate_case_id(object_type, case_id)
    client = _client()
    bucket = bucket_name()
    _ensure_bucket(client, bucket)
    remote_manifest = _load_remote_manifest(client, bucket, object_type, case_id)
    if remote_manifest is None:
        raise RemoteObjectNotFoundError(f"remote {object_type} not found: {case_id}")

    previous_manifest: CaseManifest | None = None
    base_manifest = _load_base_manifest(object_type, case_id)
    if local_dir.exists():
        local_manifest = _manifest_from_records(
            object_type=object_type,
            case_id=case_id,
            version=0,
            records=list(_scan_local_files(local_dir).values()),
            existing=base_manifest or remote_manifest,
        )
        previous_manifest = local_manifest
        report = _pull_conflicts(object_type, case_id, base_manifest, local_manifest, remote_manifest)
        if report.entries:
            raise VersionConflictError(f"pull conflict: {case_id}", report)

    tmp_parent = local_dir.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.pull_", dir=tmp_parent))
    downloaded = 0
    try:
        for record in remote_manifest.files:
            target = _safe_join(tmp_dir, record.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            client.fget_object(bucket, record.object_name, str(target))
            _verify_file(target, record)
            downloaded += 1
        backup_dir = None
        if local_dir.exists():
            backup_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.backup_", dir=tmp_parent))
            shutil.rmtree(backup_dir)
            local_dir.replace(backup_dir)
        tmp_dir.replace(local_dir)
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    _save_base_manifest(object_type, case_id, remote_manifest)
    logger.info(
        "Regression MinIO pulled | type={} case_id={} version={} files={} cost={:.2f}s",
        object_type,
        case_id,
        remote_manifest.version,
        downloaded,
        time.perf_counter() - start,
    )
    return SyncResult(
        downloaded=downloaded,
        version=remote_manifest.version,
        object_details=[_object_detail("download", object_type, case_id, previous_manifest, remote_manifest, downloaded)],
    )


def delete_remote_case(object_type: str, case_id: str) -> SyncResult:
    _validate_case_id(object_type, case_id)
    client = _client()
    bucket = bucket_name()
    _ensure_bucket(client, bucket)
    root = _remote_root(object_type, case_id) + "/"
    objects = [DeleteObject(obj.object_name) for obj in client.list_objects(bucket, prefix=root, recursive=True) if obj.object_name]
    if not objects:
        return SyncResult()
    errors = list(client.remove_objects(bucket, objects))
    if errors:
        first = errors[0]
        raise SyncError(f"delete remote objects failed: {first}")
    _delete_base_manifest(object_type, case_id)
    logger.info("Regression MinIO deleted | type={} case_id={} objects={}", object_type, case_id, len(objects))
    return SyncResult(deleted=len(objects))


def check_publish_conflicts(object_type: str, case_id: str, local_dir: Path) -> ConflictReport | None:
    local_records = _scan_local_files(local_dir)
    local_manifest = _manifest_from_records(object_type, case_id, 0, list(local_records.values()), None)
    remote_manifest = _load_remote_manifest(_client(), bucket_name(), object_type, case_id)
    base_manifest = _load_base_manifest(object_type, case_id)
    report = _publish_conflicts(object_type, case_id, base_manifest, local_manifest, remote_manifest)
    return report if report.entries else None


def _desired_publish_records(
    object_type: str,
    case_id: str,
    local_manifest: CaseManifest,
    remote_manifest: CaseManifest | None,
    base_manifest: CaseManifest | None,
) -> list[FileRecord]:
    if remote_manifest is None:
        return list(local_manifest.files)

    if base_manifest is None:
        if _same_manifest_content(local_manifest, remote_manifest):
            return list(remote_manifest.files)
        report = ConflictReport(
            object_type=object_type,
            case_id=case_id,
            base_version=None,
            remote_version=remote_manifest.version,
            generated_at=utc_timestamp(),
            entries=[
                ConflictEntry(
                    path=".",
                    base_sha256=None,
                    local_sha256=None,
                    remote_sha256=None,
                    local_status="unknown_base",
                    remote_status="exists",
                    conflict_type="missing_base_manifest",
                    choices=["pull_remote_first", "keep_local_as_new_version"],
                )
            ],
        )
        raise VersionConflictError(f"missing local base manifest: {case_id}", report)

    if remote_manifest.version == base_manifest.version:
        return _reuse_unchanged_remote_objects(local_manifest, remote_manifest)

    report = _publish_conflicts(object_type, case_id, base_manifest, local_manifest, remote_manifest)
    conflicts = [entry for entry in report.entries if entry.conflict_type != "auto"]
    if conflicts:
        raise VersionConflictError(f"publish conflict: {case_id}", report)
    return _merge_non_conflicting_records(base_manifest, local_manifest, remote_manifest)


def _publish_conflicts(
    object_type: str,
    case_id: str,
    base_manifest: CaseManifest | None,
    local_manifest: CaseManifest,
    remote_manifest: CaseManifest | None,
) -> ConflictReport:
    if base_manifest is None or remote_manifest is None:
        return ConflictReport(object_type, case_id, None, remote_manifest.version if remote_manifest else None, utc_timestamp(), [])
    entries: list[ConflictEntry] = []
    base = base_manifest.file_map()
    local = local_manifest.file_map()
    remote = remote_manifest.file_map()
    for path in sorted(set(base) | set(local) | set(remote)):
        base_rec = base.get(path)
        local_rec = local.get(path)
        remote_rec = remote.get(path)
        local_status = _change_status(base_rec, local_rec)
        remote_status = _change_status(base_rec, remote_rec)
        conflict_type = _conflict_type(base_rec, local_rec, remote_rec, local_status, remote_status)
        if conflict_type in {"auto", "none"}:
            continue
        entries.append(
            ConflictEntry(
                path=path,
                base_sha256=base_rec.sha256 if base_rec else None,
                local_sha256=local_rec.sha256 if local_rec else None,
                remote_sha256=remote_rec.sha256 if remote_rec else None,
                local_status=local_status,
                remote_status=remote_status,
                conflict_type=conflict_type,
                choices=["use_local", "use_remote", "skip"],
            )
        )
    return ConflictReport(object_type, case_id, base_manifest.version, remote_manifest.version, utc_timestamp(), entries)


def _pull_conflicts(
    object_type: str,
    case_id: str,
    base_manifest: CaseManifest | None,
    local_manifest: CaseManifest,
    remote_manifest: CaseManifest,
) -> ConflictReport:
    if base_manifest is None:
        if _same_manifest_content(local_manifest, remote_manifest):
            return ConflictReport(object_type, case_id, None, remote_manifest.version, utc_timestamp(), [])
        return ConflictReport(
            object_type,
            case_id,
            None,
            remote_manifest.version,
            utc_timestamp(),
            [
                ConflictEntry(
                    path=".",
                    base_sha256=None,
                    local_sha256=None,
                    remote_sha256=None,
                    local_status="unknown_base",
                    remote_status="exists",
                    conflict_type="missing_base_manifest",
                    choices=["use_remote", "skip"],
                )
            ],
        )
    report = _publish_conflicts(object_type, case_id, base_manifest, local_manifest, remote_manifest)
    return ConflictReport(
        object_type,
        case_id,
        base_manifest.version,
        remote_manifest.version,
        utc_timestamp(),
        [entry for entry in report.entries if entry.local_status != "unchanged"],
    )


def _merge_non_conflicting_records(
    base_manifest: CaseManifest,
    local_manifest: CaseManifest,
    remote_manifest: CaseManifest,
) -> list[FileRecord]:
    result: dict[str, FileRecord] = {}
    base = base_manifest.file_map()
    local = local_manifest.file_map()
    remote = remote_manifest.file_map()
    for path in sorted(set(base) | set(local) | set(remote)):
        base_rec = base.get(path)
        local_rec = local.get(path)
        remote_rec = remote.get(path)
        local_status = _change_status(base_rec, local_rec)
        remote_status = _change_status(base_rec, remote_rec)
        if local_status == "unchanged" and remote_rec is not None:
            result[path] = remote_rec
        elif remote_status == "unchanged" and local_rec is not None:
            result[path] = local_rec
        elif local_rec and remote_rec and local_rec.sha256 == remote_rec.sha256:
            result[path] = remote_rec
        elif local_status == "deleted" and remote_status == "unchanged":
            continue
        elif remote_status == "deleted" and local_status == "unchanged":
            continue
        elif local_rec is not None and remote_rec is None and base_rec is None:
            result[path] = local_rec
        elif remote_rec is not None and local_rec is None and base_rec is None:
            result[path] = remote_rec
    return list(result.values())


def _reuse_unchanged_remote_objects(local_manifest: CaseManifest, remote_manifest: CaseManifest) -> list[FileRecord]:
    remote = remote_manifest.file_map()
    records = []
    for record in local_manifest.files:
        remote_record = remote.get(record.path)
        if remote_record and remote_record.sha256 == record.sha256:
            records.append(remote_record)
        else:
            records.append(record)
    return records


def _upload_new_records(
    client: Minio,
    bucket: str,
    object_type: str,
    case_id: str,
    version: int,
    local_dir: Path,
    desired_records: list[FileRecord],
) -> tuple[list[FileRecord], int]:
    uploaded = 0
    staging_id = uuid.uuid4().hex
    staging_prefix = f"{_remote_root(object_type, case_id)}/staging/{staging_id}"
    final_records: list[FileRecord] = []
    staged_objects: list[str] = []
    try:
        for record in desired_records:
            if record.object_name:
                final_records.append(record)
                continue
            local_path = _safe_join(local_dir, record.path)
            staging_name = f"{staging_prefix}/files/{record.path}"
            client.fput_object(bucket, staging_name, str(local_path), content_type=record.content_type)
            staged_objects.append(staging_name)
            stat = client.stat_object(bucket, staging_name)
            if int(stat.size or 0) != record.size:
                raise SyncError(f"staged object size mismatch: {record.path}")
            final_name = f"{_remote_root(object_type, case_id)}/versions/{version}/files/{record.path}"
            client.copy_object(bucket, final_name, CopySource(bucket, staging_name))
            final_records.append(
                FileRecord(
                    path=record.path,
                    size=record.size,
                    sha256=record.sha256,
                    content_type=record.content_type,
                    object_name=final_name,
                )
            )
            uploaded += 1
    finally:
        _delete_objects_best_effort(client, bucket, staged_objects)
    return final_records, uploaded


def _write_remote_manifest(
    client: Minio,
    bucket: str,
    object_type: str,
    case_id: str,
    manifest: CaseManifest,
) -> None:
    raw = _json_bytes(_manifest_to_dict(manifest))
    manifest_name = f"{_remote_root(object_type, case_id)}/versions/{manifest.version}/manifest.json"
    current_name = f"{_remote_root(object_type, case_id)}/current.json"
    client.put_object(bucket, manifest_name, _bytes_stream(raw), length=len(raw), content_type="application/json")
    client.put_object(bucket, current_name, _bytes_stream(raw), length=len(raw), content_type="application/json")


def _remote_cleanup_target(
    client: Minio,
    bucket: str,
    object_type: str,
    case_id: str,
    keep_versions: int,
) -> dict[str, Any]:
    version_manifests = _load_remote_version_manifests(client, bucket, object_type, case_id)
    versions = sorted(version_manifests)
    if not versions:
        return _empty_remote_cleanup_target(object_type, case_id)
    current_manifest = _load_remote_manifest(client, bucket, object_type, case_id)
    current_version = current_manifest.version if current_manifest else versions[-1]
    retained_versions = set(versions[-keep_versions:])
    retained_versions.add(current_version)
    delete_versions = [version for version in versions if version not in retained_versions]

    retained_refs: set[str] = set()
    for version in retained_versions:
        manifest = version_manifests.get(version)
        if not manifest:
            continue
        retained_refs.update(record.object_name for record in manifest.files if record.object_name)

    delete_objects: list[dict[str, Any]] = []
    for version in delete_versions:
        prefix = f"{_remote_root(object_type, case_id)}/versions/{version}/"
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            object_name = obj.object_name or ""
            if not object_name or object_name in retained_refs:
                continue
            delete_objects.append(
                {
                    "object_type": object_type,
                    "case_id": case_id,
                    "version": version,
                    "object_name": object_name,
                    "size": int(obj.size or 0),
                }
            )

    return {
        "object_type": object_type,
        "case_id": case_id,
        "current_version": current_version,
        "version_count": len(versions),
        "retained_versions": sorted(retained_versions),
        "delete_versions": delete_versions,
        "delete_object_count": len(delete_objects),
        "delete_bytes": sum(int(item["size"]) for item in delete_objects),
        "delete_objects": delete_objects,
    }


def _empty_remote_cleanup_target(object_type: str, case_id: str) -> dict[str, Any]:
    return {
        "object_type": object_type,
        "case_id": case_id,
        "current_version": "",
        "version_count": 0,
        "retained_versions": [],
        "delete_versions": [],
        "delete_object_count": 0,
        "delete_bytes": 0,
        "delete_objects": [],
    }


def _load_remote_version_manifests(client: Minio, bucket: str, object_type: str, case_id: str) -> dict[int, CaseManifest]:
    root = _remote_root(object_type, case_id)
    manifests: dict[int, CaseManifest] = {}
    for obj in client.list_objects(bucket, prefix=f"{root}/versions/", recursive=True):
        name = obj.object_name or ""
        if not name.endswith("/manifest.json"):
            continue
        parts = PurePosixPath(name[len(root):].strip("/")).parts
        if len(parts) != 3 or parts[0] != "versions" or parts[2] != "manifest.json":
            continue
        try:
            version = int(parts[1])
        except ValueError:
            continue
        try:
            response = client.get_object(bucket, name)
            try:
                raw = response.read()
            finally:
                response.close()
                response.release_conn()
            manifests[version] = _manifest_from_dict(json.loads(raw.decode("utf-8")))
        except Exception as exc:
            logger.warning("Regression MinIO version manifest read failed | object={} err={}", name, exc)
    return manifests


def _load_remote_manifest(client: Minio, bucket: str, object_type: str, case_id: str) -> CaseManifest | None:
    current_name = f"{_remote_root(object_type, case_id)}/current.json"
    try:
        response = client.get_object(bucket, current_name)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            return None
        raise
    return _manifest_from_dict(json.loads(raw.decode("utf-8")))


def _list_remote_case_ids(object_type: str) -> list[str]:
    client = _client()
    bucket = bucket_name()
    _ensure_bucket(client, bucket)
    root = _object_name("cases" if object_type == "case" else "candidates")
    ids = []
    for obj in client.list_objects(bucket, prefix=root + "/", recursive=True):
        name = obj.object_name or ""
        if not name.endswith("/current.json"):
            continue
        rel = name[len(root):].strip("/")
        if rel.endswith("/current.json"):
            ids.append(rel[: -len("/current.json")])
    return sorted(ids)


def _scan_local_files(local_dir: Path) -> dict[str, FileRecord]:
    records: dict[str, FileRecord] = {}
    for path in _iter_files(local_dir):
        rel = _safe_relative_path(path.relative_to(local_dir))
        if rel in records:
            raise PathSafetyError(f"duplicate relative path: {rel}")
        records[rel] = FileRecord(
            path=rel,
            size=path.stat().st_size,
            sha256=_sha256_file(path),
            content_type=_content_type(path),
        )
    return records


def _manifest_from_records(
    object_type: str,
    case_id: str,
    version: int,
    records: list[FileRecord],
    existing: CaseManifest | None,
) -> CaseManifest:
    endpoint, case_name = _endpoint_and_name(object_type, case_id)
    now = utc_timestamp()
    return CaseManifest(
        object_type=object_type,
        case_id=case_id,
        endpoint=endpoint,
        case_name=case_name,
        version=version,
        files=sorted(records, key=lambda item: item.path),
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )


def _manifest_to_dict(manifest: CaseManifest) -> dict[str, Any]:
    data = asdict(manifest)
    data["files"] = [asdict(record) for record in manifest.files]
    data["manifest_sha256"] = _manifest_hash(data)
    return data


def _manifest_from_dict(data: dict[str, Any]) -> CaseManifest:
    files = [FileRecord(**record) for record in data.get("files", [])]
    return CaseManifest(
        object_type=str(data["object_type"]),
        case_id=str(data["case_id"]),
        endpoint=str(data.get("endpoint") or ""),
        case_name=str(data.get("case_name") or ""),
        version=int(data["version"]),
        files=files,
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
    )


def _base_manifest_path(object_type: str, case_id: str) -> Path:
    rel_parts = ["cases" if object_type == "case" else "candidates", *case_id.split("/"), "base_manifest.json"]
    return SYNC_STATE_DIR.joinpath(*rel_parts)


def _load_base_manifest(object_type: str, case_id: str) -> CaseManifest | None:
    path = _base_manifest_path(object_type, case_id)
    if not path.is_file():
        return None
    return _manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _save_base_manifest(object_type: str, case_id: str, manifest: CaseManifest) -> None:
    path = _base_manifest_path(object_type, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_manifest_to_dict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def _delete_base_manifest(object_type: str, case_id: str) -> None:
    path = _base_manifest_path(object_type, case_id)
    if path.is_file():
        path.unlink()


def _local_dir_for_case_id(object_type: str, case_id: str) -> Path:
    _validate_case_id(object_type, case_id)
    if object_type == "case":
        endpoint, case_name = case_id.split("/", 1)
        return (CASES_DIR / _safe_name(endpoint) / _safe_name(case_name)).resolve()
    return (CANDIDATES_DIR / _safe_name(case_id)).resolve()


def _archive_conflict_report(path: Path, action: str) -> Path:
    resolved_dir = CONFLICT_DIR / "resolved"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    target = resolved_dir / f"{path.stem}.{action}.resolved.json"
    index = 1
    while target.exists():
        target = resolved_dir / f"{path.stem}.{action}.resolved_{index}.json"
        index += 1
    shutil.move(str(path), str(target))
    return target


def _write_conflict_report(report: ConflictReport | None) -> Path | None:
    if report is None:
        return None
    CONFLICT_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = report.case_id.replace("/", "__")
    path = CONFLICT_DIR / f"{utc_timestamp().replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}_{report.object_type}_{safe_id}.json"
    path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _change_status(base: FileRecord | None, current: FileRecord | None) -> str:
    if base is None and current is None:
        return "absent"
    if base is None and current is not None:
        return "added"
    if base is not None and current is None:
        return "deleted"
    if base and current and base.sha256 == current.sha256:
        return "unchanged"
    return "modified"


def _conflict_type(
    base: FileRecord | None,
    local: FileRecord | None,
    remote: FileRecord | None,
    local_status: str,
    remote_status: str,
) -> str:
    if local_status == "unchanged" or remote_status == "unchanged":
        return "none"
    if local is not None and remote is not None and local.sha256 == remote.sha256:
        return "none"
    if local_status == "absent" or remote_status == "absent":
        return "none"
    if local_status == "deleted" and remote_status in {"modified", "added"}:
        return "delete_vs_modify"
    if remote_status == "deleted" and local_status in {"modified", "added"}:
        return "modify_vs_delete"
    if base is None and local_status == "added" and remote_status == "added":
        return "both_added_different"
    if local_status in {"modified", "added"} and remote_status in {"modified", "added"}:
        return "both_modified_different"
    return "auto"


def _same_manifest_content(left: CaseManifest, right: CaseManifest) -> bool:
    left_files = {path: record.sha256 for path, record in left.file_map().items()}
    right_files = {path: record.sha256 for path, record in right.file_map().items()}
    return left_files == right_files


def _object_detail(
    operation: str,
    object_type: str,
    case_id: str,
    before: CaseManifest | None,
    after: CaseManifest,
    transferred_files: int,
) -> dict[str, Any]:
    changes = _file_changes(before, after)
    if before is None:
        change_type = "created"
    elif changes:
        change_type = "updated"
    else:
        change_type = "unchanged"
    return {
        "operation": operation,
        "object_type": object_type,
        "case_id": case_id,
        "change_type": change_type,
        "before_version": before.version if before else None,
        "after_version": after.version,
        "file_count": len(after.files),
        "transferred_files": transferred_files,
        "changed_file_count": len(changes),
        "changed_files": changes[:100],
    }


def _conflict_object_detail(object_type: str, case_id: str, report: ConflictReport | None) -> dict[str, Any]:
    entries = report.entries if report else []
    return {
        "operation": "conflict",
        "object_type": object_type,
        "case_id": case_id,
        "change_type": "conflict",
        "before_version": report.base_version if report else None,
        "after_version": report.remote_version if report else None,
        "file_count": 0,
        "transferred_files": 0,
        "changed_file_count": len(entries),
        "changed_files": [
            {
                "path": entry.path,
                "change_type": entry.conflict_type,
                "before_sha256": entry.base_sha256,
                "after_sha256": entry.remote_sha256,
            }
            for entry in entries[:100]
        ],
    }


def _file_changes(before: CaseManifest | None, after: CaseManifest) -> list[dict[str, Any]]:
    before_map = before.file_map() if before else {}
    after_map = after.file_map()
    changes: list[dict[str, Any]] = []
    for path in sorted(set(before_map) | set(after_map)):
        before_record = before_map.get(path)
        after_record = after_map.get(path)
        if before_record and after_record and before_record.sha256 == after_record.sha256:
            continue
        if before_record is None:
            change_type = "added"
        elif after_record is None:
            change_type = "removed"
        else:
            change_type = "modified"
        changes.append(
            {
                "path": path,
                "change_type": change_type,
                "before_sha256": before_record.sha256 if before_record else "",
                "after_sha256": after_record.sha256 if after_record else "",
                "size": after_record.size if after_record else 0,
            }
        )
    return changes


def _verify_file(path: Path, record: FileRecord) -> None:
    stat = path.stat()
    if stat.st_size != record.size:
        raise SyncError(f"download size mismatch: {record.path}")
    if _sha256_file(path) != record.sha256:
        raise SyncError(f"download sha256 mismatch: {record.path}")


def _client() -> Minio:
    global _CLIENT
    if _CLIENT is None:
        cfg = config.minio
        _CLIENT = Minio(
            cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
        )
    return _CLIENT


def _ensure_bucket(client: Minio, bucket: str) -> None:
    try:
        exists = client.bucket_exists(bucket)
    except S3Error:
        raise
    if exists:
        return
    if not auto_create_bucket():
        raise RemoteObjectNotFoundError(f"MinIO bucket does not exist: {bucket}")
    client.make_bucket(bucket)
    logger.info("Regression MinIO bucket created | bucket={}", bucket)


def _remote_root(object_type: str, case_id: str) -> str:
    folder = "cases" if object_type == "case" else "candidates"
    return _object_name(folder, case_id)


def _object_name(*parts: str) -> str:
    values = [part.strip("/") for part in parts if part and part.strip("/")]
    prefix = object_prefix()
    if prefix:
        values.insert(0, prefix)
    return "/".join(values)


def _endpoint_and_name(object_type: str, case_id: str) -> tuple[str, str]:
    if object_type == "case":
        endpoint, case_name = case_id.split("/", 1)
        return endpoint, case_name
    return "", case_id


def _validate_case_id(object_type: str, case_id: str) -> None:
    if object_type not in {"case", "candidate"}:
        raise ValueError(f"unsupported object_type: {object_type}")
    parts = case_id.split("/")
    expected_len = 2 if object_type == "case" else 1
    if len(parts) != expected_len:
        raise PathSafetyError(f"invalid {object_type} id: {case_id}")
    for part in parts:
        _safe_name(part)


def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        raise PathSafetyError("empty or unsafe name")
    if Path(text).name != text or "\\" in text or "/" in text or ":" in text:
        raise PathSafetyError(f"unsafe name: {value}")
    return text


def _safe_relative_path(path: Path | PurePosixPath) -> str:
    raw = path.as_posix()
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PathSafetyError(f"unsafe relative path: {raw}")
    if any("\\" in part or ":" in part for part in pure.parts):
        raise PathSafetyError(f"unsafe relative path: {raw}")
    return pure.as_posix()


def _safe_join(root: Path, rel_path: str) -> Path:
    rel = _safe_relative_path(PurePosixPath(rel_path))
    target = (root / Path(*PurePosixPath(rel).parts)).resolve()
    target.relative_to(root.resolve())
    return target


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _validate_local_cleanup_target(path: Path) -> None:
    allowed_roots = [
        ROOT_DIR / "case_history",
        ROOT_DIR / "case_archive",
        ROOT_DIR / "reports" / "_archive",
        CONFLICT_DIR / "resolved",
    ]
    resolved = path.resolve()
    for root in allowed_roots:
        root_resolved = root.resolve()
        try:
            relative = resolved.relative_to(root_resolved)
        except ValueError:
            continue
        if not relative.parts:
            break
        if relative.parts[0] in {".", ".."}:
            break
        return
    raise PathSafetyError(f"unsafe local cleanup target: {path}")


def utc_timestamp_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", ".pytest_cache", ".sync_state"} for part in path.parts):
            continue
        yield path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(data: dict[str, Any]) -> str:
    payload = dict(data)
    payload.pop("manifest_sha256", None)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _bytes_stream(data: bytes):
    import io

    return io.BytesIO(data)


def _delete_objects_best_effort(client: Minio, bucket: str, object_names: list[str]) -> None:
    if not object_names:
        return
    try:
        list(client.remove_objects(bucket, [DeleteObject(name) for name in object_names]))
    except Exception as exc:
        logger.warning("Regression MinIO staging cleanup failed | count={} err={}", len(object_names), exc)


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    if suffix == ".html":
        return "text/html"
    if suffix == ".md":
        return "text/markdown"
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"
