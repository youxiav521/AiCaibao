from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.policy.policy_check_agent.single_file_policy_checks.router import (
    _build_audited_results,
    _build_completed_payload,
)
from src.agents.policy.policy_check_agent.single_file_policy_checks.schemas import (
    SingleFilePolicyAuditedPayload,
    SingleFilePolicyCheckRequest,
)
from src.agents.policy.policy_check_agent.single_file_policy_checks.single_file_policy_check_service import (
    SingleFilePolicyCheckService,
)


DEFAULT_MODEL_ID = "model_bailian_policy_check"
DEFAULT_MAX_CONCURRENCY = 12
DEFAULT_CONCURRENCY = 4
DEFAULT_OUTPUT_DIR = "storage/policy_check/single_file_policy_local_tests"

# 测试时优先改这里：不传命令行参数时，脚本会使用这些默认值。
DEFAULT_INPUT_PATH = "D:\\dev\\yusys_ai\\storage\\mineru_output\\converted_output\\42fec53197894830b73d8e43ff508ab0CUUXwEmZy6_mineru_converted.json"
DEFAULT_FRAMEWORKS_PATH = "D:\\dev\\yusys_ai\\storage\\mineru_output\\converted_output\\2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8_framework_output.qwen35_list_ids.json"
DEFAULT_CHECK_REPORT_ID = "LOCAL-POLICY-DEMO"
# 目录输入时自动把 MinerU JSON 聚合成一个资源库请求（推荐开启）。
DEFAULT_AUTO_PACK_REPOSITORY_INPUT = True


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_project_root_cwd() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "src").is_dir() and (candidate / "config").is_dir():
            os.chdir(candidate)
            return candidate
    return Path.cwd()


def _strip_wrapped_value(value: str) -> str:
    raw = value.strip()
    if len(raw) >= 2 and ((raw[0] == raw[-1] == '"') or (raw[0] == raw[-1] == "'")):
        return raw[1:-1]
    return raw


def _parse_dotenv_file(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            continue
        result[key] = _strip_wrapped_value(value)
    return result


def _parse_compose_environment(path: Path) -> Dict[str, str]:
    # 仅提取 docker-compose 风格 yml 里的 environment 块，避免误抓其他字段。
    envs: Dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    in_environment = False
    env_indent = 0
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if re.match(r"^environment\s*:\s*$", stripped):
            in_environment = True
            env_indent = indent
            continue
        if in_environment:
            if indent <= env_indent:
                in_environment = False
                continue
            # KEY: value
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$", stripped)
            if m:
                key = m.group(1).strip()
                value = _strip_wrapped_value(m.group(2))
                envs[key] = value
                continue
            # - KEY=value
            m = re.match(r"^-\s*([A-Za-z_][A-Za-z0-9_]*)=(.+?)\s*$", stripped)
            if m:
                key = m.group(1).strip()
                value = _strip_wrapped_value(m.group(2))
                envs[key] = value
    return envs


def _load_local_env_files(project_root: Path) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    # 优先读取 .env / .env.local，再补充 env.yaml/env.yml（仅 environment 块）。
    candidates = [
        project_root / ".env",
        project_root / ".env.local",
        project_root / "env.yaml",
        project_root / "env.yml",
    ]
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            if path.suffix.lower() in {".yml", ".yaml"}:
                parsed = _parse_compose_environment(path)
            else:
                parsed = _parse_dotenv_file(path)
            for key, value in parsed.items():
                if key and value is not None:
                    # 不覆盖已存在环境变量，尊重外部注入配置。
                    if key not in os.environ:
                        os.environ[key] = value
                    loaded[key] = value
        except Exception:
            # 本地测试脚本容错：解析失败不打断主流程。
            continue
    return loaded


def _build_callback_payload(*, event: str, check_report_id: str, file_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": event,
        "checkReportId": check_report_id,
        "fileId": file_id,
        "category": "COMPLIANCE",
        "data": data,
    }


def _print_payload(title: str, payload: dict[str, Any]) -> None:
    print(f"\n===== {title} =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_frameworks_payload(raw: Any) -> List[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        # 1) 完整包装：{"frameworks":[...]}
        if isinstance(raw.get("frameworks"), list):
            return [item for item in raw["frameworks"] if isinstance(item, dict)]
        # 2) 结果包装：{"result":{"frameworks":[...]}}
        result_obj = raw.get("result")
        if isinstance(result_obj, dict) and isinstance(result_obj.get("frameworks"), list):
            return [item for item in result_obj["frameworks"] if isinstance(item, dict)]
        # 3) 泛化包装：{"data":[...]}（且元素像 framework）
        if isinstance(raw.get("data"), list):
            candidates = [item for item in raw["data"] if isinstance(item, dict)]
            if candidates and any("frameworkId" in item or "requirements" in item for item in candidates):
                return candidates
        # 4) 单 framework 对象
        if "frameworkId" in raw and isinstance(raw.get("requirements"), list):
            return [raw]
    return []


def _load_frameworks_from_path(frameworks_path: Optional[str]) -> Optional[List[dict[str, Any]]]:
    if not frameworks_path:
        return None
    raw = _load_json(Path(frameworks_path))
    frameworks = _normalize_frameworks_payload(raw)
    return frameworks or None


def _sanitize_filename_part(text: str, limit: int = 40) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text or ""))
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "unknown"
    return cleaned[:limit]


def _resolve_input_file_id(check_report_id: str, index: int, file_id: Optional[str]) -> str:
    value = (file_id or "").strip()
    if value:
        return value
    return f"{check_report_id}_{index + 1}"


def _resolve_input_file_name(file_id: str, file_name: Optional[str]) -> str:
    value = (file_name or "").strip()
    if value:
        return value
    return f"{file_id}.json"


def _convert_resource_repository_to_files(repo_payload: dict[str, Any]) -> List[dict[str, Any]]:
    files_payload = repo_payload.get("files") or []
    if not isinstance(files_payload, list):
        raise ValueError("resource_repository.files 必须是数组")
    files: List[dict[str, Any]] = []
    for index, file_payload in enumerate(files_payload):
        if not isinstance(file_payload, dict):
            continue
        file_id = str(file_payload.get("fileId") or file_payload.get("file_id") or f"repo_file_{index + 1}")
        file_name = str(file_payload.get("fileName") or file_payload.get("file_name") or f"{file_id}.json")
        doc_content = file_payload.get("docContent")
        if doc_content is None:
            doc_content = file_payload.get("doc_content")
        if doc_content is None:
            doc_content = file_payload.get("content")
        if not isinstance(doc_content, list):
            raise ValueError(f"resource_repository.files[{index}] 缺少 content/docContent 数组")
        files.append(
            {
                "fileId": file_id,
                "fileName": file_name,
                "docContent": doc_content,
            }
        )
    return files


def _normalize_request_payload(
    raw: Any,
    *,
    input_path: Path,
    seq: int,
    frameworks_path: Optional[str],
) -> dict[str, Any]:
    if isinstance(raw, list):
        frameworks = _load_frameworks_from_path(frameworks_path)
        if not frameworks:
            raise ValueError("MinerU 数组输入需要 frameworks，请配置 DEFAULT_FRAMEWORKS_PATH 或 --frameworks-path")
        file_id = input_path.stem
        return {
            "checkReportId": f"{DEFAULT_CHECK_REPORT_ID}-{seq:03d}",
            "files": [
                {
                    "fileId": file_id,
                    "fileName": input_path.name,
                    "docContent": raw,
                }
            ],
            "frameworks": frameworks,
            "metadata": {"autoPackedFromMineru": True},
        }

    if not isinstance(raw, dict):
        raise ValueError("输入 JSON 必须是对象，或 MinerU 数组")

    if isinstance(raw.get("files"), list) and isinstance(raw.get("frameworks"), list):
        payload = dict(raw)
        payload.setdefault("checkReportId", raw.get("checkReportId") or f"{DEFAULT_CHECK_REPORT_ID}-{seq:03d}")
        return payload

    resource_repo = raw.get("resource_repository") or raw.get("resourceRepository")
    if isinstance(resource_repo, dict):
        files = _convert_resource_repository_to_files(resource_repo)
        frameworks = raw.get("frameworks")
        if not isinstance(frameworks, list):
            if frameworks_path:
                frameworks = _load_frameworks_from_path(frameworks_path)
            else:
                raise ValueError("resource_repository 输入缺少 frameworks，且未配置 DEFAULT_FRAMEWORKS_PATH")
        if not isinstance(frameworks, list):
            raise ValueError("frameworks 必须是数组")
        return {
            "checkReportId": raw.get("checkReportId") or f"{DEFAULT_CHECK_REPORT_ID}-{seq:03d}",
            "files": files,
            "frameworks": frameworks,
            "metadata": raw.get("metadata") or {},
        }

    if isinstance(raw.get("files"), list) and not raw.get("frameworks") and frameworks_path:
        frameworks = _load_frameworks_from_path(frameworks_path)
        if not isinstance(frameworks, list):
            raise ValueError("DEFAULT_FRAMEWORKS_PATH 内容必须是 frameworks 数组")
        payload = dict(raw)
        payload["frameworks"] = frameworks
        payload.setdefault("checkReportId", raw.get("checkReportId") or f"{DEFAULT_CHECK_REPORT_ID}-{seq:03d}")
        return payload

    raise ValueError("输入 JSON 需为完整请求（files+frameworks）或包含 resource_repository/resourceRepository")


def _extract_doc_content_from_raw(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("docContent", "doc_content", "content"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return None


def _build_request_from_mineru_directory(
    *,
    input_dir: Path,
    frameworks_path: Optional[str],
    check_report_id: str,
) -> dict[str, Any]:
    json_files = sorted(p for p in input_dir.rglob("*.json") if p.is_file())
    if not json_files:
        raise ValueError(f"目录下未发现 JSON 文件: {input_dir}")

    frameworks = _load_frameworks_from_path(frameworks_path)
    if not isinstance(frameworks, list):
        raise ValueError("目录自动聚合模式需要 frameworks，请配置 DEFAULT_FRAMEWORKS_PATH 或 --frameworks-path")

    files_payload: List[dict[str, Any]] = []
    for index, file_path in enumerate(json_files):
        raw = _load_json(file_path)
        doc_content = _extract_doc_content_from_raw(raw)
        if not isinstance(doc_content, list):
            raise ValueError(f"文件不属于 MinerU 可识别格式（缺少 content/docContent 数组）: {file_path}")

        file_id = file_path.stem
        file_name = file_path.name
        if isinstance(raw, dict):
            file_id = str(raw.get("fileId") or raw.get("file_id") or file_id)
            file_name = str(raw.get("fileName") or raw.get("file_name") or file_name)
        if not file_id:
            file_id = f"{input_dir.stem}_{index + 1}"
        if not file_name:
            file_name = f"{file_id}.json"

        files_payload.append(
            {
                "fileId": file_id,
                "fileName": file_name,
                "docContent": doc_content,
            }
        )

    return {
        "checkReportId": check_report_id,
        "files": files_payload,
        "frameworks": frameworks,
        "metadata": {
            "autoPackedFromDirectory": True,
            "sourceDirectory": str(input_dir),
            "sourceFileCount": len(files_payload),
        },
    }


def _apply_overrides(
    payload: dict[str, Any],
    *,
    model_id: str,
    max_concurrency: int,
) -> SingleFilePolicyCheckRequest:
    normalized = dict(payload)
    normalized["modelId"] = model_id
    normalized["maxConcurrency"] = max_concurrency
    return SingleFilePolicyCheckRequest.model_validate(normalized)


async def _run_one(
    *,
    input_path: Path,
    request: SingleFilePolicyCheckRequest,
    output_dir: Path,
) -> dict[str, Any]:
    service = SingleFilePolicyCheckService(
        model_id=request.model_id,
        max_concurrency=request.max_concurrency,
    )
    started_at = datetime.now(UTC)

    ordered_file_keys: list[tuple[str, str]] = []
    started_events: list[dict[str, Any]] = []
    for index, input_file in enumerate(request.files):
        file_id = _resolve_input_file_id(request.check_report_id, index, input_file.file_id)
        file_name = _resolve_input_file_name(file_id, input_file.file_name)
        ordered_file_keys.append((file_id, file_name))
        started_event = _build_callback_payload(
            event="started",
            check_report_id=request.check_report_id,
            file_id=file_id,
            data={"fileStatus": "P"},
        )
        started_events.append(started_event)
        _print_payload(f"{input_path.name} {file_id} started", started_event)

    try:
        hybrid_results = await service.analyze_request_hybrid(request)
    except Exception as exc:
        error_events: list[dict[str, Any]] = []
        per_file_status: list[dict[str, Any]] = []
        error_message = f"{type(exc).__name__}: {exc}"
        for file_id, file_name in ordered_file_keys:
            error_event = _build_callback_payload(
                event="error",
                check_report_id=request.check_report_id,
                file_id=file_id,
                data={"errorMessage": error_message},
            )
            error_events.append(error_event)
            per_file_status.append(
                {
                    "fileId": file_id,
                    "fileName": file_name,
                    "status": "FAILED",
                    "error": error_message,
                    "event": error_event,
                }
            )
            _print_payload(f"{input_path.name} {file_id} error", error_event)

        output = {
            "meta": {
                "inputFile": str(input_path),
                "checkReportId": request.check_report_id,
                "modelId": request.model_id,
                "maxConcurrency": request.max_concurrency,
                "fileCount": len(request.files),
                "failedFileCount": len(request.files),
                "startedAt": _utc_now_iso(),
                "fatalError": error_message,
            },
            "callbacks": {
                "started": started_events,
                "audited": [],
                "completed": [],
                "error": error_events,
            },
            "files": per_file_status,
        }
        readable_report = _sanitize_filename_part(request.check_report_id, limit=32)
        digest_src = f"{request.check_report_id}|{input_path.name}|fatal|{error_message}"
        digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:12]
        out_name = f"{readable_report}_{digest}.json"
        out_file = output_dir / out_name
        try:
            out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            fallback_out_file = output_dir / f"result_{digest}.json"
            fallback_out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            out_file = fallback_out_file
        print(f"\n[Saved] {out_file}")
        return output

    results_by_file_id: dict[str, list[Any]] = {}
    for result in hybrid_results:
        results_by_file_id.setdefault(result.file_id, []).append(result)

    audited_events: list[dict[str, Any]] = []
    completed_events: list[dict[str, Any]] = []
    error_events: list[dict[str, Any]] = []
    per_file_status: list[dict[str, Any]] = []
    failed_file_count = 0

    for file_id, file_name in ordered_file_keys:
        result_bucket = results_by_file_id.get(file_id) or []
        result = result_bucket.pop(0) if result_bucket else None
        if result is None:
            failed_file_count += 1
            file_error = "Hybrid execution returned no result for current file."
            error_event = _build_callback_payload(
                event="error",
                check_report_id=request.check_report_id,
                file_id=file_id,
                data={"errorMessage": file_error},
            )
            error_events.append(error_event)
            per_file_status.append(
                {
                    "fileId": file_id,
                    "fileName": file_name,
                    "status": "FAILED",
                    "error": file_error,
                    "event": error_event,
                }
            )
            _print_payload(f"{input_path.name} {file_id} error", error_event)
            continue

        audited_payload = SingleFilePolicyAuditedPayload(
            file_id=result.file_id,
            doc_file_name=result.file_name,
            file_status="F" if result.overall_status == "fail" else "S",
            results=_build_audited_results(request=request, result=result),
        ).model_dump(mode="json", by_alias=True)
        duration_seconds = (datetime.now(UTC) - started_at).total_seconds()
        completed_payload = _build_completed_payload(result, duration_seconds).model_dump(mode="json", by_alias=True)
        audited_event = _build_callback_payload(
            event="audited",
            check_report_id=request.check_report_id,
            file_id=result.file_id,
            data=audited_payload,
        )
        completed_event = _build_callback_payload(
            event="completed",
            check_report_id=request.check_report_id,
            file_id=result.file_id,
            data=completed_payload,
        )
        audited_events.append(audited_event)
        completed_events.append(completed_event)
        per_file_status.append(
            {
                "fileId": result.file_id,
                "fileName": result.file_name,
                "status": "SUCCESS",
                "audited": audited_event,
                "completed": completed_event,
                "result": result.model_dump(mode="json", by_alias=True),
            }
        )
        _print_payload(f"{input_path.name} {result.file_id} audited", audited_event)
        _print_payload(f"{input_path.name} {result.file_id} completed", completed_event)

    output = {
        "meta": {
            "inputFile": str(input_path),
            "checkReportId": request.check_report_id,
            "modelId": request.model_id,
            "maxConcurrency": request.max_concurrency,
            "fileCount": len(request.files),
            "failedFileCount": failed_file_count,
            "startedAt": _utc_now_iso(),
        },
        "callbacks": {
            "started": started_events,
            "audited": audited_events,
            "completed": completed_events,
            "error": error_events,
        },
        "files": per_file_status,
    }

    readable_report = _sanitize_filename_part(request.check_report_id, limit=32)
    digest_src = f"{request.check_report_id}|{input_path.name}|{len(request.files)}"
    digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:12]
    out_name = f"{readable_report}_{digest}.json"
    out_file = output_dir / out_name
    try:
        out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        fallback_out_file = output_dir / f"result_{digest}.json"
        fallback_out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        out_file = fallback_out_file
    print(f"\n[Saved] {out_file}")
    return output


async def _run_all(
    *,
    input_files: List[Path],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    input_root = Path(args.input)

    if args.auto_pack_repository and input_root.is_dir():
        check_report_id = f"{DEFAULT_CHECK_REPORT_ID}-{input_root.stem or 'DIR'}"
        payload = _build_request_from_mineru_directory(
            input_dir=input_root,
            frameworks_path=args.frameworks_path,
            check_report_id=check_report_id,
        )
        request = _apply_overrides(
            payload,
            model_id=args.model_id,
            max_concurrency=args.max_concurrency,
        )
        await _run_one(input_path=input_root, request=request, output_dir=output_dir)
        return

    async def worker(idx: int, path: Path) -> None:
        raw = _load_json(path)
        payload = _normalize_request_payload(
            raw,
            input_path=path,
            seq=idx + 1,
            frameworks_path=args.frameworks_path,
        )
        request = _apply_overrides(
            payload,
            model_id=args.model_id,
            max_concurrency=args.max_concurrency,
        )
        async with semaphore:
            await _run_one(input_path=path, request=request, output_dir=output_dir)

    await asyncio.gather(*(worker(idx, path) for idx, path in enumerate(input_files)))


def _discover_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.json") if p.is_file())
    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单文件合规检查本地测试脚本（资源库输入，按文件打印回调并落盘输出）"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "输入 JSON 文件路径或目录；支持完整请求（files+frameworks）"
            "或 resource_repository/resourceRepository。"
            "不传则使用文件头 DEFAULT_INPUT_PATH"
        ),
    )
    parser.add_argument(
        "--frameworks-path",
        default=DEFAULT_FRAMEWORKS_PATH,
        help="当输入不含 frameworks 时，可指定 frameworks JSON 文件路径",
    )
    parser.add_argument(
        "--auto-pack-repository",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_PACK_REPOSITORY_INPUT,
        help="目录输入时自动聚合 MinerU JSON 为单次资源库请求，默认开启",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"模型 ID，默认: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"服务内部控制执行并发，默认: {DEFAULT_MAX_CONCURRENCY}",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"多输入文件并发数（批量时生效），默认: {DEFAULT_CONCURRENCY}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录，默认: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def main() -> None:
    project_root = _ensure_project_root_cwd()
    loaded_env = _load_local_env_files(project_root)
    parser = _build_parser()
    args = parser.parse_args()
    args.input = args.input or DEFAULT_INPUT_PATH

    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency 必须 >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency 必须 >= 1")

    input_files = _discover_input_files(Path(args.input))
    if not input_files:
        raise ValueError("未发现可测试的 JSON 文件")

    print(
        json.dumps(
            {
                "projectRoot": str(project_root),
                "loadedEnvCount": len(loaded_env),
                "hasBailianBaseUrl": bool(os.getenv("BAILIAN_API_BASE_URL")),
                "hasBailianApiKey": bool(os.getenv("BAILIAN_API_KEY")),
                "inputPath": args.input,
                "inputCount": len(input_files),
                "frameworksPath": args.frameworks_path,
                "autoPackRepository": args.auto_pack_repository,
                "modelId": args.model_id,
                "maxConcurrency": args.max_concurrency,
                "batchConcurrency": args.concurrency,
                "outputDir": args.output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    asyncio.run(_run_all(input_files=input_files, args=args))


if __name__ == "__main__":
    main()
