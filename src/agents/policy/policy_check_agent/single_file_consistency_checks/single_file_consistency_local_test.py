from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.policy.policy_check_agent.single_file_consistency_checks.schemas import (
    SingleDocConsistencyCheckRequest,
)
from src.agents.policy.policy_check_agent.single_file_consistency_checks.service import (
    SingleDocConsistencyService,
)


DEFAULT_MODEL_ID = "model_bailian_policy_check"
DEFAULT_ENABLE_LLM = True
DEFAULT_LLM_RETRY_ATTEMPTS = 3
DEFAULT_MAX_CANDIDATES = 24
DEFAULT_MAX_JUDGE_CONCURRENCY = 12
DEFAULT_CONCURRENCY = 8
DEFAULT_OUTPUT_DIR = "storage/policy_check/single_file_consistency_local_tests"

# 测试时优先改这里：不传命令行参数时，脚本会使用这些默认值。
DEFAULT_INPUT_PATH = "D:\\dev\\yusys_ai\\storage\\mineru_output\\converted_output\\5e507223f37a460b89f0ced860936a69vNKBvYNfSZ_mineru_converted.json"
DEFAULT_CHECK_TYPES: Optional[List[str]] = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _to_callback_status(status: str) -> str:
    normalized = (status or "").lower()
    if normalized == "inconsistent":
        return "fail"
    if normalized in {"possibly_inconsistent", "insufficient_information"}:
        return "uncertain"
    return "pass"


def _to_callback_level(severity: str, status: str) -> str:
    callback_status = _to_callback_status(status)
    if callback_status == "pass":
        return "NONE"
    normalized = (severity or "").upper()
    if normalized in {"HIGH", "MEDIUM", "LOW"}:
        return normalized
    return "MEDIUM"


def _build_callback_payload(
    *,
    check_report_id: str,
    file_id: str,
    event: str,
    category: str = "CONSISTENCY",
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "checkReportId": check_report_id,
        "fileId": file_id,
        "event": event,
        "category": category,
        "data": data or {},
    }


def _build_audited_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    file_id = str(result.get("fileId") or "")
    file_name = str(result.get("fileName") or "")
    rows: list[dict[str, Any]] = []
    for issue in result.get("issues", []) or []:
        evidence_rows = []
        for index, evidence in enumerate(issue.get("evidence", []) or [], start=1):
            location = evidence.get("sourceLocation", {}) or {}
            evidence_rows.append(
                {
                    "sequenceId": index,
                    "fileId": location.get("fileId") or file_id,
                    "elementIds": location.get("elementIds") or [],
                    "pageNo": location.get("pageNo"),
                    "segmentId": location.get("segmentId"),
                    "blockType": location.get("blockType"),
                    "role": evidence.get("role") or "",
                    "quote": evidence.get("sourceText") or "",
                    "recommendation": issue.get("reasoning") or "",
                }
            )
        rows.append(
            {
                "scope": "file",
                "category": "CONSISTENCY",
                "fileId": file_id,
                "docFileName": file_name,
                "issueId": issue.get("issueId") or "",
                "issueType": issue.get("issueType") or "",
                "controlResult": {
                    "status": _to_callback_status(str(issue.get("status") or "")),
                    "level": _to_callback_level(str(issue.get("severity") or ""), str(issue.get("status") or "")),
                    "title": issue.get("summary") or "",
                    "detail": issue.get("reasoning") or "",
                    "suggestion": issue.get("suggestion") or "",
                    "degraded": False,
                },
                "attributes": issue.get("attributes") or {},
                "evidence": evidence_rows,
            }
        )
    if not rows:
        rows.append(
            {
                "scope": "file",
                "category": "CONSISTENCY",
                "fileId": file_id,
                "docFileName": file_name,
                "issueId": "",
                "issueType": "NO_ISSUE",
                "controlResult": {
                    "status": "pass",
                    "level": "NONE",
                    "title": result.get("summary") or "未发现一致性问题",
                    "detail": result.get("summary") or "未发现一致性问题",
                    "suggestion": "",
                    "degraded": False,
                },
                "attributes": {"resultType": "summary"},
                "evidence": [],
            }
        )
    return rows


def _build_completed_summary(result: dict[str, Any], duration_seconds: float) -> dict[str, Any]:
    issues = result.get("issues", []) or []
    inconsistent = sum(1 for issue in issues if str(issue.get("status") or "").lower() == "inconsistent")
    possibly_inconsistent = sum(
        1 for issue in issues if str(issue.get("status") or "").lower() == "possibly_inconsistent"
    )
    insufficient_information = sum(
        1 for issue in issues if str(issue.get("status") or "").lower() == "insufficient_information"
    )
    return {
        "issueCount": len(issues),
        "inconsistent": inconsistent,
        "possiblyInconsistent": possibly_inconsistent,
        "insufficientInformation": insufficient_information,
        "consistent": 1 if len(issues) == 0 else 0,
        "durationSeconds": round(duration_seconds, 2),
        "description": result.get("summary") or "",
    }


def _print_payload(title: str, payload: dict[str, Any]) -> None:
    print(f"\n===== {title} =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_filename_part(text: str, limit: int = 40) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text or ""))
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "unknown"
    return cleaned[:limit]


def _normalize_request_payload(raw: Any, input_path: Path, seq: int) -> dict[str, Any]:
    file_stem = input_path.stem
    default_report_id = f"LOCAL-CONS-{file_stem}-{seq:03d}"
    if isinstance(raw, list):
        return {
            "checkReportId": default_report_id,
            "fileId": default_report_id,
            "fileName": input_path.name,
            "docContent": raw,
        }
    if isinstance(raw, dict):
        if "docContent" in raw or "doc_content" in raw:
            payload = dict(raw)
            payload.setdefault("checkReportId", default_report_id)
            payload.setdefault("fileId", payload.get("checkReportId", default_report_id))
            payload.setdefault("fileName", input_path.name)
            return payload
        if "content" in raw and isinstance(raw["content"], list):
            return {
                "checkReportId": raw.get("checkReportId") or default_report_id,
                "fileId": raw.get("fileId") or raw.get("checkReportId") or default_report_id,
                "fileName": raw.get("fileName") or input_path.name,
                "docContent": raw["content"],
            }
    raise ValueError(
        "输入 JSON 必须是 docContent 数组，或包含 docContent/doc_content/content 的对象。"
    )


def _apply_overrides(
    payload: dict[str, Any],
    *,
    model_id: str,
    enable_llm: bool,
    max_candidates: int,
    max_judge_concurrency: int,
    llm_retry_attempts: int,
    check_types: Optional[List[str]],
) -> SingleDocConsistencyCheckRequest:
    normalized = dict(payload)
    normalized["modelId"] = model_id
    normalized["enableLlm"] = enable_llm
    normalized["maxCandidates"] = max_candidates
    normalized["maxJudgeConcurrency"] = max_judge_concurrency
    normalized["llmRetryAttempts"] = llm_retry_attempts
    if check_types is not None:
        normalized["checkTypes"] = check_types
    return SingleDocConsistencyCheckRequest.model_validate(normalized)


async def _run_one(
    *,
    input_path: Path,
    request: SingleDocConsistencyCheckRequest,
    output_dir: Path,
) -> dict[str, Any]:
    service = SingleDocConsistencyService(
        model_id=request.model_id,
        max_candidates=request.max_candidates,
        max_judge_concurrency=request.max_judge_concurrency,
        enable_llm=request.enable_llm,
        llm_retry_attempts=request.llm_retry_attempts,
    )
    started_at = datetime.now(UTC)
    started_payload = _build_callback_payload(
        check_report_id=request.check_report_id,
        file_id=request.file_id or request.check_report_id,
        event="started",
        data={"checkType": "single_doc_consistency", "fileStatus": "P"},
    )
    _print_payload(f"{input_path.name} started", started_payload)

    result = await service.analyze_request(request)
    result_payload = result.model_dump(mode="json", by_alias=True)
    duration_seconds = (datetime.now(UTC) - started_at).total_seconds()

    audited_payload = _build_callback_payload(
        check_report_id=request.check_report_id,
        file_id=result.file_id,
        event="audited",
        data={
            "fileId": result.file_id,
            "docFileName": result.file_name,
            "fileStatus": "F" if result.issue_count else "S",
            "results": _build_audited_results(result_payload),
            "status": "running",
            "checkType": "single_doc_consistency",
            "result": result_payload,
        },
    )
    completed_payload = _build_callback_payload(
        check_report_id=request.check_report_id,
        file_id=result.file_id,
        event="completed",
        data={
            **_build_completed_summary(result_payload, duration_seconds),
            "status": "completed",
            "checkType": "single_doc_consistency",
            "result": result_payload,
        },
    )
    _print_payload(f"{input_path.name} audited", audited_payload)
    _print_payload(f"{input_path.name} completed", completed_payload)

    output = {
        "meta": {
            "inputFile": str(input_path),
            "checkReportId": request.check_report_id,
            "fileId": result.file_id,
            "fileName": result.file_name,
            "modelId": request.model_id,
            "enableLlm": request.enable_llm,
            "maxCandidates": request.max_candidates,
            "maxJudgeConcurrency": request.max_judge_concurrency,
            "llmRetryAttempts": request.llm_retry_attempts,
            "checkTypes": request.check_types,
            "startedAt": _utc_now_iso(),
            "durationSeconds": round(duration_seconds, 2),
        },
        "result": result_payload,
        "callbacks": {
            "started": started_payload,
            "audited": audited_payload,
            "completed": completed_payload,
        },
    }

    readable_report = _sanitize_filename_part(request.check_report_id, limit=32)
    readable_file = _sanitize_filename_part(result.file_id, limit=32)
    digest_src = f"{request.check_report_id}|{result.file_id}|{result.file_name}"
    digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:12]
    out_name = f"{readable_report}_{readable_file}_{digest}.json"
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

    async def worker(idx: int, path: Path) -> None:
        raw = _load_json(path)
        payload = _normalize_request_payload(raw, path, idx + 1)
        request = _apply_overrides(
            payload,
            model_id=args.model_id,
            enable_llm=args.enable_llm,
            max_candidates=args.max_candidates,
            max_judge_concurrency=args.max_judge_concurrency,
            llm_retry_attempts=args.llm_retry_attempts,
            check_types=args.check_types,
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
        description="单文档一致性检查本地测试脚本（打印回调结构并落盘模型输出）"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="输入 JSON 文件路径，或包含多个 JSON 的目录路径；不传则使用文件头 DEFAULT_INPUT_PATH",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"模型 ID，默认: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--enable-llm",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_LLM,
        help=f"是否启用 LLM，默认: {DEFAULT_ENABLE_LLM}",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=f"候选上限，默认: {DEFAULT_MAX_CANDIDATES}",
    )
    parser.add_argument(
        "--max-judge-concurrency",
        type=int,
        default=DEFAULT_MAX_JUDGE_CONCURRENCY,
        help=f"LLM裁决并发，默认: {DEFAULT_MAX_JUDGE_CONCURRENCY}",
    )
    parser.add_argument(
        "--llm-retry-attempts",
        type=int,
        default=DEFAULT_LLM_RETRY_ATTEMPTS,
        help=f"LLM 重试次数，默认: {DEFAULT_LLM_RETRY_ATTEMPTS}",
    )
    parser.add_argument(
        "--check-types",
        nargs="*",
        default=DEFAULT_CHECK_TYPES,
        help="覆盖请求中的 checkTypes，例如: --check-types fact_value_consistency cross_reference_consistency",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"文件级并发数（批量输入时生效），默认: {DEFAULT_CONCURRENCY}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录，默认: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.input = args.input or DEFAULT_INPUT_PATH

    if args.concurrency < 1:
        raise ValueError("--concurrency 必须 >= 1")
    if args.max_candidates < 1:
        raise ValueError("--max-candidates 必须 >= 1")
    if args.max_judge_concurrency < 1:
        raise ValueError("--max-judge-concurrency 必须 >= 1")
    if args.llm_retry_attempts < 1:
        raise ValueError("--llm-retry-attempts 必须 >= 1")

    input_files = _discover_input_files(Path(args.input))
    if not input_files:
        raise ValueError("未发现可测试的 JSON 文件")

    print(
        json.dumps(
            {
                "inputPath": args.input,
                "inputCount": len(input_files),
                "modelId": args.model_id,
                "enableLlm": args.enable_llm,
                "maxCandidates": args.max_candidates,
                "maxJudgeConcurrency": args.max_judge_concurrency,
                "llmRetryAttempts": args.llm_retry_attempts,
                "checkTypesOverride": args.check_types,
                "concurrency": args.concurrency,
                "outputDir": args.output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    asyncio.run(_run_all(input_files=input_files, args=args))


if __name__ == "__main__":
    main()
