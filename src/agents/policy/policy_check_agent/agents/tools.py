"""
Policy Check Agent tool definitions.

These tools read parsed repository content, find candidate files, read images,
and return final requirement/control results.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextvars import ContextVar

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from src.agents.content_generate_agent.utils import count_tokens
from src.agents.policy.policy_check_agent.material_layer import (
    ControlRetrievalPlanner,
    MaterialEvidenceRetriever,
    MaterialUnderstandingService,
)
from src.agents.policy.policy_check_agent.services import MaterialRetriever, build_material_store
from src.agents.policy.policy_check_agent.schemas import ResourceFile, item_content_extract
from src.agents.policy.policy_check_agent.schemas import ControlRetrievalRequest
from src.agents.policy.policy_check_agent.utils import read_image_async
from loguru import logger

# Use ContextVar so each concurrent task keeps its own repository/runtime state.

_resource_repository_var: ContextVar[Optional[Any]] = ContextVar('resource_repository', default=None)
_resource_runtime_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar("resource_runtime", default=None)

IMAGE_BASE_DIR = "storage/mineru_output/images"
TERMINAL_ERROR_PREFIX = "TERMINAL_ERROR:"
MAX_FILE_CONTENT_TOKENS = 12000
MAX_FILE_OVERVIEW_ITEMS = 20
MAX_EXCERPT_ITEMS = 12
VALID_CHECK_RESULTS = {"pass", "fail", "uncertain", "skipped"}
MAX_FIND_FILES_CALLS_PER_REQUIREMENT = 6
MAX_READ_EXCERPT_CALLS_PER_REQUIREMENT = 18
MAX_RETRIEVE_MATERIAL_CALLS_PER_REQUIREMENT = 8
MAX_READ_EVIDENCE_BLOCKS_CALLS_PER_REQUIREMENT = 12
MAX_REPEAT_EXCERPT_QUERIES_PER_FILE = 1
MAX_EVIDENCE_QUOTED_TEXT_CHARS = 30


def _build_default_runtime_state() -> Dict[str, Any]:
    return {
        "missing_files": set(),
        "missing_images": set(),
        "control_results": {},
        "emitted_control_ids": set(),
        "file_read_counts": {},
        "formatted_file_cache": {},
        "excerpt_cache": {},
        "content_hint_cache": {},
        "file_match_cache": {},
        "tool_repeat_counts": {},
        "tool_call_counts": {},
        "material_retrieval_cache": {},
        "material_retrieval_metrics": {"calls": 0, "doc_hits": 0, "chunk_hits": 0, "empty_calls": 0},
        "event_queue": None,
    }

def set_resource_repository(repo) -> None:
    """Set the current resource repository in context."""
    _resource_repository_var.set(repo)


def get_resource_repository() -> Optional[Any]:
    """Get the current resource repository from context."""
    return _resource_repository_var.get()


class ResourceRepositoryContext:
    """Context manager for request-scoped repository and runtime state."""
    
    def __init__(self, repo, runtime_overrides: Optional[Dict[str, Any]] = None):
        self.repo = repo
        self.runtime_overrides = runtime_overrides or {}
        self.token = None
        self.runtime_token = None
    
    def __enter__(self):
        self.token = _resource_repository_var.set(self.repo)
        state = _build_default_runtime_state()
        state.update(self.runtime_overrides)
        self.runtime_token = _resource_runtime_var.set(state)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _resource_repository_var.reset(self.token)
        _resource_runtime_var.reset(self.runtime_token)
        return False


def _get_runtime_state() -> Dict[str, Any]:
    state = _resource_runtime_var.get()
    if state is None:
        state = _build_default_runtime_state()
        _resource_runtime_var.set(state)
    return state


def set_runtime_event_queue(queue: Any) -> None:
    state = _get_runtime_state()
    state["event_queue"] = queue


def set_runtime_metadata(**kwargs: Any) -> None:
    state = _get_runtime_state()
    state.update(kwargs)


def merge_runtime_material_metrics(
    *,
    calls: int = 0,
    doc_hits: int = 0,
    chunk_hits: int = 0,
    empty_calls: int = 0,
) -> None:
    state = _get_runtime_state()
    metrics = state.setdefault(
        "material_retrieval_metrics",
        {"calls": 0, "doc_hits": 0, "chunk_hits": 0, "empty_calls": 0},
    )
    metrics["calls"] = int(metrics.get("calls", 0) or 0) + int(calls or 0)
    metrics["doc_hits"] = int(metrics.get("doc_hits", 0) or 0) + int(doc_hits or 0)
    metrics["chunk_hits"] = int(metrics.get("chunk_hits", 0) or 0) + int(chunk_hits or 0)
    metrics["empty_calls"] = int(metrics.get("empty_calls", 0) or 0) + int(empty_calls or 0)


def bump_runtime_tool_call(tool_name: str, amount: int = 1) -> None:
    state = _get_runtime_state()
    call_counts = state.setdefault("tool_call_counts", {})
    call_counts[tool_name] = int(call_counts.get(tool_name, 0) or 0) + int(amount or 0)


def get_runtime_snapshot() -> Dict[str, Any]:
    state = _get_runtime_state()
    return {
        "control_results": list(state.get("control_results", {}).values()),
        "final_check_result": state.get("final_check_result"),
        "material_retrieval_metrics": dict(state.get("material_retrieval_metrics", {})),
        "tool_call_counts": dict(state.get("tool_call_counts", {})),
        "missing_files": sorted(str(item) for item in (state.get("missing_files") or set())),
        "missing_images": sorted(str(item) for item in (state.get("missing_images") or set())),
    }


def _get_material_understanding() -> Optional[Any]:
    state = _get_runtime_state()
    cached = state.get("material_understanding")
    if cached is not None:
        return cached
    repo = get_resource_repository()
    if repo is None:
        return None
    try:
        cached = MaterialUnderstandingService().build(repo)
        state["material_understanding"] = cached
        state["material_retriever"] = MaterialEvidenceRetriever(cached)
        return cached
    except Exception as exc:
        logger.warning("material understanding build failed: {}", exc)
        state["material_understanding"] = None
        return None


def _get_material_store() -> Optional[Any]:
    state = _get_runtime_state()
    cached = state.get("material_store")
    if cached is not None:
        return cached
    repo = get_resource_repository()
    if repo is None:
        return None
    try:
        cached = build_material_store(repo, case_id=repo.id)
        state["material_store"] = cached
        state["material_retriever_v2"] = MaterialRetriever(cached)
        return cached
    except Exception as exc:
        logger.warning("material store build failed: {}", exc)
        state["material_store"] = None
        return None


def _get_material_retriever_v2() -> Optional[MaterialRetriever]:
    state = _get_runtime_state()
    cached = state.get("material_retriever_v2")
    if cached is not None:
        return cached
    store = _get_material_store()
    if store is None:
        return None
    cached = MaterialRetriever(store)
    state["material_retriever_v2"] = cached
    return cached


def _get_material_retriever() -> Optional[MaterialEvidenceRetriever]:
    state = _get_runtime_state()
    cached = state.get("material_retriever")
    if cached is not None:
        return cached
    material_understanding = _get_material_understanding()
    if material_understanding is None:
        return None
    cached = MaterialEvidenceRetriever(material_understanding)
    state["material_retriever"] = cached
    return cached


def _get_control_retrieval_planner() -> ControlRetrievalPlanner:
    state = _get_runtime_state()
    cached = state.get("control_retrieval_planner")
    if cached is None:
        cached = ControlRetrievalPlanner()
        state["control_retrieval_planner"] = cached
    return cached


def _normalize_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", (value or "").lower())


def _normalize_file_stem(value: str) -> str:
    return _normalize_text(Path(value or "").stem)


def _extract_search_tokens(value: str) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+", value or "")
    return [token.lower() for token in tokens if token.strip()]


def _build_aliases(resource_file: ResourceFile) -> List[str]:
    aliases: List[str] = []
    for value in [
        resource_file.fileId,
        resource_file.fileName,
        getattr(resource_file, "originalFileName", None),
        *getattr(resource_file, "aliases", []),
    ]:
        if not value:
            continue
        if value not in aliases:
            aliases.append(value)
        stem = Path(value).stem if "." in value or "/" in value or "\\" in value else value
        if stem and stem not in aliases:
            aliases.append(stem)
    return aliases


def _build_metadata_tokens(resource_file: ResourceFile) -> List[str]:
    values: List[str] = []
    if getattr(resource_file, "docType", None):
        values.append(resource_file.docType)
    values.extend(getattr(resource_file, "businessTags", []) or [])
    tokens: List[str] = []
    for value in values:
        for token in _extract_search_tokens(value):
            if token not in tokens:
                tokens.append(token)
    return tokens


def _build_content_hint(resource_file: ResourceFile, limit: int = 12) -> str:
    runtime_state = _get_runtime_state()
    cached = runtime_state["content_hint_cache"].get(resource_file.fileId)
    if cached is not None:
        return cached

    fragments: List[str] = []
    for item in resource_file.content:
        text = item_content_extract(item)
        if not text:
            continue
        fragments.append(text[:300])
        if len(fragments) >= limit:
            break
    hint = "\n".join(fragments)
    runtime_state["content_hint_cache"][resource_file.fileId] = hint
    return hint


def _score_file_match(query: str, resource_file: ResourceFile) -> int:
    normalized_query = _normalize_text(query)
    query_tokens = set(_extract_search_tokens(query))
    aliases = _build_aliases(resource_file)
    normalized_aliases = [_normalize_text(alias) for alias in aliases if alias]
    if not normalized_query:
        return 0
    score = 0
    for alias, normalized_alias in zip(aliases, normalized_aliases):
        if not normalized_alias:
            continue
        if normalized_query == normalized_alias:
            if alias == resource_file.fileId:
                score = max(score, 5000)
            else:
                score = max(score, 3500)
        elif normalized_query in normalized_alias or normalized_alias in normalized_query:
            score = max(score, 2200)

    alias_tokens = set()
    for alias in aliases:
        alias_tokens.update(_extract_search_tokens(alias))
    overlap = query_tokens & alias_tokens
    if overlap:
        score = max(score, 800 + len(overlap) * 180)

    metadata_tokens = set(_build_metadata_tokens(resource_file))
    metadata_overlap = query_tokens & metadata_tokens
    if metadata_overlap:
        score = max(score, 700 + len(metadata_overlap) * 160)

    content_hint = _build_content_hint(resource_file)
    if content_hint:
        content_tokens = set(_extract_search_tokens(content_hint))
        content_overlap = query_tokens & content_tokens
        if content_overlap:
            score = max(score, 350 + len(content_overlap) * 60)
        normalized_hint = _normalize_text(content_hint[:2000])
        if normalized_query and normalized_query in normalized_hint:
            score = max(score, 900)

    return score


def _find_candidate_files(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    repo = get_resource_repository()
    if repo is None:
        return []
    runtime_state = _get_runtime_state()
    cache_key = f"{query}::{limit}"
    cached = runtime_state["file_match_cache"].get(cache_key)
    if cached is not None:
        return cached

    retriever = _get_material_retriever()
    material_candidates: List[Dict[str, Any]] = []
    retriever_v2 = _get_material_retriever_v2()
    if retriever_v2 is not None:
        request = ControlRetrievalRequest(control_text=query, top_k_documents=max(limit * 3, limit))
        for hit in retriever_v2.retrieve_for_control(request).candidate_documents:
            material_candidates.append(
                {
                    "fileName": hit.manifest.file_name,
                    "fileId": hit.manifest.file_id,
                    "docType": hit.manifest.doc_type,
                    "businessTags": hit.manifest.candidate_tags,
                    "score": int(hit.score * 400) + 1500,
                    "matchReasons": hit.reasons,
                }
            )
    if retriever is not None:
        for item in retriever.query_documents(query, limit=max(limit * 3, limit)):
            material_candidates.append(
                {
                    "fileName": item["fileName"],
                    "fileId": item["fileId"],
                    "docType": item.get("docClass"),
                    "businessTags": [],
                    "score": int(item["score"]) + 1200,
                    "matchReasons": item.get("reasons", []),
                }
            )

    scored: List[Dict[str, Any]] = []
    for resource_file in repo.files:
        score = _score_file_match(query, resource_file)
        if score <= 0:
            continue
        scored.append(
            {
                "fileName": resource_file.fileName,
                "fileId": resource_file.fileId,
                "originalFileName": getattr(resource_file, "originalFileName", None),
                "docType": getattr(resource_file, "docType", None),
                "businessTags": getattr(resource_file, "businessTags", []) or [],
                "score": score,
            }
        )
    merged: Dict[str, Dict[str, Any]] = {}
    for item in material_candidates + scored:
        file_id = item["fileId"]
        if file_id not in merged or int(item["score"]) > int(merged[file_id]["score"]):
            merged[file_id] = item
        elif "matchReasons" in item and "matchReasons" not in merged[file_id]:
            merged[file_id]["matchReasons"] = item["matchReasons"]
    ranked = list(merged.values())
    ranked.sort(key=lambda item: (-int(item["score"]), item["fileName"]))
    result = ranked[:limit]
    runtime_state["file_match_cache"][cache_key] = result
    return result


def _resolve_target_file(file_query: str) -> Optional[ResourceFile]:
    repo = get_resource_repository()
    if repo is None:
        return None
    normalized_query = _normalize_text(file_query)

    for resource_file in repo.files:
        if resource_file.fileId == file_query:
            return resource_file
        aliases = _build_aliases(resource_file)
        if file_query in aliases:
            return resource_file
        if normalized_query and normalized_query in [_normalize_text(alias) for alias in aliases]:
            return resource_file

    candidates = _find_candidate_files(file_query, limit=1)
    if not candidates:
        return None

    best_candidate = candidates[0]
    top_two = _find_candidate_files(file_query, limit=2)
    if best_candidate["score"] < 350:
        return None
    if len(top_two) > 1 and best_candidate["score"] < 1200:
        score_gap = best_candidate["score"] - top_two[1]["score"]
        if score_gap < 120:
            logger.info("Ambiguous file match for query={}, top_two={}", file_query, top_two)
            return None

    if best_candidate["score"] < 200:
        return None

    for resource_file in repo.files:
        if resource_file.fileId == best_candidate["fileId"]:
            return resource_file
    return None


def _build_terminal_error(message: str) -> str:
    return f"{TERMINAL_ERROR_PREFIX} {message}"


def _track_tool_repeat(tool_name: str, key: str) -> int:
    state = _get_runtime_state()
    repeat_counts = state.setdefault("tool_repeat_counts", {})
    counter_key = f"{tool_name}::{key}"
    repeat_counts[counter_key] = repeat_counts.get(counter_key, 0) + 1
    return repeat_counts[counter_key]


def _track_tool_call(tool_name: str) -> int:
    state = _get_runtime_state()
    call_counts = state.setdefault("tool_call_counts", {})
    call_counts[tool_name] = call_counts.get(tool_name, 0) + 1
    return call_counts[tool_name]


def _tool_budget_message(tool_name: str, recommended_tool: str) -> str:
    return (
        f"{tool_name} has already been called too many times for the current requirement. "
        f"Stop exploring and reuse earlier results. Prefer `{recommended_tool}` or conclude with fail/uncertain."
    )


def _normalize_query_key(value: str) -> str:
    tokens = sorted(set(_extract_search_tokens(value or "")))
    return " ".join(tokens)


def _emit_runtime_event(payload: Dict[str, Any]) -> None:
    queue = _get_runtime_state().get("event_queue")
    if queue is not None:
        queue.put_nowait(payload)


def _result_rank(result: str) -> int:
    order = {
        "fail": 4,
        "uncertain": 3,
        "pass": 2,
        "skipped": 1,
    }
    return order.get(result or "", 0)


def _normalize_check_result(result: Any, default: str = "uncertain") -> str:
    normalized = str(result or "").strip().lower()
    return normalized if normalized in VALID_CHECK_RESULTS else default


def _extract_evidence_refs(reason: str) -> set[str]:
    return set(re.findall(r"\[(evidence_[A-Za-z0-9_-]+)\]", reason or ""))


def _parse_element_ids_from_text(value: str) -> List[int]:
    ids: List[int] = []
    patterns = [
        r"ElementIds?\s*[:=：]\s*\[([^\]]+)\]",
        r"elementIds?\s*[:=：]\s*\[([^\]]+)\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, value or "")
        if not match:
            continue
        for part in re.split(r"[,\s]+", match.group(1).strip()):
            if not part:
                continue
            try:
                ids.append(int(part))
            except (TypeError, ValueError):
                continue
        if ids:
            return ids
    return ids


def _truncate_quoted_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= MAX_EVIDENCE_QUOTED_TEXT_CHARS:
        return text

    # Prefer a natural short phrase before hard truncation so the frontend sees
    # a focused snippet instead of a long chopped sentence.
    separators = ["。", "；", "，", ",", ":", "：", "\n"]
    best_candidate: Optional[str] = None
    for separator in separators:
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        for part in parts:
            if len(part) <= MAX_EVIDENCE_QUOTED_TEXT_CHARS:
                if best_candidate is None or len(part) > len(best_candidate):
                    best_candidate = part
        if best_candidate:
            return best_candidate

    return f"{text[:MAX_EVIDENCE_QUOTED_TEXT_CHARS].rstrip()}..."


def _get_evidence_block_lookup() -> Dict[str, Any]:
    state = _get_runtime_state()
    lookup = state.get("evidence_block_lookup")
    if lookup is not None:
        return lookup
    store = _get_material_store()
    if store is None:
        lookup = {}
    else:
        lookup = {block.evidence_id: block for block in store.evidence_blocks}
    state["evidence_block_lookup"] = lookup
    return lookup


def _resolve_evidence_quoted_text(
    *,
    evidence_id: Optional[str],
    file_id: str,
    element_ids: List[int],
    fallback_text: Optional[str] = None,
) -> Optional[str]:
    if evidence_id:
        block = _get_evidence_block_lookup().get(evidence_id)
        if block is not None:
            quoted = _truncate_quoted_text(getattr(block, "text", None))
            if quoted:
                return quoted

    target_file = _resolve_target_file(file_id)
    if target_file is not None and element_ids:
        element_id_set = {int(item) for item in element_ids}
        matched_segments: List[str] = []
        for item in target_file.content:
            item_element_id = getattr(item, "element_id", None)
            if item_element_id not in element_id_set:
                continue
            text = item_content_extract(item)
            if text:
                matched_segments.append(text)
        if matched_segments:
            return _truncate_quoted_text("\n".join(matched_segments))

    return _truncate_quoted_text(fallback_text)


def _infer_evidence_from_reason(reason: str) -> List[Dict[str, Any]]:
    if not reason:
        return []

    file_id = None
    file_id_patterns = [
        r"(?:证据文件ID(?:为)?|文件ID(?:为)?|fileId)\s*[:=：]?\s*([A-Za-z0-9]+)",
    ]
    for pattern in file_id_patterns:
        match = re.search(pattern, reason)
        if match:
            file_id = match.group(1)
            break

    if not file_id:
        file_name_match = re.search(r"[《\"]([^《》\"]+\.[A-Za-z0-9]+)[》\"]", reason)
        if file_name_match:
            target_file = _resolve_target_file(file_name_match.group(1))
            if target_file is not None:
                file_id = target_file.fileId

    if not file_id:
        repo = get_resource_repository()
        if repo is not None:
            normalized_reason = _normalize_text(reason)
            matched_file_ids: List[str] = []
            for resource_file in repo.files:
                aliases = _build_aliases(resource_file)
                if any(_normalize_text(alias) and _normalize_text(alias) in normalized_reason for alias in aliases):
                    matched_file_ids.append(resource_file.fileId)
            deduped_file_ids = list(dict.fromkeys(matched_file_ids))
            if len(deduped_file_ids) == 1:
                file_id = deduped_file_ids[0]

    if not file_id:
        repo = get_resource_repository()
        if repo is not None and len(repo.files) == 1:
            file_id = repo.files[0].fileId

    if not file_id:
        return []

    return [
        {
            "fileId": file_id,
            "elementIds": _parse_element_ids_from_text(reason),
            "summary": reason[:200],
            "quotedText": _truncate_quoted_text(reason[:200]),
        }
    ]


def _build_single_file_fallback_evidence(reason: str) -> List[Dict[str, Any]]:
    repo = get_resource_repository()
    if repo is None or len(repo.files) != 1:
        return []
    only_file = repo.files[0]
    return [
        {
            "fileId": only_file.fileId,
            "elementIds": [],
            "summary": reason[:200] or f"Matched file {only_file.fileName}",
            "quotedText": _truncate_quoted_text(reason[:200] or only_file.fileName),
        }
    ]


def _sanitize_evidence_items(items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        file_id = str(item.get("fileId") or "").strip()
        if not file_id:
            continue
        raw_element_ids = item.get("elementIds", []) or []
        element_ids: List[int] = []
        for element_id in raw_element_ids:
            try:
                element_ids.append(int(element_id))
            except (TypeError, ValueError):
                continue
        sanitized.append(
            {
                "evidenceId": str(item.get("evidenceId") or "").strip() or None,
                "fileId": file_id,
                "elementIds": element_ids,
                "summary": item.get("summary"),
                "quotedText": _resolve_evidence_quoted_text(
                    evidence_id=str(item.get("evidenceId") or "").strip() or None,
                    file_id=file_id,
                    element_ids=element_ids,
                    fallback_text=item.get("quotedText") or item.get("summary"),
                ),
            }
        )
    return _merge_evidence_items([], sanitized)


def _coerce_evidence_input(evidence: Any) -> List[Dict[str, Any]]:
    if evidence is None:
        return []
    if isinstance(evidence, list):
        return [item for item in evidence if isinstance(item, dict)]
    if isinstance(evidence, tuple):
        return [item for item in evidence if isinstance(item, dict)]
    if isinstance(evidence, dict):
        return [evidence]
    if isinstance(evidence, str):
        text = evidence.strip()
        if not text or text.lower() in {"none", "null", "[]"}:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
        return []
    return []


def _finalize_check_payload(
    payload_type: str,
    result: Any,
    reason: Any,
    evidence: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    normalized_result = _normalize_check_result(result)
    normalized_reason = str(reason or "").strip()
    normalized_evidence = _sanitize_evidence_items(evidence)
    if not normalized_evidence and normalized_reason:
        normalized_evidence = _sanitize_evidence_items(_infer_evidence_from_reason(normalized_reason))
    if normalized_result == "pass" and not normalized_evidence:
        normalized_evidence = _sanitize_evidence_items(_build_single_file_fallback_evidence(normalized_reason))
    evidence_ids = {item["evidenceId"] for item in normalized_evidence if item.get("evidenceId")}
    cited_evidence_ids = _extract_evidence_refs(normalized_reason)

    notes: List[str] = []
    if not normalized_reason:
        notes.append("reason_empty_normalized")
    if normalized_result == "pass" and not normalized_evidence:
        notes.append("pass_without_evidence")
    missing_refs = sorted(cited_evidence_ids - evidence_ids)
    if missing_refs:
        notes.append(f"missing_evidence_refs:{','.join(missing_refs)}")
        if normalized_result == "pass":
            normalized_result = "uncertain"

    if not normalized_reason:
        normalized_reason = "未提供有效理由。"

    logger.info(
        "normalized {} payload: result={}, evidence_count={}, notes={}",
        payload_type,
        normalized_result,
        len(normalized_evidence),
        notes,
    )
    return {
        "result": normalized_result,
        "reason": normalized_reason,
        "evidence": normalized_evidence,
    }


def _merge_evidence_items(
    existing: List[Dict[str, Any]],
    incoming: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for item in (existing or []) + (incoming or []):
        normalized = {
            "evidenceId": item.get("evidenceId"),
            "fileId": item.get("fileId"),
            "elementIds": item.get("elementIds", []) or [],
            "summary": item.get("summary"),
            "quotedText": item.get("quotedText"),
        }
        key = (
            normalized.get("fileId"),
            tuple(normalized.get("elementIds", [])),
            normalized.get("summary"),
            normalized.get("quotedText"),
            normalized.get("evidenceId"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    for index, item in enumerate(merged, start=1):
        if not item.get("evidenceId"):
            item["evidenceId"] = f"evidence_{index}"
    return merged


def merge_control_result(
    existing: Optional[Dict[str, Any]],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    if not existing:
        return {
            "requirementId": incoming.get("requirementId"),
            "controlId": incoming.get("controlId"),
            "result": incoming.get("result", "uncertain"),
            "reason": incoming.get("reason", ""),
            "evidence": _merge_evidence_items([], incoming.get("evidence", [])),
        }

    existing_reason = (existing.get("reason") or "").strip()
    incoming_reason = (incoming.get("reason") or "").strip()
    reason_parts: List[str] = []
    for part in [existing_reason, incoming_reason]:
        if part and part not in reason_parts:
            reason_parts.append(part)

    existing_result = existing.get("result", "uncertain")
    incoming_result = incoming.get("result", "uncertain")
    result = incoming_result if _result_rank(incoming_result) >= _result_rank(existing_result) else existing_result

    return {
        "requirementId": incoming.get("requirementId") or existing.get("requirementId"),
        "controlId": incoming.get("controlId") or existing.get("controlId"),
        "result": result,
        "reason": " ".join(reason_parts),
        "evidence": _merge_evidence_items(existing.get("evidence", []), incoming.get("evidence", [])),
    }


def _collect_file_segments(resource_file: ResourceFile) -> List[Dict[str, Any]]:
    retriever = _get_material_retriever()
    if retriever is not None:
        document = getattr(retriever, "docs_by_id", {}).get(resource_file.fileId)
        if document is not None:
            segments: List[Dict[str, Any]] = []
            for page in document.pages:
                for chunk in page.chunks:
                    segments.append({"element_id": chunk.elementId or 0, "text": chunk.text, "page_no": chunk.pageNo})
            if segments:
                return segments
    segments: List[Dict[str, Any]] = []
    for item in resource_file.content:
        text = item_content_extract(item)
        if not text:
            continue
        segments.append({"element_id": item.element_id, "text": text})
    return segments


def _format_segments(resource_file: ResourceFile, segments: List[Dict[str, Any]], header_lines: List[str]) -> str:
    if not segments:
        return "\n".join(header_lines + [f"File {resource_file.fileName} has no readable parsed content."])

    parts = list(header_lines)
    for segment in segments:
        parts.append(f"[ElementId: {segment['element_id']}]\n{segment['text']}\n")
    return "\n".join(parts)


def _format_evidence_blocks(file_name: str, file_id: str, blocks: List[Dict[str, Any]], header_lines: List[str]) -> str:
    if not blocks:
        return "\n".join(header_lines + [f"File {file_name} has no readable evidence blocks."])

    parts = list(header_lines)
    for block in blocks:
        page = block.get("page")
        page_text = f", Page: {page}" if page is not None else ""
        element_ids = block.get("elementIds") or []
        block_type = block.get("blockType") or "unknown"
        evidence_id = block.get("evidenceId") or ""
        parts.append(
            f"[EvidenceId: {evidence_id}, ElementIds: {element_ids}, BlockType: {block_type}{page_text}]\n{block.get('text', '')}\n"
        )
    return "\n".join(parts)


def _build_file_overview(resource_file: ResourceFile, segments: List[Dict[str, Any]]) -> str:
    sample_segments = segments[:MAX_FILE_OVERVIEW_ITEMS]
    header_lines = [
        f"File overview for {resource_file.fileName}. Showing {len(sample_segments)}/{len(segments)} segments. Use read_file_excerpt_tool for targeted retrieval.",
        f"=== 文件(fileName): {resource_file.fileName}, 文件ID(fileId): {resource_file.fileId} ===",
    ]
    return _format_segments(resource_file, sample_segments, header_lines)


def _rank_segments_for_query(segments: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    query_tokens = set(_extract_search_tokens(query))
    normalized_query = _normalize_text(query)
    ranked: List[Dict[str, Any]] = []
    for segment in segments:
        normalized_text = _normalize_text(segment["text"])
        score = 0
        if normalized_query and normalized_query in normalized_text:
            score += 1000
        overlap = query_tokens & set(_extract_search_tokens(segment["text"]))
        score += len(overlap) * 100
        score += min(len(set(normalized_query) & set(normalized_text)), 20)
        if score > 0:
            ranked.append({**segment, "score": score})
    ranked.sort(key=lambda item: (-item["score"], item["element_id"]))
    return ranked


def _format_file_content(resource_file: ResourceFile) -> str:
    runtime_state = _get_runtime_state()
    cached = runtime_state["formatted_file_cache"].get(resource_file.fileId)
    if cached is not None:
        return cached

    if not resource_file.content:
        cached = f"文件 {resource_file.fileName} 内容为空"
        runtime_state["formatted_file_cache"][resource_file.fileId] = cached
        return cached

    segments = _collect_file_segments(resource_file)
    header_lines = [f"=== 文件(fileName): {resource_file.fileName}, 文件ID(fileId): {resource_file.fileId} ==="]
    full_content = _format_segments(resource_file, segments, header_lines)
    if count_tokens(full_content) <= MAX_FILE_CONTENT_TOKENS:
        runtime_state["formatted_file_cache"][resource_file.fileId] = full_content
        return full_content

    overview = _build_file_overview(resource_file, segments)
    logger.warning("文件 {} 内容过长，已返回压缩概览，原始token数: {}", resource_file.fileName, count_tokens(full_content))
    runtime_state["formatted_file_cache"][resource_file.fileId] = overview
    return overview

@tool
def read_file_tool(file_name: str):
    """
    Read parsed file content by file name or file id.
    """
    logger.debug("read_file_tool: {}", file_name)
    repo = get_resource_repository()
    if repo is None:
        return "Error: resource_repository is not initialized"

    runtime_state = _get_runtime_state()
    if file_name in runtime_state["missing_files"]:
        return _build_terminal_error(
            f"File `{file_name}` is confirmed missing or unmatched. Do not retry it. Return fail or uncertain based on missing file."
        )

    target_file = _resolve_target_file(file_name)
    if target_file is None:
        runtime_state["missing_files"].add(file_name)
        candidates = _find_candidate_files(file_name, limit=5)
        return _build_terminal_error(
            f"File `{file_name}` is missing or cannot be matched exactly. Candidates: {candidates}"
        )

    read_counts = runtime_state["file_read_counts"]
    read_counts[target_file.fileId] = read_counts.get(target_file.fileId, 0) + 1
    if read_counts[target_file.fileId] > 1:
        return (
            f"File {target_file.fileName} has already been read {read_counts[target_file.fileId] - 1} time(s). "
            "Use read_file_excerpt_tool for targeted clauses instead of rereading the whole file."
        )

    return _format_file_content(target_file)


@tool
def read_file_excerpt_tool(file_name: str, query: str, max_items: int = MAX_EXCERPT_ITEMS):
    """
    Read only the most relevant parsed segments for a query.
    """
    repo = get_resource_repository()
    if repo is None:
        return "Error: resource_repository is not initialized"

    call_count = _track_tool_call("read_file_excerpt_tool")
    if call_count > MAX_READ_EXCERPT_CALLS_PER_REQUIREMENT:
        return _build_terminal_error(
            _tool_budget_message("read_file_excerpt_tool", "read_evidence_blocks_tool")
        )

    target_file = _resolve_target_file(file_name)
    if target_file is None:
        candidates = _find_candidate_files(file_name, limit=5)
        return _build_terminal_error(
            f"File `{file_name}` is missing or cannot be matched exactly. Candidates: {candidates}"
        )

    runtime_state = _get_runtime_state()
    prefetched_evidence_by_file = runtime_state.get("prefetched_evidence_by_file", {}) or {}
    prefetched_ids_for_file = prefetched_evidence_by_file.get(target_file.fileId, []) or []
    if len(prefetched_ids_for_file) >= 2:
        return _build_terminal_error(
            f"Direct evidence blocks are already available for file `{target_file.fileName}`. "
            f"Use read_evidence_blocks_tool with evidence ids {prefetched_ids_for_file[:6]} instead of reading more excerpts."
        )
    normalized_query_key = _normalize_query_key(query)
    semantic_repeat_key = f"{target_file.fileId}::{normalized_query_key}"
    semantic_repeat_count = _track_tool_repeat("read_file_excerpt_tool.semantic", semantic_repeat_key)
    if semantic_repeat_count > MAX_REPEAT_EXCERPT_QUERIES_PER_FILE:
        return _build_terminal_error(
            f"Similar excerpt queries for file `{target_file.fileName}` were already executed. "
            "Reuse the retrieved evidence blocks or conclude with fail/uncertain instead of rereading similar excerpts."
        )

    cache_key = f"{target_file.fileId}::{normalized_query_key}::{max_items}"
    cached = runtime_state["excerpt_cache"].get(cache_key)
    if cached is not None:
        repeat_count = _track_tool_repeat("read_file_excerpt_tool.cached", cache_key)
        if repeat_count > 1:
            return _build_terminal_error(
                f"The excerpt for file `{target_file.fileName}` and query `{query}` was already returned. "
                "Use the existing excerpt or `read_evidence_blocks_tool` to conclude."
            )
        return cached

    logger.debug("read_file_excerpt_tool: file_name={}, query={}, max_items={}", file_name, query, max_items)

    max_items = max(1, min(max_items, 30))
    retriever = _get_material_retriever()
    ranked_segments: List[Dict[str, Any]]
    if retriever is not None:
        ranked_segments = [
            {
                "element_id": item.get("elementId") or 0,
                "text": item["text"],
                "score": item["score"],
            }
            for item in retriever.query_chunks(target_file.fileId, query, limit=max_items)
        ]
        if not ranked_segments:
            segments = _collect_file_segments(target_file)
            ranked_segments = _rank_segments_for_query(segments, query)
    else:
        segments = _collect_file_segments(target_file)
        ranked_segments = _rank_segments_for_query(segments, query)
    if not ranked_segments:
        message = (
            f"No obvious segments matched query `{query}` in file {target_file.fileName}. "
            "If this file is mandatory evidence, return fail or uncertain."
        )
        runtime_state["excerpt_cache"][cache_key] = message
        return message

    selected_segments = ranked_segments[:max_items]
    content = _format_segments(
        target_file,
        selected_segments,
        [
            f"Relevant segments for query `{query}` in file {target_file.fileName}, count={len(selected_segments)}",
            f"=== 文件(fileName): {target_file.fileName}, 文件ID(fileId): {target_file.fileId} ===",
        ],
    )
    runtime_state["excerpt_cache"][cache_key] = content
    return content


@tool
def read_evidence_blocks_tool(evidence_ids: List[str], max_blocks: int = 8):
    """
    Read evidence blocks returned by retrieve_material_evidence_tool directly by evidence id.
    """
    repo = get_resource_repository()
    if repo is None:
        return "Error: resource_repository is not initialized"

    call_count = _track_tool_call("read_evidence_blocks_tool")
    if call_count > MAX_READ_EVIDENCE_BLOCKS_CALLS_PER_REQUIREMENT:
        return _build_terminal_error(
            _tool_budget_message("read_evidence_blocks_tool", "complete_check")
        )

    if not evidence_ids:
        return _build_terminal_error("No evidence ids were provided.")

    store = _get_material_store()
    if store is None:
        return _build_terminal_error("Material store is not available.")

    blocks_by_id = {block.evidence_id: block for block in store.evidence_blocks}
    selected_blocks: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    for evidence_id in evidence_ids[: max(1, min(max_blocks, 20))]:
        block = blocks_by_id.get(evidence_id)
        if block is None:
            missing_ids.append(evidence_id)
            continue
        selected_blocks.append(
            {
                "evidenceId": block.evidence_id,
                "fileId": block.file_id,
                "fileName": block.file_name,
                "page": block.page,
                "elementIds": block.source_element_ids,
                "blockType": block.block_type,
                "text": block.text,
            }
        )

    if not selected_blocks:
        return _build_terminal_error(
            f"None of the requested evidence ids could be resolved: {missing_ids}"
        )

    primary = selected_blocks[0]
    header_lines = [
        f"Direct evidence blocks for file {primary['fileName']} (fileId={primary['fileId']}), count={len(selected_blocks)}",
        f"Missing evidence ids: {missing_ids}" if missing_ids else "All requested evidence ids were resolved.",
    ]
    return _format_evidence_blocks(
        file_name=primary["fileName"],
        file_id=primary["fileId"],
        blocks=selected_blocks,
        header_lines=header_lines,
    )

@tool
def find_files_tool(query: str, limit: int = 5, doc_type: str = "", tags: Optional[List[str]] = None):
    """
    Find candidate files by natural-language query or file name fragment.
    """
    repo = get_resource_repository()
    if repo is None:
        return [{"error": "resource_repository is not initialized"}]

    call_count = _track_tool_call("find_files_tool")
    if call_count > MAX_FIND_FILES_CALLS_PER_REQUIREMENT:
        return [{"message": _tool_budget_message("find_files_tool", "retrieve_material_evidence_tool")}]

    limit = max(1, min(limit, 20))
    composite_query = " ".join([query, doc_type, " ".join(tags or [])]).strip()
    repeat_key = f"{composite_query or query}::{limit}"
    repeat_count = _track_tool_repeat("find_files_tool", repeat_key)
    if repeat_count > 2:
        return [{"message": f"Query `{query}` was already searched repeatedly. Reuse the earlier candidate files instead of searching again."}]
    logger.debug("find_files_tool: query={}, limit={}, doc_type={}, tags={}", query, limit, doc_type, tags)
    candidates = _find_candidate_files(composite_query or query, limit=max(limit * 3, limit))
    if doc_type:
        candidates = [
            item for item in candidates
            if (item.get("docType") or "").lower() == doc_type.lower()
        ] or candidates
    if tags:
        tag_set = {tag.lower() for tag in (tags or []) if tag}
        filtered = []
        for item in candidates:
            item_tags = {tag.lower() for tag in (item.get("businessTags") or [])}
            if item_tags & tag_set:
                filtered.append(item)
        candidates = filtered or candidates
    candidates = candidates[:limit]
    return candidates if candidates else [{"message": f"No candidate files matched `{query}`"}]


@tool
def retrieve_material_evidence_tool(control_text: str, limit_docs: int = 5, limit_chunks: int = 8):
    """
    Build a control-aware retrieval plan and return candidate documents/chunks.
    """
    repo = get_resource_repository()
    if repo is None:
        return {"error": "resource_repository is not initialized"}

    call_count = _track_tool_call("retrieve_material_evidence_tool")
    if call_count > MAX_RETRIEVE_MATERIAL_CALLS_PER_REQUIREMENT:
        return {
            "message": _tool_budget_message("retrieve_material_evidence_tool", "read_file_excerpt_tool"),
            "docCandidates": [],
            "chunkCandidates": [],
        }

    retriever_v2 = _get_material_retriever_v2()
    if retriever_v2 is None:
        return {"error": "material retriever is not available"}

    runtime_state = _get_runtime_state()
    cache_key = f"{control_text}::{limit_docs}::{limit_chunks}"
    cached = runtime_state["material_retrieval_cache"].get(cache_key)
    if cached is not None:
        logger.info(
            "retrieve_material_evidence_tool cache_hit requirement_id={} docs={} chunks={}",
            runtime_state.get("requirement_id"),
            len(cached.get("docCandidates", []) or []),
            len(cached.get("chunkCandidates", []) or []),
        )
        return cached

    retrieval_result = retriever_v2.retrieve_for_control(
        ControlRetrievalRequest(
            control_text=control_text,
            top_k_documents=max(1, min(limit_docs, 10)),
            top_k_blocks=max(1, min(limit_chunks, 12)),
            top_k_facts=max(1, min(limit_chunks, 12)),
        )
    )
    store = _get_material_store()
    evidence_block_by_id = {}
    if store is not None:
        evidence_block_by_id = {block.evidence_id: block for block in store.evidence_blocks}
    chunk_candidates: Dict[str, List[Dict[str, Any]]] = {}
    evidence_candidates: List[Dict[str, Any]] = []
    seen_evidence_keys = set()
    for hit in retrieval_result.evidence_hits:
        candidate = {
            "evidenceId": hit.block.evidence_id,
            "page": hit.block.page,
            "elementIds": hit.block.source_element_ids,
            "text": hit.block.text,
            "blockType": hit.block.block_type,
            "score": hit.score,
            "reasons": hit.reasons,
        }
        chunk_candidates.setdefault(hit.block.file_id, []).append(candidate)
        evidence_key = (
            hit.block.evidence_id,
            hit.block.file_id,
            tuple(hit.block.source_element_ids or []),
        )
        if evidence_key not in seen_evidence_keys:
            seen_evidence_keys.add(evidence_key)
            evidence_candidates.append(
                {
                    "evidenceId": hit.block.evidence_id,
                    "fileId": hit.block.file_id,
                    "fileName": hit.block.file_name,
                    "elementIds": hit.block.source_element_ids,
                    "page": hit.block.page,
                    "summary": hit.block.text[:200],
                    "quotedText": _truncate_quoted_text(hit.block.text),
                    "source": "chunk",
                    "score": hit.score,
                }
            )

    for hit in retrieval_result.fact_hits:
        fact = hit.fact
        fact_block = evidence_block_by_id.get(fact.evidence_id)
        evidence_key = (fact.evidence_id, fact.doc_id, ())
        if evidence_key in seen_evidence_keys:
            continue
        seen_evidence_keys.add(evidence_key)
        evidence_candidates.append(
            {
                "evidenceId": fact.evidence_id,
                "fileId": getattr(fact_block, "file_id", "") or "",
                "fileName": getattr(fact_block, "file_name", "") or "",
                "elementIds": getattr(fact_block, "source_element_ids", []) or [],
                "page": getattr(fact_block, "page", None) if fact_block is not None else fact.page,
                "summary": f"{fact.field_name}={fact.field_value}",
                "quotedText": _truncate_quoted_text(getattr(fact_block, "text", None) or f"{fact.field_name}={fact.field_value}"),
                "source": "fact",
                "score": hit.score,
            }
        )

    payload = {
        "docCandidates": [
            {
                "fileId": hit.manifest.file_id,
                "fileName": hit.manifest.file_name,
                "docType": hit.manifest.doc_type,
                "candidateTags": hit.manifest.candidate_tags,
                "score": hit.score,
                "reasons": hit.reasons,
            }
            for hit in retrieval_result.candidate_documents
        ],
        "chunkCandidates": chunk_candidates,
        "evidenceCandidates": sorted(
            evidence_candidates,
            key=lambda item: (-float(item.get("score") or 0), str(item.get("evidenceId") or "")),
        )[: max(1, min(limit_chunks, 12))],
        "factHits": [
            {
                "factId": hit.fact.fact_id,
                "fieldName": hit.fact.field_name,
                "fieldValue": hit.fact.field_value,
                "normalizedValue": hit.fact.normalized_value,
                "evidenceId": hit.fact.evidence_id,
                "score": hit.score,
                "reasons": hit.reasons,
            }
            for hit in retrieval_result.fact_hits
        ],
        "debug": retrieval_result.debug,
    }
    runtime_state["material_retrieval_cache"][cache_key] = payload
    metrics = runtime_state.setdefault("material_retrieval_metrics", {})
    doc_hits = len(payload.get("docCandidates", []) or [])
    chunk_hits = sum(len(items) for items in (payload.get("chunkCandidates", {}) or {}).values())
    metrics["calls"] = metrics.get("calls", 0) + 1
    metrics["doc_hits"] = metrics.get("doc_hits", 0) + doc_hits
    metrics["chunk_hits"] = metrics.get("chunk_hits", 0) + chunk_hits
    if doc_hits == 0 and chunk_hits == 0:
        metrics["empty_calls"] = metrics.get("empty_calls", 0) + 1
    logger.info(
        "retrieve_material_evidence_tool requirement_id={} docs={} chunks={} plan_doc_classes={} plan_fields={} queries={}",
        runtime_state.get("requirement_id"),
        doc_hits,
        chunk_hits,
        payload.get("debug", {}).get("preferred_doc_types", []),
        payload.get("debug", {}).get("fact_fields", []),
        payload.get("debug", {}).get("keywords", []),
    )
    return payload


@tool
def list_files_tool(
    pattern: str = "*",
):
    """
    List parsed files in the current resource repository.

    Args:
        pattern: Optional glob pattern, default `*`.

    Returns:
        Matching file names.
    """
    repo = get_resource_repository()
    if repo is None:
        return ["Error: resource_repository is not initialized"]
    logger.debug("list_files_tool: {}", pattern)
    
    file_names = [f.fileName for f in repo.files]
    if pattern and pattern != "*":
        import fnmatch
        file_names = [f for f in file_names if fnmatch.fnmatch(f, pattern)]
    
    return file_names if file_names else ["No matching files found"]


@tool
async def read_image_tool(
    file_name: str,
):
    """
    Read parsed image artifacts for multimodal checking.

    Args:
        file_name: File name or file id.

    Returns:
        Multimodal payloads containing text + base64 image content.
    """
    logger.debug("read_image_tool: {}", file_name)
    repo = get_resource_repository()
    if repo is None:
        return "Error: resource_repository is not initialized"

    runtime_state = _get_runtime_state()
    if file_name in runtime_state["missing_images"]:
        return _build_terminal_error(
            f"Image file `{file_name}` is already confirmed unavailable. Do not retry it. Return fail or uncertain based on missing image."
        )

    target_file = _resolve_target_file(file_name)
    if target_file is None:
        runtime_state["missing_images"].add(file_name)
        candidates = _find_candidate_files(file_name, limit=5)
        return _build_terminal_error(
            f"Image file `{file_name}` is missing or cannot be matched exactly. Candidates: {candidates}"
        )
    
    # Collect parsed image items and build multimodal payloads.
    image_contents = []
    file_id = target_file.fileId
    for idx, item in enumerate(target_file.content):
        if hasattr(item, 'type') and item.type == 'image':
            image_caption = getattr(item, 'image_caption', None)
            if isinstance(image_caption, list):
                image_caption_text = " ".join(str(part) for part in image_caption if part)
            else:
                image_caption_text = str(image_caption or "").strip()
            image_description = (
                "Image content loaded.\n"
                f"fileId={file_id}\n"
                f"fileName={target_file.fileName}\n"
                f"elementId={getattr(item, 'element_id', '')}\n"
                f"page={getattr(item, 'page', '')}\n"
                f"caption={image_caption_text or 'N/A'}"
            )
            img_path = getattr(item, 'img_path', None)
            
            if img_path:
                img_file_name = os.path.basename(img_path)
                image_relative_path = f"{file_id}/{img_file_name}"
                image_base64, mime_type = await read_image_async(image_relative_path, IMAGE_BASE_DIR)
                
                if image_base64:
                    # Build multimodal content payload.
                    image_contents.append([
                        {"type": "text", "text": image_description},
                        {
                            "type": "image",
                            "base64": image_base64,
                            "mime_type": mime_type,
                        }
                    ])
    
    if not image_contents:
        runtime_state["missing_images"].add(file_name)
        return _build_terminal_error(
            f"File `{target_file.fileName}` has no usable image content, or the image artifact is missing on disk. Do not retry it."
        )
    
    return image_contents


@tool
def complete_check(
    requirementId: str,
    result: str,
    reason: str,
    evidence: Any = None,
    runtime: ToolRuntime = None,
) -> Command:
    """
    Finalize a requirement check and store the result in runtime/state.

    Call this after all controls under the current requirement are finished.

    Args:
        requirementId: Requirement ID.
        result: Final result, one of `pass`, `fail`, `uncertain`, `skipped`.
        reason: Human-readable conclusion.
        evidence: Evidence list with `fileId` and `elementIds`.
        runtime: Tool runtime injected by LangChain.

    Returns:
        A `Command` object that updates `check_results` in state.
    """


    runtime_state = _get_runtime_state()
    control_results = list(runtime_state.get("control_results", {}).values())
    normalized_input_evidence = _coerce_evidence_input(evidence)
    normalized_payload = _finalize_check_payload(
        payload_type="requirement_result",
        result=result,
        reason=reason,
        evidence=normalized_input_evidence,
    )

    # Build final check result payload.
    check_result_json = {
        "requirementId": requirementId,
        "result": normalized_payload["result"],
        "reason": normalized_payload["reason"],
        "evidence": normalized_payload["evidence"],
        "controlResults": control_results,
    }
    runtime_state["final_check_result"] = check_result_json

    logger.info("complete_check: {}", check_result_json)

    # Return a Command that updates graph state.
    return Command(
        update={
            "check_results": check_result_json,
            "control_results": control_results,
            "messages": [
                ToolMessage(
                    content=f"Check result saved: requirementId={requirementId}, result={result}",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool
def report_control_check(
    requirementId: str,
    controlId: str,
    result: str,
    reason: str,
    evidence: Any = None,
) -> str:
    """Buffer and merge a single control result."""
    runtime_state = _get_runtime_state()
    normalized_input_evidence = _coerce_evidence_input(evidence)
    normalized_payload = _finalize_check_payload(
        payload_type="control_result",
        result=result,
        reason=reason,
        evidence=normalized_input_evidence,
    )
    incoming_result = {
        "requirementId": requirementId,
        "controlId": controlId,
        "result": normalized_payload["result"],
        "reason": normalized_payload["reason"],
        "evidence": normalized_payload["evidence"],
    }
    control_results = runtime_state.setdefault("control_results", {})
    merged_result = merge_control_result(control_results.get(controlId), incoming_result)
    control_results[controlId] = merged_result
    logger.info("report_control_check buffered: {}", merged_result)
    return f"Control result buffered: controlId={controlId}, result={merged_result['result']}"
