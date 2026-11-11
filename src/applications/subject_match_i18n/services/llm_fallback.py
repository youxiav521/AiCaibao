"""LLM 兜底匹配：分批选择题模式调用 LLM"""

from __future__ import annotations

import asyncio

from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from src.applications.subject_match_i18n.constants import MatchType
from src.applications.subject_match_i18n.utils.attributes import check_attributes_compatible
from src.applications.subject_match_i18n.utils.match_utils import infer_line_attributes
from src.applications.subject_match_i18n.utils.text_normalize import normalize, strip_ifrs_qualifiers
from src.applications.subject_match_i18n.services.template_parser import TemplateSubject
from src.utils.llm_invoke import ainvoke_llm_with_structured_output

_MAX_CANDIDATES_PER_ROW = 10
_MAX_PENDING_ROWS_PER_LLM_CALL = 10
_LLM_CONFIDENCE_THRESHOLD = 0.75


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class _LLMMatchItem(BaseModel):
    rowIndex: int
    candidateNo: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""


class _LLMMatchResponse(BaseModel):
    matches: list[_LLMMatchItem]


# ── 候选构建（多路召回） ──────────────────────────────────────────────────────

def _build_row_candidates(
    raw_name: str,
    cleaned_name: str,
    line_type: str,
    indent_level: int,
    all_subjects: list[TemplateSubject],
    matched_codes: set[str],
    matched_by_row: dict[int, TemplateSubject],
    inferred_parent: int | None,
    lookup: dict[str, list[TemplateSubject]],
    synonym_lookup: dict[str, set[str]],
    code_to_subject: dict[str, TemplateSubject],
    line_attrs: dict[str, str] | None = None,
    unresolved: list[TemplateSubject] | None = None,
    fuzzy_must: list[tuple[TemplateSubject, float, bool]] | None = None,
    embedding_sim: dict[str, float] | None = None,
) -> list[tuple[TemplateSubject, str, float]]:
    """
    多路召回候选并排序，返回 (subject, source, score) 列表。
    候选来源：
      1. exact lookup → synonym lookup
      2. parent descendants
      3. unresolved（exact/synonym 阶段未消歧候选）
      4. fuzzy_must（fuzzy 阶段 0.65-0.85 必选候选）
      5. embedding_sim（按 cos 降序补到 _MAX_CANDIDATES_PER_ROW）
    不再有盲扫全量 subjects。
    """
    prefer_leaf = line_type not in ("SUBTOTAL", "TOTAL")
    candidates: dict[str, tuple[TemplateSubject, str, float]] = {}
    # 字面命中路径（exact/synonym/unresolved）豁免属性过滤：名称完全一致时属性推断的误差不应否决
    _ATTR_FILTER_EXEMPT_SOURCES = {"exact", "synonym", "unresolved"}

    def _add(s: TemplateSubject, source: str, score: float) -> None:
        if s.code in matched_codes:
            return
        if (
            line_attrs
            and source not in _ATTR_FILTER_EXEMPT_SOURCES
            and not check_attributes_compatible(line_attrs, s.attributes)
        ):
            return
        existing = candidates.get(s.code)
        if existing is None or score > existing[2]:
            candidates[s.code] = (s, source, score)

    # 路径 1: exact lookup（与主流程 exact_match 对齐：IFRS 剥离 + normalize）
    raw_key = normalize(strip_ifrs_qualifiers(raw_name))
    clean_key = normalize(strip_ifrs_qualifiers(cleaned_name))
    for key in (raw_key, clean_key):
        if not key:
            continue
        for s in lookup.get(key, []):
            _add(s, "exact", 1.0)

    # 路径 1.5: synonym lookup（与主流程 exact_match 对齐）
    for key in (raw_key, clean_key):
        if not key:
            continue
        synonym_codes = synonym_lookup.get(key)
        if not synonym_codes:
            continue
        for code in synonym_codes:
            if code in matched_codes:
                continue
            s = code_to_subject.get(code)
            if s:
                _add(s, "synonym", 1.0)

    # 路径 2: parent descendants
    parent_subject: TemplateSubject | None = None
    if inferred_parent is not None:
        parent_subject = matched_by_row.get(inferred_parent)
    if parent_subject:
        for s in all_subjects:
            if s.parent == parent_subject.code and s.code not in matched_codes:
                existing = candidates.get(s.code)
                if existing is None:
                    _add(s, "parent_child", 0.6)

    # 路径 3: unresolved（exact/synonym 阶段未消歧候选）
    if unresolved:
        for s in unresolved:
            _add(s, "unresolved", 0.9)

    # 路径 4: fuzzy_must（fuzzy 阶段 0.65-0.85 必选）
    if fuzzy_must:
        for s, cos, via in fuzzy_must:
            _add(s, "fuzzy_must", cos)

    # 路径 5: embedding_sim 按 cos 降序补充（不重复）
    if embedding_sim and len(candidates) < _MAX_CANDIDATES_PER_ROW:
        sorted_sims = sorted(embedding_sim.items(), key=lambda x: x[1], reverse=True)
        for code, cos in sorted_sims:
            if len(candidates) >= _MAX_CANDIDATES_PER_ROW:
                break
            if code in matched_codes or code in candidates:
                continue
            s = code_to_subject.get(code)
            if s:
                _add(s, "embedding", cos)

    # 截断：source 优先级 > leaf 对齐 > score 降序，保证高分候选不被截掉
    _SOURCE_PRIORITY = {
        "exact": 0, "synonym": 0, "unresolved": 0,
        "parent_child": 1, "fuzzy_must": 2, "embedding": 3,
    }
    result = list(candidates.values())

    def _truncate_key(item: tuple[TemplateSubject, str, float]) -> tuple:
        s, source, score = item
        source_pri = _SOURCE_PRIORITY.get(source, 4)
        leaf_match = 0 if s.is_leaf == prefer_leaf else 1
        return (source_pri, leaf_match, -score, s.code)

    result.sort(key=_truncate_key)
    result = result[:_MAX_CANDIDATES_PER_ROW]
    # 展示：按 code 中性排序，消除位置偏置，让 LLM 凭语义判断而非排序先验
    result.sort(key=lambda item: item[0].code)
    return result


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def _format_anchor_lines(state: Any) -> str:
    if not state.matched_by_row:
        return "Matched anchor rows: none"
    parts = ["Matched anchor rows (fixed positions):"]
    row_by_index = {line.get("rowIndex"): line for line in state.lines if line.get("rowIndex") is not None}
    for row_idx in sorted(state.matched_by_row.keys()):
        subject = state.matched_by_row[row_idx]
        line = row_by_index.get(row_idx, {})
        parts.append(
            "  row={}, code={}, name=\"{}\", indentLevel={}".format(
                row_idx,
                subject.code,
                subject.en_name,
                line.get("indentLevel", 1),
            )
        )
    return "\n".join(parts)


def _build_prompt(
    bundles: list[dict],
    state: Any,
) -> str:
    """选择题模式 prompt。"""
    prompt_parts = [
        "You are a financial statement account mapping expert (IFRS / US GAAP / ESEF).",
        "Map each pending row to the best-matching candidate by its number.\n",
        "## Rules",
        "- Each row has its own numbered candidate list. Select ONLY from that list.",
        "- Choose the candidate that best matches the row's **financial concept/semantics**,",
        "  NOT just the one with the most similar wording.",
        "- Pay attention to the statement section context (Operating/Investing/Financing for CF,",
        "  Profit/OCI for IS, Current/Non-current for BS) to avoid cross-section mismatches.",
        "- Use parentMatched context: a matched code indicates the parent section;",
        "  an (unmatched) name indicates the parent section even though it wasn't mapped to a template code.",
        "- Reject mismatched pairs: receivable/payable, current/non-current,",
        "  income/expense, asset/liability, operating/investing/financing, gross/net.",
        "- If uncertain, set candidateNo to null and confidence to 0.\n",
        "## Output — return json",
        '{"matches": [{"rowIndex": 3, "candidateNo": "1", "confidence": 0.92, "reason": "same concept"}, '
        '{"rowIndex": 5, "candidateNo": null, "confidence": 0, "reason": "no match"}]}\n',
    ]

    prompt_parts.append(_format_anchor_lines(state))
    prompt_parts.append("")
    prompt_parts.append("## Pending Rows")

    for bundle in bundles:
        header = (
            'Row {row}: rawName="{raw}", cleanedName="{clean}", lineType={lt}, '
            "indentLevel={ind}, parentMatched={pm}, prev=\"{prev}\", next=\"{next}\""
        ).format(
            row=bundle["row_idx"],
            raw=bundle["raw_name"],
            clean=bundle["cleaned_name"],
            lt=bundle["line_type"],
            ind=bundle["indent_level"],
            pm=bundle["parent_info"] or "none",
            prev=bundle["preceding"],
            next=bundle["following"],
        )
        prompt_parts.append(header)
        prompt_parts.append("Candidates:")
        prompt_parts.extend(bundle["candidate_lines"])
        prompt_parts.append("")

    return "\n".join(prompt_parts)


# ── 主函数 ────────────────────────────────────────────────────────────────────

async def llm_fallback(
    *,
    state: Any,
    lookup: dict[str, list[TemplateSubject]],
    all_subjects: list[TemplateSubject],
    synonym_lookup: dict[str, set[str]],
    code_to_subject: dict[str, TemplateSubject],
    fuzzy_must_candidates: dict[int, list[tuple[TemplateSubject, float, bool]]] | None = None,
    embedding_sim: dict[int, dict[str, float]] | None = None,
) -> None:
    """
    LLM 兜底匹配：将 pending lines 分批以选择题模式调用 LLM。
    候选构建优先级：
      1. exact lookup / synonym lookup
      2. parent_child（已匹配父行的子科目）
      3. unresolved_candidates（exact/synonym 阶段未消歧候选）
      4. fuzzy_must_candidates（fuzzy 阶段 0.65-0.85 必选）
      5. embedding_sim（按 cos 降序补到 _MAX_CANDIDATES_PER_ROW）
    匹配结果直接写回 state.matched_by_row。
    """
    if not state.pending_lines:
        return

    matched_codes = state.matched_codes
    row_by_idx = {l.get("rowIndex"): l for l in state.lines if l.get("rowIndex") is not None}

    # ── 为每行构建候选 ─────────────────────────────────────────────────────
    row_bundles: list[dict] = []
    candidate_no_to_code: dict[tuple[int, str], str] = {}

    for line in state.pending_lines:
        row_idx = line.get("rowIndex", -1)
        raw_name = line.get("rawName", "")
        cleaned_name = line.get("cleanedName", "")
        line_type = line.get("lineType", "ITEM")
        indent_level = line.get("indentLevel", 1)
        inferred_parent = line.get("inferredParent")
        preceding = line.get("precedingName") or ""
        following = line.get("followingName") or ""

        parent_info = ""
        if inferred_parent is not None:
            ps = state.matched_by_row.get(inferred_parent)
            if ps:
                parent_info = f"{ps.code}({ps.en_name})"
            else:
                # 父行未匹配模板科目，但仍提供父行名称作为上下文
                parent_line = row_by_idx.get(inferred_parent)
                parent_raw = (parent_line.get("rawName") or "") if parent_line else ""
                if parent_raw:
                    parent_info = f"(unmatched) \"{parent_raw}\""

        line_attrs = infer_line_attributes(line, row_by_idx)

        candidates = _build_row_candidates(
            raw_name=raw_name,
            cleaned_name=cleaned_name,
            line_type=line_type,
            indent_level=indent_level,
            all_subjects=all_subjects,
            matched_codes=matched_codes,
            matched_by_row=state.matched_by_row,
            inferred_parent=inferred_parent,
            lookup=lookup,
            synonym_lookup=synonym_lookup,
            code_to_subject=code_to_subject,
            line_attrs=line_attrs,
            unresolved=state.unresolved_candidates.get(row_idx),
            fuzzy_must=(fuzzy_must_candidates or {}).get(row_idx),
            embedding_sim=(embedding_sim or {}).get(row_idx),
        )

        if not candidates:
            continue

        cand_lines = []
        for i, (s, source, score) in enumerate(candidates, 1):
            leaf = "leaf" if s.is_leaf else "group"
            syn_part = ""
            if s.synonyms:
                syns = [a for a in s.synonyms if a]
                if syns:
                    syn_part = f" | aka: {', '.join(syns)}"
            cand_lines.append(
                f"  {i}. {s.en_name}{syn_part} | Lv{s.level} | parent={s.parent} | {leaf}"
            )
            candidate_no_to_code[(row_idx, str(i))] = s.code

        row_bundles.append({
            "row_idx": row_idx,
            "raw_name": raw_name,
            "cleaned_name": cleaned_name,
            "line_type": line_type,
            "indent_level": indent_level,
            "parent_info": parent_info,
            "preceding": preceding,
            "following": following,
            "candidate_lines": cand_lines,
            "line_attrs": line_attrs,
        })

    if not row_bundles:
        return

    # ── 分批并发调用 LLM（Semaphore 控制并发） ──────────────────────────────
    batch_size = _MAX_PENDING_ROWS_PER_LLM_CALL
    concurrent_limit = 10
    sem = asyncio.Semaphore(concurrent_limit)
    write_lock = asyncio.Lock()

    async def _call_batch(batch):
        async with sem:
            prompt = _build_prompt(batch, state)
            await _call_llm(
                prompt=prompt,
                bundles=batch,
                state=state,
                candidate_no_to_code=candidate_no_to_code,
                code_to_subject=code_to_subject,
                matched_codes=matched_codes,
                write_lock=write_lock,
            )

    tasks = [
        _call_batch(row_bundles[i:i + batch_size])
        for i in range(0, len(row_bundles), batch_size)
    ]
    await asyncio.gather(*tasks)

    llm_count = sum(1 for mt in state.match_type_by_row.values() if mt == MatchType.LLM_FALLBACK)
    if llm_count:
        logger.info("LLM兜底完成 | 表={}, 命中={}, 总pending={}", state.table_id, llm_count, len(state.pending_lines))


async def _call_llm(
    *,
    prompt: str,
    bundles: list[dict],
    state: Any,
    candidate_no_to_code: dict[tuple[int, str], str],
    code_to_subject: dict[str, TemplateSubject],
    matched_codes: set[str],
    write_lock: asyncio.Lock | None = None,
) -> None:
    """选择题模式，LLM 从编号候选中选择。"""
    pending_row_indexes = {b["row_idx"] for b in bundles}
    attrs_by_row = {b["row_idx"]: b.get("line_attrs") or {} for b in bundles}
    used_codes: set[str] = set()

    model_id = "subject_match_llm_fallback"
    try:
        response = await ainvoke_llm_with_structured_output(
            prompt=prompt,
            model_id=model_id,
            response_type=_LLMMatchResponse,
        )
    except Exception:
        logger.exception("LLM 兜底匹配失败")
        return

    for item in response.matches:
        if not item.candidateNo:
            continue
        if item.confidence < _LLM_CONFIDENCE_THRESHOLD:
            logger.info("LLM低置信度跳过 | row={}, conf={:.2f}", item.rowIndex, item.confidence)
            continue
        if item.rowIndex not in pending_row_indexes:
            logger.warning("LLM返回非pending行 | rowIndex={}", item.rowIndex)
            continue

        code = candidate_no_to_code.get((item.rowIndex, item.candidateNo))
        if not code:
            logger.warning("LLM返回无效候选编号 | row={}, no={}", item.rowIndex, item.candidateNo)
            continue
        if code not in code_to_subject:
            continue
        if code in matched_codes or code in used_codes:
            logger.warning("LLM返回已占用code | row={}, code={}", item.rowIndex, code)
            continue

        # 确定性安全校验：属性硬冲突（如 current↔non-current、跨段）直接拒绝
        line_attrs = attrs_by_row.get(item.rowIndex)
        if line_attrs and not check_attributes_compatible(line_attrs, code_to_subject[code].attributes):
            logger.warning(
                "LLM选择属性冲突候选被拒 | row={}, code={}, line_attrs={}",
                item.rowIndex, code, line_attrs,
            )
            continue

        if write_lock:
            async with write_lock:
                # 加锁后再次检查，防止并发冲突
                if code in matched_codes or code in used_codes:
                    logger.warning("LLM并发冲突跳过 | row={}, code={}", item.rowIndex, code)
                    continue
                state.matched_by_row[item.rowIndex] = code_to_subject[code]
                state.match_type_by_row[item.rowIndex] = MatchType.LLM_FALLBACK
                matched_codes.add(code)
                used_codes.add(code)
        else:
            state.matched_by_row[item.rowIndex] = code_to_subject[code]
            state.match_type_by_row[item.rowIndex] = MatchType.LLM_FALLBACK
            matched_codes.add(code)
            used_codes.add(code)
        logger.info("LLM匹配 | row={}, code={}, conf={:.2f}", item.rowIndex, code, item.confidence)
