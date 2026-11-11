"""主服务：编排四级科目匹配（精确匹配 → 模糊匹配 → 上下文消歧 → LLM 兜底）"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.applications.subject_match_i18n.constants import MatchType
from src.applications.subject_match_i18n.schemas.match_schema import (
    MatchResponse,
    MatchResultItem,
    ParseLineInput,
    TemplateSubjectInput,
)
from src.applications.subject_match_i18n.services.candidate_index import CandidateIndex, embedding_rescue, embedding_verify
from src.applications.subject_match_i18n.services.context_resolver import build_descendant_index, resolve
from src.applications.subject_match_i18n.services.llm_fallback import llm_fallback
from src.applications.subject_match_i18n.services.template_parser import (
    TemplateSubject,
    build_lookup,
    propagate_liquidity,
    propagate_section,
)
from src.applications.subject_match_i18n.utils.match_utils import (
    _SKIP_LINE_TYPES,
    build_next_effective_line_index,
    build_synonym_lookup,
    convert_template_input,
    exact_match,
    filter_candidates_by_line_level,
    fuzzy_recall,
    infer_line_attributes,
)
from src.applications.subject_match_i18n.utils.attributes import check_attributes_compatible
from src.applications.subject_match_i18n.utils.text_normalize import normalize, strip_ifrs_qualifiers


class _MatchState:
    """规则匹配阶段结束后的中间状态"""
    __slots__ = (
        "table_id", "lines", "matched_by_row",
        "match_type_by_row", "pending_lines",
        "unresolved_candidates",
    )

    def __init__(
        self,
        table_id: str,
        lines: list[dict[str, Any]],
        matched_by_row: dict[int, TemplateSubject],
        match_type_by_row: dict[int, str],
        pending_lines: list[dict[str, Any]],
        unresolved_candidates: dict[int, list[TemplateSubject]] | None = None,
    ):
        self.table_id = table_id
        self.lines = lines
        self.matched_by_row = matched_by_row
        self.match_type_by_row = match_type_by_row
        self.pending_lines = pending_lines
        # 精确/同义词阶段同名歧义未消歧成功的候选 (row_idx -> [subjects])
        self.unresolved_candidates: dict[int, list[TemplateSubject]] = unresolved_candidates or {}

    @property
    def matched_codes(self) -> set[str]:
        return {s.code for s in self.matched_by_row.values()}


def _rule_match(
    table_id: str,
    lines: list[dict[str, Any]],
    lookup: dict[str, list[TemplateSubject]],
    synonym_lookup: dict[str, set[str]],
    code_to_subject: dict[str, TemplateSubject],
    descendant_index: dict[str, set[str]],
    excluded_rows: set[int] | None = None,
    pre_matched_codes: set[str] | None = None,
) -> _MatchState:
    """
    执行精确匹配 + 同义词匹配 + 上下文消歧，返回中间状态供后续 fuzzy/embedding/LLM。
    """
    matched_codes: set[str] = set(pre_matched_codes or set())
    _excluded = excluded_rows or set()

    matchable_lines = [
        line for line in lines
        if line.get("rowIndex") is not None
        and line.get("rowIndex") not in _excluded
        and line.get("lineType", "ITEM") not in _SKIP_LINE_TYPES
        and (line.get("rawName") or line.get("cleanedName"))
    ]

    # 流动性消歧需要按 rowIndex 反查父行，所有行都建索引（包括 HEADER 等区块行）
    row_by_idx: dict[int, dict[str, Any]] = {
        line["rowIndex"]: line
        for line in lines
        if line.get("rowIndex") is not None
    }

    next_effective = build_next_effective_line_index(lines)

    matched_by_row: dict[int, TemplateSubject] = {}
    match_type_by_row: dict[int, str] = {}
    ambiguous: dict[int, tuple[dict[str, Any], list[TemplateSubject], str]] = {}

    for line in matchable_lines:
        row_idx: int = line["rowIndex"]
        raw_name = line.get("rawName", "")
        cleaned_name = line.get("cleanedName", "")
        raw_key = normalize(strip_ifrs_qualifiers(raw_name))
        clean_key = normalize(strip_ifrs_qualifiers(cleaned_name))

        candidates, source = exact_match(
            raw_key, clean_key,
            lookup, synonym_lookup, code_to_subject, matched_codes,
        )

        if not candidates:
            continue

        # 属性闸门前置：先按维度属性（流动性 / 报表段 / 收付方向等）过滤掉硬冲突候选，
        # 避免 filter_candidates_by_line_level 的 leaf 偏好抢跑--例如源 "Inventory" 在 "Current assets"
        # 父链下，原先会被 leaf 偏好直接选成唯一的 leaf "Non-current inventories"。
        if len(candidates) > 1:
            line_attrs = infer_line_attributes(line, row_by_idx)
            if line_attrs:
                compatible = [
                    c for c in candidates
                    if check_attributes_compatible(line_attrs, c.attributes)
                ]
                if compatible:
                    candidates = compatible

        if len(candidates) > 1:
            candidates = filter_candidates_by_line_level(line, candidates, next_effective)

        if len(candidates) == 1:
            matched_by_row[row_idx] = candidates[0]
            match_type_by_row[row_idx] = source
            matched_codes.add(candidates[0].code)
        else:
            ambiguous[row_idx] = (line, candidates, source)

    logger.info(
        "第一阶段精确/同义词匹配完成 | 表={}, 已匹配={} 行, 同名歧义待消歧={} 行, 可匹配行总数={}",
        table_id,
        len(matched_by_row),
        len(ambiguous),
        len(matchable_lines),
    )

    disambiguated = 0
    unresolved_candidates: dict[int, list[TemplateSubject]] = {}
    for row_idx, (line, candidates, source) in ambiguous.items():
        resolved = resolve(line, candidates, matched_by_row, descendant_index, row_by_idx)
        if resolved is not None:
            matched_by_row[row_idx] = resolved
            match_type_by_row[row_idx] = source
            matched_codes.add(resolved.code)
            disambiguated += 1
        elif source == MatchType.SYNONYM:
            # 同义词等价集合的"无上下文"兜底：旧实现后写覆盖总是单命中，
            # 新实现保留等价集合后 resolve 可能无法消歧。
            # 取第一个候选中第一个（exact_match 传入时已按 code 排序）作为确定性兜底。
            # 如果这个候选人已被占用，跳过以免 UB。
            remaining = [c for c in candidates if c.code not in matched_codes]
            if remaining:
                picked = remaining[0]
                matched_by_row[row_idx] = picked
                match_type_by_row[row_idx] = source
                matched_codes.add(picked.code)
                logger.info(
                    "同义词兜底 | row={}, code={}, name='{}'",
                    row_idx, picked.code, picked.en_name,
                )
                disambiguated += 1
            else:
                unresolved_candidates[row_idx] = candidates
        else:
            # exact 同名歧义未消歧 → 保留候选供 LLM 兜底
            unresolved_candidates[row_idx] = candidates

    logger.info(
        "上下文消歧完成 | 表={}, 消歧成功={} 行, 未消歧保留候选={} 行",
        table_id,
        disambiguated,
        len(unresolved_candidates),
    )

    all_pending = [
        line for line in matchable_lines
        if line["rowIndex"] not in matched_by_row
    ]

    logger.info(
        "规则匹配（exact/synonym）完成 | 表={}, 命中={} 行, 进入fuzzy={} 行",
        table_id,
        len(matched_by_row),
        len(all_pending),
    )

    return _MatchState(
        table_id=table_id,
        lines=lines,
        matched_by_row=matched_by_row,
        match_type_by_row=match_type_by_row,
        pending_lines=all_pending,
        unresolved_candidates=unresolved_candidates,
    )


async def _fuzzy_match_with_embedding(
    state: _MatchState,
    fuzzy_subjects: list[TemplateSubject],
    descendant_index: dict[str, set[str]],
    row_by_idx: dict[int, dict[str, Any]],
    next_effective: dict[int, int | None],
) -> tuple[set[int], dict[int, tuple[dict[str, Any], list[tuple[TemplateSubject, float, bool]]]]]:
    """
    Fuzzy 阶段（异步，带 embedding 验证）：
    - 调 fuzzy_recall 拿到字面召回候选
    - 对所有 (line, candidate) 对做 embedding 余弦相似度验证
    - 三层分级：
      * cos >= EMB_DIRECT_THRESHOLD (0.85) → 直接 fuzzy 命中
      * EMB_LLM_THRESHOLD (0.65) <= cos < 0.85 → 留给 LLM 兜底
      * cos < 0.65 → 舍弃
    返回：(直接命中的 row 集合, LLM 候选 {row -> (line, [(subject, cos, via_synonym), ...])})
    """
    matched_by_row = state.matched_by_row
    match_type_by_row = state.match_type_by_row
    matched_codes = set(state.matched_codes)  # 本地快照，命中时同步更新
    pending_lines = list(state.pending_lines)

    # 第 1 步：召回所有 (line, candidate, literal_score, via_synonym)
    recall_per_row: dict[int, tuple[dict[str, Any], list[tuple[TemplateSubject, float, bool]]]] = {}
    for line in pending_lines:
        raw_name = line.get("rawName", "")
        cleaned_name = line.get("cleanedName", "")
        scored = fuzzy_recall(raw_name, cleaned_name, fuzzy_subjects, matched_codes)
        if not scored:
            continue
        recall_per_row[line["rowIndex"]] = (
            line,
            [(s, score, via) for score, s, via in scored],
        )

    if not recall_per_row:
        return set(), {}

    # 第 2 步：构建 (line_text, candidate_text) 索引，批量调 embedding
    line_text_index: dict[int, int] = {}  # row_idx -> pos in line_texts
    line_texts: list[str] = []
    cand_text_index: dict[tuple[int, str], int] = {}  # (row_idx, code) -> pos in candidate_texts
    cand_texts: list[str] = []

    for row_idx, (line, candidates) in recall_per_row.items():
        line_text = line.get("rawName") or line.get("cleanedName") or ""
        if not line_text:
            continue
        line_text_index[row_idx] = len(line_texts)
        line_texts.append(line_text)
        for s, _score, _via in candidates:
            key = (row_idx, s.code)
            if key not in cand_text_index:
                cand_text_index[key] = len(cand_texts)
                cand_texts.append(s.en_name or "")

    sim_matrix = await embedding_verify(line_texts, cand_texts)

    # 第 3 步：三层分级 + 收集所有候选（cos > 0）供 LLM 兜底使用
    direct_matched_rows: set[int] = set()
    # llm_candidates: row_idx -> (line, [(subject, cos, via_synonym), ...])
    # 包含所有 cos > 0 的召回候选（按 cos 降序），供 LLM 阶段按需选用
    llm_candidates: dict[int, tuple[dict[str, Any], list[tuple[TemplateSubject, float, bool]]]] = {}

    EMB_DIRECT_THRESHOLD = 0.85
    EMB_LLM_THRESHOLD = 0.65

    for row_idx, (line, candidates) in recall_per_row.items():
        if row_idx not in line_text_index:
            continue
        li = line_text_index[row_idx]

        # 收集每个 candidate 的 cos 分数（保留所有 > 0 的，按 cos 降序）
        scored_cands: list[tuple[TemplateSubject, float, bool]] = []
        for s, _literal, via in candidates:
            ci = cand_text_index.get((row_idx, s.code))
            if ci is None or not sim_matrix:
                continue
            cos = float(sim_matrix[li][ci])
            if cos <= 0:
                continue
            scored_cands.append((s, cos, via))

        if not scored_cands:
            continue

        scored_cands.sort(key=lambda x: x[1], reverse=True)
        best_subject, best_cos, best_via = scored_cands[0]

        if best_cos >= EMB_DIRECT_THRESHOLD and best_subject.code not in matched_codes:
            # 直接命中
            line_attrs = infer_line_attributes(line, row_by_idx)
            if not check_attributes_compatible(line_attrs, best_subject.attributes):
                # 维度反向校验失败 → 留全部 cos > 0 候选给 LLM
                llm_candidates[row_idx] = (line, scored_cands)
                continue
            matched_by_row[row_idx] = best_subject
            match_type_by_row[row_idx] = (
                MatchType.SYNONYM_FUZZY if best_via else MatchType.FUZZY
            )
            matched_codes.add(best_subject.code)
            direct_matched_rows.add(row_idx)
        else:
            # 0.65 ≤ cos < 0.85 或 cos < 0.65 → 全部留给 LLM（按 cos 排序，cos>=0.65 优先选）
            llm_candidates[row_idx] = (line, scored_cands)

    # 更新 pending_lines：移除已直接命中的行
    state.pending_lines = [l for l in pending_lines if l["rowIndex"] not in direct_matched_rows]

    logger.info(
        "Fuzzy+Embedding 完成 | 召回行={}, 直接命中={}, 留给LLM={}",
        len(recall_per_row), len(direct_matched_rows), len(llm_candidates),
    )

    return direct_matched_rows, llm_candidates


async def match(
    table_id: str,
    template: list[TemplateSubjectInput],
    parse_result: list[ParseLineInput],
) -> MatchResponse:
    """
    五级级联科目匹配主流程：
    1. 转换模板，构建查找表
    2. 倒排索引初筛 → 精确匹配 + 模糊匹配 + 上下文消歧
    3. Embedding 救回（未匹配行 × 未命中科目）→ 再次规则匹配
    4. LLM 兜底
    """
    subjects = convert_template_input(template)
    if not subjects:
        logger.warning("模板为空")
        return MatchResponse(id=table_id)

    propagate_liquidity(subjects)
    propagate_section(subjects)

    lookup = build_lookup(subjects)
    synonym_lookup = build_synonym_lookup(subjects)
    code_to_subject = {s.code: s for s in subjects}
    descendant_index = build_descendant_index(subjects)
    lines = [line.model_dump() for line in parse_result]

    matchable_lines = [
        line for line in lines
        if line.get("rowIndex") is not None
        and line.get("lineType", "ITEM") not in _SKIP_LINE_TYPES
        and (line.get("rawName") or line.get("cleanedName"))
    ]

    # 倒排初筛：始终启用，用 token 召回缩小 fuzzy 扫描集，剩余科目交由阶段 2 的 embedding 救回
    if matchable_lines:
        index = CandidateIndex(subjects)
        token_hit_codes = index.token_recall_codes(matchable_lines)
        token_hit_subjects = [s for s in subjects if s.code in token_hit_codes]
        token_miss_subjects = [s for s in subjects if s.code not in token_hit_codes]

        logger.info(
            "倒排初筛 | 原始={}, 倒排命中={}, 未命中={}",
            len(subjects), len(token_hit_subjects), len(token_miss_subjects),
        )
    else:
        token_hit_subjects = []
        token_miss_subjects = []

    # 阶段 1: exact/synonym + 消歧（全量 lookup）
    state = _rule_match(
        table_id=table_id,
        lines=lines,
        lookup=lookup,
        synonym_lookup=synonym_lookup,
        code_to_subject=code_to_subject,
        descendant_index=descendant_index,
    )

    # 阶段 2: 准备 fuzzy 候选池 = token_hit + embedding 救回的 token_miss
    fuzzy_subjects: list[TemplateSubject] = []
    per_line_sim: dict[int, dict[str, float]] = {}
    if state.pending_lines:
        matched_codes_now = state.matched_codes
        remaining_hit = [s for s in token_hit_subjects if s.code not in matched_codes_now]
        remaining_miss = [s for s in token_miss_subjects if s.code not in matched_codes_now]

        rescued_codes: set[str] = set()
        if remaining_miss:
            rescued_codes, per_line_sim = await embedding_rescue(state.pending_lines, remaining_miss)

        fuzzy_subjects = remaining_hit + [
            s for s in remaining_miss if s.code in rescued_codes
        ]

        logger.info(
            "Fuzzy 候选池准备 | token命中待复用={}, embedding救回={}, 总计={}",
            len(remaining_hit), len(rescued_codes), len(fuzzy_subjects),
        )

    # 阶段 3: fuzzy 召回 + embedding 验证 + 三层分级
    row_by_idx = {
        line["rowIndex"]: line for line in lines if line.get("rowIndex") is not None
    }
    next_effective = build_next_effective_line_index(lines)
    llm_candidates: dict[int, tuple[dict[str, Any], list[tuple[TemplateSubject, float, bool]]]] = {}
    if state.pending_lines and fuzzy_subjects:
        _, llm_candidates = await _fuzzy_match_with_embedding(
            state=state,
            fuzzy_subjects=fuzzy_subjects,
            descendant_index=descendant_index,
            row_by_idx=row_by_idx,
            next_effective=next_effective,
        )

    # 提取 fuzzy_must（0.65-0.85）和 embedding_supplement（<0.65 + per_line_sim）
    fuzzy_must_candidates: dict[int, list[tuple[TemplateSubject, float, bool]]] = {}
    # embedding_supplement = per_line_sim (token_miss) + llm_candidates 中 <0.65 的
    embedding_supplement: dict[int, dict[str, float]] = {}
    for row_idx, (line, cands) in llm_candidates.items():
        must = [(s, cos, via) for s, cos, via in cands if cos >= 0.65]
        low = {s.code: cos for s, cos, via in cands if cos < 0.65}
        if must:
            fuzzy_must_candidates[row_idx] = must
        if low:
            embedding_supplement[row_idx] = low
    # 合并 per_line_sim（token_miss 科目 embedding 分）
    for row_idx, sim_dict in per_line_sim.items():
        if row_idx not in embedding_supplement:
            embedding_supplement[row_idx] = {}
        existing = embedding_supplement[row_idx]
        for code, cos in sim_dict.items():
            if code not in existing or cos > existing[code]:
                existing[code] = cos

    if state.pending_lines:
        await llm_fallback(
            state=state,
            lookup=lookup,
            all_subjects=subjects,
            synonym_lookup=synonym_lookup,
            code_to_subject=code_to_subject,
            fuzzy_must_candidates=fuzzy_must_candidates,
            embedding_sim=embedding_supplement,
        )

    row_order = {
        line.get("rowIndex"): i
        for i, line in enumerate(state.lines)
        if line.get("rowIndex") is not None
    }
    results = [
        MatchResultItem(
            rowIndex=row_idx,
            matchedCode=subject.code,
            matchedName=subject.en_name,
            matchType=state.match_type_by_row.get(row_idx, MatchType.LLM_FALLBACK),
        )
        for row_idx, subject in state.matched_by_row.items()
    ]
    results.sort(key=lambda r: row_order.get(r.rowIndex, r.rowIndex))
    return MatchResponse(id=table_id, matchedResults=results)
