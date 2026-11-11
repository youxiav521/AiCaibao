from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Optional

from loguru import logger

from ...schemas.subject_match_schema import Cell, Report, Subject, Table
from .text_utils import FAMILY_SUFFIXES_DEDUP, classify_family_pair, deep_normalize

_AGGREGATE_SUFFIXES: tuple[str, ...] = ("净额", "净增加额", "净变动额", "净减少额")
_NET_TYPE_SUFFIXES: tuple[str, ...] = ("净额", "净值", "净损失", "净收益")

# 补充资料（间接法）调节项关键词——这些科目属于间接法附表，不应与直接法主表科目混匹
_INDIRECT_METHOD_KEYWORDS: tuple[str, ...] = (
    "存货的减少", "存货的增加",
    "经营性应收项目的减少", "经营性应收项目的增加",
    "经营性应付项目的增加", "经营性应付项目的减少",
    "待摊费用减少", "预提费用增加",
    "固定资产折旧", "无形资产摊销", "长期待摊费用摊销",
    "处置固定资产", "固定资产报废损失",
    "财务费用", "投资损失", "投资收益",
    "递延税款", "递延所得税",
    "公允价值变动损失", "公允价值变动收益",
    "计提的资产减值准备", "资产减值准备",
)

# 利润表专属科目关键词——与现金流量表科目互斥
_INCOME_STATEMENT_ONLY_KEYWORDS: tuple[str, ...] = (
    "营业收入", "营业成本", "营业总收入", "营业总成本",
    "财务费用", "管理费用", "销售费用", "研发费用",
    "资产减值损失", "信用减值损失",
    "公允价值变动收益", "投资收益",
    "营业外收入", "营业外支出",
    "所得税费用", "所得税",
)
_SUMMARY_SUFFIXES: tuple[str, ...] = ("合计", "小计", "总计")

def whole_table_same_id_dedup(
    table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    """Ensure each template subjectId is used by at most one source text across the table.

    Scans ALL cells (Java and Python) and keeps only the single best match
    per subjectId. This handles:
      - Java vs Python conflicts (e.g. Java "应收账款" + Python "应收账款净额" → same sid)
      - Java vs Java conflicts (e.g. Java "应收账款" + Java "应收账款净额" → same sid)
        which occur when Java sends all subjects at once instead of in batches.
    """
    sid_to_cells: dict[str, list] = defaultdict(list)
    for row in table.rows:
        for cell in row:
            if cell.subjectId:
                sid_to_cells[cell.subjectId].append(cell)

    for sid, cells in sid_to_cells.items():
        if len(cells) <= 1:
            continue

        tname = subject_id_to_name.get(sid, sid)

        deep_tname = deep_normalize(tname)

        scored = sorted(
            cells,
            key=lambda c: (
                deep_normalize(c.text.strip()) == deep_tname,   # 1) 深度完全一致优先
                c.text.strip() == tname,                              # 2) 原文完全一致
                SequenceMatcher(None, c.text.strip(), tname).ratio(), # 3) 相似度
                c.cellId in pre_matched_cell_ids,                     # 4) 最后才看 Java
            ),
            reverse=True,
        )
        keep = scored[0]
        for c in scored[1:]:
            origin = "Java" if c.cellId in pre_matched_cell_ids else "Python"
            c.subjectId = None
            
            # 显示同一模板被多个源文本占用的情况
            logger.warning(
                "[Same-ID] template={!r}: KEPT={!r} ({}) vs CLEARED={!r} ({}) | "
                "sim: {:.2f} vs {:.2f}",
                tname,
                keep.text.strip(), "Java" if keep.cellId in pre_matched_cell_ids else "Python",
                c.text.strip(), origin,
                SequenceMatcher(None, keep.text.strip(), tname).ratio(),
                SequenceMatcher(None, c.text.strip(), tname).ratio(),
            )

def whole_table_name_family_dedup(
    table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    """Scan all cells in the table and clear family-duplicate subjectIds.

    This is the final safety net for "多个加总科目" when summaryRelation is absent.
    It groups all assigned subjectIds by their normalized template name and,
    within each family, keeps only the member whose source text best matches
    its template name.
    """
    sid_to_cells_all: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for row in table.rows:
        for cell in row:
            if cell.subjectId:
                sid_to_cells_all[cell.subjectId].append((cell.text.strip(), cell))

    if len(sid_to_cells_all) < 2:
        return

    name_to_sid = subject_id_to_name
    sid_deep: dict[str, str] = {}
    for sid in sid_to_cells_all:
        tname = name_to_sid.get(sid, "")
        sid_deep[sid] = deep_normalize(tname)

    sids_to_clear: set[str] = set()
    processed: set[str] = set()

    sid_list = list(sid_to_cells_all.keys())
    for i, sid_a in enumerate(sid_list):
        if sid_a in sids_to_clear or sid_a in processed:
            continue
        deep_a = sid_deep.get(sid_a, "")
        if not deep_a:
            continue

        family: list[str] = [sid_a]
        for j in range(i + 1, len(sid_list)):
            sid_b = sid_list[j]
            if sid_b in sids_to_clear or sid_a == sid_b:
                continue
            deep_b = sid_deep.get(sid_b, "")
            if not deep_b:
                continue

            # 关键修复：不仅 identical，suffix/compound 也视为同家族
            family_type = classify_family_pair(deep_a, deep_b)
            if family_type is not None:
                family.append(sid_b)

        if len(family) <= 1:
            continue

        for sid in family:
            processed.add(sid)

        # Score each family member by its BEST source-cell evidence (not cells[0]).
        # Priority: deep-exact > exact > best similarity > Java pre-match > shorter template name.
        scored: list[tuple[str, bool, bool, float, bool, int]] = []
        for sid in family:
            tname = name_to_sid.get(sid, "")
            cells = sid_to_cells_all[sid]
            deep_tname = deep_normalize(tname)
            is_java = any(c.cellId in pre_matched_cell_ids for _, c in cells)
            deep = sid_deep.get(sid, "")

            best_exact_deep = False
            best_exact_raw = False
            best_sim = 0.0

            for src_text, _ in cells:
                src = src_text.strip()
                src_deep = deep_normalize(src)
                if src_deep and src_deep == deep_tname:
                    best_exact_deep = True
                if src == tname:
                    best_exact_raw = True
                sim = SequenceMatcher(None, src, tname).ratio()
                if sim > best_sim:
                    best_sim = sim

            scored.append((sid, best_exact_deep, best_exact_raw, best_sim, is_java, len(deep)))

        scored.sort(key=lambda x: (x[1], x[2], x[3], x[4], -x[5]), reverse=True)
        keep_sid = scored[0][0]
        keep_tname = name_to_sid.get(keep_sid, "")

        for sid, _ed, _er, sim, is_java, _len_deep in scored[1:]:
            tname = name_to_sid.get(sid, "")
            cells = sid_to_cells_all[sid]
            sids_to_clear.add(sid)
            origin = "Java" if is_java else "Python"
            
            # 改进日志：显示同家族科目的竞争关系
            logger.warning(
                "[Family] family_type={}: KEPT={!r} vs CLEARED={!r} ({}) | sim={:.2f}",
                family_type, keep_tname, tname, origin, sim,
            )
            for _, cell in cells:
                cell.subjectId = None

def whole_table_source_family_dedup(
    table: Table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    """按源文本家族做最终去重，避免相似来源行同时命中造成重复加总。"""

    assigned_cells: list[Cell] = []
    for row in table.rows:
        for cell in row:
            if cell.subjectId:
                assigned_cells.append(cell)

    if len(assigned_cells) <= 1:
        return

    def _source_family_key(text: str) -> str:
        src = deep_normalize((text or "").strip())
        if not src:
            return ""
        for suffix in FAMILY_SUFFIXES_DEDUP:
            if src.endswith(suffix) and len(src) > len(suffix):
                src = src[: -len(suffix)]
                break
        return src

    family_to_cells: dict[str, list[Cell]] = defaultdict(list)
    for c in assigned_cells:
        fam_key = _source_family_key(c.text)
        if fam_key:
            family_to_cells[fam_key].append(c)

    for fam_key, cells in family_to_cells.items():
        if len(cells) <= 1:
            continue

        scored: list[tuple[Cell, tuple]] = []
        for c in cells:
            sid = c.subjectId or ""
            tname = subject_id_to_name.get(sid, sid)
            src = (c.text or "").strip()
            src_deep = deep_normalize(src)
            tname_deep = deep_normalize(tname)

            score = (
                src_deep == tname_deep,
                src == tname,
                not any(src.endswith(s) for s in _NET_TYPE_SUFFIXES),
                SequenceMatcher(None, src, tname).ratio(),
                c.cellId in pre_matched_cell_ids,
            )
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_cell = scored[0][0]

        for c, _ in scored[1:]:
            origin = "Java" if c.cellId in pre_matched_cell_ids else "Python"
            c.subjectId = None
            logger.warning(
                "Source-family dedup: cleared {} cell {!r}, kept {!r} (family={!r})",
                origin, c.text.strip(), keep_cell.text.strip(), fam_key,
            )

def enforce_one_to_one(
    table: Table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    """Final safety net: enforce strict 1:1 by TEMPLATE NAME (not only subjectId).

    Why:
    - Different subjectIds may share same template name.
    - Java pre-matched cells may already occupy the same template target.
    - We must keep exactly one source cell per template-name family.
    """
    def _is_net_variant_text(text: str) -> bool:
        t = (text or "").strip()
        return any(t.endswith(s) for s in _NET_TYPE_SUFFIXES)

    # Group by deep-normalized template name (subjectId -> subjectName -> deep_name)
    tname_to_cells: dict[str, list[tuple[str, Cell]]] = defaultdict(list)
    for row in table.rows:
        for cell in row:
            sid = cell.subjectId
            if not sid:
                continue
            tname = subject_id_to_name.get(sid, sid)
            deep_tname = deep_normalize(tname)
            if not deep_tname:
                continue
            tname_to_cells[deep_tname].append((tname, cell))

    for deep_tname, pairs in tname_to_cells.items():
        if len(pairs) <= 1:
            continue

        # Keep the single best cell for this template-name family.
        # Priority:
        # 1) deep exact text==template
        # 2) raw exact text==template
        # 3) non-net variant preferred (e.g. keep "应收账款" over "应收账款净额")
        # 4) higher similarity
        # 5) Java pre-matched as last tie-breaker
        scored: list[tuple[Cell, tuple]] = []
        for tname, c in pairs:
            src = c.text.strip()
            src_deep = deep_normalize(src)
            score = (
                src_deep == deep_tname,
                src == tname,
                not _is_net_variant_text(src),
                SequenceMatcher(None, src, tname).ratio(),
                c.cellId in pre_matched_cell_ids,
            )
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_cell = scored[0][0]
        keep_score = scored[0][1]

        for c, lose_score in scored[1:]:
            origin = "Java" if c.cellId in pre_matched_cell_ids else "Python"
            c.subjectId = None
            
            # 改进日志：显示最终兜底时的竞争和决策
            logger.warning(
                "[Enforce-1:1] template_family={!r}: KEPT={!r} vs CLEARED={!r} ({}) | "
                "score: {:.2f} vs {:.2f}",
                deep_tname, keep_cell.text.strip(), c.text.strip(), origin,
                keep_score[3], lose_score[3],  # 相似度对比
            )

def apply_net_amount_exclusion(
    text_to_first_cells: dict[str, list],
    subject_id_to_name: dict[str, str],
) -> None:
    """Remove net-amount variants that conflict with the primary subject.

    If "X" and "X净额" both appear as source texts and map to different template subjects,
    this is valid. But if they map to the *same* template subject, the net-amount variant
    should be cleared to prevent double-counting.
    """
    sid_to_texts: dict[str, list[str]] = defaultdict(list)
    for txt, cells in text_to_first_cells.items():
        sid = cells[0].subjectId if cells else None
        if sid:
            sid_to_texts[sid].append(txt)

    for sid, texts in sid_to_texts.items():
        if len(texts) <= 1:
            continue
        net_texts = [t for t in texts if any(t.endswith(s) for s in _NET_TYPE_SUFFIXES)]
        non_net_texts = [t for t in texts if t not in net_texts]
        if net_texts and non_net_texts:
            for nt in net_texts:
                for cell in text_to_first_cells[nt]:
                    cell.subjectId = None
                logger.warning(
                    "Net-amount exclusion: cleared {!r} (keeping {!r}) for template {!r}",
                    nt, non_net_texts[0], subject_id_to_name.get(sid, sid),
                )


def apply_name_family_dedup(
    text_to_first_cells: dict[str, list],
    subject_id_to_name: dict[str, str],
) -> None:
    """Detect sibling subjects by template name patterns and keep only one per family.

    Works without summaryRelation by inferring families from name structure.
    """
    matched_entries: list[tuple[str, str, str]] = []
    for txt, cells in text_to_first_cells.items():
        sid = cells[0].subjectId if cells else None
        if sid:
            tname = subject_id_to_name.get(sid, "")
            matched_entries.append((txt, sid, tname))

    if len(matched_entries) < 2:
        return

    sids_to_clear: set[str] = set()
    checked: set[tuple[str, str]] = set()

    for i, (txt_a, sid_a, tname_a) in enumerate(matched_entries):
        if sid_a in sids_to_clear:
            continue
        deep_a = deep_normalize(tname_a)
        if not deep_a:
            continue

        for j in range(i + 1, len(matched_entries)):
            txt_b, sid_b, tname_b = matched_entries[j]
            if sid_b in sids_to_clear or sid_a == sid_b:
                continue

            pair = (min(sid_a, sid_b), max(sid_a, sid_b))
            if pair in checked:
                continue
            checked.add(pair)

            deep_b = deep_normalize(tname_b)
            if not deep_b:
                continue

            # 关键修复：只要是同家族（identical/suffix/compound）都参与去重
            family_type = classify_family_pair(deep_a, deep_b)
            if family_type is None:
                continue

            sim_a = SequenceMatcher(None, txt_a, tname_a).ratio()
            sim_b = SequenceMatcher(None, txt_b, tname_b).ratio()

            if sim_a >= sim_b:
                clear_sid, clear_txt, clear_tname = sid_b, txt_b, tname_b
                keep_txt, keep_tname = txt_a, tname_a
            else:
                clear_sid, clear_txt, clear_tname = sid_a, txt_a, tname_a
                keep_txt, keep_tname = txt_b, tname_b

            sids_to_clear.add(clear_sid)
            logger.warning(
                "Name-family dedup ({}): kept {!r}->{!r} (sim={:.2f}), "
                "cleared {!r}->{!r} (sim={:.2f})",
                family_type,
                keep_txt, keep_tname, max(sim_a, sim_b),
                clear_txt, clear_tname, min(sim_a, sim_b),
            )

    if sids_to_clear:
        for txt, cells in text_to_first_cells.items():
            sid = cells[0].subjectId if cells else None
            if sid and sid in sids_to_clear:
                for cell in cells:
                    cell.subjectId = None

def reject_income_statement_matches(
    text_to_first_cells: dict[str, list],
    subject_id_to_name: dict[str, str],
) -> None:
    """For cash flow tables, reject matches where the source text is clearly an income statement item."""
    for txt, cells in text_to_first_cells.items():
        sid = cells[0].subjectId if cells else None
        if not sid:
            continue
        template_name = subject_id_to_name.get(sid, "")
        if any(kw == txt or txt.startswith(kw) for kw in _INCOME_STATEMENT_ONLY_KEYWORDS):
            if "现金" not in template_name:
                for cell in cells:
                    cell.subjectId = None
                logger.warning(
                    "Cross-table rejection: income statement item {!r} should not match cash flow template {!r}",
                    txt, template_name,
                )

def resolve_aggregate_to_leaf(
    text_to_first_cells: dict[str, list],
    report: Report,
) -> None:
    """When a source text matched to an aggregate subject, try to resolve it to a leaf child.

    E.g. "应收账款" should match the leaf "应收账款", not the aggregate that sums
    "应收账款" + "应收账款净额".

    Resolution strategy (in order):
    1. Search aggregate's children (summaryRelation) that ARE in subjectList by exact name.
    2. When a child_id is NOT in subjectList, treat child_id itself as a potential subject
       name (some templates store names rather than IDs in summaryRelation) and check if
       it matches the source text – if so, look up the same name across all leaf subjects.
    3. Global fallback: scan ALL non-aggregate subjects in subjectList for an exact
       deep-normalized name match. This catches cases where a leaf subject exists but
       is not linked as a child of this aggregate in summaryRelation.
    """
    subject_by_id: dict[str, Subject] = {s.subjectId: s for s in report.subjectList}

    # Build a global leaf name→subject lookup for the fallback strategies.
    leaf_by_deep_name: dict[str, Subject] = {}
    for s in report.subjectList:
        if s.summaryRelation:  # skip aggregates
            continue
        dn = deep_normalize(s.subjectName.strip())
        if dn and dn not in leaf_by_deep_name:
            leaf_by_deep_name[dn] = s
        for syn in s.synonymSubject:
            dn_syn = deep_normalize(syn.strip())
            if dn_syn and dn_syn not in leaf_by_deep_name:
                leaf_by_deep_name[dn_syn] = s

    for txt, cells in text_to_first_cells.items():
        sid = cells[0].subjectId if cells else None
        if not sid:
            continue
        matched = subject_by_id.get(sid)
        if not matched or not matched.summaryRelation:
            continue

        deep_text = deep_normalize(txt)
        best_leaf: Optional[Subject] = None

        # Strategy 1 & 2: iterate summaryRelation children
        for child_id in matched.summaryRelation:
            child = subject_by_id.get(child_id)
            if child is None:
                # Strategy 2: child not in subjectList – try child_id as a name.
                # Some templates put subject names (not IDs) into summaryRelation.
                deep_child_id = deep_normalize(child_id.strip())
                if deep_child_id and deep_child_id == deep_text:
                    # The child's name matches the source text exactly.
                    # Try to find a leaf subject in subjectList with this name.
                    candidate = leaf_by_deep_name.get(deep_child_id)
                    if candidate:
                        best_leaf = candidate
                        break
                continue
            if deep_normalize(child.subjectName.strip()) == deep_text:
                best_leaf = child
                break
            for syn in child.synonymSubject:
                if deep_normalize(syn.strip()) == deep_text:
                    best_leaf = child
                    break
            if best_leaf:
                break

        # Strategy 3: global fallback – any leaf in subjectList with the same deep name.
        # Handles cases where the leaf exists in subjectList but is not explicitly listed
        # as a child of this aggregate (inconsistent template data).
        if not best_leaf:
            best_leaf = leaf_by_deep_name.get(deep_text)
            if best_leaf:
                logger.debug(
                    "Aggregate resolved via global leaf fallback: {!r} -> {}",
                    txt, best_leaf.subjectName,
                )

        if best_leaf:
            for cell in cells:
                cell.subjectId = best_leaf.subjectId
            logger.info(
                "Aggregate resolved to leaf: {!r}: {} (aggregate) -> {} (leaf)",
                txt, matched.subjectName, best_leaf.subjectName,
            )

def try_spread_to_leaves(
    texts: list[str],
    text_to_first_cells: dict[str, list],
    aggregate: Subject,
    subject_by_id: dict[str, Subject],
) -> bool:
    """When multiple source texts all landed on the same aggregate subject, try to assign
    each source text to a *distinct* leaf child of that aggregate.

    Returns True only when every source text has been successfully re-assigned to a
    different leaf, so the caller can skip the regular conflict-resolution logic.

    The matching uses deep-normalized name comparison (same logic as
    _resolve_aggregate_to_leaf), plus a lightweight similarity fallback for cases
    where names differ slightly (e.g. "应收账款" vs "应收账款净额").
    """
    # Build the pool of leaf children that are actually present in subjectList.
    available_leaves: list[Subject] = []
    for child_id in aggregate.summaryRelation:
        child = subject_by_id.get(child_id)
        if child is not None and not child.summaryRelation:
            available_leaves.append(child)

    if not available_leaves:
        return False  # No resolvable children – fall through to normal conflict resolution

    # Greedy assignment: try to pair each source text to the best-matching available leaf.
    assignments: dict[str, Subject] = {}  # source_text -> leaf Subject
    used_leaf_ids: set[str] = set()

    # First pass: exact deep-normalized name match.
    for txt in texts:
        deep_txt = deep_normalize(txt)
        for leaf in available_leaves:
            if leaf.subjectId in used_leaf_ids:
                continue
            if deep_normalize(leaf.subjectName.strip()) == deep_txt:
                assignments[txt] = leaf
                used_leaf_ids.add(leaf.subjectId)
                break
            if any(deep_normalize(syn.strip()) == deep_txt for syn in leaf.synonymSubject):
                assignments[txt] = leaf
                used_leaf_ids.add(leaf.subjectId)
                break

    # Second pass: similarity-based assignment for texts not yet assigned.
    unassigned = [t for t in texts if t not in assignments]
    unused_leaves = [lf for lf in available_leaves if lf.subjectId not in used_leaf_ids]
    for txt in unassigned:
        if not unused_leaves:
            break
        best_leaf = max(
            unused_leaves,
            key=lambda lf: SequenceMatcher(None, txt, lf.subjectName).ratio(),
        )
        sim = SequenceMatcher(None, txt, best_leaf.subjectName).ratio()
        if sim >= 0.5:  # Only assign if there is a reasonable similarity
            assignments[txt] = best_leaf
            used_leaf_ids.add(best_leaf.subjectId)
            unused_leaves.remove(best_leaf)

    # Only apply assignments if ALL source texts could be assigned to distinct leaves.
    if len(assignments) < len(texts):
        return False

    for txt, leaf in assignments.items():
        for cell in text_to_first_cells[txt]:
            cell.subjectId = leaf.subjectId
        logger.debug(
            "Spread to leaf: {!r} -> {} (was aggregate {})",
            txt, leaf.subjectName, aggregate.subjectName,
        )
    return True
