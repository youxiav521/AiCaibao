"""Level 2 上下文消歧：当精确匹配到多个候选时，利用 lines 结构信息缩小到唯一结果"""

from __future__ import annotations

from typing import Any, Optional

from src.applications.subject_match_i18n.services.template_parser import TemplateSubject
from src.applications.subject_match_i18n.utils.attributes import check_attributes_compatible
from src.applications.subject_match_i18n.utils.match_utils import _fuzzy_pair_score, infer_line_attributes
from src.applications.subject_match_i18n.utils.text_normalize import normalize, strip_ifrs_qualifiers

_AGGREGATE_LINE_TYPES = {"SUBTOTAL", "TOTAL"}

# 评分收敛层：分差低于此阈值时视为不确定，返回 None 交给 LLM
_SCORE_GAP_THRESHOLD = 0.15


def resolve(
    line: dict[str, Any],
    candidates: list[TemplateSubject],
    matched_by_row: dict[int, TemplateSubject],
    descendant_index: Optional[dict[str, set[str]]] = None,
    row_by_idx: Optional[dict[int, dict[str, Any]]] = None,
) -> Optional[TemplateSubject]:
    """
    从多个同名候选中，利用上下文信息选出唯一匹配。
    返回 None 表示无法消歧，交由 LLM 兜底。

    消歧规则（按优先级）：
    0. 排除已被其他行占用的候选编码
    0.5. 维度属性对齐（强约束）：源行 attributes 与候选 attributes 反向直接排除
    1. 祖先科目约束：inferredParent 行已匹配 → 候选必须是该科目的后代（含跨级）
    2. 相邻科目约束：precedingName/followingName 已匹配 → 候选落在邻居父节点的子树内（即兄弟/侄子）
    3. 行类型辅助：SUBTOTAL/TOTAL 倾向高层级（is_leaf=False），ITEM 倾向叶子节点
    4. indentLevel ↔ level 对齐：唯一最接近者返回
    5. 评分收敛层：多因子加权打分，top1 与 top2 分差 ≥ 阈值时返回 top1
    """
    if len(candidates) == 1:
        return candidates[0]

    # 规则 0：排除已被其他行占用的候选
    occupied_codes = {s.code for s in matched_by_row.values()}
    available = [c for c in candidates if c.code not in occupied_codes]
    if not available:
        return None
    if len(available) == 1:
        return available[0]
    candidates = available

    # 规则 0.5：维度属性对齐（统一闸门，含流动性/方向/段归属等）
    if row_by_idx is not None:
        line_attrs = infer_line_attributes(line, row_by_idx)
        if line_attrs:
            compatible = [
                c for c in candidates
                if check_attributes_compatible(line_attrs, c.attributes)
            ]
            # strict：候选 attributes 必须在所有维度都与 line_attrs 完全一致（非中性）
            strict = [
                c for c in compatible
                if all(c.attributes.get(dim) == val for dim, val in line_attrs.items() if val)
            ]
            if len(strict) == 1:
                return strict[0]
            if strict:
                candidates = strict
            elif compatible:
                candidates = compatible
            # 全部反向：不放弃，保留原 candidates 走后续规则

    # 规则 1：祖先科目约束（后代，不限于直接子节点）
    inferred_parent_row: Optional[int] = line.get("inferredParent")
    if inferred_parent_row is not None and descendant_index is not None:
        parent_subject = matched_by_row.get(inferred_parent_row)
        if parent_subject is not None:
            parent_descendants = descendant_index.get(parent_subject.code, set())
            parent_filtered = [c for c in candidates if c.code in parent_descendants]
            if len(parent_filtered) == 1:
                return parent_filtered[0]
            if parent_filtered:
                candidates = parent_filtered

    # 规则 2：相邻科目约束（候选应落在已匹配邻居行直接父节点的子树内，即兄弟/侄子关系）
    if descendant_index is not None and row_by_idx is not None:
        name_to_subject = _build_matched_name_index(matched_by_row, row_by_idx)
        sibling_scope_codes: set[str] = set()
        for neighbor_name in (line.get("precedingName"), line.get("followingName")):
            if not neighbor_name:
                continue
            neighbor_subject = name_to_subject.get(normalize(neighbor_name))
            if neighbor_subject and neighbor_subject.parent:
                # 邻居父节点的所有后代 = 邻居的兄弟及其子孙
                sibling_scope_codes.update(
                    descendant_index.get(neighbor_subject.parent, set())
                )

        if sibling_scope_codes:
            sibling_filtered = [c for c in candidates if c.code in sibling_scope_codes]
            if len(sibling_filtered) == 1:
                return sibling_filtered[0]
            if sibling_filtered:
                candidates = sibling_filtered

    # 规则 3：行类型辅助
    line_type: str = line.get("lineType", "ITEM")
    if line_type in _AGGREGATE_LINE_TYPES:
        non_leaf = [c for c in candidates if not c.is_leaf]
        if len(non_leaf) == 1:
            return non_leaf[0]
    else:
        leaf = [c for c in candidates if c.is_leaf]
        if len(leaf) == 1:
            return leaf[0]

    # 规则 3.5：精确名称匹配优先
    # normalize 的单复数归一可能导致不同名称碰撞到同一 key（如 "Tax paid" vs "Taxes paid"），
    # 此时优先选择 en_name 与源行名（strip_ifrs_qualifiers 后，未经 normalize）完全一致的候选。
    raw_name = line.get("rawName") or line.get("cleanedName") or ""
    if raw_name:
        source_name = strip_ifrs_qualifiers(raw_name)
        exact_name_matches = [c for c in candidates if strip_ifrs_qualifiers(c.en_name) == source_name]
        if len(exact_name_matches) == 1:
            return exact_name_matches[0]
        if exact_name_matches:
            candidates = exact_name_matches

    # 规则 4：indentLevel ↔ level 对齐（唯一最接近者返回）
    indent_level = line.get("indentLevel", 1)
    level_distances = [(c, abs(c.level - indent_level)) for c in candidates]
    min_dist = min(d for _, d in level_distances)
    closest = [c for c, d in level_distances if d == min_dist]
    if len(closest) == 1:
        return closest[0]

    # 规则 5：评分收敛层——多因子加权打分
    return _score_candidates(line, candidates, row_by_idx)


def _score_candidates(
    line: dict[str, Any],
    candidates: list[TemplateSubject],
    row_by_idx: Optional[dict[int, dict[str, Any]]],
) -> Optional[TemplateSubject]:
    """
    评分收敛层：多因子加权打分，选出最优候选。

    打分维度（保守策略）：
    - 段归属对齐：源行与候选段一致 → 加分，冲突 → 扣分，任一侧中性 → 不影响
    - indentLevel ↔ level 对齐：距离越小越好
    - 字面相似度：rawName 与候选 en_name 的 fuzzy score
    - 叶子对齐：lineType 与 is_leaf 一致 → 加分

    收敛判据：top1 与 top2 分差 ≥ _SCORE_GAP_THRESHOLD 时返回 top1，否则 None。
    """
    indent_level = line.get("indentLevel", 1)
    line_type = line.get("lineType", "ITEM")
    raw_name = line.get("rawName") or line.get("cleanedName") or ""

    # 源行段归属
    line_attrs: dict[str, str] = {}
    if row_by_idx is not None:
        line_attrs = infer_line_attributes(line, row_by_idx)

    scored: list[tuple[float, TemplateSubject]] = []
    for c in candidates:
        score = 0.0

        # 因子 1：段归属对齐（权重 0.30）
        # 反向否决、同向加分、缺失中性
        section_score = _section_alignment_score(line_attrs, c.attributes)
        score += section_score * 0.30

        # 因子 2：indentLevel ↔ level 对齐（权重 0.25）
        # 归一化：距离 0 → 1.0，距离 ≥ 4 → 0.0
        level_dist = abs(c.level - indent_level)
        level_score = max(0.0, 1.0 - level_dist / 4.0)
        score += level_score * 0.25

        # 因子 3：字面相似度（权重 0.15）
        # rawName 与候选 en_name 的 fuzzy score
        if raw_name and c.en_name:
            fuzzy = _fuzzy_pair_score(raw_name, c.en_name)
            score += fuzzy * 0.15

        # 因子 4：叶子对齐（权重 0.10）
        prefer_leaf = line_type not in _AGGREGATE_LINE_TYPES
        leaf_aligned = (c.is_leaf == prefer_leaf)
        score += (1.0 if leaf_aligned else 0.0) * 0.10

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    if len(scored) == 1:
        return scored[0][1]

    top1_score, top1 = scored[0]
    top2_score, _ = scored[1]

    if top1_score - top2_score >= _SCORE_GAP_THRESHOLD:
        return top1

    return None


def _section_alignment_score(line_attrs: dict[str, str], subject_attrs: dict[str, str]) -> float:
    """
    段归属对齐评分：反向否决(-1)、同向加分(+1)、缺失中性(0)。
    与 check_attributes_compatible 的保守策略一致——
    任一侧推断不出段归属时不扣分，避免误杀模板侧本就没标段的中性科目。
    """
    score = 0.0
    # cf_section 维度
    line_cf = line_attrs.get("cf_section", "")
    subj_cf = subject_attrs.get("cf_section", "")
    if line_cf and subj_cf:
        if line_cf == subj_cf:
            score += 1.0
        else:
            score -= 1.0

    # is_section 维度
    line_is = line_attrs.get("is_section", "")
    subj_is = subject_attrs.get("is_section", "")
    if line_is and subj_is:
        if line_is == subj_is:
            score += 1.0
        else:
            score -= 1.0

    # 归一化到 [−1, 1]：最多 2 个维度，每个 ±1
    dim_count = (1 if line_cf and subj_cf else 0) + (1 if line_is and subj_is else 0)
    if dim_count == 0:
        return 0.0
    return score / dim_count


def build_descendant_index(all_subjects: list[TemplateSubject]) -> dict[str, set[str]]:
    """
    构建 code → 所有后代 code 集合的索引（含自身）。
    报告层级和模板层级可能不对齐，用后代约束代替直接子节点约束。
    """
    children: dict[str, list[str]] = {}
    for s in all_subjects:
        children.setdefault(s.parent, []).append(s.code)

    index: dict[str, set[str]] = {}
    for s in all_subjects:
        if s.code in index:
            continue
        _collect_descendants(s.code, children, index)
    return index


def _collect_descendants(
    code: str,
    children: dict[str, list[str]],
    cache: dict[str, set[str]],
) -> set[str]:
    if code in cache:
        return cache[code]
    descendants: set[str] = {code}
    for child_code in children.get(code, []):
        descendants.update(_collect_descendants(child_code, children, cache))
    cache[code] = descendants
    return descendants


def _build_matched_name_index(
    matched_by_row: dict[int, TemplateSubject],
    row_by_idx: dict[int, dict[str, Any]],
) -> dict[str, TemplateSubject]:
    """构建已匹配行的 标准化行名 → 匹配科目 索引。
    precedingName/followingName 是源行名称，必须以源行名（而非模板标准名）为 key，
    否则邻居行通过 synonym 匹配时（行名 ≠ 模板名）查不到。
    """
    index: dict[str, TemplateSubject] = {}
    for row_idx, subject in matched_by_row.items():
        line = row_by_idx.get(row_idx, {})
        for name in (line.get("rawName"), line.get("cleanedName"), subject.en_name):
            if not name:
                continue
            key = normalize(name)
            if key:
                index.setdefault(key, subject)
    return index
