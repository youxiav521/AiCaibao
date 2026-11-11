"""科目匹配通用工具函数"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from src.applications.subject_match_i18n.constants import MatchType
from src.applications.subject_match_i18n.services.template_parser import (
    TemplateSubject,
)
from src.applications.subject_match_i18n.schemas.match_schema import (
    TemplateSubjectInput,
)
from src.applications.subject_match_i18n.utils.attributes import (
    CF_OPERATING_HINTS,
    CF_INVESTING_HINTS,
    CF_FINANCING_HINTS,
    IS_OCI_HINTS,
    IS_PROFIT_HINTS,
    SECTION_MAX_DEPTH,
    check_attributes_compatible,
    infer_text_attributes,
)
from src.applications.subject_match_i18n.utils.text_normalize import normalize, strip_ifrs_qualifiers

_TOKEN_SORT_THRESHOLD = 0.7

_SKIP_LINE_TYPES = {"BLANK", "HEADER", "NOTE", "EMPTY"}
_EN_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "at",
    "on", "by", "with", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "not", "no", "but", "if", "so", "than", "that", "this", "these",
    "those", "it", "its", "their", "them", "they", "we", "us", "our",
    "he", "she", "him", "her", "his", "you", "your", "i", "me", "my",
    "which", "who", "whom", "what", "where", "when", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "too", "very", "just", "also",
    "any", "about", "up", "out", "into", "over", "after", "before",
    "between", "under", "above", "through", "during", "until",
    "against", "without", "within", "along", "across", "behind",
    "beyond", "plus", "except", "per", "via",
})

_PARENTHESES_CONTENT_RE = re.compile(r"\([^)]*\)")

# 流动性消歧关键词。判定顺序：先 non-current（避免被 "current" 子串误命中）。
_NONCURRENT_HINTS: tuple[str, ...] = ("non-current", "noncurrent", "non current", "long-term", "long term")
_CURRENT_HINT: str = "current"
_LIQUIDITY_PARENT_MAX_DEPTH: int = 8


def infer_line_liquidity(
    line: dict[str, Any],
    row_by_idx: dict[int, dict[str, Any]],
) -> str | None:
    """
    沿 inferredParent 链向上找最近的"流动性区块名"，返回 "non-current" / "current" / None。
    限深 8 + 防环。父链无信号时返回 None（中性，交由后续兜底）。
    """
    seen: set[int] = set()
    cur = line
    for _ in range(_LIQUIDITY_PARENT_MAX_DEPTH):
        pidx = cur.get("inferredParent")
        if pidx is None or pidx in seen:
            break
        seen.add(pidx)
        parent = row_by_idx.get(pidx)
        if parent is None:
            break
        name = (parent.get("rawName") or "").lower()
        if any(h in name for h in _NONCURRENT_HINTS):
            return "non-current"
        if _CURRENT_HINT in name:
            return "current"
        cur = parent
    return None


def infer_line_section(
    line: dict[str, Any],
    row_by_idx: dict[int, dict[str, Any]],
) -> dict[str, str]:
    """
    沿 inferredParent 链向上搜索（含 HEADER 行），推断源行的报表段归属。
    返回 {"cf_section": "operating"} 或 {"is_section": "profit"} 等，空 dict 表示无法推断（中性）。

    与 infer_line_liquidity 不同的是：此函数会查看所有行类型（含 HEADER），
    因为段归属信号通常在 HEADER 行的名称中。
    """
    seen: set[int] = set()
    cur = line
    for _ in range(SECTION_MAX_DEPTH):
        pidx = cur.get("inferredParent")
        if pidx is None or pidx in seen:
            break
        seen.add(pidx)
        parent = row_by_idx.get(pidx)
        if parent is None:
            break
        name = (parent.get("rawName") or "").lower()

        # CF 段落检测
        if any(h in name for h in CF_OPERATING_HINTS):
            return {"cf_section": "operating"}
        if any(h in name for h in CF_INVESTING_HINTS):
            return {"cf_section": "investing"}
        if any(h in name for h in CF_FINANCING_HINTS):
            return {"cf_section": "financing"}

        # IS 段落检测
        if any(h in name for h in IS_OCI_HINTS):
            return {"is_section": "oci"}
        if any(h in name for h in IS_PROFIT_HINTS):
            return {"is_section": "profit"}

        cur = parent
    return {}


def infer_line_attributes(
    line: dict[str, Any],
    row_by_idx: dict[int, dict[str, Any]],
) -> dict[str, str]:
    """
    源行结构化属性推断：
    - 流动性维度：仅取自父链区块名（沿用 infer_line_liquidity 的语义）；
    - 报表段归属（cf_section / is_section）：从父链 HEADER 行名称推断；
    - 其他方向类维度：从行自身名称推断（与模板侧规则一致）。
    """
    attrs: dict[str, str] = {}
    liquidity = infer_line_liquidity(line, row_by_idx)
    if liquidity is not None:
        attrs["liquidity"] = liquidity
    # 报表段归属：沿父链向上找最近的 HEADER 行，从名称关键词推断段
    section = infer_line_section(line, row_by_idx)
    attrs.update(section)
    raw_name = line.get("rawName") or line.get("cleanedName") or ""
    if raw_name:
        for dim, val in infer_text_attributes(raw_name).items():
            if dim == "liquidity":
                # 流动性以父链推断为准（行名未必出现 current/non-current）
                continue
            attrs[dim] = val
    return attrs


def _clean_for_contains(text: str) -> str:
    if not text:
        return ""
    result = text.lower().strip()
    result = re.sub(r"[,;:]", " ", result)
    result = re.sub(r"\s+", " ", result)
    return result


# 向后兼容别名：IFRS 限定词剥离已移至 text_normalize.strip_ifrs_qualifiers
_strip_ifrs_qualifiers = strip_ifrs_qualifiers


def _contains_match(query: str, candidate: str, min_ratio: float = 0.55) -> bool:
    if query in candidate:
        return len(query) >= len(candidate) * min_ratio
    if candidate in query:
        return len(candidate) >= len(query) * min_ratio
    return False


def _token_sort_similarity(a: str, b: str) -> float:
    return fuzz.token_sort_ratio(a, b) / 100.0


def _token_ordered_contains(query: str, candidate: str) -> bool:
    q_tokens = [t for t in query.lower().split() if len(t) > 1]
    c_tokens = [t for t in candidate.lower().split() if len(t) > 1]
    if not q_tokens or not c_tokens:
        return False
    if len(q_tokens) < 2:
        return False
    if len(q_tokens) > len(c_tokens):
        return False
    idx = 0
    for ct in c_tokens:
        if idx < len(q_tokens) and ct == q_tokens[idx]:
            idx += 1
    return idx == len(q_tokens)


def _compute_verify_score(query: str, candidate: str, norm_q: str, norm_c: str) -> float:
    """3c/3d 召回后的相似度验证，返回字符/token 级的多种相似度最大值。"""
    scores: list[float] = []
    if norm_q and norm_c:
        scores.append(fuzz.ratio(norm_q, norm_c) / 100.0)
        scores.append(_token_sort_similarity(norm_q, norm_c))
    scores.append(fuzz.ratio(query, candidate) / 100.0)
    scores.append(_token_sort_similarity(query, candidate))
    return max(scores)


def _fuzzy_pair_score(query_raw: str, candidate_raw: str) -> float:
    """
    召回阶段评分：仅做 3c/3d/token_sort==1.0 召回，不做硬阈值拦阻。
    返回裸分（0.0 表示无召回），由调用方按 embedding cos 做三级决策。
    """
    query = _strip_ifrs_qualifiers(_clean_for_contains(query_raw))
    candidate = _strip_ifrs_qualifiers(_clean_for_contains(candidate_raw))
    if not query or not candidate:
        return 0.0

    # 维度属性反向校验
    q_attrs = infer_text_attributes(query)
    c_attrs = infer_text_attributes(candidate)
    if not check_attributes_compatible(q_attrs, c_attrs):
        return 0.0

    if query == candidate:
        return 1.0

    norm_q = normalize(query)
    norm_c = normalize(candidate)

    # token 集合完全等价（词序无关）→ 满配召回
    if norm_q and norm_c and _token_sort_similarity(norm_q, norm_c) == 1.0:
        return 1.0

    # 召回触发：3c 字符子串 / 3d token 有序包含 / token_sort 在 0.7-1.0 区间
    triggered = (
        _contains_match(query, candidate)
        or _token_ordered_contains(query, candidate)
    )
    if not triggered and norm_q and norm_c:
        token_sort_sim = _token_sort_similarity(norm_q, norm_c)
        if 0.7 <= token_sort_sim < 1.0:
            triggered = True

    if triggered:
        return _compute_verify_score(query, candidate, norm_q, norm_c)

    return 0.0


def fuzzy_match(
    raw_key: str,
    clean_key: str,
    subjects: list[TemplateSubject],
    matched_codes: set[str],
) -> tuple[list[TemplateSubject], set[str]]:
    """
    模糊匹配（兼容旧接口）：调用 fuzzy_recall 并提取候选列表 + 同义词 code 集合。
    """
    scored = fuzzy_recall(raw_key, clean_key, subjects, matched_codes)
    subjects_list = [s for _, s, _ in scored]
    synonym_codes = {s.code for _, s, matched_via in scored if matched_via}
    return subjects_list, synonym_codes


def fuzzy_recall(
    raw_key: str,
    clean_key: str,
    subjects: list[TemplateSubject],
    matched_codes: set[str],
) -> list[tuple[float, TemplateSubject, bool]]:
    """
    模糊召回（不打硬阈值），返回 [(literal_score, subject, matched_via_synonym), ...]
    literal_score 是字符级 fuzz 验证分数，由调用方结合 embedding cos 做三层分级。
    """
    scored: list[tuple[float, TemplateSubject, bool]] = []
    seen_codes: set[str] = set()
    for s in subjects:
        if s.code in matched_codes or s.code in seen_codes:
            continue
        best_for_subject = 0.0
        matched_via_synonym = False

        for candidate_text in (s.en_name,):
            if not candidate_text or candidate_text == "-":
                continue
            for query_text in (raw_key, clean_key):
                if not query_text:
                    continue
                score = _fuzzy_pair_score(query_text, candidate_text)
                if score > best_for_subject:
                    best_for_subject = score
                    matched_via_synonym = False

        for alias in s.synonyms:
            if not alias:
                continue
            for query_text in (raw_key, clean_key):
                if not query_text:
                    continue
                score = _fuzzy_pair_score(query_text, alias)
                if score > best_for_subject:
                    best_for_subject = score
                    matched_via_synonym = True

        if best_for_subject > 0:
            scored.append((best_for_subject, s, matched_via_synonym))
            seen_codes.add(s.code)

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def build_synonym_lookup(
    subjects: list[TemplateSubject],
) -> dict[str, set[str]]:
    """
    构建同义词标准化名称 → 等价科目 code 集合 的查找表。

    模型动机（借鉴 XBRL Versioning 的 ConceptRename 等价类）：
    若两个科目 A、B 共享同一个 alias（如 "Cash"），它们对该 alias 是等价的；
    传递闭包后形成等价集合。原"一对一覆盖式"实现会丢失先写入的科目，
    本实现保留等价集合内全部科目，匹配阶段由 exact_match 转交 resolve 上下文消歧。

    使用 Union-Find 把"共享 alias"视为等价关系：以 alias 为锚点，把
    所有指向同一 alias 的 subject code union 在一起；最终 alias 映射到该集合所有代表元素。
    """
    # alias → 出现过的 codes（先收集）
    alias_to_codes: dict[str, list[str]] = {}
    for subject in subjects:
        if not subject.synonyms:
            continue
        for raw_alias in subject.synonyms:
            key = normalize(raw_alias)
            if not key:
                continue
            alias_to_codes.setdefault(key, []).append(subject.code)

    # Union-Find
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    all_codes = {s.code for s in subjects}
    for code in all_codes:
        parent[code] = code
    for codes in alias_to_codes.values():
        if len(codes) > 1:
            head = codes[0]
            for c in codes[1:]:
                union(head, c)

    # 等价根 → 集合内所有 code
    root_to_set: dict[str, set[str]] = {}
    for code in all_codes:
        root_to_set.setdefault(find(code), set()).add(code)

    # alias → 等价集合（取该 alias 任一 code 所在等价集合）
    lookup: dict[str, set[str]] = {}
    for alias, codes in alias_to_codes.items():
        equivalence_set = root_to_set.get(find(codes[0]), set(codes))
        lookup[alias] = equivalence_set
    return lookup


def build_next_effective_line_index(
    lines: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    预计算 rowIndex → 下一有效行的映射，O(n)。
    有效行指 lineType 不在 _SKIP_LINE_TYPES 中且有 rowIndex 的行。
    """
    sorted_lines = sorted(
        (line for line in lines if line.get("rowIndex") is not None),
        key=lambda l: l["rowIndex"],
    )
    next_effective: dict[int, dict[str, Any]] = {}
    for i in range(len(sorted_lines) - 1):
        current_idx = sorted_lines[i]["rowIndex"]
        for j in range(i + 1, len(sorted_lines)):
            candidate = sorted_lines[j]
            if candidate.get("lineType", "ITEM") not in _SKIP_LINE_TYPES:
                next_effective[current_idx] = candidate
                break
    return next_effective


def filter_candidates_by_line_level(
    line: dict[str, Any],
    candidates: list[TemplateSubject],
    next_effective: dict[int, dict[str, Any]],
) -> list[TemplateSubject]:
    """
    用 indentLevel 缩小候选：下一有效行 level 更深 → 当前行有子级 → 优先非叶子；否则优先叶子。
    SUBTOTAL/TOTAL 行直接优先非叶子节点。
    """
    if len(candidates) <= 1:
        return candidates

    line_type = line.get("lineType", "ITEM")
    if line_type in ("SUBTOTAL", "TOTAL"):
        non_leaf = [c for c in candidates if not c.is_leaf]
        if non_leaf:
            return non_leaf

    current_level = line.get("indentLevel", 1)
    current_row_idx = line.get("rowIndex")
    if current_row_idx is None:
        return candidates

    next_line = next_effective.get(current_row_idx)
    if next_line is not None and next_line.get("indentLevel", 1) > current_level:
        filtered = [c for c in candidates if not c.is_leaf]
    else:
        filtered = [c for c in candidates if c.is_leaf]
    return filtered if filtered else candidates


def exact_match(
    raw_key: str,
    clean_key: str,
    lookup: dict[str, list[TemplateSubject]],
    synonym_lookup: dict[str, set[str]],
    code_to_subject: dict[str, TemplateSubject],
    global_matched_codes: set[str],
) -> tuple[list[TemplateSubject], str]:
    """
    精确匹配：标准名称 lookup 与同义词 lookup 合并召回，交由 resolve 消歧。
    key 已在调用方做过 IFRS 剥离 + normalize。
    查找顺序：raw_key -> clean_key。
    排除已占用科目。返回 (候选列表, 匹配来源)，来源为 "synonym" | "exact" | ""。

    合并而非 early-return：当源名同时是某科目的标准名和另一科目的同义词时，
    两者都应作为候选进入 resolve，避免同义词单候选直接命中绕过消歧。
    """
    for key in (raw_key, clean_key):
        if not key:
            continue

        candidates: dict[str, TemplateSubject] = {}
        has_exact = False

        # 标准名称 lookup
        found = lookup.get(key)
        if found:
            for s in found:
                if s.code not in global_matched_codes:
                    candidates[s.code] = s
                    has_exact = True

        # 同义词 lookup（补充标准 lookup 未覆盖的候选）
        synonym_codes = synonym_lookup.get(key)
        if synonym_codes:
            for code in synonym_codes:
                if code not in global_matched_codes and code in code_to_subject:
                    candidates.setdefault(code, code_to_subject[code])

        if candidates:
            result = sorted(candidates.values(), key=lambda s: s.code)
            source = MatchType.EXACT if has_exact else MatchType.SYNONYM
            return result, source

    return [], ""




def convert_template_input(inputs: list[TemplateSubjectInput]) -> list[TemplateSubject]:
    return [
        TemplateSubject(
            code=item.code,
            en_name=item.enName,
            level=item.level,
            parent=item.parent,
            is_leaf=item.isLeaf,
            industry=item.industry,
            sort=item.sort,
            synonyms=item.synonyms,
        )
        for item in inputs
    ]
