"""
结构化语义维度（attributes）：替代散落的互斥关键字硬编码。

设计动机：
- 流动性（current / non-current）、方向（receivable / payable / income / expense / gain / loss /
  surplus / deficit / asset / liability）等"互斥维度"，
  在原实现中分散在 _EXCLUSIVE_PAIRS_EN 关键字对、infer_line_liquidity、infer_subject_liquidity 等多处。
- 这里把它们统一抽象为结构化属性 dict，匹配阶段通过 check_attributes_compatible 一处校验，
  规则集中、可测、可扩展（新增维度只需在此模块加规则）。
- 兼容性：当某维度的值为 None / 未推断出，视为"中性"，与任何值兼容（保守策略，避免误杀）。

收录原则（重要）：
- 只收录"两侧成对出现即语义对立"的维度。
- 不收录 net/gross、operating/investing/financing 等高频但歧义的词
  ——"Net assets"、"Operating lease liabilities"、"Other net" 等大量中性用法会被强行打标，
  比旧的"成对出现才否决"逻辑更激进，容易误杀。
"""

from __future__ import annotations

import re


# 流动性：判定顺序敏感（先 non-current 防止被 current 子串误命中）
_LIQUIDITY_NONCURRENT_HINTS: tuple[str, ...] = ("non-current", "noncurrent", "non current", "long-term", "long term")
_LIQUIDITY_CURRENT_HINTS: tuple[str, ...] = ("current", "short-term", "short term")

# 缓存已编译的关键字词边界 pattern
_KEYWORD_CACHE: dict[str, re.Pattern] = {}


def _has_keyword(text_lower: str, kw: str) -> bool:
    """
    词边界匹配关键字，兼容复数形式（payable 匹配 payables）。
    使用 (?<!\\w) / (?!\\w) 替代 \\b，避免 \\b 在 'e'->'s' 之间失效。
    末尾 s? 使单复数通用。
    """
    pat = _KEYWORD_CACHE.get(kw)
    if pat is None:
        pat = re.compile(rf"(?<!\w){re.escape(kw)}s?(?!\w)")
        _KEYWORD_CACHE[kw] = pat
    return bool(pat.search(text_lower))

# 方向类维度：每个维度内的关键字互斥，跨维度独立。
# 仅收录"成对出现即对立"且歧义低的维度；net/gross、cf_section 等过宽词类不收录。
_DIRECTION_RULES: list[tuple[str, dict[str, tuple[str, ...]]]] = [
    ("ar_ap", {
        "receivable": ("receivable",),
        "payable": ("payable",),
    }),
    ("pnl", {
        # 注意 revenue / income 都归为 income 一侧
        "income": ("income", "revenue"),
        "expense": ("expense",),
    }),
    ("gain_loss", {
        "gain": ("gain",),
        "loss": ("loss",),
    }),
    ("surplus_deficit", {
        "surplus": ("surplus",),
        "deficit": ("deficit",),
    }),
    ("balance_side", {
        "asset": ("asset", "assets"),
        "liability": ("liability", "liabilities"),
    }),
]


def _detect_liquidity(text_lower: str) -> str | None:
    if any(h in text_lower for h in _LIQUIDITY_NONCURRENT_HINTS):
        return "non-current"
    if any(h in text_lower for h in _LIQUIDITY_CURRENT_HINTS):
        return "current"
    return None


# 汇总/其它维度：非对称约束，单独处理（不进 _DIRECTION_RULES）。
# - "other" 两侧必须严格对齐（一侧 other、另一侧非 other 即否决）
# - "total"/"subtotal" 仅与 "other" 互斥，与中性兼容
_AGGREGATION_TOTAL_KEYWORDS: tuple[str, ...] = ("total", "subtotal")
_AGGREGATION_OTHER_KEYWORDS: tuple[str, ...] = ("other",)


def _detect_aggregation(text_lower: str) -> str | None:
    if any(_has_keyword(text_lower, kw) for kw in _AGGREGATION_OTHER_KEYWORDS):
        return "other"
    if any(_has_keyword(text_lower, kw) for kw in _AGGREGATION_TOTAL_KEYWORDS):
        return "total"
    return None


def _detect_directions(text_lower: str) -> dict[str, str]:
    """对每个维度独立判定；命中多个值时不记录该维度（避免歧义）。"""
    result: dict[str, str] = {}
    for dim_name, value_to_keywords in _DIRECTION_RULES:
        hit_values: list[str] = []
        for value, keywords in value_to_keywords.items():
            if any(_has_keyword(text_lower, kw) for kw in keywords):
                hit_values.append(value)
        if len(hit_values) == 1:
            result[dim_name] = hit_values[0]
    return result


# 报表段归属：从科目名称推断（与源行侧 infer_line_section 共用关键词）。
# CF: Operating/Investing/Financing 互斥；IS: Profit/OCI 互斥。
# 模板侧通过 propagate_section 沿 parent 链传播；源行侧沿 inferredParent 链推断。
CF_OPERATING_HINTS: tuple[str, ...] = (
    "cash flows from operating", "cash generated from operating",
    "net cash from operating", "cash flow from operating",
    "operating activities",
)
CF_INVESTING_HINTS: tuple[str, ...] = (
    "cash flows from investing", "cash used in investing",
    "net cash from investing", "cash flow from investing",
    "investing activities",
)
CF_FINANCING_HINTS: tuple[str, ...] = (
    "cash flows from financing", "cash used in financing",
    "net cash from financing", "cash flow from financing",
    "financing activities",
)
IS_OCI_HINTS: tuple[str, ...] = (
    "other comprehensive income", "comprehensive income attributable",
    "items of other comprehensive",
)
IS_PROFIT_HINTS: tuple[str, ...] = (
    "profit for the year", "profit attributable to",
    "profit for the period", "net profit",
)

SECTION_MAX_DEPTH: int = 10


def detect_section_from_name(name: str) -> dict[str, str]:
    """从科目名称推断报表段归属。返回空 dict 表示无法推断（中性）。"""
    if not name:
        return {}
    name_lower = name.lower()
    if any(h in name_lower for h in CF_OPERATING_HINTS):
        return {"cf_section": "operating"}
    if any(h in name_lower for h in CF_INVESTING_HINTS):
        return {"cf_section": "investing"}
    if any(h in name_lower for h in CF_FINANCING_HINTS):
        return {"cf_section": "financing"}
    if any(h in name_lower for h in IS_OCI_HINTS):
        return {"is_section": "oci"}
    if any(h in name_lower for h in IS_PROFIT_HINTS):
        return {"is_section": "profit"}
    return {}


def infer_subject_attributes(en_name: str) -> dict[str, str]:
    """
    从模板科目英文名称推断结构化属性。空名返回空 dict。

    报表段归属（cf_section / is_section）从此处的名称检测初始化，
    再由 propagate_section 沿 parent 链向下传播到子节点。
    """
    if not en_name:
        return {}
    text_lower = en_name.lower()
    attrs: dict[str, str] = {}
    liquidity = _detect_liquidity(text_lower)
    if liquidity is not None:
        attrs["liquidity"] = liquidity
    aggregation = _detect_aggregation(text_lower)
    if aggregation is not None:
        attrs["aggregation"] = aggregation
    attrs.update(_detect_directions(text_lower))
    attrs.update(detect_section_from_name(en_name))
    return attrs


def infer_text_attributes(text: str) -> dict[str, str]:
    """从任意文本（如源行名称）推断属性，规则与模板侧一致。"""
    return infer_subject_attributes(text)


def check_attributes_compatible(
    line_attrs: dict[str, str],
    subject_attrs: dict[str, str],
) -> bool:
    """
    维度一致性闸门：仅当两侧都明确给出且不同值时返回 False（反向，否决匹配）。
    任一侧未给出某维度（""/None/缺失）视为中性，保持兼容。

    特例：aggregation 维度采用非对称严格约束——
    - 一侧为 "other" 而另一侧不是 "other"（含中性）→ 否决
    - "total" 与中性兼容；"total" 与 "other" 互斥（由通用规则覆盖）
    """
    if not line_attrs or not subject_attrs:
        # aggregation=other 的单侧出现也要否决，不能因对侧空 dict 直接放行
        line_agg = (line_attrs or {}).get("aggregation")
        subj_agg = (subject_attrs or {}).get("aggregation")
        if (line_agg == "other") != (subj_agg == "other"):
            return False
        return True

    line_agg = line_attrs.get("aggregation")
    subj_agg = subject_attrs.get("aggregation")
    if (line_agg == "other") != (subj_agg == "other"):
        return False

    for dim, line_val in line_attrs.items():
        if not line_val:
            continue
        subj_val = subject_attrs.get(dim)
        if subj_val and subj_val != line_val:
            return False
    return True
