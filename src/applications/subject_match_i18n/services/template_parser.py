"""标准科目模板数据结构与查找表构建"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.applications.subject_match_i18n.utils.attributes import infer_subject_attributes
from src.applications.subject_match_i18n.utils.text_normalize import (
    normalize,
    strip_liquidity_prefix,
)


@dataclass
class TemplateSubject:
    code: str
    en_name: str
    level: int
    parent: str          # "-" 表示根节点
    is_leaf: bool
    industry: str        # 适用行业：ALL/MFG/RETAIL/AGRI/ENERGY 等
    sort: int            # 模板排序码
    synonyms: list[str] = field(default_factory=list)
    en_name_normalized: str = field(init=False)
    en_name_liquidity_bare: str = field(init=False)
    # 结构化语义维度：从 en_name 自动推断的属性字典，如
    # {"liquidity": "current"|"non-current", "direction": "receivable"|"payable"|...}
    # 用于匹配阶段的"维度一致性闸门"，替代散落的关键字黑名单。
    attributes: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.en_name_normalized = normalize(self.en_name)
        # 流动性裸名 key：仅当模板名首部含流动性前缀时非空。
        # 让 "Non-current lease liabilities" / "Current lease liabilities" 共享 key=leaseliabilities，
        # 配合 resolve 中的流动性过滤完成消歧。
        bare = strip_liquidity_prefix(self.en_name)
        bare_normalized = normalize(bare)
        self.en_name_liquidity_bare = (
            bare_normalized if bare_normalized and bare_normalized != self.en_name_normalized else ""
        )
        self.attributes = infer_subject_attributes(self.en_name)


def propagate_liquidity(subjects: list[TemplateSubject]) -> None:
    """
    沿 parent 链向下传播流动性属性。

    CURT_NONCURT 模板中流动性编码在父级名称里（如 "Non-current provisions"），
    叶子节点名称可能不含 current/non-current 关键词，导致属性闸门失效。
    按 level 升序处理，每个节点只查直接父级（父级已先于子级处理），
    等价于追踪到根节点。自身名称已能推断的不覆盖。
    """
    code_to_subject = {s.code: s for s in subjects}
    for s in sorted(subjects, key=lambda x: x.level):
        if "liquidity" in s.attributes:
            continue
        parent = code_to_subject.get(s.parent)
        if parent:
            parent_liq = parent.attributes.get("liquidity")
            if parent_liq:
                s.attributes["liquidity"] = parent_liq


def propagate_section(subjects: list[TemplateSubject]) -> None:
    """
    沿 parent 链向下传播报表段属性（cf_section / is_section）。

    段落信号（如 "Cash flows from operating activities"）通常在高层节点名称中，
    叶子节点名称不含段落关键词。按 level 升序处理，每个节点只查直接父级
    （父级已先于子级处理），等价于追踪到根节点。自身名称已能推断的不覆盖。
    """
    code_to_subject = {s.code: s for s in subjects}
    for s in sorted(subjects, key=lambda x: x.level):
        if "cf_section" in s.attributes or "is_section" in s.attributes:
            continue
        parent = code_to_subject.get(s.parent)
        if parent:
            parent_section = {
                k: v for k, v in parent.attributes.items()
                if k in ("cf_section", "is_section")
            }
            if parent_section:
                s.attributes.update(parent_section)


def build_lookup(subjects: list[TemplateSubject]) -> dict[str, list[TemplateSubject]]:
    """
    构建标准化名称 → 候选科目列表的查找表。
    除 normalize 后的原始 key 外，额外添加：
    - 流动性前缀剥离 key（使源裸名 "Lease liabilities" 召回 Non-current/Current 两个变体进入消歧）
    同一标准化名称可能对应多个科目（同名歧义/流动性变体场景，由 resolve 二次过滤）。
    """
    lookup: dict[str, list[TemplateSubject]] = {}
    for s in subjects:
        # 原始 normalize key
        for key in (s.en_name_normalized,):
            if key:
                lookup.setdefault(key, []).append(s)
        # 流动性裸名 key（仅模板名含流动性前缀时生效）
        bare_key = s.en_name_liquidity_bare
        if bare_key and bare_key not in (s.en_name_normalized,):
            existing = lookup.get(bare_key)
            if not (existing and any(sub.code == s.code for sub in existing)):
                lookup.setdefault(bare_key, []).append(s)
    return lookup
