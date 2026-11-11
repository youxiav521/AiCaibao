"""科目匹配公共常量"""

from __future__ import annotations


class MatchType:
    """匹配方式标识，写入 MatchResultItem.matchType，对外契约。"""
    SYNONYM = "synonym"
    EXACT = "exact"
    FUZZY = "fuzzy"
    SYNONYM_FUZZY = "synonym_fuzzy"
    LLM_FALLBACK = "llm_fallback"
