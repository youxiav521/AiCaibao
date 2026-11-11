from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from src.applications.parsing.utils.table_classification.common import (
    STATEMENT_KEYWORDS,
    compact_text,
)
from src.utils.chinese_converter import traditional_to_simplified

DERIVATIVE_TAIL_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_TITLE_LEADING_NUMBERING_RE = re.compile(
    r"^\s*(?:第\s*[一二三四五六七八九十百千万\d]+\s*[章节篇]\s*"
    r"|[（(]?[一二三四五六七八九十百千万\d]+(?:\.\d+)*[）)、.．、]?\s*)"
)
_TITLE_EDGE_PUNCT_RE = re.compile(
    r"^[：:、，。；;！!？?·．.\-—_（(【\[]+|[：:、，。；;！!？?·．.\-—_）)】\]]+$"
)
_TITLE_TRAILING_PAREN_RE = re.compile(r"[（(][^）)]*[）)]$")
NON_STATEMENT_EXACT_TITLES: frozenset[str] = frozenset({"现金流"})
NON_FINANCIAL_TITLE_TERMS: Tuple[str, ...] = (
    "综合收益",
    "收入",
    "支出",
    "构成", "介绍", "概述", "概况",
    "客户", "供应商",
    "排名", "名次", "竞争",
    "核心技术", "研发投入",
    "坏账准备", "长期股权投资", "递延所得税", "应付债券",
    "合同资产", "合同负债", "存货", "预付款项", "其他应收款",
    "商誉", "无形资产",
    "应付职工薪酬", "短期薪酬", "离职后福利",
    "联营", "合营", "子公司投资",
    "非经常性损益", "主营业务", "公允价值",
    "独立董事", "董事会", "监事会", "审计委员", "薪酬与提名",
    "担保", "会计师事务所",
    "持股", "限售股份", "总股本",
    "关键审计事项",
    "释义",
    "股东权益变动", "所有者权益变动", "权益变动",
)


def is_derivative_statement_title(text: Any) -> bool:
    if not isinstance(text, str):
        return False

    stripped = traditional_to_simplified(text.strip())
    if not stripped:
        return False

    if not any(keyword in stripped for keyword in STATEMENT_KEYWORDS):
        return False

    compact = compact_text(stripped)
    derivative_markers: Tuple[str, ...] = (
        "项目注释",
        "项目注解",
        "项目附注",
        "报表附注",
        "报表注释",
        "项目说明",
        "变动说明",
        "情况说明",
        "明细说明",
        "项目变动",
        "变动情况",
        "明细情况",
        "明细表",
        "分析表",
        "构成表",
        "附注",
        "注释",
        "注解",
        "说明",
        "明细",
        "构成",
    )
    return any(marker in compact for marker in derivative_markers)


def normalize_heading_title_for_exact_match(text: str) -> str:
    normalized = traditional_to_simplified(text.strip())
    if not normalized:
        return ""
    while True:
        stripped = _TITLE_LEADING_NUMBERING_RE.sub("", normalized, count=1)
        if stripped == normalized:
            break
        normalized = stripped
    normalized = compact_text(normalized)
    while True:
        stripped = _TITLE_TRAILING_PAREN_RE.sub("", normalized, count=1)
        if stripped == normalized:
            break
        normalized = compact_text(stripped)
    while True:
        stripped = _TITLE_EDGE_PUNCT_RE.sub("", normalized, count=1)
        if stripped == normalized:
            break
        normalized = stripped
    return normalized


def is_non_statement_exact_title(label: str) -> bool:
    if not isinstance(label, str) or not label.strip():
        return False
    return normalize_heading_title_for_exact_match(label) in NON_STATEMENT_EXACT_TITLES


def is_cash_flow_summary_title(label: str) -> bool:
    return is_non_statement_exact_title(label)


def is_non_financial_title(label: str) -> bool:
    if not isinstance(label, str) or not label.strip():
        return False
    compact = "".join(label.split())
    return any(term in compact for term in NON_FINANCIAL_TITLE_TERMS)


def is_full_zh_statement_table_md(table_md: str, label: str) -> bool:
    compact = compact_text(traditional_to_simplified(table_md or ""))
    if not compact:
        return False

    if "资产负债表" in label:
        asset_total_terms = ("资产总计", "资产合计")
        liability_equity_total_terms = (
            "负债和所有者权益总计",
            "负债及所有者权益总计",
            "负债和所有者权益合计",
            "负债及所有者权益合计",
            "负债和所有者权益（或股东权益）总计",
            "负债及所有者权益（或股东权益）总计",
            "负债和所有者权益（或股东权益）合计",
            "负债及所有者权益（或股东权益）合计",
        )
        has_asset_total = any(term in compact for term in asset_total_terms)
        has_liability_equity_total = any(term in compact for term in liability_equity_total_terms)
        has_liability_total = "负债合计" in compact
        has_equity_total = "所有者权益合计" in compact or "股东权益合计" in compact
        return has_asset_total and (has_liability_equity_total or (has_liability_total and has_equity_total))

    if "利润表" in label:
        revenue_terms = ("营业收入", "营业总收入", "主营业务收入")
        net_profit_terms = ("净利润", "净亏损")
        extra_terms = ("利润总额", "每股收益", "综合收益总额", "营业利润")
        return (
            any(term in compact for term in revenue_terms)
            and any(term in compact for term in net_profit_terms)
            and any(term in compact for term in extra_terms)
        )

    if "现金流量表" in label:
        required_terms = (
            "经营活动产生的现金流量",
            "投资活动产生的现金流量",
            "筹资活动产生的现金流量",
            "期末现金及现金等价物余额",
        )
        net_increase_terms = (
            "现金及现金等价物净增加额",
            "现金及现金等价物的净增加额",
            "现金及现金等价物净增加",
            "现金及现金等价物的净增加",
        )
        return any(term in compact for term in required_terms) and any(term in compact for term in net_increase_terms)

    return False


def detect_statement_type_by_content_zh(table_md: str) -> Optional[str]:
    compact = compact_text(traditional_to_simplified(table_md or ""))
    if not compact:
        return None

    balance_terms = (
        "流动资产合计",
        "非流动资产合计",
        "资产总计",
        "资产合计",
        "流动负债合计",
        "非流动负债合计",
        "负债合计",
        "所有者权益合计",
        "股东权益合计",
        "负债和所有者权益总计",
        "负债及所有者权益总计",
        "负债和所有者权益合计",
        "负债及所有者权益合计",
    )
    profit_terms = (
        "营业总收入",
        "营业收入",
        "营业总成本",
        "营业成本",
        "利润总额",
        "净利润",
        "净亏损",
        "综合收益总额",
        "每股收益",
        "主营业务收入",
        "主营业务利润",
        "营业利润",
    )
    cash_terms = (
        "经营活动产生的现金流量",
        "投资活动产生的现金流量",
        "筹资活动产生的现金流量",
        "经营活动现金流入小计",
        "经营活动现金流出小计",
        "现金及现金等价物净增加额",
        "现金及现金等价物的净增加额",
        "期末现金及现金等价物余额",
    )

    balance_score = sum(1 for term in balance_terms if term in compact)
    profit_score = sum(1 for term in profit_terms if term in compact)
    cash_score = sum(1 for term in cash_terms if term in compact)

    best_score = max(balance_score, profit_score, cash_score)
    if best_score >= 4:
        if balance_score >= 4 and balance_score > max(profit_score, cash_score):
            return "资产负债表"
        if cash_score >= 4 and cash_score > max(balance_score, profit_score):
            return "现金流量表"
        if profit_score >= 4 and profit_score > max(balance_score, cash_score):
            return "利润表"

    return None
