from __future__ import annotations

import re

from src.utils.chinese_converter import traditional_to_simplified

from ..text_utils import (
    FAMILY_SUFFIXES_DEDUP,
    classify_family_pair,
    deep_normalize,
    has_trailing_heading_punctuation,
    normalize_subject_text,
    template_text_key,
)

AGGREGATE_SUFFIXES: tuple[str, ...] = ("净额", "净增加额", "净变动额", "净减少额")
NET_TYPE_SUFFIXES: tuple[str, ...] = ("净额", "净值", "净损失", "净收益")
SUMMARY_SUFFIXES: tuple[str, ...] = ("合计", "小计", "总计")
CASHFLOW_SECTION_KEYWORDS: tuple[str, ...] = (
    "经营活动产生的现金流量",
    "投资活动产生的现金流量",
    "筹资活动产生的现金流量",
)
EPS_KEYWORDS: tuple[str, ...] = ("每股收益", "每股净资产", "每股")
PROFIT_KEYWORDS: tuple[str, ...] = (
    "净利润", "利润总额", "营业利润", "归属于母公司股东的净利润",
    "持续经营净利润", "非持续经营净利润",
)
SECTION_HEADING_TERMS: tuple[str, ...] = (
    "流动资产", "非流动资产", "流动负债", "非流动负债",
    "长期投资", "固定资产", "在建工程", "使用权资产",
    "无形资产", "长期待摊费用", "递延资产", "其他长期资产",
)
SKIP_EXACT: frozenset[str] = frozenset({
    "合计", "小计", "总计", "项目", "科目", "科目名称",
    "期末余额", "期初余额", "本期金额", "上期金额",
    "本年累计", "上年累计", "本期数", "上期数", "期末数", "期初数",
    "本年金额", "上年金额", "本期发生额", "上期发生额",
    "收入", "成本", "金额", "余额", "备注", "说明",
    "行次", "序号", "编号",
    "调整数", "重述数", "调整后", "重分类",
    "资产", "负债", "所有者权益", "股东权益",
    "流动资产", "非流动资产", "流动负债", "非流动负债",
})
SUPP_SECTION_HEADERS: tuple[str, ...] = (
    "补充资料", "将净利润调节为经营活动现金流量",
    "不涉及现金收支的重大投资和筹资活动",
    "现金及现金等价物净变动情况",
)
SUBJECT_LIKE_TERMS: tuple[str, ...] = (
    "损失", "收益", "减值", "折旧", "摊销", "报废", "处置",
    "减少", "增加", "费用", "准备", "净额", "净收益", "净损失",
    "支付", "收到", "收回", "投入",
)


class ChineseSubjectTextRules:
    family_suffixes_dedup = FAMILY_SUFFIXES_DEDUP
    aggregate_suffixes = AGGREGATE_SUFFIXES
    net_type_suffixes = NET_TYPE_SUFFIXES
    summary_suffixes = SUMMARY_SUFFIXES
    cashflow_section_keywords = CASHFLOW_SECTION_KEYWORDS
    eps_keywords = EPS_KEYWORDS
    profit_keywords = PROFIT_KEYWORDS

    def normalize_subject_text(self, text: str) -> str:
        return normalize_subject_text(text)

    def deep_normalize(self, text: str) -> str:
        return deep_normalize(text)

    def template_text_key(self, text: str) -> str:
        return template_text_key(text)

    def has_trailing_heading_punctuation(self, text: str) -> bool:
        return has_trailing_heading_punctuation(text)

    def is_hard_non_subject_text(self, text: str) -> bool:
        if not text:
            return True
        text_s = traditional_to_simplified(text)
        compact_s = re.sub(r"\s+", "", text_s)
        if compact_s in SKIP_EXACT or text_s in SKIP_EXACT:
            return True
        if re.match(r"^[\d,.\-—\s%]+$", text):
            return True
        if re.match(r"^[—\-=\s]+$", text):
            return True
        if re.search(r"\d{4}\s*年\s*\d{1,2}\s*月", text):
            return True
        if any(kw in text_s for kw in ("编制单位", "单位：", "金额单位", "币种", "人民币")):
            return True
        if any(h in text_s for h in SUPP_SECTION_HEADERS):
            return True
        return False

    def is_soft_non_subject_text(self, text: str) -> bool:
        if not text:
            return False
        text_s = traditional_to_simplified(text)
        compact_s = re.sub(r"\s+", "", text_s)
        return (
            len(compact_s) > 3
            and any(compact_s.endswith(s) for s in SUMMARY_SUFFIXES)
            and bool(re.search(r"[及和与]", compact_s))
        )

    def looks_like_section_heading(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False
        if self.has_trailing_heading_punctuation(compact):
            return True

        compact = traditional_to_simplified(compact)
        if any(term in compact for term in SUBJECT_LIKE_TERMS):
            return False

        if not re.search(r"[及和与]", compact):
            return False
        hits = sum(1 for term in SECTION_HEADING_TERMS if term in compact)
        return hits >= 2

    def deterministic_match_key(self, text: str) -> str | None:
        compact = self.deep_normalize(text)
        if not compact:
            return None

        is_long_term_asset_purchase = (
            any(term in compact for term in ("购建", "购置"))
            and all(
                term in compact
                for term in ("固定资产", "无形资产", "长期资产", "支付", "现金")
            )
        )
        if is_long_term_asset_purchase:
            return "cashflow:purchase-long-term-assets"

        if (
            "以摊余成本计量的金融资产" in compact
            and "终止确认" in compact
            and any(term in compact for term in ("收益", "损失", "损益"))
        ):
            return "profit:amortized-cost-derecognition"

        is_disposal_cash = (
            "处置" in compact
            and "收到" in compact
            and "现金" in compact
        )
        is_joint_or_associate = "合营" in compact and "联营" in compact
        is_subsidiary_disposal = "子公司" in compact and any(
            term in compact for term in ("营业单位", "经营单位")
        )
        if is_disposal_cash and (is_joint_or_associate or is_subsidiary_disposal):
            return "cashflow:disposal-investee"

        is_equity_holder_inflow = (
            "其他权益工具持有者" in compact
            and "投入" in compact
            and "现金" in compact
        )
        is_absorbed_investment = (
            "吸收" in compact
            and "投资" in compact
            and "收到" in compact
            and "现金" in compact
            and "子公司" not in compact
            and "少数股东" not in compact
        )
        if is_equity_holder_inflow or is_absorbed_investment:
            return "cashflow:equity-investment-inflow"

        if (
            "公允价值变动" in compact
            and any(term in compact for term in ("应收款项融资", "其他债权投资"))
        ):
            return "profit:other-debt-investment-fair-value-change"

        if (
            "信用减值准备" in compact
            and any(term in compact for term in ("应收款项融资", "其他债权投资"))
        ):
            return "profit:other-debt-investment-credit-impairment"

        if "外币" in compact and "报表折算差额" in compact:
            return "profit:foreign-currency-statement-translation"

        is_subsidiary_acquisition_cash = (
            any(term in compact for term in ("取得", "收购"))
            and "子公司" in compact
            and any(term in compact for term in ("营业单位", "经营单位"))
            and "支付" in compact
            and "现金" in compact
        )
        if is_subsidiary_acquisition_cash:
            return "cashflow:subsidiary-acquisition"

        for activity in ("经营", "投资", "筹资"):
            if (
                f"{activity}活动" in compact
                and "现金流量净额" in compact
                and any(term in compact for term in ("产生", "使用", "流出"))
            ):
                return f"cashflow:{activity}-activity-net"

        has_cash_equivalents = any(
            term in compact
            for term in ("现金及现金等价物", "现金和现金等价物")
        )
        has_increase = "增加" in compact
        has_decrease = "减少" in compact
        if (
            has_cash_equivalents
            and (has_increase or has_decrease)
            and ("净" in compact or (has_increase and has_decrease))
        ):
            return "cashflow:cash-equivalents-net-change"

        return None

    def classify_family_pair(self, deep_a: str, deep_b: str) -> str | None:
        return classify_family_pair(deep_a, deep_b)
