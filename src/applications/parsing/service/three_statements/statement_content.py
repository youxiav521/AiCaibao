from __future__ import annotations

BALANCE_SHEET_TERMINAL_KWS: tuple[str, ...] = (
    "负债和所有者权益总计",
    "负债及所有者权益总计",
    "负债和所有者权益合计",
    "负债及所有者权益合计",
    "负债和所有者权益（或股东权益）总计",
    "负债及所有者权益（或股东权益）总计",
    "负债和所有者权益（或股东权益）合计",
    "负债及所有者权益（或股东权益）合计",
)
PROFIT_TERMINAL_KWS: tuple[str, ...] = (
    "综合收益总额",
)
CASH_FLOW_TERMINAL_KWS: tuple[str, ...] = (
    "期末现金及现金等价物余额",
)
BASE_TYPE_START_KEYWORDS: dict[str, tuple[str, ...]] = {
    "资产负债表": ("流动资产",),
    "利润表": ("营业收入", "营业总收入"),
    "现金流量表": ("经营活动",),
}
BASE_TYPE_CONTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "资产负债表": ("流动资产", "非流动资产", "资产总计", "资产合计", "流动负债", "非流动负债", "所有者权益", "股东权益"),
    "利润表": ("营业收入", "营业成本", "营业利润", "利润总额", "净利润", "每股收益"),
    "现金流量表": ("经营活动", "投资活动", "筹资活动", "现金及现金等价物"),
}


def get_terminal_keywords(base_type: str) -> tuple[str, ...]:
    if base_type == "资产负债表":
        return BALANCE_SHEET_TERMINAL_KWS
    if base_type == "利润表":
        return PROFIT_TERMINAL_KWS
    if base_type == "现金流量表":
        return CASH_FLOW_TERMINAL_KWS
    return ()


def content_matches_base_type(table_md: str, base_type: str) -> bool:
    keywords = BASE_TYPE_CONTENT_KEYWORDS.get(base_type)
    if not keywords:
        return False
    compact = "".join((table_md or "").split())
    return sum(1 for keyword in keywords if keyword in compact) >= 2


def content_matches_other_base_type(table_md: str, primary_base: str) -> bool:
    compact = "".join((table_md or "").split())
    primary_score = 0
    primary_keywords = BASE_TYPE_CONTENT_KEYWORDS.get(primary_base, ())
    if primary_keywords:
        primary_score = sum(1 for keyword in primary_keywords if keyword in compact)
    for base_type, keywords in BASE_TYPE_CONTENT_KEYWORDS.items():
        if base_type == primary_base:
            continue
        score = sum(1 for keyword in keywords if keyword in compact)
        if score >= 2 and score > primary_score:
            return True
    return False
