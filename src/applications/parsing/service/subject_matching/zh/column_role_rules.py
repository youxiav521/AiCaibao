from __future__ import annotations

import re

ZH_SUBJECT_HEADER_KEYWORDS: frozenset[str] = frozenset({
    "项目", "科目", "科目名称", "名称", "行项目",
})
ZH_AMOUNT_HEADER_KEYWORDS: frozenset[str] = frozenset({
    "金额", "余额", "本期金额", "上期金额", "期末余额", "期初余额",
    "本年累计", "上年累计", "本期数", "上期数", "期末数", "期初数",
    "本年金额", "上年金额", "本年累计数", "上年累计数",
    "本期发生额", "上期发生额", "本年同期数", "上年同期数",
})
ZH_AMOUNT_HEADER_RE = re.compile(
    r"(?:金额|余额|本期|上期|期末|期初|本年|上年|累计|发生额|\d{4}\s*(?:年度?|[/-]))"
)
