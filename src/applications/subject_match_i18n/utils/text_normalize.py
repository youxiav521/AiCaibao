"""文本标准化工具：用于科目名称的归一化比较"""

from __future__ import annotations

import re
import unicodedata


# 匹配空白字符和指定范围的标点符号
# \s: 空白字符 | \u3000: 全角空格 | \u00a0: 不换行空格
# \uff00-\uffef: 全角ASCII | \u2000-\u206f: 通用标点 | \u0020-\u007e: 基本ASCII范围
_PUNCT_AND_SPACE = re.compile(r"[\s\u3000\u00a0\uff00-\uffef\u2000-\u206f\u0020-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]+")


def normalize(text: str) -> str:
    r"""
    标准化科目名称：
    1. NFKC Unicode 归一化（全角转半角、合字拆分等）
    2. 转小写
    3. 按空白/标点切分为 token，对每个 token 做保守的复数→单数归一
       （inventories → inventory、assets → asset），保留 ss/oss/us/is 等结尾以免误伤
    4. 拼接所有 token，并去除空白字符和指定标点范围：
       \s\u3000\u00a0\uff00-\uffef\u2000-\u206f\u0020-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e

    两侧（模板与源行）对称走同一规则，可在 lookup / fuzzy / 倒排索引三路统一吸收单复数差异。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # 先用同一套标点/空白切分为 token，逐个归一后再无缝拼接，保持下游 key 形态稳定
    tokens = [t for t in _PUNCT_AND_SPACE.split(text) if t]
    if not tokens:
        return ""
    return "".join(_singularize_token(t) for t in tokens)


# 单复数归一：白名单防止 business / loss / gross / series / news / status 等被误伤
# 命中即跳过归一；这些词在英文财务科目语境中不会有"+s/+es"的复数形态
_SINGULAR_KEEP_AS_IS: frozenset[str] = frozenset({
    # -ss 结尾（loss/gross/business/expense-> expenses 例外见下）
    "loss", "gross", "business", "address", "process", "access", "progress",
    "class", "miss", "pass", "press", "cross",
    # -us 结尾
    "bonus", "surplus", "minus", "plus", "status", "campus", "focus",
    "consensus", "census", "virus",
    # -is 结尾
    "basis", "analysis", "thesis", "crisis", "axis", "oasis",
    # -ies/-es 非复数结尾
    "series", "species",
    # -s 结尾的不可数 / 单数
    "news", "means", "goods", "premises", "savings", "earnings",
    "headquarters", "premises", "proceeds", "remains",
    # 其他易误伤
    "less", "this", "his", "its", "yes", "us", "is", "as", "was",
    "has", "does", "gas",
})


def _singularize_token(token: str) -> str:
    """
    极简且保守的英文复数→单数归一：仅处理常见的 -ies / -es / -s 三种后缀，
    其余原样返回；命中白名单时也原样返回。

    - 不试图覆盖 child→children、man→men 等不规则复数（财务科目极少出现）
    - 优先级：白名单 > -ies > -ches/-shes/-xes/-sses/-zes > -s（排除 -ss/-us/-is）
    """
    if len(token) <= 3:
        return token
    if token in _SINGULAR_KEEP_AS_IS:
        return token

    # inventories -> inventory ; liabilities -> liability
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    # taxes -> tax ; expenses -> expense（先剥 -es 适用于 ch/sh/x/ss/z 结尾的基词）
    if token.endswith("es") and len(token) > 3:
        stem = token[:-2]
        if stem.endswith(("ch", "sh", "x", "ss", "z")):
            return stem
        # 普通 -es 视为 -s 后缀（assets/notes/fees 等已在下方分支统一处理），
        # 但 services/issues 等去 -s 即可，统一在 -s 分支处理
    # assets -> asset ; receivables -> receivable
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


# --- IFRS 限定词剥离 ---

_IFRS_REMOVABLE_SUFFIXES: list[str] = [
    "[abstract]",
    "[member]",
    "[axis]",
    "[domain]",
]

# 流动性前缀：仅用于额外召回 key（让源裸名 "lease liabilities" 能召回到
# "Non-current lease liabilities" / "Current lease liabilities" 一同进入消歧）。
# 不放进 _IFRS_REMOVABLE_PREFIXES，避免影响 strip_ifrs_qualifiers 的现有调用方。
_LIQUIDITY_REMOVABLE_PREFIXES: list[str] = [
    "non-current ",
    "noncurrent ",
    "current ",
    "long-term ",
    "long term ",
    "short-term ",
    "short term ",
]

_BRACKET_CONTENT_RE = re.compile(r"\[[^\]]*\]")


def strip_ifrs_qualifiers(text: str) -> str:
    """
    剥离 IFRS 限定词：仅剥离方括号后缀（如 [abstract]、[member]）及方括号内容。
    不剥离前缀，不剥离圆括号内容。
    """
    if not text:
        return text
    result = text.lower().strip()
    for suffix in _IFRS_REMOVABLE_SUFFIXES:
        if result.endswith(suffix):
            result = result[: -len(suffix)]
    result = _BRACKET_CONTENT_RE.sub("", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def strip_liquidity_prefix(text: str) -> str:
    """
    剥离首部流动性前缀（current / non-current / long-term 等），仅返回去前缀后的文本；
    不匹配任何前缀时返回去空白后的原文。
    用于模板侧生成"裸名召回 key"——让源裸名能召回到所有流动性变体一起进入消歧。
    """
    if not text:
        return text
    result = text.lower().strip()
    for prefix in _LIQUIDITY_REMOVABLE_PREFIXES:
        if result.startswith(prefix):
            return result[len(prefix):].strip()
    return result
