from __future__ import annotations

from functools import lru_cache
import re
from typing import Dict, Iterable, List

import jieba


CN_STOPWORDS = {
    "根据",
    "以及",
    "相关",
    "规定",
    "要求",
    "进行",
    "提交",
    "提供",
    "包括",
    "材料",
    "文件",
    "公司",
    "企业",
    "银行",
    "客户",
    "是否",
    "应当",
    "可以",
    "不得",
}

SUBTOKEN_HINTS = (
    "证",
    "章",
    "决议",
    "执照",
    "征信",
    "调查",
    "报告",
    "授权",
    "授信",
    "申请",
    "审批",
    "批复",
    "批示",
    "核准",
    "放款",
    "借款",
    "贷款",
    "还款",
    "保函",
    "担保",
    "抵押",
    "质押",
    "登记",
    "合同",
    "协议",
    "证明",
    "函",
    "单",
    "表",
    "票",
    "账",
    "流水",
    "回单",
    "发票",
    "清单",
    "台账",
    "对账",
    "用途",
    "业务",
    "供应商",
    "客户",
    "产品",
    "收入",
    "营利",
    "访谈",
    "录音",
    "文本",
    "仓库",
    "现场",
    "资产",
    "车辆",
    "审计",
    "财务",
    "报表",
    "章程",
    "尽调",
    "核查",
    "签字",
    "签名",
    "印章",
    "凭证",
    "样本",
)

QUERY_SYNONYMS = {
    "法人": ["法定代表人"],
    "法人代表": ["法定代表人"],
    "营业执照": ["营业执照副本", "统一社会信用代码"],
    "统一社会信用代码": ["营业执照", "社会信用代码", "uscc"],
    "身份证": ["居民身份证", "法定代表人身份证明"],
    "身份证明": ["法定代表人身份证明", "身份证"],
    "法定代表人": ["法人", "法人代表"],
    "章程": ["公司章程"],
    "公司章程": ["章程"],
    "董事会决议": ["董事会", "决议"],
    "决议": ["董事会决议"],
    "签字": ["签名", "签章"],
    "签名": ["签字"],
    "盖章": ["印章", "签章"],
    "印章": ["盖章"],
    "征信": ["征信报告", "信用报告"],
    "征信报告": ["征信", "信用报告"],
    "信用报告": ["征信", "征信报告"],
    "尽调": ["尽职调查", "调查分析报告"],
    "尽职调查": ["尽调", "调查分析报告"],
    "调查分析报告": ["尽职调查", "尽调"],
    "授权书": ["授权委托书"],
    "授权委托书": ["授权书"],
    "抵押": ["抵押登记", "他项权证"],
    "他项权证": ["抵押", "抵押登记"],
    "授信": ["授信申请", "授信审批", "授信批复"],
    "授信申请": ["申请业务概况", "授信业务申请书", "授信"],
    "授信业务申请书": ["授信申请", "申请业务概况"],
    "申请业务概况": ["授信申请", "授信业务申请书"],
    "借款合同": ["贷款合同", "授信合同", "合同"],
    "贷款合同": ["借款合同", "授信合同", "合同"],
    "授信合同": ["借款合同", "贷款合同", "合同"],
    "保证合同": ["担保合同", "保证"],
    "担保合同": ["保证合同", "担保"],
    "质押合同": ["质押", "担保合同"],
    "抵押合同": ["抵押", "担保合同"],
    "征信查询记录": ["征信", "查询记录", "信用报告"],
    "查询记录": ["征信查询记录", "征信"],
    "银行流水": ["账户明细", "交易明细", "对账单"],
    "账户明细": ["银行流水", "交易明细", "对账单"],
    "交易明细": ["银行流水", "账户明细"],
    "对账单": ["银行流水", "账户明细"],
    "回单": ["银行回单", "电子回单"],
    "银行回单": ["回单", "电子回单"],
    "电子回单": ["回单", "银行回单"],
    "发票": ["增值税发票", "专用发票", "普通发票"],
    "增值税发票": ["发票", "专用发票", "普通发票"],
    "专用发票": ["增值税发票", "发票"],
    "普通发票": ["增值税发票", "发票"],
    "董事会": ["董事会决议"],
    "股东会决议": ["股东会", "决议"],
    "股东会": ["股东会决议"],
    "公司章程修正案": ["公司章程", "章程"],
    "审计报告": ["财务报表", "资产负债表", "利润表", "现金流量表"],
    "财务报表": ["审计报告", "资产负债表", "利润表", "现金流量表"],
    "资产负债表": ["财务报表", "审计报告"],
    "利润表": ["财务报表", "审计报告"],
    "现金流量表": ["财务报表", "审计报告"],
    "授权审批表": ["审批表", "审批单", "授权书"],
    "审批表": ["授权审批表", "审批单"],
    "审批单": ["审批表", "授权审批表"],
    "批复": ["批示", "核准"],
    "批示": ["批复", "核准"],
    "核准": ["批复", "批示"],
    "放款凭证": ["借款借据", "入账凭证", "支付凭证"],
    "借款借据": ["放款凭证", "贷款借据"],
    "贷款借据": ["借款借据", "放款凭证"],
    "入账凭证": ["放款凭证", "支付凭证"],
    "支付凭证": ["放款凭证", "入账凭证"],
    "保函": ["担保函", "保证函"],
    "担保函": ["保函", "保证函"],
    "保证函": ["保函", "担保函"],
    "不动产登记": ["不动产权证", "他项权证", "抵押登记"],
    "不动产权证": ["不动产登记", "抵押登记"],
    "机动车登记证": ["车辆登记证", "行驶证"],
    "车辆登记证": ["机动车登记证", "行驶证"],
    "行驶证": ["机动车登记证", "车辆登记证"],
    "名单": ["清单", "台账"],
    "清单": ["名单", "台账"],
    "台账": ["清单", "名单"],
    "协议": ["合同"],
    "合同": ["协议"],
    "企业信用报告": ["信用报告", "征信报告", "企业征信"],
    "企业征信": ["企业信用报告", "征信报告", "信用报告"],
    "法人个人征信": ["个人征信", "征信报告", "信用报告"],
    "个人征信": ["法人个人征信", "征信报告", "信用报告"],
    "法人身份证": ["法定代表人身份证明", "身份证", "法定代表人"],
    "借款用途": ["用途说明", "贷款用途", "资金用途"],
    "贷款用途": ["借款用途", "资金用途"],
    "资金用途": ["借款用途", "贷款用途"],
    "主营业务介绍": ["主营业务", "业务介绍", "公司介绍"],
    "主营业务": ["主营业务介绍", "业务介绍", "主营业务收入构成", "主营业务营利分析"],
    "业务介绍": ["主营业务介绍", "公司介绍"],
    "主营业务收入构成": ["主营业务", "收入构成", "产品结构"],
    "收入构成": ["主营业务收入构成", "营收构成", "产品结构"],
    "主营业务营利分析": ["主营业务", "营利分析", "盈利分析"],
    "营利分析": ["主营业务营利分析", "盈利分析"],
    "盈利分析": ["主营业务营利分析", "营利分析"],
    "核心产品分类分析": ["产品分类", "核心产品", "产品结构"],
    "产品分类": ["核心产品分类分析", "核心产品", "产品结构"],
    "核心产品": ["核心产品分类分析", "产品分类", "产品结构"],
    "产品结构": ["核心产品分类分析", "主营业务收入构成"],
    "主要供应商": ["供应商", "供应商清单", "上游客户"],
    "供应商": ["主要供应商", "供应商清单"],
    "供应商清单": ["主要供应商", "供应商"],
    "主要客户": ["客户", "客户清单", "下游客户"],
    "客户": ["主要客户", "客户清单"],
    "客户清单": ["主要客户", "客户"],
    "仓库尽调材料": ["仓库", "尽调材料", "现场尽调"],
    "现场尽调材料": ["尽调材料", "现场走访", "实地尽调"],
    "尽调材料": ["现场尽调材料", "仓库尽调材料", "尽职调查实施情况"],
    "尽职调查实施情况": ["尽调材料", "尽职调查", "调查实施情况"],
    "现场走访": ["现场尽调材料", "实地尽调"],
    "实地尽调": ["现场尽调材料", "现场走访"],
    "访谈录音": ["访谈记录", "访谈文本", "访谈录音文本版"],
    "访谈录音文本版": ["访谈录音", "访谈文本", "访谈记录"],
    "访谈文本": ["访谈录音文本版", "访谈记录"],
    "访谈记录": ["访谈录音", "访谈录音文本版"],
    "合规性原材料": ["合规材料", "合规性资料", "资料审核"],
    "合规材料": ["合规性原材料", "合规性资料", "资料审核"],
    "授信相关信息": ["授信信息", "授信资料", "授信申请"],
    "授信信息": ["授信相关信息", "授信资料"],
    "授信资料": ["授信相关信息", "授信信息"],
    "本次申请业务概况": ["申请业务概况", "授信申请", "业务概况"],
    "申请人本行存款": ["本行存款", "存款情况", "账户流水"],
    "本行存款": ["申请人本行存款", "存款情况", "账户流水"],
    "存款情况": ["本行存款", "申请人本行存款"],
    "账户流水": ["银行流水", "账户明细"],
    "不动产": ["不动产权证", "不动产登记", "房产证明"],
    "房产证明": ["不动产", "不动产权证"],
    "机动车登记证": ["车辆登记证", "行驶证", "车辆登记"],
    "车辆登记": ["机动车登记证", "车辆登记证"],
    "其他资产证明": ["资产证明", "机动车登记证", "不动产"],
    "资产证明": ["其他资产证明", "不动产", "机动车登记证"],
    "建设银行": ["建行", "中国建设银行"],
    "建行": ["建设银行", "中国建设银行"],
    "中国建设银行": ["建设银行", "建行"],
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_text(text: str) -> str:
    return normalize_whitespace(text).lower()


@lru_cache(maxsize=2048)
def _chinese_search_tokens(token: str) -> tuple[str, ...]:
    pieces = []
    seen = set()
    for piece in jieba.cut_for_search(token):
        normalized = piece.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        pieces.append(normalized)
    return tuple(pieces)


def _fallback_sub_tokens(token: str) -> List[str]:
    result: List[str] = []
    token_len = len(token)
    if not (4 <= token_len <= 16):
        return result
    if not any(hint in token for hint in SUBTOKEN_HINTS):
        return result
    for size in (2, 3, 4):
        if token_len < size:
            continue
        for index in range(token_len - size + 1):
            result.append(token[index:index + size])
    return result


def _base_keyword_tokens(text: str) -> List[str]:
    raw_tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+", (text or "").lower())
    result: List[str] = []
    seen: set[str] = set()

    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue

        candidates = [token]
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            search_terms = [
                piece for piece in _chinese_search_tokens(token)
                if 2 <= len(piece) <= 8 and piece != token and piece not in CN_STOPWORDS
            ]
            if search_terms:
                candidates.extend(search_terms)
            else:
                candidates.extend(_fallback_sub_tokens(token))

        for candidate in candidates:
            if len(candidate) <= 1:
                continue
            if candidate.isdigit():
                continue
            if candidate in CN_STOPWORDS:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", candidate) and len(candidate) == 2 and any(
                char in candidate for char in {"的", "及", "与", "和", "按", "将", "把", "对", "向", "据", "于"}
            ):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            result.append(candidate)

    return result


def keyword_tokens(text: str) -> List[str]:
    return _base_keyword_tokens(text)


def expand_query_tokens(text: str) -> List[str]:
    return list(build_query_token_weights(text).keys())


def build_query_token_weights(text: str) -> Dict[str, float]:
    tokens = _base_keyword_tokens(text)
    weights: Dict[str, float] = {}

    for token in tokens:
        weights[token] = max(weights.get(token, 0.0), 1.0)
        for synonym in QUERY_SYNONYMS.get(token, []):
            for candidate in _base_keyword_tokens(synonym):
                if candidate == token:
                    continue
                weights[candidate] = max(weights.get(candidate, 0.0), 0.72)

    return weights


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def looks_empty(text: str) -> bool:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return True
    return cleaned in {"-", "--", "—", "n/a", "na", "无", "空"}


def guess_candidate_tags(file_name: str, sample_text: str) -> List[str]:
    content = f"{file_name} {sample_text}"
    tags = []
    hints = {
        "license": ["营业执照", "工商登记", "统一社会信用代码"],
        "identity": ["身份证", "法定代表人", "公民身份号码"],
        "credit": ["征信", "信用报告", "贷款卡"],
        "application": ["授信申请", "申请业务概况", "申请书"],
        "due_diligence": ["尽职调查", "调查分析", "尽调"],
        "compliance": ["合规", "资料审核", "真实性", "完整性"],
        "collateral": ["抵押", "质押", "不动产", "机动车登记证"],
        "resolution": ["董事会决议", "董事会"],
        "charter": ["公司章程", "章程"],
        "statement": ["银行流水", "账户明细", "交易日期"],
    }
    for tag, keywords in hints.items():
        if any(keyword in content for keyword in keywords):
            tags.append(tag)
    return tags
