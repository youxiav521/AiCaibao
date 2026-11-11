from __future__ import annotations

import re
from typing import Dict, List, Tuple

from src.agents.policy.policy_check_agent.schemas import ResourceFile, item_content_extract

from .schemas import DocumentClassification


DOC_CLASS_RULES: Dict[str, List[str]] = {
    "business_license": ["营业执照", "统一社会信用代码", "法定代表人", "注册资本"],
    "identity_document": ["身份证", "居民身份证", "公民身份号码", "签发机关"],
    "bank_statement": ["银行流水", "账户明细", "交易日期", "余额", "开户行"],
    "credit_report": ["征信", "信用报告", "信贷记录", "查询记录"],
    "credit_application": ["授信业务申请书", "授信申请", "申请业务概况", "申请人本行存款", "授信申请书"],
    "due_diligence_report": ["尽职调查", "调查分析报告", "授信调查", "尽调", "调查实施情况"],
    "compliance_material": ["合规性原材料", "资料审核", "完整性", "表面真实性", "原件核验"],
    "transaction_voucher": ["借款借据", "贴现凭证", "入账凭证", "贷款借据", "放款凭证"],
    "collateral_document": ["抵押", "质押", "担保", "他项权证", "机动车登记证", "不动产"],
    "contract": ["合同", "协议", "借款合同", "授信合同", "保证合同"],
    "invoice": ["发票", "税额", "价税合计", "开票日期"],
    "approval_form": ["审批表", "审批意见", "审批记录", "审批人"],
    "authorization_letter": ["授权书", "授权委托书", "委托人", "受托人"],
    "financial_statement": ["审计报告", "资产负债表", "利润表", "现金流量表", "财务报表"],
    "board_resolution": ["董事会决议", "董事会", "会议决议"],
    "articles_of_association": ["公司章程", "章程", "股东会", "董事会"],
}


def _extract_document_hint(resource_file: ResourceFile, limit: int = 10) -> str:
    parts: List[str] = [resource_file.fileName]
    if resource_file.originalFileName:
        parts.append(resource_file.originalFileName)
    for item in resource_file.content[:limit]:
        text = item_content_extract(item)
        if text:
            parts.append(text[:200])
    return "\n".join(parts)


def _score_rule_match(file_name: str, hint: str, keywords: List[str]) -> Tuple[int, List[str]]:
    score = 0
    matched: List[str] = []
    for keyword in keywords:
        if keyword.lower() in file_name.lower():
            score += 3
            matched.append(f"filename:{keyword}")
        elif keyword.lower() in hint.lower():
            score += 1
            matched.append(keyword)
    return score, matched


class RuleBasedDocumentClassifier:
    def classify(self, resource_file: ResourceFile) -> DocumentClassification:
        hint = _extract_document_hint(resource_file)
        best_class = "other"
        best_score = 0
        matched_rules: List[str] = []
        for doc_class, keywords in DOC_CLASS_RULES.items():
            score, matched = _score_rule_match(resource_file.fileName, hint, keywords)
            if score > best_score:
                best_class = doc_class
                best_score = score
                matched_rules = matched

        if best_score == 0:
            reason = "No strong rule matched file name or parsed content."
            confidence = 0.25
        else:
            confidence = min(0.55 + 0.12 * best_score, 0.98)
            reason = f"Matched keywords: {', '.join(matched_rules)}"

        return DocumentClassification(
            docId=resource_file.fileId,
            fileId=resource_file.fileId,
            fileName=resource_file.fileName,
            docClass=best_class,  # type: ignore[arg-type]
            confidence=confidence,
            reason=reason,
            matchedRules=matched_rules,
        )
