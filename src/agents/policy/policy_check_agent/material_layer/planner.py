from __future__ import annotations

from typing import Dict, List

from .schemas import RetrievalPlan


DOC_CLASS_HINTS: Dict[str, List[str]] = {
    "business_license": ["营业执照", "统一社会信用代码", "工商登记", "工商"],
    "identity_document": ["身份证", "法定代表人身份证明", "法人身份证"],
    "bank_statement": ["银行流水", "账户明细", "流水"],
    "credit_report": ["征信", "信用报告", "贷款卡"],
    "credit_application": ["授信业务申请书", "授信申请", "申请业务概况", "申请资料", "申请人本行存款"],
    "due_diligence_report": ["尽职调查", "调查分析报告", "授信调查", "尽调", "调查实施情况"],
    "compliance_material": ["合规性原材料", "资料审核", "完整性", "表面真实性", "原件核验"],
    "transaction_voucher": ["借款借据", "贴现凭证", "入账凭证", "贷款借据", "放款凭证"],
    "collateral_document": ["担保", "抵押", "质押", "他项权证", "抵押登记", "机动车登记证", "不动产"],
    "contract": ["合同", "协议", "借款合同", "授信合同", "担保协议", "担保书"],
    "approval_form": ["审批表", "审批意见", "审批记录"],
    "authorization_letter": ["授权书", "授权委托书", "委托书"],
    "financial_statement": ["审计报告", "财务报表", "利润表", "资产负债表"],
    "board_resolution": ["董事会决议", "董事会"],
    "articles_of_association": ["公司章程", "章程"],
}


FIELD_HINTS: Dict[str, List[str]] = {
    "company_name": ["公司名称", "企业名称", "借款人名称"],
    "uscc": ["统一社会信用代码", "社会信用代码"],
    "legal_representative": ["法定代表人", "法人"],
    "expiry_date": ["营业期限", "有效期至", "经营期限"],
}


class ControlRetrievalPlanner:
    def build_plan(self, control_text: str) -> RetrievalPlan:
        doc_classes = []
        canonical_fields = []
        entity_hints = []
        text = control_text or ""

        for doc_class, hints in DOC_CLASS_HINTS.items():
            if any(hint in text for hint in hints):
                doc_classes.append(doc_class)

        for field_name, hints in FIELD_HINTS.items():
            if any(hint in text for hint in hints):
                canonical_fields.append(field_name)

        keyword_queries = [text]
        for hint in doc_classes[:3]:
            for keyword in DOC_CLASS_HINTS.get(hint, [])[:2]:
                if keyword not in keyword_queries:
                    keyword_queries.append(keyword)

        domain_expansions = [
            (["授信调查", "调查分析", "尽职调查", "尽调"], ["due_diligence_report"], ["授信调查分析报告", "尽职调查实施情况", "授信相关信息"]),
            (["授信业务申请书", "授信申请", "申请业务概况"], ["credit_application"], ["授信业务申请书", "申请业务概况", "申请人本行存款"]),
            (["资料审核", "完整性", "表面真实性", "原件", "复印件", "签章清晰"], ["compliance_material"], ["合规性原材料", "授信相关信息", "尽职调查实施情况"]),
            (["借款借据", "贴现凭证", "入账凭证", "放款凭证"], ["transaction_voucher"], ["借款借据", "贴现凭证", "入账凭证"]),
            (["董事会决议", "董事会成员", "签字样本"], ["board_resolution"], ["董事会决议", "董事会成员", "签字样本"]),
            (["章程"], ["articles_of_association"], ["公司章程"]),
            (["担保", "抵押", "质押", "他项权证", "机动车登记证", "不动产"], ["collateral_document"], ["担保协议", "抵押登记", "机动车登记证", "不动产"]),
        ]
        for trigger_terms, classes_to_add, queries_to_add in domain_expansions:
            if any(term in text for term in trigger_terms):
                for doc_class in classes_to_add:
                    if doc_class not in doc_classes:
                        doc_classes.append(doc_class)
                for query in queries_to_add:
                    if query not in keyword_queries:
                        keyword_queries.append(query)

        if "比亚迪" in text:
            entity_hints.append("比亚迪")
        if "王传福" in text:
            entity_hints.append("王传福")

        reasoning_parts = []
        if doc_classes:
            reasoning_parts.append(f"docClasses={doc_classes}")
        if canonical_fields:
            reasoning_parts.append(f"canonicalFields={canonical_fields}")
        if entity_hints:
            reasoning_parts.append(f"entityHints={entity_hints}")
        if not reasoning_parts:
            reasoning_parts.append("No strong structured hint matched; fallback to raw control text retrieval.")

        return RetrievalPlan(
            controlText=text,
            docClasses=doc_classes,
            canonicalFields=canonical_fields,
            keywordQueries=keyword_queries[:6],
            entityHints=entity_hints,
            reasoning="; ".join(reasoning_parts),
        )
