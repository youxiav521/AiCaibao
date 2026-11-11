from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .schemas import DocType

# Shared mapping from generic doc profile types to policy-check retriever doc types.
SHARED_DOC_TYPE_TO_POLICY_DOC_TYPES: Dict[DocType, Tuple[str, ...]] = {
    DocType.BUSINESS_LICENSE: ("business_license",),
    DocType.LEGAL_REP_ID: ("identity_document",),
    DocType.LEGAL_REP_MARRIAGE_CERT: ("identity_document",),
    DocType.SPOUSE_ID: ("identity_document",),
    DocType.LEGAL_REP_CERT: ("identity_document",),
    DocType.ARTICLES_OF_ASSOCIATION: ("articles_of_association",),
    DocType.SIGNATURE_SAMPLE: ("board_resolution", "compliance_material"),
    DocType.CREDIT_REPORT: ("credit_report",),
    DocType.COLLATERAL_APPRAISAL: ("collateral_document",),
    DocType.CAPITAL_VERIFICATION_REPORT: ("financial_statement",),
    DocType.CONTRACT_AND_PAYMENT: ("transaction_voucher", "credit_application"),
    DocType.OWNERSHIP_STRUCTURE: ("compliance_material",),
    DocType.MEETING_MINUTES: ("board_resolution",),
    DocType.TAX_REG_CERT: ("compliance_material",),
}

POLICY_DOC_TYPE_ANCHOR_TERMS: Dict[str, Tuple[str, ...]] = {
    "identity_document": ("身份证", "身份证明", "公民身份号码", "签发机关", "有效期限"),
    "business_license": ("营业执照", "统一社会信用代码", "法定代表人", "住所"),
    "credit_application": ("授信申请", "贷款申请", "额度申请", "展期申请"),
    "credit_report": ("征信报告", "信用报告", "不良记录", "查询记录"),
    "due_diligence_report": ("尽职调查", "调查分析", "尽调报告"),
    "compliance_material": ("合规", "真实性", "完整性", "原件", "复印件"),
    "transaction_voucher": ("借款借据", "放款凭证", "入账凭证", "回单"),
    "board_resolution": ("董事会决议", "董事会成员", "签字样本", "会议决议"),
    "articles_of_association": ("公司章程", "章程"),
    "collateral_document": ("抵押", "质押", "评估报告", "他项权证", "不动产"),
    "bank_statement": ("银行流水", "账户明细", "交易流水"),
    "financial_statement": ("财务报表", "审计报告", "资产负债表", "利润表"),
}


def map_doc_types_to_policy_doc_types(doc_types: List[DocType]) -> Set[str]:
    mapped: Set[str] = set()
    for doc_type in doc_types or []:
        mapped.update(SHARED_DOC_TYPE_TO_POLICY_DOC_TYPES.get(doc_type, ()))
    return mapped


def collect_policy_doc_type_anchor_terms(policy_doc_types: Set[str]) -> Set[str]:
    anchors: Set[str] = set()
    for doc_type in policy_doc_types or set():
        anchors.update(POLICY_DOC_TYPE_ANCHOR_TERMS.get(doc_type, ()))
    return anchors
