from __future__ import annotations

from typing import Any, Dict, List


SECTION_RULES: Dict[str, Dict[str, Any]] = {
    "目的": {
        "check_point": "purpose_section",
        "severity": "high",
        "allow_element_only": True,
        "title_aliases": ["目的", "宗旨", "总体目标", "编制目的"],
        "content_patterns": [
            r"为.{0,20}(规范|加强|完善|明确)",
            r"本(办法|规定|制度|指引).{0,10}(目的|旨在)",
        ],
    },
    "适用范围": {
        "check_point": "scope_section",
        "severity": "high",
        "allow_element_only": True,
        "title_aliases": ["适用范围", "适用对象", "适用机构", "适用业务范围"],
        "content_patterns": [
            r"适用于.{0,30}",
            r"适用(对象|机构|范围).{0,20}",
        ],
    },
    "定义与术语": {
        "check_point": "definition_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["定义", "术语定义", "释义", "名词解释"],
        "content_patterns": [
            r"本(办法|规定|制度|协议)所称.{0,30}是指",
            r"以下简称",
        ],
    },
    "职责分工": {
        "check_point": "responsibility_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["职责分工", "职责", "责任分工", "岗位职责"],
        "content_patterns": [
            r"(负责|职责|承担).{0,30}",
            r"(部门|岗位).{0,20}(负责|职责)",
        ],
    },
    "办理流程": {
        "check_point": "process_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["办理流程", "业务流程", "审批流程", "操作流程", "流程"],
        "content_patterns": [
            r"(流程|步骤|程序).{0,20}",
            r"(申请|受理|审批|放款|归档).{0,20}",
        ],
    },
    "申请条件": {
        "check_point": "eligibility_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["申请条件", "准入条件", "适用条件", "受理条件"],
        "content_patterns": [
            r"(符合|满足).{0,30}(条件|要求)",
            r"(申请|准入).{0,20}(条件|要求)",
        ],
    },
    "申请材料": {
        "check_point": "materials_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["申请材料", "申报材料", "报送材料", "资料清单", "提交材料"],
        "content_patterns": [
            r"(提交|提供|报送).{0,20}(材料|资料|文件)",
            r"(资料|材料)清单",
        ],
    },
    "风险控制": {
        "check_point": "risk_control_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["风险控制", "风险管理", "风险防控", "内控要求"],
        "content_patterns": [
            r"风险(控制|管理|防控).{0,20}",
            r"(内控|合规).{0,20}(要求|措施)",
        ],
    },
    "附则": {
        "check_point": "supplementary_section",
        "severity": "low",
        "allow_element_only": False,
        "title_aliases": ["附则", "附加说明", "补充规定"],
        "content_patterns": [
            r"本(办法|规定|制度).{0,10}(由|自).{0,20}(解释|施行)",
        ],
    },
    "合同当事人": {
        "check_point": "party_section",
        "severity": "high",
        "allow_element_only": True,
        "title_aliases": ["合同当事人", "协议双方", "各方主体", "当事人"],
        "content_patterns": [
            r"(甲方|乙方|借款人|贷款人|保证人)",
        ],
    },
    "金额与期限": {
        "check_point": "amount_term_section",
        "severity": "high",
        "allow_element_only": True,
        "title_aliases": ["金额与期限", "授信金额与期限", "借款金额与期限", "额度与期限"],
        "content_patterns": [
            r"(金额|额度).{0,20}(期限|期间)",
            r"(借款|授信).{0,20}(金额|期限)",
        ],
    },
    "利率与费用": {
        "check_point": "rate_fee_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["利率与费用", "利率", "费用", "费率"],
        "content_patterns": [
            r"(利率|费率|费用).{0,20}",
            r"\d+(\.\d+)?%",
        ],
    },
    "还款安排": {
        "check_point": "repayment_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["还款安排", "还款方式", "还款计划"],
        "content_patterns": [
            r"(还款|偿还).{0,20}(方式|计划|安排)",
        ],
    },
    "违约责任": {
        "check_point": "breach_section",
        "severity": "high",
        "allow_element_only": True,
        "title_aliases": ["违约责任", "违约处理", "责任承担"],
        "content_patterns": [
            r"违约.{0,20}(责任|处理)",
            r"(逾期|违约).{0,20}(责任|罚息)",
        ],
    },
    "争议解决": {
        "check_point": "dispute_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["争议解决", "争议处理", "法律适用与争议解决"],
        "content_patterns": [
            r"争议.{0,20}(解决|处理)",
            r"(法院|仲裁).{0,20}(管辖|解决)",
        ],
    },
    "调查结论": {
        "check_point": "conclusion_section",
        "severity": "medium",
        "allow_element_only": True,
        "title_aliases": ["调查结论", "尽调结论", "结论与建议", "结论"],
        "content_patterns": [
            r"(结论|建议).{0,20}",
            r"(调查|尽调).{0,20}(结论|建议)",
        ],
    },
}


COMPLETENESS_PROFILES: Dict[str, Dict[str, Any]] = {
    "bank_policy_basic": {
        "label": "银行制度类文档",
        "sections": ["目的", "适用范围", "定义与术语", "职责分工", "办理流程", "附则"],
    },
    "credit_policy": {
        "label": "授信政策/管理办法",
        "sections": ["目的", "适用范围", "申请条件", "申请材料", "职责分工", "办理流程", "风险控制", "附则"],
    },
    "loan_contract": {
        "label": "借款/授信合同",
        "sections": ["合同当事人", "金额与期限", "利率与费用", "还款安排", "违约责任", "争议解决"],
    },
    "guarantee_contract": {
        "label": "保证/担保合同",
        "sections": ["合同当事人", "适用范围", "金额与期限", "违约责任", "争议解决"],
    },
    "credit_application": {
        "label": "授信申请书/调查申请材料",
        "sections": ["目的", "适用范围", "申请条件", "申请材料", "风险控制"],
    },
    "due_diligence_report": {
        "label": "尽职调查报告",
        "sections": ["目的", "适用范围", "风险控制", "调查结论"],
    },
}


DEFAULT_PROFILE_ID = "bank_policy_basic"
DEFAULT_FALLBACK_SECTIONS = ["目的", "适用范围"]

SECTION_GROUPS: Dict[str, Dict[str, Any]] = {
    "governance_core": {
        "label": "治理制度核心章节",
        "sections": ["目的", "适用范围", "职责分工", "附则"],
    },
    "procedure_core": {
        "label": "流程制度核心章节",
        "sections": ["办理流程", "申请条件", "申请材料"],
    },
    "contract_core": {
        "label": "合同文本核心章节",
        "sections": ["合同当事人", "金额与期限", "违约责任", "争议解决"],
    },
    "risk_core": {
        "label": "风险调查核心章节",
        "sections": ["风险控制", "调查结论"],
    },
}


def resolve_profile_sections(document_type: str | None) -> List[str]:
    profile_id = (document_type or DEFAULT_PROFILE_ID).strip() or DEFAULT_PROFILE_ID
    profile = COMPLETENESS_PROFILES.get(profile_id)
    if profile is None:
        profile = COMPLETENESS_PROFILES[DEFAULT_PROFILE_ID]
    return list(profile["sections"])


def resolve_group_sections(group_ids: List[str]) -> List[str]:
    sections: List[str] = []
    seen: set[str] = set()
    for group_id in group_ids:
        group = SECTION_GROUPS.get(group_id)
        if group is None:
            continue
        for section in group["sections"]:
            if section in seen:
                continue
            seen.add(section)
            sections.append(section)
    return sections or list(DEFAULT_FALLBACK_SECTIONS)
