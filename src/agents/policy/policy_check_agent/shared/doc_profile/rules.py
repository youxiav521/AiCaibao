from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .schemas import DocType


@dataclass(frozen=True)
class RulePack:
    filename_terms: tuple[str, ...]
    strong_terms: tuple[str, ...]
    weak_terms: tuple[str, ...]


DOC_TYPE_RULES: Dict[DocType, RulePack] = {
    DocType.BUSINESS_LICENSE: RulePack(
        filename_terms=("营业执照", "business_license"),
        strong_terms=("营业执照", "统一社会信用代码", "法定代表人", "成立日期", "经营范围", "登记机关"),
        weak_terms=("社会信用代码", "注册资本", "住所", "类型"),
    ),
    DocType.LEGAL_REP_ID: RulePack(
        filename_terms=("法人身份证", "法定代表人身份证", "legal_rep_id"),
        strong_terms=("居民身份证", "中华人民共和国居民身份证", "姓名", "公民身份号码", "法定代表人", "法人"),
        weak_terms=("身份证号", "签发机关", "有效期限"),
    ),
    DocType.LEGAL_REP_MARRIAGE_CERT: RulePack(
        filename_terms=("结婚证", "婚姻", "marriage"),
        strong_terms=("结婚证", "婚姻登记", "持证人", "登记日期"),
        weak_terms=("婚姻状况", "配偶", "登记机关"),
    ),
    DocType.SPOUSE_ID: RulePack(
        filename_terms=("配偶身份证", "spouse_id"),
        strong_terms=("配偶", "居民身份证", "身份证号码", "与法定代表人关系"),
        weak_terms=("婚姻", "配偶姓名"),
    ),
    DocType.LEGAL_REP_CERT: RulePack(
        filename_terms=("法人证明", "法定代表人证明", "legal_rep_cert"),
        strong_terms=("法定代表人身份证明书", "法定代表人证明", "兹证明", "法定代表人"),
        weak_terms=("代表公司", "任职", "证明书"),
    ),
    DocType.ARTICLES_OF_ASSOCIATION: RulePack(
        filename_terms=("公司章程", "章程", "articles"),
        strong_terms=("公司章程", "股东会", "董事会", "监事会", "利润分配"),
        weak_terms=("章程修正案", "表决权", "注册资本"),
    ),
    DocType.SIGNATURE_SAMPLE: RulePack(
        filename_terms=("签字样本", "签名样本", "名单", "sample"),
        strong_terms=("签字样本", "签名样本", "印鉴", "预留签章", "授权签字人"),
        weak_terms=("名单", "样本", "签章"),
    ),
    DocType.CREDIT_REPORT: RulePack(
        filename_terms=("征信", "信用报告", "credit_report"),
        strong_terms=("征信报告", "信用报告", "人民银行征信", "信贷记录", "逾期"),
        weak_terms=("查询记录", "授信", "不良记录"),
    ),
    DocType.COLLATERAL_APPRAISAL: RulePack(
        filename_terms=("评估报告", "抵押物", "质押物", "appraisal"),
        strong_terms=("评估报告", "评估价值", "市场价值", "抵押物", "质押物", "估价师"),
        weak_terms=("估值", "评估机构", "评估结论"),
    ),
    DocType.CAPITAL_VERIFICATION_REPORT: RulePack(
        filename_terms=("验资报告", "capital_verification"),
        strong_terms=("验资报告", "实收资本", "注册资本", "会计师事务所"),
        weak_terms=("验资", "出资", "资本"),
    ),
    DocType.CONTRACT_AND_PAYMENT: RulePack(
        filename_terms=("合同", "协议", "付款", "凭证", "contract", "payment"),
        strong_terms=("合同", "协议", "付款凭证", "收据", "发票", "转账", "回单"),
        weak_terms=("甲方", "乙方", "违约", "支付", "结算"),
    ),
    DocType.OWNERSHIP_STRUCTURE: RulePack(
        filename_terms=("股权结构", "股权", "ownership"),
        strong_terms=("股权结构", "股东", "持股比例", "实际控制人", "穿透"),
        weak_terms=("控制关系", "出资比例", "受益所有人"),
    ),
    DocType.AUDIO: RulePack(
        filename_terms=("录音", "音频", "audio"),
        strong_terms=("录音", "音频", "转写", "语音识别"),
        weak_terms=("通话", "访谈"),
    ),
    DocType.MEETING_MINUTES: RulePack(
        filename_terms=("会议纪要", "纪要", "minutes"),
        strong_terms=("会议纪要", "会议时间", "参会人员", "会议结论"),
        weak_terms=("会议地点", "议题", "表决"),
    ),
    DocType.VIDEO: RulePack(
        filename_terms=("视频", "video"),
        strong_terms=("视频", "录像", "画面", "时长"),
        weak_terms=("帧", "字幕", "镜头"),
    ),
    DocType.TAX_REG_CERT: RulePack(
        filename_terms=("税务登记证", "税务", "tax"),
        strong_terms=("税务登记证", "纳税人识别号", "税务机关", "税务登记"),
        weak_terms=("纳税", "税种", "税号"),
    ),
}


MEDIA_FILENAME_HINTS: Dict[DocType, tuple[str, ...]] = {
    DocType.AUDIO: (".mp3", ".wav", ".m4a", ".aac", ".flac"),
    DocType.VIDEO: (".mp4", ".avi", ".mov", ".mkv", ".wmv"),
}


SPOUSE_HINTS: List[str] = ["配偶", "妻子", "丈夫"]
MARRIAGE_HINTS: List[str] = ["结婚证", "婚姻登记", "婚姻状况"]
MEETING_HINTS: List[str] = ["会议纪要", "会议记录", "会议结论"]

