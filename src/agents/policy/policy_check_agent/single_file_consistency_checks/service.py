from __future__ import annotations

import asyncio
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from src.agents.content_generate_agent.utils import (
    abatch_llm_with_structured_output,
    ainvoke_llm_with_structured_output,
)

from ..shared.doc_profile import (
    DocType,
    classify_doc_profile,
    classify_doc_profile_with_llm_fallback,
    map_doc_types_to_policy_doc_types,
)
from ..schemas import ResourceFile, ResourceRepository
from ..schemas.material_models import FactRecord, MaterialStore
from .schemas import (
    CandidateExtractionResult,
    CandidateJudgment,
    ConsistencyCandidate,
    ConsistencyEvidence,
    ConsistencyIssue,
    ConsistencySourceLocation,
    CrossReference,
    DocumentSegment,
    FactMention,
    LlmCandidateOutline,
    SingleDocConsistencyCheckRequest,
    SingleDocConsistencyCheckResult,
    TermDefinition,
)
from ..utils import normalize_whitespace, stable_id
from ..services.material_cleaner import MaterialCleaner
from ..services.material_loader import build_material_store
from loguru import logger


TERM_PATTERNS = [
    re.compile(
        r"(?:本办法|本制度|本规定|本指引)?所称[“\"《]?([^”\"》]{2,20})[”\"》]?(?:，|,)?"
        r"(?:是指|系指)([^。；\n]{4,120})"
    ),
    re.compile(
        r"([^\s，。；“\"《》]{2,20})\s*(?:以下简称|下称)\s*[“\"《]?([^”\"》]{2,20})[”\"》]?"
    ),
]

FACT_RULES = [
    ("loan_amount", "借款金额", re.compile(r"(借款金额|贷款金额|授信额度|合同金额|金额)[:：]?\s*(人民币)?\s*([0-9][0-9,\.]*\s*(?:亿元|万元|元)?)")),
    ("effective_date", "生效日期", re.compile(r"(生效日期|签订日期|起始日期|开始日期|放款日期|到期日期|截止日期)[:：]?\s*([0-9]{4}[年/\.-][0-9]{1,2}[月/\.-][0-9]{1,2}日?)")),
    ("term_limit", "期限", re.compile(r"(借款期限|授信期限|期限|有效期)[:：]?\s*([0-9一二三四五六七八九十百\d]+\s*(?:年|个月|月|天))")),
    ("rate_value", "利率比例", re.compile(r"(利率|费率|比例|保证金比例)[:：]?\s*([0-9]+(?:\.[0-9]+)?\s*%?)")),
    ("contract_identifier", "编号", re.compile(r"(合同编号|协议编号|附件编号|条款编号|项目编号|产品名称|项目名称)[:：]?\s*([A-Za-z0-9\u4e00-\u9fff\-_\/]+)")),
    ("role_party", "主体角色", re.compile(r"(借款人|申请人|担保人|抵押人|出质人|贷款人|甲方|乙方|丙方)[:：]?\s*([^\n，。；]{2,60})")),
]

DOC_TYPE_EXTRA_FACT_RULES: Dict[DocType, List[tuple[str, str, re.Pattern[str]]]] = {
    DocType.LEGAL_REP_ID: [
        ("id_card_number", "身份证号", re.compile(r"(公民身份号码|身份证号?)[:：]?\s*([0-9Xx]{15,18})")),
        ("person_name", "姓名", re.compile(r"(姓名)[:：]?\s*([A-Za-z\u4e00-\u9fff·]{2,20})")),
        ("gender", "性别", re.compile(r"(性别)[:：]?\s*(男|女)")),
        ("ethnicity", "民族", re.compile(r"(民族)[:：]?\s*([A-Za-z\u4e00-\u9fff]{1,10})")),
        ("birth_date", "出生日期", re.compile(r"(出生)[:：]?\s*([0-9]{4}[年\./-][0-9]{1,2}[月\./-][0-9]{1,2}日?)")),
        ("issue_authority", "签发机关", re.compile(r"(签发机关)[:：]?\s*([^\n，。；]{2,40})")),
    ],
    DocType.SPOUSE_ID: [
        ("id_card_number", "身份证号", re.compile(r"(公民身份号码|身份证号?)[:：]?\s*([0-9Xx]{15,18})")),
        ("person_name", "姓名", re.compile(r"(姓名)[:：]?\s*([A-Za-z\u4e00-\u9fff·]{2,20})")),
    ],
    DocType.BUSINESS_LICENSE: [
        ("uscc_code", "统一社会信用代码", re.compile(r"(统一社会信用代码)[:：]?\s*([0-9A-Z]{18})")),
        ("company_name", "企业名称", re.compile(r"(名称|企业名称|公司名称)[:：]?\s*([^\n，。；]{2,80})")),
        ("legal_representative", "法定代表人", re.compile(r"(法定代表人)[:：]?\s*([A-Za-z\u4e00-\u9fff·]{2,20})")),
        ("registered_capital", "注册资本", re.compile(r"(注册资本)[:：]?\s*([0-9][0-9,\.]*\s*(?:万元|亿元|元|万人民币))")),
        ("established_date", "成立日期", re.compile(r"(成立日期)[:：]?\s*([0-9]{4}[年\./-][0-9]{1,2}[月\./-][0-9]{1,2}日?)")),
    ],
    DocType.TAX_REG_CERT: [
        ("taxpayer_id", "纳税人识别号", re.compile(r"(纳税人识别号|税号)[:：]?\s*([0-9A-Z]{15,20})")),
        ("taxpayer_name", "纳税人名称", re.compile(r"(纳税人名称|名称|单位名称)[:：]?\s*([^\n，。；]{2,80})")),
        ("tax_authority", "主管税务机关", re.compile(r"(主管税务机关|税务机关)[:：]?\s*([^\n，。；]{2,60})")),
    ],
    DocType.ARTICLES_OF_ASSOCIATION: [
        ("shareholder_name", "股东名称", re.compile(r"(股东|发起人)[:：]?\s*([A-Za-z\u4e00-\u9fff·（）()]{2,80})")),
        ("shareholding_ratio", "持股比例", re.compile(r"(持股比例|出资比例)[:：]?\s*([0-9]+(?:\.[0-9]+)?\s*%)")),
    ],
    DocType.OWNERSHIP_STRUCTURE: [
        ("shareholder_name", "股东名称", re.compile(r"(股东|股东名称|出资方)[:：]?\s*([A-Za-z\u4e00-\u9fff·（）()]{2,80})")),
        ("shareholding_ratio", "持股比例", re.compile(r"(持股比例|出资比例|股权比例)[:：]?\s*([0-9]+(?:\.[0-9]+)?\s*%)")),
        ("controller_name", "实际控制人", re.compile(r"(实际控制人|控制人)[:：]?\s*([A-Za-z\u4e00-\u9fff·（）()]{2,80})")),
    ],
    DocType.CONTRACT_AND_PAYMENT: [
        ("payment_amount", "支付金额", re.compile(r"(付款金额|支付金额|实付金额)[:：]?\s*(人民币)?\s*([0-9][0-9,\.]*\s*(?:万元|亿元|元)?)")),
        ("receipt_no", "回单号", re.compile(r"(回单号|流水号|交易号)[:：]?\s*([A-Za-z0-9\-_]{6,40})")),
    ],
}

REFERENCE_PATTERNS = [
    ("clause", re.compile(r"(第[一二三四五六七八九十百千0-9]+条)")),
    ("appendix", re.compile(r"(附件[一二三四五六七八九十百千0-9A-Za-z]+)")),
    ("table", re.compile(r"(表[一二三四五六七八九十百千0-9A-Za-z]+)")),
]

ANCHOR_PATTERNS = [
    re.compile(r"^(第[一二三四五六七八九十百千0-9]+条)"),
    re.compile(r"^(附件[一二三四五六七八九十百千0-9A-Za-z]+)"),
    re.compile(r"^(表[一二三四五六七八九十百千0-9A-Za-z]+)"),
]

ACCEPTED_STORE_FACT_FIELDS = {
    "company_name",
    "person_name",
    "uscc",
    "id_card_number",
    "taxpayer_id",
    "tax_no",
    "legal_representative",
    "registered_capital",
    "established_date",
    "shareholder_name",
    "shareholding_ratio",
    "controller_name",
    "property_owner",
    "property_usage_term",
    "vehicle_owner",
    "vehicle_registration_number",
    "vehicle_registration_date",
    "vehicle_vin",
    "vehicle_issue_date",
}

DEFAULT_CHECK_TYPES = {
    "term_definition_consistency",
    "fact_value_consistency",
    "cross_reference_consistency",
}

DOC_TYPE_CHECK_STRATEGY: Dict[DocType, Dict[str, Any]] = {
    DocType.BUSINESS_LICENSE: {
        "check_types": {"fact_value_consistency", "cross_reference_consistency"},
        "slot_hints": ["统一社会信用代码", "法定代表人", "注册资本", "成立日期", "经营范围", "uscc", "company_name"],
    },
    DocType.LEGAL_REP_ID: {
        "check_types": {"fact_value_consistency"},
        "slot_hints": ["身份证", "公民身份号码", "姓名", "签发机关", "有效期限", "person_name"],
    },
    DocType.LEGAL_REP_MARRIAGE_CERT: {
        "check_types": {"fact_value_consistency"},
        "slot_hints": ["婚姻", "结婚", "配偶", "登记日期"],
    },
    DocType.SPOUSE_ID: {
        "check_types": {"fact_value_consistency"},
        "slot_hints": ["配偶", "身份证", "公民身份号码", "姓名"],
    },
    DocType.LEGAL_REP_CERT: {
        "check_types": {"term_definition_consistency", "fact_value_consistency"},
        "slot_hints": ["法定代表人", "任职", "姓名", "身份证"],
    },
    DocType.ARTICLES_OF_ASSOCIATION: {
        "check_types": {
            "term_definition_consistency",
            "fact_value_consistency",
            "cross_reference_consistency",
        },
        "slot_hints": ["股东", "董事", "监事", "注册资本", "章程", "持股", "company_name"],
    },
    DocType.SIGNATURE_SAMPLE: {
        "check_types": {"fact_value_consistency"},
        "slot_hints": ["签字", "签章", "印鉴", "授权签字人"],
    },
    DocType.CREDIT_REPORT: {
        "check_types": {"term_definition_consistency", "fact_value_consistency"},
        "slot_hints": ["征信", "授信", "逾期", "查询记录", "person_name", "company_name"],
    },
    DocType.COLLATERAL_APPRAISAL: {
        "check_types": {"fact_value_consistency", "cross_reference_consistency"},
        "slot_hints": ["评估", "抵押", "质押", "权属", "所有权", "估值", "property_owner", "vehicle_owner"],
    },
    DocType.CAPITAL_VERIFICATION_REPORT: {
        "check_types": {"term_definition_consistency", "fact_value_consistency"},
        "slot_hints": ["验资", "实收资本", "注册资本", "出资"],
    },
    DocType.CONTRACT_AND_PAYMENT: {
        "check_types": {
            "term_definition_consistency",
            "fact_value_consistency",
            "cross_reference_consistency",
        },
        "slot_hints": [
            "合同",
            "协议",
            "金额",
            "期限",
            "利率",
            "甲方",
            "乙方",
            "付款",
            "回单",
            "loan_amount",
            "effective_date",
            "term_limit",
            "rate_value",
            "contract_identifier",
            "role_party",
        ],
    },
    DocType.OWNERSHIP_STRUCTURE: {
        "check_types": {"term_definition_consistency", "fact_value_consistency"},
        "slot_hints": ["股权", "持股", "股东", "控制人", "受益所有人", "company_name"],
    },
    DocType.AUDIO: {
        "check_types": {"term_definition_consistency"},
        "slot_hints": ["录音", "音频", "转写", "会议"],
    },
    DocType.MEETING_MINUTES: {
        "check_types": {
            "term_definition_consistency",
            "fact_value_consistency",
            "cross_reference_consistency",
        },
        "slot_hints": ["会议", "议题", "决议", "纪要", "表决"],
    },
    DocType.VIDEO: {
        "check_types": {"term_definition_consistency"},
        "slot_hints": ["视频", "录像", "转写"],
    },
    DocType.TAX_REG_CERT: {
        "check_types": {"fact_value_consistency"},
        "slot_hints": ["税务", "纳税人识别号", "税号", "登记", "uscc", "company_name"],
    },
    DocType.UNKNOWN: {
        "check_types": set(DEFAULT_CHECK_TYPES),
        "slot_hints": [],
    },
}


class SingleDocConsistencyService:
    def __init__(
        self,
        *,
        model_id: str = "model_bailian_policy_check",
        max_candidates: int = 24,
        max_judge_concurrency: int = 12,
        enable_llm: bool = True,
        llm_candidate_window_size: int = 10,
        llm_candidate_window_overlap: int = 2,
        llm_candidate_limit: int = 8,
        llm_retry_attempts: int = 3,
        llm_retry_initial_delay: float = 1.5,
        candidate_batch_func: Optional[
            Callable[[List[str], str], Awaitable[List[CandidateExtractionResult]]]
        ] = None,
        judge_batch_func: Optional[Callable[[List[str], str], Awaitable[List[CandidateJudgment]]]] = None,
    ) -> None:
        self.model_id = model_id
        self.max_candidates = max_candidates
        self.max_judge_concurrency = max_judge_concurrency
        self.enable_llm = enable_llm
        self.llm_candidate_window_size = llm_candidate_window_size
        self.llm_candidate_window_overlap = llm_candidate_window_overlap
        self.llm_candidate_limit = llm_candidate_limit
        self.llm_retry_attempts = llm_retry_attempts
        self.llm_retry_initial_delay = llm_retry_initial_delay
        self._uses_default_candidate_batch = candidate_batch_func is None
        self._uses_default_judge_batch = judge_batch_func is None
        self._candidate_batch_func = candidate_batch_func or self._default_candidate_batch
        self._judge_batch_func = judge_batch_func or self._default_judge_batch
        self.cleaner = MaterialCleaner()

    async def analyze_request(self, request: SingleDocConsistencyCheckRequest) -> SingleDocConsistencyCheckResult:
        resource_file = self._build_resource_file(request)
        return await self.analyze_resource_file(
            resource_file,
            metadata=request.metadata,
            check_types=request.check_types,
            max_candidates=request.max_candidates,
            enable_llm=request.enable_llm,
            effective_model_id=request.model_id or self.model_id,
            effective_max_judge_concurrency=request.max_judge_concurrency,
            effective_llm_retry_attempts=request.llm_retry_attempts,
        )

    async def analyze_resource_file(
        self,
        resource_file: ResourceFile,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        check_types: Optional[List[str]] = None,
        max_candidates: Optional[int] = None,
        enable_llm: Optional[bool] = None,
        effective_model_id: Optional[str] = None,
        effective_max_judge_concurrency: Optional[int] = None,
        effective_llm_retry_attempts: Optional[int] = None,
    ) -> SingleDocConsistencyCheckResult:
        logger.info("single_doc_consistency start file_id={} file_name={}", resource_file.fileId, resource_file.fileName)
        resolved_model_id = effective_model_id or self.model_id
        resolved_max_judge_concurrency = self._validate_max_judge_concurrency(
            effective_max_judge_concurrency
            if effective_max_judge_concurrency is not None
            else self.max_judge_concurrency
        )
        resolved_llm_retry_attempts = self._validate_llm_retry_attempts(
            effective_llm_retry_attempts if effective_llm_retry_attempts is not None else self.llm_retry_attempts
        )
        repository = ResourceRepository(
            id=stable_id("repo", resource_file.fileId, resource_file.fileName),
            title=resource_file.fileName,
            description="single_doc_consistency",
            files=[resource_file],
        )
        cleaned_items = self.cleaner.clean_file(resource_file)
        store = build_material_store(repository, case_id=repository.id)

        effective_llm_for_profile = self.enable_llm if enable_llm is None else enable_llm
        if effective_llm_for_profile:
            doc_profile = await classify_doc_profile_with_llm_fallback(
                file_name=resource_file.fileName or "",
                doc_content=resource_file.content or [],
                model_id=resolved_model_id,
                enable_llm_fallback=True,
            )
        else:
            doc_profile = classify_doc_profile(
                file_name=resource_file.fileName or "",
                doc_content=resource_file.content or [],
            )
        policy_doc_types = sorted(map_doc_types_to_policy_doc_types(doc_profile.doc_types))
        requested_check_types = self._resolve_requested_check_types(check_types)
        check_strategy = self._resolve_doc_type_check_strategy(
            primary_doc_type=doc_profile.primary_type,
            requested_check_types=requested_check_types,
        )
        effective_check_types = check_strategy["check_types"]
        slot_hints = check_strategy["slot_hints"]

        segments = self._build_segments(resource_file, store, cleaned_items)
        anchors = self._extract_anchor_labels(cleaned_items)
        terms = self._extract_term_definitions(segments)
        facts = self._extract_fact_mentions(
            segments,
            store,
            doc_type=doc_profile.primary_type,
        )
        if slot_hints:
            facts = self._filter_fact_mentions_by_slot_hints(facts, slot_hints)
        references = self._extract_references(segments)

        logger.info(
            "single_doc_consistency prepared file_id={} segments={} terms={} facts={} references={} policy_doc_types={}",
            resource_file.fileId,
            len(segments),
            len(terms),
            len(facts),
            len(references),
            policy_doc_types,
        )

        issues: List[ConsistencyIssue] = []
        if "cross_reference_consistency" in effective_check_types:
            issues.extend(self._check_missing_references(resource_file.fileId, references, anchors))

        candidate_limit = max_candidates or self.max_candidates
        rule_candidates = self._build_rule_candidates(
            resource_file.fileId,
            segments,
            terms,
            facts,
            references,
            anchors,
            candidate_limit,
            enabled_check_types=effective_check_types,
        )
        candidates = list(rule_candidates)
        effective_llm = effective_llm_for_profile
        if effective_llm and segments:
            try:
                llm_candidates = await self._build_llm_candidates(
                    resource_file.fileId,
                    segments,
                    candidate_limit,
                    effective_model_id=resolved_model_id,
                    effective_max_judge_concurrency=resolved_max_judge_concurrency,
                    effective_llm_retry_attempts=resolved_llm_retry_attempts,
                )
                llm_candidates = self._filter_candidates_by_check_types(llm_candidates, effective_check_types)
                candidates = self._merge_candidates(rule_candidates + llm_candidates, candidate_limit)
            except Exception as exc:
                logger.warning(
                    "single_doc_consistency llm_candidate_generation_failed file_id={} error={}",
                    resource_file.fileId,
                    exc,
                )
        logger.info(
            "single_doc_consistency generated candidates file_id={} candidates={} limit={}",
            resource_file.fileId,
            len(candidates),
            candidate_limit,
        )

        judged_issues: List[ConsistencyIssue] = []
        if candidates:
            if effective_llm:
                try:
                    judged_issues = await self._judge_candidates(
                        resource_file.fileId,
                        candidates,
                        effective_model_id=resolved_model_id,
                        effective_max_judge_concurrency=resolved_max_judge_concurrency,
                        effective_llm_retry_attempts=resolved_llm_retry_attempts,
                    )
                except Exception as exc:
                    logger.warning("single_doc_consistency llm_judgment_failed file_id={} error={}", resource_file.fileId, exc)
                    judged_issues = self._fallback_judge_candidates(resource_file.fileId, candidates)
            else:
                judged_issues = self._fallback_judge_candidates(resource_file.fileId, candidates)

        merged_issues = self._merge_issues(issues + judged_issues)
        result = self._build_result(
            resource_file,
            merged_issues,
            metadata=self._compose_result_metadata(
                metadata=metadata or {},
                doc_profile=doc_profile,
                effective_check_types=effective_check_types,
                slot_hints=slot_hints,
            ),
        )
        logger.info(
            "single_doc_consistency complete file_id={} overall_status={} issue_count={}",
            resource_file.fileId,
            result.overall_status,
            result.issue_count,
        )
        return result

    def _build_resource_file(self, request: SingleDocConsistencyCheckRequest) -> ResourceFile:
        if request.doc_content:
            file_id = request.file_id or self._build_default_file_id(request.file_name, request.doc_content)
            file_name = request.file_name or f"{file_id}.json"
            return ResourceFile.model_validate(
                {
                    "fileId": file_id,
                    "fileName": file_name,
                    "content": request.doc_content,
                }
            )

        if not request.input_path:
            raise ValueError("input_path or docContent is required")
        input_path = Path(request.input_path)
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if isinstance(raw.get("docContent"), list):
                doc_content = raw["docContent"]
            elif isinstance(raw.get("content"), list):
                doc_content = raw["content"]
            else:
                raise ValueError("input_path json must contain docContent/content list")
        elif isinstance(raw, list):
            doc_content = raw
        else:
            raise ValueError("input_path json must be a list or dict with docContent")
        return ResourceFile.model_validate(
            {
                "fileId": request.file_id or input_path.stem,
                "fileName": request.file_name or input_path.name,
                "content": doc_content,
            }
        )

    @staticmethod
    def _build_default_file_id(file_name: Optional[str], doc_content: List[Dict[str, Any]]) -> str:
        payload = {
            "fileName": file_name or "",
            "totalBlocks": len(doc_content),
            "headContent": doc_content[:20],
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"file_{digest}"

    @staticmethod
    def _validate_max_judge_concurrency(value: Any) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("max_judge_concurrency must be an integer >= 1")
        return value

    @staticmethod
    def _validate_llm_retry_attempts(value: Any) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("llm_retry_attempts must be an integer >= 1")
        return value

    def _build_segments(
        self,
        resource_file: ResourceFile,
        store: MaterialStore,
        cleaned_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DocumentSegment]:
        segments: List[DocumentSegment] = []
        for block in store.evidence_blocks:
            text = normalize_whitespace(block.text)
            if not text or self._is_noise_segment_text(text):
                continue
            segments.append(
                DocumentSegment(
                    segment_id=block.evidence_id,
                    file_id=resource_file.fileId,
                    text=text,
                    page_no=block.page,
                    element_ids=list(block.source_element_ids or []),
                    block_type=block.block_type,
                    anchor_label=self._extract_inline_anchor(text),
                )
            )
        cleaned_source = cleaned_items if cleaned_items is not None else self.cleaner.clean_file(resource_file)
        for cleaned in cleaned_source:
            segment = self._cleaned_item_to_segment(resource_file.fileId, cleaned)
            if segment is not None:
                segments.append(segment)
        return self._dedup_segments(segments)

    def _extract_term_definitions(self, segments: Iterable[DocumentSegment]) -> List[TermDefinition]:
        definitions: List[TermDefinition] = []
        for segment in segments:
            for pattern in TERM_PATTERNS:
                for match in pattern.finditer(segment.text):
                    left = normalize_whitespace(match.group(1))
                    right = normalize_whitespace(match.group(2))
                    if len(left) < 2 or len(right) < 2:
                        continue
                    if "以下简称" in match.group(0) or "下称" in match.group(0):
                        term = right
                        aliases = [left]
                        definition = f"{left} 在后文简称为 {right}"
                    else:
                        term = left
                        aliases = []
                        definition = right
                    definitions.append(
                        TermDefinition(
                            term_id=stable_id("term", segment.segment_id, term, definition),
                            term=term,
                            normalized_term=self._normalize_term(term),
                            definition=definition,
                            aliases=aliases,
                            segment_id=segment.segment_id,
                            evidence=self._segment_evidence(role="definition", text=match.group(0), segment=segment),
                        )
                    )
        return definitions

    def _extract_fact_mentions(
        self,
        segments: Iterable[DocumentSegment],
        store: MaterialStore,
        *,
        doc_type: DocType = DocType.UNKNOWN,
    ) -> List[FactMention]:
        segment_map = {segment.segment_id: segment for segment in segments}
        mentions: List[FactMention] = []
        effective_fact_rules = self._build_effective_fact_rules(doc_type)
        for fact in store.facts:
            slot_key = self._canonical_slot_from_fact_record(fact)
            if not slot_key:
                continue
            segment = segment_map.get(fact.evidence_id)
            if segment is None:
                continue
            if self._should_skip_fact_mention(segment, slot_key, fact.field_value, fact.field_value):
                continue
            mentions.append(
                FactMention(
                    fact_id=stable_id("fact_slot", fact.fact_id, slot_key),
                    slot_key=slot_key,
                    slot_label=fact.field_name,
                    value=fact.field_value,
                    normalized_value=self._normalize_fact_value(slot_key, fact.normalized_value or fact.field_value),
                    segment_id=segment.segment_id,
                    evidence=self._segment_evidence(role="mention_a", text=fact.field_value, segment=segment),
                )
            )

        for segment in segments:
            for slot_key, slot_label, pattern in effective_fact_rules:
                for match in pattern.finditer(segment.text):
                    value = normalize_whitespace(match.group(match.lastindex or 1))
                    label = normalize_whitespace(match.group(1))
                    if not value:
                        continue
                    canonical_slot = self._canonical_slot(slot_key, label or slot_label)
                    if self._should_skip_fact_mention(segment, canonical_slot, value, match.group(0)):
                        continue
                    mentions.append(
                        FactMention(
                            fact_id=stable_id("fact_slot", segment.segment_id, canonical_slot, value),
                            slot_key=canonical_slot,
                            slot_label=label or slot_label,
                            value=value,
                            normalized_value=self._normalize_fact_value(canonical_slot, value),
                            segment_id=segment.segment_id,
                            evidence=self._segment_evidence(role="mention_a", text=match.group(0), segment=segment),
                        )
                    )
        return self._dedup_fact_mentions(mentions)

    def _extract_references(self, segments: Iterable[DocumentSegment]) -> List[CrossReference]:
        references: List[CrossReference] = []
        for segment in segments:
            for reference_type, pattern in REFERENCE_PATTERNS:
                for match in pattern.finditer(segment.text):
                    label = normalize_whitespace(match.group(1))
                    references.append(
                        CrossReference(
                            reference_id=stable_id("ref", segment.segment_id, reference_type, label),
                            reference_type=reference_type,
                            reference_label=label,
                            segment_id=segment.segment_id,
                            evidence=self._segment_evidence(role="reference", text=match.group(0), segment=segment),
                        )
                    )
        return self._dedup_references(references)

    def _extract_anchor_labels(self, cleaned_items: Iterable[Dict[str, Any]]) -> Dict[str, str]:
        anchors: Dict[str, str] = {}
        for item in cleaned_items:
            text = normalize_whitespace(item.get("text") or "")
            if not text:
                continue
            for pattern in ANCHOR_PATTERNS:
                match = pattern.search(text)
                if match:
                    label = normalize_whitespace(match.group(1))
                    anchors[self._normalize_reference_label(label)] = label
        return anchors

    def _build_rule_candidates(
        self,
        file_id: str,
        segments: List[DocumentSegment],
        terms: List[TermDefinition],
        facts: List[FactMention],
        references: List[CrossReference],
        anchors: Dict[str, str],
        max_candidates: int,
        *,
        enabled_check_types: set[str],
    ) -> List[ConsistencyCandidate]:
        candidates: List[ConsistencyCandidate] = []

        if "term_definition_consistency" in enabled_check_types:
            for term in terms:
                usages = self._find_term_usages(term, segments)
                for usage in usages[:4]:
                    candidates.append(
                        ConsistencyCandidate(
                            candidate_id=stable_id("cand_term", term.term_id, usage.segment_id),
                            candidate_type="term_usage",
                            subject=term.term,
                            normalized_key=term.normalized_term,
                            issue_type_hint="TERM_USE_MISMATCH",
                            left_evidence=term.evidence.model_copy(update={"role": "definition"}),
                            right_evidence=self._segment_evidence(role="usage", text=usage.text, segment=usage),
                            left_summary=term.definition[:240],
                            right_summary=usage.text[:240],
                            attributes={"aliases": term.aliases},
                        )
                    )

        if "fact_value_consistency" in enabled_check_types:
            grouped_facts: Dict[str, List[FactMention]] = {}
            for fact in facts:
                grouped_facts.setdefault(fact.slot_key, []).append(fact)
            for slot_key, mentions in grouped_facts.items():
                distinct_by_value: Dict[str, FactMention] = {}
                for mention in mentions:
                    distinct_by_value.setdefault(mention.normalized_value, mention)
                distinct_mentions = list(distinct_by_value.values())
                if len(distinct_mentions) <= 1:
                    continue
                for left_index in range(len(distinct_mentions) - 1):
                    for right_index in range(left_index + 1, len(distinct_mentions)):
                        left = distinct_mentions[left_index]
                        right = distinct_mentions[right_index]
                        candidates.append(
                            ConsistencyCandidate(
                                candidate_id=stable_id("cand_fact", slot_key, left.fact_id, right.fact_id),
                                candidate_type="fact_pair",
                                subject=slot_key,
                                normalized_key=slot_key,
                                issue_type_hint=self._fact_issue_type(slot_key),
                                left_evidence=left.evidence.model_copy(update={"role": "mention_a"}),
                                right_evidence=right.evidence.model_copy(update={"role": "mention_b"}),
                                left_summary=f"{left.slot_label}: {left.value}",
                                right_summary=f"{right.slot_label}: {right.value}",
                                attributes={"leftValue": left.normalized_value, "rightValue": right.normalized_value},
                            )
                        )

        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        logger.info(
            "single_doc_consistency candidate_breakdown file_id={} term_candidates={} fact_candidates={} reference_checks={} anchors={}",
            file_id,
            len([item for item in candidates if item.candidate_type == "term_usage"]),
            len([item for item in candidates if item.candidate_type == "fact_pair"]),
            len(references),
            len(anchors),
        )
        return candidates

    async def _build_llm_candidates(
        self,
        file_id: str,
        segments: List[DocumentSegment],
        limit: int,
        *,
        effective_model_id: str,
        effective_max_judge_concurrency: int,
        effective_llm_retry_attempts: int,
    ) -> List[ConsistencyCandidate]:
        prompts = self._build_candidate_extraction_prompts(segments)
        if not prompts:
            return []
        if self._uses_default_candidate_batch:
            batch_func = lambda: self._default_candidate_batch(
                prompts,
                effective_model_id,
                max_judge_concurrency=effective_max_judge_concurrency,
            )
        else:
            batch_func = lambda: self._candidate_batch_func(prompts, effective_model_id)
        extraction_results = await self._call_llm_batch_with_retries(
            batch_func,
            stage="candidate_generation",
            file_id=file_id,
            llm_retry_attempts=effective_llm_retry_attempts,
        )
        segment_by_id = {segment.segment_id: segment for segment in segments}
        candidates: List[ConsistencyCandidate] = []
        for extraction in extraction_results:
            for item in extraction.candidates[: self.llm_candidate_limit]:
                candidate = self._candidate_from_outline(file_id, item, segment_by_id)
                if candidate is not None:
                    candidates.append(candidate)
        merged = self._merge_candidates(candidates, limit)
        logger.info(
            "single_doc_consistency llm_candidates file_id={} prompts={} extracted={} merged={}",
            file_id,
            len(prompts),
            sum(len(item.candidates) for item in extraction_results),
            len(merged),
        )
        return merged

    def _build_candidate_extraction_prompts(self, segments: List[DocumentSegment]) -> List[str]:
        if not segments:
            return []
        prompt_template = self._load_candidate_extractor_prompt()
        prompts: List[str] = []
        if len(segments) <= 12:
            prompts.append(self._render_candidate_extraction_prompt(prompt_template, segments))
            return prompts

        if len(segments) <= 28:
            prompts.append(self._render_candidate_extraction_prompt(prompt_template, segments))

        window_size = max(4, self.llm_candidate_window_size)
        step = max(1, window_size - self.llm_candidate_window_overlap)
        for start in range(0, len(segments), step):
            window = segments[start : start + window_size]
            if not window:
                continue
            prompts.append(self._render_candidate_extraction_prompt(prompt_template, window))
            if start + window_size >= len(segments):
                break
        return prompts

    def _render_candidate_extraction_prompt(self, prompt_template: str, segments: List[DocumentSegment]) -> str:
        payload = {
            "segments": [
                {
                    "segmentId": segment.segment_id,
                    "pageNo": segment.page_no,
                    "blockType": segment.block_type,
                    "anchorLabel": segment.anchor_label,
                    "elementIds": segment.element_ids,
                    "text": segment.text[:600],
                }
                for segment in segments
            ]
        }
        return f"{prompt_template}\n\n[segments]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    def _candidate_from_outline(
        self,
        file_id: str,
        outline: LlmCandidateOutline,
        segment_by_id: Dict[str, DocumentSegment],
    ) -> Optional[ConsistencyCandidate]:
        left_segment = segment_by_id.get(outline.left_segment_id)
        if left_segment is None:
            return None
        right_segment = segment_by_id.get(outline.right_segment_id) if outline.right_segment_id else None
        return ConsistencyCandidate(
            candidate_id=stable_id(
                "candidate_llm",
                file_id,
                outline.candidate_type,
                outline.left_segment_id,
                outline.right_segment_id or "",
                outline.issue_type_hint,
            ),
            candidate_type=outline.candidate_type,
            subject=outline.subject,
            normalized_key=self._normalize_candidate_key(outline),
            issue_type_hint=outline.issue_type_hint,
            left_evidence=self._segment_evidence(role="mention_a", text=left_segment.text, segment=left_segment),
            right_evidence=(
                self._segment_evidence(role="mention_b", text=right_segment.text, segment=right_segment)
                if right_segment is not None
                else None
            ),
            left_summary=outline.left_summary or "",
            right_summary=outline.right_summary or "",
            attributes=dict(outline.attributes),
        )

    async def _default_candidate_batch(
        self,
        prompts: List[str],
        model_id: str,
        *,
        max_judge_concurrency: int,
    ) -> List[CandidateExtractionResult]:
        return await abatch_llm_with_structured_output(
            prompts,
            model_id=model_id,
            response_type=CandidateExtractionResult,
        )

    async def _judge_candidates(
        self,
        file_id: str,
        candidates: List[ConsistencyCandidate],
        *,
        effective_model_id: str,
        effective_max_judge_concurrency: int,
        effective_llm_retry_attempts: int,
    ) -> List[ConsistencyIssue]:
        prompts = [self._build_candidate_prompt(candidate) for candidate in candidates]
        logger.info(
            "single_doc_consistency llm_judge_start file_id={} candidates={} model_id={}",
            file_id,
            len(candidates),
            effective_model_id,
        )
        if self._uses_default_judge_batch:
            batch_func = lambda: self._default_judge_batch(
                prompts,
                effective_model_id,
                max_judge_concurrency=effective_max_judge_concurrency,
            )
        else:
            batch_func = lambda: self._judge_batch_func(prompts, effective_model_id)
        try:
            judgments = await self._call_llm_batch_with_retries(
                batch_func,
                stage="candidate_judgment",
                file_id=file_id,
                llm_retry_attempts=effective_llm_retry_attempts,
            )
        except Exception as exc:
            if not self._uses_default_judge_batch:
                raise
            logger.warning(
                "single_doc_consistency candidate_judgment_batch_failed file_id={} error={} fallback=per_item",
                file_id,
                exc,
            )
            judgments = await self._judge_candidates_one_by_one(
                file_id,
                prompts,
                effective_model_id=effective_model_id,
                effective_llm_retry_attempts=effective_llm_retry_attempts,
            )
        issues: List[ConsistencyIssue] = []
        for candidate, judgment in zip(candidates, judgments):
            issue = self._issue_from_judgment(candidate, judgment)
            if issue is not None:
                issues.append(issue)
        return issues

    async def _judge_candidates_one_by_one(
        self,
        file_id: str,
        prompts: List[str],
        *,
        effective_model_id: str,
        effective_llm_retry_attempts: int,
    ) -> List[CandidateJudgment]:
        judgments: List[CandidateJudgment] = []
        for index, prompt in enumerate(prompts, start=1):
            try:
                judgments.append(
                    await self._call_llm_batch_with_retries(
                        lambda prompt_text=prompt: self._default_judge_single(
                            self._build_repair_prompt(prompt_text),
                            model_id=effective_model_id,
                        ),
                        stage=f"candidate_judgment_single_{index}",
                        file_id=file_id,
                        retry_on_parse_error=True,
                        llm_retry_attempts=effective_llm_retry_attempts,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "single_doc_consistency candidate_judgment_single_failed file_id={} index={} error={}",
                    file_id,
                    index,
                    exc,
                )
                judgments.append(
                    CandidateJudgment(
                        comparable=False,
                        status="insufficient_information",
                        issue_type="INSUFFICIENT_INFORMATION",
                        severity="low",
                        summary="",
                        reasoning="single candidate judgment failed",
                        confidence=0.0,
                    )
                )
        return judgments

    async def _default_judge_batch(
        self,
        prompts: List[str],
        model_id: str,
        *,
        max_judge_concurrency: int,
    ) -> List[CandidateJudgment]:
        return await abatch_llm_with_structured_output(
            prompts,
            model_id=model_id,
            response_type=CandidateJudgment,
        )

    async def _default_judge_single(self, prompt: str, *, model_id: str) -> CandidateJudgment:
        return await ainvoke_llm_with_structured_output(
            prompt=prompt,
            model_id=model_id,
            response_type=CandidateJudgment,
        )

    def _fallback_judge_candidates(self, file_id: str, candidates: List[ConsistencyCandidate]) -> List[ConsistencyIssue]:
        logger.info("single_doc_consistency fallback_judge file_id={} candidates={}", file_id, len(candidates))
        issues: List[ConsistencyIssue] = []
        for candidate in candidates:
            if candidate.candidate_type == "fact_pair":
                issues.append(
                    ConsistencyIssue(
                        issue_id=stable_id("issue", candidate.candidate_id),
                        issue_type=candidate.issue_type_hint,
                        severity="medium",
                        status="possibly_inconsistent",
                        summary=f"同一事实字段出现多个不同值：{candidate.subject}",
                        reasoning=f"字段 `{candidate.subject}` 在文档中出现了多个不同取值，需要人工复核。",
                        confidence=0.55,
                        evidence=[candidate.left_evidence, candidate.right_evidence] if candidate.right_evidence else [candidate.left_evidence],
                        suggestion="请核对同一字段在正文、表格、附件中的取值是否保持一致。",
                        attributes=dict(candidate.attributes),
                    )
                )
            elif candidate.candidate_type == "term_usage":
                issues.append(
                    ConsistencyIssue(
                        issue_id=stable_id("issue", candidate.candidate_id),
                        issue_type=candidate.issue_type_hint,
                        severity="low",
                        status="possibly_inconsistent",
                        summary=f"术语 `{candidate.subject}` 的定义与使用需要人工复核。",
                        reasoning="系统发现术语定义片段与使用片段存在可疑差异，但当前未经过模型裁决。",
                        confidence=0.45,
                        evidence=[candidate.left_evidence, candidate.right_evidence] if candidate.right_evidence else [candidate.left_evidence],
                        suggestion="请确认该术语在全文中的指代对象和职责是否一致。",
                        attributes=dict(candidate.attributes),
                    )
                )
        return issues

    def _check_missing_references(
        self,
        file_id: str,
        references: List[CrossReference],
        anchors: Dict[str, str],
    ) -> List[ConsistencyIssue]:
        issues: List[ConsistencyIssue] = []
        anchor_types = {self._reference_type_from_label(label) for label in anchors.values()}
        for reference in references:
            normalized = self._normalize_reference_label(reference.reference_label)
            if normalized in anchors:
                continue
            if not self._should_enforce_missing_reference(reference, anchor_types):
                continue
            issues.append(
                ConsistencyIssue(
                    issue_id=stable_id("issue", reference.reference_id),
                    issue_type="CROSS_REFERENCE_ERROR",
                    severity="high",
                    status="inconsistent",
                    summary=f"引用 `{reference.reference_label}` 未找到对应目标。",
                    reasoning=f"文档中引用了 `{reference.reference_label}`，但在正文标题、条款或附件锚点中未发现对应目标。",
                    confidence=0.98,
                    evidence=[reference.evidence],
                    suggestion="请补充对应条款/附件/表格，或修正引用编号。",
                    attributes={"referenceType": reference.reference_type, "referenceLabel": reference.reference_label},
                )
            )
        logger.info(
            "single_doc_consistency reference_check file_id={} missing_references={} anchors={}",
            file_id,
            len(issues),
            len(anchors),
        )
        return issues

    def _build_result(
        self,
        resource_file: ResourceFile,
        issues: List[ConsistencyIssue],
        *,
        metadata: Dict[str, Any],
    ) -> SingleDocConsistencyCheckResult:
        statistics = {
            "high": sum(1 for issue in issues if issue.severity == "high"),
            "medium": sum(1 for issue in issues if issue.severity == "medium"),
            "low": sum(1 for issue in issues if issue.severity == "low"),
        }
        statuses = {issue.status for issue in issues}
        if "inconsistent" in statuses:
            overall_status = "inconsistent"
        elif "possibly_inconsistent" in statuses:
            overall_status = "possibly_inconsistent"
        elif "insufficient_information" in statuses and issues:
            overall_status = "insufficient_information"
        else:
            overall_status = "consistent"

        if not issues:
            summary = "未发现明显的一致性问题。"
        else:
            summary = (
                f"共发现 {len(issues)} 条疑似一致性问题，"
                f"其中 high={statistics['high']}、medium={statistics['medium']}、low={statistics['low']}。"
            )

        return SingleDocConsistencyCheckResult(
            document_id=stable_id("doc_consistency", resource_file.fileId, resource_file.fileName),
            file_id=resource_file.fileId,
            file_name=resource_file.fileName,
            overall_status=overall_status,
            summary=summary,
            issue_count=len(issues),
            issues=issues,
            statistics=statistics,
            metadata=metadata,
        )

    def _build_candidate_prompt(self, candidate: ConsistencyCandidate) -> str:
        prompt_template = self._load_prompt_template()
        payload = {
            "candidateType": candidate.candidate_type,
            "subject": candidate.subject,
            "normalizedKey": candidate.normalized_key,
            "issueTypeHint": candidate.issue_type_hint,
            "leftEvidence": candidate.left_evidence.model_dump(mode="json", by_alias=True),
            "rightEvidence": candidate.right_evidence.model_dump(mode="json", by_alias=True) if candidate.right_evidence else None,
            "leftSummary": candidate.left_summary,
            "rightSummary": candidate.right_summary,
            "attributes": candidate.attributes,
        }
        return f"{prompt_template}\n\n[candidate]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    def _issue_from_judgment(
        self,
        candidate: ConsistencyCandidate,
        judgment: CandidateJudgment,
    ) -> Optional[ConsistencyIssue]:
        if not judgment.comparable or judgment.status == "consistent":
            return None
        evidence = [candidate.left_evidence]
        if candidate.right_evidence is not None:
            evidence.append(candidate.right_evidence)
        issue = ConsistencyIssue(
            issue_id=stable_id("issue", candidate.candidate_id, judgment.issue_type, judgment.status),
            issue_type=judgment.issue_type or candidate.issue_type_hint,
            severity=judgment.severity,
            status=judgment.status,
            summary=judgment.summary,
            reasoning=judgment.reasoning,
            confidence=judgment.confidence,
            evidence=evidence,
            suggestion=judgment.suggestion,
            attributes=dict(candidate.attributes),
        )
        if not self._should_emit_issue(issue):
            return None
        return issue

    def _find_term_usages(self, definition: TermDefinition, segments: List[DocumentSegment]) -> List[DocumentSegment]:
        aliases = {definition.term, *definition.aliases}
        usages: List[DocumentSegment] = []
        seen = set()
        for segment in segments:
            if segment.segment_id == definition.segment_id:
                continue
            haystack = segment.text
            if not any(alias and alias in haystack for alias in aliases):
                continue
            if segment.segment_id in seen:
                continue
            seen.add(segment.segment_id)
            usages.append(segment)
        return usages

    @staticmethod
    def _canonical_slot(slot_key: str, label: str) -> str:
        normalized_label = normalize_whitespace(label).replace("：", "").replace(":", "")
        return f"{slot_key}.{normalized_label}"

    @staticmethod
    def _canonical_slot_from_fact_record(fact: FactRecord) -> Optional[str]:
        field_name = str(fact.field_name or "").strip()
        if not field_name or field_name not in ACCEPTED_STORE_FACT_FIELDS:
            return None
        return f"field.{field_name}"

    @staticmethod
    def _fact_issue_type(slot_key: str) -> str:
        if "date" in slot_key or "term" in slot_key:
            return "DATE_TERM_MISMATCH"
        if "role" in slot_key:
            return "ENTITY_ROLE_MISMATCH"
        return "FACT_VALUE_MISMATCH"

    @staticmethod
    def _normalize_term(term: str) -> str:
        return re.sub(r"[\s“”\"《》()（）]", "", term or "").lower()

    @staticmethod
    def _normalize_fact_value(slot_key: str, value: str) -> str:
        normalized = normalize_whitespace(value).lower()
        normalized = normalized.replace("人民币", "").replace(",", "")
        if "date" in slot_key:
            normalized = normalized.replace("年", "-").replace("月", "-").replace("日", "")
            normalized = normalized.replace("/", "-").replace(".", "-")
        if "rate" in slot_key and not normalized.endswith("%"):
            if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
                normalized = f"{normalized}%"
        return normalized

    @staticmethod
    def _normalize_reference_label(label: str) -> str:
        return re.sub(r"[\s（）()]", "", label or "")

    @staticmethod
    def _extract_inline_anchor(text: str) -> Optional[str]:
        for pattern in ANCHOR_PATTERNS:
            match = pattern.search(text)
            if match:
                return normalize_whitespace(match.group(1))
        return None

    @staticmethod
    def _is_noise_segment_text(text: str) -> bool:
        lowered = text.lower()
        return "rowspan=" in lowered or "colstart=" in lowered or "text='" in lowered

    def _segment_evidence(self, *, role: str, text: str, segment: DocumentSegment) -> ConsistencyEvidence:
        return ConsistencyEvidence(
            role=role,
            source_text=text[:500],
            source_location=ConsistencySourceLocation(
                file_id=segment.file_id,
                element_ids=list(segment.element_ids),
                page_no=segment.page_no,
                segment_id=segment.segment_id,
                block_type=segment.block_type,
                anchor_label=segment.anchor_label,
            ),
        )

    def _cleaned_item_to_segment(self, file_id: str, cleaned: Dict[str, Any]) -> Optional[DocumentSegment]:
        text = normalize_whitespace(cleaned.get("text") or cleaned.get("md") or "")
        if not text or self._is_noise_segment_text(text):
            return None
        block_type = str(cleaned.get("type") or "paragraph")
        return DocumentSegment(
            segment_id=stable_id("segment", file_id, cleaned.get("element_id"), block_type, text[:40]),
            file_id=file_id,
            text=text,
            page_no=cleaned.get("page"),
            element_ids=[eid for eid in [cleaned.get("element_id")] if eid is not None],
            block_type=block_type,
            anchor_label=self._extract_inline_anchor(text),
        )

    @staticmethod
    def _should_skip_fact_mention(
        segment: DocumentSegment,
        slot_key: str,
        value: str,
        raw_match_text: str,
    ) -> bool:
        normalized_value = normalize_whitespace(value)
        if segment.block_type == "table":
            if "|" in segment.text or " | " in segment.text:
                if slot_key.startswith("role_party."):
                    return True
                if any(keyword in normalized_value for keyword in ("材料", "文书", "凭证", "清单", "样本")):
                    return True
            if len(normalized_value) > 40:
                return True
        if slot_key.startswith("role_party."):
            if any(keyword in normalized_value for keyword in ("董事会", "身份证明", "授权委托书", "签字样本", "收妥凭证")):
                return True
            if any(symbol in raw_match_text for symbol in ("|", "｜")):
                return True
        if slot_key.startswith("contract_identifier."):
            if len(normalized_value) > 30:
                return True
            if any(keyword in normalized_value for keyword in ("一致", "填写", "完整", "应", "需", "必须", "要求", "主合同")):
                return True
            if not re.search(r"[0-9A-Za-z\-_/]{2,}", normalized_value):
                return True
        return False

    def _should_enforce_missing_reference(self, reference: CrossReference, anchor_types: set[Optional[str]]) -> bool:
        reference_type = reference.reference_type or self._reference_type_from_label(reference.reference_label)
        if reference_type == "clause":
            return "clause" in anchor_types
        if reference_type in {"appendix", "table"}:
            return reference_type in anchor_types
        return False

    @staticmethod
    def _reference_type_from_label(label: str) -> Optional[str]:
        normalized = normalize_whitespace(label)
        if normalized.startswith("第") and "条" in normalized:
            return "clause"
        if normalized.startswith("附件"):
            return "appendix"
        if normalized.startswith("表"):
            return "table"
        return None

    async def _call_llm_batch_with_retries(
        self,
        func: Callable[[], Awaitable[Any]],
        *,
        stage: str,
        file_id: str,
        llm_retry_attempts: int,
        retry_on_parse_error: bool = False,
    ) -> Any:
        delay = self.llm_retry_initial_delay
        last_error: Optional[Exception] = None
        for attempt in range(1, llm_retry_attempts + 1):
            try:
                return await func()
            except Exception as exc:
                last_error = exc
                if attempt >= llm_retry_attempts or not (
                    self._is_retryable_llm_error(exc)
                    or (retry_on_parse_error and self._is_structured_output_error(exc))
                ):
                    raise
                logger.warning(
                    "single_doc_consistency {} retry file_id={} attempt={}/{} delay={} error={}",
                    stage,
                    file_id,
                    attempt,
                    llm_retry_attempts,
                    round(delay, 2),
                    exc,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{stage} failed without specific error")

    @staticmethod
    def _is_retryable_llm_error(error: Exception) -> bool:
        error_text = f"{type(error).__name__} {error}".lower()
        retryable_markers = [
            "502",
            "bad gateway",
            "503",
            "service temporarily unavailable",
            "connection error",
            "connecterror",
            "connection reset",
            "connection refused",
            "timeout",
        ]
        return any(marker in error_text for marker in retryable_markers)

    @staticmethod
    def _is_structured_output_error(error: Exception) -> bool:
        error_text = f"{type(error).__name__} {error}".lower()
        markers = [
            "validationerror",
            "field required",
            "input should be",
            "structured output",
        ]
        return any(marker in error_text for marker in markers)

    @staticmethod
    def _build_repair_prompt(prompt: str) -> str:
        return (
            f"{prompt}\n\n[format_repair]\n"
            "必须返回完整结构化字段，即使信息不足也不能省略字段。"
            "至少补全 `status`、`issue_type`、`severity`、`summary`、`reasoning`、`confidence`。"
        )

    @staticmethod
    def _dedup_fact_mentions(mentions: List[FactMention]) -> List[FactMention]:
        deduped: Dict[str, FactMention] = {}
        for mention in mentions:
            key = f"{mention.slot_key}::{mention.normalized_value}::{mention.segment_id}"
            deduped.setdefault(key, mention)
        return list(deduped.values())

    @staticmethod
    def _dedup_references(references: List[CrossReference]) -> List[CrossReference]:
        deduped: Dict[str, CrossReference] = {}
        for reference in references:
            key = f"{reference.reference_type}::{reference.reference_label}::{reference.segment_id}"
            deduped.setdefault(key, reference)
        return list(deduped.values())

    @staticmethod
    def _dedup_segments(segments: List[DocumentSegment]) -> List[DocumentSegment]:
        deduped: Dict[str, DocumentSegment] = {}
        for segment in segments:
            normalized_text = normalize_whitespace(segment.text)
            if len(normalized_text) < 2:
                continue
            key = f"{segment.page_no}::{segment.block_type}::{tuple(segment.element_ids)}::{normalized_text[:200]}"
            existing = deduped.get(key)
            if existing is None or len(segment.text) > len(existing.text):
                deduped[key] = segment
        return sorted(
            deduped.values(),
            key=lambda item: ((item.page_no or 0), min(item.element_ids) if item.element_ids else 0, item.segment_id),
        )

    def _merge_candidates(self, candidates: List[ConsistencyCandidate], limit: int) -> List[ConsistencyCandidate]:
        merged: Dict[str, ConsistencyCandidate] = {}
        for candidate in candidates:
            key = self._candidate_merge_key(candidate)
            merged.setdefault(key, candidate)
        ordered = list(merged.values())
        return ordered[:limit]

    @staticmethod
    def _build_effective_fact_rules(doc_type: DocType) -> List[tuple[str, str, re.Pattern[str]]]:
        rules = list(FACT_RULES)
        rules.extend(DOC_TYPE_EXTRA_FACT_RULES.get(doc_type, []))
        return rules

    @staticmethod
    def _resolve_requested_check_types(check_types: Optional[List[str]]) -> set[str]:
        if not check_types:
            return set(DEFAULT_CHECK_TYPES)
        normalized = {item.strip() for item in check_types if item and item.strip()}
        filtered = normalized & DEFAULT_CHECK_TYPES
        return filtered or set(DEFAULT_CHECK_TYPES)

    @staticmethod
    def _resolve_doc_type_check_strategy(
        *,
        primary_doc_type: DocType,
        requested_check_types: set[str],
    ) -> Dict[str, Any]:
        strategy = DOC_TYPE_CHECK_STRATEGY.get(primary_doc_type) or DOC_TYPE_CHECK_STRATEGY[DocType.UNKNOWN]
        allowed_check_types = set(strategy.get("check_types") or DEFAULT_CHECK_TYPES)
        effective_check_types = requested_check_types & allowed_check_types
        if not effective_check_types:
            effective_check_types = requested_check_types
        return {
            "check_types": effective_check_types,
            "slot_hints": list(strategy.get("slot_hints") or []),
        }

    @staticmethod
    def _filter_fact_mentions_by_slot_hints(facts: List[FactMention], slot_hints: List[str]) -> List[FactMention]:
        if not slot_hints:
            return facts
        normalized_hints = [hint.lower() for hint in slot_hints]
        filtered: List[FactMention] = []
        for fact in facts:
            haystack = f"{fact.slot_key} {fact.slot_label} {fact.value}".lower()
            if any(hint in haystack for hint in normalized_hints):
                filtered.append(fact)
        return filtered

    def _filter_candidates_by_check_types(
        self,
        candidates: List[ConsistencyCandidate],
        enabled_check_types: set[str],
    ) -> List[ConsistencyCandidate]:
        if not candidates:
            return candidates
        return [candidate for candidate in candidates if self._candidate_allowed_by_check_types(candidate, enabled_check_types)]

    @staticmethod
    def _candidate_allowed_by_check_types(candidate: ConsistencyCandidate, enabled_check_types: set[str]) -> bool:
        if candidate.candidate_type == "term_usage":
            return "term_definition_consistency" in enabled_check_types
        if candidate.candidate_type == "fact_pair":
            return "fact_value_consistency" in enabled_check_types
        # LLM extracted candidate types are usually fact/term/clause-reference like checks.
        issue_type_hint = (candidate.issue_type_hint or "").upper()
        if "REFERENCE" in issue_type_hint:
            return "cross_reference_consistency" in enabled_check_types
        if issue_type_hint in {"TERM_USE_MISMATCH"}:
            return "term_definition_consistency" in enabled_check_types
        return "fact_value_consistency" in enabled_check_types or "term_definition_consistency" in enabled_check_types

    @staticmethod
    def _compose_result_metadata(
        *,
        metadata: Dict[str, Any],
        doc_profile: Any,
        effective_check_types: set[str],
        slot_hints: List[str],
    ) -> Dict[str, Any]:
        return {
            **metadata,
            "docProfile": doc_profile.model_dump(by_alias=True, mode="json"),
            "primaryDocType": doc_profile.primary_type.value,
            "docTypes": [doc_type.value for doc_type in doc_profile.doc_types],
            "classificationConfidence": doc_profile.confidence,
            "effectiveCheckTypes": sorted(effective_check_types),
            "slotHints": slot_hints,
        }

    def _merge_issues(self, issues: List[ConsistencyIssue]) -> List[ConsistencyIssue]:
        merged: Dict[str, ConsistencyIssue] = {}
        for issue in issues:
            if not self._should_emit_issue(issue):
                continue
            merge_key = self._issue_merge_key(issue)
            existing = merged.get(merge_key)
            if existing is None:
                merged[merge_key] = issue
                continue
            if issue.confidence > existing.confidence:
                existing.summary = issue.summary
                existing.reasoning = issue.reasoning
                existing.confidence = issue.confidence
                existing.severity = issue.severity
                existing.status = issue.status
                if issue.suggestion:
                    existing.suggestion = issue.suggestion
            existing_keys = {
                (evidence.role, evidence.source_location.segment_id, tuple(evidence.source_location.element_ids))
                for evidence in existing.evidence
            }
            for evidence in issue.evidence:
                evidence_key = (evidence.role, evidence.source_location.segment_id, tuple(evidence.source_location.element_ids))
                if evidence_key not in existing_keys:
                    existing.evidence.append(evidence)
                    existing_keys.add(evidence_key)
        return sorted(
            merged.values(),
            key=lambda item: ({"high": 0, "medium": 1, "low": 2}.get(item.severity, 3), -item.confidence),
        )

    @staticmethod
    def _issue_merge_key(issue: ConsistencyIssue) -> str:
        segment_ids = sorted(
            {
                evidence.source_location.segment_id or ""
                for evidence in issue.evidence
                if evidence.source_location.segment_id
            }
        )
        if len(segment_ids) >= 2:
            return f"pair::{segment_ids[0]}::{segment_ids[1]}"
        if len(segment_ids) == 1:
            return f"single::{segment_ids[0]}::{issue.issue_type}"
        normalized_key = str(issue.attributes.get("referenceLabel") or issue.attributes.get("leftValue") or issue.issue_type)
        return f"{issue.issue_type}::{normalized_key}"

    def _should_emit_issue(self, issue: ConsistencyIssue) -> bool:
        if issue.issue_type == "CROSS_REFERENCE_ERROR" and "referenceLabel" not in issue.attributes:
            return False

        evidence_block_types = {evidence.source_location.block_type or "" for evidence in issue.evidence}
        all_table = bool(evidence_block_types) and evidence_block_types == {"table"}
        summary = normalize_whitespace(issue.summary)
        reasoning = normalize_whitespace(issue.reasoning)
        weak_terms = ("命名", "术语", "可能", "需确认", "潜在", "歧义", "不统一")
        direct_conflict_terms = ("矛盾", "冲突", "必须", "不得", "可不", "互斥", "不一致")

        if issue.status == "possibly_inconsistent" and issue.confidence < 0.8:
            if all_table and any(term in summary or term in reasoning for term in weak_terms):
                if not any(term in summary or term in reasoning for term in direct_conflict_terms):
                    return False

        if (
            issue.issue_type in {"TERM_USE_MISMATCH", "ENTITY_ROLE_MISMATCH"}
            and issue.status == "possibly_inconsistent"
            and all_table
            and issue.confidence <= 0.75
        ):
            return False

        return True

    def _build_candidate_prompt(self, candidate: ConsistencyCandidate) -> str:
        prompt_template = self._load_prompt_template()
        payload = {
            "candidateType": candidate.candidate_type,
            "subject": candidate.subject,
            "normalizedKey": candidate.normalized_key,
            "issueTypeHint": candidate.issue_type_hint,
            "leftEvidence": candidate.left_evidence.model_dump(mode="json", by_alias=True),
            "rightEvidence": candidate.right_evidence.model_dump(mode="json", by_alias=True) if candidate.right_evidence else None,
            "leftSummary": candidate.left_summary,
            "rightSummary": candidate.right_summary,
            "attributes": candidate.attributes,
        }
        extra_guardrail = (
            "\n\n[extra_rules]\n"
            "- 如果一处是一般规则，另一处是特定例外、补充条件或适用范围更窄的特殊情形，"
            "除非文本明确说明两者适用于同一范围且要求互斥，否则不要判定为冲突。\n"
            "- 对于仅因“未明确写出例外”“适用范围可能不同”而产生的疑问，优先返回 `insufficient_information` 或 `consistent`，不要直接报不一致。"
        )
        return f"{prompt_template}{extra_guardrail}\n\n[candidate]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    def _should_emit_issue(self, issue: ConsistencyIssue) -> bool:
        if issue.issue_type == "CROSS_REFERENCE_ERROR" and "referenceLabel" not in issue.attributes:
            return False

        evidence_block_types = {evidence.source_location.block_type or "" for evidence in issue.evidence}
        all_table = bool(evidence_block_types) and evidence_block_types == {"table"}
        summary = normalize_whitespace(issue.summary)
        reasoning = normalize_whitespace(issue.reasoning)
        weak_terms = ("命名", "术语", "可能", "需确认", "潜在", "歧义", "不统一")
        direct_conflict_terms = ("矛盾", "冲突", "必须", "不得", "可不", "互斥", "不一致")
        exception_ambiguity_terms = ("例外", "适用范围", "特定情形", "特殊情形", "未明确排除", "可能不同", "豁免", "一般规则")

        if issue.status == "possibly_inconsistent" and issue.confidence < 0.8:
            if all_table and any(term in summary or term in reasoning for term in weak_terms):
                if not any(term in summary or term in reasoning for term in direct_conflict_terms):
                    return False

        if (
            issue.status == "possibly_inconsistent"
            and all_table
            and issue.confidence <= 0.8
            and issue.issue_type in {"CLAUSE_LOGIC_CONFLICT", "FACT_VALUE_MISMATCH", "TERM_USE_MISMATCH"}
            and any(term in summary or term in reasoning for term in exception_ambiguity_terms)
        ):
            return False

        if (
            issue.issue_type in {"TERM_USE_MISMATCH", "ENTITY_ROLE_MISMATCH"}
            and issue.status == "possibly_inconsistent"
            and all_table
            and issue.confidence <= 0.75
        ):
            return False

        return True

    def _should_emit_issue(self, issue: ConsistencyIssue) -> bool:
        if issue.issue_type == "CROSS_REFERENCE_ERROR" and "referenceLabel" not in issue.attributes:
            return False

        evidence_block_types = {evidence.source_location.block_type or "" for evidence in issue.evidence}
        all_table = bool(evidence_block_types) and evidence_block_types == {"table"}
        combined_text = normalize_whitespace(f"{issue.summary} {issue.reasoning}").lower()
        evidence_count = len(issue.evidence)

        direct_conflict_terms = (
            "矛盾",
            "冲突",
            "互斥",
            "不一致",
            "conflict",
            "contradict",
            "inconsistent",
            "must not",
            "forbidden",
        )
        ambiguity_terms = (
            "术语",
            "命名",
            "角色",
            "可能",
            "待确认",
            "潜在",
            "歧义",
            "不统一",
            "例外",
            "适用范围",
            "scope",
            "exception",
            "carve-out",
        )

        if issue.status == "insufficient_information":
            return False

        if issue.issue_type in {"TERM_USE_MISMATCH", "ENTITY_ROLE_MISMATCH"} and issue.status != "inconsistent":
            return False

        if (
            issue.issue_type == "FACT_VALUE_MISMATCH"
            and issue.status == "possibly_inconsistent"
            and not all_table
            and evidence_count >= 2
        ):
            return True

        if issue.status == "possibly_inconsistent":
            has_direct_conflict = any(term in combined_text for term in direct_conflict_terms)
            has_ambiguity = any(term in combined_text for term in ambiguity_terms)
            if issue.confidence < 0.9 or not has_direct_conflict or has_ambiguity:
                return False

        if all_table and issue.status != "inconsistent" and issue.confidence < 0.92:
            return False

        return True

    @staticmethod
    def _candidate_merge_key(candidate: ConsistencyCandidate) -> str:
        left_segment = candidate.left_evidence.source_location.segment_id or ""
        right_segment = candidate.right_evidence.source_location.segment_id if candidate.right_evidence else ""
        return f"{candidate.issue_type_hint}::{candidate.normalized_key}::{left_segment}::{right_segment}"

    @staticmethod
    def _normalize_candidate_key(outline: LlmCandidateOutline) -> str:
        base = normalize_whitespace(outline.subject).lower()
        return f"{outline.issue_type_hint}.{base or outline.left_segment_id}"

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_prompt_template() -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "consistency.md"
        return prompt_path.read_text(encoding="utf-8")

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_candidate_extractor_prompt() -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "candidate_extractor.md"
        return prompt_path.read_text(encoding="utf-8")
