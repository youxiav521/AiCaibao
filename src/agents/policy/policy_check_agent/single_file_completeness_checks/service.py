from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from src.agents.content_generate_agent.utils import ainvoke_llm_with_prompt, ainvoke_llm_with_structured_output
from loguru import logger

from ..shared.doc_profile import (
    DocType,
    classify_doc_profile_with_llm_fallback,
    map_doc_types_to_policy_doc_types,
)
from ..schemas import ResourceFile, ResourceRepository
from ..services.material_cleaner import MaterialCleaner
from ..services.material_loader import build_material_store
from ..utils import normalize_whitespace, stable_id
from .profiles import (
    DEFAULT_FALLBACK_SECTIONS,
    DEFAULT_PROFILE_ID,
    COMPLETENESS_PROFILES,
    SECTION_GROUPS,
    SECTION_RULES,
    resolve_group_sections,
    resolve_profile_sections,
)
from .schemas import (
    CompletenessCheckItem,
    CompletenessEvidence,
    CompletenessSourceLocation,
    DocumentTypeClassification,
    DocumentSegment,
    SectionExistenceJudgment,
    SectionGroupClassification,
    SingleFileCompletenessCheckRequest,
    SingleFileCompletenessCheckResult,
)


TITLE_PREFIX_RE = re.compile(r"^(第[一二三四五六七八九十百零\d]+[章节条款]\s*|[一二三四五六七八九十]+[、.]\s*|\d+[、.]\s*)")
NOISE_TOKENS = ("目录", "页码", "备注", "附件清单")


SectionRule = Dict[str, Any]
INTERNAL_SEGMENT_TOKEN_RE = re.compile(
    r"(?:cleaned_[A-Za-z0-9:_-]+|seg_[cb]\d{4}_p\d+_e\d+)",
    flags=re.IGNORECASE,
)

DOC_TYPE_COMPLETENESS_RULES: Dict[DocType, List[SectionRule]] = {
    DocType.LEGAL_REP_ID: [
        {
            "section_name": "姓名",
            "check_point": "id_name",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["姓名"],
            "content_patterns": [r"(姓名)[:：]?\s*[A-Za-z\u4e00-\u9fff·]{2,20}"],
        },
        {
            "section_name": "身份证号",
            "check_point": "id_number",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["公民身份号码", "身份证号", "身份证号码"],
            "content_patterns": [r"(公民身份号码|身份证号|身份证号码)[:：]?\s*[0-9Xx]{15,18}"],
        },
        {
            "section_name": "有效期限",
            "check_point": "id_validity",
            "severity": "medium",
            "allow_element_only": True,
            "title_aliases": ["有效期限", "签发日期", "有效期"],
            "content_patterns": [r"(有效期限|签发日期|有效期)[:：]?\s*[0-9]{4}"],
        },
    ],
    DocType.SPOUSE_ID: [
        {
            "section_name": "姓名",
            "check_point": "spouse_id_name",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["姓名"],
            "content_patterns": [r"(姓名)[:：]?\s*[A-Za-z\u4e00-\u9fff·]{2,20}"],
        },
        {
            "section_name": "身份证号",
            "check_point": "spouse_id_number",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["公民身份号码", "身份证号", "身份证号码"],
            "content_patterns": [r"(公民身份号码|身份证号|身份证号码)[:：]?\s*[0-9Xx]{15,18}"],
        },
    ],
    DocType.BUSINESS_LICENSE: [
        {
            "section_name": "统一社会信用代码",
            "check_point": "bl_uscc",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["统一社会信用代码"],
            "content_patterns": [r"(统一社会信用代码)[:：]?\s*[0-9A-Z]{18}"],
        },
        {
            "section_name": "法定代表人",
            "check_point": "bl_legal_representative",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["法定代表人"],
            "content_patterns": [r"(法定代表人)[:：]?\s*[A-Za-z\u4e00-\u9fff·]{2,20}"],
        },
        {
            "section_name": "成立日期",
            "check_point": "bl_established_date",
            "severity": "medium",
            "allow_element_only": True,
            "title_aliases": ["成立日期"],
            "content_patterns": [r"(成立日期)[:：]?\s*[0-9]{4}"],
        },
    ],
    DocType.TAX_REG_CERT: [
        {
            "section_name": "纳税人识别号",
            "check_point": "tax_id",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["纳税人识别号", "税号"],
            "content_patterns": [r"(纳税人识别号|税号)[:：]?\s*[0-9A-Z]{15,20}"],
        },
        {
            "section_name": "主管税务机关",
            "check_point": "tax_authority",
            "severity": "medium",
            "allow_element_only": True,
            "title_aliases": ["主管税务机关", "税务机关"],
            "content_patterns": [r"(主管税务机关|税务机关)[:：]?\s*[^\n，。；]{2,60}"],
        },
    ],
    DocType.CONTRACT_AND_PAYMENT: [
        {
            "section_name": "合同主体",
            "check_point": "contract_party",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["合同当事人", "甲方", "乙方", "借款人", "贷款人"],
            "content_patterns": [r"(甲方|乙方|借款人|贷款人)[:：]?\s*[^\n，。；]{2,80}"],
        },
        {
            "section_name": "金额条款",
            "check_point": "contract_amount",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["贷款金额", "授信额度", "合同金额", "付款金额"],
            "content_patterns": [r"(贷款金额|授信额度|合同金额|付款金额)[:：]?\s*([0-9][0-9,\.]*)"],
        },
    ],
    DocType.OWNERSHIP_STRUCTURE: [
        {
            "section_name": "股东信息",
            "check_point": "ownership_shareholder",
            "severity": "high",
            "allow_element_only": True,
            "title_aliases": ["股东", "出资方", "持股比例"],
            "content_patterns": [r"(股东|出资方|持股比例|股权比例)"],
        },
        {
            "section_name": "实际控制人",
            "check_point": "ownership_controller",
            "severity": "medium",
            "allow_element_only": True,
            "title_aliases": ["实际控制人", "控制人", "受益所有人"],
            "content_patterns": [r"(实际控制人|控制人|受益所有人)"],
        },
    ],
}

DOC_TYPE_PREFER_TYPE_ONLY = {
    DocType.LEGAL_REP_ID,
    DocType.SPOUSE_ID,
    DocType.BUSINESS_LICENSE,
    DocType.TAX_REG_CERT,
}


class SingleFileCompletenessCheckService:
    def __init__(
        self,
        *,
        model_id: str = "model_bailian_policy_check",
        enable_llm: bool = True,
        llm_retry_attempts: int = 2,
        profile_classifier_func: Optional[
            Callable[[str, str], Awaitable[DocumentTypeClassification]]
        ] = None,
        group_classifier_func: Optional[
            Callable[[str, str], Awaitable[SectionGroupClassification]]
        ] = None,
        section_judge_func: Optional[
            Callable[[str, str], Awaitable[SectionExistenceJudgment]]
        ] = None,
    ) -> None:
        self.cleaner = MaterialCleaner()
        self.model_id = model_id
        self.enable_llm = enable_llm
        self.llm_retry_attempts = llm_retry_attempts
        self._profile_classifier_func = profile_classifier_func or self._default_profile_classifier
        self._group_classifier_func = group_classifier_func or self._default_group_classifier
        self._section_judge_func = section_judge_func or self._default_section_judge

    async def analyze_request(self, request: SingleFileCompletenessCheckRequest) -> SingleFileCompletenessCheckResult:
        resource_file = ResourceFile.model_validate(
            {
                "fileId": request.file_id or self._build_default_file_id(request.file_name, request.doc_content),
                "fileName": request.file_name or f"{request.file_id or 'completeness_file'}.json",
                "content": request.doc_content,
            }
        )
        return await self.analyze_resource_file(
            resource_file,
            required_sections=request.required_sections,
            document_type=request.document_type,
            enable_llm=request.enable_llm,
            effective_model_id=request.model_id or self.model_id,
            effective_enable_llm=request.enable_llm,
            effective_llm_retry_attempts=request.llm_retry_attempts,
        )

    async def analyze_resource_file(
        self,
        resource_file: ResourceFile,
        *,
        required_sections: Optional[List[str]] = None,
        document_type: Optional[str] = None,
        enable_llm: Optional[bool] = None,
        effective_model_id: Optional[str] = None,
        effective_enable_llm: Optional[bool] = None,
        effective_llm_retry_attempts: Optional[int] = None,
    ) -> SingleFileCompletenessCheckResult:
        resolved_model_id = effective_model_id or self.model_id
        resolved_enable_llm = self.enable_llm if effective_enable_llm is None else effective_enable_llm
        if enable_llm is not None:
            resolved_enable_llm = enable_llm
        resolved_llm_retry_attempts = self._validate_llm_retry_attempts(
            effective_llm_retry_attempts if effective_llm_retry_attempts is not None else self.llm_retry_attempts
        )
        repository = ResourceRepository(
            id=stable_id("repo", resource_file.fileId, resource_file.fileName),
            title=resource_file.fileName,
            description="single_file_completeness",
            files=[resource_file],
        )
        doc_profile = await classify_doc_profile_with_llm_fallback(
            file_name=resource_file.fileName or "",
            doc_content=resource_file.content or [],
            model_id=resolved_model_id,
            enable_llm_fallback=resolved_enable_llm,
        )
        policy_doc_types = sorted(map_doc_types_to_policy_doc_types(doc_profile.doc_types))
        cleaned_items = self.cleaner.clean_file(resource_file)
        store = build_material_store(repository, case_id=repository.id)
        segments = self._build_segments(resource_file, store, cleaned_items)
        section_names, resolution_meta = await self._resolve_required_sections(
            required_sections,
            document_type,
            segments,
            enable_llm=resolved_enable_llm,
            effective_model_id=resolved_model_id,
            effective_llm_retry_attempts=resolved_llm_retry_attempts,
        )
        effective_section_names, effective_rules = self._build_effective_section_plan(
            section_names,
            doc_profile.primary_type,
            explicit_sections=bool(required_sections),
            explicit_profile=bool(document_type),
            resolution_source=str(resolution_meta.get("source", "")),
        )
        items = await self._check_sections(
            resource_file.fileId,
            effective_section_names,
            segments,
            section_rules=effective_rules,
            enable_llm=resolved_enable_llm,
            effective_model_id=resolved_model_id,
            effective_llm_retry_attempts=resolved_llm_retry_attempts,
        )
        logger.info(
            (
                "single_file_completeness resolved sections file_id={} source={} profile={} "
                "groups={} doc_type={} policy_doc_types={} sections={}"
            ),
            resource_file.fileId,
            resolution_meta.get("source", ""),
            resolution_meta.get("profileId", ""),
            resolution_meta.get("groupIds", []),
            doc_profile.primary_type.value,
            policy_doc_types,
            effective_section_names,
        )
        return self._build_result(resource_file, items)

    async def _check_sections(
        self,
        file_id: str,
        section_names: List[str],
        segments: List[DocumentSegment],
        *,
        section_rules: Mapping[str, Dict[str, Any]],
        enable_llm: bool,
        effective_model_id: str,
        effective_llm_retry_attempts: int,
    ) -> List[CompletenessCheckItem]:
        items: List[CompletenessCheckItem] = []
        for section_name in section_names:
            item = self._check_section(file_id, section_name, segments, section_rules=section_rules)
            if enable_llm and item.status == "fail":
                item = await self._fallback_section_check_with_llm(
                    file_id,
                    section_name,
                    segments,
                    item,
                    section_rules=section_rules,
                    effective_model_id=effective_model_id,
                    effective_llm_retry_attempts=effective_llm_retry_attempts,
                )
            items.append(item)
        return items

    def _build_segments(
        self,
        resource_file: ResourceFile,
        store: Any,
        cleaned_items: Optional[List[Dict[str, Any]]],
    ) -> List[DocumentSegment]:
        segments: List[DocumentSegment] = []
        cleaned_index = 0
        for cleaned in cleaned_items or []:
            text = normalize_whitespace(cleaned.get("text") or "")
            if not text or self._is_noise_segment_text(text):
                continue
            element_ids = [int(v) for v in cleaned.get("elementIds", []) if str(v).isdigit()]
            if not element_ids and cleaned.get("element_id") is not None and str(cleaned.get("element_id")).isdigit():
                element_ids = [int(cleaned["element_id"])]
            page_no = cleaned.get("pageNo") or cleaned.get("page")
            cleaned_index += 1
            segments.append(
                DocumentSegment(
                    segment_id=self._build_public_segment_id(
                        source_prefix="c",
                        order=cleaned_index,
                        page_no=page_no,
                        element_ids=element_ids,
                    ),
                    file_id=resource_file.fileId,
                    text=text,
                    page_no=page_no,
                    element_ids=element_ids,
                    block_type=str(cleaned.get("type") or cleaned.get("blockType") or "paragraph"),
                )
            )
        block_index = 0
        for block in getattr(store, "evidence_blocks", []):
            text = normalize_whitespace(getattr(block, "text", ""))
            if not text or self._is_noise_segment_text(text):
                continue
            page_no = getattr(block, "page", None)
            element_ids = list(getattr(block, "source_element_ids", []) or [])
            block_index += 1
            segments.append(
                DocumentSegment(
                    segment_id=self._build_public_segment_id(
                        source_prefix="b",
                        order=block_index,
                        page_no=page_no,
                        element_ids=element_ids,
                    ),
                    file_id=resource_file.fileId,
                    text=text,
                    page_no=page_no,
                    element_ids=element_ids,
                    block_type=getattr(block, "block_type", "paragraph"),
                )
            )
        return self._dedup_segments(segments)

    @staticmethod
    def _build_public_segment_id(
        *,
        source_prefix: str,
        order: int,
        page_no: Optional[int],
        element_ids: List[int],
    ) -> str:
        page_part = int(page_no) if isinstance(page_no, int) else 0
        first_element = int(element_ids[0]) if element_ids else 0
        return f"seg_{source_prefix}{order:04d}_p{page_part}_e{first_element}"

    def _check_section(
        self,
        file_id: str,
        section_name: str,
        segments: List[DocumentSegment],
        *,
        section_rules: Mapping[str, Dict[str, Any]],
    ) -> CompletenessCheckItem:
        rule = section_rules.get(section_name) or {
            "check_point": stable_id("section", section_name),
            "severity": "medium",
            "title_aliases": [section_name],
            "content_patterns": [],
            "rule_scope": "general",
        }
        title_match = self._find_title_match(segments, rule["title_aliases"])
        if title_match is not None:
            return CompletenessCheckItem(
                item_id=stable_id("item", file_id, section_name),
                check_point=rule["check_point"],
                section_name=section_name,
                status="pass",
                severity="low",
                summary=f"已识别到独立章节“{section_name}”。",
                reasoning=f"文档中存在标题或标题样式文本，能够明确支持“{section_name}”这一检查项。",
                evidence=[self._segment_evidence("title", title_match.text, title_match)],
                attributes={
                    "matchedBy": "title",
                    "satisfactionMode": "section",
                    "isStandaloneSection": True,
                    "ruleScope": rule.get("rule_scope", "general"),
                },
            )

        content_match = self._find_content_match(segments, rule["content_patterns"])
        if content_match is not None:
            return CompletenessCheckItem(
                item_id=stable_id("item", file_id, section_name),
                check_point=rule["check_point"],
                section_name=section_name,
                status="pass",
                severity="low",
                summary=f"未识别到独立章节，但正文已覆盖“{section_name}”要素。",
                reasoning=f"文档未单独列出“{section_name}”章节，但正文中存在与该检查项语义一致的内容。",
                evidence=[self._segment_evidence("content", content_match.text, content_match)],
                attributes={
                    "matchedBy": "content",
                    "satisfactionMode": "element",
                    "isStandaloneSection": False,
                    "allowElementOnly": bool(rule.get("allow_element_only", True)),
                    "ruleScope": rule.get("rule_scope", "general"),
                },
            )

        return CompletenessCheckItem(
            item_id=stable_id("item", file_id, section_name),
            check_point=rule["check_point"],
            section_name=section_name,
            status="fail",
            severity=rule["severity"],
            summary=f"未识别到“{section_name}”对应的章节或要素。",
            reasoning=f"当前未在文档标题或正文中识别到能够支持“{section_name}”的章节或明确内容。",
            evidence=[],
            attributes={
                "matchedBy": "missing",
                "satisfactionMode": "missing",
                "isStandaloneSection": False,
                "allowElementOnly": bool(rule.get("allow_element_only", True)),
                "ruleScope": rule.get("rule_scope", "general"),
            },
        )

    async def _fallback_section_check_with_llm(
        self,
        file_id: str,
        section_name: str,
        segments: List[DocumentSegment],
        base_item: CompletenessCheckItem,
        *,
        section_rules: Mapping[str, Dict[str, Any]],
        effective_model_id: str,
        effective_llm_retry_attempts: int,
    ) -> CompletenessCheckItem:
        rule = section_rules.get(section_name) or {
            "title_aliases": [section_name],
            "content_patterns": [],
        }
        context_segments = self._select_section_context(segments, section_name, rule["title_aliases"])
        if not context_segments:
            return base_item

        prompt = self._load_prompt("section_existence_judge.md").format(
            section_name=section_name,
            aliases=json.dumps(rule["title_aliases"], ensure_ascii=False),
            document_excerpt=self._format_segment_excerpt(context_segments),
        )
        judgment = await self._call_with_retry(
            self._section_judge_func,
            prompt,
            "section_judge",
            effective_model_id=effective_model_id,
            effective_llm_retry_attempts=effective_llm_retry_attempts,
        )
        if not isinstance(judgment, SectionExistenceJudgment):
            return base_item
        if judgment.status not in ("pass", "uncertain") and not judgment.exists:
            return base_item

        evidence = self._evidence_from_segment_ids(
            judgment.evidence_segment_ids,
            context_segments,
            role="llm_support",
        )
        if judgment.status == "pass" and judgment.exists:
            return CompletenessCheckItem(
                item_id=base_item.item_id,
                check_point=base_item.check_point,
                section_name=base_item.section_name,
                status="pass",
                severity="low",
                summary=f"未识别到独立章节，但已通过语义识别确认“{section_name}”要素存在。",
                reasoning=self._sanitize_llm_text(judgment.reasoning or base_item.reasoning),
                evidence=evidence,
                attributes={
                    **base_item.attributes,
                    "matchedBy": "llm_semantic",
                    "satisfactionMode": "element",
                    "isStandaloneSection": False,
                    "matchedSectionName": self._sanitize_llm_text(judgment.matched_section_name),
                    "confidence": judgment.confidence,
                },
            )
        if judgment.status == "uncertain":
            return CompletenessCheckItem(
                item_id=base_item.item_id,
                check_point=base_item.check_point,
                section_name=base_item.section_name,
                status="uncertain",
                severity=base_item.severity,
                summary=f"“{section_name}”是否已在正文中覆盖需要人工复核。",
                reasoning=self._sanitize_llm_text(judgment.reasoning or base_item.reasoning),
                evidence=evidence,
                attributes={
                    **base_item.attributes,
                    "matchedBy": "llm_uncertain",
                    "satisfactionMode": "missing",
                    "isStandaloneSection": False,
                    "matchedSectionName": self._sanitize_llm_text(judgment.matched_section_name),
                    "confidence": judgment.confidence,
                },
            )
        return base_item

    def _find_title_match(self, segments: Iterable[DocumentSegment], aliases: List[str]) -> Optional[DocumentSegment]:
        for segment in segments:
            text = segment.text.strip()
            normalized = self._normalize_title_text(text)
            if len(normalized) > 24:
                continue
            for alias in aliases:
                if alias == normalized or normalized.startswith(alias):
                    return segment
        return None

    def _find_content_match(self, segments: Iterable[DocumentSegment], patterns: List[str]) -> Optional[DocumentSegment]:
        compiled = [re.compile(pattern) for pattern in patterns]
        for segment in segments:
            text = segment.text.strip()
            for pattern in compiled:
                if pattern.search(text):
                    return segment
        return None

    def _build_effective_section_plan(
        self,
        base_sections: List[str],
        primary_doc_type: DocType,
        *,
        explicit_sections: bool,
        explicit_profile: bool,
        resolution_source: str,
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        effective_rules: Dict[str, Dict[str, Any]] = {
            section_name: {**rule, "rule_scope": "general"}
            for section_name, rule in SECTION_RULES.items()
        }
        sections = list(base_sections)
        if explicit_sections:
            return sections, effective_rules

        type_rules = DOC_TYPE_COMPLETENESS_RULES.get(primary_doc_type, [])
        prefer_type_only = (
            primary_doc_type in DOC_TYPE_PREFER_TYPE_ONLY
            and bool(type_rules)
            and not explicit_profile
            and resolution_source in {"default_profile", "default_fallback", "llm_profile", "llm_groups"}
        )
        if prefer_type_only:
            sections = []

        for rule in type_rules:
            section_name = str(rule["section_name"])
            if section_name not in sections:
                sections.append(section_name)
            effective_rules[section_name] = {
                "check_point": rule["check_point"],
                "severity": rule["severity"],
                "allow_element_only": bool(rule.get("allow_element_only", True)),
                "title_aliases": list(rule["title_aliases"]),
                "content_patterns": list(rule["content_patterns"]),
                "rule_scope": "type",
                "rule_doc_type": primary_doc_type.value,
            }
        return sections, effective_rules

    def _build_result(
        self,
        resource_file: ResourceFile,
        items: List[CompletenessCheckItem],
    ) -> SingleFileCompletenessCheckResult:
        failed = sum(1 for item in items if item.status == "fail")
        passed = sum(1 for item in items if item.status == "pass")
        uncertain = sum(1 for item in items if item.status == "uncertain")
        if failed:
            overall_status = "fail"
            summary = f"发现 {failed} 个必备章节缺失，已命中 {passed} 个章节。"
        elif uncertain:
            overall_status = "uncertain"
            summary = f"存在 {uncertain} 个章节需人工复核，其余章节已识别。"
        else:
            overall_status = "pass"
            summary = "文档已识别到当前要求的全部必备章节与要素。"
        return SingleFileCompletenessCheckResult(
            document_id=stable_id("completeness_doc", resource_file.fileId, resource_file.fileName),
            file_id=resource_file.fileId,
            file_name=resource_file.fileName,
            overall_status=overall_status,
            summary=summary,
            item_count=len(items),
            items=items,
            statistics={"pass": passed, "fail": failed, "uncertain": uncertain},
        )

    def _segment_evidence(self, role: str, text: str, segment: DocumentSegment) -> CompletenessEvidence:
        return CompletenessEvidence(
            role=role,
            source_text=text,
            source_location=CompletenessSourceLocation(
                file_id=segment.file_id,
                element_ids=segment.element_ids,
                page_no=segment.page_no,
                segment_id=segment.segment_id,
                block_type=segment.block_type,
            ),
        )

    def _evidence_from_segment_ids(
        self,
        segment_ids: List[str],
        segments: List[DocumentSegment],
        *,
        role: str,
    ) -> List[CompletenessEvidence]:
        if not segment_ids:
            return []
        index = {segment.segment_id: segment for segment in segments}
        evidence: List[CompletenessEvidence] = []
        for segment_id in segment_ids:
            segment = index.get(segment_id)
            if segment is None:
                continue
            evidence.append(self._segment_evidence(role, segment.text, segment))
        return evidence

    def _dedup_segments(self, segments: Iterable[DocumentSegment]) -> List[DocumentSegment]:
        deduped_by_key: Dict[str, DocumentSegment] = {}
        order: List[str] = []
        for segment in segments:
            key = stable_id("seg_dedup", segment.file_id, segment.text, segment.page_no or "", segment.block_type)
            existing = deduped_by_key.get(key)
            if existing is None:
                deduped_by_key[key] = segment
                order.append(key)
                continue
            if self._is_better_segment(segment, existing):
                deduped_by_key[key] = segment
        return [deduped_by_key[key] for key in order]

    def _is_better_segment(self, candidate: DocumentSegment, existing: DocumentSegment) -> bool:
        candidate_span = len(candidate.element_ids) if candidate.element_ids else 9999
        existing_span = len(existing.element_ids) if existing.element_ids else 9999
        if candidate_span != existing_span:
            return candidate_span < existing_span
        return len(candidate.text) < len(existing.text)

    def _is_noise_segment_text(self, text: str) -> bool:
        if len(text) <= 2:
            return True
        return any(token == text for token in NOISE_TOKENS)

    def _normalize_title_text(self, text: str) -> str:
        normalized = TITLE_PREFIX_RE.sub("", text).strip()
        normalized = normalized.lstrip("：:、.。").strip()
        return normalized

    async def _resolve_required_sections(
        self,
        required_sections: Optional[List[str]],
        document_type: Optional[str],
        segments: List[DocumentSegment],
        *,
        enable_llm: bool,
        effective_model_id: str,
        effective_llm_retry_attempts: int,
    ) -> Tuple[List[str], Dict[str, Any]]:
        if required_sections:
            return list(required_sections), {"source": "explicit_sections"}
        if document_type:
            return resolve_profile_sections(document_type), {
                "source": "explicit_profile",
                "profileId": document_type,
            }
        if not enable_llm:
            return resolve_profile_sections(DEFAULT_PROFILE_ID), {
                "source": "default_profile",
                "profileId": DEFAULT_PROFILE_ID,
            }

        doc_brief = self._build_document_brief(segments)
        profile_result = await self._classify_document_type_with_llm(
            doc_brief,
            effective_model_id=effective_model_id,
            effective_llm_retry_attempts=effective_llm_retry_attempts,
        )
        if (
            profile_result.profile_id in COMPLETENESS_PROFILES
            and profile_result.profile_id != "unknown"
            and profile_result.confidence >= 0.7
        ):
            return resolve_profile_sections(profile_result.profile_id), {
                "source": "llm_profile",
                "profileId": profile_result.profile_id,
                "confidence": profile_result.confidence,
                "reason": profile_result.reason,
            }

        group_result = await self._classify_section_groups_with_llm(
            doc_brief,
            effective_model_id=effective_model_id,
            effective_llm_retry_attempts=effective_llm_retry_attempts,
        )
        group_sections = resolve_group_sections(group_result.group_ids)
        if group_result.group_ids and group_result.confidence >= 0.55:
            return group_sections, {
                "source": "llm_groups",
                "groupIds": group_result.group_ids,
                "confidence": group_result.confidence,
                "reason": group_result.reason,
            }
        return list(DEFAULT_FALLBACK_SECTIONS), {
            "source": "default_fallback",
            "profileId": DEFAULT_PROFILE_ID,
        }

    def _build_document_brief(self, segments: List[DocumentSegment], limit: int = 12) -> str:
        lines: List[str] = []
        for segment in segments[:limit]:
            preview = segment.text.strip()
            if len(preview) > 120:
                preview = f"{preview[:120]}..."
            lines.append(f"[{segment.segment_id}] {preview}")
        return "\n".join(lines)

    def _format_segment_excerpt(self, segments: List[DocumentSegment], limit: int = 8) -> str:
        lines: List[str] = []
        for segment in segments[:limit]:
            preview = segment.text.strip()
            if len(preview) > 180:
                preview = f"{preview[:180]}..."
            lines.append(f"[{segment.segment_id}] {preview}")
        return "\n".join(lines)

    @staticmethod
    def _sanitize_llm_text(text: str) -> str:
        cleaned = INTERNAL_SEGMENT_TOKEN_RE.sub("", str(text or ""))
        return " ".join(cleaned.split())

    def _select_section_context(
        self,
        segments: List[DocumentSegment],
        section_name: str,
        aliases: List[str],
        *,
        limit: int = 8,
    ) -> List[DocumentSegment]:
        selected: List[DocumentSegment] = []
        query_terms = [section_name, *aliases]
        lowered_terms = [term.lower() for term in query_terms if term]
        for segment in segments:
            text = segment.text.lower()
            if any(term.lower() in text for term in lowered_terms):
                selected.append(segment)
                continue
            if len(selected) >= limit:
                break
        if len(selected) < min(4, len(segments)):
            for segment in segments:
                if segment in selected:
                    continue
                selected.append(segment)
                if len(selected) >= limit:
                    break
        return selected[:limit]

    async def _classify_document_type_with_llm(
        self,
        doc_brief: str,
        *,
        effective_model_id: str,
        effective_llm_retry_attempts: int,
    ) -> DocumentTypeClassification:
        prompt = self._load_prompt("profile_classifier.md").format(
            available_profiles=json.dumps(
                {
                    key: {"label": value["label"], "sections": value["sections"]}
                    for key, value in COMPLETENESS_PROFILES.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            document_excerpt=doc_brief,
        )
        return await self._call_with_retry(
            self._profile_classifier_func,
            prompt,
            "profile_classifier",
            effective_model_id=effective_model_id,
            effective_llm_retry_attempts=effective_llm_retry_attempts,
        )

    async def _classify_section_groups_with_llm(
        self,
        doc_brief: str,
        *,
        effective_model_id: str,
        effective_llm_retry_attempts: int,
    ) -> SectionGroupClassification:
        prompt = self._load_prompt("group_classifier.md").format(
            available_groups=json.dumps(
                {
                    key: {"label": value["label"], "sections": value["sections"]}
                    for key, value in SECTION_GROUPS.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            document_excerpt=doc_brief,
        )
        return await self._call_with_retry(
            self._group_classifier_func,
            prompt,
            "group_classifier",
            effective_model_id=effective_model_id,
            effective_llm_retry_attempts=effective_llm_retry_attempts,
        )

    async def _call_with_retry(
        self,
        func: Callable[[str, str], Awaitable[Any]],
        prompt: str,
        stage: str,
        *,
        effective_model_id: str,
        effective_llm_retry_attempts: int,
    ) -> Any:
        last_error: Optional[Exception] = None
        for _ in range(effective_llm_retry_attempts):
            try:
                return await func(prompt, effective_model_id)
            except Exception as exc:
                last_error = exc
        logger.warning("single_file_completeness {} failed error={}", stage, last_error)
        if stage == "profile_classifier":
            return DocumentTypeClassification(profile_id="unknown", confidence=0.0, reason="classification_failed")
        if stage == "section_judge":
            return SectionExistenceJudgment(
                exists=False,
                status="uncertain",
                matched_section_name="",
                reasoning="section_judge_failed",
                confidence=0.0,
                evidence_segment_ids=[],
            )
        return SectionGroupClassification(group_ids=[], confidence=0.0, reason="classification_failed")

    async def _default_profile_classifier(self, prompt: str, model_id: str) -> DocumentTypeClassification:
        return await ainvoke_llm_with_structured_output(
            prompt,
            model_id=model_id,
            response_type=DocumentTypeClassification,
        )

    async def _default_group_classifier(self, prompt: str, model_id: str) -> SectionGroupClassification:
        try:
            return await ainvoke_llm_with_structured_output(
                prompt,
                model_id=model_id,
                response_type=SectionGroupClassification,
            )
        except Exception as exc:
            logger.warning(
                "single_file_completeness group_classifier structured parse failed, fallback to raw parsing error={}",
                exc,
            )
            raw = await ainvoke_llm_with_prompt(prompt, model_id=model_id)
            return self._parse_group_classifier_raw(raw)

    async def _default_section_judge(self, prompt: str, model_id: str) -> SectionExistenceJudgment:
        return await ainvoke_llm_with_structured_output(
            prompt,
            model_id=model_id,
            response_type=SectionExistenceJudgment,
        )

    def _parse_group_classifier_raw(self, raw: str) -> SectionGroupClassification:
        text = (raw or "").strip()
        if not text:
            return SectionGroupClassification(group_ids=[], confidence=0.0, reason="empty_raw_output")

        payload: Any = text
        try:
            payload = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    payload = json.loads(text[start : end + 1])
                except Exception:
                    payload = text
            else:
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end > start:
                    try:
                        payload = json.loads(text[start : end + 1])
                    except Exception:
                        payload = text

        if isinstance(payload, list):
            group_ids = [str(v) for v in payload if isinstance(v, str)]
            return SectionGroupClassification(
                group_ids=group_ids[:2],
                confidence=0.5 if group_ids else 0.0,
                reason="parsed_from_list_output",
            )

        if isinstance(payload, dict):
            candidate_ids = payload.get("groupIds", payload.get("group_ids", []))
            if isinstance(candidate_ids, list):
                group_ids = [str(v) for v in candidate_ids if isinstance(v, str)]
            else:
                group_ids = []

            confidence = payload.get("confidence", 0.0)
            try:
                confidence_value = float(confidence)
            except Exception:
                confidence_value = 0.0
            confidence_value = max(0.0, min(1.0, confidence_value))

            reason = payload.get("reason") or "parsed_from_dict_output"
            return SectionGroupClassification(
                group_ids=group_ids[:2],
                confidence=confidence_value,
                reason=str(reason),
            )

        return SectionGroupClassification(group_ids=[], confidence=0.0, reason="unparseable_raw_output")

    def _load_prompt(self, file_name: str) -> str:
        return (Path(__file__).resolve().parent / "prompts" / file_name).read_text(encoding="utf-8")

    @staticmethod
    def _validate_llm_retry_attempts(value: Any) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("llm_retry_attempts must be an integer >= 1")
        return value

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
