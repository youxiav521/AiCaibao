from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
from .schemas import (
    DocumentSegment,
    RiskSignalCandidate,
    RiskSignalEvidence,
    RiskSignalIssue,
    RiskSignalJudgment,
    RiskSignalSourceLocation,
    SingleFileRiskSignalCheckRequest,
    SingleFileRiskSignalCheckResult,
)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")
NOISE_TOKENS = ("目录", "页码", "备注", "附件清单")
VAGUE_CONTEXT_TOKENS = (
    "应",
    "应当",
    "须",
    "需要",
    "应在",
    "通知",
    "完成",
    "履行",
    "工作日",
    "期限",
    "责任",
    "义务",
    "处理",
    "审批",
)
SEVERITY_PRIORITY = {"high": 3, "medium": 2, "low": 1}
RISK_RULES = [
    ("VAGUE_EXPRESSION", "medium", ("原则上", "视情况", "酌情", "适当", "必要时", "尽快", "及时", "可根据情况")),
    ("EXTREME_EXPRESSION", "high", ("绝不负责", "任何情况均不承担", "概不负责", "全部免责", "无条件免责")),
    (
        "LIABILITY_DISCLAIMER",
        "high",
        ("不承担责任", "不承担任何责任", "免责", "不作保证", "不作承诺", "风险自担", "无需负责", "责任自负"),
    ),
    ("UNILATERAL_INTERPRETATION", "high", ("最终解释权归", "我方有权单方解释", "单方解释", "以我方解释为准", "无需另行说明")),
]

DOC_TYPE_RISK_RULES: Dict[DocType, List[tuple[str, str, tuple[str, ...]]]] = {
    DocType.LEGAL_REP_ID: [
        ("PERSONAL_INFO_EXPOSURE", "high", ("身份证号全号", "完整身份证号", "明文展示", "公开展示", "对外披露", "截图外传")),
        ("ID_FIELD_ANOMALY", "medium", ("号码不一致", "证件过期", "签发机关不详", "信息缺失", "有效期异常")),
    ],
    DocType.SPOUSE_ID: [
        ("PERSONAL_INFO_EXPOSURE", "high", ("身份证号全号", "完整身份证号", "明文展示", "公开展示", "对外披露", "截图外传")),
        ("ID_FIELD_ANOMALY", "medium", ("号码不一致", "证件过期", "签发机关不详", "信息缺失", "有效期异常")),
    ],
    DocType.BUSINESS_LICENSE: [
        ("LICENSE_FIELD_ANOMALY", "medium", ("吊销", "注销", "经营异常", "异常名录", "代码不一致", "证照过期")),
    ],
    DocType.ARTICLES_OF_ASSOCIATION: [
        ("RIGHTS_DUTIES_CONFLICT", "high", ("股东会", "董事会", "监事会", "表决权", "利润分配")),
    ],
    DocType.OWNERSHIP_STRUCTURE: [
        ("CONTROL_RELATION_RISK", "high", ("实际控制人", "持股比例", "股权结构", "受益所有人")),
    ],
    DocType.CONTRACT_AND_PAYMENT: [
        ("PAYMENT_RESPONSIBILITY_AMBIGUITY", "high", ("付款", "回单", "支付", "违约责任", "结算")),
        ("CLAUSE_EDGE_RISK", "medium", ("另行约定", "最终解释", "以甲方为准", "以我方为准")),
    ],
    DocType.TAX_REG_CERT: [
        ("TAX_ID_ANOMALY", "medium", ("税号不一致", "税务异常", "非正常户", "欠税", "证照过期")),
    ],
    DocType.AUDIO: [
        ("TRACEABILITY_RISK", "medium", ("无法识别说话人", "无时间戳", "转写缺失", "疑似剪辑", "音质不清")),
    ],
    DocType.VIDEO: [
        ("TRACEABILITY_RISK", "medium", ("无时间戳", "关键画面缺失", "疑似剪辑", "画面模糊", "来源不明")),
    ],
    DocType.MEETING_MINUTES: [
        ("MINUTES_INTEGRITY_RISK", "medium", ("未记录结论", "无参会名单", "口头决定", "未经审议", "决议缺失")),
    ],
}


class SingleFileRiskSignalCheckService:
    def __init__(
        self,
        *,
        model_id: str = "model_bailian_policy_check",
        max_candidates: int = 24,
        max_judge_concurrency: int = 12,
        enable_llm: bool = True,
        llm_retry_attempts: int = 3,
    ) -> None:
        self.model_id = model_id
        self.max_candidates = max_candidates
        self.max_judge_concurrency = max_judge_concurrency
        self.enable_llm = enable_llm
        self.llm_retry_attempts = llm_retry_attempts
        self.cleaner = MaterialCleaner()
        self._risk_signal_judge_template = (
            Path(__file__).resolve().parent / "prompts" / "risk_signal_judge.md"
        ).read_text(encoding="utf-8")

    async def analyze_request(self, request: SingleFileRiskSignalCheckRequest) -> SingleFileRiskSignalCheckResult:
        resource_file = ResourceFile.model_validate(
            {
                "fileId": request.file_id or self._build_default_file_id(request.file_name, request.doc_content),
                "fileName": request.file_name or f"{request.file_id or 'risk_file'}.json",
                "content": request.doc_content,
            }
        )
        return await self.analyze_resource_file(
            resource_file,
            max_candidates=request.max_candidates,
            enable_llm=request.enable_llm,
            model_id=request.model_id or self.model_id,
            max_judge_concurrency=request.max_judge_concurrency,
            llm_retry_attempts=request.llm_retry_attempts,
        )

    async def analyze_resource_file(
        self,
        resource_file: ResourceFile,
        *,
        max_candidates: Optional[int] = None,
        enable_llm: Optional[bool] = None,
        model_id: Optional[str] = None,
        max_judge_concurrency: Optional[int] = None,
        llm_retry_attempts: Optional[int] = None,
    ) -> SingleFileRiskSignalCheckResult:
        repository = ResourceRepository(
            id=stable_id("repo", resource_file.fileId, resource_file.fileName),
            title=resource_file.fileName,
            description="single_file_risk_signal",
            files=[resource_file],
        )
        doc_profile = await classify_doc_profile_with_llm_fallback(
            file_name=resource_file.fileName or "",
            doc_content=resource_file.content or [],
            model_id=model_id or self.model_id,
            enable_llm_fallback=self.enable_llm if enable_llm is None else enable_llm,
        )
        policy_doc_types = sorted(map_doc_types_to_policy_doc_types(doc_profile.doc_types))
        cleaned_items = self.cleaner.clean_file(resource_file)
        store = build_material_store(repository, case_id=repository.id)
        segments = self._build_segments(resource_file, store, cleaned_items)
        candidates = self._build_candidates(
            resource_file.fileId,
            segments,
            max_candidates or self.max_candidates,
            primary_doc_type=doc_profile.primary_type,
        )
        logger.info(
            "single_file_risk_signal prepared file_id={} segments={} candidates={} policy_doc_types={}",
            resource_file.fileId,
            len(segments),
            len(candidates),
            policy_doc_types,
        )
        effective_llm = self.enable_llm if enable_llm is None else enable_llm
        effective_model_id = model_id or self.model_id
        effective_max_judge_concurrency = self._validate_max_judge_concurrency(
            max_judge_concurrency if max_judge_concurrency is not None else self.max_judge_concurrency
        )
        effective_llm_retry_attempts = self._validate_llm_retry_attempts(
            llm_retry_attempts if llm_retry_attempts is not None else self.llm_retry_attempts
        )
        if effective_llm and candidates:
            issues = await self._judge_candidates(
                candidates,
                model_id=effective_model_id,
                max_judge_concurrency=effective_max_judge_concurrency,
                llm_retry_attempts=effective_llm_retry_attempts,
            )
        else:
            issues = self._fallback_judge_candidates(candidates)
        result = self._build_result(
            resource_file,
            self._merge_issues(issues),
            metadata={
                "docProfile": doc_profile.model_dump(by_alias=True, mode="json"),
                "primaryDocType": doc_profile.primary_type.value,
                "docTypes": [doc_type.value for doc_type in doc_profile.doc_types],
                "classificationConfidence": doc_profile.confidence,
                "effectiveRiskRuleScopes": ["general", "type"],
            },
        )
        logger.info(
            "single_file_risk_signal complete file_id={} overall_status={} issue_count={}",
            resource_file.fileId,
            result.overall_status,
            result.issue_count,
        )
        return result

    def _build_segments(
        self,
        resource_file: ResourceFile,
        store: Any,
        cleaned_items: Optional[List[Dict[str, Any]]],
    ) -> List[DocumentSegment]:
        segments: List[DocumentSegment] = []
        for cleaned in cleaned_items or []:
            text = normalize_whitespace(cleaned.get("text") or "")
            if not text or self._is_noise_segment_text(text):
                continue
            element_ids = [int(v) for v in cleaned.get("elementIds", []) if str(v).isdigit()]
            if not element_ids and cleaned.get("element_id") is not None and str(cleaned.get("element_id")).isdigit():
                element_ids = [int(cleaned["element_id"])]
            segments.append(
                DocumentSegment(
                    segment_id=str(cleaned.get("id") or stable_id("cleaned", resource_file.fileId, text[:80])),
                    file_id=resource_file.fileId,
                    text=text,
                    page_no=cleaned.get("pageNo") or cleaned.get("page"),
                    element_ids=element_ids,
                    block_type=str(cleaned.get("type") or cleaned.get("blockType") or "paragraph"),
                )
            )
        for block in getattr(store, "evidence_blocks", []):
            text = normalize_whitespace(getattr(block, "text", ""))
            if not text or self._is_noise_segment_text(text):
                continue
            segments.append(
                DocumentSegment(
                    segment_id=getattr(block, "evidence_id", stable_id("seg", resource_file.fileId, text[:40])),
                    file_id=resource_file.fileId,
                    text=text,
                    page_no=getattr(block, "page", None),
                    element_ids=list(getattr(block, "source_element_ids", []) or []),
                    block_type=getattr(block, "block_type", "paragraph"),
                )
            )
        deduped_by_key: Dict[str, DocumentSegment] = {}
        dedup_order: List[str] = []
        for segment in segments:
            key = stable_id("seg_dedup", segment.file_id, segment.text, segment.page_no or "", segment.block_type)
            existing = deduped_by_key.get(key)
            if existing is None:
                deduped_by_key[key] = segment
                dedup_order.append(key)
                continue
            if self._is_better_segment(segment, existing):
                deduped_by_key[key] = segment
        return [deduped_by_key[key] for key in dedup_order]

    def _build_candidates(
        self,
        file_id: str,
        segments: Iterable[DocumentSegment],
        max_candidates: int,
        *,
        primary_doc_type: DocType,
    ) -> List[RiskSignalCandidate]:
        effective_rules = self._resolve_effective_risk_rules(primary_doc_type)
        candidates: List[RiskSignalCandidate] = []
        for segment in segments:
            for sentence in self._split_sentences(segment.text):
                sentence = normalize_whitespace(sentence)
                if len(sentence) < 6:
                    continue
                best_match: Optional[tuple[str, str, List[str], str]] = None
                for issue_type, severity, keywords, rule_scope in effective_rules:
                    matched = [keyword for keyword in keywords if keyword in sentence]
                    if not matched:
                        continue
                    if issue_type == "VAGUE_EXPRESSION" and not self._is_actionable_vague_sentence(sentence):
                        continue
                    if best_match is None or SEVERITY_PRIORITY[severity] > SEVERITY_PRIORITY[best_match[1]]:
                        best_match = (issue_type, severity, matched, rule_scope)
                if best_match is not None:
                    issue_type, severity, matched, rule_scope = best_match
                    candidates.append(
                        RiskSignalCandidate(
                            candidate_id=stable_id("risk", segment.segment_id, issue_type, sentence[:80]),
                            issue_type_hint=issue_type,
                            severity_hint=severity,
                            text=sentence,
                            segment_id=segment.segment_id,
                            evidence=self._segment_evidence("mention", sentence, segment),
                            attributes={
                                "matchedKeywords": matched,
                                "ruleScope": rule_scope,
                                "ruleDocType": primary_doc_type.value,
                            },
                        )
                    )
                if len(candidates) >= max_candidates:
                    return self._dedup_candidates(candidates)
        return self._dedup_candidates(candidates)

    async def _judge_candidates(
        self,
        candidates: List[RiskSignalCandidate],
        *,
        model_id: str,
        max_judge_concurrency: int,
        llm_retry_attempts: int,
    ) -> List[RiskSignalIssue]:
        prompts = [self._build_judge_prompt(candidate) for candidate in candidates]
        judgments = await self._invoke_with_retry(
            prompts,
            model_id=model_id,
            max_judge_concurrency=max_judge_concurrency,
            llm_retry_attempts=llm_retry_attempts,
        )
        issues: List[RiskSignalIssue] = []
        for candidate, judgment in zip(candidates, judgments):
            if not judgment.is_risk:
                continue
            issues.append(
                RiskSignalIssue(
                    issue_id=stable_id("issue", candidate.candidate_id, judgment.issue_type, judgment.summary),
                    issue_type=judgment.issue_type or candidate.issue_type_hint,
                    severity=judgment.severity,
                    status=judgment.status,
                    summary=judgment.summary or candidate.text[:120],
                    reasoning=judgment.reasoning or "模型判断该表述存在风险信号。",
                    confidence=judgment.confidence,
                    evidence=[candidate.evidence],
                    suggestion=judgment.suggestion or self._default_suggestion(candidate.issue_type_hint),
                    attributes=candidate.attributes,
                )
            )
        return issues

    async def _invoke_with_retry(
        self,
        prompts: List[str],
        *,
        model_id: str,
        max_judge_concurrency: int,
        llm_retry_attempts: int,
    ) -> List[RiskSignalJudgment]:
        semaphore = asyncio.Semaphore(max_judge_concurrency)

        async def _judge_one(prompt: str) -> RiskSignalJudgment:
            last_error: Optional[Exception] = None
            for attempt in range(1, llm_retry_attempts + 1):
                try:
                    async with semaphore:
                        return await self._invoke_single_judgment(prompt, model_id=model_id)
                except Exception as exc:
                    last_error = exc
                    if attempt < llm_retry_attempts:
                        await asyncio.sleep(min(1.5 * attempt, 4.0))
            logger.warning("single_file_risk_signal single_judge_failed error={}", last_error)
            return self._fallback_judgment()

        results = await asyncio.gather(*[_judge_one(prompt) for prompt in prompts], return_exceptions=True)
        judgments: List[RiskSignalJudgment] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("single_file_risk_signal gather_exception error={}", result)
                judgments.append(self._fallback_judgment())
            else:
                judgments.append(result)
        return judgments

    async def _invoke_single_judgment(self, prompt: str, *, model_id: str) -> RiskSignalJudgment:
        try:
            return await ainvoke_llm_with_structured_output(
                prompt,
                model_id=model_id,
                response_type=RiskSignalJudgment,
            )
        except Exception as exc:
            logger.warning(
                "single_file_risk_signal structured_output_failed fallback_to_raw error={}",
                exc,
            )
            raw = await ainvoke_llm_with_prompt(prompt, model_id=model_id)
            return self._parse_judgment_raw(raw)

    def _parse_judgment_raw(self, raw: str) -> RiskSignalJudgment:
        text = (raw or "").strip()
        if not text:
            return self._fallback_judgment()

        payload: Any = text
        try:
            payload = json.loads(text)
        except Exception:
            start_obj = text.find("{")
            end_obj = text.rfind("}")
            if start_obj != -1 and end_obj > start_obj:
                try:
                    payload = json.loads(text[start_obj : end_obj + 1])
                except Exception:
                    payload = text
            else:
                start_arr = text.find("[")
                end_arr = text.rfind("]")
                if start_arr != -1 and end_arr > start_arr:
                    try:
                        payload = json.loads(text[start_arr : end_arr + 1])
                    except Exception:
                        payload = text

        if isinstance(payload, list):
            payload = payload[0] if payload and isinstance(payload[0], dict) else {}
        if not isinstance(payload, dict):
            return self._fallback_judgment()

        try:
            return RiskSignalJudgment.model_validate(payload)
        except Exception:
            merged = {
                "is_risk": payload.get("is_risk", payload.get("isRisk", True)),
                "issue_type": payload.get("issue_type", payload.get("issueType", "VAGUE_EXPRESSION")),
                "severity": payload.get("severity", "medium"),
                "status": payload.get("status", "uncertain"),
                "summary": payload.get("summary", payload.get("title", "检测到疑似高风险表述，请人工复核。")),
                "reasoning": payload.get("reasoning", payload.get("reason", "模型原始输出字段不完整，已降级解析。")),
                "confidence": payload.get("confidence", 0.45),
                "suggestion": payload.get("suggestion"),
            }
            try:
                return RiskSignalJudgment.model_validate(merged)
            except Exception:
                return self._fallback_judgment()

    def _fallback_judgment(self) -> RiskSignalJudgment:
        return RiskSignalJudgment(
            is_risk=True,
            issue_type="VAGUE_EXPRESSION",
            severity="medium",
            status="uncertain",
            summary="检测到疑似高风险表述，请人工复核。",
            reasoning="模型裁决失败，按规则候选保守降级输出。",
            confidence=0.45,
        )

    def _fallback_judge_candidates(self, candidates: List[RiskSignalCandidate]) -> List[RiskSignalIssue]:
        issues: List[RiskSignalIssue] = []
        for candidate in candidates:
            issues.append(
                RiskSignalIssue(
                    issue_id=stable_id("issue", candidate.candidate_id),
                    issue_type=candidate.issue_type_hint,
                    severity=candidate.severity_hint,
                    status="uncertain",
                    summary="检测到疑似高风险表述，请人工复核。",
                    reasoning="基于规则词命中召回，未经过模型确认。",
                    confidence=0.4,
                    evidence=[candidate.evidence],
                    suggestion=self._default_suggestion(candidate.issue_type_hint),
                    attributes=candidate.attributes,
                )
            )
        return issues

    def _build_result(
        self,
        resource_file: ResourceFile,
        issues: List[RiskSignalIssue],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SingleFileRiskSignalCheckResult:
        if not issues:
            overall_status = "pass"
            summary = "未发现明显高风险表述。"
        elif any(issue.status == "fail" for issue in issues):
            overall_status = "fail"
            summary = f"发现 {len(issues)} 条高风险表述，建议重点复核免责、模糊或单方解释类语句。"
        else:
            overall_status = "uncertain"
            summary = f"发现 {len(issues)} 条疑似高风险表述，建议人工复核。"
        return SingleFileRiskSignalCheckResult(
            document_id=stable_id("risk_doc", resource_file.fileId, resource_file.fileName),
            file_id=resource_file.fileId,
            file_name=resource_file.fileName,
            overall_status=overall_status,
            summary=summary,
            issue_count=len(issues),
            issues=issues,
            statistics={
                "high": sum(1 for issue in issues if issue.severity == "high"),
                "medium": sum(1 for issue in issues if issue.severity == "medium"),
                "low": sum(1 for issue in issues if issue.severity == "low"),
            },
            metadata=metadata or {},
        )

    def _segment_evidence(self, role: str, text: str, segment: DocumentSegment) -> RiskSignalEvidence:
        return RiskSignalEvidence(
            role=role,
            source_text=text,
            source_location=RiskSignalSourceLocation(
                file_id=segment.file_id,
                element_ids=segment.element_ids,
                page_no=segment.page_no,
                segment_id=segment.segment_id,
                block_type=segment.block_type,
            ),
        )

    def _dedup_candidates(self, candidates: Iterable[RiskSignalCandidate]) -> List[RiskSignalCandidate]:
        deduped: List[RiskSignalCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = stable_id("cand_dedup", candidate.issue_type_hint, candidate.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _merge_issues(self, issues: Iterable[RiskSignalIssue]) -> List[RiskSignalIssue]:
        deduped: List[RiskSignalIssue] = []
        seen: set[str] = set()
        for issue in issues:
            key = stable_id("issue_dedup", issue.issue_type, issue.summary, *(e.source_text for e in issue.evidence))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return deduped

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
    def _resolve_effective_risk_rules(
        primary_doc_type: DocType,
    ) -> List[tuple[str, str, tuple[str, ...], str]]:
        rules: List[tuple[str, str, tuple[str, ...], str]] = []
        rules.extend((issue_type, severity, keywords, "general") for issue_type, severity, keywords in RISK_RULES)
        rules.extend(
            (issue_type, severity, keywords, "type")
            for issue_type, severity, keywords in DOC_TYPE_RISK_RULES.get(primary_doc_type, [])
        )
        return rules

    def _split_sentences(self, text: str) -> List[str]:
        return [item.strip() for item in SENTENCE_SPLIT_RE.split(text) if item and item.strip()]

    def _is_noise_segment_text(self, text: str) -> bool:
        if len(text) <= 2:
            return True
        return any(token == text for token in NOISE_TOKENS)

    def _is_actionable_vague_sentence(self, sentence: str) -> bool:
        return any(token in sentence for token in VAGUE_CONTEXT_TOKENS)

    def _is_better_segment(self, candidate: DocumentSegment, existing: DocumentSegment) -> bool:
        candidate_span = len(candidate.element_ids) if candidate.element_ids else 9999
        existing_span = len(existing.element_ids) if existing.element_ids else 9999
        if candidate_span != existing_span:
            return candidate_span < existing_span
        return len(candidate.text) < len(existing.text)

    def _build_judge_prompt(self, candidate: RiskSignalCandidate) -> str:
        prompt = (
            self._risk_signal_judge_template.replace("{issue_type_hint}", candidate.issue_type_hint)
            .replace("{severity_hint}", candidate.severity_hint.upper())
            .replace("{text}", candidate.text)
            .replace("{matched_keywords}", ", ".join(candidate.attributes.get("matchedKeywords", [])))
        )
        rule_scope = str(candidate.attributes.get("ruleScope") or "")
        if rule_scope:
            prompt += f"\n[rule_scope]\n{rule_scope}\n"
        return prompt

    def _default_suggestion(self, issue_type: str) -> str:
        mapping = {
            "VAGUE_EXPRESSION": "建议改为可执行、可验证的明确条件或时限。",
            "EXTREME_EXPRESSION": "建议删除绝对化表述，改为边界清晰的责任描述。",
            "LIABILITY_DISCLAIMER": "建议补充责任边界和适用条件，避免泛化免责。",
            "UNILATERAL_INTERPRETATION": "建议删除单方解释权表述，改为双方确认或按制度解释。",
        }
        return mapping.get(issue_type, "建议补充清晰、可执行的责任和边界描述。")
