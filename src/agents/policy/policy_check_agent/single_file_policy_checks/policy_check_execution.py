from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


from ..legacy_schemas import Framework, Requirement, ResourceRepository
from .policy_check_aggregation import PolicyCheckAggregation
from .policy_check_config import DEGRADED_REASON, PolicyCheckConfig
from .schemas import SingleFilePolicyControlResult
from loguru import logger

_INTERNAL_EVIDENCE_ID_RE = re.compile(r"\bevidence[_:\-]?[A-Za-z0-9]+\b", flags=re.IGNORECASE)


@dataclass(slots=True)
class ExecutionOutcome:
    control_results: List[SingleFilePolicyControlResult]
    degraded: bool


class PolicyCheckExecution:
    def __init__(
        self,
        *,
        config: PolicyCheckConfig,
        aggregation: PolicyCheckAggregation,
        policy_agent_provider: Callable[[str], Any],
    ) -> None:
        self.config = config
        self.aggregation = aggregation
        self.policy_agent_provider = policy_agent_provider

    def build_fast_path_control_results(
        self,
        *,
        requirement: Requirement,
        relevant_controls: List[Any],
        relevance_by_control: Dict[str, Any],
    ) -> Optional[List[SingleFilePolicyControlResult]]:
        if not relevant_controls:
            return None
        if not all(relevance_by_control[control.controlId].is_fast_fail_candidate for control in relevant_controls):
            return None
        results: List[SingleFilePolicyControlResult] = []
        for control in relevant_controls:
            control_meta = relevance_by_control[control.controlId]
            direct_fail = bool(control_meta.allow_direct_fail)
            control_result = "fail" if direct_fail else "uncertain"
            reason = (
                "当前单文档在高置信条件下无法提供该规则所需材料，已直接判定 fail。"
                if direct_fail
                else "当前单文档疑似缺少该规则所需材料，触发 fast-fail 候选并保守输出 uncertain。"
            )
            results.append(
                SingleFilePolicyControlResult(
                    requirement_id=requirement.requirementId,
                    control_id=control.controlId,
                    result=control_result,
                    reason=reason,
                    evidence=[],
                    degraded=False,
                    is_relevant=True,
                    relevance_reason=control_meta.reason,
                    retrieval_stats=control_meta.stats,
                    reason_code=control_meta.reason_code,
                    decision_stage="control_fast_fail",
                    decision_confidence=float(control_meta.decision_confidence or 0.0),
                    feature_scores=dict(control_meta.feature_scores or {}),
                    llm_override_used=bool(control_meta.llm_override_used),
                    execution_gate_passed=bool(control_meta.execution_gate_passed),
                    fail_suppressed=False,
                )
            )
        return results

    async def execute_relevant_controls(
        self,
        *,
        framework: Framework,
        requirement: Requirement,
        relevant_controls: List[Any],
        relevance_by_control: Dict[str, Any],
        repository: ResourceRepository,
        execution_semaphore: asyncio.Semaphore,
        effective_model_id: str,
    ) -> ExecutionOutcome:
        policy_agent = self.policy_agent_provider(effective_model_id)
        filtered_requirement = requirement.model_copy(update={"controls": relevant_controls})
        try:
            invoke_result = await policy_agent._execute_requirement_with_retry(
                resource_repository=repository,
                requirement=filtered_requirement,
                framework_id=framework.frameworkId,
                event_queue=asyncio.Queue(),
                execution_semaphore=execution_semaphore,
            )
        except Exception as exc:
            logger.warning(
                "single_file_policy_check requirement execution failed requirement_id={} framework_id={} error={}",
                requirement.requirementId,
                framework.frameworkId,
                exc,
            )
            invoke_result = self._build_degraded_requirement_result(filtered_requirement)

        raw_control_results = [dict(item) for item in (invoke_result.get("control_results") or [])]
        aggregated = self.aggregation.aggregate_requirement_result(filtered_requirement, raw_control_results)
        check_results = invoke_result.get("check_results") or {}
        if not isinstance(check_results, dict) or not check_results.get("result"):
            check_results = aggregated
        elif not check_results.get("controlResults"):
            check_results["controlResults"] = aggregated["controlResults"]

        reported_ids = {item.get("controlId") for item in raw_control_results}
        for control in relevant_controls:
            if control.controlId in reported_ids:
                continue
            raw_control_results.append(
                self._build_fallback_control_payload(
                    control_id=control.controlId,
                    requirement_result=check_results,
                    requirement_id=requirement.requirementId,
                )
            )

        matched_control_results = []
        for item in raw_control_results:
            control_id = item.get("controlId") or ""
            control_meta = relevance_by_control.get(control_id)
            if control_meta is None:
                continue
            raw_result = item.get("result", "uncertain")
            raw_reason = self._normalize_external_reason(item.get("reason", ""))
            fail_suppressed = False
            confirmed_missing_artifact = self._is_confirmed_missing_artifact_reason(raw_reason)
            if confirmed_missing_artifact and raw_result != "pass":
                raw_result = "fail"
                if "关键材料已确认缺失" not in raw_reason:
                    raw_reason = f"{raw_reason}（关键材料已确认缺失）"
            if raw_result == "fail" and not confirmed_missing_artifact and self._should_suppress_fail(control_meta):
                fail_suppressed = True
                raw_result = "uncertain"
                raw_reason = self._normalize_external_reason(
                    f"{raw_reason}（当前证据强度不足，结论调整为待人工复核）"
                )
            matched_control_results.append(
                SingleFilePolicyControlResult(
                    requirement_id=requirement.requirementId,
                    control_id=control_id,
                    result=raw_result,
                    reason=raw_reason,
                    evidence=self.aggregation.normalize_evidence(item.get("evidence", [])),
                    degraded=bool(item.get("degraded")),
                    is_relevant=True,
                    relevance_reason=control_meta.reason,
                    retrieval_stats=control_meta.stats,
                    reason_code=control_meta.reason_code,
                    decision_stage=str(control_meta.decision_stage or "control"),
                    decision_confidence=float(control_meta.decision_confidence or 0.0),
                    feature_scores=dict(control_meta.feature_scores or {}),
                    llm_override_used=bool(control_meta.llm_override_used),
                    execution_gate_passed=bool(control_meta.execution_gate_passed),
                    fail_suppressed=fail_suppressed,
                )
            )
        degraded = any(item.degraded for item in matched_control_results)
        return ExecutionOutcome(control_results=matched_control_results, degraded=degraded)

    @staticmethod
    def _normalize_external_reason(reason: str) -> str:
        text = str(reason or "").strip()
        if not text:
            return DEGRADED_REASON
        if _INTERNAL_EVIDENCE_ID_RE.search(text):
            return "当前文件未命中直接证据。"
        lowered = text.lower()
        english_degraded_markers = (
            "automatic fallback was used because the verification process could not complete reliably",
            "fallback was used",
            "verification process could not complete reliably",
            "fallback to requirement-level result",
            "returned non-json content",
        )
        if any(marker in lowered for marker in english_degraded_markers):
            return DEGRADED_REASON
        timeout_markers = (
            "timeout",
            "timed out",
            "soft timeout",
            "split execution hit soft timeout",
            "exceeded control-level time budget",
            "was not completed within requirement-level time budget",
        )
        if any(marker in lowered for marker in timeout_markers):
            return "校验过程超时，已自动降级为待人工复核。"
        translation_map = (
            ("has no usable image content", "关键图片材料不可用或缺失。"),
            ("image artifact is missing on disk", "关键图片材料缺失。"),
            ("missing or cannot be matched exactly", "关键材料缺失或无法精确匹配。"),
            ("confirmed missing", "关键材料已确认缺失。"),
            ("file does not exist", "关键材料文件不存在。"),
            ("cannot read", "关键材料无法读取。"),
        )
        for marker, zh in translation_map:
            if marker in lowered:
                return zh
        return text

    def _should_suppress_fail(self, control_meta: Any) -> bool:
        if bool(control_meta.allow_direct_fail):
            return False
        feature_scores = dict(control_meta.feature_scores or {})
        decision_score = float(control_meta.decision_score or 0.0)
        retrieval_strength = float(feature_scores.get("retrieval_strength", 0.0))
        if decision_score < self.config.fail_relevance_floor:
            return True
        if retrieval_strength < self.config.fail_retrieval_floor:
            return True
        if str(control_meta.decision_stage or "").endswith("hold"):
            return True
        return False

    @staticmethod
    def _is_confirmed_missing_artifact_reason(reason: str) -> bool:
        text = str(reason or "")
        if not text:
            return False
        markers = (
            "confirmed missing",
            "已确认缺失",
            "已确认不存在",
            "missing or cannot be matched exactly",
            "has no usable image content",
            "image artifact is missing on disk",
            "图片文件不存在",
            "文件不存在",
            "无法读取",
        )
        lowered = text.lower()
        return any(marker in text or marker in lowered for marker in markers)

    @staticmethod
    def _build_fallback_control_payload(
        *,
        control_id: str,
        requirement_result: Dict[str, Any],
        requirement_id: str,
    ) -> Dict[str, Any]:
        return {
            "requirementId": requirement_id,
            "controlId": control_id,
            "result": requirement_result.get("result", "uncertain"),
            "reason": requirement_result.get(
                "reason",
                "未产出规则级独立结果，已回退为要求级结果。",
            ),
            "evidence": requirement_result.get("evidence", []),
            "degraded": True,
        }

    @staticmethod
    def _build_degraded_requirement_result(requirement: Requirement) -> Dict[str, Any]:
        control_results = [
            {
                "requirementId": requirement.requirementId,
                "controlId": control.controlId,
                "result": "uncertain",
                "reason": DEGRADED_REASON,
                "evidence": [],
                "degraded": True,
            }
            for control in requirement.controls
        ]
        return {
            "result": {"messages": [], "todos": []},
            "check_results": {
                "requirementId": requirement.requirementId,
                "result": "uncertain",
                "reason": DEGRADED_REASON,
                "evidence": [],
                "controlResults": control_results,
            },
            "control_results": control_results,
        }
