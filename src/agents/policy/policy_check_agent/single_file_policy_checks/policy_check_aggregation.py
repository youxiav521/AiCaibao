from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..legacy_schemas import EvidenceItem, Framework, Requirement
from .policy_check_config import UNRELATED_CONTROL_REASON
from .policy_relevance_profiles import RelevanceDecision
from .schemas import (
    ControlRetrievalStats,
    SingleFilePolicyControlResult,
    SingleFilePolicyRequirementResult,
)


class PolicyCheckAggregation:
    def __init__(self, *, include_skipped_controls: bool) -> None:
        self.include_skipped_controls = include_skipped_controls

    @staticmethod
    def empty_retrieval_stats() -> ControlRetrievalStats:
        return ControlRetrievalStats()

    @staticmethod
    def build_skipped_control_result(
        *,
        requirement_id: str,
        control_id: str,
        reason: str,
        is_relevant: bool,
        relevance_reason: str,
        retrieval_stats: ControlRetrievalStats,
        reason_code: str = "",
        decision_stage: str = "",
        decision_confidence: float = 0.0,
        feature_scores: Optional[Dict[str, float]] = None,
        llm_override_used: bool = False,
        execution_gate_passed: bool = False,
        fail_suppressed: bool = False,
    ) -> SingleFilePolicyControlResult:
        return SingleFilePolicyControlResult(
            requirement_id=requirement_id,
            control_id=control_id,
            result="skipped",
            reason=reason,
            evidence=[],
            degraded=False,
            is_relevant=is_relevant,
            relevance_reason=relevance_reason,
            retrieval_stats=retrieval_stats,
            reason_code=reason_code,
            decision_stage=decision_stage,
            decision_confidence=decision_confidence,
            feature_scores=feature_scores or {},
            llm_override_used=llm_override_used,
            execution_gate_passed=execution_gate_passed,
            fail_suppressed=fail_suppressed,
        )

    def build_requirement_level_skipped_controls(
        self,
        *,
        requirement: Requirement,
        requirement_decision: RelevanceDecision,
        requirement_llm_override: bool,
        relevance_by_control: Mapping[str, Any],
    ) -> List[SingleFilePolicyControlResult]:
        if requirement_decision.decision != "hard_skip":
            skipped_controls: List[SingleFilePolicyControlResult] = []
            for control in requirement.controls:
                control_meta = relevance_by_control.get(control.controlId)
                if control_meta is None or bool(control_meta.is_relevant):
                    continue
                skipped_controls.append(
                    self.build_skipped_control_result(
                        requirement_id=requirement.requirementId,
                        control_id=control.controlId,
                        reason=control_meta.skip_reason or UNRELATED_CONTROL_REASON,
                        is_relevant=False,
                        relevance_reason=str(control_meta.reason or ""),
                        retrieval_stats=control_meta.stats or self.empty_retrieval_stats(),
                        reason_code=str(control_meta.reason_code or ""),
                        decision_stage=str(control_meta.decision_stage or "control"),
                        decision_confidence=float(control_meta.decision_confidence or 0.0),
                        feature_scores=dict(control_meta.feature_scores or {}),
                        llm_override_used=bool(control_meta.llm_override_used),
                        execution_gate_passed=bool(control_meta.execution_gate_passed),
                    )
                )
            return skipped_controls

        reason_text = "; ".join(requirement_decision.reasons) or "当前单文档与该要求目标材料类型高置信不匹配，已自动跳过。"
        skip_reason = "当前单文档与该要求目标材料类型高置信不匹配，已自动跳过。"
        return [
            self.build_skipped_control_result(
                requirement_id=requirement.requirementId,
                control_id=control.controlId,
                reason=skip_reason,
                is_relevant=False,
                relevance_reason=reason_text,
                retrieval_stats=self.empty_retrieval_stats(),
                reason_code=requirement_decision.reason_code,
                decision_stage="requirement",
                decision_confidence=requirement_decision.confidence,
                feature_scores=requirement_decision.feature_scores,
                llm_override_used=requirement_llm_override,
            )
            for control in requirement.controls
        ]

    def build_requirement_result(
        self,
        *,
        framework: Framework,
        requirement: Requirement,
        control_results: List[SingleFilePolicyControlResult],
        skipped_control_results: List[SingleFilePolicyControlResult],
        degraded: bool,
        reason_override: Optional[str] = None,
        result_override: Optional[str] = None,
    ) -> SingleFilePolicyRequirementResult:
        included_controls = list(control_results)
        if self.include_skipped_controls:
            included_controls.extend(skipped_control_results)

        if result_override:
            result = result_override
            reason = reason_override or ""
            evidence: List[EvidenceItem] = []
        else:
            aggregated = self.aggregate_requirement_result(
                requirement,
                [item.model_dump(by_alias=True, mode="python") for item in [*control_results, *skipped_control_results]],
            )
            result = aggregated.get("result", "uncertain")
            reason = reason_override or aggregated.get("reason", "")
            evidence = self._normalize_evidence(aggregated.get("evidence", []))

        return SingleFilePolicyRequirementResult(
            framework_id=framework.frameworkId,
            framework_name=framework.name,
            requirement_id=requirement.requirementId,
            requirement_title=requirement.title,
            result=result,
            reason=reason,
            evidence=evidence,
            control_results=included_controls,
            matched_control_count=len(control_results),
            skipped_control_count=len(skipped_control_results),
            degraded=degraded,
        )

    @staticmethod
    def aggregate_requirement_result(requirement: Requirement, control_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        deduped_by_control: Dict[str, Dict[str, Any]] = {}
        for item in control_results:
            control_id = item.get("controlId")
            if not control_id or control_id in deduped_by_control:
                continue
            deduped_by_control[control_id] = item
        ordered_results = [
            deduped_by_control[control.controlId]
            for control in requirement.controls
            if control.controlId in deduped_by_control
        ]
        statuses = [item.get("result", "uncertain") for item in ordered_results]
        if any(status == "fail" for status in statuses):
            final_result = "fail"
        elif any(status == "uncertain" for status in statuses):
            final_result = "uncertain"
        elif statuses and all(status == "skipped" for status in statuses):
            final_result = "skipped"
        else:
            final_result = "pass"

        reason_parts: List[str] = []
        evidence: List[Dict[str, Any]] = []
        seen_evidence = set()
        for item in ordered_results:
            control_id = item.get("controlId", "")
            reason = item.get("reason", "")
            if reason:
                reason_parts.append(f"{control_id}: {reason}")
            for ev in item.get("evidence", []):
                key = (
                    ev.get("fileId", ""),
                    tuple(ev.get("elementIds", []) or []),
                    ev.get("pageNo", None),
                    ev.get("summary", "") or ev.get("evidenceId", ""),
                )
                if key in seen_evidence:
                    continue
                seen_evidence.add(key)
                evidence.append(
                    {
                        "evidenceId": ev.get("evidenceId") or f"evidence_{len(evidence) + 1}",
                        "fileId": ev.get("fileId", ""),
                        "elementIds": ev.get("elementIds", []) or [],
                        "pageNo": ev.get("pageNo"),
                        "summary": ev.get("summary"),
                        "quotedText": ev.get("quotedText"),
                    }
                )

        if not reason_parts:
            reason_parts.append("未产出有效的规则结果，当前要求无法可靠判定。")
            final_result = "uncertain"

        return {
            "requirementId": requirement.requirementId,
            "result": final_result,
            "reason": "; ".join(reason_parts),
            "evidence": evidence,
            "controlResults": ordered_results,
        }

    @staticmethod
    def aggregate_file_status(requirements: List[SingleFilePolicyRequirementResult]) -> str:
        statuses = [item.result for item in requirements]
        if any(status == "fail" for status in statuses):
            return "fail"
        if any(status == "uncertain" for status in statuses):
            return "uncertain"
        if statuses and all(status == "skipped" for status in statuses):
            return "skipped"
        if not statuses:
            return "skipped"
        return "pass"

    @staticmethod
    def build_statistics(
        requirements: List[SingleFilePolicyRequirementResult],
        matched_controls: List[SingleFilePolicyControlResult],
        skipped_controls: List[SingleFilePolicyControlResult],
    ) -> Dict[str, int]:
        requirement_counts = {
            "totalRequirements": len(requirements),
            "passedRequirements": 0,
            "failedRequirements": 0,
            "uncertainRequirements": 0,
            "skippedRequirements": 0,
        }
        for requirement in requirements:
            if requirement.result == "pass":
                requirement_counts["passedRequirements"] += 1
            elif requirement.result == "fail":
                requirement_counts["failedRequirements"] += 1
            elif requirement.result == "skipped":
                requirement_counts["skippedRequirements"] += 1
            else:
                requirement_counts["uncertainRequirements"] += 1

        control_counts = {
            "matchedControls": len(matched_controls),
            "skippedControls": len(skipped_controls),
            "passedControls": 0,
            "failedControls": 0,
            "uncertainControls": 0,
            "degradedControls": 0,
        }
        for control in matched_controls:
            if control.result == "pass":
                control_counts["passedControls"] += 1
            elif control.result == "fail":
                control_counts["failedControls"] += 1
            elif control.result == "uncertain":
                control_counts["uncertainControls"] += 1
            if control.degraded:
                control_counts["degradedControls"] += 1

        return {
            **requirement_counts,
            **control_counts,
            "totalControls": len(matched_controls) + len(skipped_controls),
        }

    @staticmethod
    def build_summary(file_name: str, overall_status: str, statistics: Dict[str, int]) -> str:
        return (
            f"文件 {file_name} 审查完成，整体状态为 {overall_status}。"
            f"共处理 {statistics.get('totalRequirements', 0)} 条要求、{statistics.get('totalControls', 0)} 条规则，"
            f"其中命中 {statistics.get('matchedControls', 0)} 条，跳过 {statistics.get('skippedControls', 0)} 条。"
        )

    @staticmethod
    def build_llm_gate_stats(
        matched_controls: List[SingleFilePolicyControlResult],
        skipped_controls: List[SingleFilePolicyControlResult],
    ) -> Dict[str, int]:
        controls = list(matched_controls or []) + list(skipped_controls or [])
        vetoed = 0
        promoted = 0
        hold = 0
        for control in controls:
            code = str(getattr(control, "reason_code", "") or "")
            if "_LLM_GATE_VETO" in code:
                vetoed += 1
            elif "_LLM_GATE_PROMOTE" in code:
                promoted += 1
            elif "_LLM_GATE_HOLD" in code:
                hold += 1
        return {
            "llmGateOverridden": vetoed + promoted + hold,
            "llmGateVetoed": vetoed,
            "llmGatePromoted": promoted,
            "llmGateHeld": hold,
        }

    @staticmethod
    def requirement_order_key(framework: Framework, requirement: Requirement) -> str:
        return f"{framework.frameworkId}::{requirement.requirementId}"

    @staticmethod
    def _normalize_evidence(items: List[Dict[str, Any]]) -> List[EvidenceItem]:
        normalized: List[EvidenceItem] = []
        for item in items or []:
            normalized.append(EvidenceItem.model_validate(item))
        return normalized

    def normalize_evidence(self, items: List[Dict[str, Any]]) -> List[EvidenceItem]:
        return self._normalize_evidence(items)
