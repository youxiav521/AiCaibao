from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..legacy_schemas import EvidenceItem
from .policy_check_aggregation import PolicyCheckAggregation
from .policy_check_config import DEGRADED_REASON, PolicyCheckConfig
from .policy_relevance_profiles import ControlProfile, RequirementProfile
from .schemas import SingleFilePolicyControlResult


class PolicyCheckProjection:
    def __init__(self, *, config: PolicyCheckConfig, aggregation: PolicyCheckAggregation) -> None:
        self.config = config
        self.aggregation = aggregation

    def max_candidate_files_for_control(
        self,
        *,
        control_profile: Optional[ControlProfile],
        requirement_profile: RequirementProfile,
    ) -> int:
        if control_profile is None:
            return self.config.multi_evidence_max_files
        mode = str(control_profile.check_mode or "mixed")
        expectation = str(control_profile.evidence_expectation or "mixed")
        if mode in {"existence", "validity"} and expectation in {"doc_first", "evidence_first"}:
            return self.config.single_evidence_max_files
        if requirement_profile.collection_heavy or expectation in {"mixed", "fact_first"} or mode in {"relation", "consistency"}:
            return self.config.multi_evidence_max_files
        if mode == "mixed":
            return self.config.multi_evidence_max_files
        return self.config.repository_evidence_max_files

    def select_candidate_files_for_control(
        self,
        *,
        candidates: List[Dict[str, Any]],
        control_profile: Optional[ControlProfile],
        requirement_profile: RequirementProfile,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        budget = self.max_candidate_files_for_control(
            control_profile=control_profile,
            requirement_profile=requirement_profile,
        )
        best_score = float(candidates[0].get("decision_score", 0.0))
        selected: List[Dict[str, Any]] = []
        for item in candidates:
            if len(selected) >= budget:
                break
            score = float(item.get("decision_score", 0.0))
            if selected and (best_score - score) > self.config.candidate_file_score_margin:
                continue
            selected.append(item)
        return selected or candidates[:1]

    def project_global_control_result_to_file(
        self,
        *,
        file_id: str,
        global_result: Optional[SingleFilePolicyControlResult],
        fallback_meta: Any,
    ) -> SingleFilePolicyControlResult:
        if global_result is None:
            return SingleFilePolicyControlResult(
                requirement_id=str(fallback_meta.request.requirement_id if fallback_meta.request else ""),
                control_id=str(fallback_meta.request.control_id if fallback_meta.request else ""),
                result="uncertain",
                reason=DEGRADED_REASON,
                evidence=[],
                degraded=True,
                is_relevant=True,
                relevance_reason=str(fallback_meta.reason or ""),
                retrieval_stats=fallback_meta.stats or self.aggregation.empty_retrieval_stats(),
                reason_code=str(fallback_meta.reason_code or ""),
                decision_stage=str(fallback_meta.decision_stage or "control"),
                decision_confidence=float(fallback_meta.decision_confidence or 0.0),
                feature_scores=dict(fallback_meta.feature_scores or {}),
                llm_override_used=bool(fallback_meta.llm_override_used),
                execution_gate_passed=bool(fallback_meta.execution_gate_passed),
                fail_suppressed=False,
            )

        file_evidence = [item for item in global_result.evidence if item.fileId == file_id]
        if file_evidence:
            return global_result.model_copy(update={"evidence": file_evidence})

        inferred_evidence = self.infer_file_evidence_from_retrieval(
            file_id=file_id,
            fallback_meta=fallback_meta,
            control_id=global_result.control_id,
        )
        if global_result.result == "fail":
            if inferred_evidence:
                return global_result.model_copy(
                    update={
                        "result": "uncertain",
                        "reason": f"{global_result.reason}（当前文件存在候选证据信号，保守降级为 uncertain）",
                        "evidence": inferred_evidence,
                    }
                )
            return global_result.model_copy(
                update={
                    "result": "uncertain",
                    "reason": f"{global_result.reason}（该结论未命中当前文件证据，当前文件降级为 uncertain）",
                    "evidence": [],
                }
            )
        if global_result.result == "pass":
            if inferred_evidence:
                return global_result.model_copy(
                    update={
                        "result": "pass",
                        "reason": f"{global_result.reason}（根据候选检索信号已归属当前文件）",
                        "evidence": inferred_evidence,
                    }
                )
            return global_result.model_copy(
                update={
                    "result": "skipped",
                    "reason": "该控制项在全库执行中通过，但当前文件未作为关键证据，按单文件视角标记 skipped。",
                    "evidence": [],
                }
            )
        if inferred_evidence:
            return global_result.model_copy(
                update={
                    "result": "uncertain",
                    "reason": f"{global_result.reason}（当前文件存在候选证据信号）",
                    "evidence": inferred_evidence,
                }
            )
        return global_result.model_copy(
            update={
                "result": "uncertain",
                "reason": f"{global_result.reason}（当前文件未命中直接证据）",
                "evidence": [],
            }
        )

    def infer_file_evidence_from_retrieval(
        self,
        *,
        file_id: str,
        fallback_meta: Any,
        control_id: str,
    ) -> List[EvidenceItem]:
        retrieval = fallback_meta.retrieval
        if retrieval is None:
            return []

        inferred: List[EvidenceItem] = []
        doc_ids_by_file: Set[str] = set()
        for doc_hit in list(getattr(retrieval, "candidate_documents", []) or []):
            manifest = getattr(doc_hit, "manifest", None)
            if manifest is None or str(getattr(manifest, "file_id", "")) != file_id:
                continue
            doc_ids_by_file.add(str(getattr(manifest, "doc_id", "")))
            score = float(getattr(doc_hit, "score", 0.0) or 0.0)
            if score >= 2.0:
                inferred.append(
                    EvidenceItem(
                        evidenceId=f"infer_doc:{control_id}:{getattr(manifest, 'doc_id', '')}",
                        fileId=file_id,
                        elementIds=[],
                        pageNo=None,
                        summary=f"候选文档命中（score={score:.2f}）",
                        quotedText=str(getattr(manifest, "file_name", "") or ""),
                    )
                )

        for evidence_hit in list(getattr(retrieval, "evidence_hits", []) or []):
            block = getattr(evidence_hit, "block", None)
            if block is None or str(getattr(block, "file_id", "")) != file_id:
                continue
            score = float(getattr(evidence_hit, "score", 0.0) or 0.0)
            if score <= 0.0:
                continue
            inferred.append(
                EvidenceItem(
                    evidenceId=f"infer_block:{control_id}:{getattr(block, 'evidence_id', '')}",
                    fileId=file_id,
                    elementIds=list(getattr(block, "source_element_ids", []) or [])[:8],
                    pageNo=getattr(block, "page_no", None),
                    summary=f"候选证据块命中（score={score:.2f}）",
                    quotedText=str(getattr(block, "text", "") or "")[:120],
                )
            )

        for fact_hit in list(getattr(retrieval, "fact_hits", []) or []):
            fact = getattr(fact_hit, "fact", None)
            if fact is None:
                continue
            if str(getattr(fact, "doc_id", "")) not in doc_ids_by_file:
                continue
            score = float(getattr(fact_hit, "score", 0.0) or 0.0)
            if score <= 0.0:
                continue
            inferred.append(
                EvidenceItem(
                    evidenceId=f"infer_fact:{control_id}:{getattr(fact, 'fact_id', '')}",
                    fileId=file_id,
                    elementIds=[],
                    pageNo=None,
                    summary=f"候选事实命中（field={getattr(fact, 'field_name', '')}, score={score:.2f}）",
                    quotedText=str(getattr(fact, "field_value", "") or "")[:120],
                )
            )

        dedup: Dict[str, EvidenceItem] = {}
        for item in inferred:
            dedup[item.evidenceId] = item
        return list(dedup.values())[:3]

    def has_projection_signal(self, *, file_id: str, fallback_meta: Any) -> bool:
        retrieval = fallback_meta.retrieval
        if retrieval is None:
            return False
        for doc_hit in list(getattr(retrieval, "candidate_documents", []) or []):
            manifest = getattr(doc_hit, "manifest", None)
            if manifest is None:
                continue
            if str(getattr(manifest, "file_id", "")) == file_id and float(getattr(doc_hit, "score", 0.0) or 0.0) >= 2.0:
                return True
        for evidence_hit in list(getattr(retrieval, "evidence_hits", []) or []):
            block = getattr(evidence_hit, "block", None)
            if block is None:
                continue
            if str(getattr(block, "file_id", "")) == file_id and float(getattr(evidence_hit, "score", 0.0) or 0.0) > 0.0:
                return True
        return False
