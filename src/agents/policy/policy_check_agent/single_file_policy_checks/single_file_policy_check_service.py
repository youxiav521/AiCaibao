from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set


from ..agents.agent import PolicyCheckAgent
from ..legacy_schemas import Framework, Requirement, ResourceFile, ResourceRepository
from ..services.material_loader import build_material_store
from ..services.material_retriever import MaterialRetriever
from ..utils import stable_id
from .policy_check_aggregation import PolicyCheckAggregation
from loguru import logger
from .policy_check_builders import (
    build_default_file_id,
    build_multi_file_repository,
    build_resource_file_from_input,
    build_single_file_repository,
    validate_max_concurrency,
)
from .policy_check_config import PolicyCheckConfig, RELEVANCE_ONLY_REASON, UNRELATED_CONTROL_REASON
from .policy_check_context import PolicyCheckRuntimeContext
from .policy_check_execution import ExecutionOutcome, PolicyCheckExecution
from .policy_check_projection import PolicyCheckProjection
from .policy_check_relevance import ControlRelevanceRecord, PolicyCheckRelevance, RequirementReviewState
from .policy_relevance_matcher import PolicyRelevanceMatcher
from .schemas import (
    SingleFilePolicyCheckRequest,
    SingleFilePolicyCheckResult,
    SingleFilePolicyControlResult,
    SingleFilePolicyInputFile,
    SingleFilePolicyRequirementResult,
)



class SingleFilePolicyCheckService:
    def __init__(
        self,
        *,
        model_id: str = "model_bailian_policy_check",
        max_concurrency: int = 12,
        relevance_threshold: float = 4.5,
        include_skipped_controls: bool = True,
        policy_agent: Optional[Any] = None,
        enable_relevance_llm_profiles: bool = False,
        enable_relevance_llm_adjudication: bool = False,
        gray_zone_lower: float = 0.35,
        gray_zone_upper: float = 0.55,
        execution_retrieval_floor: float = 0.22,
        fail_relevance_floor: float = 0.58,
        fail_retrieval_floor: float = 0.2,
        review_execute_score_floor: float = 0.58,
        review_execute_retrieval_floor: float = 0.28,
        max_review_execute_controls_per_requirement: int = 12,
        single_evidence_max_files: int = 2,
        multi_evidence_max_files: int = 4,
        repository_evidence_max_files: int = 6,
        candidate_file_score_margin: float = 0.06,
        llm_veto_confidence: float = 0.62,
        llm_promote_confidence: float = 0.78,
        llm_review_execute_confirm_confidence: float = 0.62,
    ) -> None:
        self.config = PolicyCheckConfig(
            model_id=model_id,
            max_concurrency=max_concurrency,
            relevance_threshold=relevance_threshold,
            include_skipped_controls=include_skipped_controls,
            enable_relevance_llm_profiles=enable_relevance_llm_profiles,
            enable_relevance_llm_adjudication=enable_relevance_llm_adjudication,
            gray_zone_lower=gray_zone_lower,
            gray_zone_upper=gray_zone_upper,
            execution_retrieval_floor=execution_retrieval_floor,
            fail_relevance_floor=fail_relevance_floor,
            fail_retrieval_floor=fail_retrieval_floor,
            review_execute_score_floor=review_execute_score_floor,
            review_execute_retrieval_floor=review_execute_retrieval_floor,
            max_review_execute_controls_per_requirement=max_review_execute_controls_per_requirement,
            single_evidence_max_files=single_evidence_max_files,
            multi_evidence_max_files=multi_evidence_max_files,
            repository_evidence_max_files=repository_evidence_max_files,
            candidate_file_score_margin=candidate_file_score_margin,
            llm_veto_confidence=llm_veto_confidence,
            llm_promote_confidence=llm_promote_confidence,
            llm_review_execute_confirm_confidence=llm_review_execute_confirm_confidence,
        )
        self.model_id = self.config.model_id
        self.max_concurrency = self.config.max_concurrency
        self.relevance_threshold = self.config.relevance_threshold
        self.include_skipped_controls = self.config.include_skipped_controls
        self.enable_relevance_llm_profiles = self.config.enable_relevance_llm_profiles
        self.enable_relevance_llm_adjudication = self.config.enable_relevance_llm_adjudication
        self.gray_zone_lower = self.config.gray_zone_lower
        self.gray_zone_upper = self.config.gray_zone_upper
        self.execution_retrieval_floor = self.config.execution_retrieval_floor
        self.fail_relevance_floor = self.config.fail_relevance_floor
        self.fail_retrieval_floor = self.config.fail_retrieval_floor
        self.review_execute_score_floor = self.config.review_execute_score_floor
        self.review_execute_retrieval_floor = self.config.review_execute_retrieval_floor
        self.max_review_execute_controls_per_requirement = self.config.max_review_execute_controls_per_requirement
        self.single_evidence_max_files = self.config.single_evidence_max_files
        self.multi_evidence_max_files = self.config.multi_evidence_max_files
        self.repository_evidence_max_files = self.config.repository_evidence_max_files
        self.candidate_file_score_margin = self.config.candidate_file_score_margin
        self.llm_veto_confidence = self.config.llm_veto_confidence
        self.llm_promote_confidence = self.config.llm_promote_confidence
        self.llm_review_execute_confirm_confidence = self.config.llm_review_execute_confirm_confidence

        self._matcher = PolicyRelevanceMatcher()
        self._policy_agents_by_model: Dict[str, Any] = {}
        self._llm_gate_cache: Dict[str, Any] = {}
        if policy_agent is not None:
            self._policy_agents_by_model[model_id] = policy_agent

        self.aggregation = PolicyCheckAggregation(include_skipped_controls=self.include_skipped_controls)
        self.relevance = PolicyCheckRelevance(
            config=self.config,
            matcher=self._matcher,
            llm_gate_cache=self._llm_gate_cache,
        )
        self.projection = PolicyCheckProjection(config=self.config, aggregation=self.aggregation)
        self.execution = PolicyCheckExecution(
            config=self.config,
            aggregation=self.aggregation,
            policy_agent_provider=self._get_policy_agent,
        )

    async def analyze_request(self, request: SingleFilePolicyCheckRequest) -> SingleFilePolicyCheckResult:
        hybrid_results = await self.analyze_request_hybrid(request)
        if not hybrid_results:
            raise ValueError("files is required")
        return hybrid_results[0]

    async def analyze_request_hybrid(self, request: SingleFilePolicyCheckRequest) -> List[SingleFilePolicyCheckResult]:
        if not request.files:
            raise ValueError("files is required")
        effective_model_id = (request.model_id or self.model_id).strip()
        effective_max_concurrency = self._validate_max_concurrency(request.max_concurrency or self.max_concurrency)
        effective_relevance_threshold = self.relevance_threshold
        return await self.analyze_input_files_hybrid(
            request.files,
            frameworks=request.frameworks,
            metadata={"checkReportId": request.check_report_id},
            relevance_only=False,
            effective_model_id=effective_model_id,
            effective_max_concurrency=effective_max_concurrency,
            effective_relevance_threshold=effective_relevance_threshold,
        )

    async def analyze_input_files_hybrid(
        self,
        input_files: List[SingleFilePolicyInputFile],
        *,
        frameworks: List[Framework],
        metadata: Optional[Dict[str, Any]] = None,
        relevance_only: bool = False,
        effective_model_id: Optional[str] = None,
        effective_max_concurrency: Optional[int] = None,
        effective_relevance_threshold: Optional[float] = None,
    ) -> List[SingleFilePolicyCheckResult]:
        resolved_model_id = (effective_model_id or self.model_id).strip()
        resolved_max_concurrency = self._validate_max_concurrency(effective_max_concurrency or self.max_concurrency)
        resolved_relevance_threshold = (
            self.relevance_threshold if effective_relevance_threshold is None else float(effective_relevance_threshold)
        )
        resource_files = [self.build_resource_file_from_input(input_file) for input_file in input_files]
        if not resource_files:
            return []

        repository = self._build_multi_file_repository(resource_files)
        store = build_material_store(repository, case_id=repository.id)
        retriever = MaterialRetriever(store)
        execution_semaphore = asyncio.Semaphore(resolved_max_concurrency)
        profile_tasks = [
            asyncio.create_task(
                self.relevance.build_document_profile(resource_file, effective_model_id=resolved_model_id),
            )
            for resource_file in resource_files
        ]
        built_profiles = await asyncio.gather(*profile_tasks) if profile_tasks else []
        profiles_by_file_id = {resource_file.fileId: profile for resource_file, profile in zip(resource_files, built_profiles)}
        context = PolicyCheckRuntimeContext(
            model_id=resolved_model_id,
            max_concurrency=resolved_max_concurrency,
            relevance_threshold=resolved_relevance_threshold,
            repository=repository,
            retriever=retriever,
            execution_semaphore=execution_semaphore,
            document_profiles=profiles_by_file_id,
        )

        review_tasks = [
            asyncio.create_task(
                self._review_requirement_hybrid(
                    framework=framework,
                    requirement=requirement,
                    context=context,
                    relevance_only=relevance_only,
                )
            )
            for framework in frameworks
            for requirement in framework.requirements
        ]
        reviewed_requirements = await asyncio.gather(*review_tasks) if review_tasks else []
        reviewed_requirements.sort(key=lambda item: item["order"])

        requirement_results_by_file: Dict[str, List[SingleFilePolicyRequirementResult]] = defaultdict(list)
        matched_controls_by_file: Dict[str, List[SingleFilePolicyControlResult]] = defaultdict(list)
        skipped_controls_by_file: Dict[str, List[SingleFilePolicyControlResult]] = defaultdict(list)
        for reviewed in reviewed_requirements:
            per_file = reviewed.get("per_file", {})
            for file_id, payload in per_file.items():
                requirement_results_by_file[file_id].append(payload["result"])
                matched_controls_by_file[file_id].extend(payload["matched_controls"])
                skipped_controls_by_file[file_id].extend(payload["skipped_controls"])

        results: List[SingleFilePolicyCheckResult] = []
        for resource_file in resource_files:
            file_requirements = requirement_results_by_file.get(resource_file.fileId, [])
            file_matched_controls = matched_controls_by_file.get(resource_file.fileId, [])
            file_skipped_controls = skipped_controls_by_file.get(resource_file.fileId, [])
            overall_status = self.aggregation.aggregate_file_status(file_requirements)
            statistics = self.aggregation.build_statistics(file_requirements, file_matched_controls, file_skipped_controls)
            summary = self.aggregation.build_summary(resource_file.fileName, overall_status, statistics)
            result_metadata = {
                "modelId": resolved_model_id,
                "relevanceThreshold": resolved_relevance_threshold,
                "relevanceOnly": relevance_only,
                "executionMode": "hybrid_repository_per_requirement",
                "llmGateStats": self.aggregation.build_llm_gate_stats(file_matched_controls, file_skipped_controls),
                **(metadata or {}),
            }
            results.append(
                SingleFilePolicyCheckResult(
                    document_id=stable_id("single-file-policy-check", resource_file.fileId, resource_file.fileName),
                    file_id=resource_file.fileId,
                    file_name=resource_file.fileName,
                    overall_status=overall_status,
                    summary=summary,
                    statistics=statistics,
                    requirements=file_requirements,
                    matched_controls=file_matched_controls,
                    skipped_controls=file_skipped_controls if self.include_skipped_controls else [],
                    metadata=result_metadata,
                )
            )
        return results

    async def analyze_input_file(
        self,
        input_file: SingleFilePolicyInputFile,
        *,
        frameworks: List[Framework],
        metadata: Optional[Dict[str, Any]] = None,
        relevance_only: bool = False,
    ) -> SingleFilePolicyCheckResult:
        resource_file = self.build_resource_file_from_input(input_file)
        return await self.analyze_resource_file(
            resource_file,
            frameworks=frameworks,
            metadata=metadata,
            relevance_only=relevance_only,
            effective_model_id=self.model_id,
            effective_max_concurrency=self.max_concurrency,
            effective_relevance_threshold=self.relevance_threshold,
        )

    async def analyze_resource_file(
        self,
        resource_file: ResourceFile,
        *,
        frameworks: List[Framework],
        metadata: Optional[Dict[str, Any]] = None,
        relevance_only: bool = False,
        effective_model_id: Optional[str] = None,
        effective_max_concurrency: Optional[int] = None,
        effective_relevance_threshold: Optional[float] = None,
    ) -> SingleFilePolicyCheckResult:
        resolved_model_id = (effective_model_id or self.model_id).strip()
        resolved_max_concurrency = self._validate_max_concurrency(effective_max_concurrency or self.max_concurrency)
        resolved_relevance_threshold = (
            self.relevance_threshold if effective_relevance_threshold is None else float(effective_relevance_threshold)
        )
        repository = self._build_single_file_repository(resource_file)
        store = build_material_store(repository, case_id=repository.id)
        retriever = MaterialRetriever(store)
        execution_semaphore = asyncio.Semaphore(resolved_max_concurrency)
        document_profile = await self.relevance.build_document_profile(
            resource_file,
            effective_model_id=resolved_model_id,
        )
        context = PolicyCheckRuntimeContext(
            model_id=resolved_model_id,
            max_concurrency=resolved_max_concurrency,
            relevance_threshold=resolved_relevance_threshold,
            repository=repository,
            retriever=retriever,
            execution_semaphore=execution_semaphore,
            document_profiles={resource_file.fileId: document_profile},
        )
        tasks = [
            asyncio.create_task(
                self._review_requirement(
                    framework=framework,
                    requirement=requirement,
                    context=context,
                    relevance_only=relevance_only,
                    document_profile=document_profile,
                )
            )
            for framework in frameworks
            for requirement in framework.requirements
        ]
        reviewed_requirements = await asyncio.gather(*tasks) if tasks else []
        reviewed_requirements.sort(key=lambda item: item["order"])

        requirement_results = [item["result"] for item in reviewed_requirements]
        matched_controls = [control for item in reviewed_requirements for control in item["matched_controls"]]
        skipped_controls = [control for item in reviewed_requirements for control in item["skipped_controls"]]
        overall_status = self.aggregation.aggregate_file_status(requirement_results)
        statistics = self.aggregation.build_statistics(requirement_results, matched_controls, skipped_controls)
        summary = self.aggregation.build_summary(resource_file.fileName, overall_status, statistics)
        result_metadata = {
            "modelId": resolved_model_id,
            "relevanceThreshold": resolved_relevance_threshold,
            "relevanceOnly": relevance_only,
            "llmGateStats": self.aggregation.build_llm_gate_stats(matched_controls, skipped_controls),
            **(metadata or {}),
        }
        return SingleFilePolicyCheckResult(
            document_id=stable_id("single-file-policy-check", resource_file.fileId, resource_file.fileName),
            file_id=resource_file.fileId,
            file_name=resource_file.fileName,
            overall_status=overall_status,
            summary=summary,
            statistics=statistics,
            requirements=requirement_results,
            matched_controls=matched_controls,
            skipped_controls=skipped_controls if self.include_skipped_controls else [],
            metadata=result_metadata,
        )

    async def _review_requirement(
        self,
        *,
        framework: Framework,
        requirement: Requirement,
        context: PolicyCheckRuntimeContext,
        relevance_only: bool,
        document_profile: Any,
    ) -> Dict[str, Any]:
        file_id = next(iter(context.document_profiles.keys()))
        state = await self._prepare_requirement_review_state(requirement=requirement, context=context)
        requirement_decision = state.requirement_decisions_by_file[file_id]
        requirement_llm_override_used = state.requirement_llm_override_by_file[file_id]
        if requirement_decision.decision in {"hard_skip", "soft_skip"}:
            skipped_control_results = [
                self.aggregation.build_skipped_control_result(
                    requirement_id=requirement.requirementId,
                    control_id=control.controlId,
                    reason=(
                        "当前单文档与该要求相关性不足，已自动跳过。"
                        if requirement_decision.decision == "soft_skip"
                        else "当前单文档与该要求目标材料类型高置信不匹配，已自动跳过。"
                    ),
                    is_relevant=False,
                    relevance_reason="; ".join(requirement_decision.reasons),
                    retrieval_stats=self.aggregation.empty_retrieval_stats(),
                    reason_code=requirement_decision.reason_code,
                    decision_stage="requirement",
                    decision_confidence=requirement_decision.confidence,
                    feature_scores=requirement_decision.feature_scores,
                    llm_override_used=requirement_llm_override_used,
                )
                for control in requirement.controls
            ]
            requirement_result = self.aggregation.build_requirement_result(
                framework=framework,
                requirement=requirement,
                control_results=[],
                skipped_control_results=skipped_control_results,
                degraded=False,
                reason_override="; ".join(requirement_decision.reasons),
                result_override="skipped",
            )
            return self._single_requirement_payload(framework, requirement, requirement_result, [], skipped_control_results)

        relevance_by_control: Dict[str, ControlRelevanceRecord] = {}
        relevant_controls: List[Any] = []
        skipped_control_results: List[SingleFilePolicyControlResult] = []
        for control in requirement.controls:
            record = await self.relevance.evaluate_control_relevance(
                requirement=requirement,
                control=control,
                retriever=context.retriever,
                document_profile=document_profile,
                relevance_threshold=context.relevance_threshold,
                requirement_profile=state.requirement_profile,
                control_profile=state.control_profiles[control.controlId],
                enriched_request=state.enriched_requests[control.controlId],
                effective_model_id=context.model_id,
            )
            relevance_by_control[control.controlId] = record
            if record.is_relevant:
                relevant_controls.append(control)
            else:
                skipped_control_results.append(self._build_control_skip_result(requirement.requirementId, control.controlId, record))

        if not relevant_controls:
            requirement_result = self.aggregation.build_requirement_result(
                framework=framework,
                requirement=requirement,
                control_results=[],
                skipped_control_results=skipped_control_results,
                degraded=False,
                reason_override="该要求下未识别到与当前单文档明确相关的规则。",
                result_override="skipped",
            )
            return self._single_requirement_payload(framework, requirement, requirement_result, [], skipped_control_results)

        if relevance_only:
            matched = [
                self._build_relevance_only_result(requirement.requirementId, control.controlId, relevance_by_control[control.controlId])
                for control in relevant_controls
            ]
            requirement_result = self.aggregation.build_requirement_result(
                framework=framework,
                requirement=requirement,
                control_results=matched,
                skipped_control_results=skipped_control_results,
                degraded=False,
                reason_override="已识别到相关规则，但因仅启用相关性判断模式，已跳过正式审查。",
                result_override="skipped",
            )
            return self._single_requirement_payload(framework, requirement, requirement_result, matched, skipped_control_results)

        relevant_controls, execution_dropped_ids = self.relevance.prune_controls_for_execution(
            relevant_controls=relevant_controls,
            relevance_by_control=relevance_by_control,
        )
        self._append_execution_dropped_skips(
            requirement=requirement,
            execution_dropped_ids=execution_dropped_ids,
            relevance_by_control=relevance_by_control,
            skipped_controls=skipped_control_results,
        )
        if not relevant_controls:
            requirement_result = self.aggregation.build_requirement_result(
                framework=framework,
                requirement=requirement,
                control_results=[],
                skipped_control_results=skipped_control_results,
                degraded=False,
                reason_override="该要求下候选规则相关性较弱或执行优先级不足，已自动跳过正式检查。",
                result_override="skipped",
            )
            return self._single_requirement_payload(framework, requirement, requirement_result, [], skipped_control_results)

        matched_control_results, degraded = await self._run_requirement_execution(
            framework=framework,
            requirement=requirement,
            relevant_controls=relevant_controls,
            relevance_by_control=relevance_by_control,
            context=context,
        )
        requirement_result = self.aggregation.build_requirement_result(
            framework=framework,
            requirement=requirement,
            control_results=matched_control_results,
            skipped_control_results=skipped_control_results,
            degraded=degraded,
        )
        return self._single_requirement_payload(
            framework,
            requirement,
            requirement_result,
            matched_control_results,
            skipped_control_results,
        )

    async def _review_requirement_hybrid(
        self,
        *,
        framework: Framework,
        requirement: Requirement,
        context: PolicyCheckRuntimeContext,
        relevance_only: bool,
    ) -> Dict[str, Any]:
        file_ids = [resource_file.fileId for resource_file in context.repository.files]
        state = await self._prepare_requirement_review_state(requirement=requirement, context=context)

        relevance_by_file_control: Dict[str, Dict[str, ControlRelevanceRecord]] = {file_id: {} for file_id in file_ids}
        for control in requirement.controls:
            for file_id in file_ids:
                if file_id in state.requirement_hard_skipped_file_ids:
                    continue
                record = await self.relevance.evaluate_control_relevance(
                    requirement=requirement,
                    control=control,
                    retriever=context.retriever,
                    document_profile=context.document_profiles[file_id],
                    relevance_threshold=context.relevance_threshold,
                    requirement_profile=state.requirement_profile,
                    control_profile=state.control_profiles[control.controlId],
                    enriched_request=state.enriched_requests[control.controlId],
                    effective_model_id=context.model_id,
                )
                relevance_by_file_control[file_id][control.controlId] = record

        control_candidate_files: Dict[str, Set[str]] = defaultdict(set)
        control_global_meta: Dict[str, ControlRelevanceRecord] = {}
        relevant_controls: List[Any] = []
        for control in requirement.controls:
            control_id = control.controlId
            candidates: List[Dict[str, Any]] = []
            for file_id in file_ids:
                meta = relevance_by_file_control[file_id].get(control_id)
                if meta is None or not meta.is_relevant:
                    continue
                candidates.append(
                    {
                        "file_id": file_id,
                        "record": meta,
                        "decision_score": float(meta.decision_score or 0.0),
                        "retrieval_strength": float((meta.feature_scores or {}).get("retrieval_strength", 0.0)),
                        "decision_confidence": float(meta.decision_confidence or 0.0),
                    }
                )
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    float(item.get("decision_score", 0.0)),
                    float(item.get("retrieval_strength", 0.0)),
                    float(item.get("decision_confidence", 0.0)),
                ),
                reverse=True,
            )
            selected = self.projection.select_candidate_files_for_control(
                candidates=candidates,
                control_profile=state.control_profiles.get(control_id),
                requirement_profile=state.requirement_profile,
            )
            if not selected:
                continue
            control_candidate_files[control_id] = {item["file_id"] for item in selected}
            best = selected[0]["record"]
            global_reason = str(best.reason or "规则相关性已评估。")
            if len(control_candidate_files[control_id]) > 1:
                global_reason = f"{global_reason}; 候选文件数={len(control_candidate_files[control_id])}"
            control_global_meta[control_id] = replace(best, reason=global_reason)
            relevant_controls.append(control)

        per_file_payload: Dict[str, Dict[str, Any]] = {
            file_id: {"result": None, "matched_controls": [], "skipped_controls": []}
            for file_id in file_ids
        }
        if relevance_only:
            for file_id in file_ids:
                skipped = self.aggregation.build_requirement_level_skipped_controls(
                    requirement=requirement,
                    requirement_decision=state.requirement_decisions_by_file[file_id],
                    requirement_llm_override=state.requirement_llm_override_by_file[file_id],
                    relevance_by_control=relevance_by_file_control[file_id],
                )
                matched = [
                    self._build_relevance_only_result(requirement.requirementId, control.controlId, meta)
                    for control in requirement.controls
                    if (meta := relevance_by_file_control[file_id].get(control.controlId)) is not None and meta.is_relevant
                ]
                requirement_result = self.aggregation.build_requirement_result(
                    framework=framework,
                    requirement=requirement,
                    control_results=matched,
                    skipped_control_results=skipped,
                    degraded=False,
                    reason_override="已识别到相关规则，但因仅启用相关性判断模式，已跳过正式审查。",
                    result_override="skipped",
                )
                per_file_payload[file_id] = {
                    "result": requirement_result,
                    "matched_controls": matched,
                    "skipped_controls": skipped,
                }
            return {"order": self.aggregation.requirement_order_key(framework, requirement), "per_file": per_file_payload}

        relevant_controls, execution_dropped_ids = self.relevance.prune_controls_for_execution(
            relevant_controls=relevant_controls,
            relevance_by_control=control_global_meta,
        )
        if not relevant_controls:
            for file_id in file_ids:
                skipped = self.aggregation.build_requirement_level_skipped_controls(
                    requirement=requirement,
                    requirement_decision=state.requirement_decisions_by_file[file_id],
                    requirement_llm_override=state.requirement_llm_override_by_file[file_id],
                    relevance_by_control=relevance_by_file_control[file_id],
                )
                self._append_execution_dropped_skips(
                    requirement=requirement,
                    execution_dropped_ids=execution_dropped_ids,
                    relevance_by_control=relevance_by_file_control[file_id],
                    skipped_controls=skipped,
                )
                requirement_result = self.aggregation.build_requirement_result(
                    framework=framework,
                    requirement=requirement,
                    control_results=[],
                    skipped_control_results=skipped,
                    degraded=False,
                    reason_override="该要求下未识别到与当前单文档明确相关的规则。",
                    result_override="skipped",
                )
                per_file_payload[file_id] = {
                    "result": requirement_result,
                    "matched_controls": [],
                    "skipped_controls": skipped,
                }
            return {"order": self.aggregation.requirement_order_key(framework, requirement), "per_file": per_file_payload}

        executed_results, degraded = await self._run_requirement_execution(
            framework=framework,
            requirement=requirement,
            relevant_controls=relevant_controls,
            relevance_by_control=control_global_meta,
            context=context,
        )
        executed_by_control = {item.control_id: item for item in executed_results}
        for file_id in file_ids:
            skipped = self.aggregation.build_requirement_level_skipped_controls(
                requirement=requirement,
                requirement_decision=state.requirement_decisions_by_file[file_id],
                requirement_llm_override=state.requirement_llm_override_by_file[file_id],
                relevance_by_control=relevance_by_file_control[file_id],
            )
            matched: List[SingleFilePolicyControlResult] = []
            if file_id not in state.requirement_hard_skipped_file_ids:
                for control in relevant_controls:
                    control_id = control.controlId
                    allowed_file_ids = control_candidate_files.get(control_id, set())
                    file_meta = relevance_by_file_control[file_id].get(control_id)
                    if allowed_file_ids and file_id not in allowed_file_ids:
                        if file_meta is None or not self.projection.has_projection_signal(file_id=file_id, fallback_meta=file_meta):
                            continue
                    if file_meta is None:
                        continue
                    signal = self.projection.has_projection_signal(file_id=file_id, fallback_meta=file_meta)
                    if not file_meta.is_relevant and not signal:
                        continue
                    projected = self.projection.project_global_control_result_to_file(
                        file_id=file_id,
                        global_result=executed_by_control.get(control_id),
                        fallback_meta=file_meta,
                    )
                    if projected.result == "skipped":
                        skipped.append(projected)
                    else:
                        matched.append(projected)
                self._append_execution_dropped_skips(
                    requirement=requirement,
                    execution_dropped_ids=execution_dropped_ids,
                    relevance_by_control=relevance_by_file_control[file_id],
                    skipped_controls=skipped,
                )

            matched_ids = {item.control_id for item in matched}
            if matched_ids:
                skipped = [item for item in skipped if item.control_id not in matched_ids]

            requirement_result = self.aggregation.build_requirement_result(
                framework=framework,
                requirement=requirement,
                control_results=matched,
                skipped_control_results=skipped,
                degraded=degraded,
            )
            per_file_payload[file_id] = {
                "result": requirement_result,
                "matched_controls": matched,
                "skipped_controls": skipped,
            }
        return {"order": self.aggregation.requirement_order_key(framework, requirement), "per_file": per_file_payload}

    async def _prepare_requirement_review_state(
        self,
        *,
        requirement: Requirement,
        context: PolicyCheckRuntimeContext,
    ) -> RequirementReviewState:
        return await self.relevance.build_requirement_state(
            requirement=requirement,
            retriever=context.retriever,
            document_profiles=context.document_profiles,
            effective_model_id=context.model_id,
        )

    async def _run_requirement_execution(
        self,
        *,
        framework: Framework,
        requirement: Requirement,
        relevant_controls: List[Any],
        relevance_by_control: Dict[str, ControlRelevanceRecord],
        context: PolicyCheckRuntimeContext,
    ) -> tuple[List[SingleFilePolicyControlResult], bool]:
        fast_path = self.execution.build_fast_path_control_results(
            requirement=requirement,
            relevant_controls=relevant_controls,
            relevance_by_control=relevance_by_control,
        )
        if fast_path is not None:
            return fast_path, False
        outcome: ExecutionOutcome = await self.execution.execute_relevant_controls(
            framework=framework,
            requirement=requirement,
            relevant_controls=relevant_controls,
            relevance_by_control=relevance_by_control,
            repository=context.repository,
            execution_semaphore=context.execution_semaphore,
            effective_model_id=context.model_id,
        )
        return outcome.control_results, outcome.degraded

    def _build_control_skip_result(
        self,
        requirement_id: str,
        control_id: str,
        record: ControlRelevanceRecord,
    ) -> SingleFilePolicyControlResult:
        return self.aggregation.build_skipped_control_result(
            requirement_id=requirement_id,
            control_id=control_id,
            reason=record.skip_reason or UNRELATED_CONTROL_REASON,
            is_relevant=False,
            relevance_reason=record.reason,
            retrieval_stats=record.stats,
            reason_code=record.reason_code,
            decision_stage=record.decision_stage,
            decision_confidence=record.decision_confidence,
            feature_scores=record.feature_scores,
            llm_override_used=record.llm_override_used,
            execution_gate_passed=record.execution_gate_passed,
        )

    def _build_relevance_only_result(
        self,
        requirement_id: str,
        control_id: str,
        record: ControlRelevanceRecord,
    ) -> SingleFilePolicyControlResult:
        return self.aggregation.build_skipped_control_result(
            requirement_id=requirement_id,
            control_id=control_id,
            reason=RELEVANCE_ONLY_REASON,
            is_relevant=True,
            relevance_reason=record.reason,
            retrieval_stats=record.stats,
            reason_code=record.reason_code,
            decision_stage=record.decision_stage or "control",
            decision_confidence=record.decision_confidence,
            feature_scores=record.feature_scores,
            llm_override_used=record.llm_override_used,
            execution_gate_passed=record.execution_gate_passed,
        )

    def _append_execution_dropped_skips(
        self,
        *,
        requirement: Requirement,
        execution_dropped_ids: Set[str],
        relevance_by_control: Dict[str, ControlRelevanceRecord],
        skipped_controls: List[SingleFilePolicyControlResult],
    ) -> None:
        for control in requirement.controls:
            if control.controlId not in execution_dropped_ids:
                continue
            control_meta = relevance_by_control.get(control.controlId)
            if control_meta is None or not control_meta.is_relevant:
                continue
            skipped_controls.append(
                self.aggregation.build_skipped_control_result(
                    requirement_id=requirement.requirementId,
                    control_id=control.controlId,
                    reason="相关性处于灰区且执行优先级较低，已跳过正式检查。",
                    is_relevant=False,
                    relevance_reason=str(control_meta.reason or ""),
                    retrieval_stats=control_meta.stats or self.aggregation.empty_retrieval_stats(),
                    reason_code=str(control_meta.reason_code or ""),
                    decision_stage=str(control_meta.decision_stage or "control"),
                    decision_confidence=float(control_meta.decision_confidence or 0.0),
                    feature_scores=dict(control_meta.feature_scores or {}),
                    llm_override_used=bool(control_meta.llm_override_used),
                    execution_gate_passed=bool(control_meta.execution_gate_passed),
                )
            )

    def _single_requirement_payload(
        self,
        framework: Framework,
        requirement: Requirement,
        result: SingleFilePolicyRequirementResult,
        matched_controls: List[SingleFilePolicyControlResult],
        skipped_controls: List[SingleFilePolicyControlResult],
    ) -> Dict[str, Any]:
        return {
            "order": self.aggregation.requirement_order_key(framework, requirement),
            "result": result,
            "matched_controls": matched_controls,
            "skipped_controls": skipped_controls,
        }

    def _get_policy_agent(self, model_id: str) -> Any:
        normalized_model_id = (model_id or self.model_id).strip()
        if normalized_model_id not in self._policy_agents_by_model:
            self._policy_agents_by_model[normalized_model_id] = PolicyCheckAgent(model_id=normalized_model_id)
        return self._policy_agents_by_model[normalized_model_id]

    @staticmethod
    def build_resource_file_from_input(input_file: SingleFilePolicyInputFile) -> ResourceFile:
        return build_resource_file_from_input(input_file)

    @staticmethod
    def _build_default_file_id(file_name: Optional[str], doc_content: List[Dict[str, Any]]) -> str:
        return build_default_file_id(file_name, doc_content)

    @staticmethod
    def _build_single_file_repository(resource_file: ResourceFile) -> ResourceRepository:
        return build_single_file_repository(resource_file)

    @staticmethod
    def _build_multi_file_repository(resource_files: List[ResourceFile]) -> ResourceRepository:
        return build_multi_file_repository(resource_files)

    @staticmethod
    def _validate_max_concurrency(value: Any) -> int:
        return validate_max_concurrency(value)

    def _infer_doc_types(self, text: str) -> set[str]:
        raise NotImplementedError("Deprecated wrapper removed: document typing now relies on profile builders.")

    @staticmethod
    def _profile_anchor_overlap_count(control_text: str, document_profile: Dict[str, Any]) -> int:
        raise NotImplementedError("Deprecated wrapper removed: overlap logic moved to relevance matcher.")

    def _should_skip_by_doc_type_gate(
        self,
        *,
        control_text: str,
        document_profile: Dict[str, Any],
        preferred_doc_types: Optional[List[str]] = None,
    ) -> bool:
        raise NotImplementedError("Deprecated wrapper removed: doc-type gate logic moved to relevance matcher.")

    def _build_control_match_terms(self, *, control: Any, enriched_request: Any) -> List[str]:
        raise NotImplementedError("Deprecated wrapper removed: control matching terms are built in profile modules.")

    @staticmethod
    def _count_term_hits(terms: List[str], text: str) -> int:
        raise NotImplementedError("Deprecated wrapper removed: term-hit counting is no longer used.")

    def _count_evidence_term_hits(self, *, retrieval: Any, match_terms: List[str]) -> int:
        raise NotImplementedError("Deprecated wrapper removed: evidence term-hit counting is no longer used.")

    def _build_requirement_match_terms(self, requirement: Requirement) -> List[str]:
        raise NotImplementedError("Deprecated wrapper removed: requirement terms are built in profile modules.")

    def _evaluate_requirement_scope(self, *, requirement: Requirement, document_profile: Dict[str, Any]) -> tuple[bool, str]:
        raise NotImplementedError("Deprecated wrapper removed: requirement scope gate is handled by matcher.")

    @staticmethod
    def _profile_overlap_score(text: str, *, document_profile: Dict[str, Any]) -> int:
        raise NotImplementedError("Deprecated wrapper removed: overlap scoring moved to matcher.")

    @staticmethod
    def _is_collection_heavy_requirement(requirement: Requirement) -> bool:
        raise NotImplementedError("Deprecated wrapper removed: collection-heavy inference moved to profile builder.")

    def _should_fast_fail_for_missing_evidence(
        self,
        *,
        control: Any,
        requirement: Requirement,
        retrieval: Any,
        enriched_request: Any,
        document_profile: Dict[str, Any],
        match_terms: List[str],
        evidence_term_hits: int,
    ) -> bool:
        raise NotImplementedError("Deprecated wrapper removed: fast-fail decision moved to relevance engine.")
