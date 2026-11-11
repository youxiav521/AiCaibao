"""
Policy Check Agent - 主智能体实现

使用 LangChain create_agent 和中间件实现多智能体架构
支持并发执行多个 requirement
"""
import asyncio
import json
import time
import traceback
from uuid import uuid4
from typing import Dict, Any, List, Annotated, AsyncGenerator, Tuple, Optional
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolCallId

from src.base.langfuse_observe import trace_span
from src.utils.model_factory import ModelFactory
from ..schemas import ResourceRepository, Framework, Requirement
from ..schemas.control_models import ControlRetrievalRequest
from ..services import MaterialRetriever, build_material_store
from ..material_layer import MaterialEvidenceRetriever, MaterialUnderstandingService
from .tools import (
    ResourceRepositoryContext,
    read_file_tool,
    read_file_excerpt_tool,
    read_evidence_blocks_tool,
    find_files_tool,
    retrieve_material_evidence_tool,
    list_files_tool,
    read_image_tool,
    report_control_check,
    complete_check,
    get_runtime_snapshot,
    set_runtime_event_queue,
    set_runtime_metadata,
    merge_runtime_material_metrics,
    bump_runtime_tool_call,
)
from .subagents import (
    create_document_checker_subagent,
    create_multimodal_checker_subagent,
)
from ..utils import get_system_prompt
from ..middlewares.invoke_model_middleware import InvokeModelMiddleware
from loguru import logger


class PolicyCheckAgent:
    """策略检查智能体 - 创建并执行策略合规检查"""

    def __init__(self, model_id: str = "model_bailian_policy_check"):
        self.model_id = model_id
        self.requirement_retry_limit = 2
        self.requirement_retry_backoff_seconds = 1.5
        self.requirement_timeout_seconds = 180
        self.requirement_timeout_floor_seconds = 75
        self.requirement_timeout_ceiling_seconds = 240
        self.requirement_timeout_base_seconds = 45
        self.requirement_timeout_per_control_seconds = 20
        self.requirement_soft_timeout_buffer_seconds = 20
        self.split_control_threshold = 8
        self.split_collection_threshold = 4
        self.split_multimodal_threshold = 3
        self.split_high_risk_threshold = 5
        self.split_prompt_char_threshold = 6000
        self.split_run_max_concurrency = 3
        self._material_stores_by_repo: Dict[str, Optional[Any]] = {}
        self._material_retrievers_by_repo: Dict[str, Optional[MaterialRetriever]] = {}
        self._material_understanding_by_repo: Dict[str, Optional[Any]] = {}
        self._material_retriever_v1_by_repo: Dict[str, Optional[MaterialEvidenceRetriever]] = {}
        self._missing_files_by_repo: Dict[str, set[str]] = {}
        self._missing_images_by_repo: Dict[str, set[str]] = {}
        self.model = ModelFactory.get_model(key=model_id)
        self.middlewares = self._create_middlewares()
        self.tools = self._create_tools()
        self.system_prompt = get_system_prompt()
        self.agent = self._create_agent()

    def _create_middlewares(self) -> List:
        """创建全局中间件"""
        global_tool_limiter = ToolCallLimitMiddleware(thread_limit=120, run_limit=120)
        # 重试和token limit中间件
        invoke_model_middleware = InvokeModelMiddleware(max_retries=5, model_id=self.model_id)
        summarization_middleware = SummarizationMiddleware(
            model=self.model,
            max_tokens_before_summary=200000,
            messages_to_keep=4
        )

        return [
            invoke_model_middleware,
            global_tool_limiter,
            summarization_middleware
        ]

    @staticmethod
    def _normalize_subagent_payload(raw_content: Any, payload_type: str) -> str:
        if isinstance(raw_content, dict):
            return json.dumps(raw_content, ensure_ascii=False)

        content = str(raw_content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        fallback_payload = {
            "result": "uncertain",
            "reason": f"{payload_type} 返回了非 JSON 内容，已按降级结果处理。",
            "rawContent": str(raw_content or ""),
        }
        return json.dumps(fallback_payload, ensure_ascii=False)

    def _create_tools(self) -> List:
        base_tools = [retrieve_material_evidence_tool, read_evidence_blocks_tool, read_file_tool, read_file_excerpt_tool, find_files_tool, list_files_tool, read_image_tool]
        file_tools = [retrieve_material_evidence_tool, read_evidence_blocks_tool, read_file_tool, read_file_excerpt_tool, find_files_tool, list_files_tool]
        image_tools = [retrieve_material_evidence_tool, read_evidence_blocks_tool, read_image_tool, find_files_tool, list_files_tool]
        document_checker = create_document_checker_subagent(self.model, file_tools, self.middlewares)
        multimodal_checker = create_multimodal_checker_subagent(self.model, image_tools, self.middlewares)

        @tool("check_document_content", description="专注执行规则检查的 subagent tool，检查文档内容，执行文本规则检查")
        async def check_document_content(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
            logger.info("check_document_content: {}", query)
            @trace_span(name="document_checker.ainvoke")
            async def invoke_document_checker():
                return await document_checker.ainvoke(
                    {"messages": [HumanMessage(content=query)]}
                )
            response = await invoke_document_checker()
            return self._normalize_subagent_payload(
                response["messages"][-1].content,
                payload_type="document_checker",
            )

        @tool(
            "check_multimodal_content", 
            description="专门处理图片内容的subagent tool，进行检查过程中需要判定图片内容时，使用此工具执行多模态规则检查"
        )
        async def check_multimodal_content(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
            logger.info("check_multimodal_content: {}", query)
            @trace_span(name="multimodal_checker.ainvoke")
            async def invoke_multimodal_checker():
                return await multimodal_checker.ainvoke(
                    {"messages": [HumanMessage(content=query)]}
                )
            response = await invoke_multimodal_checker()
            return self._normalize_subagent_payload(
                response["messages"][-1].content,
                payload_type="multimodal_checker",
            )

        return [
            retrieve_material_evidence_tool,
            read_evidence_blocks_tool,
            find_files_tool,
            read_file_excerpt_tool,
            list_files_tool, 
            check_document_content, 
            check_multimodal_content, 
            report_control_check,
            complete_check
        ]

    def _create_agent(self):
        return create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            middleware=[*self.middlewares]
        )

    def _get_material_retriever(self, resource_repository: ResourceRepository) -> Optional[MaterialRetriever]:
        repo_key = getattr(resource_repository, "id", "") or "__default__"
        if repo_key in self._material_retrievers_by_repo:
            return self._material_retrievers_by_repo[repo_key]
        try:
            store = self._get_material_store(resource_repository)
            if store is None:
                retriever = None
            else:
                retriever = MaterialRetriever(store)
        except Exception as exc:
            logger.warning("build material retriever failed for repo {}: {}", repo_key, exc)
            retriever = None
        self._material_retrievers_by_repo[repo_key] = retriever
        return retriever

    def _get_material_store(self, resource_repository: ResourceRepository) -> Optional[Any]:
        repo_key = getattr(resource_repository, "id", "") or "__default__"
        if repo_key in self._material_stores_by_repo:
            return self._material_stores_by_repo[repo_key]
        try:
            store = build_material_store(resource_repository, case_id=repo_key)
        except Exception as exc:
            logger.warning("build material store failed for repo {}: {}", repo_key, exc)
            store = None
        self._material_stores_by_repo[repo_key] = store
        return store

    def _get_material_understanding(self, resource_repository: ResourceRepository) -> Optional[Any]:
        repo_key = getattr(resource_repository, "id", "") or "__default__"
        if repo_key in self._material_understanding_by_repo:
            return self._material_understanding_by_repo[repo_key]
        try:
            understanding = MaterialUnderstandingService().build(resource_repository)
        except Exception as exc:
            logger.warning("build material understanding failed for repo {}: {}", repo_key, exc)
            understanding = None
        self._material_understanding_by_repo[repo_key] = understanding
        return understanding

    def _get_material_retriever_v1(self, resource_repository: ResourceRepository) -> Optional[MaterialEvidenceRetriever]:
        repo_key = getattr(resource_repository, "id", "") or "__default__"
        if repo_key in self._material_retriever_v1_by_repo:
            return self._material_retriever_v1_by_repo[repo_key]
        understanding = self._get_material_understanding(resource_repository)
        retriever = MaterialEvidenceRetriever(understanding) if understanding is not None else None
        self._material_retriever_v1_by_repo[repo_key] = retriever
        return retriever

    def _build_runtime_overrides(self, resource_repository: ResourceRepository) -> Dict[str, Any]:
        runtime_overrides: Dict[str, Any] = {}
        repo_key = getattr(resource_repository, "id", "") or "__default__"
        store = self._get_material_store(resource_repository)
        if store is not None:
            runtime_overrides["material_store"] = store
        retriever_v2 = self._get_material_retriever(resource_repository)
        if retriever_v2 is not None:
            runtime_overrides["material_retriever_v2"] = retriever_v2
        understanding = self._get_material_understanding(resource_repository)
        if understanding is not None:
            runtime_overrides["material_understanding"] = understanding
        retriever_v1 = self._get_material_retriever_v1(resource_repository)
        if retriever_v1 is not None:
            runtime_overrides["material_retriever"] = retriever_v1
        runtime_overrides["missing_files"] = set(self._missing_files_by_repo.get(repo_key, set()))
        runtime_overrides["missing_images"] = set(self._missing_images_by_repo.get(repo_key, set()))
        return runtime_overrides

    def _merge_missing_artifact_cache(self, resource_repository: ResourceRepository, runtime_snapshot: Dict[str, Any]) -> None:
        repo_key = getattr(resource_repository, "id", "") or "__default__"
        missing_files = {str(item) for item in (runtime_snapshot.get("missing_files") or []) if str(item).strip()}
        missing_images = {str(item) for item in (runtime_snapshot.get("missing_images") or []) if str(item).strip()}
        if missing_files:
            cached_files = self._missing_files_by_repo.setdefault(repo_key, set())
            cached_files.update(missing_files)
        if missing_images:
            cached_images = self._missing_images_by_repo.setdefault(repo_key, set())
            cached_images.update(missing_images)

    def _build_requirement_prefetch(
        self,
        resource_repository: ResourceRepository,
        requirement: Requirement,
    ) -> Dict[str, Any]:
        retriever = self._get_material_retriever(resource_repository)
        if retriever is None:
            return {"text": "", "evidence_ids": [], "evidence_by_file": {}}

        sections: List[str] = []
        evidence_ids: List[str] = []
        evidence_by_file: Dict[str, List[str]] = {}
        prefetch_metrics = {"calls": 0, "doc_hits": 0, "chunk_hits": 0, "empty_calls": 0}
        controls = requirement.controls or []
        for control in controls[: min(len(controls), 4)]:
            retrieval = retriever.retrieve_for_control(
                ControlRetrievalRequest(
                    control_id=control.controlId,
                    requirement_id=requirement.requirementId,
                    control_text=self._control_text(control),
                    top_k_documents=3,
                    top_k_blocks=5,
                    top_k_facts=4,
                )
            )
            prefetch_metrics["calls"] += 1
            prefetch_metrics["doc_hits"] += len(retrieval.candidate_documents)
            prefetch_metrics["chunk_hits"] += len(retrieval.evidence_hits)
            if not retrieval.candidate_documents and not retrieval.evidence_hits and not retrieval.fact_hits:
                prefetch_metrics["empty_calls"] += 1
            section_lines = [f"- controlId={control.controlId}"]
            if retrieval.candidate_documents:
                section_lines.append(
                    "  candidate_docs="
                    + str([hit.manifest.file_name for hit in retrieval.candidate_documents[:3]])
                )
            if retrieval.evidence_hits:
                section_lines.append("  evidence_blocks:")
                for hit in retrieval.evidence_hits[:4]:
                    evidence_ids.append(hit.block.evidence_id)
                    evidence_by_file.setdefault(hit.block.file_id, []).append(hit.block.evidence_id)
                    section_lines.append(
                        f"    {hit.block.evidence_id} file={hit.block.file_name} page={hit.block.page} text={hit.block.text[:180]}"
                    )
            if retrieval.fact_hits:
                section_lines.append(
                    "  fact_hits="
                    + str(
                        [
                            f"{hit.fact.field_name}={hit.fact.field_value} evidence={hit.fact.evidence_id}"
                            for hit in retrieval.fact_hits[:3]
                        ]
                    )
                )
            sections.append("\n".join(section_lines))

        deduped_ids: List[str] = []
        seen_ids = set()
        for evidence_id in evidence_ids:
            if evidence_id in seen_ids:
                continue
            seen_ids.add(evidence_id)
            deduped_ids.append(evidence_id)

        return {
            "text": "\n".join(sections).strip(),
            "evidence_ids": deduped_ids,
            "evidence_by_file": evidence_by_file,
            "metrics": prefetch_metrics,
        }

    def _format_requirement(
        self,
        resource_repository: ResourceRepository,
        requirement: Requirement,
        prefetched_context: str = "",
    ) -> str:
        controls = requirement.controls or []
        controls_prompt = "\n".join([
            f"{i}. 控制项ID: {c.controlId}\n   - 类型: {c.controlType}, 风险: {c.risk}\n   - 逻辑: {c.logicDescription}\n   - 原文: {c.sourceText}"
            for i, c in enumerate(controls, 1)
        ]) if controls else "无控制项"

        prompt = f"""请执行以下政策规则检查：

## 资源库信息
- 标题: {resource_repository.title}
- 描述: {resource_repository.description}
- 文件列表: {', '.join([f.fileName for f in resource_repository.files])}

## 当前检查的需求
- 需求ID: {requirement.requirementId}
- 需求标题: {requirement.title}
- 需求原文: {requirement.sourceText}
- 需求检查说明: {requirement.description}

## 控制项列表
{controls_prompt}
"""
        if prefetched_context:
            prompt += (
                "\n## 预取证据\n"
                f"{prefetched_context}\n"
                "如果已经有 evidenceId，优先使用 read_evidence_blocks_tool，不要再反复调用 read_file_excerpt_tool。\n"
            )
        prompt += "\n请按照工作流程依次执行检查任务，检查上述需求相关的规则。"
        return prompt

    @staticmethod
    def _control_text(control: Any) -> str:
        return " ".join(
            str(value or "")
            for value in [
                getattr(control, "controlType", ""),
                getattr(control, "risk", ""),
                getattr(control, "logicDescription", ""),
                getattr(control, "sourceText", ""),
            ]
        )

    def _control_is_collection_like(self, control: Any) -> bool:
        text = self._control_text(control)
        keywords = [
            "是否存在", "是否齐全", "是否提供", "收集", "提供", "提交", "具备",
            "文件", "资料", "文书", "凭证", "证明", "协议", "决议", "授权书",
            "证书", "报告", "登记", "原件", "复印件", "查询记录",
        ]
        return any(keyword in text for keyword in keywords)

    def _control_is_multimodal(self, control: Any) -> bool:
        text = self._control_text(control)
        if str(getattr(control, "controlType", "")).lower() in {"image", "multimodal"}:
            return True
        keywords = ["图片", "影像", "签字", "签名", "盖章", "骑缝", "手写", "书写", "图像", "扫描件"]
        return any(keyword in text for keyword in keywords)

    def _control_is_high_risk(self, control: Any) -> bool:
        return str(getattr(control, "risk", "")).lower() == "high"

    def _control_is_explicit_document_presence_check(self, control: Any) -> bool:
        text = self._control_text(control)
        keywords = [
            "申请书", "合同", "协议", "决议", "证明", "凭证", "报告", "授权书", "证书",
            "登记证明", "批复", "批文", "收据", "担保函", "借据",
        ]
        return self._control_is_collection_like(control) and any(keyword in text for keyword in keywords)

    def _requirement_timeout_budget(self, requirement: Requirement, should_split: bool = False) -> int:
        control_count = max(1, len(requirement.controls or []))
        budget = self.requirement_timeout_base_seconds + self.requirement_timeout_per_control_seconds * control_count
        if should_split:
            budget += self.requirement_soft_timeout_buffer_seconds
        budget = max(self.requirement_timeout_floor_seconds, budget)
        budget = min(self.requirement_timeout_ceiling_seconds, budget)
        return int(budget)

    def _split_concurrency_for_requirement(self, requirement: Requirement) -> int:
        control_count = len(requirement.controls or [])
        if control_count <= 3:
            return 1
        if control_count <= 6:
            return 2
        return self.split_run_max_concurrency

    def _control_timeout_budget(
        self,
        control: Any,
        requirement: Requirement,
        split_soft_timeout_seconds: int,
    ) -> int:
        control_count = max(1, len(requirement.controls or []))
        # Based on the previous full run, 4-6 control document-heavy requirements were
        # spending roughly 27-41s per control before converging or timing out.
        baseline = split_soft_timeout_seconds / control_count
        budget = max(28, min(50, int(round(baseline))))
        if self._control_is_collection_like(control):
            budget += 4
        if self._control_is_high_risk(control):
            budget += 5
        if self._control_is_multimodal(control):
            budget += 8
        return min(55, budget)

    def _fast_path_control_result(
        self,
        resource_repository: ResourceRepository,
        requirement: Requirement,
        framework_id: str,
        control: Any,
    ) -> Optional[Dict[str, Any]]:
        if not self._control_is_explicit_document_presence_check(control):
            return None
        retriever = self._get_material_retriever(resource_repository)
        if retriever is None:
            return None

        retrieval = retriever.retrieve_for_control(
            ControlRetrievalRequest(
                control_id=control.controlId,
                requirement_id=requirement.requirementId,
                control_text=self._control_text(control),
                top_k_documents=3,
                top_k_blocks=4,
                top_k_facts=3,
            )
        )
        strongest_doc_score = max((hit.score for hit in retrieval.candidate_documents), default=0.0)
        has_strong_candidate = (
            bool(retrieval.evidence_hits)
            or bool(retrieval.fact_hits)
            or strongest_doc_score >= 4.5
        )
        if has_strong_candidate:
            return None

        result = "uncertain" if self._control_is_multimodal(control) else "fail"
        reason = (
            "Deterministic retrieval found no candidate documents, evidence blocks, or facts for this required document control. "
            "The expected material appears to be missing from the repository."
        )
        return {
            "control_results": [
                {
                    "requirementId": requirement.requirementId,
                    "controlId": control.controlId,
                    "result": result,
                    "reason": reason,
                    "evidence": [],
                    "degraded": False,
                }
            ],
            "runtime_snapshot": {
                "tool_call_counts": {"agent_fast_path_control": 1},
                "material_retrieval_metrics": {
                    "calls": 1,
                    "doc_hits": len(retrieval.candidate_documents),
                    "chunk_hits": len(retrieval.evidence_hits),
                    "empty_calls": 1 if not retrieval.candidate_documents and not retrieval.evidence_hits and not retrieval.fact_hits else 0,
                },
            },
        }

    def _build_budget_fallback_control_result(
        self,
        resource_repository: ResourceRepository,
        requirement: Requirement,
        framework_id: str,
        control: Any,
        *,
        reason: str,
    ) -> Dict[str, Any]:
        fast_path = self._fast_path_control_result(
            resource_repository=resource_repository,
            requirement=requirement,
            framework_id=framework_id,
            control=control,
        )
        if fast_path is not None:
            control_result = dict(fast_path["control_results"][0])
            if reason and reason not in (control_result.get("reason") or ""):
                control_result["reason"] = f"{control_result.get('reason', '')} {reason}".strip()
            control_result["degraded"] = True
            runtime_snapshot = fast_path.get("runtime_snapshot", {})
            return {
                "control_results": [control_result],
                "runtime_snapshot": runtime_snapshot,
            }
        return {
            "control_results": [
                {
                    "requirementId": requirement.requirementId,
                    "controlId": control.controlId,
                    "result": "uncertain",
                    "reason": reason,
                    "evidence": [],
                    "degraded": True,
                }
            ],
            "runtime_snapshot": {
                "tool_call_counts": {"agent_control_budget_fallback": 1},
                "material_retrieval_metrics": {"calls": 0, "doc_hits": 0, "chunk_hits": 0, "empty_calls": 0},
            },
        }

    def _split_strategy(self, resource_repository: ResourceRepository, requirement: Requirement) -> Tuple[bool, List[str]]:
        controls = requirement.controls or []
        reasons: List[str] = []
        control_count = len(controls)
        prompt_chars = len(self._format_requirement(resource_repository, requirement))
        collection_like_count = sum(1 for control in controls if self._control_is_collection_like(control))
        multimodal_count = sum(1 for control in controls if self._control_is_multimodal(control))
        high_risk_count = sum(1 for control in controls if self._control_is_high_risk(control))

        if control_count >= self.split_control_threshold:
            reasons.append(f"control_count>={self.split_control_threshold}")
        if prompt_chars > self.split_prompt_char_threshold:
            reasons.append(f"prompt_chars>{self.split_prompt_char_threshold}")
        if collection_like_count >= self.split_collection_threshold:
            reasons.append(f"collection_like_controls>={self.split_collection_threshold}")
        if multimodal_count >= self.split_multimodal_threshold:
            reasons.append(f"multimodal_controls>={self.split_multimodal_threshold}")
        if control_count >= 6 and high_risk_count >= self.split_high_risk_threshold:
            reasons.append(f"high_risk_controls>={self.split_high_risk_threshold}")
        if control_count >= 5 and collection_like_count >= max(3, control_count - 1):
            reasons.append("requirement_is_document_collection_heavy")

        return bool(reasons), reasons

    async def _invoke_requirement_once(
        self,
        resource_repository: ResourceRepository,
        requirement: Requirement,
        framework_id: str,
        event_queue: asyncio.Queue,
    ) -> Dict[str, Any]:
        with ResourceRepositoryContext(
            resource_repository,
            runtime_overrides=self._build_runtime_overrides(resource_repository),
        ):
            prefetched = self._build_requirement_prefetch(resource_repository, requirement)
            set_runtime_event_queue(event_queue)
            set_runtime_metadata(
                framework_id=framework_id,
                requirement_id=requirement.requirementId,
                requirement_title=requirement.title,
                prefetched_evidence_ids=prefetched.get("evidence_ids", []),
                prefetched_evidence_by_file=prefetched.get("evidence_by_file", {}),
                prefetched_evidence_text=prefetched.get("text", ""),
            )
            prefetch_metrics = prefetched.get("metrics", {}) or {}
            if prefetch_metrics.get("calls"):
                merge_runtime_material_metrics(
                    calls=prefetch_metrics.get("calls", 0),
                    doc_hits=prefetch_metrics.get("doc_hits", 0),
                    chunk_hits=prefetch_metrics.get("chunk_hits", 0),
                    empty_calls=prefetch_metrics.get("empty_calls", 0),
                )
                bump_runtime_tool_call("agent_prefetch_retrieval", prefetch_metrics.get("calls", 0))
            result = await self.agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=self._format_requirement(
                                resource_repository,
                                requirement,
                                prefetched_context=prefetched.get("text", ""),
                            )
                        )
                    ]
                }
            )
            runtime_snapshot = get_runtime_snapshot()
            self._merge_missing_artifact_cache(resource_repository, runtime_snapshot)
            return {
                "result": result,
                "check_results": runtime_snapshot.get("final_check_result") or result.get("check_results", {}),
                "control_results": runtime_snapshot.get("control_results", []),
                "runtime_snapshot": runtime_snapshot,
            }

    def _build_fallback_control_result(
        self,
        requirement: Requirement,
        framework_id: str,
        control_id: str,
        check_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "event": "control_result",
            "requirement_id": requirement.requirementId,
            "requirement_title": requirement.title,
            "framework_id": framework_id,
            "control_id": control_id,
            "result": check_results.get("result", "uncertain"),
            "reason": check_results.get("reason", "未产出规则级独立结果，已回退为要求级结果。"),
            "evidence": check_results.get("evidence", []),
            "fallback": True,
        }

    def _build_stream_control_event(
        self,
        requirement: Requirement,
        framework_id: str,
        control_result: Dict[str, Any],
        fallback: bool = False,
    ) -> Dict[str, Any]:
        return {
            "event": "control_result",
            "requirement_id": requirement.requirementId,
            "requirement_title": requirement.title,
            "framework_id": framework_id,
            "control_id": control_result.get("controlId"),
            "result": control_result.get("result", "uncertain"),
            "reason": control_result.get("reason", ""),
            "evidence": control_result.get("evidence", []),
            "degraded": bool(control_result.get("degraded")),
            "fallback": fallback,
        }

    def _build_degraded_requirement_result(
        self,
        requirement: Requirement,
        framework_id: str,
    ) -> Dict[str, Any]:
        control_results: List[Dict[str, Any]] = []
        for control in requirement.controls:
            control_results.append(
                {
                    "requirementId": requirement.requirementId,
                    "controlId": control.controlId,
                    "result": "uncertain",
                    "reason": "校验流程未能稳定完成，已使用自动降级兜底结果。",
                    "evidence": [],
                    "degraded": True,
                }
            )

        return {
            "result": {"messages": [], "todos": []},
            "check_results": {
                "requirementId": requirement.requirementId,
                "result": "uncertain",
                "reason": "校验流程未能稳定完成，已使用自动降级兜底结果。",
                "evidence": [],
                "controlResults": control_results,
            },
            "control_results": control_results,
            "framework_id": framework_id,
        }

    def _aggregate_requirement_result(
        self,
        requirement: Requirement,
        control_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
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
                        "summary": ev.get("summary"),
                        "quotedText": ev.get("quotedText"),
                    }
                )

        if not reason_parts:
            reason_parts.append("No valid control result was produced, so the requirement cannot be determined confidently.")
            final_result = "uncertain"

        return {
            "requirementId": requirement.requirementId,
            "result": final_result,
            "reason": "; ".join(reason_parts),
            "evidence": evidence,
            "controlResults": ordered_results,
        }

    @staticmethod
    def _merge_runtime_snapshots(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        tool_call_counts: Dict[str, int] = {}
        material_metrics = {
            "calls": 0,
            "doc_hits": 0,
            "chunk_hits": 0,
            "empty_calls": 0,
        }
        for snapshot in snapshots:
            for tool_name, count in (snapshot.get("tool_call_counts") or {}).items():
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + int(count or 0)
            metrics = snapshot.get("material_retrieval_metrics") or {}
            for key in material_metrics:
                material_metrics[key] += int(metrics.get(key) or 0)
        return {
            "tool_call_counts": tool_call_counts,
            "material_retrieval_metrics": material_metrics,
        }

    @staticmethod
    def _build_completion_summary(
        *,
        all_requirements: List[Dict[str, Any]],
        control_results_by_key: Dict[Tuple[str, str, str], Dict[str, Any]],
        duration_seconds: float,
    ) -> Dict[str, Any]:
        total_requirements = len(all_requirements)
        total_controls = len(control_results_by_key)
        status_counts = {
            "passed": 0,
            "failed": 0,
            "uncertain": 0,
            "skipped": 0,
        }
        for item in control_results_by_key.values():
            result = str(item.get("result") or "uncertain").lower()
            if result == "pass":
                status_counts["passed"] += 1
            elif result == "fail":
                status_counts["failed"] += 1
            elif result == "skipped":
                status_counts["skipped"] += 1
            else:
                status_counts["uncertain"] += 1
        pass_rate = 0.0 if total_controls <= 0 else (status_counts["passed"] / total_controls) * 100
        return {
            "totalRequirements": total_requirements,
            "totalControls": total_controls,
            "passed": status_counts["passed"],
            "failed": status_counts["failed"],
            "uncertain": status_counts["uncertain"],
            "skipped": status_counts["skipped"],
            "description": (
                f"共完成 {total_requirements} 条requirement、{total_controls} 条control，"
                f"通过 {status_counts['passed']} 条，失败 {status_counts['failed']} 条，"
                f"待确认 {status_counts['uncertain']} 条，跳过 {status_counts['skipped']} 条。"
            ),
            "durationSeconds": round(duration_seconds, 2),
            "passRate": f"{pass_rate:.1f}%",
        }

    async def _run_split_requirement(
        self,
        resource_repository: ResourceRepository,
        requirement: Requirement,
        framework_id: str,
        event_queue: asyncio.Queue,
        execution_semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        should_split, split_reasons = self._split_strategy(resource_repository, requirement)
        logger.info(
            "requirement {} split execution, controls={}, reasons={}",
            requirement.requirementId,
            len(requirement.controls),
            split_reasons,
        )
        aggregated_control_results: List[Dict[str, Any]] = []
        runtime_snapshots: List[Dict[str, Any]] = []
        split_limit = self._split_concurrency_for_requirement(requirement)
        semaphore = asyncio.Semaphore(split_limit)
        soft_timeout_seconds = max(
            self.requirement_timeout_floor_seconds,
            self._requirement_timeout_budget(requirement, should_split=True) - self.requirement_soft_timeout_buffer_seconds,
        )

        async def run_control(control: Any) -> Dict[str, Any]:
            fast_path = self._fast_path_control_result(
                resource_repository=resource_repository,
                requirement=requirement,
                framework_id=framework_id,
                control=control,
            )
            if fast_path is not None:
                return fast_path

            try:
                async with semaphore:
                    async with execution_semaphore:
                        split_requirement = requirement.model_copy(update={"controls": [control]})
                        invoke_result = await asyncio.wait_for(
                            self._invoke_requirement_once(
                                resource_repository=resource_repository,
                                requirement=split_requirement,
                                framework_id=framework_id,
                                event_queue=event_queue,
                            ),
                            timeout=self._control_timeout_budget(control, requirement, soft_timeout_seconds),
                        )
            except asyncio.TimeoutError:
                return self._build_budget_fallback_control_result(
                    resource_repository=resource_repository,
                    requirement=requirement,
                    framework_id=framework_id,
                    control=control,
                    reason="该控制项超过控制级时间预算，已自动降级。",
                )
            except Exception as exc:
                logger.warning(
                    "split control {} under requirement {} failed and will degrade: {}",
                    control.controlId,
                    requirement.requirementId,
                    str(exc),
                )
                return self._build_budget_fallback_control_result(
                    resource_repository=resource_repository,
                    requirement=requirement,
                    framework_id=framework_id,
                    control=control,
                    reason="该控制项执行失败，已自动降级。",
                )
            control_results = invoke_result["control_results"]
            if control_results:
                return {
                    "control_results": control_results,
                    "runtime_snapshot": invoke_result.get("runtime_snapshot", {}),
                }

            fallback = self._build_fallback_control_result(
                requirement=requirement,
                framework_id=framework_id,
                control_id=control.controlId,
                check_results=invoke_result["check_results"],
            )
            await event_queue.put(fallback)
            return {
                        "control_results": [
                            {
                                "requirementId": requirement.requirementId,
                                "controlId": control.controlId,
                                "result": fallback["result"],
                                "reason": fallback["reason"],
                                "evidence": fallback["evidence"],
                                "degraded": True,
                            }
                        ],
                "runtime_snapshot": invoke_result.get("runtime_snapshot", {}),
            }

        task_by_control_id = {
            control.controlId: asyncio.create_task(run_control(control))
            for control in requirement.controls
        }
        done, pending = await asyncio.wait(
            task_by_control_id.values(),
            timeout=soft_timeout_seconds,
        )
        for task in done:
            task_result = await task
            control_results = task_result["control_results"]
            runtime_snapshots.append(task_result.get("runtime_snapshot", {}))
            aggregated_control_results.extend(control_results)
            for control_result in control_results:
                await event_queue.put(
                    self._build_stream_control_event(
                        requirement=requirement,
                        framework_id=framework_id,
                        control_result=control_result,
                    )
                )

        if pending:
            logger.warning(
                "requirement {} split execution hit soft timeout={}s, pending_controls={}, split_limit={}",
                requirement.requirementId,
                soft_timeout_seconds,
                len(pending),
                split_limit,
            )
            completed_control_ids = {
                item.get("controlId")
                for item in aggregated_control_results
            }
            for control in requirement.controls:
                if control.controlId in completed_control_ids:
                    continue
                fallback = {
                    **self._build_budget_fallback_control_result(
                        resource_repository=resource_repository,
                        requirement=requirement,
                        framework_id=framework_id,
                        control=control,
                        reason="该控制项未在要求项时间预算内完成，已自动降级。",
                    )
                }
                fallback_control_result = fallback["control_results"][0]
                runtime_snapshots.append(fallback.get("runtime_snapshot", {}))
                aggregated_control_results.append(fallback_control_result)
                await event_queue.put(
                    self._build_stream_control_event(
                        requirement=requirement,
                        framework_id=framework_id,
                        control_result=fallback_control_result,
                        fallback=True,
                    )
                )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        return {
            "result": {"messages": [], "todos": []},
            "check_results": self._aggregate_requirement_result(requirement, aggregated_control_results),
            "control_results": aggregated_control_results,
            "runtime_snapshot": self._merge_runtime_snapshots(runtime_snapshots),
        }

    async def _execute_requirement_with_retry(
        self,
        resource_repository: ResourceRepository,
        requirement: Requirement,
        framework_id: str,
        event_queue: asyncio.Queue,
        execution_semaphore: asyncio.Semaphore,
        should_split: Optional[bool] = None,
        split_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        last_error: Exception | None = None
        if should_split is None or split_reasons is None:
            should_split, split_reasons = self._split_strategy(resource_repository, requirement)
        logger.info(
            "requirement {} split_decision={} reasons={}",
            requirement.requirementId,
            should_split,
            split_reasons,
        )
        timeout_budget_seconds = self._requirement_timeout_budget(requirement, should_split=should_split)
        for attempt in range(1, self.requirement_retry_limit + 2):
            try:
                if should_split:
                    return await self._run_split_requirement(
                        resource_repository=resource_repository,
                        requirement=requirement,
                        framework_id=framework_id,
                        event_queue=event_queue,
                        execution_semaphore=execution_semaphore,
                    )
                async with execution_semaphore:
                    return await asyncio.wait_for(
                        self._invoke_requirement_once(
                            resource_repository=resource_repository,
                            requirement=requirement,
                            framework_id=framework_id,
                            event_queue=event_queue,
                        ),
                        timeout=timeout_budget_seconds,
                    )
            except Exception as exc:
                last_error = exc
                if self._is_confirmed_missing_artifact_error(exc):
                    logger.warning(
                        "requirement {} hit confirmed missing artifact, skip retries and fail fast: {}",
                        requirement.requirementId,
                        str(exc),
                    )
                    return self._build_missing_artifact_requirement_result(
                        requirement=requirement,
                        framework_id=framework_id,
                        reason=str(exc),
                    )
                logger.warning(
                    "requirement {} attempt {}/{} failed: {}",
                    requirement.requirementId,
                    attempt,
                    self.requirement_retry_limit + 1,
                    str(exc),
                )
                if attempt > self.requirement_retry_limit:
                    break
                await asyncio.sleep(self.requirement_retry_backoff_seconds * attempt)

        logger.error(
            "requirement {} exhausted retries and will use degraded fallback: {}",
            requirement.requirementId,
            str(last_error),
        )
        if last_error is not None and self._is_confirmed_missing_artifact_error(last_error):
            return self._build_missing_artifact_requirement_result(
                requirement=requirement,
                framework_id=framework_id,
                reason=str(last_error),
            )
        return self._build_degraded_requirement_result(
            requirement=requirement,
            framework_id=framework_id,
        )

    @staticmethod
    def _is_confirmed_missing_artifact_error(exc: Exception) -> bool:
        text = str(exc or "")
        if not text:
            return False
        lowered = text.lower()
        markers = (
            "terminal_error:",
            "confirmed missing",
            "missing or cannot be matched exactly",
            "image artifact is missing on disk",
            "图片文件不存在",
            "文件不存在",
            "已确认缺失",
            "已确认不存在",
        )
        return any(marker in lowered or marker in text for marker in markers)

    @staticmethod
    def _build_missing_artifact_requirement_result(
        *,
        requirement: Requirement,
        framework_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        fail_reason = f"关键材料缺失已确认，直接判定 fail。{reason}"
        control_results: List[Dict[str, Any]] = []
        for control in requirement.controls:
            control_results.append(
                {
                    "requirementId": requirement.requirementId,
                    "controlId": control.controlId,
                    "result": "fail",
                    "reason": fail_reason,
                    "evidence": [],
                    "degraded": False,
                }
            )
        return {
            "result": {"messages": [], "todos": []},
            "check_results": {
                "requirementId": requirement.requirementId,
                "result": "fail",
                "reason": fail_reason,
                "evidence": [],
                "controlResults": control_results,
            },
            "control_results": control_results,
            "framework_id": framework_id,
        }

    async def run(
        self,
        resource_repository: ResourceRepository,
        frameworks: List[Framework],
        max_concurrency: int = 5,
        check_report_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式执行策略检查"""
        time_start = time.time()
        report_id = (check_report_id or "").strip() or f"rpt_{uuid4().hex}"
        with ResourceRepositoryContext(
            resource_repository,
            runtime_overrides=self._build_runtime_overrides(resource_repository),
        ):
            all_requirements = [
                {"requirement": req, "framework_id": fw.frameworkId}
                for fw in frameworks for req in fw.requirements
            ]

            if not all_requirements:
                yield {"status": "error", "error": "No requirements to check"}
                return

            async for result in self._run_concurrent(all_requirements, resource_repository, max_concurrency):
                yield result
        time_end = time.time()
        time_cost = time_end - time_start
        logger.info("策略检查完成，总耗时: {} 秒", time_cost)

    async def _run_concurrent(
        self,
        all_requirements: List[Dict[str, Any]],
        resource_repository: ResourceRepository,
        max_concurrency: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """并发执行所有 requirements"""
        total = len(all_requirements)
        logger.info("开始并发执行 {} 个 requirements，最大并发数: {}", total, max_concurrency)

        completed = 0
        execution_semaphore = asyncio.Semaphore(max_concurrency)
        event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        yielded_control_event_keys: set[tuple[str, str, str]] = set()

        @trace_span(name="PolicyCheckAgent.run_single")
        async def run_single(req_info: Dict[str, Any]) -> None:
            nonlocal completed
            # 每个任务在自己的上下文中运行，避免 ContextVar 错误
            with ResourceRepositoryContext(
                resource_repository,
                runtime_overrides=self._build_runtime_overrides(resource_repository),
            ):
                requirement_start = time.monotonic()
                requirement, framework_id = req_info["requirement"], req_info["framework_id"]
                should_split, split_reasons = self._split_strategy(resource_repository, requirement)
                set_runtime_event_queue(event_queue)
                set_runtime_metadata(
                    framework_id=framework_id,
                    requirement_id=requirement.requirementId,
                    requirement_title=requirement.title,
                )
                logger.info("开始检查 requirement: {}", requirement.requirementId)
                try:
                    invoke_result = await self._execute_requirement_with_retry(
                        resource_repository=resource_repository,
                        requirement=requirement,
                        framework_id=framework_id,
                        event_queue=event_queue,
                        execution_semaphore=execution_semaphore,
                        should_split=should_split,
                        split_reasons=split_reasons,
                    )
                    result = invoke_result["result"]
                    check_results = invoke_result["check_results"]
                    degraded = bool(invoke_result.get("degraded"))
                    aggregated_requirement_result = self._aggregate_requirement_result(
                        requirement,
                        invoke_result["control_results"],
                    )
                    control_results = aggregated_requirement_result["controlResults"]
                    if not isinstance(check_results, dict) or not check_results.get("result"):
                        check_results = aggregated_requirement_result
                    elif not check_results.get("reason") and aggregated_requirement_result.get("reason"):
                        check_results["reason"] = aggregated_requirement_result["reason"]
                    if not check_results.get("controlResults"):
                        check_results["controlResults"] = control_results
                    if not should_split:
                        for control_result in control_results:
                            await event_queue.put(
                                self._build_stream_control_event(
                                    requirement=requirement,
                                    framework_id=framework_id,
                                    control_result=control_result,
                                )
                            )
                    reported_control_ids = {item.get("controlId") for item in control_results}
                    for control in requirement.controls:
                        if control.controlId in reported_control_ids:
                            continue
                        fallback_control_result = self._build_fallback_control_result(
                            requirement=requirement,
                            framework_id=framework_id,
                            control_id=control.controlId,
                            check_results=check_results,
                        )
                        control_results.append(
                            {
                                "requirementId": requirement.requirementId,
                                "controlId": control.controlId,
                                "result": fallback_control_result["result"],
                                "reason": fallback_control_result["reason"],
                                "evidence": fallback_control_result["evidence"],
                                "degraded": True,
                            }
                        )
                        await event_queue.put(fallback_control_result)
                    completed += 1
                    elapsed = round(time.monotonic() - requirement_start, 2)
                    runtime_snapshot = invoke_result.get("runtime_snapshot", {})
                    logger.info(
                        "完成检查 {} ({}/{}) result={} degraded={} elapsed={}s tool_calls={} material_metrics={}",
                        requirement.requirementId,
                        completed,
                        total,
                        check_results.get("result"),
                        degraded,
                        elapsed,
                        runtime_snapshot.get("tool_call_counts", {}),
                        runtime_snapshot.get("material_retrieval_metrics", {}),
                    )
                    messages_data = []
                    for msg in result.get("messages", []):
                        if hasattr(msg, "content") and hasattr(msg, "tool_call"):
                            messages_data.append({"type": type(msg).__name__, "content": msg.content, "tool_call": msg.tool_call})
                        elif hasattr(msg, "content"):
                            messages_data.append({"type": type(msg).__name__, "content": msg.content})
                    await event_queue.put({
                        "event": "requirement_result",
                        "requirement_id": requirement.requirementId,
                        "requirement_title": requirement.title,
                        "framework_id": framework_id,
                        "status": "completed",
                        "progress": f"{completed}/{total}",
                        "elapsed_seconds": elapsed,
                        "check_results": check_results,
                        "control_results": control_results,
                        "tool_call_counts": runtime_snapshot.get("tool_call_counts", {}),
                        "material_retrieval_metrics": runtime_snapshot.get("material_retrieval_metrics", {}),
                        "messages": messages_data,
                        "todos": result.get("todos", []),
                    })
                except Exception as e:
                    completed += 1
                    elapsed = round(time.monotonic() - requirement_start, 2)
                    logger.error("检查 {} 出错: {}", requirement.requirementId, str(e))
                    traceback.print_exc()
                    await event_queue.put({
                        "event": "requirement_result",
                        "requirement_id": requirement.requirementId,
                        "requirement_title": requirement.title,
                        "framework_id": framework_id,
                        "status": "completed",
                        "progress": f"{completed}/{total}",
                        "elapsed_seconds": elapsed,
                        "check_results": {
                            "requirementId": requirement.requirementId,
                            "result": "uncertain",
                            "reason": "校验流程未能稳定完成，已使用自动降级兜底结果。",
                            "evidence": [],
                            "controlResults": [],
                        },
                        "control_results": [],
                        "messages": [],
                        "todos": [],
                    })

        tasks = [asyncio.create_task(run_single(r)) for r in all_requirements]

        while tasks:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                if event.get("event") == "control_result":
                    event_key = (
                        event.get("framework_id") or "",
                        event.get("requirement_id") or "",
                        event.get("control_id") or "",
                    )
                    if event_key in yielded_control_event_keys:
                        logger.debug("skip duplicate streamed control_result: {}", event_key)
                        continue
                    if event_key[2]:
                        yielded_control_event_keys.add(event_key)
                yield event
            except asyncio.TimeoutError:
                pass

            finished = [task for task in tasks if task.done()]
            for task in finished:
                await task
                tasks.remove(task)

        while not event_queue.empty():
            event = await event_queue.get()
            if event.get("event") == "control_result":
                event_key = (
                    event.get("framework_id") or "",
                    event.get("requirement_id") or "",
                    event.get("control_id") or "",
                )
                if event_key in yielded_control_event_keys:
                    logger.debug("skip duplicate streamed control_result in drain: {}", event_key)
                    continue
                if event_key[2]:
                    yielded_control_event_keys.add(event_key)
            yield event


def create_policy_check_agent(model_id: str = "model_bailian_policy_check") -> Any:
    """创建 Policy Check Agent（兼容旧接口）"""
    return PolicyCheckAgent(model_id=model_id).agent


async def run_policy_check(
    resource_repository: ResourceRepository,
    frameworks: List[Framework],
    model_id: str = "model_bailian_policy_check",
    max_concurrency: int = 5
) -> AsyncGenerator[Dict[str, Any], None]:
    """运行策略检查（兼容旧接口）"""
    async for result in PolicyCheckAgent(model_id=model_id).run(resource_repository, frameworks, max_concurrency):
        yield result
