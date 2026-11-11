
from typing import List, Optional

from langgraph.graph import StateGraph, END, START


from src.base.langfuse_observe import trace_agent_workflow, trace_span

from .planning_agent_nodes import record_metrics_node, planning_agent_node,planning_tool_node, is_tool_call
from ..state import PlanningState
from loguru import logger

planning_agent_builder = StateGraph(PlanningState)

planning_agent_builder.add_node("planning_agent_node", planning_agent_node)
planning_agent_builder.add_node("planning_tool_node", planning_tool_node)
planning_agent_builder.add_node("record_metrics_node", record_metrics_node)

planning_agent_builder.add_edge(START, "planning_agent_node")

planning_agent_builder.add_conditional_edges(
    "planning_agent_node", 
    is_tool_call,
    {
        "planning_tool_node": "planning_tool_node",
        "record_metrics_node": "record_metrics_node"
    }
)
planning_agent_builder.add_edge("planning_tool_node", "planning_agent_node")

# Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler
# 编译工作流
planning_agent_workflow = planning_agent_builder.compile().with_config(
    config={"recursion_limit": 100}
)

class PlanningAgentService:
    @trace_span(name="PlanningAgent.run_planning")
    async def run_planning(
        self,
        question: str,
        prompt: Optional[str] = None,
        documents: Optional[List[dict]] = None,
        answer_language: str = "zh_CN",
    ):
        from ...models.document_model import Document
        doc_list = [Document.from_dict(d) for d in (documents or [])]
        state = PlanningState(
            question_description=question,
            relative_documents_list=doc_list,
            user_prompt=prompt,
            answer_language=answer_language,
        )
        result = await planning_agent_workflow.ainvoke(state)
        answer_keypoints = result.get("answer_keypoints", "")
        need_calculate_metrics = result.get("need_calculate_metrics", {})
        return {
            "answer_keypoints": answer_keypoints,
            "need_calculate_metrics": need_calculate_metrics
        }
