from langgraph.graph import END, START, StateGraph

from ......utils.langfuse_tracing import trace_span
from ..state import ExtractState
from .extract_agent_node import split_documents_node, judge_need_extract_node, extract_node

extract_agent_builder = StateGraph(ExtractState)
extract_agent_builder.add_node("split_documents_node", split_documents_node)
extract_agent_builder.add_node("extract_node", extract_node)
extract_agent_builder.add_edge(START, "split_documents_node")
extract_agent_builder.add_conditional_edges(
    "split_documents_node",
    judge_need_extract_node,
    {
        "extract_node": "extract_node",
        "END": END
    }
)
extract_agent_builder.add_edge("extract_node", END)

# Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler
config={"recursion_limit": 50}
extract_agent_workflow = extract_agent_builder.compile().with_config(
    config=config
)

class ExtractAgentService:
    @trace_span(name="ExtractAgent.run_extract")
    async def run_extract(self, state: ExtractState):
        result = await extract_agent_workflow.ainvoke(state)
        return result