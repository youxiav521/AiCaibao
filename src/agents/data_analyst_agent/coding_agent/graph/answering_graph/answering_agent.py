from langgraph.graph import END, START, StateGraph

from ......utils.langfuse_tracing import trace_span
from ..state import AnsweringState
from .answering_agent_node import answering_node

answering_agent_builder = StateGraph(AnsweringState)
answering_agent_builder.add_node("answering_node", answering_node)
answering_agent_builder.add_edge(START, "answering_node")
answering_agent_builder.add_edge("answering_node", END)

# Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler
config={"recursion_limit": 50}
answering_agent_workflow = answering_agent_builder.compile().with_config(
    config=config
)

class AnsweringAgentService:
    @trace_span(name="AnsweringAgent.run_answering")
    async def run_answering(self, state: AnsweringState):
        result = await answering_agent_workflow.ainvoke(state)
        return result