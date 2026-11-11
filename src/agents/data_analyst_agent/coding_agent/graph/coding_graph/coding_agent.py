import os
import time
from typing import Dict, List, Optional
from typing_extensions import Any

from langgraph.graph import END, START, StateGraph
from ..state import CodingState
from .coding_agent_nodes import is_tool_call, should_review_error
from .coding_agent_nodes import code_error_reviewer_node, generate_code_node, coding_tool_node, entry_data_handler_node
from src.base.langfuse_observe import trace_span
from ...models.document_model import Document
from dotenv import load_dotenv
from loguru import logger
load_dotenv()

coding_agent_builder = StateGraph(CodingState)

coding_agent_builder.add_node("entry_data_handler_node", entry_data_handler_node)
coding_agent_builder.add_node("generate_code_node", generate_code_node)
coding_agent_builder.add_node("coding_tool_node", coding_tool_node)
coding_agent_builder.add_node("code_error_reviewer_node", code_error_reviewer_node)
# coding_agent_builder.add_node("record_execution_answer_node", record_execution_answer_node)

coding_agent_builder.add_edge(START, "entry_data_handler_node")
coding_agent_builder.add_edge("entry_data_handler_node", "generate_code_node")

coding_agent_builder.add_conditional_edges( 
    "generate_code_node", 
    is_tool_call,
    {
        "coding_tool_node": "coding_tool_node",
        # "generate_code_node": "generate_code_node", # 如果工具调用次数达到最大迭代次数，生成最后回答
        "END": END
    }
)
coding_agent_builder.add_conditional_edges(
    "coding_tool_node",
    should_review_error,
    {
        "code_error_reviewer_node": "code_error_reviewer_node",
        "generate_code_node": "generate_code_node"
    }
)
coding_agent_builder.add_edge("code_error_reviewer_node", "generate_code_node")

# coding_agent_builder.add_edge("record_execution_answer_node", END)
# Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler
config = {"recursion_limit": 100}
# 编译工作流
coding_agent_workflow = coding_agent_builder.compile().with_config(
    config=config
)
# coding_agent_workflow = coding_agent_builder.compile()

class CodingAgentService:
    # 使用langfuse的trace_agent_workflow装饰器，追踪全链路
    @trace_span(name="CodingAgent.run_question")
    async def run_question(
        self, 
        question: str,
        prompt: Optional[str]=None, 
        documents: Optional[List[dict]] = None,
        need_calculate_metrics: Optional[Dict[str, Any]] = None,
        max_iterations: Optional[int] = 5
    ):
        doc_list = [Document.from_dict(d) for d in (documents or [])]
        state = CodingState(
            question_description=question,
            user_prompt=prompt,
            relative_documents_list=doc_list,
            need_calculate_metrics=need_calculate_metrics or {},
            max_iterations=max_iterations,
            iteration=0
        )
        
        time_start = time.time()
        try:
            result=await coding_agent_workflow.ainvoke(state)
        except Exception as e:
            logger.error(f"调用coding agent时发生异常: {e}")
            # 尝试删除临时保存的表格文件
            saved_table_docs = result.get("saved_table_docs", [])
            if saved_table_docs:
                for doc in saved_table_docs:
                    if os.path.exists(doc["file_path"]):
                        os.remove(doc["file_path"])
                
        time_end = time.time()
        time_cost = time_end - time_start
        code_execution_result = result.get("code_execution_result", "")
        last_message=result['messages'][-1].content
        logger.info(f"模型最后消息：{last_message[:100]}...")
        return {
            "result": last_message, 
            "code_execution_result": code_execution_result, 
            "time_cost": time_cost if time_cost else 0
        }