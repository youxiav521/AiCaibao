import os
import time
from typing import List, Optional

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from src.agents.data_analyst_agent.coding_agent.config.invoke_model_config import model_invoke_config

from ....models.document_model import Document
from ..table_agent_state import TableAgentState
from .fill_table_nodes import is_tool_call, should_review_error, out_put_table_node
from .fill_table_nodes import code_error_reviewer_node, table_code_node, table_tool_node, entry_data_handler_node
from .......utils.langfuse_tracing import trace_span
from loguru import logger
from ....utils.token_utils import (
    count_doc_list_tokens,
    split_doc_list_by_token_limit,
    DEFAULT_TOKEN_LIMIT,
    DEFAULT_PARTITION_TOKEN_VALUE,
)
from src.agents.content_generate_agent.utils import get_localized_message, normalize_locale
from ....utils.table_saver_utils import cleanup_saved_tables
from dotenv import load_dotenv
load_dotenv()

fill_table_agent_builder = StateGraph(TableAgentState)

fill_table_agent_builder.add_node("entry_data_handler_node", entry_data_handler_node)
fill_table_agent_builder.add_node("table_code_node", table_code_node)
fill_table_agent_builder.add_node("table_tool_node", table_tool_node)
fill_table_agent_builder.add_node("code_error_reviewer_node", code_error_reviewer_node)
fill_table_agent_builder.add_node("out_put_table_node", out_put_table_node)

fill_table_agent_builder.add_edge(START, "entry_data_handler_node")
fill_table_agent_builder.add_edge("entry_data_handler_node", "table_code_node")

fill_table_agent_builder.add_conditional_edges( 
    "table_code_node", 
    is_tool_call,
    {
        "table_tool_node": "table_tool_node",
        "out_put_table_node": "out_put_table_node"
    }
)
fill_table_agent_builder.add_conditional_edges(
    "table_tool_node",
    should_review_error,
    {
        "code_error_reviewer_node": "code_error_reviewer_node",
        "table_code_node": "table_code_node"
    }
)
fill_table_agent_builder.add_edge("code_error_reviewer_node", "table_code_node")
fill_table_agent_builder.add_edge("out_put_table_node", END)
# Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler

config = {"recursion_limit": 100}
# 编译工作流
fill_table_agent_workflow = fill_table_agent_builder.compile().with_config(
    config=config
)

class FillTableAgentService:
    @trace_span(name="FillTableAgent.run_fill_table")
    async def run_fill_table(
        self,
        question: str,
        prompt: Optional[str] = None,
        documents: Optional[List[dict]] = None,
        max_iterations: Optional[int] = 5,
        model_id: Optional[str] = "model_for_data_analyst",
        history_messages: Optional[list[AnyMessage]] = None,
        answer_lang: str = "zh_CN",
    ):
        answer_lang = normalize_locale(answer_lang)
        doc_list = [Document.from_dict(d) for d in (documents or [])]
        original_model_id = model_invoke_config.get_model_id()
        model_invoke_config.set_model_id(model_id)
        # 历史消息传入 state，供 table_code_node 等节点拼入 system_messages 后面；state 中直接存 List[Document]
        table_state = TableAgentState(
            question_description=question,
            user_prompt=prompt,
            relative_documents_list=doc_list,
            max_iterations=max_iterations,
            iteration=0,
            history_messages=history_messages if history_messages is not None else [],
            answer_lang=answer_lang,
        )
        
        time_start = time.time()
        table_result = {}
        try:
            table_result = await fill_table_agent_workflow.ainvoke(table_state)
            code_execution_result = table_result.get("code_execution_result", [])
            if code_execution_result and len(code_execution_result) > 0:
                code_execution_result = code_execution_result[-1]
            last_message = table_result["messages"][-1].content
            output_table_data = table_result.get("output_table_data", "")
            logger.info(f"模型最后消息：{last_message[:100]}...")
            time_end = time.time()
            time_cost = time_end - time_start
            # 成功：output_table_data 为 str（HTML）或已符合规范的错误 dict
            return {
                "result": last_message,
                "code_execution_result": code_execution_result,
                "output_table_data": output_table_data,
                "time_cost": time_cost if time_cost else 0,
            }
        except Exception as e:
            logger.error(f"调用fill_table_agent时发生异常: {e}")
            import traceback
            logger.error(f"详细异常堆栈: {traceback.format_exc()}")
            # 尝试删除临时保存的表格文件（ainvoke 抛异常时 table_result 可能为空）
            saved_info = table_result.get("saved_info", []) if table_result else []
            cleanup_saved_tables(saved_info)
            # 返回规范错误结构：含 error 与 filled_table_html（空，由上层补原始表格）
            return {
                "result": f"调用fill_table_agent时发生异常: {e}",
                "code_execution_result": "",
                "output_table_data": {
                    "error": get_localized_message("table_fill_error", answer_lang, error=str(e)),
                    "filled_table_html": "",
                },
                "time_cost": 0,
            }
        finally:
            # 恢复默认model_id
            model_invoke_config.set_model_id(original_model_id)

    @trace_span(name="FillTableAgent.run_fill_table_with_token_limit")
    async def run_fill_table_with_token_limit(
        self,
        question: str,
        prompt: Optional[str] = None,
        documents: Optional[List[dict]] = None,
        max_iterations: Optional[int] = 5,
        model_id: Optional[str] = "model_for_data_analyst",
        history_messages: Optional[list[AnyMessage]] = None,
        token_limit: Optional[int] = None,
        partition_token_value: Optional[int] = None,
        answer_lang: str = "zh_CN",
    ):
        """
        在 token 上限约束下，以滑动窗口方式调用 run_fill_table。

        使用 token_utils 计算 doc_list 总 token 数；超过 token_limit 时，
        按 partition_token_value 将 doc_list 划分为多块（不拆开单个 document），
        再对每一块依次调用 run_fill_table，最后返回最后一次调用的结果。
        """
        # 使用默认 token 上限与划分值
        answer_lang = normalize_locale(answer_lang)
        if token_limit is None:
            token_limit = DEFAULT_TOKEN_LIMIT
        if partition_token_value is None:
            partition_token_value = DEFAULT_PARTITION_TOKEN_VALUE

        # 转为 Document 列表以便计算 token 与划分
        doc_list = [Document.from_dict(d) for d in (documents or [])]
        chunks = split_doc_list_by_token_limit(
            doc_list,
            token_limit=token_limit,
            partition_token_value=partition_token_value,
        )

        # 滑动窗口：依次对每一块调用 run_fill_table
        # 为避免直接修改外部传入的 history_messages，这里统一拷贝一份本地列表
        history_messages = list(history_messages) if history_messages is not None else []
        last_result = None
        for i, chunk in enumerate(chunks):
            # 如果存在上一个窗口的输出结果，则将其作为一条历史消息加入，增强当前窗口的大模型上下文
            if last_result is not None:
                prev_output = last_result.get("output_table_data", "")
                # 仅在存在非空输出时才追加，避免无效噪声
                prev_content = get_localized_message(
                    "table_history_previous_output",
                    answer_lang,
                    output=prev_output,
                )
                if prev_output:
                    history_messages.append(HumanMessage(content=prev_content))
            chunk_documents = [doc.to_dict() for doc in chunk]
            logger.info(
                f"---run_fill_table_with_token_limit---: "
                f"窗口 {i + 1}/{len(chunks)}，本块文档数 {len(chunk)}，"
                f"本块 token 约 {count_doc_list_tokens(chunk)}"
            )
            last_result = await self.run_fill_table(
                question=question,
                prompt=prompt,
                documents=chunk_documents,
                max_iterations=max_iterations,
                model_id=model_id,
                history_messages=history_messages,
                answer_lang=answer_lang,
            )
            # 若某次调用返回错误结构（含 output_table_data.error），可在此选择提前退出或继续下一窗口
            if last_result and isinstance(last_result.get("output_table_data"), dict):
                err = last_result["output_table_data"].get("error")
                if err:
                    logger.warning(f"窗口 {i + 1} 调用返回 error，继续下一窗口")

        # 无任何窗口（doc_list 为空）时仍走一次原方法以保持行为一致
        if last_result is None:
            last_result = await self.run_fill_table(
                question=question,
                prompt=prompt,
                documents=[],
                max_iterations=max_iterations,
                model_id=model_id,
                history_messages=history_messages,
                answer_lang=answer_lang,
            )
        return last_result

