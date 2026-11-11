
from langchain_core.messages import HumanMessage, SystemMessage
from src.agents.content_generate_agent.utils import get_answering_prompt_locale_vars
from src.base.prompt_template_factory import PromptManager
from ..state import language_guidance

from ...decorators.trace_decorator import trace_answer
from ...utils.model_invoke_tool import ainvoke_model_with_message
from ...utils.table_saver_utils import cleanup_saved_tables
from .answering_agent_prompts import answer_user_prompt
from ..state import AnsweringState
from loguru import logger


@trace_answer()
async def answering_node(state: AnsweringState):
    """回答节点"""
    logger.info("---record_execution_answer_node---: 记录代码执行结果并生成最终回答")
    # todo 生成一条最后回答 待优化prompt 可以考虑不把原始数据给他，直接让他做回答
    # 构建用户消息
    question_description = state["question_description"]
    user_prompt = state.get("user_prompt", "")
    context_str = state.get("context_str", "")
    code_execution_result_list = state.get("code_execution_results", [])
    need_calculate_metrics = state["need_calculate_metrics"]
    answer_language = state.get("answer_language", "zh_CN")
    prompt_locale_vars = get_answering_prompt_locale_vars(answer_language)

    related_data_calculation_results_str = ""
    for i, result in enumerate(code_execution_result_list, 1):
        related_data_calculation_results_str += f"第{i}次相关计算结果：\n {result}\n"
    related_data_calculation_results_str = related_data_calculation_results_str.strip()
    user_message = HumanMessage(content=answer_user_prompt.format(
        user_question=question_description,
        user_prompt=user_prompt,
        context_str=context_str,
        related_data_calculation_results=related_data_calculation_results_str,
        answer_cot=need_calculate_metrics,
        **prompt_locale_vars,
    ))

    # 构建系统消息
    language_instruction = language_guidance(answer_language)
    answer_system_prompt = PromptManager.getPrompt(
        templateId="data_analyst_answering",
        param={
            "answer_language_instruction": language_instruction,
            **prompt_locale_vars,
        }
    )
    system_message = SystemMessage(content=answer_system_prompt)
    # 与 coding_agent_nodes 一致：将 history_messages 拼到 system 后、user 前
    history_messages = state.get("history_messages", []) or []
    messages = [system_message] + list(history_messages) + [user_message]
    result, model_time = await ainvoke_model_with_message(messages=messages)
    
    # 更新性能指标
    state["model_call_count"] += 1
    state["model_call_times_ms"].append(model_time)
    
    state["messages"].append(result)

    # 删除临时保存的表格
    saved_info = state.get("saved_info", [])
    cleanup_saved_tables(saved_info)

    return state
