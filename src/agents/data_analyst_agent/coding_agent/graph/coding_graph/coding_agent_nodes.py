import json
import os
import json_repair
import time
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START
from langgraph.types import Command, Send
from langgraph.prebuilt import ToolNode
from loguru import logger

from src.base.prompt_template_factory import PromptManager
from ...decorators.trace_decorator import trace_code, trace_code_error, trace_data

from ...utils.model_invoke_tool import ainvoke_model_with_message

from ...models.python_code_model import PythonCodeModel
from ..state import CodingState
from ...code_executor import PythonCodeExecutor, _get_executor_instance
from .coding_agent_prompts import execute_success_prompt, input_question_prompt
from .coding_agent_prompts import plan_text, execute_error_prompt, review_code_result
from ..state import language_guidance
from ...utils.code_extractor import extract_bash_result_from_tool_message, extract_code_error
from ...utils.table_saver_utils import cleanup_saved_tables, format_documents_context, save_documents_to_json

from dotenv import load_dotenv
load_dotenv()

table_import="""
# 导入表格工具
from table_utils_for_agent import read_table
from precision_calculator import quantize,add,sub,mul,div

"""
# 执行代码的工具
@tool("execute_code_tool")
async def execute_code_tool(
    # imports: str, 
    code: str, 
    description:str
):
    """
    执行代码的工具函数
    参数:
        code: 代码逻辑的部分
        description: 代码逻辑的描述，描述解决问题的计算步骤
    返回:
        response: 包含执行结果或错误信息。
    """
    # logger.info(f"---execute_code_tool---: 执行代码\n{imports[:50]}\n{code[:50]}...")
    logger.info(f"---execute_code_tool---: 执行代码\n{code[:50]}...")

    # 把imports中的\n字符转化为换行符
    # if isinstance(imports, str):
    #     imports = imports.replace('\\n', '\n')
    # 在执行前将table_import添加到imports里
    # imports = table_import + imports
    executor = _get_executor_instance()
    code_model = PythonCodeModel(
        imports=table_import,
        code=code,
        description=description
    )
    exec_result = await executor.execute(code_model)

    # 根据执行结果拼接代码和结果给大模型
    success=exec_result.get("success")
    error=exec_result.get("error")
    output=exec_result.get("output")
    
    output_str = str(output).strip() if output is not None else ""
    # 检查代码执行成功但output为空的情况
    if success and not output_str:
        logger.warning(f"---execute_code_tool---: 代码执行成功但输出为空，原始output: {repr(output)}, success: {success}, error: {error}")
        success = False
        if not error:
            error = "Error-代码执行成功但没有输出结果，你的代码没有正确使用print()函数打印计算结果，或者打印对象为空（空列表，空字典等），请确认代码中的输出逻辑，修改后重新执行"
    
    if success:
        return execute_success_prompt.format(
            imports=table_import,
            code=code,
            execute_result=output
        )
    else:
        return execute_error_prompt.format(
            imports=table_import,
            code=code,
            execute_error_message=error
        )
        

tools = [execute_code_tool]
tools_by_name = {tool.name: tool for tool in tools}
# coding_tool_node = ToolNode(tools)

@trace_code()
async def coding_tool_node(state: CodingState):
    """
    执行工具调用节点
    """
    tool_calls = state["messages"][-1].tool_calls

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_args=tool_call["args"]
        observations.append(await tool.ainvoke(tool_args))

        code_model = PythonCodeModel(
            imports=table_import,
            code=tool_args["code"],
            description=tool_args["description"]
        )
        state["last_execution_code"]=code_model

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]
    for i, observation in enumerate(observations):
        code_execution_result = extract_bash_result_from_tool_message(observation)
        code_execution_result_list = state.get("code_execution_results", [])

        # 检查是否为错误结果，记录错误代码用于后续审查
        # 错误信息在 observation 中包含 "Error-出现错误"，错误详情在 ```error``` 块中
        if "Error-出现错误" in observation:
            tool_call = tool_calls[i]
            error_message = extract_code_error(observation)
            # 临时保存错误代码，用于后续错误审查
            state["last_error_code"] = tool_call["args"].get("code", "")
            state["last_error_message"] = error_message
            logger.warning(f"---coding_tool_node---: 代码执行错误: {error_message[:100]}...")

        if code_execution_result:
            result_str = str(code_execution_result)
            code_execution_result_list.append(result_str)
            state["code_execution_results"] = code_execution_result_list
    return {
        **state,
        "messages": tool_outputs
    }

# 入口数据处理节点
# @trace_data()
def entry_data_handler_node(state: CodingState):
    """入口数据处理节点（异常安全：保证节点异常时临时文件也能被清理）"""
    # state 中 relative_documents_list 为 List[Document]，save_documents_to_json 支持 List[Document] | List[dict]
    relative_documents = state.get("relative_documents_list", [])
    saved_info = save_documents_to_json(relative_documents)

    try:
        # 将每个表格和上下文信息拼接为文本：- 表格名 {title}\n 表结构示例 {dataframe}\n
        context_str, table_names = format_documents_context(saved_info)
        return {
            **state,
            "table_names": table_names,
            "context_str": context_str,
            "saved_info": saved_info
        }
    except Exception:
        # 节点处理失败：清理已保存的临时文件
        cleanup_saved_tables(saved_info)
        raise

# 代码生成节点
async def generate_code_node(state: CodingState):
    # 生成简单代码来调试 
    """根据用户输入生成简单的Python代码"""
    logger.info(f"---generate_code_node---:agent生成")
    
    # 构建用户消息
    question_description = state["question_description"]
    user_prompt = state.get("user_prompt", "")
    need_calculate_metrics = state.get("need_calculate_metrics", {})
    need_calculate_metrics_str = json.dumps(need_calculate_metrics, ensure_ascii=False)
    answer_keypoints = need_calculate_metrics.get("keypoints", "")
    
    table_names = state.get("table_names", "")
    if not table_names or len(table_names) == 0: table_names = "无相关表格数据提供"
    context_str = state.get("context_str", "")
    
    plan_text_str = plan_text.format(
        user_prompt=user_prompt,
        question_description=question_description,
        answer_keypoints=answer_keypoints,
        need_calculate_metrics=need_calculate_metrics_str
    )
   
    user_message = HumanMessage(
        content=input_question_prompt.format(
            plan_text=plan_text_str,
            context=context_str,
            table_info=table_names
        )
    )

    # 构建系统消息 使用全局方法
    answer_language = state.get("answer_language", "zh_CN")
    language_instruction = language_guidance(answer_language)
    coding_agent_prompt = PromptManager.getPrompt(
        "data_analyst_coding",
        param={"answer_language_instruction": language_instruction}
    )
    system_message = SystemMessage(content = coding_agent_prompt)
    # 将 history_messages 拼到 system 后、user 前，与 fill_table 的 table_code_node 一致，供滑动窗口等场景传入上一轮回答
    history_messages = state.get("history_messages", []) or []
    recent_messages = state["messages"]
    messages = [system_message] + list(history_messages) + [user_message] + recent_messages
    result, model_time = await ainvoke_model_with_message(messages=messages, tools=tools)
    
    # 更新性能指标
    state["model_call_count"] += 1
    state["model_call_times_ms"].append(model_time)
    
    state["messages"].append(result)
    last_message = state["messages"][-1]
    
    # 检查是否有工具调用，更新迭代次数
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        state["iteration"] = state.get("iteration", 0) + 1
    # extracted_code = extract_python_code(result.content)
    return state

def is_tool_call(state: CodingState):
    last_message = state["messages"][-1]
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info(f"---is_tool_call---: 工具调用次数：{state['iteration']}")
        # 如果迭代次数超过最大迭代次数，直接到最后回答节点
        if state["iteration"] > state["max_iterations"]:
            logger.info(f"---is_tool_call---: 迭代次数超过最大迭代次数，直接到最后回答")
            # state["messages"].append(HumanMessage(content=over_max_iterations_prompt.format(max_iterations=state["max_iterations"])))
            return "END"
        return "coding_tool_node"
    return "END"

# 代码审查节点
@trace_code_error()
async def code_error_reviewer_node(state: CodingState):
    """代码错误审查节点"""
    logger.info("---code_error_reviewer_node---: 开始代码错误审查")
    
    # 获取工具执行结果
    last_message = state["messages"][-1]
    # 构建用户消息
    question_description = state["question_description"]
    user_prompt = state.get("user_prompt", "")
    need_calculate_metrics = state.get("need_calculate_metrics", {})
    need_calculate_metrics_str = json.dumps(need_calculate_metrics, ensure_ascii=False)
    answer_keypoints = need_calculate_metrics.get("keypoints", "")
    
    table_names = state.get("table_names", "")
    if not table_names or len(table_names) == 0: table_names = "无相关表格数据提供"
    context_str = state.get("context_str", "")
    
    plan_text_str = plan_text.format(
        user_prompt=user_prompt,
        question_description=question_description,
        answer_keypoints=answer_keypoints,
        need_calculate_metrics=need_calculate_metrics_str
    )
    question_description_str=input_question_prompt.format(
            plan_text=plan_text_str,
            context=context_str,
            table_info=table_names
    )
    error_content = last_message.content

    # 获取语言设置
    answer_language = state.get("answer_language", "zh_CN")
    language_instruction = language_guidance(answer_language)

    # 构建审查提示
    error_reviewer_prompt = PromptManager.getPrompt(
        templateId = "data_analyst_error_review",
        param = {
            "table_names": table_names,
            "question_description_prompt": question_description_str,
            "answer_language_instruction": language_instruction
        }
    )
    system_message = SystemMessage(content=error_reviewer_prompt)
    user_message = HumanMessage(content=error_content)
    
    # 调用模型进行错误审查
    review_messages = [system_message, user_message]
    review_result, review_time = await ainvoke_model_with_message(messages=review_messages)
    
    # 更新性能指标
    state["model_call_count"] += 1
    state["model_call_times_ms"].append(review_time)
    
    # 将审查结果添加到消息历史
    review_content = review_result.content
    logger.info(f"---code_error_reviewer_node---: 错误审查完成，返回修改建议重新生成代码：{review_content[:50]}...")
    review_message = HumanMessage(content=review_code_result.format(review_suggestions=review_content))
    state["messages"].append(review_message)
    
    # 记录错误信息（包含错误代码、错误原因和修改建议）
    error_info = {
        "code": state.get("last_error_code", ""),
        "error_message": state.get("last_error_message", ""),
        "review_suggestion": review_content,
        "timestamp": time.time()
    }
    state["code_errors"].append(error_info)
    state["code_error_count"] += 1
    logger.info(f"---code_error_reviewer_node---: 记录代码执行错误，错误计数: {state['code_error_count']}")
    
    return state

def should_review_error(state: CodingState):
    """判断是否需要错误审查"""
    last_message = state["messages"][-1]
    # 检查是否包含错误信息
    if hasattr(last_message, 'content') and "Error-出现错误" in last_message.content:
        logger.info("---should_review_error---: 检测到错误，需要审查")
        return "code_error_reviewer_node"
    else:
        return "generate_code_node"
        # return "record_execution_answer_node"