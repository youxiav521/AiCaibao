from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
import re
import os
import pandas as pd

from ....utils.model_invoke_tool import ainvoke_model_with_message
from ....models.python_code_model import PythonCodeModel
from ..table_agent_state import TableAgentState
from ....code_executor import PythonCodeExecutor, _get_executor_instance
from .fill_table_prompts import (
    EXECUTION_ERROR_MARKER,
    render_banned_text_prompt,
    render_code_reviewer_prompt,
    render_execute_error_prompt,
    render_execute_success_prompt,
    render_filling_agent_prompt,
    render_input_question_prompt,
    render_review_code_result,
)
from loguru import logger
from ....utils.code_extractor import extract_bash_result_from_tool_message
from ....utils.table_saver_utils import format_documents_context, save_documents_to_json, cleanup_saved_tables
from src.agents.content_generate_agent.utils import contains_cjk, get_localized_message, normalize_locale

from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "code_executor", "tmp_tables")
# 执行代码的工具
table_import="""
# 导入表格工具
from table_utils_for_agent import read_table, save_table
from precision_calculator import quantize,add,sub,mul,div

"""
@tool("execute_code_tool")
async def execute_code_tool(
    # imports: str, 
    code: str, 
    description: str,
    answer_lang: str = "zh_CN",
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
    # 导入read_table
    
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
    # 检查危险操作，禁止关键行为，包含open("xxx.csv")，df.to_csv，df.to_excel, with open, pickle.dump, df.to_dict, df.to_json
    forbidden_patterns = [
        "open(",
        ".to_csv(",
        ".to_excel(",
        "with open(",
        "pickle.dump(",
        ".to_dict(",
        ".to_json("
    ]
    ban_text = render_banned_text_prompt(answer_lang)
    # 检查imports和code中是否含有禁止内容
    check_strings = (table_import or "") + "\n" + (code or "")
    for pattern in forbidden_patterns:
        if pattern in check_strings:
            return ban_text
    exec_result = await executor.execute(code_model)

    # 根据执行结果拼接代码和结果给大模型
    success=exec_result.get("success")
    error=exec_result.get("error")
    output=exec_result.get("output")
    if success:
        return render_execute_success_prompt(
            answer_lang,
            imports=table_import,
            code=code,
            execute_result=output,
        )
    else:
        return render_execute_error_prompt(
            answer_lang,
            imports=table_import,
            code=code,
            execute_error_message=error,
        )

tools = [execute_code_tool]
tools_by_name = {tool.name: tool for tool in tools}

# @trace_code()
async def table_tool_node(state: TableAgentState):
    """
    执行工具调用节点
    """
    tool_calls = state["messages"][-1].tool_calls

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool = tools_by_name[tool_name]
        tool_args=tool_call["args"]
        if tool_name == "execute_code_tool" and "answer_lang" not in tool_args:
            tool_args = {**tool_args, "answer_lang": state.get("answer_lang", "zh_CN")}
        observations.append(await tool.ainvoke(tool_args))
        if tool_name == "execute_code_tool":
            # 安全获取 imports，如果不存在则使用空字符串
            # imports_value = tool_args.get("imports", "")
            code_model = PythonCodeModel(
                imports=table_import,
                code=tool_args["code"],
                description=tool_args["description"]
            )
            state["last_execution_code"]=code_model

    # Create tool message outputs
    tool_outputs = []
    for observation, tool_call in zip(observations, tool_calls):
        tool_name=tool_call["name"]
        tool_output = ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        tool_outputs.append(tool_output)
        if tool_name == "execute_code_tool":
            code_execution_result = extract_bash_result_from_tool_message(observation)
            # 使用非贪婪匹配，匹配"保存表格"和"成功"之间的所有内容（包括空格）
            # 支持两种格式：1) save_table:保存表格 xxx 成功  2) 保存表格 xxx 成功
            match = re.search(r"(?:save_table:)?保存表格\s+(.+?)\s+成功", str(code_execution_result))
            last_saved_table_name = match.group(1).strip() if match else ""
            state["last_saved_table_name"] = last_saved_table_name
            code_execution_result_list = state.get("code_execution_result", [])
            if code_execution_result:
                code_execution_result_list.append(str(code_execution_result))
                state["code_execution_result"] = code_execution_result_list
        
    return {
        **state,
        "messages": tool_outputs
    }

# 入口数据处理节点
# @trace_data()
def entry_data_handler_node(state: TableAgentState):
    """入口数据处理节点（异常安全：保证节点异常时临时文件也能被清理）"""
    # logger.info("---entry_data_handler_node---: 数据处理")
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
async def table_code_node(state: TableAgentState):
    logger.info(f"---table_code_node---:代码生成")
    
    # 构建用户消息
    question_description = state["question_description"]
    user_prompt = state.get("user_prompt", "")
    table_names = state.get("table_names", "")
    context_str = state.get("context_str", "")
    answer_lang = state.get("answer_lang", "zh_CN")
   
    user_message = HumanMessage(
        content=render_input_question_prompt(
            answer_lang,
            user_prompt=user_prompt,
            question_description=question_description,
            context=context_str,
            table_info=table_names,
        )
    )

    # 构建系统消息
    system_message = SystemMessage(
        content=render_filling_agent_prompt(answer_lang, table_names)
    )

    # 将历史消息拼到 system_messages 后面，再拼当前 user_message 和本工作流已有 messages
    history_messages = state.get("history_messages", []) or []
    messages = [system_message] + list(history_messages) + [user_message] + list(state["messages"])
    result, model_time = await ainvoke_model_with_message(messages=messages, tools=tools)
    
    state["messages"].append(result)
    last_message = state["messages"][-1]
    
    # 检查是否有工具调用，更新迭代次数
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        state["iteration"] = state.get("iteration", 0) + 1
    # extracted_code = extract_python_code(result.content)
    return state

def is_tool_call(state: TableAgentState):
    last_message = state["messages"][-1]
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info(f"---is_tool_call---: 工具调用次数：{state['iteration']}")
        # 如果迭代次数超过最大迭代次数，直接到最后回答节点
        if state["iteration"] > state["max_iterations"]:
            logger.info(f"---is_tool_call---: 迭代次数超过最大迭代次数，截断")
            # state["messages"].append(HumanMessage(content=over_max_iterations_prompt.format(max_iterations=state["max_iterations"])))
            return "out_put_table_node"
        return "table_tool_node"
    return "out_put_table_node"

# 代码审查节点
# @trace_code_error()
async def code_error_reviewer_node(state: TableAgentState):
    """代码错误审查节点"""
    logger.info("---code_error_reviewer_node---: 代码错误审查生成")
    
    # 获取工具执行结果
    last_message = state["messages"][-1]
    # 构建用户消息
    question_description = state["question_description"]
    user_prompt = state.get("user_prompt", "")
    table_names = state.get("table_names", "")
    context_str = state.get("context_str", "")
    answer_lang = state.get("answer_lang", "zh_CN")
    
    question_description_str = render_input_question_prompt(
            answer_lang,
            user_prompt=user_prompt,
            question_description=question_description,
            context=context_str,
            table_info=table_names,
    )
    error_content = last_message.content
    
    # 构建审查提示
    # system_message = SystemMessage(content=code_reviewer_prompt + "no_think")
    system_message = SystemMessage(
        content=render_code_reviewer_prompt(
            answer_lang,
            table_names=table_names,
            question_description_prompt=question_description_str,
        )
    )
    user_message = HumanMessage(content=error_content)
    
    # 调用模型进行错误审查
    review_messages = [system_message, user_message]
    review_result, review_time = await ainvoke_model_with_message(messages=review_messages)

    # 将审查结果添加到消息历史
    review_content = review_result.content
    logger.info(f"---code_error_reviewer_node---: 错误审查完成，返回修改建议重新生成代码：{review_content[:50]}...")
    review_message = HumanMessage(
        content=render_review_code_result(answer_lang, review_suggestions=review_content)
    )
    state["messages"].append(review_message)
    
    return state

def should_review_error(state: TableAgentState):
    """判断是否需要错误审查"""
    last_message = state["messages"][-1]
    # 检查是否包含错误信息
    if hasattr(last_message, 'content') and EXECUTION_ERROR_MARKER in last_message.content:
        logger.info("---should_review_error---: 检测到错误，需要审查")
        return "code_error_reviewer_node"
    else:
        return "table_code_node"
        # return "record_execution_answer_node"

def out_put_table_node(state: TableAgentState):
    """输出表格节点"""
    last_saved_table_name = state.get("last_saved_table_name", "")
    answer_lang = state.get("answer_lang", "zh_CN")
    logger.info(f"---out_put_table_node---: 输出表格: {last_saved_table_name}")
    # TODO 将表格数据进行类型转换为需要的格式，作为模型的表格输出？
    # 写死文件位置，直接按last_saved_table_name读取文件
    # 删除saved_info中保存的临时文件
    saved_info = state.get("saved_info", [])
    cleanup_saved_tables(saved_info)
    if last_saved_table_name:
        try:
            # 写死目录路径
            table_path = os.path.join(
                BASE_URL,
                last_saved_table_name
            )
            logger.info(f"---out_put_table_node---: 读取表格文件路径: {table_path}")
            if not os.path.exists(table_path):
                raise FileNotFoundError(f"文件不存在: {table_path}")
            # 直接pandas解析
            df = pd.read_json(table_path, orient="split")
            # TODO 输出的格式有待商榷，这里为测试先使用html
            # 为避免科学记数法显示，将所有数值列格式化
            import numpy as np
            def format_number(x):
                if isinstance(x, float) or isinstance(x, np.floating):
                    # 保留有效小数位，去掉科学记数法
                    return '{0:.8f}'.format(x).rstrip('0').rstrip('.') if '.' in '{0:.8f}'.format(x) else '{0:.8f}'.format(x)
                return x
            df_no_sci = df.copy()
            for col in df_no_sci.select_dtypes(include=[np.number]).columns:
                df_no_sci[col] = df_no_sci[col].apply(format_number)
            df_html = df_no_sci.to_html()
            # 成功后删除文件
            os.remove(table_path)
            # 成功：填表结果规范为 str（填充后的表格 HTML）
            state["output_table_data"] = df_html
        except Exception as e:
            logger.error(f"读取保存的表格失败: {e}")
            # 失败：规范错误结构，filled_table_html 由上游 answer_table_workflow 补充
            state["output_table_data"] = {
                "error": get_localized_message("table_read_saved_error", answer_lang, error=str(e)),
                "filled_table_html": "",
            }
    else:
        logger.warning("未找到已保存的表格名（last_saved_table_name为空）")
        # 记录最后一次代码执行结果以便排查问题
        code_execution_result_list = state.get("code_execution_result", [])
        last_execution_code = state.get("last_execution_code")
        normalized_locale = normalize_locale(answer_lang)

        warning_lines = [get_localized_message("table_missing_related_info_prefix", answer_lang)]

        if last_execution_code:
            if hasattr(last_execution_code, "description") and last_execution_code.description:
                description = last_execution_code.description
                if not (normalized_locale == "en_US" and contains_cjk(description)):
                    warning_lines.append(
                        f"{get_localized_message('table_generated_description_label', answer_lang)}: "
                        f"{description}"
                    )

        if code_execution_result_list and len(code_execution_result_list) > 0:
            last_result = code_execution_result_list[-1]
            if not (normalized_locale == "en_US" and contains_cjk(last_result)):
                warning_lines.append(
                    f"{get_localized_message('table_execution_result_label', answer_lang)}:\n{last_result}"
                )

        if len(warning_lines) == 1:
            warning_lines.append(get_localized_message("table_missing_key_info", answer_lang))

        warning_msg = "\n".join(warning_lines)
        logger.warning(warning_msg)
        # 失败：规范错误结构
        state["output_table_data"] = {"error": warning_msg, "filled_table_html": ""}
    return state
