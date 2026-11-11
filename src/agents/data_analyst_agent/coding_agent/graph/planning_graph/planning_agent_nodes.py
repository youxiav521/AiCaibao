import json
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
import threading
from pydantic import BaseModel

from ...decorators.trace_decorator import trace_plan
from src.base.prompt_template_factory import PromptManager
from ...utils.model_invoke_tool import ainvoke_model_with_message
from .planning_agent_prompt import planning_user_input
from ..state import language_guidance

from ...models.document_model import Document
from ..state import PlanningState
from loguru import logger

class CalMetric(BaseModel):
    name: str
    description: str
    formula: str

# 使用threading.local()为每个线程维护独立的need_calculate_metrics变量，确保线程安全
# ！错误 因为tool_call的执行在不同线程，使用get_need_calculate_metrics获得不到同一个need_calculate_metrics
# shuu: 修改tool_node的实现后解决
_thread_local = threading.local()

def get_need_calculate_metrics():
    if not hasattr(_thread_local, 'need_calculate_metrics'):
        _thread_local.need_calculate_metrics = {
            "keypoints": "",
            "metrics": [],
            "limitations": ""
        }
    return _thread_local.need_calculate_metrics

def set_need_calculate_metrics(value):
    _thread_local.need_calculate_metrics = value
# 后续代码中都应通过get_need_calculate_metrics()访问need_calculate_metrics（如需赋值，可用set_need_calculate_metrics）

@tool("add_metric")
def add_metric(metrics: list[CalMetric]):
    """
    批量添加需要计算的多个指标
    参数:
        metrics (list): 指标的json列表，每个元素为dict，包含"name", "description", "formula"字段
            - name (str): 指标名称
            - description (str): 指标描述，参数解释
            - formula (str): 指标的计算公式，不要直接使用数据作为公式中的内容
    返回:
        str: 添加成功或失败的信息，包含每个指标的处理结果
    """
    if not isinstance(metrics, list):
        return "参数必须为指标的json列表（list）。"
    results = []
    need_calculate_metrics = get_need_calculate_metrics()
    for metric in metrics:
        name = metric.name
        description = metric.description
        formula = metric.formula
        if not name:
            results.append("缺少指标名称，未添加。")
            continue
        logger.info(f"---add_metric---: 批量添加指标到计划文本：{name}...")
        # 检查是否已存在同名指标，避免重复添加
        exists = False
        for m in need_calculate_metrics["metrics"]:
            if m["name"] == name:
                exists = True
                break
        if exists:
            results.append(f"指标'{name}'已存在，未重复添加。")
        else:
            need_calculate_metrics["metrics"].append({
                "name": name,
                "description": description,
                "formula": formula
            })
            results.append(f"成功添加指标：{name}")
    set_need_calculate_metrics(need_calculate_metrics)
    return "；".join(results)

@tool("add_limitations")
def add_limitations(limitations:str):
    """
    添加所有指标的限制条件，可能是统一单位、小数精度等
    Args:
        limitations (str): 限制条件
    Returns:
        str: 添加成功或失败的信息
    """
    logger.info(f"---add_limitations---: 添加限制到计划文本：{limitations[:50]}...")
    need_calculate_metrics = get_need_calculate_metrics()
    if limitations and limitations not in need_calculate_metrics["limitations"]:
        if need_calculate_metrics["limitations"]:
            need_calculate_metrics["limitations"] += f"，{limitations}"
        else:
            need_calculate_metrics["limitations"] = limitations
        set_need_calculate_metrics(need_calculate_metrics)
        return "添加限制到计划文本成功"
    else:
        return "添加限制到计划文本失败，请重新添加" 

@tool("add_keypoints")
def add_keypoints(answer_keypoints:str):
    """
    添加回答要点到计划文本
    Args:
        answer_keypoints: 回答要点
    Returns:
        添加成功或添加失败
    """
    logger.info(f"---add_keypoints---: 添加回答要点到计划文本：{answer_keypoints[:50]}...")
    # 支持拓展方式添加keypoints
    if answer_keypoints:
        need_calculate_metrics = get_need_calculate_metrics()
        if not need_calculate_metrics.get("keypoints"):
            need_calculate_metrics["keypoints"] = answer_keypoints
        else:
            # 如果已有keypoints，追加新要点，避免重复
            existing = need_calculate_metrics["keypoints"]
            # 判断是否已包含该要点，避免重复
            if answer_keypoints not in existing:
                # 用分号或换行分隔追加
                need_calculate_metrics["keypoints"] = existing.rstrip() + "\n" + answer_keypoints
            else:
                return "该回答要点已存在，无需重复添加"
        set_need_calculate_metrics(need_calculate_metrics)
        return "添加回答要点到计划文本成功"
    else:
        return "添加回答要点到计划文本失败，请重新添加"

tools = [add_metric, add_limitations, add_keypoints]
tools_by_name = {tool.name: tool for tool in tools}
# planning_tool_node = ToolNode(tools)

def planning_tool_node(state: PlanningState):
    """
    执行工具调用节点
    """
    tool_calls = state["messages"][-1].tool_calls
    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_args=tool_call["args"]
        observations.append(tool.invoke(tool_args))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]
    return {
        **state,
        "messages": tool_outputs
    }

# 关键信息提取agent节点
async def planning_agent_node(state:PlanningState):
    """
    规划回答plan的agent节点
    """
    try:
        logger.info("---planning_agent_node---: 规划回答计划")
        user_question = state.get("question_description", "")
        raw_docs = state.get("relative_documents_list", [])
        # state 中为 List[Document]，反序列化后可能为 dict，统一转为 Document 再序列化
        docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in raw_docs]
        user_prompt = state.get("user_prompt", "")
        answer_language = state.get("answer_language", "zh_CN")
        language_instruction = language_guidance(answer_language)
        logger.info(f"---planning_agent_node---: language_instruction: {answer_language}")
        # 构建系统消息
        planning_agent_system_prompt = PromptManager.getPrompt(
            templateId="data_analyst_planning",
            param={"answer_language_instruction": language_instruction}
        )
        system_message = SystemMessage(
            content=planning_agent_system_prompt
        )
        # 将文档列表转为字符串，多个文档用换行分隔（Document 用 to_dict 后 json.dumps）
        relative_documents = "\n".join([json.dumps(doc.to_dict(), ensure_ascii=False) for doc in docs])
        user_message = HumanMessage(
            content=planning_user_input.format(
                relative_documents_list=relative_documents, 
                question_description=user_question, 
                user_prompt=user_prompt
            )
        )
        # 与 coding_agent_nodes 一致：将 history_messages 拼到 system 后、user 前
        history_messages = state.get("history_messages", []) or []
        messages = [system_message] + list(history_messages) + [user_message] + state.get("messages", [])
        result, model_time = await ainvoke_model_with_message(messages=messages, tools=tools)
        
        # 更新性能指标
        state["model_call_count"] += 1
        state["model_call_times_ms"].append(model_time)

        state["messages"].append(result)
        return state
    except Exception:
        set_need_calculate_metrics({
            "keypoints": "",
            "metrics": [],
            "limitations": ""
        })
        raise

def is_tool_call(state: PlanningState):
    last_message = state["messages"][-1]
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "planning_tool_node"
    return "record_metrics_node"

@trace_plan()
def record_metrics_node(state: PlanningState):
    """
    根据need_calculate_metrics和plan_text，生成plan文本
    """
    logger.info(f"---form_plan_text_node---: 生成最终计划文本")
    need_calculate_metrics = get_need_calculate_metrics()
    logger.info(f"---record_metrics_node---: need_calculate_metrics: {need_calculate_metrics}")
    state["need_calculate_metrics"] = need_calculate_metrics
    set_need_calculate_metrics( {
        "keypoints": "",
        "metrics": [],
        "limitations": ""
    })
    return state


