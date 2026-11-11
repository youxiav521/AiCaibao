from typing import List, Dict, Any, Optional, Sequence, TypedDict

from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import MessagesState, add_messages

from src.agents.data_analyst_agent.coding_agent.models.document_model import Document
from src.agents.data_analyst_agent.coding_agent.models.python_code_model import PythonCodeModel
from src.agents.content_generate_agent.utils import get_answer_language_instruction

LANGUAGE_GUIDANCE_TEMPLATE = """### 输出语言要求
{answer_language_instruction}
- 除 Python 代码本身外，tool call 的 `description`、说明文字、错误说明等自然语言内容应尽量遵守上述语言要求。
- Python 代码、函数名、表名和字段名保持原样，不需要翻译。
"""

def language_guidance(locale: str) -> str:
    return LANGUAGE_GUIDANCE_TEMPLATE.format(
        answer_language_instruction=get_answer_language_instruction(locale)
    )

class PlanningState(MessagesState):
    """Planning Agent 的状态定义"""
    question_description: str
    user_prompt: str = ""
    relative_documents_list: List[Document] = []
    need_calculate_metrics: Dict[str, Any] = {}
    trace_steps: Sequence[Any] = []
    history_messages: Sequence[BaseMessage] = []
    max_iterations: int = 5
    answer_language: Optional[str] = "zh_CN"

    planning_messages: Annotated[Sequence[BaseMessage], add_messages] = []
    
    # 性能指标
    model_call_count: int = 0
    model_call_times_ms: List[float] = []


class CodingState(MessagesState):
    """Coding Agent 的状态定义"""
    question_description: str
    user_prompt: str = ""
    relative_documents_list: List[Document] = []
    need_calculate_metrics: Dict[str, Any] = {}
    trace_steps: Sequence[Any] = []
    saved_info: List[dict] = []
    context_str: str = ""
    code_execution_results: list[str] = []
    history_messages: Sequence[BaseMessage] = []
    answer_language: Optional[str] = "zh_CN"

    last_execution_code: PythonCodeModel
    coding_messages: Annotated[Sequence[BaseMessage], add_messages] = []
    table_names: str = ""
    max_iterations: int = 5
    iteration: int = 0
    
    # 临时保存错误信息，用于错误审查
    last_error_code: str = ""
    last_error_message: str = ""
    
    # 性能指标
    model_call_count: int = 0
    model_call_times_ms: List[float] = []
    code_error_count: int = 0
    code_errors: List[Dict[str, Any]] = []


class AnsweringState(MessagesState):
    """Answering Agent 的状态定义"""
    question_description: str
    user_prompt: str = ""
    relative_documents_list: List[Document] = []
    need_calculate_metrics: Dict[str, Any] = {}
    trace_steps: Sequence[Any] = []
    context_str: str = ""
    code_execution_results: list[str] = []
    history_messages: Sequence[BaseMessage] = []
    answer_language: Optional[str] = "zh_CN"
    code_errors: List[Dict[str, Any]] = []

    saved_info: List[dict] = []
    answering_messages: Annotated[Sequence[BaseMessage], add_messages] = []
    
    # 性能指标
    model_call_count: int = 0
    model_call_times_ms: List[float] = []
    code_error_count: int = 0


class ExtractState(MessagesState):
    """Extract Agent 的状态定义"""
    question_description: str
    user_prompt: str = ""
    relative_documents_list: List[Document] = []
    need_calculate_metrics: Dict[str, Any] = {}
    max_iterations: int = 5
    trace_steps: Sequence[Any] = []
    context_str: str = ""
    text_documents: List[dict] = []
    table_documents: List[dict] = []
    saved_info: List[dict] = []
    history_messages: Sequence[BaseMessage] = []
    answer_language: Optional[str] = "zh_CN"
    extract_messages: Annotated[Sequence[BaseMessage], add_messages] = []
    
    # 性能指标
    model_call_count: int = 0
    model_call_times_ms: List[float] = []