from langgraph.graph.message import BaseMessage, MessagesState, add_messages
from typing_extensions import Annotated, Any, List, Sequence

from ...models.document_model import Document
from ...models.python_code_model import PythonCodeModel


class TableAgentState(MessagesState):
    """Table Agent 的状态定义"""
    question_description: str = ""
    user_prompt: str = ""
    relative_documents_list: List[Document] = []
    table_data:str = ""
    target_table:str = ""
    last_execution_code: PythonCodeModel 
    code_execution_result: list[str] = []
    trace_steps: Sequence[Any] = []
    table_agent_messages: Annotated[Sequence[BaseMessage], add_messages]=[]
    saved_info: List[dict] = []
    last_saved_table_name: str = ""
    # todo 确定输出表格数据的方式，或者作为参数提供给外部
    output_table_data: str = ""
    table_names: str = ""
    context_str: str = ""
    max_iterations: int = 5
    iteration: int = 0
    missing_info: str = ""
    history_messages: Sequence[BaseMessage] = []
    answer_lang: str = "zh_CN"
