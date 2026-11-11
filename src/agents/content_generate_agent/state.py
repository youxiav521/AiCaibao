from __future__ import annotations

from typing import Any, List, Optional, Tuple, Literal
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel
from .models import FinalAnswer, AnswerAgentStepItem

class PlannerResponse(BaseModel):
    """Planner 节点的结构化输出模型"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    needs_computation: bool = Field(..., description="是否需要进一步计算处理")
    reasoning: str = Field(..., description="解释为什么需要或不需要计算处理")
    computation_prompt: str = Field(description="如果需要计算，提供具体的处理逻辑描述")
    computation_input: List[Any] = Field(default=[], description="如果需要计算，明确计算所需的输入内容，格式为doc_id的列表")
class QuotedAnswer(BaseModel):
    """Reporter节点的结构化输出模型"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    is_answerable: bool = Field(..., description="Whether the answer is answerable from the given sources.")
    answer: str = Field(...,description="The answer to the user question, which includes proper inline citations in [doc_id] format.")

class ContentGenState(MessagesState):
    """内容生成工作流的状态结构"""
    # 输入
    question: str
    content: List[dict]
    content_cnt: int # 应等于len(content)，目前靠初始化时候保证相等，暂无验证逻辑。暂时仅用于citation的编号
    model_id: str

    # planner 输出
    reasoning: str
    computation_prompt: str
    computation_input: List[Any]
    max_iterations: int
    plan_iteration: int

    # coder输出
    coder_output: List[str]
    
    # answer_agent_steps: 在节点中直接添加步骤
    answer_agent_steps: List[AnswerAgentStepItem] = []
    # reporter 
    reporter_instruction: str
    reporter_template: Literal["default", "text_json", "text_json_citation"]
    reporter_model_id: Optional[str] = None
    answer: FinalAnswer
    docid_to_source: dict
    # 可选：用于reporter多轮对话的历史消息（仅影响reporter）
    history_messages: Optional[List[Tuple[str, str]]] = None
    # 回答语言 locale（如 zh_CN / en_US），用于生成 answer_language_instruction
    answer_lang: Optional[str] = "zh_CN"


def create_content_gen_state_with_defaults(state: dict) -> ContentGenState:
    """创建ContentGenState实例，自动补全默认值"""
    defaults = {
        "question": "",
        "content": [],
        "content_cnt": 0,
        "model_id": "model_for_content_generation_backup",
        "reasoning": "",
        "computation_prompt": "",
        "computation_input": [],
        "max_iterations": 3,
        "plan_iteration": 0,
        "coder_output": [],
        "answer_agent_steps": [],
        "reporter_instruction": "",
        "reporter_template": "default",
        "reporter_model_id": None,
        "answer": None,
        "docid_to_source": {},
        "history_messages": None,
        "answer_lang": "zh_CN",
        "messages": [],  # MessagesState的默认字段
    }
    
    # 合并默认值和传入的state
    full_state = {**defaults, **state}
    
    return ContentGenState(**full_state)


