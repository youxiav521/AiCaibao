from enum import Enum
from typing import Any, Sequence
from pydantic import BaseModel

class BaseStepTypeEnum(str, Enum):
    """基类型Enum，可被子类继承并扩展"""
    pass

class BaseTraceStep(BaseModel):
    step_id: int
    type: BaseStepTypeEnum
    next_step: int

class TraceTask(BaseModel):
    trace_id: str
    original_input: Any
    trace_steps:Sequence[Any]
    final_output: Any

