
import code
from typing import Any, Union
from unittest import result
from pydantic import Field

from .trace_task import BaseStepTypeEnum, BaseTraceStep

class CodingStepTypeEnum(BaseStepTypeEnum):
    """Coding流程类型Enum"""
    start = "start"
    data = "data"
    plan = "plan"
    code = "code"
    code_error = "code_error"
    answer = "answer"

class StartTraceStep(BaseTraceStep):
    original_input: Any
    type: CodingStepTypeEnum = Field(default=CodingStepTypeEnum.start, Literal=True)

class DataTraceStep(BaseTraceStep):
    input_data: Any
    saved_data: Any
    type: CodingStepTypeEnum = Field(default=CodingStepTypeEnum.data, Literal=True)

class PlanTraceStep(BaseTraceStep):
    need_calculate_metrics: dict
    type: CodingStepTypeEnum = Field(default=CodingStepTypeEnum.plan, Literal=True)

class CodeTraceStep(BaseTraceStep):
    code: str
    code_description: str
    success: bool
    code_execution_results: list[str]
    type: CodingStepTypeEnum = Field(default=CodingStepTypeEnum.code, Literal=True)

class CodeErrorTraceStep(BaseTraceStep):
    code:str
    error:str
    suggestion: str 
    type: CodingStepTypeEnum = Field(default=CodingStepTypeEnum.code_error, Literal=True)

class AnswerTraceStep(BaseTraceStep):
    answer: str
    reseaning: str
    type: CodingStepTypeEnum = Field(default=CodingStepTypeEnum.answer, Literal=True)

CodingTraceStep=Union[StartTraceStep, DataTraceStep, PlanTraceStep, CodeTraceStep, CodeErrorTraceStep, AnswerTraceStep]