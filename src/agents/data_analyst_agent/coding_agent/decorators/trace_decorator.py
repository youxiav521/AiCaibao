from functools import wraps
from typing import Dict, Callable, Type
import inspect
import asyncio
import types
import traceback

from pydantic import BaseModel

from ..utils.code_extractor import extract_code_error
from ....agent_trace.models.coding_trace_steps import (
    CodingStepTypeEnum, 
    StartTraceStep, 
    DataTraceStep, 
    PlanTraceStep, 
    CodeTraceStep, 
    CodeErrorTraceStep, 
    AnswerTraceStep
)
from ....agent_trace.models.trace_task import BaseTraceStep
from loguru import logger

# 步骤类型映射
STEP_TYPE_MAPPING: Dict[CodingStepTypeEnum, Type[BaseTraceStep]] = {
    CodingStepTypeEnum.start: StartTraceStep,
    CodingStepTypeEnum.data: DataTraceStep,
    CodingStepTypeEnum.plan: PlanTraceStep,
    CodingStepTypeEnum.code: CodeTraceStep,
    CodingStepTypeEnum.code_error: CodeErrorTraceStep,
    CodingStepTypeEnum.answer: AnswerTraceStep,
}

def trace_step(step_type: CodingStepTypeEnum,  step_data_factory: Callable, **default_kwargs):
    """
    追踪步骤装饰器
    Args:
        step_type: 步骤类型枚举
        step_data_factory: 用于生成特定字段的函数
        **default_kwargs: 默认参数值
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 执行原函数
            result = func(*args, **kwargs)
            result = _trace_after_call(func, args, kwargs, result, step_type, step_data_factory, default_kwargs)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 异步执行原函数
            result = await func(*args, **kwargs)
            result = _trace_after_call(func, args, kwargs, result, step_type, step_data_factory, default_kwargs)
            return result

        def _safe_call_factory(step_data_factory, bound_args, result, step_type_value):
            """安全调用 step_data_factory，失败时返回空字典"""
            try:
                return step_data_factory(bound_args, result)
            except Exception as e:
                logger.warning(f"step_data_factory failed for {step_type_value}: {e}, using default data only")
                return {}

        def _safe_create_step(step_type, step_data, trace_steps, step_type_value):
            """安全创建 step 对象，失败时返回 None"""
            try:
                step_class = STEP_TYPE_MAPPING[step_type]
                step = step_class(**step_data)
                trace_steps.append(step)
                return True
            except Exception as e:
                logger.warning(f"Failed to create step {step_type_value}: {e}, skip this trace step")
                return False

        def _trace_after_call(func, args, kwargs, result, step_type, step_data_factory, default_kwargs):
            """
            追踪函数调用后的逻辑，包裹在异常处理中，确保追踪失败不影响主流程
            """
            try:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                logger.info(f"---Trace step---: 记录溯源步骤 {step_type.value} ")
                if 'state' not in bound_args.arguments:
                    return result

                trace_steps = bound_args.arguments['state'].get("trace_steps")
                if trace_steps is None:
                    trace_steps = []
                elif isinstance(trace_steps, tuple):
                    trace_steps = list(trace_steps)

                step_id = len(trace_steps) + 1
                next_step = step_id + 1
                step_data = {
                    'step_id': step_id,
                    'next_step': next_step,
                    **default_kwargs
                }

                factory_data = _safe_call_factory(step_data_factory, bound_args, result, step_type.value)
                step_data.update(factory_data)

                if not _safe_create_step(step_type, step_data, trace_steps, step_type.value):
                    return result

                return _merge_result_with_trace_steps(result, trace_steps)

            except Exception as e:
                logger.warning(f"Trace step failed for {step_type.value}: {e}, returning original result without trace")
                logger.debug(f"Trace step error details: {traceback.format_exc()}")
                return result

        def _safe_copy_model(result, trace_steps):
            """安全地复制 BaseModel 对象并添加 trace_steps，失败时返回 None"""
            try:
                model_fields = getattr(result, "model_fields", None)
                if isinstance(model_fields, dict) and ("trace_steps" in model_fields):
                    return result.model_copy(update={"trace_steps": trace_steps})
            except Exception as e:
                logger.warning(f"Failed to copy model with trace_steps: {e}")
            return None

        def _merge_result_with_trace_steps(result, trace_steps:list):
            """
            将 trace_steps 合并到原函数返回值中。
            如果合并失败，返回原结果，确保不影响主流程。
            """
            try:
                if result is None:
                    return {"trace_steps": trace_steps}
                if isinstance(result, dict):
                    merged = dict(result)
                    merged["trace_steps"] = trace_steps
                    return merged
                if isinstance(result, BaseModel):
                    copied_model = _safe_copy_model(result, trace_steps)
                    if copied_model is not None:
                        return copied_model
                    return {"result": result, "trace_steps": trace_steps}
                return {"result": result, "trace_steps": trace_steps}
            except Exception as e:
                logger.warning(f"merge trace_steps into result failed: {e}, returning original result")
                logger.debug(f"Merge error details: {traceback.format_exc()}")
                return result
        if is_async:
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

# 便捷装饰器（需要手动传入steps列表），每个装饰器负责添加自己的字段
def trace_start(**kwargs):
    """追踪开始步骤"""
    def factory(bound_args, result):
        return {
            'original_input': bound_args.arguments['state'].get('original_input', {})
        }
    return trace_step(CodingStepTypeEnum.start, factory, **kwargs)

def trace_data(**kwargs):
    """追踪数据提取步骤"""
    def factory(bound_args, result):
        # 决定用什么保存后的数据返回
        saved_info = result.get('saved_info', [])
        # INSERT_YOUR_CODE
        # 从saved_info这个list中抽取所有object的dataframe，作为list返回
        dataframes = []
        for item in saved_info:
            dataframe = item.get("dataframe")
            if dataframe is not None:
                dataframes.append(dataframe)
        return {
            'input_data': bound_args.arguments['state'].get('relative_documents_list', []),
            'saved_data': dataframes
        }
    return trace_step(CodingStepTypeEnum.data, factory, **kwargs)

def trace_plan(**kwargs):
    """追踪计划步骤"""
    def factory(bound_args, result):
        return {
            'need_calculate_metrics': result.get('need_calculate_metrics', {})
        }
    return trace_step(CodingStepTypeEnum.plan, factory, **kwargs)

def trace_code(**kwargs):
    """追踪代码步骤"""
    def factory(bound_args, result):
        last_execution_code = result.get('last_execution_code')
        code_execution_results = result.get('code_execution_results', [])
        last_message = result.get('messages')[-1]
        if hasattr(last_message, 'content') and "Error-出现错误" in last_message.content:
            success = False
        else:
            success = True
        
        code = last_execution_code.imports+'\n'+last_execution_code.code
        code_description = last_execution_code.description
        return {
            'code': code,
            'success': success,
            'code_description': code_description,
            'code_execution_results': code_execution_results
        }
    return trace_step(CodingStepTypeEnum.code, factory, **kwargs)

def trace_code_error(**kwargs):
    """追踪代码错误步骤"""
    def factory(bound_args, result):
        last_execution_code = bound_args.arguments['state'].get('last_execution_code')
        # todo error信息还没有溯源
        # code_execution_result = bound_args.arguments['state'].get('code_execution_result', [])
        suggestion = result.get('messages')[-1].content

        code = last_execution_code.imports+'\n'+last_execution_code.code
        error = "错误原文：\n"+extract_code_error(result.get('messages')[-2].content)
        return {
            'code': code,
            'error': error,
            'suggestion': suggestion
        }
    return trace_step(CodingStepTypeEnum.code_error, factory, **kwargs)

def trace_answer(**kwargs):
    """追踪答案步骤"""
    def factory(bound_args, result):
        answer = result.get('messages')[-1].content
        return {
            'answer': answer,
            'reseaning': bound_args.arguments['state'].get('answer', '')
        }
    return trace_step(CodingStepTypeEnum.answer, factory, **kwargs)
