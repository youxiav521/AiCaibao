from typing import Any, Sequence

from ..models.trace_task import TraceTask,BaseTraceStep,BaseStepTypeEnum
import random
import string
import time
from loguru import logger


def order_trace_steps(trace_steps: Sequence[Any]) -> Sequence[Any]:
    # 以 step_id 建立映射
    step_map = {step.step_id: step for step in trace_steps}
    # 统计所有 next_step
    next_step_ids = set(step.next_step for step in trace_steps if step.next_step is not None and step.next_step != -1)
    # 查找第一个 step: 它的 step_id 不被任何 step 的 next_step 指向（即链表的头部）
    start_candidates = [step for step in trace_steps if step.step_id not in next_step_ids]
    if not start_candidates:
        # 如果没有头部，直接返回原顺序
        return trace_steps
    start_step = start_candidates[0]

    # 按照 next_step 组装链表
    ordered_steps = []
    current = start_step
    visited = set()
    while current and current.step_id not in visited:
        ordered_steps.append(current)
        visited.add(current.step_id)
        next_step_id = current.next_step
        if next_step_id is None or next_step_id == -1:
            break
        current = step_map.get(next_step_id, None)
    return ordered_steps

def form_trace_task(original_input: dict, trace_steps: Sequence[Any], final_output: dict) -> TraceTask:
    """生成唯一的trace_id，格式为trace_XXXXXX+时间戳。"""
    prefix = "trace_"
    suffix = ''.join(random.choices(string.digits, k=6))
    timestamp = str(int(time.time() * 1000))
    trace_id = f"{prefix}{suffix}{timestamp}"
    # trace_steps = order_trace_steps(trace_steps)
    # 确保trace_steps中的每一项均为BaseTraceStep或其子类的实例
    processed_trace_steps = []
    # logger.info(f"before processed_trace_steps, trace_steps_type:{type(trace_steps)}, trace_steps:{trace_steps}")
    for step in trace_steps:
        # isinstance检查本类及子类
        # logger.info(f"form_trace_task: step_type:{type(step)}, step:{step}")
        if isinstance(step, BaseTraceStep):
            # logger.info(f"form_trace_task: step is BaseTraceStep, step_type:{type(step)}, step:{step}")
            processed_trace_steps.append(step)
        else:
            raise TypeError(f"trace_steps中的元素必须为BaseTraceStep及其子类实例, 但收到: {type(step)}")
    trace_steps = processed_trace_steps
    return TraceTask(trace_id=trace_id, original_input=original_input, trace_steps=trace_steps, final_output=final_output)



def reform_trace_steps(trace_task: TraceTask) -> TraceTask:
    """
    遍历trace_task中的trace_steps，修改字段
    将不同类型的step转换为统一的数据结构
    plan -> analysis, code -> code, code_error -> error, answer -> answer
    做好错误捕获和处理，出错时日志warn不抛到上层
    返回修改后的TraceTask，其中trace_steps字段已被转换为字典列表
    """
    from ..models.coding_trace_steps import (
        PlanTraceStep, 
        CodeTraceStep, 
        CodeErrorTraceStep, 
        AnswerTraceStep,
        CodingStepTypeEnum
    )
    import traceback

    reformed_steps = []
    
    for idx, step in enumerate(getattr(trace_task, "trace_steps", [])):
        try:
            # 跳过 start 和 data 类型的步骤
            if hasattr(step, "type") and step.type in [CodingStepTypeEnum.start, CodingStepTypeEnum.data]:
                continue

            if isinstance(step, PlanTraceStep):
                # plan 转换为 analysis
                items = []
                need_calculate_metrics = getattr(step, "need_calculate_metrics", None)
                if need_calculate_metrics and isinstance(need_calculate_metrics, dict):
                    # need_calculate_metrics 结构: {"keypoints": "", "metrics": [...], "limitations": ""}
                    metrics_list = need_calculate_metrics.get("metrics", [])
                    if isinstance(metrics_list, list):
                        for metric in metrics_list:
                            if isinstance(metric, dict):
                                items.append({
                                    "type": "metrics",
                                    "text": metric.get("name", ""),
                                    "description": metric.get("description", ""),
                                    "formula": metric.get("formula", "")
                                })
                
                # 构建 text，包含 keypoints 信息
                if need_calculate_metrics and isinstance(need_calculate_metrics, dict):
                    keypoints = need_calculate_metrics.get("keypoints", "")
                    text = keypoints
                else:
                    text = ""

                reformed_steps.append({
                    "type": "analysis",
                    "text": text,
                    "items": items
                })

            elif isinstance(step, CodeTraceStep):
                # code 转换为 code
                reformed_steps.append({
                    "type": "code",
                    "text": "Python代码",
                    "description": getattr(step, "code_description", "") or "",
                    "code": getattr(step, "code", "") or ""
                })

            elif isinstance(step, CodeErrorTraceStep):
                # code_error 转换为 error
                reformed_steps.append({
                    "type": "error",
                    "text": getattr(step, "suggestion", "") or ""
                })

            elif isinstance(step, AnswerTraceStep):
                # answer 转换为 answer
                reformed_steps.append({
                    "type": "answer",
                    "text": getattr(step, "answer", "") or ""
                })

            else:
                # 如果不是已知类型，直接跳过
                continue

        except Exception as e:
            logger.warning(f"Error occurred while reformatting step at idx={idx}: {e}\n{traceback.format_exc()}")
            continue  # 错误时跳过，保证不会抛到上层

    # 创建新的TraceTask实例，更新trace_steps字段
    return TraceTask(
        trace_id=trace_task.trace_id,
        original_input=trace_task.original_input,
        trace_steps=reformed_steps,
        final_output=trace_task.final_output
    )