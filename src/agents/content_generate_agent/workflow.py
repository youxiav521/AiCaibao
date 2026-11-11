from langgraph.graph import START, END, StateGraph
from .state import ContentGenState, create_content_gen_state_with_defaults
from .nodes import planner, coder, reporter
import time
import asyncio
from typing import Optional
from loguru import logger

workflow = StateGraph(ContentGenState)

# 添加节点
workflow.add_node("planner", planner)
workflow.add_node("coder", coder)
workflow.add_node("reporter", reporter)

# 设置入口点
workflow.set_entry_point("planner")

# 其余路由由各节点的 Command(goto=...) 控制

# Langfuse 支持：使用 update_trace=True 以追踪完整链路
# Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler
content_generate_graph = workflow.compile()

class ContentGenService:

    @classmethod
    async def run_question(cls, state: dict) -> dict:
        """运行工作流：接收外部 state，补全默认值，避免缺字段 KeyError。

        同时记录每个节点（planner, coder, reporter）的执行时间。
        """
        # 使用自定义函数补全默认值，避免手动维护字段映射
        full_state = create_content_gen_state_with_defaults(state)

        # 使用 astream 追踪每个节点的执行时间
        answer_timings = {}
        result_state = None
        step_start_time = time.perf_counter()

        async for step_result in content_generate_graph.astream(full_state):
            for node_name, node_state in step_result.items():
                elapsed = time.perf_counter() - step_start_time
                answer_timings[f"answer_{node_name}"] = elapsed
                logger.info(f"答案生成节点 {node_name} 完成，耗时: {elapsed:.3f}s")
                result_state = node_state
                step_start_time = time.perf_counter()

        # 将耗时信息合并到结果中
        if result_state is not None:
            result_state["answer_timings"] = answer_timings
        else:
            # 如果 astream 没有产出（极端情况），回退到 ainvoke
            result_state = await content_generate_graph.ainvoke(full_state)

        return result_state

