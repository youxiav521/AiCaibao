from langgraph.graph import START, END, StateGraph

from src.agents.content_generate_agent.utils import log_error_type_and_message

from src.base.langfuse_observe import trace_span
from .state import RetrieverState, create_retriever_state_with_defaults
from .nodes import chunk_index_retrival, summary_index_retrival, document_reading_retrival, judge_if_continue, custom_retrieval_workflow

import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
import sys
from loguru import logger

workflow = StateGraph(RetrieverState)

# 添加节点
workflow.add_node("chunk_index_retrival", chunk_index_retrival)
workflow.add_node("summary_index_retrival", summary_index_retrival)
workflow.add_node("document_reading_retrival", document_reading_retrival)
workflow.add_node("judge_if_continue", judge_if_continue)

# 设置入口点
workflow.set_entry_point("chunk_index_retrival")

# 其余路由由各节点的 Command(goto=...) 控制

# Langfuse 支持：使用 update_trace=True 以追踪完整链路
# Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler
content_generate_graph = workflow.compile()

# 创建可配置检索工作流（使用custom_retrieval_workflow）
custom_workflow = StateGraph(RetrieverState)
custom_workflow.add_node("custom_retrieval_workflow", custom_retrieval_workflow)
custom_workflow.set_entry_point("custom_retrieval_workflow")
custom_retrieval_graph = custom_workflow.compile()

class RetrivalService:
    @trace_span(name="RetrivalService.run_streaming")
    async def run_streaming(self, state: dict) -> AsyncGenerator[Dict[str, Any], None]:
        """
        运行流式检索工作流（生产者）。
        
        这是一个异步生成器，会持续产出finished_question_ids供消费者使用。
        当检测到finished_question时，会立即yield相关信息。
        
        Args:
            state: 检索工作流的初始状态
            
        Yields:
            Dict[str, Any]: 包含finished_question_ids和检索结果的数据
        """
        logger.info("开始启动流式检索工作流")
        
        # 构建完整状态
        # 使用自定义函数补全默认值，避免手动维护字段映射
        # 允许外部传入批量控制开关，默认 False
        full_state = create_retriever_state_with_defaults({
            **state,
            "use_batch_for_llm": state.get("use_batch_for_llm", False)
        })
        
        logger.info(
            f"流式检索工作流状态初始化完成，问题数量: {len(full_state['question_tasks'])}, "
            f"answer_lang={full_state.get('answer_lang')}"
        )
        
        # 用于跟踪最终状态（用于缓存摘要树）
        final_state = None
        
        try:
            # 使用astream进行流式处理
            async for step_result in content_generate_graph.astream(full_state):
                logger.info(f"收到工作流步骤结果: {list(step_result.keys())}")
                
                # 更新最终状态（合并所有步骤的更新）
                if final_state is None:
                    final_state = {}
                for node_name, node_state in step_result.items():
                    if isinstance(node_state, dict):
                        final_state.update(node_state)
                
                # 检查是否有finished_question_ids产出
                # 注：step_result只会返回本步更新的那部分state字段而不是完整的state
                for node_name, node_state in step_result.items():
                    if isinstance(node_state, dict) and "finished_question_ids" in node_state:
                        finished_ids = node_state["finished_question_ids"]
                        if finished_ids:
                            logger.info(f"检测到finished_question_ids: {finished_ids}")
                            
                            # 构建产出数据
                            yield_data = {
                                "type": "finished_questions",
                                "finished_question_ids": finished_ids,
                                "retrieval_data": node_state,  # 包含完整的检索结果
                                "node_name": node_name,
                                "timestamp": asyncio.get_event_loop().time()
                            }
                            
                            logger.info(f"产出finished_question资源: {finished_ids}")
                            yield yield_data
                            
                            # 清空finished_question_ids，避免重复产出
                            node_state["finished_question_ids"] = []
                
                # 检查工作流是否结束
                if "__end__" in step_result:
                    logger.info("检索工作流已结束")
                    break
                    
        except Exception as e:
            # 获取抛出异常的函数名称，使用更安全的方式
            try:
                import traceback
                tb_str = traceback.format_exc()
                # 尝试从 traceback 中提取函数名和位置信息
                exc_msg = f"流式检索工作流出现异常: {e}\n{tb_str}"
            except Exception:
                # 如果获取 traceback 失败，只记录异常信息
                exc_msg = f"流式检索工作流出现异常: {e}"
            logger.error(exc_msg)

            # 产出错误信息
            yield {
                "type": "error",
                "error": exc_msg,
                "timestamp": asyncio.get_event_loop().time()
            }
        finally:
            if final_state:
                yield {
                    "type": "final_state",
                    "retrieval_data": final_state,
                    "timestamp": asyncio.get_event_loop().time()
                }
        
        logger.info("流式检索工作流完成")
    
    @trace_span(name="RetrivalService.run_custom")
    async def run_custom(self, state: dict) -> dict:
        """
        运行可配置检索工作流（使用custom_retrieval_workflow）。
        
        这是一个非流式的检索工作流，直接返回所有检索结果。
        
        Args:
            state: 检索工作流的初始状态
            
        Returns:
            dict: 包含检索结果的完整状态
        """
        logger.info("开始启动可配置检索工作流")
        
        # 构建完整状态
        full_state = create_retriever_state_with_defaults({
            **state,
            "use_batch_for_llm": state.get("use_batch_for_llm", False)
        })
        
        logger.info(
            f"可配置检索工作流状态初始化完成，问题数量: {len(full_state['question_tasks'])}, "
            f"answer_lang={full_state.get('answer_lang')}"
        )
        
        try:
            # 使用ainvoke直接执行工作流
            result = await custom_retrieval_graph.ainvoke(full_state)
            logger.info("可配置检索工作流执行完成")
            return result
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="可配置检索工作流出现异常",
                e=e,
                exc_info=True,
                max_message_chars=200,
            )
            raise
