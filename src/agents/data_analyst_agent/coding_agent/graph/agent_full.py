import json
import time
import traceback
from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from src.agents.data_analyst_agent.coding_agent.graph.extract_graph.extract_agent import extract_agent_workflow
from src.agents.data_analyst_agent.coding_agent.models.document_model import Document

from ....agent_trace.utils.trace_task_utils import form_trace_task, reform_trace_steps
from .planning_graph.planning_agent import planning_agent_workflow
from .coding_graph.coding_agent import coding_agent_workflow
from .answering_graph.answering_agent import answering_agent_workflow
from .state_former.state_former import form_coding_state, form_answering_state, form_extract_state, form_planning_state
from ..config.invoke_model_config import model_invoke_config
from ..utils.table_saver_utils import cleanup_saved_tables
from .....utils.langfuse_tracing import trace_span
from ..utils.token_utils import (
    count_doc_list_tokens,
    split_doc_list_by_token_limit,
    DEFAULT_TOKEN_LIMIT,
    DEFAULT_PARTITION_TOKEN_VALUE,
)
from src.agents.content_generate_agent.utils import get_localized_message, normalize_locale
from loguru import logger
from src.utils.metric_monitoring import get_monitoring


class AgentFullService:
    """
    组合 Agent 服务类，提供完整的 planning + coding 工作流
    """

    def _messages_tuples_to_any_messages(
        self, messages: list[tuple[str, str]]
    ) -> list[AnyMessage]:
        """
        将 list[(role, content)] 转为 list[AnyMessage]。
        role 支持: user/human -> HumanMessage, assistant/ai -> AIMessage，其余视为 HumanMessage。
        """
        result: list[AnyMessage] = []
        for role, content in messages:
            r = (role or "").strip().lower()
            if r in ("user", "human"):
                result.append(HumanMessage(content=content or ""))
            elif r in ("assistant", "ai"):
                result.append(AIMessage(content=content or ""))
            else:
                result.append(HumanMessage(content=content or ""))
        return result

    @trace_span(name="CodingAgent.run_full_analysis_optional")
    async def run_full_analysis_optional(
        self,
        question: str,
        prompt: Optional[str] = None,
        documents: Optional[List[dict]] = [],
        need_calculate_metrics: Optional[Dict[str, Any]] = {},
        disable_planning: Optional[bool] = False,
        disable_extract: Optional[bool] = False,
        model_id: Optional[str] = "data_analyst_agent",
        history_messages: Optional[list[AnyMessage]] = None,
        max_iterations: Optional[int] = 5,
        answer_language: Optional[str] = "zh_CN"
    ):
        """
        支持 planning、extract 两个子流程分别独立关闭的完整链路入口。
        """
        answer_language = normalize_locale(answer_language)
        time_start = time.time()
        # 与 answer_table_agent.answer_single_table 一致：直接使用 list[AnyMessage]，无则用空列表
        history_messages_normalized: list[AnyMessage] = list(history_messages) if history_messages else []

        # 保存当前的model_id，在函数结束时恢复
        original_model_id = model_invoke_config.get_model_id()
        # 跟踪所有环节生成的临时文件信息，确保异常时也能清理
        last_saved_info = []
        code_generation_count = 0
        code_error_count = 0
        try:
            model_invoke_config.set_model_id(model_id)

            # 将传入的 documents 转为 Document 列表，state 中直接存 List[Document]
            doc_list = [Document.from_dict(d) for d in (documents or [])]
            logger.info(
                f"---run_full_analysis_optional---: \n"
                f"入参调试: question={question[:100] if question else ''}..., \n"
                f"prompt={prompt[:100] if prompt else 'None'}..., \n"
                f"documents_count={len(documents) if documents else 0}, \n"
                f"documents={[(doc.id,doc.title,doc.text[:100],doc.table[:100]) for doc in doc_list] if documents else []}, \n"
                f"need_calculate_metrics={need_calculate_metrics}, \n"
                f"disable_planning={disable_planning}, \n"
                f"disable_extract={disable_extract}, \n"
                f"model_id={model_id}, \n"
                f"history_messages_count={len(history_messages) if history_messages else 0}, \n"
                f"max_iterations={max_iterations}, \n"
                f"answer_language={answer_language}"
            )
            # 初始状态（含 history_messages，供 planning/extract/coding 传递并在 coding 节点使用）
            state = {
                "question_description": question,
                "relative_documents_list": doc_list,
                "user_prompt": prompt,
                "need_calculate_metrics": need_calculate_metrics,
                "history_messages": history_messages_normalized,
                "max_iterations": max_iterations,
                "answer_language": answer_language
            }

            # 运行 planning 环节
            if not disable_planning:
                logger.info("---run_full_analysis_optional---: planning enabled")
                planning_state = form_planning_state(state)
                planning_result = await planning_agent_workflow.ainvoke(planning_state)
                # 覆盖最新的 state
                state = planning_result
                # 规划获得的 latest need_calculate_metrics
                need_calculate_metrics = planning_result.get("need_calculate_metrics", {})

            # 运行 extract 环节
            if not disable_extract:
                logger.info("---run_full_analysis_optional---: extract enabled")
                extract_state = form_extract_state(state)
                extract_result = await extract_agent_workflow.ainvoke(extract_state)
                # 覆盖最新的 state
                state = extract_result

            # 运行 coding 环节
            coding_state = form_coding_state(state)
            coding_result = await coding_agent_workflow.ainvoke(coding_state)
            
            code_execution_results = coding_result.get("code_execution_results", [])
            last_execution_code = coding_result.get("last_execution_code", "")
            # 统计代码生成次数（根据迭代次数）
            code_generation_count = coding_result.get("iteration", 0)
            # 跟踪最新的 saved_info
            if "saved_info" in coding_result:
                last_saved_info = coding_result["saved_info"]

            # 运行 answering 环节
            answering_state = form_answering_state(coding_result)
            answering_result = await answering_agent_workflow.ainvoke(answering_state)
            
            last_message = answering_result['messages'][-1].content
            trace_steps = answering_result.get("trace_steps", [])
            # answering 环节已在内部删除临时文件，清空跟踪列表避免重复删除
            last_saved_info = []

            time_end = time.time()
            time_cost = time_end - time_start

            # 直接使用 answering_result 中的累计指标
            total_model_call_count = answering_result.get("model_call_count", 0)
            total_model_call_times = answering_result.get("model_call_times_ms", [])
            code_error_count = answering_result.get("code_error_count", 0)
            code_errors = answering_result.get("code_errors", [])

            trace_task = reform_trace_steps(form_trace_task(
                "",
                trace_steps,
                {
                    "result": last_message,
                    "time_cost": time_cost if time_cost else 0
                }
            ))

            return {
                "result": last_message,
                "code_execution_result": code_execution_results,
                "need_calculate_metrics": need_calculate_metrics,
                "last_execution_code": last_execution_code,
                "time_cost": time_cost if time_cost else 0,
                "trace_task": trace_task,
                "metrics": {
                    "code_generation_count": code_generation_count,
                    "code_error_count": code_error_count,
                    "code_errors": code_errors,
                    "model_call_count": total_model_call_count,
                    "model_call_times_ms": total_model_call_times
                }
            }
        except Exception as e:
            # 统一处理大模型调用异常
            error_msg = str(e)
            # 打印堆栈
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            if self._is_model_api_error(e, error_msg):
                logger.error(f"调用大模型时发生异常: {error_msg}")
                return self._get_error_response(time_start, answer_language)
            return self._get_error_response(time_start, answer_language)
        finally:
            # 清理所有生成的临时表格文件（确保异常时也能删除）
            cleanup_saved_tables(last_saved_info)
            # 恢复原来的model_id，确保不影响后续请求
            model_invoke_config.set_model_id(original_model_id)

    @trace_span(name="CodingAgent.run_full_analysis_with_msg")
    async def run_full_analysis_with_msg(
        self,
        question: str,
        history_messages: list[tuple[str, str]],
        prompt: Optional[str] = None,
        documents: Optional[List[dict]] = None,
        need_calculate_metrics: Optional[Dict[str, Any]] = None,
        disable_planning: Optional[bool] = False,
        disable_extract: Optional[bool] = False,
        model_id: Optional[str] = "data_analyst_agent",
        max_iterations: Optional[int] = 5,
        answer_language: Optional[str] = "zh_CN"
    ):
        """
        带历史消息的完整分析：接收 list[(role, content)]，转换为 AnyMessage 后调用 run_full_analysis_optional。
        """
        if documents is None:
            documents = []
        if need_calculate_metrics is None:
            need_calculate_metrics = {}
        any_messages = self._messages_tuples_to_any_messages(history_messages)
        return await self.run_full_analysis_optional(
            question=question,
            prompt=prompt,
            documents=documents,
            need_calculate_metrics=need_calculate_metrics,
            disable_planning=disable_planning,
            disable_extract=disable_extract,
            model_id=model_id,
            history_messages=any_messages,
            max_iterations=max_iterations,
            answer_language=answer_language
        )

    # @trace_span(name="CodingAgent.run_full_analysis_with_token_limit")
    async def run_full_analysis_with_token_limit(
        self,
        question: str,
        prompt: Optional[str] = None,
        documents: Optional[List[dict]] = [],
        need_calculate_metrics: Optional[Dict[str, Any]] = {},
        disable_planning: Optional[bool] = False,
        disable_extract: Optional[bool] = False,
        model_id: Optional[str] = "data_analyst_agent",
        history_messages: Optional[list[tuple[str, str]]] = None,
        token_limit: Optional[int] = None,
        max_iterations: Optional[int] = 5,
        answer_language: Optional[str] = "zh_CN"
    ):
        """
        在 token 上限约束下，以滑动窗口方式调用 run_full_analysis_optional。

        滑动窗口策略：
        1. 第一个窗口：使用原始问题，完整执行 planning + extract + coding + answering
        2. 后续窗口：基于上一窗口结果"继续分析"，沿用前面窗口的 need_calculate_metrics
        3. 所有窗口完成后，返回最后一个窗口的累积分析结果
        """
        monitor = get_monitoring()
        with monitor.track_metric_context("coding_agent_full_analysis") as tracker:
            time_start = time.time()
            answer_language = normalize_locale(answer_language)
            model_call_count = 0
            code_generation_count = 0
            code_error_count = 0
            model_call_times = []
            code_errors = []
            try:
                # 使用默认 token 上限与划分值
                partition_token_value = None
                if token_limit is None:
                    token_limit = DEFAULT_TOKEN_LIMIT
                    partition_token_value = DEFAULT_PARTITION_TOKEN_VALUE
                else:
                    partition_token_value = int(token_limit * 4 / 5)

                # 将历史消息转为 list[AnyMessage]
                history_messages_list: list[AnyMessage] = (
                    self._messages_tuples_to_any_messages(history_messages)
                    if history_messages
                    else []
                )

                # 转为 Document 列表并按 token 限制分块
                doc_list = [Document.from_dict(d) for d in (documents or [])]
                chunks = split_doc_list_by_token_limit(
                    doc_list,
                    token_limit=token_limit,
                    partition_token_value=partition_token_value,
                )

                # 当前累积的 need_calculate_metrics（窗口间传递）
                current_metrics: Dict[str, Any] = dict(need_calculate_metrics) if need_calculate_metrics else {}
                last_result = None

                for i, chunk in enumerate(chunks):
                    chunk_documents = [doc.to_dict() for doc in chunk]
                    is_first_window = (i == 0)

                    logger.info(
                        f"---run_full_analysis_with_token_limit---: "
                        f"窗口 {i + 1}/{len(chunks)}，本块文档数 {len(chunk)}，"
                        f"本块 token 约 {count_doc_list_tokens(chunk)},"
                        f"using model_id:{model_id}, max_iterations:{max_iterations}"
                    )

                    # 若不是第一个窗口，将上一窗口结果作为历史消息加入，并调整问题引导"继续分析"
                    if not is_first_window and last_result is not None:
                        prev_answer = last_result.get("result", "") or ""
                        if prev_answer:
                            # 将上一窗口结果加入历史消息，供当前窗口模型参考
                            prev_content = get_localized_message(
                                "analysis_previous_window_result",
                                answer_language,
                                answer=prev_answer,
                            )
                            history_messages_list.append(HumanMessage(content=prev_content))

                        # 沿用前面窗口的 need_calculate_metrics，避免重复规划
                        current_metrics = last_result.get("need_calculate_metrics", current_metrics)

                    # 执行分析
                    last_result = await self.run_full_analysis_optional(
                        question=question,
                        prompt=prompt,
                        documents=chunk_documents,
                        need_calculate_metrics=current_metrics,
                        # 仅第一个窗口启用 planning（如果未禁用），后续窗口沿用前面的规划结果
                        disable_planning=disable_planning if is_first_window else True,
                        disable_extract=disable_extract,
                        model_id=model_id,
                        history_messages=history_messages_list,
                        max_iterations=max_iterations,
                        answer_language=answer_language
                    )
                    
                    # 从结果中获取性能指标
                    metrics = last_result.get("metrics", {})
                    code_generation_count += metrics.get("code_generation_count", 0)
                    code_error_count += metrics.get("code_error_count", 0)
                    code_errors.extend(metrics.get("code_errors", []))
                    model_call_count += metrics.get("model_call_count", 0)
                    model_call_times.extend(metrics.get("model_call_times_ms", []))
                    
                    # 打印详细的性能指标
                    logger.info(f"---run_full_analysis_with_token_limit---: 窗口 {i + 1} 性能指标:")
                    logger.info(f"  代码生成次数: {metrics.get('code_generation_count', 0)}")
                    logger.info(f"  代码错误次数: {metrics.get('code_error_count', 0)}")
                    logger.info(f"  代码错误信息: {str(code_errors)[:50] if code_errors else ''}")
                    logger.info(f"  模型调用次数: {metrics.get('model_call_count', 0)}")
                    logger.info(f"  模型调用时间: {metrics.get('model_call_times_ms', [])}")

                    # 若某次调用返回错误，继续下一窗口
                    if last_result and last_result.get("error"):
                        logger.warning(f"窗口 {i + 1} 调用返回 error，继续下一窗口")

                # 无任何窗口（doc_list 为空）时仍走一次原方法以保持行为一致
                if last_result is None:
                    # 执行分析
                    last_result = await self.run_full_analysis_optional(
                        question=question,
                        prompt=prompt,
                        documents=[],
                        need_calculate_metrics=current_metrics,
                        disable_planning=disable_planning,
                        disable_extract=disable_extract,
                        model_id=model_id,
                        history_messages=history_messages_list,
                        max_iterations=max_iterations,
                        answer_language=answer_language
                    )
                    
                    # 从结果中获取性能指标
                    metrics = last_result.get("metrics", {})
                    code_generation_count += metrics.get("code_generation_count", 0)
                    code_error_count += metrics.get("code_error_count", 0)
                    model_call_count += metrics.get("model_call_count", 0)
                    model_call_times.extend(metrics.get("model_call_times_ms", []))
                    
                    # 打印详细的性能指标
                    logger.info("---run_full_analysis_with_token_limit---: 空文档列表性能指标:")
                    logger.info(f"  代码生成次数: {metrics.get('code_generation_count', 0)}")
                    logger.info(f"  代码错误次数: {metrics.get('code_error_count', 0)}")
                    logger.info(f"  模型调用次数: {metrics.get('model_call_count', 0)}")
                    logger.info(f"  模型调用时间: {metrics.get('model_call_times_ms', [])}")

                # 计算总耗时
                time_end = time.time()
                total_duration = (time_end - time_start) * 1000

                # 记录指标和错误信息
                if tracker:
                    tracker.set_metric("total_duration_ms", total_duration)
                    tracker.set_metric("model_call_count", model_call_count)
                    tracker.set_metric("code_error_count", code_error_count)
                    tracker.set_metric("model_call_times_ms", model_call_times)
                    tracker.set_metric("window_count", len(chunks))
                    # 将code_errors转为字符串列表，避免langfuse序列化时丢失详细数据
                    # code_errors_str = [str(error) for error in code_errors]
                    logger.info(f"about to set code_errors:{str(code_errors)[:50]}")
                    tracker.set_event_info("code_errors", code_errors)
                    # tracker.set_metric("code_errors", code_errors)

                return last_result
            except Exception as e:
                error_msg = str(e)
                logger.error(f"run_full_analysis_with_token_limit 发生异常: {error_msg}")
                logger.error(f"堆栈: {traceback.format_exc()}")
                
                # 记录异常指标
                if tracker:
                    tracker.set_metric("error", error_msg)
                    tracker.set_metric("exception_type", type(e).__name__)
                
                return self._get_error_response(time_start, answer_language)

    def _is_model_api_error(self, exception: Exception, error_msg: str) -> bool:
        """
        判断是否为大模型API调用相关的异常
        Args:
            exception: 异常对象
            error_msg: 异常消息字符串
        Returns:
            bool: 如果是大模型API相关异常返回True，否则返回False
        """
        error_msg_lower = error_msg.lower()
        
        # 检查400错误：输入长度超出范围
        if "400" in error_msg or "invalidparameter" in error_msg_lower:
            if "range of input length" in error_msg_lower or "input length should be" in error_msg_lower:
                return True
        
        # 检查流式处理异常：不支持的消息类型
        if "unsupported message type" in error_msg_lower:
            return True
        
        # 检查其他常见的API错误
        if any(keyword in error_msg_lower for keyword in [
            "invalid_request_error",
            "api error",
            "model error",
            "rate limit",
            "timeout",
            "connection error"
        ]):
            return True
        
        return False

    def _get_error_response(self, time_start: float, answer_language: str = "zh_CN") -> Dict[str, Any]:
        """
        生成错误响应信息
        Args:
            time_start: 开始时间
        Returns:
            Dict: 包含错误信息的响应字典
        """
        time_end = time.time()
        time_cost = time_end - time_start
        
        error_message = get_localized_message("model_service_error", answer_language)
        
        return {
            "result": error_message,
            "code_execution_result": [],
            "last_execution_code": "",
            "time_cost": time_cost if time_cost else 0,
            "trace_task": None,
            "error": True
        }
