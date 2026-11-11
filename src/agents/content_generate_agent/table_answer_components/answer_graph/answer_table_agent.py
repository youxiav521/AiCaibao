from typing import List, Optional
import asyncio

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
import traceback

from typing import AsyncGenerator
from src.agents.content_generate_agent.models import ContentGenerateResponseEvent, FinalAnswer, QuestionTask
from src.agents.content_generate_agent.utils import (
    ensure_retrieval_queries_for_tasks,
    get_localized_message,
    normalize_locale,
)
from src.agents.content_generate_agent.input_models import Item
from src.agents.content_generate_agent.table_answer_components.answer_graph.answer_table_state import FillResult, TableAnswerState
from src.agents.content_generate_agent.table_answer_components.answer_graph.answer_table_workflow import (
    table_entry_node,
    table_retrieval_node,
    table_retrieval_data_sources_node,
    table_fill_node,
    table_filter_node,
    should_skip_fill_node,
    should_use_data_sources_retrieval,
    table_output_node,
)
from src.base.langfuse_observe import trace_span
from loguru import logger

answer_table_graph = StateGraph(TableAnswerState)

# 添加节点
answer_table_graph.add_node("table_entry_node", table_entry_node)
answer_table_graph.add_node("table_retrieval_node", table_retrieval_node)
answer_table_graph.add_node("table_retrieval_data_sources_node", table_retrieval_data_sources_node)
answer_table_graph.add_node("table_filter_node", table_filter_node)
answer_table_graph.add_node("table_fill_node", table_fill_node)
answer_table_graph.add_node("table_output_node", table_output_node)

# 连接节点
answer_table_graph.add_edge(START, "table_entry_node")
# 条件边在 workflow 中实现：根据 data_sources 是否含 ChunkIndexRetrievalTool 选择检索节点
answer_table_graph.add_conditional_edges(
    "table_entry_node",
    should_use_data_sources_retrieval,
    {
        "table_retrieval_data_sources_node": "table_retrieval_data_sources_node",
        "table_retrieval_node": "table_retrieval_node",
    },
)
answer_table_graph.add_edge("table_retrieval_node", "table_filter_node")
answer_table_graph.add_edge("table_retrieval_data_sources_node", "table_filter_node")
# 使用条件边：根据过滤结果决定是执行 fill 节点还是直接到 output 节点
answer_table_graph.add_conditional_edges(
    "table_filter_node",
    should_skip_fill_node,
    {
        "table_fill_node": "table_fill_node",
        "table_output_node": "table_output_node"
    }
)
answer_table_graph.add_edge("table_fill_node", "table_output_node")
answer_table_graph.add_edge("table_output_node", END)

# Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler
config = {"recursion_limit": 50}
answer_table_agent_workflow = answer_table_graph.compile().with_config(
    config=config
)

class AnswerTableAgentService:

    @trace_span(name="AnswerTableAgentService.answer_single_table_question")
    async def answer_single_table(
        self,
        typed_items: List[Item],
        question_task: QuestionTask,
        model_id: Optional[str] = "model_for_content_generation_bailian",
        disable_keyword_extract: bool = True,
        history_messages: Optional[list[AnyMessage]] = None,
        max_iterations: Optional[int] = 5,
        answer_lang: str = "zh_CN",
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        """
        回答单个表格类型问题，返回事件流（通常只yield一次）。
        """
        answer_lang = normalize_locale(answer_lang)
        try:
            # 检查并转换 question_task，确保它是 QuestionTask 对象
            # 由于可能从外部传入字典格式，需要统一转换为对象
            if isinstance(question_task, dict):
                logger.info(f"question_task 为字典类型，尝试转换为 QuestionTask 对象")
                question_task = QuestionTask.model_validate(question_task)
            elif not isinstance(question_task, QuestionTask):
                logger.warning(f"question_task 类型异常: {type(question_task)}，尝试转换")
                question_task = QuestionTask.model_validate(question_task)
            
            # 构造每个问题的 state（可选传入历史 history_messages；state 的 messages 键由 MessagesState 定义）
            state = TableAnswerState(
                content_list=typed_items,
                question_task=question_task,
                model_id=model_id,
                disable_keyword_extract=disable_keyword_extract,
                messages=history_messages if history_messages is not None else [],
                max_iterations=max_iterations,
                answer_lang=answer_lang,
            )
            result = await answer_table_agent_workflow.ainvoke(state)
            merged_raw = result.get("merged_table_result", "")
            question_citations = result.get("question_citations", [])
            origin_table_html = question_task.html or ""
            if question_task.type != "table":
                raise ValueError(f"question_task的type必须为'table'，当前为: {question_task.type}")
            # 统一用 FillResult 解析（兼容 str/dict），避免下游对类型分支判断
            fill_result = FillResult.from_raw(merged_raw, locale=answer_lang)
            answer_text = fill_result.to_answer_text()
            is_answerable = fill_result.is_success() and bool(answer_text) and (answer_text != origin_table_html)

            # 输出结果事件
            yield ContentGenerateResponseEvent(
                question_id=question_task.id,
                question_type="table",
                question_text=question_task.description,
                answer=FinalAnswer(
                    is_answerable=is_answerable,
                    answer_text=answer_text,
                    citations=question_citations
                )
            )
        except Exception as e:
            logger.error(f"表格agent调用失败: 问题 {question_task.id if question_task.id else 'null id'}，错误: {e}")
            logger.error(f"详细异常堆栈: {traceback.format_exc()}")
            # 输出现失败事件
            yield ContentGenerateResponseEvent(
                question_id=question_task.id if question_task.id else "null id",
                question_type="table",
                question_text=question_task.description if question_task.description else "null description",
                answer=FinalAnswer(
                    is_answerable=False,
                    answer_text=get_localized_message("table_agent_failed", answer_lang, error=str(e)),
                    citations=[]
                )
            )
            return

    async def table_answering(
        self,
        typed_items: List[Item],
        question_tasks: List[QuestionTask],
        model_id: Optional[str] = "model_for_content_generation_bailian",
        disable_keyword_extract: bool = False,
        max_iterations: Optional[int] = 5,
        answer_lang: str = "zh_CN",
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        logger.info(f"table_answering开始，共 {len(typed_items)} 个Item对象，共 {len(question_tasks)} 个问题")

        answer_lang = normalize_locale(answer_lang)

        async def run_with_semaphore(question_task: QuestionTask) -> ContentGenerateResponseEvent:
            """运行单个问题并返回事件"""
            try:
                async for event in self.answer_single_table(
                    typed_items=typed_items,
                    question_task=question_task,
                    model_id=model_id,
                    max_iterations=max_iterations,
                    disable_keyword_extract=disable_keyword_extract,
                    answer_lang=answer_lang,
                ):
                    return event
            except Exception as e:
                    logger.error(f"问题 {question_task.id} 执行失败: {e}")
                    return ContentGenerateResponseEvent(
                        question_id=question_task.id if question_task.id else "null id",
                        question_type="table",
                        question_text=question_task.description if question_task.description else "null description",
                        answer=FinalAnswer(
                            is_answerable=False,
                            answer_text=get_localized_message("table_agent_failed", answer_lang, error=str(e)),
                            citations=[]
                        )
                    )

        # 创建所有任务
        tasks = [asyncio.create_task(run_with_semaphore(qt)) for qt in question_tasks]

        # 使用 as_completed 实现每个任务完成时立即 yield
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if isinstance(result, Exception):
                logger.error(f"并行任务执行失败: {result}")
                continue
            yield result

    def _messages_tuples_to_any_messages(
        self, messages: list[tuple[str, str]]
    ) -> list[AnyMessage]:
        """
        将 list[(role, content)] 转为 list[AnyMessage]。
        role 支持: user/human -> HumanMessage, assistant/ai -> AIMessage, system -> SystemMessage。
        """
        result: list[AnyMessage] = []
        for role, content in messages:
            r = (role or "").strip().lower()
            if r in ("user", "human"):
                result.append(HumanMessage(content=content or ""))
            elif r in ("assistant", "ai"):
                result.append(AIMessage(content=content or ""))
            # elif r == "system":
            #     result.append(SystemMessage(content=content or ""))
            else:
                result.append(HumanMessage(content=content or ""))
        return result

    @trace_span(name="AnswerTableAgentService.answer_single_table_with_msg")
    async def answer_single_table_with_msg(
        self,
        typed_items: List[Item],
        question_task: QuestionTask,
        history_messages: list[tuple[str, str]],
        model_id: Optional[str] = "model_for_content_generation_bailian",
        disable_keyword_extract: bool = True,
        max_iterations: Optional[int] = 5,
        answer_lang: str = "zh_CN",
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        """
        带历史消息的单个表格回答：接收 list[(role, content)]，转换为 AnyMessage 后调用 answer_single_table。
        """
        any_messages = self._messages_tuples_to_any_messages(history_messages)
        async for event in self.answer_single_table(
            typed_items=typed_items,
            question_task=question_task,
            model_id=model_id,
            disable_keyword_extract=disable_keyword_extract,
            history_messages=any_messages,
            max_iterations=max_iterations,
            answer_lang=answer_lang,
        ):
            yield event

