import os
import re
import time
from typing import Any, Optional
from langfuse import Langfuse
from loguru import logger
from arq import Retry
from arq.worker import no_result

from src.base.task_manager import TaskManager, TaskStatus
from src.agents.content_generate_agent.content_generate_agent import (
    ContentGenerateAgent,
)
from src.agents.content_generate_agent.models import (
    Document,
    FinalAnswer,
    QuestionTask,
    ContentGenerateResponseEvent,
)
from src.agents.content_generate_agent.question_scheduler import (
    ContentGenerateQuestionScheduler,
)
from src.agents.content_generate_agent.utils import (
    content_generate_llm_stats_session,
    get_localized_message,
    resolve_answer_locale,
)
from src.applications.content_generate.tasks.callback_service import CallbackService
from src.applications.content_generate.services.log_redaction import redact_data_uris_for_log
from src.base.langfuse_observe import observe
from src.utils.langfuse_tracing import (
    reset_trace_context,
    set_trace_context,
)


def _allows_empty_content(questions: list[QuestionTask]) -> bool:
    """图片任务可仅依赖 dataSources.fileIds 检索原图。"""
    return bool(questions) and all(question.type == "image" for question in questions)


def _strip_citations_for_unanswerable(
    event: ContentGenerateResponseEvent,
) -> ContentGenerateResponseEvent:
    try:
        if not event.answer.is_answerable:
            event.answer.citations = []
            citation_pattern = re.compile(r"\[\d+\]")
            answer_text = event.answer.answer_text
            if answer_text:
                event.answer.answer_text = citation_pattern.sub("", answer_text).strip()
            logger.info(
                "移除不可回答答案中的引用标记: {} -> {}",
                answer_text, event.answer.answer_text,
            )
    except Exception:
        logger.exception("处理不可回答答案时去除引用信息失败")
    return event


def _log_content_generate_task_metrics(
    *,
    task_id: str,
    elapsed_s: float,
    questions_completed: int,
    questions_total: int,
    llm_api_calls: int,
    llm_scheduled_units: int,
) -> None:
    qps = (questions_completed / elapsed_s) if elapsed_s > 0 else 0.0
    logger.warning(
        "CONTENT_GENERATE_TASK_METRICS "
        "task_id={} elapsed_s={:.3f} questions_completed={} questions_total={} "
        "questions_per_sec={:.4f} llm_api_calls_total={} llm_scheduled_units_total={}",
        task_id, elapsed_s, questions_completed, questions_total,
        qps, llm_api_calls, llm_scheduled_units,
    )
    logger.warning(
        "【内容生成-任务汇总】 "
        "task_id={} | 耗时={:.2f}s | 完成题数={}/{} | 生成速度={:.4f}题/秒 "
        "| LLM实际调用次数(含重试放大)={} | LLM计划调用次数={}",
        task_id, elapsed_s, questions_completed, questions_total,
        qps, llm_api_calls, llm_scheduled_units,
    )


def _build_error_callback_data(
    *,
    message: str,
    completed_questions_count: int,
    questions: list[QuestionTask],
) -> dict[str, Any]:
    answer = FinalAnswer(
        is_answerable=False,
        answer_text=message,
        citations=[],
    )
    data: dict[str, Any] = {
        "success": False,
        "message": message,
        "completed_questions_counts": completed_questions_count,
    }
    question = questions[0] if len(questions) == 1 else None
    event = ContentGenerateResponseEvent(
        question_id=question.id if question else "",
        question_text=question.description if question else message,
        question_type=question.type if question else "",
        answer=answer,
    )
    data.update(event.model_dump(by_alias=True))
    return data


@observe(name="worker.process_content_generate", capture_input=False, capture_output=True)
async def process_content_generate(
    ctx: dict,
    content_list: list[dict[str, Any]],
    question_tasks: list[dict[str, Any]],
    task_id: str,
    callback_url: Optional[str] = None,
    answer_lang: str = "zh_CN",
    trace_id: Optional[str] = None,
    trace_carrier: Optional[dict[str, str]] = None,
) -> None:
    job_id = ctx.get("job_id")
    job_try = ctx.get("job_try", 1)

    assert job_id is not None, "job_id not found in task context"

    Langfuse().update_current_trace(name="worker.process_content_generate")

    trace_token = None

    try:
        if trace_id:
            trace_token = set_trace_context(trace_id)

        logger.info(
            "[i18n] content_generate ARQ任务开始: job_id={} task_id={} answer_lang={} job_try={} upstream_trace_id={}",
            job_id, task_id, answer_lang, job_try, trace_id,
        )

        await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)

        # 注册该批次到 CallbackService，防止并发批次间回调被误关闭
        CallbackService.register_batch(task_id)

        answer_lang = resolve_answer_locale(answer_lang)
        logger.info(
            "[i18n] content_generate ARQ任务语言归一化完成: job_id={} task_id={} answer_lang={}",
            job_id, task_id, answer_lang,
        )

        documents = [Document.model_validate(doc) for doc in content_list]
        questions = [QuestionTask.model_validate(qt) for qt in question_tasks]
        results: list[ContentGenerateResponseEvent] = []
        questions_total = len(questions)
        t0 = time.perf_counter()

        async with content_generate_llm_stats_session() as llm_stats:
            try:
                CallbackService.send(
                    "start",
                    {"msg": get_localized_message("callback_start", answer_lang)},
                    task_id,
                    callback_url,
                )

                try:
                    default_model_id = (
                        os.getenv("content_generate_model_id") or "model_bailian_80b"
                    )
                    content_agent = ContentGenerateAgent(
                        model_id=default_model_id, answer_lang=answer_lang
                    )

                    merged_content: list[dict[str, Any]] = []
                    if documents:
                        for doc in documents:
                            if doc.content:
                                merged_content.extend(doc.content)
                                logger.info(
                                    "添加文档 {} 的 {} 个内容片段",
                                    doc.file_name, len(doc.content),
                                )

                    if not merged_content and not _allows_empty_content(questions):
                        logger.warning("输入内容为空，无法生成。")
                        error_message = get_localized_message(
                            "callback_empty_input", answer_lang
                        )
                        error_data = _build_error_callback_data(
                            message=error_message,
                            completed_questions_count=0,
                            questions=questions,
                        )
                        CallbackService.send(
                            "error", error_data, task_id, callback_url
                        )
                        await CallbackService.signal_task_end(task_id)
                        return

                    logger.info(
                        "多文件内容合并完成，总计 {} 个内容片段", len(merged_content)
                    )

                    from src.agents.content_generate_agent.input_models import (
                        DocumentItems,
                    )

                    try:
                        typed_items = DocumentItems.model_validate(merged_content).root
                        logger.info(
                            "内容转换完成，共 {} 个Item对象", len(typed_items)
                        )
                    except Exception as e:
                        logger.exception("内容列表校验失败")
                        error_message = get_localized_message(
                            "callback_validation_failed", answer_lang, error=str(e)
                        )
                        error_data = _build_error_callback_data(
                            message=error_message,
                            completed_questions_count=0,
                            questions=questions,
                        )
                        CallbackService.send(
                            "error", error_data, task_id, callback_url
                        )
                        await CallbackService.signal_task_end(task_id)
                        return

                    logger.info("问题任务数量: {}", len(questions))

                    scheduler = ContentGenerateQuestionScheduler()
                    async for event in scheduler.run(
                        typed_items=typed_items,
                        question_tasks=questions,
                        content_agent=content_agent,
                        model_id_for_table=default_model_id,
                    ):
                        event = _strip_citations_for_unanswerable(event)
                        results.append(event)
                        logger.info(
                            "处理问题: {} 完成，输出结果: {}",
                            event.question_id,
                            redact_data_uris_for_log(event.model_dump(by_alias=True)),
                        )
                        event_data = event.model_dump(by_alias=True)
                        CallbackService.send(
                            "message", event_data, task_id, callback_url
                        )
                        if CallbackService.is_interrupted(task_id):
                            await CallbackService.signal_task_end(task_id)
                            await TaskManager.delete(
                                job_id,
                                task_type="report_generate",
                                remove_arq_keys=False,
                            )
                            logger.warning(
                                "content_generate ARQ任务已取消 | job_id={} task_id={}",
                                job_id, task_id,
                            )
                            return no_result

                    success_data = {
                        "success": True,
                        "message": get_localized_message(
                            "callback_success", answer_lang, count=len(results)
                        ),
                        "completed_questions_counts": len(results),
                    }
                    CallbackService.send(
                        "complete", success_data, task_id, callback_url
                    )

                except Exception as e:
                    logger.exception(
                        "内容生成过程中发生未知错误: {}", str(e)
                    )
                    error_message = get_localized_message(
                        "callback_unknown_error", answer_lang, error=str(e)
                    )
                    error_data = _build_error_callback_data(
                        message=error_message,
                        completed_questions_count=len(results),
                        questions=questions,
                    )
                    CallbackService.send(
                        "error", error_data, task_id, callback_url
                    )
            finally:
                elapsed_s = time.perf_counter() - t0
                _log_content_generate_task_metrics(
                    task_id=task_id,
                    elapsed_s=elapsed_s,
                    questions_completed=len(results),
                    questions_total=questions_total,
                    llm_api_calls=llm_stats.llm_api_attempts,
                    llm_scheduled_units=llm_stats.llm_scheduled_units,
                )

        await TaskManager.update(
            job_id, TaskStatus.SUCCESS, result={"completed": len(results)}, attempts=job_try
        )
        logger.info(
            "content_generate ARQ任务完成 | job_id={} task_id={} completed={}",
            job_id, task_id, len(results),
        )
        await CallbackService.signal_task_end(task_id)

    except Exception as e:
        logger.exception(
            "content_generate ARQ任务失败 | job_id={} task_id={} error={} attempts={}",
            job_id, task_id, str(e), job_try,
        )
        await CallbackService.signal_task_end(task_id)

        if job_try >= 2:
            await TaskManager.update(
                job_id, TaskStatus.FAILED, error=str(e), attempts=job_try
            )
            logger.error(
                "content_generate ARQ任务达到最大重试次数，最终标记为FAILED | job_id={} task_id={} attempts={}",
                job_id, task_id, job_try,
            )
        else:
            retry_delay = 10 if job_try == 1 else 30
            logger.info(
                "content_generate ARQ任务将重试 | job_id={} task_id={} delay={}s",
                job_id, task_id, retry_delay,
            )
            raise Retry(defer=retry_delay)
    finally:
        if trace_token is not None:
            reset_trace_context(trace_token)
