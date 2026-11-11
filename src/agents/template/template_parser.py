import time
from typing import Optional, List
from loguru import logger
from tqdm import tqdm
from src.agents.template.context.analyse import AnalyseQuestion
from src.agents.template.context.document import DocumentContext, Question
from src.agents.template.graph import template_graph
from src.agents.template.utils import is_empty, read_file
from src.agents.template.llm_utils import abatch_extract_table_html
from src.agents.template.callback.callback_utils import (
    get_callback_url,
    send_start_callback,
    send_processing_callback,
    send_completed_callback,
    send_error_callback
)
from src.utils.model_enhancements.llm_metrics import metrics_context_session

async def parse_template_processor(
    id: str,
    template_id: str,
    qids: Optional[List[str]] = None,
    callback_url: Optional[str] = None,
    model_id: str = "model_bailian_80b",
    batch_size: int = 15,
    use_english: bool = False,
    send_callback_enabled: bool = True
):
    """
    模板解析处理器 - 异步回调方式

    Args:
        id: 模板ID
        template_id: 模板任务ID
        qids: 可选的问题ID列表，如果指定则只处理指定的问题
        callback_url: 回调URL，用于debug模式
        model_id: 模型ID，默认为 "model_bailian_80b"
        batch_size: 批处理大小，默认为15
        use_english: 是否使用英文模式，默认False
        send_callback_enabled: 是否发送回调信息，默认True
    """
    time_start = time.time()

    # 使用传入的回调URL，如果没有则使用环境变量中的默认值
    callback_url = callback_url if callback_url else get_callback_url()

    # 开启 LLM 调用统计（紧凑格式输出）
    # async with metrics_context_session(auto_print=True, compact=True):
    try:
        # 读取模板并获取问题
        graph_result = await template_graph.ainvoke({
            "template": await read_file(id),
            "model_id": model_id
        })
        questions: list[Question] = graph_result['questions']
        a = AnalyseQuestion(graph_result['context'], model_id=model_id)
        c: DocumentContext = graph_result['document_context']

        # 如果指定了 qids，则只处理指定的问题
        if qids and len(qids) > 0:
            questions = [q for q in questions if q.id in qids]

        total_questions = len(questions)

        # 发送开始回调
        interrupted = False
        if send_callback_enabled:
            interrupted = await send_start_callback(
                data={
                    "total": total_questions,
                    "message": f"开始处理模板，共 {total_questions} 个问题"
                },
                template_id=template_id,
                callback_url=callback_url
            )
            if interrupted:
                logger.warning(f"模板 {template_id} 收到中断指令，停止处理")
                return

        # 将问题分成批进行处理
        processed_count = 0
        with tqdm(total=total_questions, desc="问题分析") as pbar:
            for i in range(0, total_questions, batch_size):
                batch_questions = questions[i:i + batch_size] # 批量问题

                # 批量分析当前批次的问题
                batch_details = await a.analyse_batch_async(
                    batch_questions,
                    use_english=use_english,
                    model_id=model_id
                )

                # 收集需要提取 HTML 的表格类型问题
                table_items = []
                for idx, qdetail in enumerate(batch_details):
                    original_q = batch_questions[idx]
                    q_html = a.extract_html_qtag(qdetail.id, a.context.html)

                    # 收集表格类型的问题（包括有 HTML 和没有 HTML 的）
                    if qdetail.type == 'table' or qdetail.type == 'financial_table':
                        if is_empty(q_html):
                            # 没有 HTML 的表格直接转换为 text
                            logger.warning(f"问题 {qdetail.id} 没有 HTML 内容，将 type 改为 'text'")
                            qdetail.type = 'text'
                        else:
                            # 有 HTML 的表格加入批量处理列表
                            table_items.append((qdetail, q_html))

                # 批量提取表格 HTML（如果有需要处理的表格）
                if table_items:
                    await abatch_extract_table_html(
                        table_items=table_items,
                        model_id=model_id,
                    )

                # 对每个问题详情，先发送 top_toc，再发送问题详情
                for idx, qdetail in enumerate(batch_details):
                    original_q = batch_questions[idx]

                    q_text = a.get_sub_text(qdetail.id, a.context.markdown)
                    # 查询并发送 top_toc
                    top_toc = c.question_top_toc(original_q)
                    if top_toc:
                        for t in top_toc:
                            if send_callback_enabled:
                                interrupted = await send_processing_callback(
                                    data={
                                        "id": f'h_{t.id}',
                                        "description": t.text,
                                        "type": "header",
                                        "level": t.level
                                    },
                                    template_id=template_id,
                                    callback_url=callback_url
                                )
                                if interrupted:
                                    logger.warning(f"模板 {template_id} 收到中断指令，停止处理")
                                    return

                    # 发送问题详情
                    if send_callback_enabled:
                        interrupted = await send_processing_callback(
                            data=qdetail.model_dump(exclude_none=True),
                            template_id=template_id,
                            callback_url=callback_url
                        )
                        if interrupted:
                            logger.warning(f"模板 {template_id} 收到中断指令，停止处理")
                            return

                    processed_count += 1

                # 更新进度条
                pbar.update(batch_size)

        # 保存结果
        await a.write_jsonl(f'Q-{a.context.name}', a.context.questions_detail, model_id=a.model_id)
        time_end = time.time()
        processing_time = time_end - time_start
        logger.info(f"模板解析时间: {processing_time}秒")

        # 发送完成回调
        if send_callback_enabled:
            await send_completed_callback(
                data={
                    "total": total_questions,
                    "processed": processed_count,
                    "success": True,
                    "processing_time": processing_time
                },
                template_id=template_id,
                callback_url=callback_url
            )

    except Exception as e:
        logger.error(f"模板解析失败: {e}")
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"详细堆栈信息:\n{tb_str}")

        # 发送错误回调
        if send_callback_enabled:
            await send_error_callback(
                data={
                    "error": "模板解析失败",
                    "detail": str(e)
                },
                template_id=template_id,
                callback_url=callback_url
            )
