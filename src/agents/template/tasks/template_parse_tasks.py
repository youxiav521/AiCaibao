import time
from typing import Optional, List
from loguru import logger
from arq import Retry

from src.base.task_manager import TaskManager, TaskStatus
from src.agents.template.template_parser import parse_template_processor

_TEMPLATE_TASK_TYPE = "template_parse"
_TEMPLATE_TASK_NAME = "process_template_parse"


async def process_template_parse(
    ctx: dict,
    id: str,
    template_id: str,
    q_ids: Optional[List[str]] = None,
    callback_url: Optional[str] = None,
    model_id: str = "model_bailian_80b",
    batch_size: int = 5,
    use_english: bool = False,
    send_callback: bool = True
) -> None:
    """
    模板解析 ARQ 任务处理函数
    
    Args:
        ctx: ARQ 上下文，包含 job_id 等信息
        id: 模板ID
        template_id: 模板任务ID
        q_ids: 可选的问题ID列表
        callback_url: 回调URL
        model_id: 模型ID
        batch_size: 批处理大小
        use_english: 是否使用英文模式
        send_callback: 是否发送回调信息
    """
    job_id = ctx.get("job_id")
    job_try = ctx.get("job_try", 1)

    assert job_id is not None, "job_id not found in task context"

    logger.info(
        "模板解析 ARQ 任务开始: job_id={}, template_id={}, job_try={}",
        job_id, template_id, job_try
    )

    try:
        # 更新任务状态为 RUNNING
        await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)

        # 调用原有的模板解析逻辑
        time_start = time.time()
        
        await parse_template_processor(
            id=id,
            template_id=template_id,
            qids=q_ids,
            callback_url=callback_url,
            model_id=model_id,
            batch_size=batch_size,
            use_english=use_english,
            send_callback_enabled=send_callback
        )

        time_end = time.time()
        processing_time = time_end - time_start

        # 更新任务状态为 SUCCESS
        await TaskManager.update(
            job_id,
            TaskStatus.SUCCESS,
            result={
                "template_id": template_id,
                "processing_time": processing_time
            },
            attempts=job_try
        )

        logger.info(
            "模板解析 ARQ 任务完成: job_id={}, template_id={}, 耗时={:.2f}秒",
            job_id, template_id, processing_time
        )

    except Exception as e:
        logger.exception(
            "模板解析 ARQ 任务失败: job_id={}, template_id={}, error={}, attempts={}",
            job_id, template_id, str(e), job_try
        )

        if job_try >= 2:
            # 达到最大重试次数，标记为 FAILED
            await TaskManager.update(
                job_id,
                TaskStatus.FAILED,
                error=str(e),
                attempts=job_try
            )
            logger.error(
                "模板解析 ARQ 任务达到最大重试次数，最终标记为 FAILED: job_id={}, template_id={}",
                job_id, template_id
            )
        else:
            # 重试
            retry_delay = 10 if job_try == 1 else 30
            logger.info(
                "模板解析 ARQ 任务将重试: job_id={}, template_id={}, delay={}秒",
                job_id, template_id, retry_delay
            )
            raise Retry(defer=retry_delay)
