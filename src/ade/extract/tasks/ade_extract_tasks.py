"""
ADE Extract 异步任务

流程：
  1. 从 MinIO 下载原始文件
  2. 调用 ExtractService → LLM/VLM 提取
  3. 将 ExtractionItem JSON 上传到 MinIO
  4. 通过 TaskManager 写回 SUCCESS/FAILED，result 中仅保存引用路径
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from arq import Retry
from loguru import logger

from src.ade.extract.schema.extract_schema import ExtractRule, ModelEnum
from src.ade.extract.service.extract_service import ExtractService
from src.base.storage_service import storage_service
from src.base.task_manager import TaskManager, TaskStatus

_ADE_EXTRACT_BUCKET = os.getenv("ADE_EXTRACT_BUCKET", "ade-extract")


async def process_ade_extract(
    ctx: dict,
    file_path: str,
    filename: str,
    rules_json: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    ARQ 异步任务：执行 ADE 结构化提取。

    Args:
        ctx:          ARQ 上下文（含 job_id、job_try 等）
        file_path:    MinIO 路径，格式 "{bucket}/{object_name}"
        filename:     原始文件名
        rules_json:   ExtractRule 列表的 JSON 字符串
        model:        模型枚举值字符串（yusys_llm / yusys_vlm）
    """
    job_id: str = ctx.get("job_id")
    job_try: int = ctx.get("job_try", 1)

    assert job_id is not None, "job_id not found in task context"

    logger.info(
        "ADE extract 任务开始 | job_id={} file={} model={} try={}",
        job_id, filename, model, job_try,
    )

    try:
        await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)

        if "/" not in file_path:
            raise ValueError(f"非法的 file_path，应为 '{{bucket}}/{{object_name}}'：{file_path}")
        bucket, object_name = file_path.split("/", 1)
        file_bytes = await storage_service.download(bucket, object_name)

        rules_data = json.loads(rules_json)
        rules = [ExtractRule(**r) for r in rules_data]
        model_enum = ModelEnum(model) if model else None

        service = ExtractService()
        extraction_item = await service.sync_extract(
            file_bytes=file_bytes,
            filename=filename,
            rules=rules,
            model=model_enum,
        )

        date_prefix = datetime.now().strftime("%Y_%m_%d")
        result_object = f"results/{date_prefix}/{job_id}.json"
        result_json_bytes = extraction_item.model_dump_json().encode("utf-8")
        await storage_service.upload(
            _ADE_EXTRACT_BUCKET,
            result_object,
            result_json_bytes,
            "application/json",
        )

        result: Dict[str, Any] = {
            "extraction_path": f"{_ADE_EXTRACT_BUCKET}/{result_object}",
            "filename": filename,
        }

        await TaskManager.update(
            job_id, TaskStatus.SUCCESS, result=result, attempts=job_try
        )
        logger.info(
            "ADE extract 任务完成 | job_id={} result_path={}",
            job_id, result["extraction_path"],
        )

        return result

    except Exception as exc:
        logger.exception("ADE extract 任务失败 | job_id={} 尝试次数={}", job_id, job_try)

        if job_try >= 3:
            await TaskManager.update(
                job_id, TaskStatus.FAILED, error=str(exc), attempts=job_try
            )
            logger.error("ADE extract 任务达到最大重试次数，最终标记为 FAILED | job_id={} 总尝试次数={}", job_id, job_try)
        else:
            retry_delay = 10 if job_try == 1 else 30
            raise Retry(defer=retry_delay)
