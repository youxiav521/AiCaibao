"""
ADE 结构化提取 Service — 统一入口

按 model 参数分发到 LLM / VLM 子服务。
多条规则的结果合并到同一个 ExtractionItem 中，extractName 作为内部 key。
"""
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.ade.extract.schema.extract_schema import (
    ExtractRule,
    ExtractionItem,
    ModelEnum,
)
from src.ade.extract.service.llm_extract_service import LlmExtractService
from src.ade.extract.service.vlm_extract_service import VlmExtractService
from src.base.storage_service import storage_service
from src.base.task_manager import TaskManager, TaskStatus

_ADE_EXTRACT_BUCKET = os.getenv("ADE_EXTRACT_BUCKET", "ade-extract")


class ExtractService:
    """ADE 结构化提取服务 — 统一入口"""

    def __init__(self):
        self.llm_service = LlmExtractService()
        self.vlm_service = VlmExtractService()

    async def sync_extract(
        self,
        file_bytes: bytes,
        filename: str,
        rules: List[ExtractRule],
        model: Optional[ModelEnum] = None,
    ) -> ExtractionItem:
        """
        同步提取主入口，按 model 分发。

        Returns:
            ExtractionItem，extraction 和 extraction_metadata 内部以 extractName 为 key。
        """
        if model == ModelEnum.yusys_vlm:
            logger.info("使用 VLM 模式提取, file={}", filename)
            per_rule: Dict[str, ExtractionItem] = await self.vlm_service.extract(
                file_bytes, filename, rules
            )
            return _merge_per_rule(per_rule)
        else:
            logger.info("使用 LLM 模式提取, file={}", filename)
            # LLM 新版直接返回合并好的 ExtractionItem
            return await self.llm_service.extract(file_bytes, filename, rules, model)

    async def submit_extract_job(
        self,
        file_bytes: bytes,
        filename: str,
        rules: List[ExtractRule],
        model: Optional[ModelEnum] = None,
    ) -> str:
        """
        异步提取任务提交：先把文件上传到存储，再把轻量参数入队。

        Returns:
            job_id
        """
        job_id = str(uuid.uuid4())
        date_prefix = datetime.now().strftime("%Y_%m_%d")
        object_name = f"inputs/{date_prefix}/{job_id}/{filename}"

        await storage_service.upload(
            _ADE_EXTRACT_BUCKET,
            object_name,
            file_bytes,
            "application/octet-stream",
        )
        file_path = f"{_ADE_EXTRACT_BUCKET}/{object_name}"

        rules_json = json.dumps(
            [r.model_dump() for r in rules], ensure_ascii=False
        )

        await TaskManager.submit_with_job_id(
            job_id=job_id,
            task_name="process_ade_extract",
            task_type="ade_extract",
            kwargs={
                "file_path": file_path,
                "filename": filename,
                "rules_json": rules_json,
                "model": model.value if model else None,
            },
        )

        logger.info(
            "ADE extract 任务已入队 | job_id={} file={} model={}",
            job_id, filename, model,
        )
        return job_id


def _merge_per_rule(per_rule: Dict[str, ExtractionItem]) -> ExtractionItem:
    """将 {extractName: ExtractionItem} 合并为单个 ExtractionItem。"""
    merged_extraction: Dict[str, Any] = {}
    merged_metadata: Dict[str, Any] = {}
    for extract_name, item in per_rule.items():
        merged_extraction[extract_name] = item.extraction
        merged_metadata[extract_name] = item.extraction_metadata
    return ExtractionItem(
        extraction=merged_extraction,
        extraction_metadata=merged_metadata,
    )
