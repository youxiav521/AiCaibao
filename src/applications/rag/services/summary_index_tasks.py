from __future__ import annotations

"""Summary Index 异步任务处理模块。

本模块定义了三种 summary 相关的异步任务处理函数：
1. process_summary_index_summaries: 生成文本摘要孙切片（从 MinIO 下载子切片快照）
2. process_image_summaries: 生成图片摘要孙切片（从 MinIO 永久路径读取解析产物和子切片快照）
3. process_table_enrichment: 生成表格语义增强孙切片（从 MinIO 永久路径读取解析产物和子切片快照）

每个任务函数遵循统一的生命周期：
- 校验文件索引是否存在、summary_job_id 是否过期
- 从 MinIO 下载子切片快照并生成对应的孙切片
- 写入向量存储、刷新索引
- 更新任务状态（RUNNING -> SUCCESS/FAILED）
"""

import os
from typing import Any

from loguru import logger

from src.applications.rag.services.summary_job_state import get_current_summary_job_id
from src.applications.rag.schemas import ChildChunk
from src.applications.rag.services.summary_index import (
    build_summary_grandchildren_from_children,
    _is_excel_like_source,
)
from src.applications.rag.services.image_summary_service import (
    build_image_summary_grandchildren,
)
from src.applications.rag.services.table_semantic_enrichment import (
    build_table_semantic_nodes_async,
    _build_table_nodes,
)
from src.applications.rag.services.vectorstore import VectorStoresService
from src.base.distributed_lock import DistributedLock
from src.base.storage_service import storage_service
from src.base.task_manager import TaskManager, TaskStatus
from src.utils import json_utils

_MINERU_OUTPUT_BUCKET = os.getenv("MINERU_OUTPUT_BUCKET", "mineru-output")


async def _load_mineru_json_from_storage(file_id: str) -> list:
    """从 MinIO 永久路径加载解析产物，失败时回退到空列表。"""
    try:
        return json_utils.loads(
            (await storage_service.download(_MINERU_OUTPUT_BUCKET, f"{file_id}/converted.json")).decode("utf-8")
        )
    except Exception:
        logger.warning("无法从MinIO下载解析产物，回退到空列表 | file_id={}", file_id)
        return []


async def _load_children_from_storage(file_id: str, summary_job_id: str) -> list[ChildChunk]:
    """从 MinIO 下载子切片快照，按 summary_job_id 做版本隔离。

    失败时返回空列表，调用方据此判断是否跳过处理。
    """
    try:
        object_name = f"{file_id}/children/{summary_job_id}.json"
        data = await storage_service.download(_MINERU_OUTPUT_BUCKET, object_name)
        items = json_utils.loads(data.decode("utf-8"))
        return [ChildChunk(**item) for item in items]
    except Exception:
        logger.warning(
            "无法从MinIO下载子切片快照 | file_id={} summary_job_id={}",
            file_id, summary_job_id,
        )
        return []


async def process_summary_index_summaries(ctx: dict, file_id: str, summary_job_id: str,
                                           file_name: str = "", folder_id: str = "") -> dict[str, Any]:
    """处理 Summary Index 文本摘要任务。

    对文件的子切片（children）调用 LLM 生成摘要孙切片（summary grandchildren），
    写入向量存储并刷新索引。

    跳过的条件（幂等保证）：
    - 文件索引不存在（解析产物已被删除）
    - summary_job_id 已过期（有更新的 summary job 覆盖）
    - 源文件为 Excel/CSV 类型（表格式文件不生成文本摘要）
    - 子切片列表为空

    Args:
        ctx: 任务上下文，包含 job_id 和 job_try 等信息。
        file_id: 文件 ID，用于从 MinIO 下载子切片快照。
        summary_job_id: 摘要任务 ID，用于幂等校验。
        file_name: 文件名（可选），用于 Excel/CSV 类型判断。
        folder_id: 文件夹 ID（可选）。

    Returns:
        dict[str, Any]: 包含以下键：
            - file_id (str): 处理的文件 ID。
            - summary_grandchildren (int): 成功写入的孙切片数量。
            - skipped (bool, 可选): 是否跳过处理。
            - reason (str, 可选): 跳过原因。
            - error (str, 可选): 异常信息。

    Raises:
        RuntimeError: 当孙切片写入数量与生成数量不一致时抛出。
    """
    job_id = ctx.get("job_id")
    job_try = int(ctx.get("job_try", 1))
    try:
        if job_id:
            await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)

        logger.info("SummaryIndex摘要任务开始 | job_id={} file_id={}", job_id, file_id)
        async with DistributedLock.acquire(f"rag:index:{file_id}", blocking=True):
            # 检查文件索引是否存在
            if not await VectorStoresService.exists_file_id(file_id):
                logger.warning("SummaryIndex摘要任务跳过，文件索引不存在 | job_id={} file_id={}", job_id, file_id)
                result = {
                    "file_id": file_id,
                    "summary_grandchildren": 0,
                    "skipped": True,
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result
            # 检查 summary_job_id 是否仍然有效，避免过期任务重复执行
            current_summary_job_id = await get_current_summary_job_id(file_id)
            if current_summary_job_id != summary_job_id:
                logger.warning(
                    "SummaryIndex摘要任务跳过，summary job 已变化 | job_id={} file_id={}",
                    job_id,
                    file_id,
                )
                result = {
                    "file_id": file_id,
                    "summary_grandchildren": 0,
                    "skipped": True,
                    "reason": "stale_summary_job_id",
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result

            # Excel/CSV 类文件不生成 summary 孙切片
            if _is_excel_like_source([], file_name):
                logger.info(
                    "SummaryIndex摘要任务跳过，Excel/CSV 文件 | job_id={} file_id={}",
                    job_id,
                    file_id,
                )
                result = {
                    "file_id": file_id,
                    "summary_grandchildren": 0,
                    "skipped": True,
                    "reason": "excel_like_source",
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result

            # 从 MinIO 下载该文件的子切片快照
            children = await _load_children_from_storage(file_id, summary_job_id)
            if not children:
                logger.info(
                    "SummaryIndex摘要任务跳过，无子切片 | job_id={} file_id={}",
                    job_id,
                    file_id,
                )
                result = {
                    "file_id": file_id,
                    "summary_grandchildren": 0,
                    "skipped": True,
                    "reason": "no_children",
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result

            # 基于子切片生成摘要孙切片（核心逻辑，调用 LLM）
            summary_grandchildren = await build_summary_grandchildren_from_children(
                children,
                file_id=file_id,
                folder_id=folder_id,
                file_name=file_name,
            )

            summary_count = 0
            if summary_grandchildren:
                # 批量写入孙切片到向量存储
                summary_count = await VectorStoresService.upsert_grandchildren(
                    summary_grandchildren
                )
                # 写入数量校验，确保数据完整性
                if summary_count != len(summary_grandchildren):
                    raise RuntimeError(
                        "SummaryIndex摘要写入不完整: "
                        f"file_id={file_id}, "
                        f"summary_grandchildren={summary_count}/{len(summary_grandchildren)}"
                    )

            # 刷新向量存储索引，使新数据立即可检索
            await VectorStoresService.refresh_indices()

        result = {
            "file_id": file_id,
            "summary_grandchildren": summary_count,
        }
        if job_id:
            await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
        logger.info(
            "SummaryIndex摘要任务完成 | job_id={} file_id={} summary_grandchildren={}",
            job_id,
            file_id,
            summary_count,
        )
        return result
    except Exception as exc:
        error = str(exc)
        if job_id:
            await TaskManager.update(job_id, TaskStatus.FAILED, error=error, attempts=job_try)
        return {"error": error}

async def process_image_summaries(ctx: dict, file_id: str, summary_job_id: str,
                                   file_name: str = "", folder_id: str = "") -> dict[str, Any]:
    """处理图片摘要任务。

    从 MinIO 永久路径下载解析产物，对文件中的图片元素（通过 mineru_json 描述）
    调用 VLM 生成图片摘要孙切片，写入向量存储并刷新索引。

    与 process_summary_index_summaries 共享相同的幂等校验逻辑：
    文件索引存在性检查、summary_job_id 过期检查、子切片非空检查。

    Args:
        ctx: 任务上下文，包含 job_id 和 job_try 等信息。
        file_id: 文件 ID，用于从 MinIO 下载子切片快照和定位解析产物。
        summary_job_id: 摘要任务 ID，用于幂等校验。
        file_name: 文件名（可选）。
        folder_id: 文件夹 ID（可选）。

    Returns:
        dict[str, Any]: 包含以下键：
            - file_id (str): 处理的文件 ID。
            - image_grandchildren (int): 成功写入的图片孙切片数量。
            - skipped (bool, 可选): 是否跳过处理。
            - reason (str, 可选): 跳过原因。
            - error (str, 可选): 异常信息。

    Raises:
        RuntimeError: 当孙切片写入数量与生成数量不一致时抛出。
    """
    job_id = ctx.get("job_id")
    job_try = int(ctx.get("job_try", 1))
    try:
        if job_id:
            await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)

        logger.info("图片摘要任务开始 | job_id={} file_id={}", job_id, file_id)
        async with DistributedLock.acquire(f"rag:index:{file_id}", blocking=True):
            # 检查文件索引是否存在
            if not await VectorStoresService.exists_file_id(file_id):
                logger.warning("图片摘要任务跳过，文件索引不存在 | job_id={} file_id={}", job_id, file_id)
                result = {"file_id": file_id, "image_grandchildren": 0, "skipped": True}
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result
            # 检查 summary_job_id 是否仍然有效
            current_summary_job_id = await get_current_summary_job_id(file_id)
            if current_summary_job_id != summary_job_id:
                logger.warning("图片摘要任务跳过，summary job 已变化 | job_id={} file_id={}", job_id, file_id)
                result = {
                    "file_id": file_id, "image_grandchildren": 0,
                    "skipped": True, "reason": "stale_summary_job_id",
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result

            # 从 MinIO 下载该文件的子切片快照
            children = await _load_children_from_storage(file_id, summary_job_id)
            if not children:
                logger.info("图片摘要任务跳过，无子切片 | job_id={} file_id={}", job_id, file_id)
                result = {
                    "file_id": file_id, "image_grandchildren": 0,
                    "skipped": True, "reason": "no_children",
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result

            # 基于 mineru_json 中的图片元素生成图片摘要孙切片（核心逻辑，调用 VLM）
            mineru_json = await _load_mineru_json_from_storage(file_id)
            image_grandchildren = await build_image_summary_grandchildren(
                mineru_json,
                children=children,
                file_id=file_id,
                folder_id=folder_id,
                file_name=file_name,
            )

            image_count = 0
            if image_grandchildren:
                # 批量写入孙切片到向量存储
                image_count = await VectorStoresService.upsert_grandchildren(image_grandchildren)
                if image_count != len(image_grandchildren):
                    raise RuntimeError(
                        "图片摘要写入不完整: "
                        f"file_id={file_id}, image_grandchildren={image_count}/{len(image_grandchildren)}"
                    )

            # 刷新向量存储索引并持久化快照
            await VectorStoresService.refresh_indices()

        result = {"file_id": file_id, "image_grandchildren": image_count}
        if job_id:
            await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
        logger.info("图片摘要任务完成 | job_id={} file_id={} image_grandchildren={}", job_id, file_id, image_count)
        return result
    except Exception as exc:
        error = str(exc)
        if job_id:
            await TaskManager.update(job_id, TaskStatus.FAILED, error=error, attempts=job_try)
        return {"error": error}


async def process_table_enrichment(ctx: dict, file_id: str, summary_job_id: str,
                                     file_name: str = "", folder_id: str = "") -> dict[str, Any]:
    """处理表格语义增强任务。

    从 MinIO 永久路径下载解析产物，对文件中的表格元素（通过 mineru_json 描述）
    调用 LLM 生成语义增强孙切片，写入向量存储并刷新索引。

    与文本/图片摘要任务不同的是，本任务先通过 _build_table_nodes 预处理
    表格节点，若没有表格节点则直接跳过，避免无效的 LLM 调用。

    幂等校验逻辑与 process_summary_index_summaries 一致：
    文件索引存在性检查、summary_job_id 过期检查、子切片非空检查。

    Args:
        ctx: 任务上下文，包含 job_id 和 job_try 等信息。
        file_id: 文件 ID，用于从 MinIO 下载子切片快照和定位解析产物。
        summary_job_id: 摘要任务 ID，用于幂等校验。
        file_name: 文件名（可选）。
        folder_id: 文件夹 ID（可选）。

    Returns:
        dict[str, Any]: 包含以下键：
            - file_id (str): 处理的文件 ID。
            - semantic_grandchildren (int): 成功写入的语义孙切片数量。
            - skipped (bool, 可选): 是否跳过处理。
            - reason (str, 可选): 跳过原因。
            - error (str, 可选): 异常信息。

    Raises:
        RuntimeError: 当孙切片写入数量与生成数量不一致时抛出。
    """
    job_id = ctx.get("job_id")
    job_try = int(ctx.get("job_try", 1))
    try:
        if job_id:
            await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)

        logger.info("表格语义增强任务开始 | job_id={} file_id={}", job_id, file_id)
        async with DistributedLock.acquire(f"rag:index:{file_id}", blocking=True):
            # 检查文件索引是否存在
            if not await VectorStoresService.exists_file_id(file_id):
                logger.warning("表格语义增强任务跳过，文件索引不存在 | job_id={} file_id={}", job_id, file_id)
                result = {"file_id": file_id, "semantic_grandchildren": 0, "skipped": True}
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result
            # 检查 summary_job_id 是否仍然有效
            current_summary_job_id = await get_current_summary_job_id(file_id)
            if current_summary_job_id != summary_job_id:
                logger.warning("表格语义增强任务跳过，summary job 已变化 | job_id={} file_id={}", job_id, file_id)
                result = {
                    "file_id": file_id, "semantic_grandchildren": 0,
                    "skipped": True, "reason": "stale_summary_job_id",
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result

            # 从 MinIO 下载该文件的子切片快照
            children = await _load_children_from_storage(file_id, summary_job_id)
            if not children:
                logger.info("表格语义增强任务跳过，无子切片 | job_id={} file_id={}", job_id, file_id)
                result = {
                    "file_id": file_id, "semantic_grandchildren": 0,
                    "skipped": True, "reason": "no_children",
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result

            # 从 MinIO 永久路径加载解析产物并预处理表格节点
            mineru_json = await _load_mineru_json_from_storage(file_id)
            prepared_table_nodes = await _build_table_nodes(mineru_json)
            if not prepared_table_nodes:
                logger.info("表格语义增强任务跳过，无表格节点 | job_id={} file_id={}", job_id, file_id)
                result = {
                    "file_id": file_id, "semantic_grandchildren": 0,
                    "skipped": True, "reason": "no_table_nodes",
                }
                if job_id:
                    await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
                return result

            # 基于表格节点生成语义增强孙切片（核心逻辑，调用 LLM）
            semantic_grandchildren = await build_table_semantic_nodes_async(
                mineru_json,
                file_id=file_id,
                children=children,
                folder_id=folder_id,
                file_name=file_name,
                prepared_nodes=prepared_table_nodes,
            )

            semantic_count = 0
            if semantic_grandchildren:
                # 批量写入孙切片到向量存储
                semantic_count = await VectorStoresService.upsert_grandchildren(semantic_grandchildren)
                if semantic_count != len(semantic_grandchildren):
                    raise RuntimeError(
                        "表格语义增强写入不完整: "
                        f"file_id={file_id}, semantic_grandchildren={semantic_count}/{len(semantic_grandchildren)}"
                    )

            # 刷新向量存储索引并持久化快照
            await VectorStoresService.refresh_indices()

        result = {"file_id": file_id, "semantic_grandchildren": semantic_count}
        if job_id:
            await TaskManager.update(job_id, TaskStatus.SUCCESS, result=result, attempts=job_try)
        logger.info("表格语义增强任务完成 | job_id={} file_id={} semantic_grandchildren={}", job_id, file_id, semantic_count)
        return result
    except Exception as exc:
        error = str(exc)
        if job_id:
            await TaskManager.update(job_id, TaskStatus.FAILED, error=error, attempts=job_try)
        return {"error": error}
