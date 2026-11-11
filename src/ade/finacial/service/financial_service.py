"""ADE Financial 服务层。

薄封装层，聚焦三件事：
1. 根据 ``file_id`` 从 MinIO 读取 MinerU 解析后的 ``{file_id}/converted.json``；
2. 调用 ``src/applications/parsing`` 里已存在的抽取函数（元数据 +
   原始三大表 / 前端结构化三大表）；
3. 将 dict 结果转成本模块定义的 pydantic schema，供路由层直接返回。

为什么"传入 file_id 即可"：
    上传阶段（``POST /api/parsing/mineru/upload`` 或等价 ADE 上传接口）服务端
    会为每个文件生成 42 位的 ``file_id``（UUID + 10 随机字符）。MinerU
    解析后的扁平 items 列表存储在 MinIO 的
    ``{MINERU_OUTPUT_BUCKET}/{file_id}/converted.json``。由于 ``file_id`` 与
    该对象一一对应，后续所有只读抽取接口只需该 id 即可重新加载 items 并执行
    下游提取，无需再次上传文件。
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from loguru import logger

from src.ade.finacial.schema.financial_metadata_schema import (
    ExtractFinancialMetadataData,
    ExtractThreeStatementsData,
    FinancialJobStatus,
    FinancialMetadataByJobData,
    FinancialMetadata,
    ThreeStatementsByJobData,
)
from src.ade.finacial.schema.three_statements_schema import StatementTableFrontend
from src.applications.parsing.service.three_statements_frontend_service import (
    extract_three_statements_frontend,
    extract_three_statements_frontend_by_file_id,
)
from src.applications.parsing.utils.financial_metadata_extractor import (
    parse_financial_metadata,
    parse_three_statements,
)
from src.applications.parsing.utils.mineru_storage import load_mineru_items
from src.base.storage_service import storage_service
from src.base.task_manager import TaskManager, TaskStatus
from src.utils import json_utils

_ADE_FINANCIAL_BUCKET = os.getenv("ADE_FINANCIAL_BUCKET", "ade-financial-results")


class AdeFinancialService:
    _TASK_STATUS_TO_ADE: dict[TaskStatus, FinancialJobStatus] = {
        TaskStatus.PENDING: FinancialJobStatus.pending,
        TaskStatus.RUNNING: FinancialJobStatus.processing,
        TaskStatus.SUCCESS: FinancialJobStatus.completed,
        TaskStatus.FAILED: FinancialJobStatus.failed,
    }

    @staticmethod
    def _progress_of(status: TaskStatus) -> float:
        if status == TaskStatus.SUCCESS:
            return 1.0
        if status == TaskStatus.RUNNING:
            return 0.5
        return 0.0

    @staticmethod
    def _received_at_of(created_at: str) -> int:
        try:
            return int(
                datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            )
        except Exception:
            return 0

    @staticmethod
    async def _resolve_file_id_by_job_id(job_id: str) -> str:
        """通过异步解析 job_id 反查 file_id。"""
        record = await TaskManager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")

        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        file_id = metadata.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            raise HTTPException(
                status_code=404,
                detail=(
                    "任务未关联 file_id，无法执行 file_id 侧财务提取；"
                    "请使用 file_id 接口或确认任务来源为 ADE parse 异步任务"
                ),
            )
        return file_id.strip()

    """ADE 财报抽取服务。"""

    @staticmethod
    async def _upload_result_to_minio(
        *,
        job_id: str,
        result_type: str,
        payload: Dict[str, Any],
    ) -> str:
        """将异步接口结果上传到财报模块独立桶并返回 MinIO 路径。"""
        date_prefix = datetime.now().strftime("%Y_%m_%d")
        object_name = f"results/{result_type}/{date_prefix}/{job_id}.json"
        result_bytes = json_utils.dumps_bytes(payload)
        await storage_service.upload(
            _ADE_FINANCIAL_BUCKET,
            object_name,
            result_bytes,
            "application/json",
        )
        return f"{_ADE_FINANCIAL_BUCKET}/{object_name}"

    # ---- 读取 items ---------------------------------------------------

    @classmethod
    async def _load_items(cls, file_id: str) -> List[Dict[str, Any]]:
        """根据 file_id 从 MinIO 读取 MinerU 转换后的 items 列表。

        - 找不到对象 -> 404
        """
        try:
            items = await load_mineru_items(file_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"未找到 file_id={file_id} 对应的解析结果，"
                    "请先通过 MinerU 上传接口完成解析"
                ),
            )

        if not isinstance(items, list):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"file_id={file_id} 对应的 MinerU converted_json 内容不是列表结构"
                ),
            )

        if not items:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"未找到 file_id={file_id} 对应的解析结果，"
                    "请先通过 MinerU 上传接口完成解析"
                ),
            )
        return items

    # ---- 可选落盘 -----------------------------------------------------

    @classmethod
    async def _save_metadata_files(
        cls,
        file_id: str,
        metadata: Dict[str, Any],
        three_statements_result: Dict[str, Any],
    ) -> tuple[str, str]:
        """把元数据和三大表原始结果各自上传 MinIO，返回 MinIO 路径对。"""
        metadata_path = await cls._upload_result_to_minio(
            job_id=file_id,
            result_type="metadata",
            payload=metadata,
        )
        three_statements_path = await cls._upload_result_to_minio(
            job_id=file_id,
            result_type="three_statements",
            payload=three_statements_result,
        )

        return metadata_path, three_statements_path

    # ---- 对外方法 -----------------------------------------------------

    @classmethod
    async def extract_metadata(
        cls,
        file_id: str,
        save: bool = False,
        locale: Optional[str] = None,
        include_three_statements: bool = False,
    ) -> ExtractFinancialMetadataData:
        """提取财报元数据 + 前端结构化三大表。

        Args:
            file_id: 上传阶段生成的 42 位文件 id，对应
                MinIO 中的 ``{file_id}/converted.json``。
            save: 为 True 时把元数据 / 原始三大表分别落 JSON，便于离线调试。
            locale: ``Accept-Language``，透传给前端三大表服务做本地化。
            include_three_statements: 是否在 metadata 接口中附带三大表。
        """
        start = time.time()
        logger.info(
            "ADE Financial metadata | file_id={} save={} locale={} include_three_statements={}",
            file_id,
            save,
            locale,
            include_three_statements,
        )

        items = await cls._load_items(file_id)

        # 元数据（同步 CPU/IO 轻计算，内部可能调用 LLM，视配置而定）
        try:
            metadata_dict = parse_financial_metadata(items)
        except Exception as exc:
            logger.exception("元数据提取失败 | file_id={}", file_id)
            raise HTTPException(status_code=500, detail=f"元数据提取失败: {exc}")

        if not isinstance(metadata_dict, dict):
            raise HTTPException(
                status_code=500, detail="元数据提取结果不是字典结构"
            )

        # 原始三大表（只在 save=True 时落盘；前端展示走下面的 frontend 版本）
        raw_three_statements: Optional[Dict[str, Any]] = None
        metadata_path: Optional[str] = None
        three_statements_path: Optional[str] = None

        if save:
            try:
                raw_three_statements = parse_three_statements(items)
                if not isinstance(raw_three_statements, dict):
                    raise ValueError("三大表提取结果不是字典结构")
                metadata_path, three_statements_path = await cls._save_metadata_files(
                    file_id, metadata_dict, raw_three_statements
                )
            except Exception as exc:
                logger.exception("保存抽取结果失败 | file_id={}", file_id)
                raise HTTPException(status_code=500, detail=f"保存结果失败: {exc}")

        metadata_model = FinancialMetadata.model_validate(metadata_dict)
        statements_model: Optional[List[StatementTableFrontend]] = None
        if include_three_statements:
            # 前端结构化三大表（面向最终用户渲染）
            try:
                frontend_three_statements = extract_three_statements_frontend(
                    items, locale=locale
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except Exception as exc:
                logger.exception("三大表前端结构化失败 | file_id={}", file_id)
                raise HTTPException(status_code=500, detail=f"三大表提取失败: {exc}")

            statements_model = [
                StatementTableFrontend.model_validate(s)
                for s in (frontend_three_statements or [])
            ]

        logger.success(
            "ADE Financial metadata done | file_id={} statements={} cost={:.2f}s",
            file_id,
            len(statements_model or []),
            time.time() - start,
        )

        return ExtractFinancialMetadataData(
            fileId=file_id,
            jobId=None,
            metadata=metadata_model,
            threeStatements=statements_model,
            metadataPath=metadata_path,
            threeStatementsPath=three_statements_path,
        )

    @classmethod
    async def extract_three_statements(
        cls,
        file_id: str,
        locale: Optional[str] = None,
    ) -> ExtractThreeStatementsData:
        """仅提取前端结构化的财务三大表。"""
        start = time.time()
        logger.info(
            "ADE Financial three-statements | file_id={} locale={}", file_id, locale
        )

        try:
            result = await extract_three_statements_frontend_by_file_id(
                file_id, locale=locale
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.exception("三大表提取失败 | file_id={}", file_id)
            raise HTTPException(status_code=500, detail=f"三大表提取失败: {exc}")

        statements: List[StatementTableFrontend] = [
            StatementTableFrontend.model_validate(s) for s in (result or [])
        ]
        logger.success(
            "ADE Financial three-statements done | file_id={} count={} cost={:.2f}s",
            file_id,
            len(statements),
            time.time() - start,
        )
        return ExtractThreeStatementsData(
            fileId=file_id,
            jobId=None,
            threeStatements=statements,
        )

    @classmethod
    async def extract_metadata_by_job_id(
        cls,
        job_id: str,
        save: bool = False,
        locale: Optional[str] = None,
        include_three_statements: bool = False,
    ) -> FinancialMetadataByJobData:
        """按 job_id 返回财报元数据提取状态与结果。"""
        record = await TaskManager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")

        ade_status = cls._TASK_STATUS_TO_ADE.get(
            record.status, FinancialJobStatus.failed
        )
        file_id = (
            (record.metadata.get("file_id") if isinstance(record.metadata, dict) else None)
            or None
        )

        result: Optional[ExtractFinancialMetadataData] = None
        if record.status == TaskStatus.SUCCESS and isinstance(file_id, str) and file_id:
            result = await cls.extract_metadata(
                file_id=file_id,
                save=save,
                locale=locale,
                include_three_statements=include_three_statements,
            )
            result.jobId = job_id
            result.metadataPath = await cls._upload_result_to_minio(
                job_id=job_id,
                result_type="metadata",
                payload=result.model_dump(mode="json", by_alias=True, exclude_none=True),
            )

        return FinancialMetadataByJobData(
            jobId=job_id,
            status=ade_status,
            receivedAt=cls._received_at_of(record.created_at),
            progress=cls._progress_of(record.status),
            fileId=file_id,
            result=result,
            failureReason=record.error,
        )

    @classmethod
    async def extract_three_statements_by_job_id(
        cls,
        job_id: str,
        locale: Optional[str] = None,
    ) -> ThreeStatementsByJobData:
        """按 job_id 返回三大表提取状态与结果。"""
        record = await TaskManager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")

        ade_status = cls._TASK_STATUS_TO_ADE.get(
            record.status, FinancialJobStatus.failed
        )
        file_id = (
            (record.metadata.get("file_id") if isinstance(record.metadata, dict) else None)
            or None
        )

        result: Optional[ExtractThreeStatementsData] = None
        if record.status == TaskStatus.SUCCESS and isinstance(file_id, str) and file_id:
            result = await cls.extract_three_statements(file_id=file_id, locale=locale)
            result.jobId = job_id
            result.threeStatementsPath = await cls._upload_result_to_minio(
                job_id=job_id,
                result_type="three_statements",
                payload=result.model_dump(mode="json", by_alias=True, exclude_none=True),
            )

        return ThreeStatementsByJobData(
            jobId=job_id,
            status=ade_status,
            receivedAt=cls._received_at_of(record.created_at),
            progress=cls._progress_of(record.status),
            fileId=file_id,
            result=result,
            failureReason=record.error,
        )
