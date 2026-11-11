"""ADE Financial 路由。

接口列表：
  GET /openapi/ade/finacial/metadata/{file_id}          — 提取财报元数据 + 财务三大表
  GET /openapi/ade/finacial/three-statements/{file_id}  — 仅提取前端结构化的财务三大表

``file_id`` 由上传接口生成（MinerU 上传链路 / ADE upload-file），其正式语义为
MinerU 解析产物在 MinIO 中的 ``{file_id}/converted.json``。
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, Path, Query

from src.ade.finacial.schema import (
    ExtractFinancialMetadataResponse,
    ExtractThreeStatementsResponse,
    FinancialMetadataByJobResponse,
    ThreeStatementsByJobResponse,
)
from src.ade.finacial.service import AdeFinancialService

router = APIRouter(prefix="/openapi/ade/finacial", tags=["ADE 财报抽取"])

_service = AdeFinancialService()


@router.get(
    "/metadata/{file_id}",
    summary="基于 file_id 提取财报元数据（可选附带财务三大表）",
    response_model=ExtractFinancialMetadataResponse,
)
async def extract_financial_metadata(
    file_id: str = Path(
        ..., description="解析后生成的文件 ID（对应 MinIO 中的 {file_id}/converted.json）", min_length=1
    ),
    save: bool = Query(
        False,
        description=(
            "是否额外将元数据 / 原始三大表结果写入本地调试目录 "
            "`{STORAGE_BASE}/mineru_output/metadata/`；默认读取源仍以 MinIO 解析结果为准"
        ),
    ),
    include_three_statements: bool = Query(
        False,
        alias="includeThreeStatements",
        description=(
            "是否在 metadata 接口中附带返回 threeStatements。"
            "默认 false，建议通过 /three-statements/{file_id} 单独获取。"
        ),
    ),
    accept_language: Optional[str] = Header(
        None, alias="Accept-Language", description="本地化语言，透传给三大表服务"
    ),
) -> ExtractFinancialMetadataResponse:
    """聚合接口：一次返回财报元数据 + 前端结构化三大表。

    - 元数据：企业全称 / 证券代码 / 交易简称 / 当前期数 / 审计意见 /
      指标变动说明，均带 ``source_element_id`` 溯源。
    - 三大表：按 ``STATEMENT_ORDER`` 顺序输出最多 6 张，结构与
      ``/three-statements/{file_id}`` 完全一致。
    """
    data = await _service.extract_metadata(
        file_id=file_id,
        save=save,
        locale=accept_language,
        include_three_statements=include_three_statements,
    )
    return ExtractFinancialMetadataResponse(
        code=200,
        displayMessage="Operation successful",
        data=data,
    )


@router.get(
    "/three-statements/{file_id}",
    summary="基于 file_id 提取财务三大表",
    response_model=ExtractThreeStatementsResponse,
)
async def extract_three_statements(
    file_id: str = Path(
        ..., description="解析后生成的文件 ID（对应 MinIO 中的 {file_id}/converted.json）", min_length=1
    ),
    accept_language: Optional[str] = Header(
        None, alias="Accept-Language", description="本地化语言"
    ),
) -> ExtractThreeStatementsResponse:
    """仅返回前端结构化的财务三大表列表。

    返回的每一项 ``StatementTableFrontend`` 均包含：
    - ``id`` / ``mergeIds``: 合并表追踪 id
    - ``text``: 归一化后的报表标题（``StatementLabel`` 枚举之一）
    - ``currencyUnit``: 金额单位字符串（中文：元/千元/万元/亿元；英文：One/Thousands/Millions/Billions）
    - ``rows``: 单元格二维网格（带 rowSpan/colSpan 与页内 bBox）
    - ``lines``: 行级结构字段（与 rows 按 rowIndex 对齐）
    """
    data = await _service.extract_three_statements(
        file_id=file_id,
        locale=accept_language,
    )
    return ExtractThreeStatementsResponse(
        code=200,
        displayMessage="Operation successful",
        data=data,
    )


@router.get(
    "/metadata/by-job/{job_id}",
    summary="基于 job_id 提取财报元数据（可选附带财务三大表）",
    response_model=FinancialMetadataByJobResponse,
)
async def extract_financial_metadata_by_job(
    job_id: str = Path(..., description="ADE parse 异步任务 ID", min_length=1),
    save: bool = Query(
        False,
        description=(
            "是否额外将元数据 / 原始三大表结果写入本地调试目录 "
            "`{STORAGE_BASE}/mineru_output/metadata/`；默认读取源仍以 MinIO 解析结果为准"
        ),
    ),
    include_three_statements: bool = Query(
        False,
        alias="includeThreeStatements",
        description=(
            "是否在 metadata 接口中附带返回 threeStatements。"
            "默认 false，建议通过 /three-statements/by-job/{job_id} 单独获取。"
        ),
    ),
    accept_language: Optional[str] = Header(
        None, alias="Accept-Language", description="本地化语言，透传给三大表服务"
    ),
) -> FinancialMetadataByJobResponse:
    """按 job_id 查询任务状态；完成后返回财报元数据结果。"""
    data = await _service.extract_metadata_by_job_id(
        job_id=job_id,
        save=save,
        locale=accept_language,
        include_three_statements=include_three_statements,
    )
    return FinancialMetadataByJobResponse(
        code=200,
        displayMessage="Operation successful",
        data=data,
    )


@router.get(
    "/three-statements/by-job/{job_id}",
    summary="基于 job_id 提取财务三大表",
    response_model=ThreeStatementsByJobResponse,
)
async def extract_three_statements_by_job(
    job_id: str = Path(..., description="ADE parse 异步任务 ID", min_length=1),
    accept_language: Optional[str] = Header(
        None, alias="Accept-Language", description="本地化语言"
    ),
) -> ThreeStatementsByJobResponse:
    """按 job_id 查询任务状态；完成后返回三大表结果。"""
    data = await _service.extract_three_statements_by_job_id(
        job_id=job_id,
        locale=accept_language,
    )
    return ThreeStatementsByJobResponse(
        code=200,
        displayMessage="Operation successful",
        data=data,
    )
