"""ADE Financial 元数据 / 财务三大表响应 Schema。

对应服务侧实现：
- ``src/applications/parsing/extractor/orchestrator.py::FinancialMetadataOrchestrator.run()``
- ``src/applications/parsing/extractor/three_statements_extractor.py::ThreeStatementsExtractor.extract()``

说明：
- 元数据每个字段形如 ``{"value": str | None, "source_element_id": int | None}``，
  ``source_element_id`` 指向 MinerU 解析后 items 中的 ``element_id``，便于前端
  回溯高亮。
- 三大表（原始）为 ``ThreeStatementsExtractor`` 的输出，附带 ``element.rows``
  等原始行结构，面向"需要自己再加工"的调用方。若需要前端直接渲染，请使用
  ``three_statements_schema.py`` 中的 ``StatementTableFrontend``。

所有模型均允许额外字段（``extra="allow"``），以便服务端在不破坏兼容性的前提下
迭代。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.ade.finacial.schema.three_statements_schema import (
    StatementTableFrontend,
)

# 统一响应信封：复用 ade/parse 的风格（code / displayMessage / data）
# 不直接依赖 ade.parse 的内部类型，这里做了等价定义以避免循环依赖。


class AdeFinancialBaseResponse(BaseModel):
    """统一响应信封（与 AdeBaseResponse 对齐）。"""

    code: int = Field(200, description="响应码，成功统一为 200", examples=[200])
    displayMessage: str = Field(
        "Operation successful",
        description="响应描述；成功为 Operation successful",
        examples=["Operation successful"],
    )


# ---------------------------------------------------------------------------
# 元数据字段通用结构：{"value": ..., "source_element_id": ...}
# ---------------------------------------------------------------------------


class BaseInfoValue(BaseModel):
    """带溯源的单字段：value + source_element_id。"""

    value: Optional[str] = Field(
        None,
        description="字段取值；未识别时为 null",
        examples=["某某银行股份有限公司"],
    )
    source_element_id: Optional[int] = Field(
        None,
        description=(
            "来源 element_id，对应 MinerU 解析后 items 列表中该字段文本所在的"
            "元素 id（1-based，和 items 顺序一致），便于前端回溯高亮"
        ),
        examples=[12],
    )

    model_config = ConfigDict(extra="allow")


class CompanyFullNameInfo(BaseInfoValue):
    """企业全称。"""


class CurrentPeriodInfo(BaseInfoValue):
    """当前期数。

    服务端会把"2024 年度报告"这种描述标准化为 ``YYYY-MM-DD``：
    - 年度报告 → YYYY-12-31
    - 半年度报告 → YYYY-06-30
    - 第一季度报告 → YYYY-03-31
    - 第三季度报告 → YYYY-09-30
    """

    value: Optional[str] = Field(
        None,
        description="标准化后的报告截止日期（YYYY-MM-DD），若无法识别则为原文",
        examples=["2024-12-31"],
    )


class SecurityCodeEntry(BaseInfoValue):
    """单一市场的证券代码。"""

    value: Optional[str] = Field(
        None,
        description="证券代码；未在该市场上市时为 null",
        examples=["600001"],
    )


class SecurityCode(BaseModel):
    """三大市场的证券代码聚合。"""

    A股: SecurityCodeEntry = Field(
        default_factory=SecurityCodeEntry,
        description="A 股代码",
    )
    港股: SecurityCodeEntry = Field(
        default_factory=SecurityCodeEntry,
        description="港股代码",
    )
    美股: SecurityCodeEntry = Field(
        default_factory=SecurityCodeEntry,
        description="美股代码",
    )

    model_config = ConfigDict(extra="allow")


class SecurityShortNameEntry(BaseInfoValue):
    """单一市场的交易简称。"""

    value: Optional[str] = Field(
        None,
        description="交易简称；未在该市场上市时为 null",
        examples=["某某银行"],
    )


class SecurityShortName(BaseModel):
    """交易简称（当前仅区分 A 股 / 港股）。"""

    A股: SecurityShortNameEntry = Field(
        default_factory=SecurityShortNameEntry,
        description="A 股交易简称",
    )
    港股: SecurityShortNameEntry = Field(
        default_factory=SecurityShortNameEntry,
        description="港股交易简称",
    )

    model_config = ConfigDict(extra="allow")


class AuditInfo(BaseInfoValue):
    """审计意见类型。

    取值建议来自以下五类之一：
    - 标准无保留意见
    - 带强调事项段的无保留意见
    - 保留意见
    - 否定意见
    - 无法表示意见

    若文档为季报/半年报仅"审阅"而未审计，也可能取到 ``未经审计``。
    """

    value: Optional[str] = Field(
        None,
        description="审计意见类型或审阅结论",
        examples=["标准无保留意见"],
    )


class ChangeExplanationItem(BaseModel):
    """指标变动说明单项。

    该结构由 ``ChangesExtractorLangGraph`` 产出，字段可能依实际召回内容而变，
    这里仅约束最常见的核心字段，其余字段通过 ``extra="allow"`` 透传。
    """

    指标: Optional[str] = Field(None, description="指标名称", examples=["营业收入"])
    变动原因: Optional[str] = Field(
        None,
        description="变动原因描述",
        examples=["主要系业务规模扩大导致"],
    )
    source_element_id: Optional[int] = Field(
        None, description="来源 element_id", examples=[312]
    )

    model_config = ConfigDict(extra="allow")


class FinancialMetadata(BaseModel):
    """财报元数据聚合结构。

    来自 ``FinancialMetadataOrchestrator.run()`` 的返回值，键为中文以保持与
    原始业务语义一致。
    """

    企业全称: CompanyFullNameInfo = Field(
        default_factory=CompanyFullNameInfo, description="企业全称"
    )
    证券代码: SecurityCode = Field(
        default_factory=SecurityCode, description="多市场证券代码"
    )
    交易简称: SecurityShortName = Field(
        default_factory=SecurityShortName, description="多市场交易简称"
    )
    当前期数: CurrentPeriodInfo = Field(
        default_factory=CurrentPeriodInfo,
        description="当前报告期（标准化为 YYYY-MM-DD）",
    )
    审计: AuditInfo = Field(default_factory=AuditInfo, description="审计意见")
    指标变动说明: List[ChangeExplanationItem] = Field(
        default_factory=list, description="关键指标变动原因说明列表"
    )

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# 三大表（原始 LLM 版本）—— ThreeStatementsExtractor 输出
# ---------------------------------------------------------------------------


class StatementUnit(BaseModel):
    """报表金额单位识别结果。"""

    value: Optional[str] = Field(
        None,
        description="金额单位原文（如 '人民币元' / '万元' / '千元' / '亿元'）",
        examples=["人民币元"],
    )
    source_element_id: Optional[int] = Field(
        None,
        description="来源 element_id（通常为 '单位：...' 所在元素）",
        examples=[1022],
    )

    model_config = ConfigDict(extra="allow")


class RawStatementElement(BaseModel):
    """原始三大表元素（对应 items 里的 table/excel_table 节选）。"""

    element_id: int = Field(..., description="表格在 items 中的 element_id")
    table_type: Optional[str] = Field(
        None,
        description=(
            "表格分类标签。九种合法值之一："
            "合并资产负债表 / 母公司资产负债表 / 资产负债表 /"
            " 合并利润表 / 母公司利润表 / 利润表 /"
            " 合并现金流量表 / 母公司现金流量表 / 现金流量表"
        ),
        examples=["合并资产负债表"],
    )
    page: Optional[int] = Field(None, description="表格所在页码", ge=1)
    rows: Optional[List[Any]] = Field(
        None,
        description="表格原始行数据（MinerU 后处理后的二维结构）",
    )
    html: Optional[str] = Field(None, description="表格 HTML 原文")
    two_d_data: Optional[List[Any]] = Field(
        None, description="表格二维化表示（行×列）"
    )

    model_config = ConfigDict(extra="allow")


class RawThreeStatementItem(BaseModel):
    """单张原始三大表（未经前端精加工）。"""

    element: RawStatementElement = Field(..., description="原始表格元素")
    单位: StatementUnit = Field(..., description="金额单位识别结果")

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# 统一响应：/metadata/{file_id}
# ---------------------------------------------------------------------------


class ExtractFinancialMetadataData(BaseModel):
    """财报元数据接口的 ``data`` 载荷。"""

    fileId: str = Field(..., description="来源文件 ID（与请求路径一致）")
    jobId: Optional[str] = Field(
        None,
        description="可选任务 ID（按 job_id 查询时返回）",
    )
    metadata: FinancialMetadata = Field(..., description="财报元数据聚合结果")
    threeStatements: Optional[List[StatementTableFrontend]] = Field(
        None,
        description=(
            "可选：前端结构化的三大表列表（与 /three-statements 接口同构）。"
            "默认不返回；当 includeThreeStatements=true 时返回。"
        ),
    )
    metadataPath: Optional[str] = Field(
        None,
        description="save=true 时，元数据 JSON 落盘的绝对路径；否则为 null",
    )
    threeStatementsPath: Optional[str] = Field(
        None,
        description="save=true 时，三大表 JSON 落盘的绝对路径；否则为 null",
    )

    model_config = ConfigDict(extra="allow")


class ExtractFinancialMetadataResponse(AdeFinancialBaseResponse):
    """财报元数据接口响应体。"""

    data: Optional[ExtractFinancialMetadataData] = Field(
        None, description="成功时返回元数据 + 三大表；失败时为 null"
    )


# ---------------------------------------------------------------------------
# 统一响应：/three-statements/{file_id}
# ---------------------------------------------------------------------------


class ExtractThreeStatementsData(BaseModel):
    """三大表接口 data 载荷。"""

    fileId: str = Field(..., description="来源文件 ID")
    jobId: Optional[str] = Field(
        None,
        description="可选任务 ID（按 job_id 查询时返回）",
    )
    threeStatements: List[StatementTableFrontend] = Field(
        default_factory=list,
        description="按固定顺序输出的三大表列表",
    )
    threeStatementsPath: Optional[str] = Field(
        None,
        description="三大表结果在 MinIO 的路径（{bucket}/{object_name}）",
    )


class ExtractThreeStatementsResponse(AdeFinancialBaseResponse):
    """三大表前端结构化接口响应体。"""

    data: Optional[ExtractThreeStatementsData] = Field(
        None,
        description="三大表提取结果；成功时包含 fileId/jobId/threeStatements",
    )


class FinancialJobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class FinancialMetadataByJobData(BaseModel):
    """按 job_id 查询元数据提取状态与结果。"""

    jobId: str = Field(..., description="任务 ID")
    status: FinancialJobStatus = Field(..., description="任务状态")
    receivedAt: int = Field(..., ge=0, description="任务创建时间戳（秒）")
    progress: float = Field(..., ge=0, le=1, description="任务进度（0~1）")
    fileId: Optional[str] = Field(None, description="任务关联的 file_id")
    result: Optional[ExtractFinancialMetadataData] = Field(
        None, description="任务完成后返回的财报元数据结果"
    )
    failureReason: Optional[str] = Field(None, description="失败原因（若有）")


class FinancialMetadataByJobResponse(AdeFinancialBaseResponse):
    data: Optional[FinancialMetadataByJobData] = Field(
        None, description="任务状态与结果"
    )


class ThreeStatementsByJobData(BaseModel):
    """按 job_id 查询三大表提取状态与结果。"""

    jobId: str = Field(..., description="任务 ID")
    status: FinancialJobStatus = Field(..., description="任务状态")
    receivedAt: int = Field(..., ge=0, description="任务创建时间戳（秒）")
    progress: float = Field(..., ge=0, le=1, description="任务进度（0~1）")
    fileId: Optional[str] = Field(None, description="任务关联的 file_id")
    result: Optional[ExtractThreeStatementsData] = Field(
        None, description="任务完成后返回的三大表结果"
    )
    failureReason: Optional[str] = Field(None, description="失败原因（若有）")


class ThreeStatementsByJobResponse(AdeFinancialBaseResponse):
    data: Optional[ThreeStatementsByJobData] = Field(
        None, description="任务状态与结果"
    )


__all__ = [
    "AdeFinancialBaseResponse",
    "BaseInfoValue",
    "CompanyFullNameInfo",
    "CurrentPeriodInfo",
    "SecurityCode",
    "SecurityCodeEntry",
    "SecurityShortName",
    "SecurityShortNameEntry",
    "AuditInfo",
    "ChangeExplanationItem",
    "FinancialMetadata",
    "StatementUnit",
    "RawStatementElement",
    "RawThreeStatementItem",
    "ExtractFinancialMetadataData",
    "ExtractThreeStatementsData",
    "ExtractFinancialMetadataResponse",
    "ExtractThreeStatementsResponse",
    "FinancialJobStatus",
    "FinancialMetadataByJobData",
    "FinancialMetadataByJobResponse",
    "ThreeStatementsByJobData",
    "ThreeStatementsByJobResponse",
]
