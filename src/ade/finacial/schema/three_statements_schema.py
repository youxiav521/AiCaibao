"""
财务三大表（前端结构）通用 Schema。

对应 `extract_three_statements_frontend(items)` 的输出结构：
- 顶层：List[StatementTableFrontend]
- 每个元素：一张报表（资产负债表/利润表/现金流量表，含合并/母公司/无前缀）

设计目标：
- 与 OpenAPI 3.0 规范兼容（字段描述/示例/约束清晰）
- 通用可扩展（允许额外字段，兼容后续演进）

v2 变动（手动合并支持）：
- StatementTableFrontend 新增 id / mergeIds，用于追踪合并来源
- 新增 OrgTableFrontend 描述原始表格块结构

v3 变动（财报识别字段）：
- primaryElementId / category / accountingPeriod / statementType / gaapType / isFinancialStatementTable

v4 变动（行级结构字段）：
- StatementTableFrontend 新增 lines，用于描述每行的科目结构、行类型、缩进和父级。
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class StatementLabel(str, Enum):
    """三大表标题枚举（与服务端 `STATEMENT_ORDER` 对齐）。"""

    merge_balance_sheet = "合并资产负债表"
    parent_balance_sheet = "母公司资产负债表"
    merge_income_statement = "合并利润表"
    parent_income_statement = "母公司利润表"
    merge_cashflow = "合并现金流量表"
    parent_cashflow = "母公司现金流量表"
    balance_sheet = "资产负债表"
    income_statement = "利润表"
    cashflow = "现金流量表"
    # 英文标签（Accept-Language: en 场景）
    en_merge_balance_sheet = "Consolidated Balance Sheet"
    en_parent_balance_sheet = "Standalone Balance Sheet"
    en_merge_income_statement = "Consolidated Income Statement"
    en_parent_income_statement = "Standalone Income Statement"
    en_merge_cashflow = "Consolidated Cashflow Statement"
    en_parent_cashflow = "Standalone Cashflow Statement"
    en_balance_sheet = "Balance Sheet"
    en_income_statement = "Income Statement"
    en_cashflow = "Cashflow Statement"

# --- 坐标盒模型 ---
class BBox(BaseModel):
    """
    盒模型坐标。用于定位元素/单元格在页面的位置。
    """
    # 与 parsing 侧兼容：部分单元格可能返回空 dict，此时使用默认零值与 TOPLEFT。
    l: float = Field(0, description="left", example=100.5)
    t: float = Field(0, description="top", example=200.0)
    r: float = Field(0, description="right", example=300.5)
    b: float = Field(0, description="bottom", example=220.0)
    coordOrigin: Literal["TOPLEFT", "BOTTOMLEFT"] = Field(
        "TOPLEFT", description="坐标原点", example="TOPLEFT"
    )

    # 允许额外字段，对应 OpenAPI additionalProperties: true
    model_config = ConfigDict(extra="allow")

# --- 单元格定义 ---
class TableCellFrontend(BaseModel):
    """
    表格单元格（归一化字段）。用于前端渲染/后续抽取。
    """
    text: str = Field(..., description="单元格文本（已转为字符串）", example="货币资金")
    rowSpan: int = Field(..., ge=1, description="行合并数（>=1）", example=1)
    colSpan: int = Field(..., ge=1, description="列合并数（>=1）", example=1)
    pageAt: Optional[int] = Field(None, description="单元格页码（若单元格自身无 pageAt，则回退使用表级 pageAt）", ge=1)
    bBox: BBox = Field(..., description="单元格坐标")
    rowStart: Optional[int] = Field(None, description="（可选）单元格起始行索引（0-based）", ge=0)
    colStart: Optional[int] = Field(None, description="（可选）单元格起始列索引（0-based）", ge=0)

    model_config = ConfigDict(extra="allow")

# --- 二维网格结构 ---
# 直接使用类型别名定义 List[List[...]]，对应 JSON Schema 中的嵌套数组
TableCellGrid = List[List[TableCellFrontend]]

# --- 单张报表结构 ---
class StatementTableFrontend(BaseModel):
    """
    单张报表（资产负债表/利润表/现金流量表）的前端结构化结果。
    """
    years: Optional[int] = Field(
        None,
        description="整份文档统一的报告年度（四位整数），来自封面/目录/报告期表述及主表表头；无则 null。",
        ge=1990,
        le=2050,
        examples=[2024],
    )
    period: str = Field(
        default="年报",
        description="整份文档统一的财报周期，与 /api/doc/fin_extract 规则一致：年报、半年报、一季度报、三季度报、月报。",
        examples=["年报"],
    )
    id: str = Field(
        ...,
        description="本条返回记录的唯一标识（唯一即可；与 /org-tables 的 id 规则一致时可带 "
        "_ss_l/_ss_r/_grp/_std 等派生后缀；多表纵向合并时可为 UUID）。",
    )
    mergeIds: List[str] = Field(
        ...,
        description="逻辑表 id 列表，与同文档 GET /org-tables 返回条目的 id 字段对齐（可含 "
        "_ss_l/_ss_r/_grp/_std）；单列未拆分时通常一项；多表合并时多项。",
    )
    primaryElementId: Optional[str] = Field(
        None,
        description="主来源 MinerU element_id（已去掉 _ss_* / _grp / _std 等派生后缀），便于按上游主键索引",
    )
    category: Optional[str] = Field(
        None,
        description="财报分类标准中文标签（如 合并资产负债表）",
    )
    accountingPeriod: Optional[Dict[str, Any]] = Field(
        None,
        description="会计周期：含 years（int 列表）、period_label（可读字符串，可为 null）",
    )
    statementType: Optional[Literal["BS", "IS", "CF"]] = Field(
        None,
        description="资产负债表/利润表/现金流量表 类型缩写",
    )
    gaapType: Optional[str] = Field(
        None,
        description="会计准则关键词识别结果：IFRS / US_GAAP / ESEF / SEC / CAS",
    )
    isFinancialStatementTable: Optional[bool] = Field(
        None,
        description="是否为三大表主表条目（本接口恒为 true）",
    )
    text: StatementLabel = Field(..., description="报表标题（已归一化）", examples=["合并资产负债表"])
    pageAt: Optional[int] = Field(None, description="报表所在页码", ge=1, example=1)
    bBox: BBox = Field(..., description="报表整体坐标")
    currencyUnit: str = Field(
        ...,
        description=(
            "金额单位展示字符串：默认中文为 元 / 千元 / 万元 / 亿元；"
            "``Accept-Language: en`` 时为 One / Thousands / Millions / Billions"
        ),
        examples=["千元"],
    )
    lineHeaderRowCount: int = Field(
        0,
        ge=0,
        description="rows 开头明确属于表头的行数；lines 生成只依据该字段识别 HEADER 行",
    )
    rows: TableCellGrid = Field(..., description="表格二维单元格网格")
    lines: List["StatementLineItem"] = Field(default_factory=list, description="行级结构字段列表，与 rows 按 rowIndex 对齐")

    model_config = ConfigDict(extra="allow")


# --- 原始表格块结构（未合并） ---
class OrgTableFrontend(BaseModel):
    """
    PDF 中的单个原始表格块（MinerU 解析后、自动跨页合并后的基本单元）。
    用于前端展示可选的合并候选项。
    """
    years: Optional[int] = Field(
        None,
        description="整份文档统一的报告年度（四位整数）",
        ge=1990,
        le=2050,
        examples=[2024],
    )
    period: str = Field(
        default="年报",
        description="整份文档统一的财报周期",
        examples=["年报"],
    )
    id: str = Field(..., description="表格唯一标识（来自 MinerU element_id）")
    text: str = Field("", description="表格分类标签（如资产负债表），可为空")
    pageAt: Optional[int] = Field(None, description="表格所在页码", ge=1)
    bBox: BBox = Field(..., description="表格坐标")
    currencyUnit: str = Field(
        "元",
        description="金额单位（中文默认 元；英文场景为 One 等，与三大表接口一致）",
        examples=["元"],
    )
    lineHeaderRowCount: int = Field(
        0,
        ge=0,
        description="rows 开头明确属于表头的行数；lines 生成只依据该字段识别 HEADER 行",
    )
    rows: TableCellGrid = Field(..., description="表格二维单元格网格")

    model_config = ConfigDict(extra="allow")



class StatementLineItem(BaseModel):
    """三大表行级结构字段。"""

    rowIndex: int = Field(..., description="行序号")
    pageNumber: Optional[int] = Field(None, description="页码")
    rawName: str = Field("", description="原始科目名")
    cleanedName: str = Field("", description="清洗后科目名")
    rawAmounts: List[str] = Field(default_factory=list, description="金额列原始文本")
    numberingPrefix: str = Field("", description="编号前缀")
    lineType: str = Field(..., description="行类型：ITEM/SUBTOTAL/TOTAL/HEADER/BLANK/NOTE")
    lineTypeSignals: Dict[str, Any] = Field(default_factory=dict, description="lineType 判定信号")
    precedingName: Optional[str] = Field(None, description="上一行 rawName")
    followingName: Optional[str] = Field(None, description="下一行 rawName")
    fontSize: Optional[float] = Field(None, description="科目列 PDF 文本层字号；无法匹配时为 null")
    isBold: Optional[bool] = Field(None, description="科目列 PDF 文本层是否加粗；无法匹配时为 null")
    isItalic: Optional[bool] = Field(None, description="科目列 PDF 文本层是否斜体；无法匹配时为 null")
    styleSource: str = Field("", description="样式来源：PDF_SPAN/UNMATCHED/UNAVAILABLE_IMAGE_ONLY")
    indentLevel: int = Field(0, le=3, description="缩进层级")
    indentSource: str = Field("", description="缩进来源")
    inferredParent: Optional[int] = Field(None, description="推断父级 rowIndex")

    model_config = ConfigDict(extra="allow")

# 顶层输出：对应 JSON Schema 中的 type: array
ThreeStatementsFrontendResponse = List[StatementTableFrontend]
OrgTablesFrontendResponse = List[OrgTableFrontend]
