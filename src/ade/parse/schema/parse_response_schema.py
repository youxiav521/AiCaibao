"""
ADE Parse 请求/响应模型（与官方 OpenAPI 及 parse_response.json 对齐）

- 请求体：multipart/form-data — ParseRequest
- 响应体：application/json — ParseResponse

ParseChunk.type 取值见官方「Chunk Types」：
https://docs.landing.ai/ade/ade-chunk-types
ADE Parse 接口说明：https://docs.landing.ai/api-reference/tools/ade-parse
"""
from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from fastapi import File, UploadFile
from pydantic import BaseModel, ConfigDict, Field


# 路由侧文件参数类型别名（与 OpenAPI document: binary 对应）
AdeParseDocumentUpload = Annotated[
    Optional[UploadFile],
    File(
        description="待解析文件（PDF/图片等）；与 document_url 二选一",
    ),
]


# ---------- 文件上传（POST /openapi/ade/upload-file）----------

AdeUploadFile = Annotated[
    UploadFile,
    File(
        ...,
        description="待上传文件（如 pdf/docx/pptx/md/txt/xlsx/csv/png/jpg/jpeg）",
    ),
]


class AdeBaseResponse(BaseModel):
    """
    ADE Parse 统一响应信封。

    所有 ADE 接口（同步/异步）均遵循相同壳：
    {
      "code": 200,
      "displayMessage": "Operation successful",
      "data": {...}
    }

    约定：
    - 成功时 `code = 200`
    - 业务失败时 `code` 返回非 200 的业务错误码（HTTP 层仍可能是 200/4xx/5xx）
    """

    code: int = Field(200, description="响应码，成功统一为 200；业务失败为其他值")
    displayMessage: str = Field(
        "Operation successful",
        description="响应描述；成功为 Operation successful",
    )


class UploadFileResponseData(BaseModel):
    """上传文件成功后的 data 载荷。"""

    fileId: str = Field(..., description="服务端生成的文件 ID")


class UploadFileResponse(AdeBaseResponse):
    """
    文件上传接口响应体。

    示例：
    {
      "code": 200,
      "displayMessage": "Operation successful",
      "data": {"fileId": "..."}
    }
    """

    data: Optional[UploadFileResponseData] = Field(
        None,
        description="成功时返回 fileId，失败时可为 null",
    )


# ---------- 创建解析任务（POST /openapi/ade/create-parse-job）----------

class SplitType(str, Enum):
    """OpenAPI SplitType：请求参数 split=page 时按页输出 splits。"""

    page = "page"


class SubmitParseJobRequest(BaseModel):
    """
    创建异步解析任务请求体。

    主路径使用 fileId（由 upload-file 返回）；同时保留 file/file_url 以兼容后续扩展。
    """

    fileId: str = Field(..., description="上传接口返回的文件 ID")
    file: Optional[str] = Field(
        None,
        description="预留字段：文件标识/内容（当前可不传）",
    )
    file_url: Optional[str] = Field(
        None,
        description="预留字段：文件 URL（当前可不传）",
    )
    split: Optional[SplitType] = Field(
        None,
        description="分段策略，可选 page；默认不分段",
    )
    callback_url: Optional[str] = Field(
        None,
        description="任务完成/失败后回调的 HTTP URL（POST JSON）",
    )


class SubmitParseJobResponseData(BaseModel):
    """创建异步任务成功后的 data 载荷。"""

    jobId: str = Field(..., description="新创建的异步任务 ID")


class SubmitParseJobResponse(AdeBaseResponse):
    """
    创建异步解析任务响应体（通常为 202 Accepted）。

    示例：
    {
      "code": 200,
      "displayMessage": "Operation successful",
      "data": {"jobId": "..."}
    }
    """

    data: Optional[SubmitParseJobResponseData] = Field(
        None,
        description="成功时返回 jobId，失败时可为 null",
    )

# ---------- ADE Parse 请求（Body: multipart/form-data）----------
#
# 与官方 OpenAPI 请求体 schema 对齐（原始 schema 名称为 ParseRequestWithEncryptedPassword）。
# 二进制字段 document 不在本 Pydantic 模型内，请在路由中单独使用 UploadFile = File(None)。


class CustomPrompts(BaseModel):
    """
    自定义解析提示（对应官方 custom_prompts 请求字段）。

    目前仅支持 figure 键，用于指导图形类块的描述生成方式。
    """

    figure: Optional[str] = Field(
        None,
        description='图形块自定义提示，如 "Describe axis labels in detail."',
    )


class ParseRequest(BaseModel):
    """
    ADE Parse `POST /openapi/ade/parse` 

    与官方 Body schema 名称一致；`document` 须单独用 FastAPI `File()` 上传。
    """

    model: Optional[str] = Field(None, description="解析使用的模型版本，例如 dpt-2-latest")
    document_url: Optional[str] = Field(
        None,
        description="待解析文件 URL；与 document（二进制上传）二选一",
    )
    split: Optional[SplitType] = Field(
        None,
        description="分段策略；设为 page 时响应中 splits 按页给出",
    )
    password: Optional[str] = Field(
        None,
        description="加密 PDF 等文件的密码；非加密文档可忽略",
    )
    custom_prompts: Optional[CustomPrompts] = Field(
        None,
        description="解析文件时，解析结果包含图像描述，称为图注, 自定义解析提示 JSON（目前仅支持 figure 键）",
    )

# ---------- ParseChunk.type（语义块类型，与官方 chunk types 一致）----------
#
# OpenAPI 中 ParseChunk.properties.type 为 string；具体取值以文档 Chunk Types 为准。
# logo / card / attestation / scan_code 仅在使用 DPT-2 等部分模型时可能出现。
# title：官方将标题归入 text；保留枚举值以兼容历史示例与本仓库 parse_response.json。

class ParseChunkType(str, Enum):
    """
    ADE Parse 响应 chunks[].type（与官方 GroundingType 枚举对齐）。

    对应关系：GroundingType 的 "chunk*" 前缀去掉后转 snake_case 即为此处值，
    另有 table / tableCell 等结构级类型。
    """

    text = "text"
    table = "table"
    marginalia = "marginalia"
    figure = "figure"
    logo = "logo"
    card = "card"
    attestation = "attestation"
    scan_code = "scan_code"
    title = "title"
    form = "form"  # 对应 chunkForm
    page_header = "page_header" # 对应 chunkPageHeader
    page_footer = "page_footer" # 对应 chunkPageFooter
    page_number = "page_number" # 对应 chunkPageNumber
    key_value = "key_value" # 对应 chunkKeyValue


# ---------- 枚举（与 ADE Parse OpenAPI GroundingType 一致）----------

class GroundingType(str, Enum):
    chunkLogo = "chunkLogo"
    chunkCard = "chunkCard"
    chunkAttestation = "chunkAttestation"
    chunkScanCode = "chunkScanCode"
    chunkForm = "chunkForm"
    chunkTable = "chunkTable"
    chunkFigure = "chunkFigure"
    chunkText = "chunkText"
    chunkMarginalia = "chunkMarginalia"
    chunkTitle = "chunkTitle"
    chunkPageHeader = "chunkPageHeader"
    chunkPageFooter = "chunkPageFooter"
    chunkPageNumber = "chunkPageNumber"
    chunkKeyValue = "chunkKeyValue"
    table = "table"
    tableCell = "tableCell"


# ParseResponseGrounding 的 type（排除 tableCell，由单独分支处理）
class NonTableCellGroundingType(str, Enum):
    chunkLogo = "chunkLogo"
    chunkCard = "chunkCard"
    chunkAttestation = "chunkAttestation"
    chunkScanCode = "chunkScanCode"
    chunkForm = "chunkForm"
    chunkTable = "chunkTable"
    chunkFigure = "chunkFigure"
    chunkText = "chunkText"
    chunkMarginalia = "chunkMarginalia"
    chunkTitle = "chunkTitle"
    chunkPageHeader = "chunkPageHeader"
    chunkPageFooter = "chunkPageFooter"
    chunkPageNumber = "chunkPageNumber"
    chunkKeyValue = "chunkKeyValue"
    table = "table"


# ---------- 块内最小 grounding（ParseGrounding：仅 box + page）----------

class ParseGroundingBox(BaseModel):
    """版面矩形框，左上-右下"""

    left: float = Field(..., description="左边界")
    top: float = Field(..., description="上边界")
    right: float = Field(..., description="右边界")
    bottom: float = Field(..., description="下边界")


class ParseChunkGrounding(BaseModel):
    """chunk.grounding：无 type/confidence，仅定位"""

    box: ParseGroundingBox
    page: int = Field(..., ge=0, description="页码，从 0 开始")


class ParseChunk(BaseModel):
    """语义块（ParseChunk）"""

    markdown: str = Field(..., description="该块对应 Markdown 片段")
    type: ParseChunkType = Field(
        ...,
        description="语义块类型：text/table/marginalia/figure 及 DPT-2 专有 logo/card/attestation/scan_code 等（见 ParseChunkType）",
    )
    id: str = Field(..., description="块唯一 id，用于 splits.chunks 与 grounding 键引用")
    grounding: ParseChunkGrounding


# ---------- splits（分页视图，ParseSplit）----------

class ParseSplit(BaseModel):
    """文档分段；请求 split=page 时通常为每页一条"""

    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(
        ...,
        validation_alias="class",
        serialization_alias="class",
        description="分段类型，如 page",
    )
    identifier: str = Field(..., description="分段标识，如 page_0")
    pages: List[int] = Field(..., description="该段覆盖的页码列表（0-based）")
    markdown: str = Field(..., description="该段范围内 Markdown")
    chunks: List[str] = Field(..., description="该段包含的 chunk id 列表")


# ---------- grounding 响应条目 ----------

class ParseLowConfidenceSpan(BaseModel):
    """low_confidence_spans 单项（对应官方 Patch schema）"""

    text: str = Field(..., description="低置信子串文本")
    span: Tuple[int, int] = Field(
        ...,
        description="子串在块 markdown 中的字符偏移 [start, end)",
    )
    confidence: float = Field(..., description="该子串置信度")


class ParseResponseTableCellGroundingPosition(BaseModel):
    """tableCell 在表格中的位置"""

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    rowspan: int = Field(..., ge=1)
    colspan: int = Field(..., ge=1)
    chunk_id: str = Field(..., description="所属表格 chunk 的 id")


class ParseResponseGrounding(BaseModel):
    """
    ParseResponseGrounding：普通块 grounding（非 tableCell）。
    """

    box: ParseGroundingBox
    page: int = Field(..., ge=0)
    type: NonTableCellGroundingType = Field(
        ...,
        description="普通 grounding 类型（不含 tableCell）",
    )
    confidence: Optional[float] = Field(None, description="可选置信度")
    markdown: Optional[str] = Field(
        None,
        description="该 grounding 对应的值文本（markdown 形式，可承载复杂结构）",
    )
    low_confidence_spans: List[ParseLowConfidenceSpan] = Field(
        default_factory=list,
        description="低置信片段",
    )


class ParseResponseTableCellGrounding(BaseModel):
    """ParseResponseTableCellGrounding：tableCell grounding。"""

    box: ParseGroundingBox
    page: int = Field(..., ge=0)
    type: Literal["tableCell"] = Field(
        ...,
        description="固定为 tableCell",
    )
    confidence: Optional[float] = Field(None, description="可选置信度")
    markdown: Optional[str] = Field(
        None,
        description="该单元格对应的值文本（markdown 形式，可承载复杂结构）",
    )
    position: Optional[ParseResponseTableCellGroundingPosition] = Field(
        None,
        description="tableCell 在表中的位置（官方定义为可空）",
    )


# grounding 条目统一类型（不使用 discriminator，避免 Pydantic v2 Literal 限制导致 schema 构建失败）
ParseResponseGroundingEntry = Union[ParseResponseGrounding, ParseResponseTableCellGrounding]


# ---------- metadata（ParseMetadata）----------

class ParseMetadata(BaseModel):
    """解析任务元数据"""

    filename: str = Field(..., description="原始文件名")
    org_id: Optional[str] = Field(None, description="组织 id，可为 null")
    page_count: int = Field(..., ge=0)
    duration_ms: int = Field(..., ge=0, description="耗时（毫秒）")
    credit_usage: float = Field(..., description="额度消耗")
    job_id: str = Field(..., description="任务 id")
    file_id: Optional[str] = Field(
        None,
        description="文件 ID（对应 MinIO 中的 {file_id}/converted.json）",
    )
    version: Optional[str] = Field(None, description="解析模型/服务版本")
    failed_pages: List[int] = Field(default_factory=list, description="解析失败的页码列表")


# ---------- 完整响应（与 parse_response.json 一致）----------

class ParseResponse(BaseModel):
    """
    ADE Parse 200 响应体形状（与 extract/schema/parse_response.json 一致）。

    官方字段定义见：ADE Parse OpenAPI — ParseResponse。
    """

    markdown: str = Field(..., description="整篇文档 Markdown")
    chunks: List[ParseChunk] = Field(default_factory=list, description="语义块列表")
    splits: List[ParseSplit] = Field(default_factory=list, description="分段列表，常与 split=page 对应")
    grounding: Optional[Dict[str, ParseResponseGroundingEntry]] = Field(
        default=None,
        description=(
            "版面证据映射（官方 API 中非 required 字段）；"
            "键为 chunk id / table id / tableCell id，值为对应的定位、置信度与 markdown 内容"
        ),
    )
    metadata: ParseMetadata


# ---------- ADE Get Parse Jobs（GET /openapi/ade/parse/jobs/{job_id}）----------
# 参考：https://docs.landing.ai/api-reference/tools/ade-get-parse-jobs

class SpreadsheetChunk(BaseModel):
    """SpreadsheetParseResponse.chunks[]。"""

    markdown: str = Field(
        ...,
        description="Chunk 内容：表格通常为 HTML table；图片衍生 chunk 可为 markdown",
    )
    type: str = Field(
        ...,
        description="Chunk 类型：table，或 /parse 系列类型（text/table/figure/form 等）",
    )
    id: str = Field(..., description="Chunk ID")
    grounding: Optional[ParseChunkGrounding] = Field(
        None,
        description="仅来自嵌入图片解析的 chunk 可能包含 grounding",
    )


class SpreadsheetSplit(BaseModel):
    """SpreadsheetParseResponse.splits[]。"""

    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(
        ...,
        validation_alias="class",
        serialization_alias="class",
        description="Split 类别：page（按 sheet）或 full（全部 sheet）",
    )
    identifier: str = Field(..., description="Split 标识：sheet 名称或 full")
    sheets: List[int] = Field(..., description="包含的 sheet 索引列表")
    markdown: str = Field(..., description="该 split 的合并 markdown")
    chunks: List[str] = Field(..., description="该 split 对应的 chunk ID 列表")


class SpreadsheetParseMetadata(BaseModel):
    """SpreadsheetParseResponse.metadata。"""

    filename: str = Field(..., description="原始文件名")
    org_id: Optional[str] = Field(None, description="组织 ID")
    sheet_count: int = Field(..., ge=0, description="处理的 sheet 数")
    total_rows: int = Field(..., ge=0, description="所有 sheet 总行数")
    total_cells: int = Field(..., ge=0, description="所有 sheet 非空单元格总数")
    total_chunks: int = Field(..., ge=0, description="提取出的 chunk 总数")
    total_images: int = Field(0, ge=0, description="提取出的图片总数")
    duration_ms: int = Field(..., ge=0, description="处理耗时（毫秒）")
    credit_usage: float = Field(0, description="额度消耗")
    job_id: str = Field("", description="任务 ID")
    version: Optional[str] = Field(None, description="解析版本")


class SpreadsheetParseResponse(BaseModel):
    """
    Spreadsheet 解析输出。

    结构与 ParseResponse 类似，但无 grounding 顶层字段。
    """

    markdown: str = Field(..., description="完整文档（HTML）")
    chunks: List[SpreadsheetChunk] = Field(default_factory=list)
    splits: List[SpreadsheetSplit] = Field(default_factory=list)
    metadata: SpreadsheetParseMetadata


class JobStatusData(BaseModel):
    """
    ADE Get Parse Jobs 单个任务的核心数据（放在 JobStatusResponse.data 中）。

    parse_result 可能是 ParseResponse、SpreadsheetParseResponse 或 null。
    """

    job_id: str = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    received_at: int = Field(..., description="任务接收时间戳")
    progress: float = Field(..., ge=0, le=1, description="任务进度，0~1")
    org_id: Optional[str] = Field(None, description="组织 ID")
    version: Optional[str] = Field(None, description="版本")
    parse_result: Optional[Union[ParseResponse, SpreadsheetParseResponse]] = Field(
        None,
        description="任务完成且未指定 output_save_url 时的输出结果",
    )
    output_url: Optional[str] = Field(
        None,
        description="结果下载链接（大结果或指定 output_save_url 时返回）",
    )
    metadata: Optional[ParseMetadata] = Field(
        None,
        description="文档 parse 元信息（文件类 parse 场景）",
    )
    failure_reason: Optional[str] = Field(None, description="失败原因")


class JobStatusResponse(AdeBaseResponse):
    """
    ADE Get Parse Jobs 统一响应体（HTTP 200）。

    data 结构见 JobStatusData。
    """

    data: Optional[JobStatusData] = Field(
        None, description="任务详情；不存在或查询失败时为 null"
    )


class ParseSyncResponseData(BaseModel):
    """同步解析 data 载荷。"""

    fileId: str = Field(
        ...,
        description="解析文件 ID，对应 MinIO 中的 {fileId}/converted.json，可直接用于财务提取接口路径参数 file_id",
    )
    parseResult: ParseResponse = Field(..., description="解析结果（与原 /parse 返回结构一致）")


class ParseSyncResponse(AdeBaseResponse):
    """
    POST /openapi/ade/parse 同步解析响应体。

    data 为同步解析结果，包含：
    - fileId: 对应 MinIO 解析结果对象（`{fileId}/converted.json`）的文件标识
    - parseResult: 解析结果 ParseResponse（保留原结构）
    """

    data: Optional[ParseSyncResponseData] = Field(
        None, description="同步解析结果；成功时必返回，失败时为 null"
    )


# ---------- ADE List Parse Jobs（GET /openapi/ade/parse/jobs）----------
# 参考：https://docs.landing.ai/api-reference/tools/ade-list-parse-jobs

class ParseJobStatus(str, Enum):
    """List Parse Jobs 状态过滤值。"""

    cancelled = "cancelled"
    completed = "completed"
    failed = "failed"
    pending = "pending"
    processing = "processing"


class ListParseJobsQuery(BaseModel):
    """ADE List Parse Jobs 的查询参数（query）。"""

    page: int = Field(0, ge=0, description="页码（0-indexed）")
    pageSize: int = Field(10, ge=1, le=100, description="每页数量")
    status: Optional[ParseJobStatus] = Field(None, description="按任务状态过滤")


class JobSummary(BaseModel):
    """JobsListResponse.jobs[] 单项摘要。"""

    job_id: str = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    received_at: int = Field(..., description="任务接收时间戳")
    progress: float = Field(
        ...,
        ge=0,
        le=1,
        description="任务进度（0~1）",
    )
    failure_reason: Optional[str] = Field(None, description="失败原因")


class JobsListData(BaseModel):
    """ADE List Parse Jobs 响应 data 载荷。"""

    jobs: List[JobSummary] = Field(default_factory=list, description="任务列表")
    org_id: Optional[str] = Field(None, description="组织 ID")
    has_more: bool = Field(False, description="是否存在下一页")


class JobsListResponse(AdeBaseResponse):
    """ADE List Parse Jobs 统一响应体。"""

    data: Optional[JobsListData] = Field(
        None, description="任务列表数据，失败时为 null"
    )
