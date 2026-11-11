"""
ADE Jobs 请求/响应模型（与 docs/new.md 规范对齐）

所有接口响应统一信封：
{
  "code": 200,
  "displayMessage": "Operation successful",
  "data": {...}
}

字段命名约定：
- 任务标识统一使用 snake_case `job_id`（与 docs/new.md 第 14 行一致）
- 时间戳为 ISO8601(UTC) 字符串
- content_list.content 保持现有 converted.json 结构，不约束 item 字段
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AdeJobsBaseResponse(BaseModel):
    """
    ADE Jobs 统一响应信封。

    与 src/ade/parse/schema/parse_response_schema.py 的 AdeBaseResponse 等价；
    独立定义以避免跨模块循环依赖（参考 AdeExtractBaseResponse / AdeFinancialBaseResponse）。
    """

    code: int = Field(200, description="响应码，成功统一为 200；业务失败为其他值")
    displayMessage: str = Field(
        "Operation successful",
        description="响应描述；成功为 Operation successful",
    )


# ── 接口 1：提交解析任务 ───────────────────────────────────────────────────────

class SubmitJobData(BaseModel):
    """提交解析任务成功后的 data 载荷。"""

    job_id: str = Field(..., description="新创建的任务 ID")


class SubmitJobResponse(AdeJobsBaseResponse):
    """POST /openapi/ade/jobs/submit 响应体。"""

    data: Optional[SubmitJobData] = Field(
        None, description="成功时返回 job_id，失败时为 null"
    )


# ── 接口 2：查询任务状态 ───────────────────────────────────────────────────────

class JobStatus(str, Enum):
    """ADE Jobs 任务状态枚举（docs/new.md 第 3 节）。

    取值固定为 pending / processing / completed / failed / cancelled。
    """

    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobStatusData(BaseModel):
    """查询任务状态的 data 载荷。"""

    job_id: str = Field(..., description="任务 ID")
    status: JobStatus = Field(..., description="任务状态")
    error_message: Optional[str] = Field(
        None, description="失败原因；status=failed 时返回，其他为 null"
    )
    progress: float = Field(
        0.0, ge=0.0, le=1.0, description="任务进度（0~1），不支持可恒为 0"
    )
    created_at: str = Field(..., description="任务创建时间（ISO8601, UTC）")
    updated_at: str = Field(..., description="任务最近更新时间（ISO8601, UTC）")


class JobStatusResponse(AdeJobsBaseResponse):
    """GET /openapi/ade/jobs/{job_id} 响应体。"""

    data: Optional[JobStatusData] = Field(
        None, description="任务状态详情；不存在时为 null"
    )


class CancelJobData(BaseModel):
    """取消任务后的状态。"""

    job_id: str
    status: JobStatus


class CancelJobResponse(AdeJobsBaseResponse):
    """POST /openapi/ade/jobs/{job_id}/cancel 响应体。"""

    data: Optional[CancelJobData] = None


# ── 接口 3：拉取解析结果 ───────────────────────────────────────────────────────

class ContentListPayload(BaseModel):
    """
    content_list 载荷：保持现有 converted.json 结构不变。

    不约束 item 字段（type/page/element_id/file_id/md/html/bBox/rows/standard2D/
    hasHeader/headerCells/table_caption/img_path 等字段一个都不少、不改名）。
    """

    content: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="converted.json item 数组，结构原样返回",
    )


class ImageItem(BaseModel):
    """图片清单单项。"""

    name: str = Field(..., description="图片标识（用于接口 4 下载路径 {image_name}）")
    path: str = Field(..., description="图片相对路径，与 content_list item.img_path 对应")
    size: int = Field(0, ge=0, description="图片字节数")


class ImagesPayload(BaseModel):
    """图片清单。"""

    count: int = Field(0, ge=0, description="图片数量")
    list: List[ImageItem] = Field(
        default_factory=list,
        description="图片列表；item.img_path 必须与 list[].name 一一对应",
    )


class JobMetadata(BaseModel):
    """任务结果元数据。"""

    external_file_id: Optional[str] = Field(
        None, description="Parsing 侧 file_id，原样透传回来便于对账"
    )
    filename: str = Field(..., description="原始文件名")
    page_count: int = Field(0, ge=0, description="解析页数")
    duration_ms: int = Field(0, ge=0, description="解析耗时（毫秒）")
    version: Optional[str] = Field(None, description="ADE 服务版本号")


class JobDataData(BaseModel):
    """拉取解析结果的 data 载荷。"""

    job_id: str = Field(..., description="任务 ID")
    content_list: ContentListPayload = Field(
        default_factory=ContentListPayload,
        description="converted.json item 数组",
    )
    images: ImagesPayload = Field(default_factory=ImagesPayload, description="图片清单")
    metadata: JobMetadata = Field(..., description="任务元数据")


class JobDataResponse(AdeJobsBaseResponse):
    """GET /openapi/ade/jobs/{job_id}/data 响应体。"""

    data: Optional[JobDataData] = Field(
        None, description="解析结果；未完成时为 null"
    )


# 注：健康检查的 HealthData / HealthResponse 已迁移至
# src/ade/health/schema/health_schema.py，避免 health 模块反向依赖 jobs 模块。
