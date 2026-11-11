"""
ADE Health 响应模型

响应信封与 src/ade/jobs/schema/jobs_schema.py 的 AdeJobsBaseResponse 等价；
独立定义以避免 health 模块反向依赖 jobs 模块（参考 AdeExtractBaseResponse /
AdeFinancialBaseResponse 的独立定义模式）。
"""
from typing import Optional

from pydantic import BaseModel, Field


class _HealthBaseResponse(BaseModel):
    """ADE Health 统一响应信封。"""

    code: int = Field(200, description="响应码，成功统一为 200")
    displayMessage: str = Field(
        "Operation successful",
        description="响应描述；成功为 Operation successful",
    )


class HealthData(BaseModel):
    """健康检查 data 载荷。"""

    status: str = Field("healthy", description="服务状态")
    version: str = Field(..., description="ADE 服务版本号")
    timestamp: str = Field(..., description="当前时间（ISO8601, UTC）")


class HealthResponse(_HealthBaseResponse):
    """GET /openapi/ade/health 响应体。"""

    data: Optional[HealthData] = Field(None, description="健康状态信息")
