"""
ADE Health 路由

接口：
  GET /openapi/ade/health — 检查 ADE 服务健康状态

设计要点：
- 简单返回服务状态、版本号、当前 UTC 时间
- 不依赖 Redis / MinIO（避免外部依赖抖动导致健康检查失败）
"""
import os
from datetime import datetime, timezone

from fastapi import APIRouter

from src.ade.health.schema import HealthData, HealthResponse

router = APIRouter(prefix="/openapi/ade", tags=["ADE Health"])


@router.get(
    "/health",
    summary="检查 ADE 服务健康状态",
    response_model=HealthResponse,
)
async def health():
    """返回 ADE 服务健康状态、版本号与当前 UTC 时间。"""
    return HealthResponse(
        code=200,
        displayMessage="Operation successful",
        data=HealthData(
            status="healthy",
            version=os.getenv("ADE_VERSION", "unknown"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
