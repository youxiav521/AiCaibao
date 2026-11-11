"""
图片类型识别与印章标记 - 数据模型
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class SealTaggingRequest(BaseModel):
    """印章标记请求"""

    file_id: str = Field(..., description="MinerU 解析结果的 file_id")


class SealTaggingResult(BaseModel):
    """印章标记结果"""

    file_id: str = Field(..., description="文件 ID")
    seal_element_ids: List[int] = Field(..., description="识别为印章的 element_id 列表")
    items: List[Dict[str, Any]] = Field(..., description="处理后的 converted.json items")
