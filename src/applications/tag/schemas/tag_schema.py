"""文档标签匹配的数据模型定义"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TagItem(BaseModel):
    """标签项"""
    tagId: str = Field(..., description="标签唯一标识")
    tagName: str = Field(..., description="标签名称")


class TagRequest(BaseModel):
    """标签匹配请求
    
    注意：docContent应使用从minerU解析得到的JSON数据，
    可以直接传入原始的Dict列表，会在Agent内部处理。
    """
    tags: List[TagItem] = Field(..., description="待匹配的标签列表")
    docContent: List[Dict[str, Any]] = Field(..., description="解析后的文档内容列表（minerU格式的原始JSON）")


class TagResponse(BaseModel):
    """标签匹配响应"""
    tagIds: str = Field(..., description="命中的标签ID列表（逗号分隔）")
    explanation: Optional[str] = Field(None, description="匹配说明（可选）")
