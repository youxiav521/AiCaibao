"""
文档提取V1模型
"""
from pydantic import BaseModel, Field
from typing import List

from .common import ExtractRule, ObjectSourceItem


class ExtractRequest(BaseModel):
    """提取请求模型"""
    docContent: str = Field(..., description="JSON 字符串，包含 file_id、content、convert_json 等")
    extract: List[ExtractRule] = Field(..., description="提取规则数组")


class ExtractResult(BaseModel):
    """提取结果模型"""
    extractId: str = Field(..., description="提取任务唯一标识")
    fileId: str = Field(..., description="文件唯一标识")
    objectName: str = Field(..., description="生成的对象名称")
    objectValue: str = Field(..., description="结构化数据")
    objectSource: List[ObjectSourceItem] = Field(..., description="溯源信息")
