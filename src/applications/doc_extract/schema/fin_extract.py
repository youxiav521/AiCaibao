"""
财报提取模型
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal

from .common import ObjectSourceItem
from .extract_v2 import FileInfo, MinioConfig, MODE_DESCRIPTION


class FinExtractRequest(BaseModel):
    """财报提取请求模型"""
    docContent: Optional[str] = Field(None, description="JSON 字符串，包含 file_id、content、convert_json 等。LLM模式必填")
    mode: Literal["llm", "vlm"] = Field(..., description=MODE_DESCRIPTION)
    minio: Optional[MinioConfig] = Field(None, description="MinIO配置，VLM模式必填")
    files: Optional[Dict[str, List[FileInfo]]] = Field(None, description="文件信息，key为桶名，value为文件列表，VLM模式必填")


class FinExtractResult(BaseModel):
    """财报提取结果模型"""
    extractId: str = Field(..., description="提取任务唯一标识")
    objectName: str = Field(..., description="生成的对象名称")
    objectValue: str = Field(..., description="结构化数据（JSON字符串）")
    objectSource: Optional[List[ObjectSourceItem]] = Field(None, description="溯源信息")
