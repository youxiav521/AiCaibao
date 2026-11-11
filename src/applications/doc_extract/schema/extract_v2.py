"""
文档提取V2模型
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

from .common import ExtractRule, ObjectSourceItem

MODE_DESCRIPTION = "提取模式：vlm 适用于单图片、短文档、不提取表格或表格内容简单的场景；llm 适用于复杂表格提取或长文档场景"


class MinioConfig(BaseModel):
    server_url: str = Field(..., description="MinIO服务器地址")
    access_key: str = Field(..., description="MinIO访问密钥")
    secret_key: str = Field(..., description="MinIO密钥")


class FileInfo(BaseModel):
    fileId: str = Field(..., description="MinIO中的文件ID/路径")
    fileName: str = Field(..., description="文件名称，用于识别文件类型")


class ExtractRequest(BaseModel):
    """提取V2请求模型"""
    docContent: Optional[str] = Field(None, description="JSON字符串，包含file_id、convert_json等。LLM模式必填")
    extract: List[ExtractRule] = Field(..., description="提取规则数组")
    mode: Literal["llm", "vlm"] = Field(..., description=MODE_DESCRIPTION)
    minio: Optional[MinioConfig] = Field(None, description="MinIO配置，VLM模式必填")
    files: Optional[Dict[str, List[FileInfo]]] = Field(None, description="文件信息，key为桶名，value为文件列表，VLM模式必填")


class ExtractResult(BaseModel):
    """提取结果模型"""
    extractId: str = Field(..., description="提取任务唯一标识")
    objectName: str = Field(..., description="生成的对象名称")
    objectValue: str = Field(..., description="结构化数据")
    objectSource: Optional[List[ObjectSourceItem]] = Field(None, description="溯源信息")
