"""
文档分类的数据模型定义
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class MinioConfig(BaseModel):
    """MinIO配置"""
    server_url: str = Field(..., description="MinIO服务器地址")
    access_key: str = Field(..., description="MinIO访问密钥")
    secret_key: str = Field(..., description="MinIO密钥")


class FileInfo(BaseModel):
    """文件信息"""
    fileId: str = Field(..., description="MinIO中的文件ID/路径")
    fileName: str = Field(..., description="文件名称")


class ClassifyItem(BaseModel):
    """分类项"""
    name: str = Field(..., description="分类名称")
    prompt: Optional[str] = Field(None, description="该分类的提示词，用于辅助LLM判断")


class DocClassifyRequest(BaseModel):
    """文档分类请求"""
    minio: MinioConfig = Field(..., description="MinIO配置信息")
    files: Dict[str, List[FileInfo]] = Field(..., description="文件信息,key为文件分组(如ddis)")
    classify: List[ClassifyItem] = Field(..., description="分类列表，每项包含分类名称和可选提示词")
    assign_classify: Optional[str] = Field(None, description="指定分类结果，传入后跳过大模型判断直接返回该值")
    thumb: Optional[str] = Field(None, description="缩略图尺寸")


class DocClassifyMinioResponse(BaseModel):
    """文档分类响应"""
    classify: str = Field(..., description="分类结果")
    fileId: str = Field(..., description="文件ID")
    fileName: str = Field(..., description="文件名称")
    thumb: str = Field(..., description="第一页缩略图 base64 编码")
