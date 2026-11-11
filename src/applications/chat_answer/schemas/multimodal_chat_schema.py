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


class MultimodalChatRequest(BaseModel):
    """统一问答请求"""
    minio: Optional[MinioConfig] = None
    files: Optional[Dict[str, List[FileInfo]]] = Field(default_factory=dict)
    question: str = Field(..., description="用户问题")