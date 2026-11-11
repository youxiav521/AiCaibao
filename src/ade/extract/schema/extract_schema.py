"""
ADE 结构化提取 - 请求/响应模型
"""
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


# ---------- 枚举 ----------

class ModelEnum(str, Enum):
    yusys_llm = "yusys_llm"
    yusys_vlm = "yusys_vlm"


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# ---------- 请求 ----------

class ExtractRule(BaseModel):
    """提取规则"""
    extractName: str = Field(..., description="提取任务名称")
    extractPrompt: Optional[str] = Field(None, description="提取提示词")
    jsonSchema: str = Field(..., description="JSON Schema 字符串，定义输出格式")


# ---------- 溯源 ----------

class FieldReference(BaseModel):
    bbox: Optional[List[float]] = Field(None, description="位置信息 [x1, y1, x2, y2]")
    pageIndex: Optional[int] = Field(None, description="所在页码，从 1 开始")


class FieldMeta(BaseModel):
    value: Any = Field(None, description="字段最终值")
    references: FieldReference = Field(default_factory=FieldReference)


# ---------- 提取结果 ----------

class ExtractionItem(BaseModel):
    """提取结果，extraction 以 extractName 为 key 聚合多规则结果"""
    extraction: Dict[str, Any] = Field(default_factory=dict, description="结构化提取结果，与 jsonSchema 同构")
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict, description="字段级溯源，与 extraction 同构，叶子节点为 FieldMeta")


# ---------- 统一响应基类 ----------

class AdeExtractBaseResponse(BaseModel):
    """
    ADE Extract 统一响应信封，与 parse 的 AdeBaseResponse 保持一致风格。
    {
      "code": 200,
      "displayMessage": "Operation successful",
      ...
    }
    """
    code: int = Field(200, description="响应码，成功统一为 200")
    displayMessage: str = Field(
        "Operation successful",
        description="响应描述；成功为 Operation successful",
    )


# ---------- 响应 ----------

class SubmitJobResponse(AdeExtractBaseResponse):
    jobId: str


class QueryJobResponse(AdeExtractBaseResponse):
    jobId: str
    status: JobStatus
    data: Optional[ExtractionItem] = None


class SyncExtractResponse(AdeExtractBaseResponse):
    data: ExtractionItem = Field(default_factory=ExtractionItem)
