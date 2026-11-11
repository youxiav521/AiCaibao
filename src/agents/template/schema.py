from typing import List, Optional
from pydantic import BaseModel, Field

from src.agents.template.context.toc.document_context import TocInfo
class Question(BaseModel):
    """问题信息，包含标题和完整描述"""
    id: str =  Field(..., description="q_数字格式id")
    description: str = Field(..., description="id对应的问题描述信息")
    # title: str = Field(..., description="id在原始文本中的title")
class QuestionDetail(Question):
    """问题明细"""
    type: Optional[str] = Field("text", description="问题类型")
    html: Optional[str] = Field("", description="html格式描述")
    prompt: Optional[str] = Field(None, description="问题提示信息")
    contents: Optional[List[TocInfo]] = Field("", description="目录结构")
    
class Context(BaseModel):
    file_path: str = Field(..., description="文件路径")

    title: Optional[str] = Field(None, description="标题")
    abstract: Optional[str] = Field(None, description="摘要")
    questions: Optional[List[Question]] = Field(
        [], description="文档中全部Question清单"
    )
    questions_detail: Optional[List[QuestionDetail]] = Field(
        [], description="文档中全部QuestionDetail清单"
    )
    toc: Optional[List[TocInfo]] = Field(
        [], description="目录结构"
    )
    markdown: Optional[str] = Field(None, description="markdown格式文本")
    html: Optional[str] = Field(None, description="html格式文本")
    md5: Optional[str] = Field(None, description="文件md5")
    name: Optional[str] = Field(None, description="文件名")