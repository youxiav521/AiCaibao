"""科目匹配（国际化）请求/响应数据模型"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class TemplateSubjectInput(BaseModel):
    """标准科目模板中的单个科目"""
    code: str = Field(..., description="科目编码，如 AST11")
    enName: str = Field("", description="英文名称")
    cnName: str = Field("", description="中文名称")
    level: int = Field(..., description="层级深度 1-4")
    parent: str = Field(..., description="父节点编码，根节点为 '-'")
    isLeaf: bool = Field(False, description="是否叶子节点")
    industry: str = Field("ALL", description="适用行业：ALL/MFG/RETAIL/AGRI/ENERGY")
    sort: int = Field(..., description="模板排序码，与标准模板定义一致，用于展示顺序")
    synonyms: list[str] = Field(default_factory=list, description="同义词列表，如 ['Cash', 'Cash on hand']")


class ParseLineInput(BaseModel):
    """解析结果中的单行数据"""
    rowIndex: int = Field(..., description="行序号")
    rawName: str = Field("", description="原始科目名称")
    cleanedName: str = Field("", description="清洗后的科目名称")
    lineType: str = Field("ITEM", description="行类型：ITEM/SUBTOTAL/TOTAL/BLANK/HEADER/NOTE/EMPTY")
    indentLevel: int = Field(1, description="层级深度，从 1 开始")
    inferredParent: Optional[int] = Field(None, description="推断的父行 rowIndex")
    precedingName: str = Field("", description="前一行科目名称")
    followingName: str = Field("", description="后一行科目名称")


class MatchRequest(BaseModel):
    """科目匹配请求"""
    id: str = Field(..., description="结果表 id")
    template: list[TemplateSubjectInput] = Field(
        ...,
        description="标准科目模板列表，每个元素代表一个标准科目",
    )
    parseResult: list[ParseLineInput] = Field(
        ...,
        description="解析结果行列表",
    )


class MatchResultItem(BaseModel):
    """单行匹配结果"""
    rowIndex: int = Field(..., description="对应 parse_result 中的行序号")
    matchedCode: str = Field(..., description="匹配到的标准科目编码")
    matchedName: str = Field(..., description="匹配到的标准科目名称")
    matchType: str = Field(..., description="匹配方式：synonym/exact/synonym_fuzzy/fuzzy/llm_fallback")


class MatchResponse(BaseModel):
    """科目匹配响应"""
    id: str = Field(..., description="请求中传入的 id，原样返回")
    matchedResults: list[MatchResultItem] = Field(default_factory=list, description="匹配成功的行列表")
