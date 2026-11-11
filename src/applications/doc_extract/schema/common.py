"""
文档提取公共模型
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class ObjectSourceItem(BaseModel):
    """溯源信息项模型"""
    element_id: str = Field(..., description="元素ID")
    value: Dict[str, Any] = Field(..., description="该元素贡献的字段值")
    type: str = Field(..., description="元素类型（text/list/table/excel_table）")
    table_caption: Optional[str] = Field(None, description="表格标题（仅table类型有效）")


class ExtractRule(BaseModel):
    """提取规则模型"""
    extractId: str = Field(..., description="提取任务唯一标识")
    extractName: str = Field(..., description="提取任务名称")
    extractPrompt: str = Field("", description="提取提示词")
    jsonSchema: str = Field(..., description="JSON Schema字符串，定义输出格式")
    namingRule: str = Field("", description="命名规则模板，如 'xxx的身份证'")
