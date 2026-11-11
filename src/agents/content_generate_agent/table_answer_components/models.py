"""
数据模型定义：表格拆分结果的数据结构
"""

from typing import List
from pydantic import BaseModel, Field


class TableRows(BaseModel):
    """单个关系表的行号分组（基于1的行号）"""
    row_numbers: List[int] = Field(
        ..., 
        description="该关系表包含的行号列表（基于1的行号，即第一行为1）"
    )


class Tables(BaseModel):
    """表格拆分结果：包含多个关系表的行号分组"""
    table_list: List[TableRows] = Field(
        ..., 
        description="拆分后的关系表列表，每个元素包含一组行号"
    )


class TableRowsWithHTML(BaseModel):
    """单个关系表的行号分组 + 生成的HTML"""
    row_numbers: List[int] = Field(
        ..., 
        description="该关系表包含的行号列表（基于1的行号，即第一行为1）"
    )
    html: str = Field(
        ..., 
        description="该关系表对应的完整HTML表格（<table>...</table>）"
    )


class TablesWithHTML(BaseModel):
    """表格拆分结果：包含多个关系表的行号分组和HTML"""
    table_list: List[TableRowsWithHTML] = Field(
        ..., 
        description="拆分后的关系表列表，每个元素包含行号和对应的HTML表格"
    )

