"""Excel 表格子区域的坐标模型。

与 MinerU 路径区分：
- MinerU 表格无 region 字段。
- Excel 表格的 region 使用 0 起始闭区间，与 cell.rowstart/colstart 一致，
  便于前端 xlsx 渲染库（SheetJS / Luckysheet 等，均为 0-indexed）直接对接。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExcelRegion(BaseModel):
    """Excel 子区域（0 起始，闭区间，与 cell.rowstart/colstart 一致）。"""

    model_config = ConfigDict(extra="allow")

    rowstart: int = Field(..., ge=0)
    rowend: int = Field(..., ge=0)
    colstart: int = Field(..., ge=0)
    colend: int = Field(..., ge=0)
