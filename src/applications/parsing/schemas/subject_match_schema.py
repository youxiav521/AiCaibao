"""财报科目匹配的数据模型定义"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class BBox(BaseModel):
    """边界框"""
    model_config = ConfigDict(populate_by_name=True)
    
    b: float = Field(0.0, description="底部坐标")
    coordOrigin: str = Field("TOPLEFT", alias="coord_origin", serialization_alias="coord_origin", description="坐标原点")
    l: float = Field(0.0, description="左侧坐标")
    r: float = Field(0.0, description="右侧坐标")
    t: float = Field(0.0, description="顶部坐标")
    
    @model_validator(mode='before')
    @classmethod
    def validate_empty(cls, data):
        """如果输入是空对象或None，返回默认值"""
        if data is None or (isinstance(data, dict) and len(data) == 0):
            return {"b": 0.0, "l": 0.0, "r": 0.0, "t": 0.0, "coordOrigin": "TOPLEFT"}
        return data


class Cell(BaseModel):
    """单元格"""
    model_config = ConfigDict(populate_by_name=True)
    
    cellId: str = Field(..., description="单元格唯一标识")
    subjectId: Optional[str] = Field(None, description="科目ID（匹配后填充）")
    text: str = Field(..., description="单元格文本内容")
    rowSpan: int = Field(1, alias="rowspan", serialization_alias="rowspan", description="行跨度")
    colSpan: int = Field(1, alias="colspan", serialization_alias="colspan", description="列跨度")
    pageAt: int = Field(..., alias="page", serialization_alias="page", description="页码")
    bBox: BBox = Field(default_factory=lambda: BBox(b=0.0, l=0.0, r=0.0, t=0.0, coordOrigin="TOPLEFT"), description="边界框")
    
    # 可选：保留业务字段（允许额外字段，但不强制）
    reportId: Optional[str] = Field(None, description="报表ID")
    instanceId: Optional[str] = Field(None, description="实例ID")
    docId: Optional[str] = Field(None, description="文档ID")
    docFolderId: Optional[str] = Field(None, description="文档文件夹ID")
    areaId: Optional[str] = Field(None, description="区域ID")
    financialsId: Optional[str] = Field(None, description="财务ID")
    beTitle: Optional[bool] = Field(None, description="是否为标题")
    rowNum: Optional[int] = Field(None, description="行号")
    colNum: Optional[int] = Field(None, description="列号")


class Table(BaseModel):
    """表格"""
    model_config = ConfigDict(populate_by_name=True)
    
    reportId: Optional[str] = Field(None, description="报表ID")
    reportName: Optional[str] = Field(None, description="报表名称")
    text: Optional[str] = Field(None, description="表格文本内容")
    pageAt: int = Field(..., description="页码")
    bBox: BBox = Field(default_factory=lambda: BBox(b=0.0, l=0.0, r=0.0, t=0.0, coordOrigin="TOPLEFT"), description="边界框")
    rows: List[List[Cell]] = Field(..., description="行数据")
    
    def model_post_init(self, __context):
        """后处理：如果reportId或reportName为空，尝试从rows中获取"""
        if not self.reportId and self.rows:
            # 尝试从第一个单元格获取reportId
            for row in self.rows:
                for cell in row:
                    if hasattr(cell, 'reportId') and cell.reportId:
                        self.reportId = cell.reportId
                        break
                if self.reportId:
                    break
        
        if not self.reportName and self.text:
            # 如果reportName为空，使用text作为reportName
            self.reportName = self.text


class Subject(BaseModel):
    """科目"""
    subjectId: str = Field(..., description="科目唯一标识")
    subjectName: str = Field(..., description="科目名称")
    synonymSubject: List[str] = Field(default_factory=list, description="同义词科目列表")
    summaryRelation: List[str] = Field(default_factory=list, description="汇总关系列表（子科目）")


class Report(BaseModel):
    """报表"""
    model_config = ConfigDict(populate_by_name=True, extra='allow')  # 允许额外字段
    
    reportId: str = Field(..., description="报表唯一标识")
    reportName: str = Field(..., description="报表名称")
    reportCaliber: int = Field(..., description="报表口径：1=单体，2=合并")
    subjectList: List[Subject] = Field(..., description="科目列表")
    
    @field_validator('reportCaliber', mode='before')
    @classmethod
    def validate_report_caliber(cls, v):
        """支持字符串转整数"""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"reportCaliber必须是整数或可转换为整数的字符串，当前值: {v}")
        return v


class SubjectMatchRequest(BaseModel):
    """财报科目匹配请求"""
    fileId: str = Field(..., description="文件ID")
    reportList: List[Report] = Field(..., description="报表列表")
    tables: List[Table] = Field(..., description="表格列表")


class SubjectMatchResponse(BaseModel):
    """财报科目匹配响应（内部使用，router层会转换为二维数组）"""
    fileId: str = Field(..., description="文件ID")
    tables: List[Table] = Field(..., description="匹配后的表格列表（subjectId已填充）")
