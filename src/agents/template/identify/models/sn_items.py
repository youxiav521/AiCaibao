from typing import Optional, List
from pydantic import BaseModel, Field

class SNItem(BaseModel):
    """SN 问题项结构"""
    # 答案类型，应该为'static_table'或'dynamic_table'或'default'
    answer_type: str = Field(..., description="答案类型，应该为'static_table'或'dynamic_table'或'default'")
    # 开始 SN 编号
    start_sn: Optional[str] = Field(None, description="开始位置的 SN 编号，如 '#1'")
    # 结束 SN 编号（与 start_sn 相同表示单个位置）
    end_sn: Optional[str] = Field(None, description="结束位置的 SN 编号，如 '#5'，与 start_sn 相同表示单个位置")

class SNExtractionResult(BaseModel):
    """SN 提取结果"""
    sn_items: List[SNItem] = Field(..., description="提取的答案位置 SN 列表，按在文档中出现的顺序排列")