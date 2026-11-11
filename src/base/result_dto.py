from typing import Any
from pydantic import BaseModel
from typing import Optional
class ResultDto(BaseModel):
    """返回对象"""
    code: Optional[str] = "0"
    message: Optional[str] = "success"
    total: Optional[int] = 0
    data: Optional[Any] = None