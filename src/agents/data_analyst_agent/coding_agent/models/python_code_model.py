
from pydantic import BaseModel, Field

class PythonCodeModel(BaseModel):
    """表示一段Python代码的数据模型。"""
    description: str = Field(..., description="代码片段的描述，用于描述代码的用途，或是执行完成后回答用户")
    imports: str = Field(..., description="需要导入的包，导入语句")
    code: str = Field(..., description="主要代码内容")
