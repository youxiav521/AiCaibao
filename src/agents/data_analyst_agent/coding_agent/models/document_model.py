import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# 支持 dict 或 Document，便于与现有 List[dict] 兼容
DocumentLike = Union["Document", Dict[str, Any]]

def _normalize_element_ids(d: Dict[str, Any]) -> List[str]:
    """从 dict 中解析 element_ids，支持 element_id（单值）或 element_ids（列表）。"""
    eids = d.get("element_ids")
    if eids is not None:
        if isinstance(eids, list):
            return [str(x) for x in eids if x is not None]
        return [str(eids)]
    eid = d.get("element_id")
    if eid is not None:
        return [str(eid)]
    return []


class Document(BaseModel):
    """
    通用文档模型，支持多种类型的文档。
    可通过 from_dict 从接口/上游 dict 统一转换。
    """
    id: str = Field(..., description="文档ID")
    title: str = Field(..., description="文档标题")
    text: str = Field(..., description="文档内容")
    table: str = Field(..., description="结构化表格内容")
    file_id: Optional[str] = Field(None, description="源文件ID")
    element_ids: Optional[List[str]] = Field(default_factory=list, description="源文件元素ID列表")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        """从上游 dict 构造 Document，兼容 id 为 int、element_id 单值等。"""
        if not isinstance(d, dict):
            return cls(id="", title="", text="", table="")
        id_val = d.get("id")
        id_str = str(id_val) if id_val is not None else ""
        title = d.get("title") or ""
        if title is not None and not isinstance(title, str):
            title = str(title)
        text = d.get("text") or ""
        if text is not None and not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False) if isinstance(text, (dict, list)) else str(text)
        table = d.get("table") or ""
        if table is not None and not isinstance(table, str):
            if isinstance(table, (dict, list)):
                table = json.dumps(table, ensure_ascii=False)
            else:
                table = str(table)
        return cls(
            id=id_str,
            title=title or "",
            text=text or "",
            table=table or "",
            file_id=d.get("file_id"),
            element_ids=_normalize_element_ids(d),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转为可写入 state / 向下游传递的 dict。"""
        return self.model_dump()
