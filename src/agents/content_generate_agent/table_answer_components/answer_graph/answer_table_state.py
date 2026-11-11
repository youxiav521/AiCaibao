from typing import Any, Dict, List, Optional, Union
from langgraph.graph import MessagesState
from pydantic import BaseModel

from src.agents.content_generate_agent.input_models import Item
from src.agents.content_generate_agent.models import QuestionTask
from src.agents.content_generate_agent.utils import get_localized_message


class FillResult(BaseModel):
    """
    填表结果统一类型：成功为 html，失败为 error + filled_table_html。
    用于解析 workflow 返回的 str/dict，避免下游对类型做分支判断。
    """
    html: Optional[str] = None
    error: Optional[str] = None
    filled_table_html: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Union[str, dict, None], locale: str = "zh_CN") -> "FillResult":
        """从当前链路中的 str 或 dict 解析为 FillResult。"""
        if raw is None:
            return cls(error=get_localized_message("table_result_empty", locale))
        if isinstance(raw, str):
            return cls(html=raw)
        if isinstance(raw, dict):
            if "error" in raw:
                return cls(
                    error=raw.get("error", get_localized_message("table_unknown_error", locale)),
                    filled_table_html=raw.get("filled_table_html") or "",
                )
            return cls(html=raw.get("html") or raw.get("filled_table_html") or "")
        return cls(error=get_localized_message("table_unknown_result_type", locale))

    def is_success(self) -> bool:
        """是否为成功结果（有填充后的 HTML）。"""
        return self.error is None

    def to_answer_text(self) -> str:
        """转为最终展示给用户的文本：成功为 HTML，失败为错误说明 + 回退 HTML。"""
        if self.is_success():
            return self.html or ""
        return (self.error or "未知错误") + "\n" + (self.filled_table_html or "")

    @classmethod
    def error_result(cls, error_msg: str, filled_table_html: str = "") -> Dict[str, Any]:
        """构造错误结果的字典，供 workflow 写入 state（保持可序列化）。"""
        return {"error": error_msg, "filled_table_html": filled_table_html}

    @classmethod
    def is_error_dict(cls, value: Any) -> bool:
        """判断是否为错误结果字典。"""
        return isinstance(value, dict) and "error" in value

    @classmethod
    def ensure_filled_html(cls, fill_result: Dict[str, Any], fallback_html: str) -> Dict[str, Any]:
        """若错误结果中缺少 filled_table_html，用 fallback 补充，返回新 dict。"""
        if "filled_table_html" in fill_result and fill_result["filled_table_html"] != "":
            return fill_result
        return {**fill_result, "filled_table_html": fallback_html}


class TableAnswerState(MessagesState):
    """
    LangGraph State for Table Answering with message passing.
    """
    content_list: List[Item]
    question_task: QuestionTask
    table_with_documents: Optional[Dict[str, Any]] = None
    question_citations: Optional[List[Any]] = None
    merged_table_result: Optional[Any] = None
    table_items: Optional[List[Item]] = None
    result: Optional[Any] = None
    model_id: Optional[str] = None
    disable_keyword_extract: Optional[bool] = True
    skip_fill: Optional[bool] = False
    answer_lang: Optional[str] = "zh_CN"
    max_iterations: Optional[int] = 5
