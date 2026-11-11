from __future__ import annotations
from typing import Any, List, Literal, Optional, Union, Dict
from pydantic import BaseModel, Field, RootModel, model_validator, ConfigDict
import json
from pathlib import Path

from src.applications.parsing.entity.excel_region import ExcelRegion
from loguru import logger

class BBox(BaseModel):
    """
    ConfigDict(extra="ignore")表示：
    当输入数据中包含模型未定义的额外字段时，这些字段会被忽略（不报错、不出现在实例上）
    - 多余字段时报错：设置 model_config = ConfigDict(extra="forbid")
    - 保留多余字段：设置 model_config = ConfigDict(extra="allow")
    - Pydantic v2 的默认行为是忽略多余字段(ignore)
    """
    # 因为MinerU经常修改schema，所以设置为allow来提高鲁棒性，在合适的时候再进行严格校验
    model_config = ConfigDict(extra="allow")

    l: float
    t: float
    r: float
    b: float
    coord_origin: Literal["TOPLEFT"] = "TOPLEFT"  # 修改：根据MinerU schema，coord_origin是必填字段且固定为"TOPLEFT"


class TextItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    page: int
    type: Literal["text"]
    bBox: BBox
    value: str
    md: str
    file_id: str
    textLevel: Optional[int] = None
    headingLevel: Optional[int] = None
    element_id: int


class TableCell(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    rowspan: int = 1
    colspan: int = 1
    rowstart: Optional[int] = None # 单元格起始行索引（0起始）
    colstart: Optional[int] = None # 单元格起始列索引（0起始）


class HeaderCell(BaseModel):
    """Header cell model for table header structure"""
    model_config = ConfigDict(extra="allow")
    
    text: str
    rowspan: int = 1
    colspan: int = 1
    rowstart: Optional[int] = None # 单元格起始行索引（0起始）
    colstart: Optional[int] = None # 单元格起始列索引（0起始）


class Region(BaseModel):
    """MinerU 表格子区域（1 起始，闭区间，保留历史契约）。"""
    model_config = ConfigDict(extra="allow")

    rowstart: int = Field(..., ge=1)
    rowend: int = Field(..., ge=1)
    colstart: int = Field(..., ge=1)
    colend: int = Field(..., ge=1)


class TableItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    page: int
    type: Literal["table"]
    bBox: BBox
    rows: List[List[TableCell]]
    md: str
    html: str
    perfectTable: Optional[bool] = None  # schema 中未强制要求
    img_path: Optional[str] = None  # 适用于 table/image，但对 table 非必填
    table_type: Optional[str] = None  # 仅适用于 table，可选
    table_caption: Optional[List[str]] = None  # 根据 schema 可选
    
    table_footnote: Optional[List[str]] = None  # 可能存在 table_footnote
    standard2D: bool  # 是否为标准二维表（无跨行/跨列）
    hasHeader: bool  # 是否包含表头（列标题）
    headerCells: Optional[List[List[HeaderCell]]] = None  # 完整表头结构
    two_d_data: Optional[Dict[str, Dict[str, Union[float, int, str, bool, None]]]] = None  # 二维数据字典
    file_id: str
    element_id: int


class ImageItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    page: int
    type: Literal["image"]
    bBox: BBox
    img_path: str
    image_caption: Optional[List[str]] = None
    image_footnote: Optional[List[str]] = None
    file_id: str
    element_id: int


class PageNumberItem(BaseModel):
    """新增：根据MinerU schema，支持page_number类型"""
    model_config = ConfigDict(extra="allow")

    page: int
    type: Literal["page_number"]
    bBox: BBox
    value: str
    md: str
    file_id: str
    element_id: int


class ListItem(BaseModel):
    """列表类型（list）"""
    model_config = ConfigDict(extra="allow")

    type: Literal["list"]
    value: Optional[str] = None
    md: Optional[str] = None
    sub_type: Optional[str] = None
    list_items: Optional[List[str]] = None
    page: Optional[int] = None
    bBox: Optional[BBox] = None
    file_id: str
    element_id: int


class ExcelTableItem(BaseModel):
    """Excel 表格类型（excel_table）"""
    model_config = ConfigDict(extra="allow")

    type: Literal["excel_table"]
    rows: List[List[TableCell]]
    html: str
    md: str
    sheetName: str
    sheetIndex: int = Field(..., ge=0)
    region: ExcelRegion
    tableSubtype: Literal["structured", "semi_structured", "unstructured"]
    hasHeader: bool
    standard2D: bool
    headerCells: Optional[List[List[HeaderCell]]] = None
    two_d_data: Optional[Dict[str, Dict[str, Union[float, int, str, bool, None]]]] = None
    file_id: str
    element_id: int

Item = Union[TextItem, TableItem, ImageItem, PageNumberItem, ListItem, ExcelTableItem]


def extract_text_yusys_ai(item: Item) -> Optional[str]:
    """
    统一的文本提取函数：
    - TextItem: 返回 md
    - TableItem: 返回 md（html token 过多）
    - 其他类型暂不参与文本提取，返回 None
    """ 
    # logger.info("正在执行extract_text_yusys_ai函数")
    if isinstance(item, TextItem):
        # 将md字段视为value
        value = item.md
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("警告：TextItem的md字段为空，已跳过。")
        return None

    if isinstance(item, TableItem):
        # 将md字段视为value
        value = item.md
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("警告：TableItem的md字段为空，已跳过。")
        return None

    if isinstance(item, ListItem):
        # 将md字段视为value
        value = item.md
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("警告：ListItem的md字段为空，已跳过。")
        return None

    if isinstance(item, ExcelTableItem):
        # 将md字段视为value
        value = item.md
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("警告：ExcelTableItem的md字段为空，已跳过。")
        return None

    if isinstance(item, ImageItem):
        # 部分 image 元素（如组织架构图）被 MinerU 解析后会在 value/md 中存放
        # mermaid 流程图文字，尝试提取其中任意非空文本
        value = getattr(item, "md", None) or getattr(item, "value", None)
        if isinstance(value, str) and value.strip():
            return value
        return None

    if isinstance(item, PageNumberItem):
        # 不确定如何处理的字段type = page_number，暂时忽略
        # logger.info("PageNumberItem，不确定后续是否保留，暂时不进行解析")
        return None

    # 其他类型（如 ImageItem）不参与文本提取
    logger.warning(f"警告：暂时不支持解析为文字的条目类型{item.type}，已跳过。")
    return None

def extract_text_only_table(item: Item) -> Optional[str]:
    """
    文本提取函数，仅提取TableItem的html字段：
    供表格检索使用
    """ 
    # logger.info("正在执行extract_text_only_table函数")

    if isinstance(item, TableItem):
        # 将html字段视为value
        value = item.html
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("警告：TableItem的html字段为空，已跳过。")
        return None

    # 其他类型不参与文本提取
    # logger.info(f"对于TableItem以外的类型，不进行文本提取。")
    return None


class DocumentItems(RootModel[List[Item]]):
    @staticmethod
    def _normalize_candidate_item(obj: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """
        对 MinerU 不同版本的 item 做轻量归一化，避免因字段细节变化导致整批校验失败。
        """
        patched = dict(obj)

        # 兼容 bbox 命名差异（MinerU 版本间常见变化）
        if "bBox" not in patched and isinstance(patched.get("bbox"), dict):
            patched["bBox"] = patched["bbox"]

        # bBox 缺省坐标原点时兜底为 TOPLEFT（与当前模型默认值一致）
        if isinstance(patched.get("bBox"), dict):
            bbox = dict(patched["bBox"])
            bbox.setdefault("coord_origin", "TOPLEFT")
            patched["bBox"] = bbox

        # 兼容 file_id / element_id 命名与缺失
        if not isinstance(patched.get("file_id"), str) or not patched.get("file_id"):
            file_id = patched.get("fileId") or patched.get("doc_id") or patched.get("docId")
            if isinstance(file_id, str) and file_id:
                patched["file_id"] = file_id
            else:
                patched["file_id"] = "__unknown__"
        if not isinstance(patched.get("element_id"), int):
            patched["element_id"] = idx

        item_type = patched.get("type")
        # text/page_number/list 在部分版本仅提供 value，补齐 md 便于后续处理
        if item_type in {"text", "page_number", "list"}:
            if not isinstance(patched.get("md"), str):
                value = patched.get("value")
                patched["md"] = value if isinstance(value, str) else ""

        # table/excel_table 在新版本可能缺省部分字段，做最小兜底
        if item_type in {"table", "excel_table"}:
            if not isinstance(patched.get("rows"), list):
                patched["rows"] = []
            if not isinstance(patched.get("md"), str):
                patched["md"] = ""
            if not isinstance(patched.get("html"), str):
                patched["html"] = ""
            if not isinstance(patched.get("hasHeader"), bool):
                patched["hasHeader"] = False
            if not isinstance(patched.get("standard2D"), bool):
                patched["standard2D"] = False

        return patched

    @model_validator(mode="before")
    @classmethod
    def _coerce_items(cls, value):  # type: ignore[override]
        if not isinstance(value, list):
            raise TypeError("Top-level JSON must be a list of items")
        coerced: List[Item] = []
        for idx, obj in enumerate(value):
            if not isinstance(obj, dict):
                continue
            obj = cls._normalize_candidate_item(obj, idx)
            item_type = obj.get("type")
            try:
                if item_type == "text":
                    coerced.append(TextItem.model_validate(obj))
                elif item_type == "table":
                    coerced.append(TableItem.model_validate(obj))
                elif item_type == "image":
                    coerced.append(ImageItem.model_validate(obj))
                elif item_type == "page_number":  # 新增：支持page_number类型
                    coerced.append(PageNumberItem.model_validate(obj))
                elif item_type == "list":  # 支持 list 类型
                    coerced.append(ListItem.model_validate(obj))
                elif item_type == "excel_table":  # 支持 excel_table 类型
                    coerced.append(ExcelTableItem.model_validate(obj))
                else:
                    logger.warning("Item解析失败: {}", obj)
                    # Unknown type; skip silently to be robust
                    continue
            except Exception as e:
                # 兼容策略：单条不合法时跳过，避免整批请求因新旧 schema 差异失败
                logger.warning(
                    "Item校验失败已跳过 | idx={}, type={}, error={}, keys={}",
                    idx,
                    item_type,
                    e,
                    sorted(list(obj.keys())) if isinstance(obj, dict) else [],
                )
                continue
        return coerced

def load_items(path: Path) -> DocumentItems:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return DocumentItems.model_validate(data)