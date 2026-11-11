from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class CoordOrigin(str, Enum):
    BOTTOMLEFT = "BOTTOMLEFT"
    TOPLEFT = "TOPLEFT"


class BBox(BaseModel):
    """边界框。输入 JSON 使用 l/t/r/b 与 coord_origin。

    额外提供只读的便捷属性：x、y、w、h、left、top、right、bottom。
    注意：由于无页面高度，y 直接使用 t；h 使用 |b - t|。
    """

    l: float = Field(..., description="left")
    t: float = Field(..., description="top")
    r: float = Field(..., description="right")
    b: float = Field(..., description="bottom")
    coord_origin: Optional[CoordOrigin] = Field(
        default=None, description="坐标系原点（可能为 BOTTOMLEFT 或 TOPLEFT）"
    )

    @property
    def left(self) -> float:
        return float(self.l)

    @property
    def right(self) -> float:
        return float(self.r)

    @property
    def top(self) -> float:
        return float(self.t)

    @property
    def bottom(self) -> float:
        return float(self.b)

    @property
    def x(self) -> float:
        return self.left

    @property
    def y(self) -> float:
        # 无页面高度信息，直接暴露 t 值
        return self.top

    @property
    def w(self) -> float:
        return abs(float(self.right) - float(self.left))

    @property
    def h(self) -> float:
        return abs(float(self.bottom) - float(self.top))

    def to_tuple(self) -> tuple:
        return (self.l, self.t, self.r, self.b)


class Element(BaseModel):
    """文档元素基类。"""

    page: Optional[int] = Field(default=None, description="页码（从 1 开始）")
    type: str = Field(..., description="元素类型标签")
    bBox: Optional[BBox] = Field(default=None, description="边界框")
    md: Optional[str] = Field(default=None, description="Markdown 表示")

    @property
    def text(self) -> str:
        """便捷访问文本内容。子类可覆盖。"""
        return ""


class TextElement(Element):
    value: Optional[str] = Field(default=None, description="原始文本")

    @property
    def text(self) -> str:
        return self.value or ""


class Heading(TextElement):
    """标题元素。逻辑上等同于 TextElement，仅作语义区分。"""


class TableCell(BaseModel):
    text: str = Field(default="")
    row_span: int = Field(default=1)
    col_span: int = Field(default=1)
    bBox: Optional[BBox] = None
    # Excel 溯源：单元格起始行/列（1-based）
    rowstart: Optional[int] = None
    colstart: Optional[int] = None


class TableElement(Element):
    rows: List[List[TableCell]] = Field(default_factory=list)
    html: Optional[str] = None
    perfectTable: bool = Field(default=False)
    standard2D: bool = Field(default=False)
    hasHeader: bool = Field(default=False)
    header: Optional[List[str]] = None
    headerCells: Optional[List[List[Dict[str, Any]]]] = None
    # Excel 溯源：表所在工作表与区域
    sheetName: Optional[str] = None
    sheetIndex: Optional[int] = None
    tableSubtype: Optional[str] = None  # structured | semi_structured | unstructured
    region: Optional[Dict[str, int]] = None  # {rowstart,rowend,colstart,colend}
    # md 字段已在父类

    @property
    def text(self) -> str:
        """拼接表格所有单元格文本，按行用换行、列用制表符连接。"""
        lines: List[str] = []
        for row in self.rows:
            cols = [cell.text if cell and cell.text is not None else "" for cell in row]
            lines.append("\t".join(cols))
        return "\n".join(lines)

    def to_2d_array(self) -> List[List[str]]:
        return [[cell.text for cell in row] for row in self.rows]


class ConvertedDocument(BaseModel):
    """整个转换后 JSON 的对象化封装。"""

    items: List[Element] = Field(default_factory=list)

    # -------- 构造/加载 --------
    @classmethod
    def from_json_data(cls, data: Iterable[Dict[str, Any]]) -> "ConvertedDocument":
        elements: List[Element] = []
        for obj in data:
            try:
                elements.append(_element_from_dict(obj))
            except Exception:
                # 忽略单条解析失败
                continue
        return cls(items=elements)

    @classmethod
    def from_json_file(cls, path: str) -> "ConvertedDocument":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("转换后的 JSON 顶层应为数组")
        return cls.from_json_data(data)

    # -------- 便捷查询 API --------
    def by_page(self, page: int) -> List[Element]:
        return [e for e in self.items if e.page == page]

    def texts(self) -> List[TextElement]:
        return [e for e in self.items if isinstance(e, TextElement) and not isinstance(e, Heading)]

    def headings(self) -> List[Heading]:
        return [e for e in self.items if isinstance(e, Heading)]

    def tables(self) -> List[TableElement]:
        return [e for e in self.items if isinstance(e, TableElement)]

    def find_by_type(self, *types: str) -> List[Element]:
        s = {t.lower() for t in types}
        return [e for e in self.items if e.type.lower() in s]


# ---------------- 工具方法 -----------------
def _parse_bbox(raw: Optional[Dict[str, Any]]) -> Optional[BBox]:
    if not raw:
        return None
    try:
        # 兼容 coord_origin 大小写
        origin = raw.get("coord_origin")
        if isinstance(origin, str):
            try:
                raw["coord_origin"] = CoordOrigin(origin.upper())
            except Exception:
                raw["coord_origin"] = origin
        return BBox(**raw)
    except Exception:
        return None


def _parse_cell(raw: Dict[str, Any]) -> TableCell:
    # 使用 rowstart/colstart
    r_idx_val = raw.get("rowstart")
    c_idx_val = raw.get("colstart")

    return TableCell(
        text=str(raw.get("text") or ""),
        row_span=int(raw.get("row_span") or 1),
        col_span=int(raw.get("col_span") or 1),
        bBox=_parse_bbox(raw.get("bBox")),
        rowstart=(int(r_idx_val) if r_idx_val is not None else None),
        colstart=(int(c_idx_val) if c_idx_val is not None else None),
    )


def _element_from_dict(obj: Dict[str, Any]) -> Element:
    common: Dict[str, Any] = {
        "page": obj.get("page"),
        "type": str(obj.get("type") or ""),
        "bBox": _parse_bbox(obj.get("bBox")),
        "md": obj.get("md"),
    }

    # 表格
    if "rows" in obj:
        raw_rows: List[List[Dict[str, Any]]] = obj.get("rows") or []
        rows: List[List[TableCell]] = [[_parse_cell(c) for c in row] for row in raw_rows]
        return TableElement(
            **common,
            rows=rows,
            html=obj.get("html"),
            perfectTable=bool(obj.get("perfectTable") or False),
            standard2D=bool(obj.get("standard2D") or False),
            hasHeader=bool(obj.get("hasHeader") or False),
            header=obj.get("header"),
            headerCells=obj.get("headerCells"),
            sheetName=obj.get("sheetName"),
            sheetIndex=(int(obj.get("sheetIndex")) if obj.get("sheetIndex") is not None else None),
            tableSubtype=obj.get("tableSubtype"),
            region=obj.get("region"),
        )

    # 文本/标题
    value = obj.get("value")
    element_cls = TextElement
    t = (common["type"] or "").lower()
    if t in {"section_header", "header", "heading", "title"}:
        element_cls = Heading
    else:
        heading_level = obj.get("headingLevel")
        if heading_level is not None:
            try:
                if int(heading_level) > 0:
                    element_cls = Heading
            except Exception:
                pass
    return element_cls(**common, value=value)


__all__ = [
    "CoordOrigin",
    "BBox",
    "Element",
    "TextElement",
    "Heading",
    "TableCell",
    "TableElement",
    "ConvertedDocument",
    "_element_from_dict",
]


