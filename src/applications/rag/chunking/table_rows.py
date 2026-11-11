from __future__ import annotations

"""表格按行切分工具。

将过大的表格转换为多个带表头的 Markdown 片段，便于后续向量化与检索保留行级
语义。
"""

from typing import Any, Dict, List, Optional

from ..utils.chunking_helpers import cell_text as _cell_text
from ..vector_document import utf8_len


def _format_table_header_md(
    header_cells: Optional[List[List[Dict[str, Any]]]],
    rows: List[List[Any]],
    has_header: bool,
) -> str:
    """将表头区域格式化为 Markdown 表头。

    Args:
        header_cells: 结构化表头单元格数据。
        rows: 原始表格行数据。
        has_header: 表格是否显式声明包含表头。

    Returns:
        str: Markdown 表头字符串；如果无法生成有效表头则返回空字符串。
    """

    if not has_header:
        return ""

    header_texts: List[str] = []

    if header_cells and isinstance(header_cells, list):
        for row in header_cells:
            if not isinstance(row, list):
                continue
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                text = str(cell.get("text", "")).strip()
                if text:
                    header_texts.append(text)
        if not header_texts and header_cells:
            last_row = header_cells[-1] if header_cells else []
            if isinstance(last_row, list):
                header_texts = [
                    str(cell.get("text", "") if isinstance(cell, dict) else cell).strip()
                    for cell in last_row
                ]
    elif rows and isinstance(rows, list):
        first_row = rows[0]
        if isinstance(first_row, list):
            header_texts = [_cell_text(cell) for cell in first_row]

    if not header_texts:
        return ""

    header_line = "| " + " | ".join(header_texts) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header_texts)) + " |"
    return header_line + "\n" + separator_line


def _format_table_row_md(row: List[Any]) -> str:
    """将单行表格数据格式化为 Markdown 行。

    Args:
        row: 单行表格数据。

    Returns:
        str: 格式化后的 Markdown 行文本；无效输入时返回空字符串。
    """

    if not row or not isinstance(row, list):
        return ""
    cells = [_cell_text(cell) for cell in row]
    return "| " + " | ".join(cells) + " |"


def split_table_by_rows(
    entry: Dict[str, Any],
    text_limit: int,
    element_id: Any,
) -> List[Dict[str, Any]]:
    """按行切分大表格，并为每个片段补齐表头。

    Args:
        entry: 归一化后的表格结构，包含 `rows`、`hasHeader`、`headerCells` 等字段。
        text_limit: 每个输出片段允许的 UTF-8 字节上限。
        element_id: 原始表格元素 ID，用于回填元数据。

    Returns:
        List[Dict[str, Any]]: 按行切分后的片段列表。每个片段包含文本内容、来源 ID、
        切分策略以及覆盖的起止行号。
    """

    rows = entry.get("rows") or []
    if not rows:
        return []

    has_header = bool(entry.get("hasHeader"))
    header_cells = entry.get("headerCells") if isinstance(entry.get("headerCells"), list) else None
    header_md = _format_table_header_md(header_cells, rows, has_header)
    header_len = utf8_len(header_md) if header_md else 0

    data_start = 0
    if has_header:
        # 表头行只用于生成上下文，不再重复计入数据行切分范围。
        data_start = len(header_cells) if header_cells else 1

    pieces: List[Dict[str, Any]] = []
    current_rows: List[str] = []
    current_len = header_len
    current_row_start: Optional[int] = None
    current_row_end: Optional[int] = None

    def flush_piece() -> None:
        nonlocal current_rows, current_len, current_row_start, current_row_end
        if not current_rows:
            return
        if header_md:
            piece_text = header_md + "\n" + "\n".join(current_rows)
        else:
            piece_text = "\n".join(current_rows)
        piece_text = piece_text.strip()
        if piece_text:
            pieces.append(
                {
                    "text": piece_text,
                    "ids": [element_id] if element_id is not None else [],
                    "strategy": "table_row",
                    "row_start": current_row_start,
                    "row_end": current_row_end,
                }
            )
        current_rows = []
        current_len = header_len
        current_row_start = None
        current_row_end = None

    for row_index in range(data_start, len(rows)):
        row = rows[row_index]
        if not isinstance(row, list):
            continue
        row_md = _format_table_row_md(row)
        if not row_md:
            continue
        row_len = utf8_len(row_md)

        # 单行已超过阈值时，仍单独保留该行，避免内容丢失。
        if header_len + row_len > text_limit:
            if current_rows:
                flush_piece()
            if header_md:
                piece_text = header_md + "\n" + row_md
            else:
                piece_text = row_md
            pieces.append(
                {
                    "text": piece_text.strip(),
                    "ids": [element_id] if element_id is not None else [],
                    "strategy": "table_row",
                    "row_start": row_index,
                    "row_end": row_index,
                }
            )
            continue

        separator_len = 1 if current_rows else 0
        new_len = current_len + separator_len + row_len
        if new_len > text_limit and current_rows:
            flush_piece()
            new_len = header_len + row_len

        if current_row_start is None:
            current_row_start = row_index
        current_rows.append(row_md)
        current_row_end = row_index
        current_len = new_len

    flush_piece()
    return pieces


__all__ = ["split_table_by_rows"]
