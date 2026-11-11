from __future__ import annotations
import csv
import html as _html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from loguru import logger
from openpyxl import load_workbook

from src.utils.excel_display import excel_cell_to_display_text

_ROW_HEIGHT_PX = 20.0
_COL_WIDTH_PX = 100.0

class ExcelParsingService:
    """直接解析 Excel，生成与 MinerU 兼容的条目，并保留 Excel 溯源信息。

    输出的表包含：
      - sheetName, sheetIndex（工作表名称与索引）
      - region {rowstart,rowend,colstart,colend}（从 0 开始，闭区间，与 cell.rowstart/colstart 一致）
      - tableSubtype: structured | semi_structured | unstructured（表类型）
      - hasHeader（是否检测到表头）
      - rows：包含 rowstart/colstart 与合并信息的单元格
    """

    def parse_to_items(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        # CSV 文件单独处理
        if path.suffix.lower() == ".csv":
            return self._parse_csv(path)



        def _parse_once(data_only_flag: bool) -> List[Dict[str, Any]]:
            wb_local = load_workbook(filename=str(path), read_only=False, data_only=data_only_flag)
            items_local: List[Dict[str, Any]] = []
            for sheet_idx, ws in enumerate(wb_local.worksheets):
                try:
                    sheet_items = self._parse_sheet(ws, sheet_idx)
                    items_local.extend(sheet_items)
                except Exception as e:
                    logger.warning(f"Failed to parse sheet {ws.title!r}: {e}")
            return items_local

        def _has_data_rows(items_in: List[Dict[str, Any]]) -> bool:
            for it in items_in or []:
                try:
                    region = it.get("region") or {}
                    rs = int(region.get("rowstart", 0))
                    re = int(region.get("rowend", 0))
                    if re > rs:
                        return True
                except Exception:
                    continue
            return False

        # 第一次：优先读取缓存值（若表内为公式且无缓存，可能导致仅识别到表头）
        items = _parse_once(data_only_flag=True)
        if not _has_data_rows(items):
            # 回退：读取公式为文本，避免把含公式的数据行当作空值丢弃
            logger.info("ExcelParsingService: fallback with data_only=False due to header-only regions detected")
            items_fallback = _parse_once(data_only_flag=False)
            # 若回退后包含数据行，则采用回退结果
            if _has_data_rows(items_fallback):
                return items_fallback

        return items

    # --------------- CSV 解析 ---------------
    def _parse_csv(self, path: Path) -> List[Dict[str, Any]]:
        """支持多表格自动拆分。"""

        encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]
        raw_grid: List[List[str]] = []
        
        # 1. 读取 CSV 到二维网格
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    sample = f.read(8192)
                    f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    except csv.Error:
                        dialect = csv.excel
                    reader = csv.reader(f, dialect)
                    raw_grid = [row for row in reader]
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if not raw_grid:
            return []

        # 2. 封装为适配器对象
        adapter = CSVSheetAdapter(raw_grid, title=path.stem)
        
        # 3. 直接复用 Excel 的解析逻辑 (自动处理空行、空列拆分、表头检测)
        # sheet_index 默认为 0
        return self._parse_sheet(adapter, 0)

    # --------------- 工作表解析 ---------------
    def _parse_sheet(self, ws, sheet_index: int) -> List[Dict[str, Any]]:
        """将一个工作表解析为一个或多个表条目。"""
        sheet_name = ws.title or f"Sheet{sheet_index+1}"

        # 先获取合并单元格信息，用于正确判断行是否为空
        merge_top_left_to_span, covered_cells = self._merged_map(ws)
        
        # 计算每行被合并单元格覆盖的列（这些位置虽然值为 None，但不应视为空）
        rows_with_merge_coverage = self._get_rows_covered_by_merges(merge_top_left_to_span)

        # 发现包含非空内容的连续行块
        row_blocks: List[Tuple[int, int]] = []
        in_block = False
        block_start = 0

        max_row = ws.max_row or 1

        # 流式遍历行
        for r_idx, row in enumerate(ws.iter_rows(values_only=False), start=1):
            # 检测该行是否有非空单元格
            non_empty_cols = [c_idx for c_idx, cell in enumerate(row, start=1) if self._is_non_empty_cell(cell)]
            # 检测该行是否被合并单元格覆盖（即使值为 None 也不算空行）
            has_merge_coverage = r_idx in rows_with_merge_coverage
            
            if non_empty_cols or has_merge_coverage:
                if not in_block:
                    in_block = True
                    block_start = r_idx
            else:
                if in_block:
                    row_blocks.append((block_start, r_idx - 1))
                    in_block = False
        if in_block:
            # 收尾最后一个行块
            row_blocks.append((block_start, max_row))

        if not row_blocks:
            return []

        items: List[Dict[str, Any]] = []
        # 将每个行块按完全空列拆分为多个列段
        for (r_start, r_end) in row_blocks:
            col_segments = self._split_by_empty_columns(ws, r_start, r_end, merge_top_left_to_span)
            for (c_start, c_end) in col_segments:
                region = (r_start, r_end, c_start, c_end)
                items.extend(self._tables_from_region(ws, sheet_name, sheet_index, region))

        return items

    def _get_rows_covered_by_merges(
        self, merge_top_left_to_span: Dict[Tuple[int, int], Tuple[int, int]]
    ) -> Set[int]:
        """返回所有被合并单元格覆盖的行号集合。
        
        对于跨多行的合并单元格，其覆盖的所有行都应该被标记为"非空"，
        即使这些行在 iter_rows(values_only=True) 中返回 None。
        """
        covered_rows: Set[int] = set()
        for (top_row, top_col), (row_span, col_span) in merge_top_left_to_span.items():
            for r in range(top_row, top_row + row_span):
                covered_rows.add(r)
        return covered_rows

    # --------------- 区域 -> 表 ---------------
    def _tables_from_region(
        self,
        ws,
        sheet_name: str,
        sheet_index: int,
        region: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        row_start, row_end, col_start, col_end = region
        # 跳过无效/空区域
        if row_end < row_start or col_end < col_start:
            return []

        # 在区域的前最多 10 行内进行表头检测
        header_row_rel_index = self._detect_header_row(ws, row_start, row_end, col_start, col_end)

        tables: List[Dict[str, Any]] = []

        # 表头之前的前言行（若存在）作为非结构化输出
        if header_row_rel_index is not None and header_row_rel_index > 0:
            preface_start = row_start
            preface_end = row_start + header_row_rel_index - 1
            if self._region_has_any_content(ws, preface_start, preface_end, col_start, col_end):
                tables.append(
                    self._build_table_item(
                        ws,
                        sheet_name,
                        sheet_index,
                        preface_start,
                        preface_end,
                        col_start,
                        col_end,
                        subtype="unstructured",
                        header_info=None,
                    )
                )

        if header_row_rel_index is not None:
            header_abs_row = row_start + header_row_rel_index
            # 数据部分包含表头行作为第一行以保持输出一致性
            subtype = "structured"
            if header_row_rel_index > 0:
                subtype = "semi_structured"
            header_values = self._row_texts(ws, header_abs_row, col_start, col_end)
            header_info = (True, header_values)
            tables.append(
                self._build_table_item(
                    ws,
                    sheet_name,
                    sheet_index,
                    header_abs_row,
                    row_end,
                    col_start,
                    col_end,
                    subtype=subtype,
                    header_info=header_info,
                )
            )
        else:
            # 未检测到表头时，整个区域作为非结构化输出
            tables.append(
                self._build_table_item(
                    ws,
                    sheet_name,
                    sheet_index,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                    subtype="unstructured",
                    header_info=None,
                )
            )

        return tables

    # --------------- 构建表条目 ---------------
    def _build_table_item(
        self,
        ws,
        sheet_name: str,
        sheet_index: int,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        subtype: str,
        header_info: Optional[Tuple[bool, List[str]]],
    ) -> Dict[str, Any]:
        merge_top_left_to_span, covered_cells = self._merged_map(ws)

        rows: List[List[Dict[str, Any]]]= []
        for r in range(row_start, row_end + 1):
            row_cells: List[Dict[str, Any]] = []
            for c in range(col_start, col_end + 1):
                if (r, c) in covered_cells:
                    # 被合并覆盖（非左上角单元格）：跳过，下游靠 colspan/rowspan 自行展开
                    continue

                cell = ws.cell(row=r, column=c)
                span = merge_top_left_to_span.get((r, c), (1, 1))
                cell_obj: Dict[str, Any] = {
                    "text": self._cell_to_text(cell),
                    "rowspan": int(span[0]),
                    "colspan": int(span[1]),
                    "rowstart": r - 1,
                    "colstart": c - 1,
                    "bBox": {
                        "l": float((c - 1) * _COL_WIDTH_PX),
                        "t": float((r - 1) * _ROW_HEIGHT_PX),
                        "r": float((c - 1 + int(span[1])) * _COL_WIDTH_PX),
                        "b": float((r - 1 + int(span[0])) * _ROW_HEIGHT_PX),
                        "coordOrigin": "TOPLEFT",
                    },
                }
                row_cells.append(cell_obj)
            rows.append(row_cells)

        # 使用整表展开（包含合并单元格的纵向展开）生成 Markdown
        def _render_markdown_full(rows_in: List[List[Dict[str, Any]]]) -> str:
            grid = ExcelParsingService.rows_to_markdown_grid(rows_in)
            if not grid:
                return ""
            def _join(line: List[str]) -> str:
                return "| " + " | ".join(line) + " |"
            lines: List[str] = []
            for rr_idx, row_vals in enumerate(grid):
                lines.append(_join(row_vals))
                if rr_idx == 0:
                    lines.append("| " + " | ".join(["---"] * len(row_vals)) + " |")
            return "\n".join(lines)

        has_header = bool(header_info[0]) if header_info is not None else False

        item: Dict[str, Any] = {
            "type": "excel_table",
            "page": sheet_index,
            "sheetName": sheet_name,
            "sheetIndex": sheet_index,
            "region": {
                "rowstart": row_start - 1,
                "rowend": row_end - 1,
                "colstart": col_start - 1,
                "colend": col_end - 1,
            },
            "bBox": {
                "l": float((col_start - 1) * _COL_WIDTH_PX),
                "t": float((row_start - 1) * _ROW_HEIGHT_PX),
                "r": float(col_end * _COL_WIDTH_PX),
                "b": float(row_end * _ROW_HEIGHT_PX),
                "coordOrigin": "TOPLEFT",
            },            
            "tableSubtype": subtype,
            "hasHeader": bool(has_header),
            "headerCells":  [],
            "standard2D": subtype in ("structured", "semi_structured"),
            "rows": rows,
            "md": _render_markdown_full(rows),
            "html": self.rows_to_html(rows),
        }
        return item

    # --------------- 帮助方法 ---------------
    def _render_markdown_from_rows(self, rows_in: List[List[Dict[str, Any]]]) -> str:
        """将 rows 转换为 Markdown 表格格式。"""
        grid = ExcelParsingService.rows_to_markdown_grid(rows_in)
        if not grid:
            return ""
        def _join(line: List[str]) -> str:
            return "| " + " | ".join(line) + " |"
        lines: List[str] = []
        for rr_idx, row_vals in enumerate(grid):
            lines.append(_join(row_vals))
            if rr_idx == 0:
                lines.append("| " + " | ".join(["---"] * len(row_vals)) + " |")
        return "\n".join(lines)

    def _split_by_empty_columns(
        self, 
        ws, 
        r_start: int, 
        r_end: int,
        merge_top_left_to_span: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None
    ) -> List[Tuple[int, int]]:
        """将 [r_start, r_end] 按列的完全空白情况拆分为若干列段。

        返回覆盖非空列的连续区间列表 (c_start, c_end)。
        同时考虑合并单元格覆盖的列。
        """
        # 计算被合并单元格覆盖的列
        cols_covered_by_merges: Set[int] = set()
        if merge_top_left_to_span:
            for (top_row, top_col), (row_span, col_span) in merge_top_left_to_span.items():
                # 检查该合并单元格是否与当前行范围有交集
                merge_row_end = top_row + row_span - 1
                if top_row <= r_end and merge_row_end >= r_start:
                    for c in range(top_col, top_col + col_span):
                        cols_covered_by_merges.add(c)

        # 先统计该行块内各列是否存在非空值
        col_has_value: Dict[int, bool] = {}
        max_col = 0
        for r in range(r_start, r_end + 1):
            row = ws.iter_rows(min_row=r, max_row=r, values_only=False)
            for row_cells in row:
                for c_idx, cell in enumerate(row_cells, start=1):
                    if self._is_non_empty_cell(cell):
                        col_has_value[c_idx] = True
                    max_col = max(max_col, c_idx)

        # 将被合并单元格覆盖的列也标记为有值
        for c in cols_covered_by_merges:
            col_has_value[c] = True
            max_col = max(max_col, c)

        non_empty_cols = [c for c in range(1, max_col + 1) if col_has_value.get(c)]
        if not non_empty_cols:
            return []

        segments: List[Tuple[int, int]] = []
        seg_start = non_empty_cols[0]
        prev = seg_start
        for c in non_empty_cols[1:]:
            if c == prev + 1:
                prev = c
                continue
            segments.append((seg_start, prev))
            seg_start = c
            prev = c
        segments.append((seg_start, prev))

        return segments

    def _detect_header_row(self, ws, r_start: int, r_end: int, c_start: int, c_end: int) -> Optional[int]:
        limit = min(r_end, r_start + 9)
        best_row_rel: Optional[int] = None
        best_weight = 0
        fallback_row_rel: Optional[int] = None
        fallback_ratio = 0.0
        for r in range(r_start, limit + 1):
            vals = self._row_values(ws, r, c_start, c_end)
            non_empty_vals = [v for v in vals if self._is_non_empty(v)]
            if len(non_empty_vals) < 2:
                continue
            # 表头权重：年份/日期/期初期末关键词计数（普通文本不计分）
            weight = sum(self._header_cell_weight(v) for v in non_empty_vals)
            # 文本占比作为兜底：无年份/期初期末的纯文本表头（如无日期的列名行）
            str_like = sum(1 for v in non_empty_vals if self._looks_like_header_text(v))
            ratio = str_like / max(1, len(non_empty_vals))
            if ratio >= 0.5 and ratio > fallback_ratio:
                fallback_ratio = ratio
                fallback_row_rel = r - r_start
            if weight > best_weight:
                best_weight = weight
                best_row_rel = r - r_start

        return best_row_rel if best_row_rel is not None else fallback_row_rel

    def _region_has_any_content(self, ws, r_start: int, r_end: int, c_start: int, c_end: int) -> bool:
        for r in range(r_start, r_end + 1):
            vals = self._row_values(ws, r, c_start, c_end)
            if any(self._is_non_empty(v) for v in vals):
                return True
        return False

    def _row_values(self, ws, r: int, c_start: int, c_end: int) -> List[Any]:
        if r < 1:
            return []
        row_iter = ws.iter_rows(min_row=r, max_row=r, min_col=c_start, max_col=c_end, values_only=False)
        for cells in row_iter:
            return [self._cell_to_text(cell) for cell in cells]
        return []

    def _row_texts(self, ws, r: int, c_start: int, c_end: int) -> List[str]:
        return [self._to_text(v) for v in self._row_values(ws, r, c_start, c_end)]

    def _merged_map(self, ws) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], set]:
        """返回：左上角单元格 -> (row_span, col_span) 的映射，以及被合并覆盖的单元格集合。
        """
        top_left_to_span = {}
        covered = set()
        # 适配 CSVSheetAdapter 或 openpyxl
        try:
            merged_cells = getattr(ws, "merged_cells", None)
            ranges = merged_cells.ranges if merged_cells else []
        except Exception:
            ranges = []
        for m in ranges:
            # m 为 CellRange：min_row, max_row, min_col, max_col
            try:
                min_row, max_row = int(m.min_row), int(m.max_row)
                min_col, max_col = int(m.min_col), int(m.max_col)
            except Exception:
                continue
            row_span = max(1, max_row - min_row + 1)
            col_span = max(1, max_col - min_col + 1)
            top_left_to_span[(min_row, min_col)] = (row_span, col_span)
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    if r == min_row and c == min_col:
                        continue
                    covered.add((r, c))
        return top_left_to_span, covered

    @staticmethod
    def _to_text(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        return s if s is not None else ""

    @staticmethod
    def _cell_to_text(cell: Any) -> str:
        try:
            return excel_cell_to_display_text(cell)
        except Exception:
            return ExcelParsingService._to_text(getattr(cell, "value", None))

    @staticmethod
    def _is_non_empty_cell(cell: Any) -> bool:
        return ExcelParsingService._is_non_empty(ExcelParsingService._cell_to_text(cell))

    @staticmethod
    def _is_non_empty(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip() != ""
        return True

    @staticmethod
    def _looks_like_header_text(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, (int, float)):
            return False
        s = str(v).strip()
        if not s:
            return False
        # 避免纯数值字符串被视为表头文本
        try:
            float(s)
            return False
        except Exception:
            return True

    @staticmethod
    def _header_cell_weight(v: Any) -> int:
        """表头单元格权重：年份/日期/期初期末关键词各计 1，普通文本计 0。

        用「计数」而非「比率」判定表头，避免「金額|％|金額|％」这类纯文本
        单位行因占比高被误选，而带年份的真正表头行因含科目名被稀释落选。
        """
        if v is None:
            return 0
        # 数字：仅 4 位年份算表头信号
        if isinstance(v, (int, float)):
            try:
                return 1 if 1900 <= int(v) <= 2100 else 0
            except (TypeError, ValueError):
                return 0
        s = str(v).strip()
        if not s:
            return 0
        # 纯 4 位年份
        if re.fullmatch(r"(19|20)\d{2}", s):
            return 1
        # 日期：2024-12-31 / 2024年12月31日 / 2024/12/31 等
        if re.search(r"(19|20)\d{2}\s*[年/.\-]\s*\d{1,2}", s):
            return 1
        # 期初/期末/年初/年末/本期/上期/本年/上年（与年份权重相等）
        if any(kw in s for kw in ("期初", "期末", "年初", "年末", "本期", "上期", "本年", "上年")):
            return 1
        return 0

    # 提供独立方法：将 Excel 解析得到的 rows 转为 HTML（保留合并信息），便于复用
    @staticmethod
    def rows_to_html(rows_in: List[List[Dict[str, Any]]]) -> str:

        parts: List[str] = ["<table>"]
        for raw_row in rows_in:
            parts.append("  <tr>")
            for cell in raw_row:
                text = _html.escape(str(cell.get("text", "") or ""))
                rs = cell.get("rowspan")
                cs = cell.get("colspan")
                attrs: List[str] = []
                try:
                    if int(rs or 1) > 1:
                        attrs.append(f"rowspan=\"{int(rs)}\"")
                except Exception:
                    pass
                try:
                    if int(cs or 1) > 1:
                        attrs.append(f"colspan=\"{int(cs)}\"")
                except Exception:
                    pass
                attr_str = (" " + " ".join(attrs)) if attrs else ""
                parts.append(f"    <td{attr_str}>{text}</td>")
            parts.append("  </tr>")
        parts.append("</table>")
        return "".join(parts)

    @staticmethod
    def rows_to_markdown_grid(rows_in: List[List[Dict[str, Any]]]) -> List[List[str]]:
        """将含有合并信息的 rows 展开为二维文本网格，用于 Markdown。

        规则：
        - 按列跨度 (colspan) 水平展开。
        - 按行跨度 (rowspan) 纵向复制顶格文本，避免 Markdown 丢失信息。
        - 网格宽度取各行 (sum colspans) 的最大值，空位以空串补齐。
        """
        if not rows_in:
            return []

        # 计算网格最大列数
        def _row_width(row: List[Dict[str, Any]]) -> int:
            w = 0
            for cell in row:
                try:
                    cs = int(cell.get("colspan") or 1)
                except Exception:
                    cs = 1
                w += max(1, cs)
            return w

        grid_width = 0
        for r in rows_in:
            grid_width = max(grid_width, _row_width(r))
        grid_width = max(1, grid_width)

        num_rows = len(rows_in)
        grid: List[List[str]] = [[""] * grid_width for _ in range(num_rows)]
        # 每列记录向下还需填充的 (剩余行数, 文本)
        carry_down: List[Optional[Tuple[int, str]]] = [None] * grid_width  # type: ignore[name-defined]

        for r_idx, raw_row in enumerate(rows_in):
            # 先填充由上一行带下来的纵向合并文本
            for c_idx in range(grid_width):
                if carry_down[c_idx] is not None and grid[r_idx][c_idx] == "":
                    remaining, text = carry_down[c_idx]
                    grid[r_idx][c_idx] = str(text)
                    remaining -= 1
                    carry_down[c_idx] = (remaining, text) if remaining > 0 else None

            # 再放置本行的单元格内容
            col_ptr = 0
            for cell in raw_row:
                text = str(cell.get("text", "") or "")
                try:
                    cs = int(cell.get("colspan") or 1)
                except Exception:
                    cs = 1
                try:
                    rs = int(cell.get("rowspan") or 1)
                except Exception:
                    rs = 1
                cs = max(1, cs)
                rs = max(1, rs)

                # 找到下一个可用列位
                while col_ptr < grid_width and grid[r_idx][col_ptr] != "":
                    col_ptr += 1
                # 写入水平展开
                for k in range(cs):
                    if r_idx < num_rows and col_ptr + k < grid_width:
                        grid[r_idx][col_ptr + k] = text
                # 记录纵向展开
                if rs > 1:
                    for k in range(cs):
                        c = col_ptr + k
                        if c < grid_width:
                            carry_down[c] = (rs - 1, text)
                col_ptr += cs

            # 行尾剩余的 carry_down 在下一行继续处理

        return grid


__all__ = ["ExcelParsingService"]



class CSVSheetAdapter:
    """模拟 openpyxl 的 worksheet 接口，使 CSV 能够复用 Excel 解析逻辑"""
    def __init__(self, data: List[List[str]], title: str):
        self.data = data
        self.title = title
        self.max_row = len(data)
        self.max_column = max(len(r) for r in data) if data else 0
        # 模拟 openpyxl 的 merged_cells 属性
        self.merged_cells = type("MergedCells", (), {"ranges": []})()

    def cell(self, row: int, column: int):
        """模拟 ws.cell(row=r, column=c)"""
        # openpyxl 是 1-indexed
        try:
            val = self.data[row - 1][column - 1]
        except IndexError:
            val = None
        return type("Cell", (), {"value": val, "row": row, "number_format": "General"})()

    def iter_rows(self, min_row=1, max_row=None, min_col=1, max_col=None, values_only=False):
        """模拟 ws.iter_rows"""
        if max_row is None: max_row = self.max_row
        if max_col is None: max_col = self.max_column
        
        for r in range(min_row, max_row + 1):
            row_data = []
            for c in range(min_col, max_col + 1):
                try:
                    val = self.data[r - 1][c - 1]
                    # 将空字符串转为 None，保持与 openpyxl 行为一致
                    val = val if val.strip() != "" else None
                except IndexError:
                    val = None
                if values_only:
                    row_data.append(val)
                else:
                    row_data.append(self.cell(r, c))
            yield tuple(row_data)