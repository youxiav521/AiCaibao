from typing import List, Dict, Any
from pathlib import Path
import re
import logging
from src.applications.parsing.utils.ragflow_parsers import RAGFlowMarkdownParser, MarkdownElementExtractor, get_text

logger = logging.getLogger(__name__)

class MarkdownParsingService:
    """Markdown 文件解析服务"""

    def __init__(self, chunk_token_num=128):
        self.chunk_token_num = chunk_token_num

    @staticmethod
    def _build_pseudo_bbox(start_row: int, end_row: int) -> Dict[str, float]:
        top = float(max(0, start_row))
        bottom = float(max(top, end_row))
        return {
            "l": 0.0,
            "t": top,
            "r": 1000.0,
            "b": bottom,
            "coord_origin": "TOPLEFT",
        }

    @staticmethod
    def _is_standard_2d(rows: List[List[Dict[str, Any]]]) -> bool:
        for row in rows or []:
            if not isinstance(row, list):
                return False
            for cell in row:
                if not isinstance(cell, dict):
                    return False
                try:
                    if int(cell.get("rowspan", 1)) != 1 or int(cell.get("colspan", 1)) != 1:
                        return False
                except Exception:
                    return False
        return True

    def parse_to_items(self, file_path: str) -> List[Dict[str, Any]]:
        """
        将 markdown 文件解析为元素列表（文本块、表格等）。
        
        Args:
            file_path: markdown 文件的绝对路径。
            
        Returns:
            表示解析后元素的字典列表。
            每个元素包含 'type'（'text', 'table', 'header' 等）、'md'、'value' 等字段。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        # 1. 读取内容
        content = get_text(file_path)

        # 2. 提取表格
        parser = RAGFlowMarkdownParser(chunk_token_num=self.chunk_token_num)
        text_only, tables = parser.extract_tables_and_remainder(content)

        items: List[Dict[str, Any]] = []

        # 3. 处理文本元素
        extractor = MarkdownElementExtractor(text_only)
        elements = extractor.extract_elements(include_meta=True)

        for el in elements:
            raw_content = el.get("content", "")
            if not raw_content.strip():
                continue
            
            # 过滤表格占位符
            if raw_content.strip() == "# RAGFLOW_TABLE_PLACEHOLDER":
                continue
                
            el_type = el.get("type", "text")
            
            # 类型映射：Markdown 解析结果直接对齐 pseudo-MinerU，避免输出额外的 code 类型
            item_type = "text"
            if el_type == "list_block":
                item_type = "list"

            start_row = el.get("start_line", 0) + 1
            end_row = el.get("end_line", 0) + 1
            item = {
                "type": item_type,
                "page": 1,
                "bBox": self._build_pseudo_bbox(start_row, end_row),
                "value": raw_content,
                "md": raw_content,
                "startRow": start_row,
                "endRow": end_row,
            }
            if el_type == "header":
                match = re.match(r"^(#{1,6})\s+", raw_content.lstrip())
                if match:
                    item["headingLevel"] = len(match.group(1))
            items.append(item)

        # 4. 处理表格
        for tbl_data in tables:
            tbl_content = tbl_data["content"]
            rows, html_content = self._parse_table_to_rows(tbl_content)
            start_row = tbl_data["start_line"] + 1
            end_row = tbl_data["end_line"] + 1
            has_header = bool(rows)
            item = {
                "type": "table",
                "page": 1,
                "bBox": self._build_pseudo_bbox(start_row, end_row),
                "tableSubtype": "structured",
                "value": tbl_content,
                "md": tbl_content,
                "html": html_content,
                "rows": rows,
                "startRow": start_row,
                "endRow": end_row,
                "standard2D": self._is_standard_2d(rows),
                "hasHeader": has_header,
            }
            if has_header:
                item["headerCells"] = [rows[0]]
            items.append(item)

        # 按 startRow 排序以恢复文档顺序
        items.sort(key=lambda x: x.get("startRow", 0))

        return items

    def _parse_table_to_rows(self, table_content: str) -> tuple[List[List[Dict[str, Any]]], str]:
        """
        将 Markdown/HTML 表格解析为 rows 结构。
        返回 (rows, html_content)。
        """
        import markdown
        from src.applications.parsing.utils.convert_minerU_script import parse_html_table_rows

        html_content = ""
        # 检查内容是否为 HTML
        if "<table>" in table_content.lower():
            html_content = table_content
        else:
            # 将 Markdown 表格转换为 HTML
            try:
                html_content = markdown.markdown(table_content, extensions=['tables'])
            except Exception as e:
                logger.warning(f"Markdown 表格转 HTML 失败: {e}")
                html_content = ""

        rows = []
        if html_content:
            try:
                rows = parse_html_table_rows(html_content)
            except Exception as e:
                logger.warning(f"解析 HTML 表格行失败: {e}")
                
        return rows, html_content
