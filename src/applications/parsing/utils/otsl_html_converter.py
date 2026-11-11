"""
OTSL与HTML格式互转脚本

OTSL标签说明：
- <fcel> : 填充单元格（有内容的单元格）
- <ecel> : 空单元格
- <lcel> : 左延续单元格（水平合并的延续部分）
- <ucel> : 上延续单元格（垂直合并的延续部分）
- <xcel> : 交叉延续单元格（既是水平又是垂直合并的延续部分）
- <nl>   : 新行（换行符）
"""

import re
import html
import itertools
from typing import List, Dict, Any
from dataclasses import dataclass, field
from html.parser import HTMLParser


# ============================================================================
# OTSL标签定义
# ============================================================================
OTSL_NL = "<nl>"      # 新行
OTSL_FCEL = "<fcel>"  # 填充单元格
OTSL_ECEL = "<ecel>"  # 空单元格
OTSL_LCEL = "<lcel>"  # 左延续单元格（水平合并）
OTSL_UCEL = "<ucel>"  # 上延续单元格（垂直合并）
OTSL_XCEL = "<xcel>"  # 交叉延续单元格（2D合并）


# ============================================================================
# 数据结构定义
# ============================================================================
@dataclass
class TableCell:
    """表格单元格数据结构"""
    text: str = ""
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int = 0
    end_row_offset_idx: int = 1
    start_col_offset_idx: int = 0
    end_col_offset_idx: int = 1
    column_header: bool = False
    row_header: bool = False


@dataclass
class TableData:
    """表格数据结构"""
    table_cells: List[TableCell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0

    def get_grid(self) -> List[List[TableCell]]:
        """生成表格网格"""
        # 初始化空表格
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]

        # 填充实际单元格内容
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell

        return table_data


# ============================================================================
# OTSL -> HTML 转换功能
# ============================================================================
def otsl_extract_tokens_and_text(s: str):
    """从OTSL字符串中提取标签和文本"""
    pattern = r"(" + r"|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]) + r")"
    
    # 查找所有标签
    tokens = re.findall(pattern, s)
    
    # 按标签分割字符串获取文本部分
    text_parts = re.split(pattern, s)
    
    # 移除空白或纯空格的字符串
    text_parts = [part for part in text_parts if part.strip()]
    
    return tokens, text_parts


def otsl_parse_texts(texts, tokens):
    """解析OTSL文本和标签，生成表格单元格"""
    split_word = OTSL_NL
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    # 检查并补全矩阵
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)

        # 向标签中插入额外的<ecel>
        for row_idx, row in enumerate(split_row_tokens):
            while len(row) < max_cols:
                row.append(OTSL_ECEL)

        # 向文本中插入额外的<ecel>
        new_texts = []
        text_idx = 0

        for row_idx, row in enumerate(split_row_tokens):
            for col_idx, token in enumerate(row):
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if (text_idx < len(texts) and 
                        texts[text_idx] not in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]):
                        new_texts.append(texts[text_idx])
                        text_idx += 1

            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1

        texts = new_texts

    def count_right(tokens, c_idx, r_idx, which_tokens):
        """计算向右的跨度"""
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        """计算向下的跨度"""
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [OTSL_FCEL, OTSL_ECEL]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                right_offset = 2

            # 检查下一个元素是否为lcel/ucel/xcel，设置row_span和col_span
            next_right_cell = ""
            if i + right_offset < len(texts):
                next_right_cell = texts[i + right_offset]

            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [OTSL_LCEL, OTSL_XCEL]:
                # 水平跨单元格或2D跨单元格
                col_span += count_right(
                    split_row_tokens,
                    c_idx + 1,
                    r_idx,
                    [OTSL_LCEL, OTSL_XCEL],
                )
            
            if next_bottom_cell in [OTSL_UCEL, OTSL_XCEL]:
                # 垂直跨单元格或2D跨单元格
                row_span += count_down(
                    split_row_tokens,
                    c_idx,
                    r_idx + 1,
                    [OTSL_UCEL, OTSL_XCEL],
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        
        if text in [OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    
    return table_cells, split_row_tokens


def export_to_html(table_data: TableData) -> str:
    """将表格数据导出为HTML"""
    nrows = table_data.num_rows
    ncols = table_data.num_cols

    if len(table_data.table_cells) == 0:
        return ""

    body = ""
    grid = table_data.get_grid()
    
    for i in range(nrows):
        body += "<tr>"
        for j in range(ncols):
            cell: TableCell = grid[i][j]

            rowspan, rowstart = cell.row_span, cell.start_row_offset_idx
            colspan, colstart = cell.col_span, cell.start_col_offset_idx

            if rowstart != i:
                continue
            if colstart != j:
                continue

            content = html.escape(cell.text.strip())
            celltag = "th" if cell.column_header else "td"

            opening_tag = f"{celltag}"
            if rowspan > 1:
                opening_tag += f' rowspan="{rowspan}"'
            if colspan > 1:
                opening_tag += f' colspan="{colspan}"'

            body += f"<{opening_tag}>{content}</{celltag}>"
        body += "</tr>"

    body = f"<table>{body}</table>"
    return body


def convert_otsl_to_html(otsl_content: str) -> str:
    """
    将OTSL格式转换为HTML格式
    
    参数:
        otsl_content (str): OTSL格式的字符串
    
    返回:
        str: HTML格式的表格字符串
    """
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)

    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=(
            max(len(row) for row in split_row_tokens) if split_row_tokens else 0
        ),
        table_cells=table_cells,
    )

    return export_to_html(table_data)


# ============================================================================
# HTML -> OTSL 转换功能
# ============================================================================
class HTMLTableParser(HTMLParser):
    """HTML表格解析器"""
    
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell_text = ""
        self.current_cell_attrs = {}
        self.rows = []
        self.current_row = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "table":
            self.in_table = True
            self.rows = []
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ["td", "th"] and self.in_row:
            self.in_cell = True
            self.current_cell_text = ""
            self.current_cell_attrs = {
                "rowspan": int(attrs_dict.get("rowspan", 1)),
                "colspan": int(attrs_dict.get("colspan", 1)),
                "is_header": tag == "th"
            }
    
    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []
        elif tag in ["td", "th"] and self.in_cell:
            self.in_cell = False
            cell_data = {
                "text": self.current_cell_text.strip(),
                "rowspan": self.current_cell_attrs.get("rowspan", 1),
                "colspan": self.current_cell_attrs.get("colspan", 1),
                "is_header": self.current_cell_attrs.get("is_header", False)
            }
            self.current_row.append(cell_data)
    
    def handle_data(self, data):
        if self.in_cell:
            self.current_cell_text += data


def convert_html_to_otsl(html_content: str) -> str:
    """
    将HTML格式转换为OTSL格式
    
    参数:
        html_content (str): HTML格式的表格字符串
    
    返回:
        str: OTSL格式的字符串
    """
    # 解析HTML表格
    parser = HTMLTableParser()
    parser.feed(html_content)
    
    if not parser.rows:
        return ""
    
    # 构建单元格网格
    max_cols = 0
    for row in parser.rows:
        cols = sum(cell["colspan"] for cell in row)
        max_cols = max(max_cols, cols)
    
    # 创建网格矩阵
    num_rows = len(parser.rows)
    grid = [[None for _ in range(max_cols)] for _ in range(num_rows)]
    
    # 填充网格
    for row_idx, row in enumerate(parser.rows):
        col_idx = 0
        for cell in row:
            # 找到下一个空位置
            while col_idx < max_cols and grid[row_idx][col_idx] is not None:
                col_idx += 1
            
            if col_idx >= max_cols:
                break
            
            # 填充单元格
            rowspan = cell["rowspan"]
            colspan = cell["colspan"]
            
            for r in range(rowspan):
                for c in range(colspan):
                    if row_idx + r < num_rows and col_idx + c < max_cols:
                        # 标记单元格类型
                        if r == 0 and c == 0:
                            # 主单元格
                            grid[row_idx + r][col_idx + c] = {
                                "type": "main",
                                "text": cell["text"]
                            }
                        elif r > 0 and c > 0:
                            # 交叉延续单元格
                            grid[row_idx + r][col_idx + c] = {"type": "xcel"}
                        elif c > 0:
                            # 水平延续单元格
                            grid[row_idx + r][col_idx + c] = {"type": "lcel"}
                        elif r > 0:
                            # 垂直延续单元格
                            grid[row_idx + r][col_idx + c] = {"type": "ucel"}
            
            col_idx += colspan
    
    # 生成OTSL字符串
    otsl_lines = []
    for row in grid:
        row_tokens = []
        for cell in row:
            if cell is None:
                row_tokens.append(OTSL_ECEL)
            elif cell["type"] == "main":
                if cell["text"]:
                    row_tokens.append(f'{OTSL_FCEL} {cell["text"]}')
                else:
                    row_tokens.append(OTSL_ECEL)
            elif cell["type"] == "lcel":
                row_tokens.append(OTSL_LCEL)
            elif cell["type"] == "ucel":
                row_tokens.append(OTSL_UCEL)
            elif cell["type"] == "xcel":
                row_tokens.append(OTSL_XCEL)
        
        otsl_lines.append(" ".join(row_tokens))
    
    return f" {OTSL_NL} ".join(otsl_lines)


# ============================================================================
# 辅助功能
# ============================================================================
def format_html(html_str: str) -> str:
    """格式化HTML字符串，使其更易读"""
    html_str = html_str.replace("<table>", "<table>\n  ")
    html_str = html_str.replace("<tr>", "<tr>\n    ")
    html_str = html_str.replace("</tr>", "\n  </tr>")
    html_str = html_str.replace("</table>", "\n</table>")
    html_str = re.sub(r'<(td|th)', r'\n    <\1', html_str)
    return html_str


# ============================================================================
# 示例和测试
# ============================================================================
def main():
    """主函数，演示OTSL和HTML的互相转换"""
    
    print("=" * 80)
    print("OTSL与HTML格式互转演示")
    print("=" * 80)
    print()
    
    # 示例1：简单的2x3表格
    print("【示例1】简单的2x3表格")
    print("-" * 80)
    otsl_example1 = "<fcel> A1 <fcel> B1 <fcel> C1 <nl> <fcel> A2 <fcel> B2 <fcel> C2"
    print(f"OTSL格式：\n{otsl_example1}\n")
    
    html_result1 = convert_otsl_to_html(otsl_example1)
    print(f"转换为HTML：\n{format_html(html_result1)}\n")
    
    otsl_back1 = convert_html_to_otsl(html_result1)
    print(f"转换回OTSL：\n{otsl_back1}\n")
    print()
    
    # 示例2：包含合并单元格的表格
    print("【示例2】包含水平合并单元格的表格")
    print("-" * 80)
    otsl_example2 = "<fcel> 标题 <lcel> <lcel> <nl> <fcel> A1 <fcel> B1 <fcel> C1"
    print(f"OTSL格式：\n{otsl_example2}\n")
    
    html_result2 = convert_otsl_to_html(otsl_example2)
    print(f"转换为HTML：\n{format_html(html_result2)}\n")
    
    otsl_back2 = convert_html_to_otsl(html_result2)
    print(f"转换回OTSL：\n{otsl_back2}\n")
    print()
    
    # 示例3：包含垂直合并单元格的表格
    print("【示例3】包含垂直合并单元格的表格")
    print("-" * 80)
    otsl_example3 = "<fcel> A <fcel> B1 <fcel> C1 <nl> <ucel> <fcel> B2 <fcel> C2"
    print(f"OTSL格式：\n{otsl_example3}\n")
    
    html_result3 = convert_otsl_to_html(otsl_example3)
    print(f"转换为HTML：\n{format_html(html_result3)}\n")
    
    otsl_back3 = convert_html_to_otsl(html_result3)
    print(f"转换回OTSL：\n{otsl_back3}\n")
    print()
    
    # 示例4：从HTML转换为OTSL
    print("【示例4】从HTML转换为OTSL")
    print("-" * 80)
    html_example4 = '<table><tr><td>姓名</td><td>年龄</td><td>城市</td></tr><tr><td>张三</td><td>25</td><td>北京</td></tr><tr><td>李四</td><td>30</td><td>上海</td></tr></table>'
    print(f"HTML格式：\n{format_html(html_example4)}\n")
    
    otsl_result4 = convert_html_to_otsl(html_example4)
    print(f"转换为OTSL：\n{otsl_result4}\n")
    
    html_back4 = convert_otsl_to_html(otsl_result4)
    print(f"转换回HTML：\n{format_html(html_back4)}\n")
    print()
    
    # 示例5：复杂表格（包含多种合并）
    print("【示例5】复杂表格（包含colspan和rowspan）")
    print("-" * 80)
    html_example5 = '<table><tr><td colspan="2">合并列</td><td rowspan="2">合并行</td></tr><tr><td>A</td><td>B</td></tr></table>'
    print(f"HTML格式：\n{format_html(html_example5)}\n")
    
    otsl_result5 = convert_html_to_otsl(html_example5)
    print(f"转换为OTSL：\n{otsl_result5}\n")
    
    html_back5 = convert_otsl_to_html(otsl_result5)
    print(f"转换回HTML：\n{format_html(html_back5)}\n")
    
    print("=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

