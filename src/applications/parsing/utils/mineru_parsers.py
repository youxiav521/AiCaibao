import re
import os
import unicodedata
from fast_langdetect import detect_language
from copy import deepcopy
from bs4 import BeautifulSoup

class BlockType:
    IMAGE = 'image'
    TABLE = 'table'
    IMAGE_BODY = 'image_body'
    TABLE_BODY = 'table_body'
    IMAGE_CAPTION = 'image_caption'
    TABLE_CAPTION = 'table_caption'
    IMAGE_FOOTNOTE = 'image_footnote'
    TABLE_FOOTNOTE = 'table_footnote'
    TEXT = 'text'
    TITLE = 'title'
    INTERLINE_EQUATION = 'interline_equation'
    LIST = 'list'
    INDEX = 'index'
    DISCARDED = 'discarded'

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    ALGORITHM = "algorithm"
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"

class ContentType:
    IMAGE = 'image'
    TABLE = 'table'
    TEXT = 'text'
    INTERLINE_EQUATION = 'interline_equation'
    INLINE_EQUATION = 'inline_equation'
    EQUATION = 'equation'
    CODE = 'code'
class MakeMode:
    MM_MD = 'mm_markdown'
    NLP_MD = 'nlp_markdown'
    CONTENT_LIST = 'content_list'
    CONTENT_LIST_V2 = 'content_list_v2'
class SplitFlag:
    CROSS_PAGE = 'cross_page'
    LINES_DELETED = 'lines_deleted'    
    
default_delimiters = {
    'display': {'left': '$$', 'right': '$$'},
    'inline': {'left': '$', 'right': '$'}
}

delimiters =  default_delimiters

display_left_delimiter = delimiters['display']['left']
display_right_delimiter = delimiters['display']['right']
inline_left_delimiter = delimiters['inline']['left']
inline_right_delimiter = delimiters['inline']['right']

def mk_blocks_to_markdown(para_blocks, make_mode, formula_enable, table_enable, img_buket_path=''):
    page_markdown = []
    for para_block in para_blocks:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.TEXT, BlockType.INTERLINE_EQUATION, BlockType.PHONETIC, BlockType.REF_TEXT]:
            para_text = merge_para_with_text(para_block, formula_enable=formula_enable, img_buket_path=img_buket_path)
        elif para_type == BlockType.LIST:
            for block in para_block['blocks']:
                item_text = merge_para_with_text(block, formula_enable=formula_enable, img_buket_path=img_buket_path)
                para_text += f"{item_text}  \n"
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
        elif para_type == BlockType.IMAGE:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                # 检测是否存在图片脚注
                has_image_footnote = any(block['type'] == BlockType.IMAGE_FOOTNOTE for block in para_block['blocks'])
                # 如果存在图片脚注，则将图片脚注拼接到图片正文后面
                if has_image_footnote:
                    for block in para_block['blocks']:  # 1st.拼image_caption
                        if block['type'] == BlockType.IMAGE_CAPTION:
                            para_text += merge_para_with_text(block) + '  \n'
                    for block in para_block['blocks']:  # 2nd.拼image_body
                        if block['type'] == BlockType.IMAGE_BODY:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if span['type'] == ContentType.IMAGE:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                    for block in para_block['blocks']:  # 3rd.拼image_footnote
                        if block['type'] == BlockType.IMAGE_FOOTNOTE:
                            para_text += '  \n' + merge_para_with_text(block)
                else:
                    for block in para_block['blocks']:  # 1st.拼image_body
                        if block['type'] == BlockType.IMAGE_BODY:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if span['type'] == ContentType.IMAGE:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                    for block in para_block['blocks']:  # 2nd.拼image_caption
                        if block['type'] == BlockType.IMAGE_CAPTION:
                            para_text += '  \n' + merge_para_with_text(block)

        elif para_type == BlockType.TABLE:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                for block in para_block['blocks']:  # 1st.拼table_caption
                    if block['type'] == BlockType.TABLE_CAPTION:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:  # 2nd.拼table_body
                    if block['type'] == BlockType.TABLE_BODY:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.TABLE:
                                    # if processed by table model
                                    if table_enable:
                                        if span.get('html', ''):
                                            para_text += f"\n{span['html']}\n"
                                        elif span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                                    else:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                for block in para_block['blocks']:  # 3rd.拼table_footnote
                    if block['type'] == BlockType.TABLE_FOOTNOTE:
                        para_text += '\n' + merge_para_with_text(block) + '  '
        elif para_type == BlockType.CODE:
            sub_type = para_block["sub_type"]
            for block in para_block['blocks']:  # 1st.拼code_caption
                if block['type'] == BlockType.CODE_CAPTION:
                    para_text += merge_para_with_text(block) + '  \n'
            for block in para_block['blocks']:  # 2nd.拼code_body
                if block['type'] == BlockType.CODE_BODY:
                    if sub_type == BlockType.CODE:
                        guess_lang = para_block["guess_lang"]
                        para_text += f"```{guess_lang}\n{merge_para_with_text(block)}\n```"
                    elif sub_type == BlockType.ALGORITHM:
                        para_text += merge_para_with_text(block)

        if para_text.strip() == '':
            continue
        else:
            # page_markdown.append(para_text.strip() + '  ')
            page_markdown.append(para_text.strip())

    return page_markdown


def merge_para_with_text(para_block, formula_enable=True, img_buket_path=''):
    block_text = ''
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] in [ContentType.TEXT]:
                span['content'] = full_to_half_exclude_marks(span['content'])
                block_text += span['content']
    block_lang = detect_lang(block_text)

    para_text = ''
    for i, line in enumerate(para_block['lines']):
        for j, span in enumerate(line['spans']):
            span_type = span['type']
            content = ''
            if span_type == ContentType.TEXT:
                content = span['content']
            elif span_type == ContentType.INLINE_EQUATION:
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
            elif span_type == ContentType.INTERLINE_EQUATION:
                if formula_enable:
                    content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
                else:
                    if span.get('image_path', ''):
                        content = f"![]({img_buket_path}/{span['image_path']})"

            content = content.strip()
            if content:

                if span_type == ContentType.INTERLINE_EQUATION:
                    para_text += content
                    continue

                # 定义CJK语言集合(中日韩)
                cjk_langs = {'zh', 'ja', 'ko'}
                # logger.info(f'block_lang: {block_lang}, content: {content}')

                # 判断是否为行末span
                is_last_span = j == len(line['spans']) - 1

                if block_lang in cjk_langs:  # 中文/日语/韩文语境下，换行不需要空格分隔,但是如果是行内公式结尾，还是要加空格
                    if is_last_span and span_type != ContentType.INLINE_EQUATION:
                        para_text += content
                    else:
                        para_text += f'{content} '
                else:
                    # 西方文本语境下 每行的最后一个span判断是否要去除连字符
                    if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
                        # 如果span是line的最后一个且末尾带有-连字符，那么末尾不应该加空格,同时应该把-删除
                        if (
                                is_last_span
                                and span_type == ContentType.TEXT
                                and is_hyphen_at_line_end(content)
                        ):
                            # 如果下一行的第一个span是小写字母开头，删除连字符
                            if (
                                    i+1 < len(para_block['lines'])
                                    and para_block['lines'][i + 1].get('spans')
                                    and para_block['lines'][i + 1]['spans'][0].get('type') == ContentType.TEXT
                                    and para_block['lines'][i + 1]['spans'][0].get('content', '')
                                    and para_block['lines'][i + 1]['spans'][0]['content'][0].islower()
                            ):
                                para_text += content[:-1]
                            else:  # 如果没有下一行，或者下一行的第一个span不是小写字母开头，则保留连字符但不加空格
                                para_text += content
                        else:  # 西方文本语境下 content间需要空格分隔
                            para_text += f'{content} '
    return para_text


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level > 5:
        title_level = 5
    elif title_level < 1:
        title_level = 0
    return title_level

def full_to_half_exclude_marks(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters and numbers (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)


def is_hyphen_at_line_end(line):
    """Check if a line ends with one or more letters followed by a hyphen.

    Args:
    line (str): The line of text to check.

    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # Use regex to check if the line ends with one or more letters followed by a hyphen
    return bool(re.search(r'[A-Za-z]+-\s*$', line))

def detect_lang(text: str) -> str:

    if len(text) == 0:
        return ""

    text = text.replace("\n", "")
    text = remove_invalid_surrogates(text)

    # print(text)
    try:
        lang_upper = detect_language(text)
    except:
        html_no_ctrl_chars = ''.join([l for l in text if unicodedata.category(l)[0] not in ['C', ]])
        lang_upper = detect_language(html_no_ctrl_chars)

    try:
        lang = lang_upper.lower()
    except:
        lang = ""
    return lang


def remove_invalid_surrogates(text):
    # 移除无效的 UTF-16 代理对
    return ''.join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))
def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters, numbers and punctuation (FF01-FF5E)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)





CONTINUATION_MARKERS = [
    "(续)",
    "(续表)",
    "(续上表)",
    "(continued)",
    "(cont.)",
    "(cont'd)",
    "续",
    "续表",
    "续上表",
    "continued",
    "cont.",
    "cont'd",
]


def calculate_table_total_columns(soup):
    """计算表格的总列数，通过分析整个表格结构来处理rowspan和colspan

    Args:
        soup: BeautifulSoup解析的表格

    Returns:
        int: 表格的总列数
    """
    rows = soup.find_all("tr")
    if not rows:
        return 0

    # 创建一个矩阵来跟踪每个位置的占用情况
    max_cols = 0
    occupied = {}  # {row_idx: {col_idx: True}}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])

        if row_idx not in occupied:
            occupied[row_idx] = {}

        for cell in cells:
            # 找到下一个未被占用的列位置
            while col_idx in occupied[row_idx]:
                col_idx += 1

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # 标记被这个单元格占用的所有位置
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True

            col_idx += colspan
            max_cols = max(max_cols, col_idx)

    return max_cols


def calculate_row_columns(row):
    """
    计算表格行的实际列数，考虑colspan属性

    Args:
        row: BeautifulSoup的tr元素对象

    Returns:
        int: 行的实际列数
    """
    cells = row.find_all(["td", "th"])
    column_count = 0

    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        column_count += colspan

    return column_count


def calculate_visual_columns(row):
    """
    计算表格行的视觉列数（实际td/th单元格数量，不考虑colspan）

    Args:
        row: BeautifulSoup的tr元素对象

    Returns:
        int: 行的视觉列数（实际单元格数）
    """
    cells = row.find_all(["td", "th"])
    return len(cells)


def detect_table_headers(soup1, soup2, max_header_rows=5):
    """
    检测并比较两个表格的表头

    Args:
        soup1: 第一个表格的BeautifulSoup对象
        soup2: 第二个表格的BeautifulSoup对象
        max_header_rows: 最大可能的表头行数

    Returns:
        tuple: (表头行数, 表头是否一致, 表头文本列表)
    """
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for i in range(min_rows):
        # 提取当前行的所有单元格
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])

        # 检查两行的结构和内容是否一致
        structure_match = True

        # 首先检查单元格数量
        if len(cells1) != len(cells2):
            structure_match = False
        else:
            # 然后检查单元格的属性和内容
            for cell1, cell2 in zip(cells1, cells2):
                colspan1 = int(cell1.get("colspan", 1))
                rowspan1 = int(cell1.get("rowspan", 1))
                colspan2 = int(cell2.get("colspan", 1))
                rowspan2 = int(cell2.get("rowspan", 1))

                # 去除所有空白字符（包括空格、换行、制表符等）
                text1 = ''.join(full_to_half(cell1.get_text()).split())
                text2 = ''.join(full_to_half(cell2.get_text()).split())

                if colspan1 != colspan2 or rowspan1 != rowspan2 or text1 != text2:
                    structure_match = False
                    break

        if structure_match:
            header_rows += 1
            row_texts = [full_to_half(cell.get_text().strip()) for cell in cells1]
            header_texts.append(row_texts)  # 添加表头文本
        else:
            headers_match = header_rows > 0  # 只有当至少匹配了一行时，才认为表头匹配
            break

    # 如果没有找到匹配的表头行，则返回失败
    if header_rows == 0:
        headers_match = False

    return header_rows, headers_match, header_texts


def can_merge_tables(current_table_block, previous_table_block, has_continuation_signal=False):
    """判断两个表格是否可以合并"""
    # 检查表格是否有caption和footnote
    # 如果有TABLE_CAPTION类型的块,检查是否至少有一个以"(续)"结尾
    caption_blocks = [block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_CAPTION]
    has_caption_continuation = any(
        _has_continuation_marker_text(merge_para_with_text(block))
        for block in caption_blocks
    )
    if caption_blocks and not has_caption_continuation and not has_continuation_signal:
        return False, None, None, None, None

    if any(block["type"] == BlockType.TABLE_FOOTNOTE for block in previous_table_block["blocks"]):
        return False, None, None, None, None

    # 获取两个表格的HTML内容
    current_html = ""
    previous_html = ""

    for block in current_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block.get("lines") and block["lines"][0].get("spans")):
            current_html = block["lines"][0]["spans"][0].get("html", "")

    for block in previous_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block.get("lines") and block["lines"][0].get("spans")):
            previous_html = block["lines"][0]["spans"][0].get("html", "")

    if not current_html or not previous_html:
        return False, None, None, None, None

    # 检查表格宽度差异
    x0_t1, y0_t1, x1_t1, y1_t1 = current_table_block["bbox"]
    x0_t2, y0_t2, x1_t2, y1_t2 = previous_table_block["bbox"]
    table1_width = x1_t1 - x0_t1
    table2_width = x1_t2 - x0_t2

    if abs(table1_width - table2_width) / min(table1_width, table2_width) >= 0.1:
        return False, None, None, None, None

    # 解析HTML并检查表格结构
    soup1 = BeautifulSoup(previous_html, "html.parser")
    soup2 = BeautifulSoup(current_html, "html.parser")

    # 检查整体列数匹配
    table_cols1 = calculate_table_total_columns(soup1)
    table_cols2 = calculate_table_total_columns(soup2)
    # logger.debug(f"Table columns - Previous: {table_cols1}, Current: {table_cols2}")
    tables_match = table_cols1 == table_cols2

    # 检查首末行列数匹配
    rows_match = check_rows_match(soup1, soup2)

    return (tables_match or rows_match), soup1, soup2, current_html, previous_html


def check_rows_match(soup1, soup2):
    """检查表格行是否匹配"""
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    if not (rows1 and rows2):
        return False

    # 获取第一个表的最后一行数据行
    last_row = None
    for row in reversed(rows1):
        if row.find_all(["td", "th"]):
            last_row = row
            break

    # 检测表头行数，以便获取第二个表的首个数据行
    header_count, _, _ = detect_table_headers(soup1, soup2)

    # 获取第二个表的首个数据行
    first_data_row = None
    if len(rows2) > header_count:
        first_data_row = rows2[header_count]  # 第一个非表头行

    if not (last_row and first_data_row):
        return False

    # 计算实际列数（考虑colspan）和视觉列数
    last_row_cols = calculate_row_columns(last_row)
    first_row_cols = calculate_row_columns(first_data_row)
    last_row_visual_cols = calculate_visual_columns(last_row)
    first_row_visual_cols = calculate_visual_columns(first_data_row)

    # logger.debug(f"行列数 - 前表最后一行: {last_row_cols}(视觉列数:{last_row_visual_cols}), 当前表首行: {first_row_cols}(视觉列数:{first_row_visual_cols})")

    # 同时考虑实际列数匹配和视觉列数匹配
    return last_row_cols == first_row_cols or last_row_visual_cols == first_row_visual_cols


def check_row_columns_match(row1, row2):
    # 逐个cell检测colspan属性是否一致
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        colspan1 = int(cell1.get("colspan", 1))
        colspan2 = int(cell2.get("colspan", 1))
        if colspan1 != colspan2:
            return False
    return True


def adjust_table_rows_colspan(rows, start_idx, end_idx,
                              reference_structure, reference_visual_cols,
                              target_cols, current_cols, reference_row):
    """调整表格行的colspan属性以匹配目标列数

    Args:
        rows: 表格行列表
        start_idx: 起始行索引
        end_idx: 结束行索引（不包含）
        reference_structure: 参考行的colspan结构列表
        reference_visual_cols: 参考行的视觉列数
        target_cols: 目标总列数
        current_cols: 当前总列数
        reference_row: 参考行对象
    """
    reference_row_copy = deepcopy(reference_row)

    for i in range(start_idx, end_idx):
        row = rows[i]
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        current_row_cols = calculate_row_columns(row)
        if current_row_cols >= target_cols:
            continue

        # 检查是否与参考行结构匹配
        if calculate_visual_columns(row) == reference_visual_cols and check_row_columns_match(row, reference_row_copy):
            # 尝试应用参考结构
            if len(cells) <= len(reference_structure):
                for j, cell in enumerate(cells):
                    if j < len(reference_structure) and reference_structure[j] > 1:
                        cell["colspan"] = str(reference_structure[j])
        else:
            # 扩展最后一个单元格以填补列数差异
            last_cell = cells[-1]
            current_last_span = int(last_cell.get("colspan", 1))
            last_cell["colspan"] = str(current_last_span + (target_cols - current_cols))


def perform_table_merge(soup1, soup2, previous_table_block, wait_merge_table_footnotes):
    """执行表格合并操作"""
    # 检测表头有几行，并确认表头内容是否一致
    header_count, headers_match, header_texts = detect_table_headers(soup1, soup2)
    # logger.debug(f"检测到表头行数: {header_count}, 表头匹配: {headers_match}")
    # logger.debug(f"表头内容: {header_texts}")

    # 找到第一个表格的tbody，如果没有则查找table元素
    tbody1 = soup1.find("tbody") or soup1.find("table")

    # 获取表1和表2的所有行
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")


    if rows1 and rows2 and header_count < len(rows2):
        # 获取表1最后一行和表2第一个非表头行
        last_row1 = rows1[-1]
        first_data_row2 = rows2[header_count]

        # 计算表格总列数
        table_cols1 = calculate_table_total_columns(soup1)
        table_cols2 = calculate_table_total_columns(soup2)
        if table_cols1 >= table_cols2:
            reference_structure = [int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(last_row1)
            # 以表1的最后一行为参考，调整表2的行
            adjust_table_rows_colspan(
                rows2, header_count, len(rows2),
                reference_structure, reference_visual_cols,
                table_cols1, table_cols2, first_data_row2
            )

        else:  # table_cols2 > table_cols1
            reference_structure = [int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(first_data_row2)
            # 以表2的第一个数据行为参考，调整表1的行
            adjust_table_rows_colspan(
                rows1, 0, len(rows1),
                reference_structure, reference_visual_cols,
                table_cols2, table_cols1, last_row1
            )

    # 将第二个表格的行添加到第一个表格中
    if tbody1:
        tbody2 = soup2.find("tbody") or soup2.find("table")
        if tbody2:
            # 将第二个表格的行添加到第一个表格中（跳过表头行）
            for row in rows2[header_count:]:
                row.extract()
                tbody1.append(row)

    # 添加待合并表格的footnote到前一个表格中
    for table_footnote in wait_merge_table_footnotes:
        temp_table_footnote = table_footnote.copy()
        temp_table_footnote[SplitFlag.CROSS_PAGE] = True
        previous_table_block["blocks"].append(temp_table_footnote)

    return str(soup1)


def _has_continuation_marker_text(text):
    normalized = full_to_half(str(text or "")).strip().lower()
    if not normalized:
        return False
    return any(marker.lower() in normalized for marker in CONTINUATION_MARKERS)


def _is_table_content_english(table_block):
    """检测表格块的内容是否为英文财报。

    通过解析表格 HTML 中的文本内容，使用 fast_langdetect 判定语言。
    """
    if not isinstance(table_block, dict):
        return False
    table_html = ""
    for block in table_block.get("blocks", []):
        if (isinstance(block, dict) and block.get("type") == BlockType.TABLE_BODY
                and block.get("lines") and block["lines"][0].get("spans")):
            table_html = block["lines"][0]["spans"][0].get("html", "")
            break
    if not table_html:
        return False
    soup = BeautifulSoup(table_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    if len(text) < 20:
        return False
    lang = detect_lang(text)
    return lang == "en"


def _blocks_before_table_identical(page_info, table_idx, prev_page_info, prev_table_idx):
    """比较两页中表格之前各 block 的文本内容是否逐对完全相同。

    用于英文财报中跨页表格的合并判定：如果上一页和当前页在表格之前的
    header/text/title 等 block 内容完全相同（如重复的公司名和报表标题），
    则两个页面的表格视为可合并的跨页续表。
    """
    para_blocks = page_info.get("para_blocks") if isinstance(page_info, dict) else None
    prev_para_blocks = prev_page_info.get("para_blocks") if isinstance(prev_page_info, dict) else None
    if not isinstance(para_blocks, list) or not isinstance(prev_para_blocks, list):
        return False
    if table_idx != prev_table_idx:
        return False
    if table_idx is None or table_idx <= 0:
        return False

    blocks_before = para_blocks[:table_idx]
    prev_blocks_before = prev_para_blocks[:prev_table_idx]
    if len(blocks_before) != len(prev_blocks_before):
        return False

    for b1, b2 in zip(blocks_before, prev_blocks_before):
        if not isinstance(b1, dict) or not isinstance(b2, dict):
            return False
        # 只比较文本类 block，跳过 image/image_caption 等无 lines 字段的类型
        if b1.get("type") not in {BlockType.TITLE, BlockType.TEXT, BlockType.TABLE_CAPTION}:
            continue
        if b2.get("type") not in {BlockType.TITLE, BlockType.TEXT, BlockType.TABLE_CAPTION}:
            continue
        text1 = full_to_half(merge_para_with_text(b1).strip())
        text2 = full_to_half(merge_para_with_text(b2).strip())
        if text1 != text2 and text2 not in text1 and text1 not in text2:
            return False
    return True


def _has_continuation_signal_before_table(page_info, table_idx):
    para_blocks = page_info.get("para_blocks") if isinstance(page_info, dict) else None
    if not isinstance(para_blocks, list) or table_idx is None:
        return False
    for block in para_blocks[:table_idx]:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in {BlockType.TITLE, BlockType.TEXT, BlockType.TABLE_CAPTION}:
            continue
        if _has_continuation_marker_text(merge_para_with_text(block)):
            return True
    return False


# 报表基础类型识别（中英文），与 table_classification_utils 中
# extract_statement_table_type / base_statement_type 的判定保持一致，
# 用于跨页合并时阻止不同类报表被错合（例如资产负债表 vs 利润表）。
_STATEMENT_BASE_KEYWORDS_ZH = (
    ("所有者权益变动表", "所有者权益变动表"),
    ("股东权益变动表", "所有者权益变动表"),
    ("权益变动表", "所有者权益变动表"),
    ("资产负债表", "资产负债表"),
    ("利润分配表", "利润表"),
    ("利润表", "利润表"),
    ("现金流量表", "现金流量表"),
)

_STATEMENT_BASE_KEYWORDS_EN = (
    ("statement of changes in equity", "所有者权益变动表"),
    ("statements of changes in equity", "所有者权益变动表"),
    ("changes in equity", "所有者权益变动表"),
    ("statement of financial position", "资产负债表"),
    ("statements of financial position", "资产负债表"),
    ("balance sheet", "资产负债表"),
    # 损益表家族：包含 P&L、单表式综合收益表、P&L+OCI 合并表，统一归"利润表"。
    ("statement of profit or loss", "利润表"),
    ("statements of profit or loss", "利润表"),
    ("statement of operations", "利润表"),
    ("income statement", "利润表"),
    ("profit and loss", "利润表"),
    ("profit or loss", "利润表"),
    ("statement of comprehensive income", "利润表"),
    ("statements of comprehensive income", "利润表"),
    ("statement of cash flows", "现金流量表"),
    ("statements of cash flows", "现金流量表"),
    ("cash flow statement", "现金流量表"),
    ("cashflow statement", "现金流量表"),
)


def _extract_statement_base_type(text):
    """从一段文本中识别报表基础类型；识别不到返回 None。

    与 ``is_derivative_statement_title`` 同源：命中关键词后若尾部仍含
    ≥2 个连续中文字符（如"资产负债表项目变动说明"），视为衍生标题，跳过。
    """
    if not isinstance(text, str):
        return None
    normalized = full_to_half(text).strip()
    if not normalized:
        return None
    compact = re.sub(r'\s+', '', normalized)
    for kw, base in _STATEMENT_BASE_KEYWORDS_ZH:
        idx = compact.find(kw)
        if idx < 0:
            continue
        tail = compact[idx + len(kw):]
        cleaned = re.sub(r'[（(][^）)]*[）)]', '', tail)
        cleaned = re.sub(r'[\s\d\-—:：、，。；！？·．.]', '', cleaned)
        if re.search(r'[\u4e00-\u9fff]{2,}', cleaned):
            continue
        return base
    lower = normalized.lower()
    for kw, base in _STATEMENT_BASE_KEYWORDS_EN:
        if kw in lower:
            return base
    return None


def _find_nearest_statement_base_before_table(page_info, table_idx):
    """在指定页面 table_idx 之前倒序扫描，返回最近一次匹配到的报表基础类型。"""
    if not isinstance(page_info, dict):
        return None
    para_blocks = page_info.get("para_blocks")
    if not isinstance(para_blocks, list) or table_idx is None:
        return None
    upper = min(table_idx, len(para_blocks))
    for idx in range(upper - 1, -1, -1):
        block = para_blocks[idx]
        if not isinstance(block, dict):
            continue
        if block.get("type") not in {BlockType.TITLE, BlockType.TEXT, BlockType.TABLE_CAPTION}:
            continue
        base = _extract_statement_base_type(merge_para_with_text(block))
        if base:
            return base
    return None


# 基于表格自身内容识别报表基础类型时使用的关键词集合。
# 关键词来源对齐 src.applications.parsing.utils.table_classification.en_rules 中
# 的 EN_BALANCE_TERMS / EN_PROFIT_TERMS / EN_CASH_TERMS，但用于跨页合并判定时
# 仅关心强信号，避免把损益表里的 "revenue" 误判到资产负债表家族。
_TABLE_CONTENT_BALANCE_TERMS_EN = (
    "total assets", "total liabilities", "total equity",
    "total equity and liabilities", "total liabilities and equity",
    "current assets", "non-current assets",
    "current liabilities", "non-current liabilities",
    "shareholders' equity", "stockholders' equity",
)
_TABLE_CONTENT_PROFIT_TERMS_EN = (
    "gross profit", "operating profit", "operating income",
    "profit before tax", "profit before income tax",
    "profit for the year", "profit for the period",
    "net profit", "net income", "net loss", "loss for the year",
    "earnings per share", "income tax expense",
    "total comprehensive income", "other comprehensive income",
    "cost of revenue", "cost of sales", "cost of goods sold",
    "statement of income", "nature of expense", "revenue",
)
_TABLE_CONTENT_CASH_TERMS_EN = (
    "cash flows from operating", "cash flows from investing",
    "cash flows from financing", "cash flow from operating",
    "cash flow from investing", "cash flow from financing",
    "net cash from operating", "net cash used in",
    "cash and cash equivalents at",
    "net increase in cash", "net change in cash",
    "operating activities", "investing activities", "financing activities",
)
_TABLE_CONTENT_EQUITY_TERMS_EN = (
    "balance at", "translation reserve", "treasury shares",
    "retained earnings", "share capital",
    "attributable to owners", "attributable to equity holders",
)

_TABLE_CONTENT_BALANCE_TERMS_ZH = (
    "资产总计", "负债总计", "负债合计", "所有者权益合计", "股东权益合计",
    "负债和所有者权益总计", "负债和股东权益总计",
    "流动资产合计", "非流动资产合计", "流动负债合计", "非流动负债合计",
)
_TABLE_CONTENT_PROFIT_TERMS_ZH = (
    "营业收入", "营业总收入", "营业成本", "营业总成本",
    "毛利", "营业利润", "利润总额", "净利润",
    "归属于母公司所有者的净利润", "每股收益", "所得税费用",
)
_TABLE_CONTENT_CASH_TERMS_ZH = (
    "经营活动产生的现金流量", "投资活动产生的现金流量", "筹资活动产生的现金流量",
    "现金及现金等价物净增加额", "期末现金及现金等价物余额",
    "经营活动现金流入", "投资活动现金流入", "筹资活动现金流入",
)
_TABLE_CONTENT_EQUITY_TERMS_ZH = (
    "实收资本", "资本公积", "盈余公积", "未分配利润",
    "本年增减变动", "上年年末余额", "本年年末余额",
)


def _extract_html_text(table_block):
    """提取表格 block 的 HTML 文本（无控制字符），用于内容关键词匹配。"""
    if not isinstance(table_block, dict):
        return ""
    for block in table_block.get("blocks", []):
        if (
            isinstance(block, dict)
            and block.get("type") == BlockType.TABLE_BODY
            and block.get("lines")
            and block["lines"][0].get("spans")
        ):
            html = block["lines"][0]["spans"][0].get("html", "")
            if not html:
                return ""
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    return ""


def _classify_table_content_base_type(table_block):
    """根据表格文本中出现的科目关键词识别报表基础类型；无法判定返回 None。

    采用"得分制 + 强势胜出"策略：四类关键词分别统计命中次数，取分数最高者；
    分数全部为 0 或最高分与次高分相等时返回 None，以避免误判。
    """
    text = _extract_html_text(table_block)
    if not text:
        return None
    full_lower = full_to_half(text).lower()
    compact_zh = re.sub(r"\s+", "", text)
    scores = {
        "资产负债表": (
            sum(1 for kw in _TABLE_CONTENT_BALANCE_TERMS_EN if kw in full_lower)
            + sum(1 for kw in _TABLE_CONTENT_BALANCE_TERMS_ZH if kw in compact_zh)
        ),
        "利润表": (
            sum(1 for kw in _TABLE_CONTENT_PROFIT_TERMS_EN if kw in full_lower)
            + sum(1 for kw in _TABLE_CONTENT_PROFIT_TERMS_ZH if kw in compact_zh)
        ),
        "现金流量表": (
            sum(1 for kw in _TABLE_CONTENT_CASH_TERMS_EN if kw in full_lower)
            + sum(1 for kw in _TABLE_CONTENT_CASH_TERMS_ZH if kw in compact_zh)
        ),
        "所有者权益变动表": (
            sum(1 for kw in _TABLE_CONTENT_EQUITY_TERMS_EN if kw in full_lower)
            + sum(1 for kw in _TABLE_CONTENT_EQUITY_TERMS_ZH if kw in compact_zh)
        ),
    }
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ordered[0]
    if top_score >= 2:
        second_score = ordered[1][1] if len(ordered) > 1 else 0
        if top_score - second_score >= 2:
            return top_label

    # 科目词判定失败时，从表格前三行做表名识别（仅用于排他）
    first_rows_text = _extract_first_rows_text(table_block, n=3)
    if first_rows_text:
        return _extract_statement_base_type(first_rows_text)
    return None


def _extract_first_rows_text(table_block, n=3):
    """提取表格前 n 行文本，用于表名识别。"""
    if not isinstance(table_block, dict):
        return ""
    for block in table_block.get("blocks", []):
        if (
            isinstance(block, dict)
            and block.get("type") == BlockType.TABLE_BODY
            and block.get("lines")
            and block["lines"][0].get("spans")
        ):
            html = block["lines"][0]["spans"][0].get("html", "")
            if not html:
                return ""
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")
            texts = []
            for row in rows[:n]:
                row_text = " ".join(
                    cell.get_text(strip=True) for cell in row.find_all(["td", "th"])
                )
                if row_text:
                    texts.append(row_text)
            return " ".join(texts)
    return ""


def _is_ignorable_block_before_first_table(block):
    if not isinstance(block, dict):
        return True
    block_type = block.get("type")
    if block_type in {
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.DISCARDED,
    }:
        return True
    if block_type in {BlockType.TITLE, BlockType.TEXT}:
        text = full_to_half(merge_para_with_text(block).strip()).lower()
        if not text:
            return True
        continuation_markers = tuple(marker.lower() for marker in CONTINUATION_MARKERS)
        if any(marker in text for marker in continuation_markers):
            return True
        if "registration no." in text or "incorporated in" in text:
            return True
        if len(text) <= 4 and text.isdigit():
            return True
    return False


def _find_first_table_block_index(page_info, previous_page_info=None):
    """查找页面中第一个 table block 的索引。

    在原有逻辑基础上增加英文财报支持：当表格前存在非可忽略的 block（如公司名、
    报表标题等）时，若该页表和上一页表对应索引相同且表格前 block 逐对完全相同
    （即页面顶部为重复的报表标题），也返回有效的表格索引。"""
    para_blocks = page_info.get("para_blocks") if isinstance(page_info, dict) else None
    if not isinstance(para_blocks, list) or not para_blocks:
        return None
    for idx, block in enumerate(para_blocks):
        if isinstance(block, dict) and block.get("type") == BlockType.TABLE:
            if all(_is_ignorable_block_before_first_table(prev) for prev in para_blocks[:idx]):
                return idx
            # 英文财报：表格前 block 与上一页完全相同 → 视为可忽略，也返回有效索引
            if previous_page_info is not None:
                prev_table_idx = _find_last_table_block_index(previous_page_info)
                if prev_table_idx is not None and _blocks_before_table_identical(
                    page_info, idx, previous_page_info, prev_table_idx
                ):
                    return idx
            return None
    return None


def _find_last_table_block_index(page_info):
    para_blocks = page_info.get("para_blocks") if isinstance(page_info, dict) else None
    if not isinstance(para_blocks, list) or not para_blocks:
        return None
    for idx in range(len(para_blocks) - 1, -1, -1):
        block = para_blocks[idx]
        if isinstance(block, dict) and block.get("type") == BlockType.TABLE:
            return idx
    return None


def merge_table(page_info_list):
    """合并跨页表格"""
    # 倒序遍历每一页
    for page_idx in range(len(page_info_list) - 1, -1, -1):
        # 跳过第一页，因为它没有前一页
        if page_idx == 0:
            continue

        page_info = page_info_list[page_idx]
        previous_page_info = page_info_list[page_idx - 1]

        current_table_idx = _find_first_table_block_index(page_info, previous_page_info)
        if current_table_idx is None:
            continue

        previous_table_idx = _find_last_table_block_index(previous_page_info)
        if previous_table_idx is None:
            continue

        current_table_block = page_info["para_blocks"][current_table_idx]
        previous_table_block = previous_page_info["para_blocks"][previous_table_idx]

        # 收集待合并表格的footnote
        wait_merge_table_footnotes = [
            block for block in current_table_block["blocks"]
            if block["type"] == BlockType.TABLE_FOOTNOTE
        ]

        # 检查两个表格是否可以合并
        # 0) 报表类型排他：分别从两侧识别报表基础类型（先看表格上方标题，
        #    标题缺失时再看表格内容关键词），两侧都识别成功且基础类型不同
        #    （资产负债表 / 利润表 / 现金流量表 / 所有者权益变动表），
        #    则视为新的一类报表，直接拒绝合并。
        #    注：单表式 Statement of Comprehensive Income 归入利润表家族。
        #    适用于：
        #    - 中文财报：表格上方有显式标题
        #    - 英文财报抬头复读但表内科目特征明显（如 BS 收尾 "TOTAL EQUITY AND
        #      LIABILITIES" vs 利润表起始 "Revenue"）
        current_stmt_base = (
            _find_nearest_statement_base_before_table(page_info, current_table_idx)
            or _classify_table_content_base_type(current_table_block)
        )
        previous_stmt_base = (
            _find_nearest_statement_base_before_table(previous_page_info, previous_table_idx)
            or _classify_table_content_base_type(previous_table_block)
        )
        if current_stmt_base and previous_stmt_base and current_stmt_base != previous_stmt_base:
            continue

        # 1) 中文财报：检查当前页表格前是否有"（续）"/"（续表）"等续标信号
        has_continuation_signal = _has_continuation_signal_before_table(page_info, current_table_idx)
        # 2) 英文财报：如果当前表格内容为英文，且两页表格索引相同，且表格前的
        #    header/text/title 等 block 逐对完全相同（如重复的公司名和报表标题），
        #    则视为跨页续表可合并
        is_english_merge = False
        if not has_continuation_signal and _is_table_content_english(current_table_block):
            if _blocks_before_table_identical(page_info, current_table_idx,
                                               previous_page_info, previous_table_idx):
                has_continuation_signal = True
                is_english_merge = True
        can_merge, soup1, soup2, current_html, previous_html = can_merge_tables(
            current_table_block, previous_table_block, has_continuation_signal=has_continuation_signal
        )

        if not can_merge:
            continue

        # 执行表格合并
        merged_html = perform_table_merge(
            soup1, soup2, previous_table_block, wait_merge_table_footnotes
        )

        # 更新previous_table_block的html
        for block in previous_table_block["blocks"]:
            if (block["type"] == BlockType.TABLE_BODY and block.get("lines") and block["lines"][0].get("spans")):
                block["lines"][0]["spans"][0]["html"] = merged_html
                break

        # 删除当前页的table
        for block in current_table_block["blocks"]:
            block['lines'] = []
            block[SplitFlag.LINES_DELETED] = True
        # 英文财报：当前页表格前的内容与上一页表格前完全相同（如重复报表标题），
        # 合并后当前页这些重复内容也应清除，避免输出中出现两份相同的标题/文字
        if is_english_merge:
            for idx in range(current_table_idx):
                block = page_info["para_blocks"][idx]
                if isinstance(block, dict):
                    block['lines'] = []
                    block[SplitFlag.LINES_DELETED] = True