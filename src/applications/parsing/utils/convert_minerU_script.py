import json
import time
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, DefaultDict, Tuple, Optional, Union
from collections import defaultdict
import re
import html as html_lib
from .heading_utils import assign_heading_levels as _assign_heading_levels, assign_ids_in_flat as _assign_ids_in_flat
from .table_classification_utils import classify_tables_async as _classify_tables_async
from .heading_utils_llm import llm_aided_title
from loguru import logger
 
MINERU_NORMALIZED_RANGE = 1000
# ---------------- 小工具函数 ----------------

# 财报文档关键词（用于判断是否启用标题分级/稳定ID）
_FIN_REPORT_KEYWORDS = (
    "年报", "年度报告", "半年报", "半年度报告",
    "季报", "季度报告", "财报摘要",
    "年度报告摘要", "半年度报告摘要", "季度报告摘要",
)

def ensure_table_defaults(entry: Dict[str, Any]) -> None:
    """为表格元素设置默认字段"""
    entry.setdefault('rows', [])
    entry.setdefault('html', "")
    entry.setdefault('md', "")
    entry.setdefault('perfectTable', False)

def ensure_rows_from_html(entry: Dict[str, Any]) -> None:
    """若存在 html 且 rows 为空，则解析 rows。"""
    try:
        html_val = entry.get('html')
        if html_val and (not entry.get('rows')):
            rows = parse_html_table_rows(html_val)
            if rows:
                entry['rows'] = rows
    except Exception:
        pass

def normalize_bbox(entry: Dict[str, Any], default_origin: str) -> None:
    """为 bBox 填充默认坐标系。"""
    if 'bBox' in entry and isinstance(entry['bBox'], dict):
        entry['bBox']['coord_origin'] = entry['bBox'].get('coord_origin', default_origin)

def _extract_text_from_nested_content(raw: Any) -> Optional[str]:
    """从 MinerU 3.x 嵌套 content 结构中提取可读文本。"""
    if isinstance(raw, str):
        text = raw.strip()
        return text if text else None

    if isinstance(raw, dict):
        # 直出文本字段
        for key in ("text", "content", "value"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # 容器字段（paragraph/title/page_header/page_number 等）
        container_keys = (
            "paragraph_content",
            "title_content",
            "page_header_content",
            "page_footer_content",
            "page_number_content",
            "list_content",
            "content",
        )
        parts: List[str] = []
        for key in container_keys:
            val = raw.get(key)
            extracted = _extract_text_from_nested_content(val)
            if extracted:
                parts.append(extracted)
        if parts:
            return "\n".join(parts)
        return None

    if isinstance(raw, list):
        parts: List[str] = []
        for item in raw:
            extracted = _extract_text_from_nested_content(item)
            if extracted:
                parts.append(extracted)
        if parts:
            return "\n".join(parts)
    return None

def _extract_string_list_from_nested(raw: Any) -> List[str]:
    """将嵌套 caption/footnote 结构归一为字符串列表。"""
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
                continue
            extracted = _extract_text_from_nested_content(item)
            if extracted:
                out.append(extracted)
        return out
    if isinstance(raw, dict):
        extracted = _extract_text_from_nested_content(raw)
        return [extracted] if extracted else []
    return []


def _extract_list_items_from_nested(raw: Any) -> List[str]:
    """从 MinerU list/list_items 结构中提取字符串列表。"""
    if isinstance(raw, dict) and isinstance(raw.get("content"), dict):
        raw = raw.get("content")

    if isinstance(raw, dict):
        return _extract_list_items_from_nested(raw.get("list_items"))

    if not isinstance(raw, list):
        return []

    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue

        if not isinstance(item, dict):
            extracted = _extract_text_from_nested_content(item)
            if extracted:
                out.append(extracted)
            continue

        item_content = item.get("item_content")
        extracted = _extract_text_from_nested_content(item_content)
        if not extracted:
            extracted = _extract_text_from_nested_content(item)
        if extracted:
            out.append(extracted)
    return out


def _has_required_text_payload(entry: Dict[str, Any]) -> bool:
    """校验 text 条目是否具备非空 value/md。"""
    value = entry.get("value")
    md = entry.get("md")
    return (
        isinstance(value, str)
        and value.strip() != ""
        and isinstance(md, str)
        and md.strip() != ""
    )

def convert_bbox_from_normalized(bbox: List[float], page_width: int = 595, page_height: int = 842, normalized_range: int = MINERU_NORMALIZED_RANGE) -> Dict[str, float]:
    """将归一化坐标系的 bbox 转换为实际页面坐标。
    
    Args:
        bbox: [left, top, right, bottom] 格式的归一化坐标
        page_width: 页面实际宽度（单位：点）。默认595点为 A4 纸宽度，实际应从 middle_json 的 pdf_info 中获取
        page_height: 页面实际高度（单位：点）。默认842点为 A4 纸高度，实际应从 middle_json 的 pdf_info 中获取
        normalized_range: 归一化坐标的范围（默认1000，minerU源码中使用1000）
    
    Returns:
        包含 l, t, r, b, coord_origin 的字典
    
    参考 bbox_demo.py 的转换逻辑：
        x0 = round(bbox_normalized[0] * W / normalized_range)
        y0 = round(bbox_normalized[1] * H / normalized_range)
    
    注意：调用此函数时应从 middle_json 的 pdf_info 中获取实际的页面尺寸，
          而不是使用默认值，以确保坐标转换的准确性。
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {'l': 0, 't': 0, 'r': 0, 'b': 0, 'coord_origin': 'TOPLEFT'}
    
    l = round(bbox[0] * page_width / normalized_range)
    t = round(bbox[1] * page_height / normalized_range)
    r = round(bbox[2] * page_width / normalized_range)
    b = round(bbox[3] * page_height / normalized_range)
    
    return {
        'l': float(l),
        't': float(t),
        'r': float(r),
        'b': float(b),
        'coord_origin': 'TOPLEFT'
    }


# ---------------- 以下转化函数 ----------------

def attach_file_id(items: List[Dict[str, Any]], file_id: Any) -> List[Dict[str, Any]]:
    """为每个元素附加 file_id（若提供）。"""
    if file_id is None:
        return items
    try:
        for it in items:
            it["file_id"] = file_id
    except Exception:
        pass
    return items

def load_json(path: Path) -> Any:
    """安全地加载JSON文件。"""
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
        return None

def save_json(path: Path, data: Any) -> None:
    """将数据保存为格式化的JSON文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def html_to_markdown_table(html: str) -> str:
    """将简单 HTML 表格 (<tr><td>...) 转为 Markdown 表格。"""
    if not html:
        return ""
    clean = re.sub(r"\s+", " ", html)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", clean, flags=re.IGNORECASE)
    if not rows:
        return ""
    md_rows: List[str] = []
    max_cols = 0
    for i, row_html in enumerate(rows):
        cells = re.findall(r"<(td|th)(?: [^>]*)?>(.*?)</(td|th)>", row_html, flags=re.IGNORECASE)
        texts: List[str] = []
        for _topen, inner, _tclose in cells:
            text = re.sub(r"<[^>]+>", "", inner).strip()
            text = text.replace("|", r"\|")
            texts.append(text)
        if not texts:
            continue
        max_cols = max(max_cols, len(texts))
        md_rows.append("| " + " | ".join(texts) + " |")
        if i == 0:
            md_rows.insert(1, "| " + " | ".join(["---"] * len(texts)) + " |")
    # 补全分隔符长度
    if len(md_rows) > 1:
        md_rows[1] = "| " + " | ".join(["---"] * max_cols) + " |"
    return "\n".join(md_rows)

def parse_html_table_rows(table_html: str, source_page_no: Optional[int] = None) -> List[List[Dict[str, Any]]]:
    """解析 <table> 的 HTML，提取每个单元格文本、跨行/跨列与起始位置。

    返回的数据结构为二维数组（按行），每个单元格包含：
    - text: 文本
    - rowspan: 行合并数
    - colspan: 列合并数
    - rowstart: 起始行索引（从 0 开始）
    - colstart: 起始列索引（从 0 开始）
    - source_page_no: 物理来源页码（若调用方提供 source_page_no）
    """
    if not table_html:
        return []
    clean = re.sub(r"\s+", " ", table_html)
    row_html_list = re.findall(r"<tr[^>]*>(.*?)</tr>", clean, flags=re.IGNORECASE | re.DOTALL)
    if not row_html_list:
        return []

    parsed_rows: List[List[Dict[str, Any]]] = []
    cell_pattern = re.compile(r"<(td|th)([^>]*)>(.*?)</(td|th)>", flags=re.IGNORECASE | re.DOTALL)

    # 记录来自上一行(及以上)的 rowspan 占位：future_row -> 已占用列索引集合
    carry_span: Dict[int, set] = {}

    for r_idx, row_html in enumerate(row_html_list):
        row_cells: List[Dict[str, Any]] = []
        taken_cols = set(carry_span.pop(r_idx, set()))  # 当前行已被上方行占用的列

        # 遍历本行所有单元格，计算其 colstart
        for _tag_open, attrs, inner, _tag_close in cell_pattern.findall(row_html):
            rs = 1
            cs = 1
            m_rs = re.search(r"rowspan\s*=\s*[\"']?(\d+)[\"']?", attrs, flags=re.IGNORECASE)
            m_cs = re.search(r"colspan\s*=\s*[\"']?(\d+)[\"']?", attrs, flags=re.IGNORECASE)
            if m_rs:
                try:
                    rs = max(int(m_rs.group(1)), 1)
                except Exception:
                    rs = 1
            if m_cs:
                try:
                    cs = max(int(m_cs.group(1)), 1)
                except Exception:
                    cs = 1

            text = re.sub(r"<[^>]+>", "", inner).strip()
            text = html_lib.unescape(text)

            # 计算此单元格的 colstart：寻找第一个未被占用的列
            c_start = 0
            while True:
                # 如果当前范围 [c_start, c_start+cs) 有任一列被占用，则右移
                conflict = any(((c_start + k) in taken_cols) for k in range(cs))
                if not conflict:
                    break
                c_start += 1

            # 标记当前行占用
            for k in range(cs):
                taken_cols.add(c_start + k)

            # 为后续行登记 rowspan 占用
            if rs > 1:
                for rr in range(1, rs):
                    target_r = r_idx + rr
                    s = carry_span.setdefault(target_r, set())
                    for k in range(cs):
                        s.add(c_start + k)

            cell_obj = {
                'text': text,
                'rowspan': rs,
                'colspan': cs,
                'rowstart': r_idx,
                'colstart': c_start,
            }
            if isinstance(source_page_no, int):
                cell_obj["source_page_no"] = source_page_no
            row_cells.append(cell_obj)

        parsed_rows.append(row_cells)

    return parsed_rows 

def _render_markdown_from_values(rows: List[List[str]]) -> str:
    """将二维文本行数据渲染为简单 Markdown 表格（首行作表头）。"""
    if not rows:
        return ""
    try:
        def _join(line: List[str]) -> str:
            return "| " + " | ".join((str(v) if v is not None else "") for v in line) + " |"
        if len(rows) == 1:
            return _join(rows[0])
        header = _join(rows[0])
        sep = "| " + " | ".join(["---" for _ in rows[0]]) + " |"
        body = [_join(r) for r in rows[1:]]
        return "\n".join([header, sep, *body])
    except Exception:
        return ""

def _ensure_normalized_table_fields(entry: Dict[str, Any]) -> None:
    """为 table 元素计算并填充 html/md（将合并单元格展开）。"""
    try:
        if not isinstance(entry, dict) or entry.get("type") != "table":
            return
        rows_obj = entry.get("rows")
        if not rows_obj and entry.get("html"):
            try:
                rows_obj = parse_html_table_rows(entry.get("html") or "")
                if rows_obj:
                    entry["rows"] = rows_obj
            except Exception:
                rows_obj = None
        if not rows_obj or not isinstance(rows_obj, list):
            return
        grid_vals = _expand_table_rows_with_spans(rows_obj)
        md_norm = _render_markdown_from_values(grid_vals)
        if isinstance(md_norm, str) and md_norm.strip():
            entry["md"] = md_norm
    except Exception:
        pass

"""
表格类型识别逻辑统一在 table_classification_utils 中实现；本文件仅调用 _classify_tables_async。
"""

# ---------------- 二维表格json数据构建 ----------------

def _is_financial_three_statement(label: Optional[str]) -> bool:
    if not isinstance(label, str):
        return False
    label = label.strip()
    financial_labels = {
        "合并资产负债表",
        "母公司资产负债表",
        "合并利润表",
        "母公司利润表",
        "合并现金流量表",
        "母公司现金流量表",

    }
    return label in financial_labels

def _normalize_numeric(value: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return ""
    # 特殊值保持为原始字符串
    if text in {"不适用", "N/A", "NA", "--"}:
        return text
    # 移除中文空格、千分位、普通空格
    cleaned = text.replace("\u3000", " ")
    cleaned = cleaned.replace(",", "").replace(" ", "")
    # 百分号保留为字符串
    if cleaned.endswith("%"):
        return text
    # 处理可能的中文负号
    cleaned = cleaned.replace("－", "-")
    # 小数点可能被错误加空格点等，简单规范化
    cleaned = cleaned.replace("。", ".")
    try:
        return float(cleaned)
    except Exception:
        return value

def _expand_table_rows_with_spans(rows: List[List[Dict[str, Any]]]) -> List[List[str]]:
    """将带有 rowspan/colspan 的行列展开为二维文本网格。

    若单元格包含 rowstart/colstart，则严格按其起始位置放置；否则回退为“第一个可用空位”策略。
    """
    grid: List[List[str]] = []
    pending: Dict[int, List[Tuple[int, str]]] = {}

    def _ensure_len(arr: List[Optional[str]], n: int) -> None:
        while len(arr) <= n:
            arr.append(None)

    for r_idx, raw_row in enumerate(rows or []):
        # 使用 None 占位，便于判断空位
        row_vals: List[Optional[str]] = []

        # 1) 预填充来自上方的 rowspan 文本：严格放在指定列索引
        carry = pending.pop(r_idx, []) if r_idx in pending else []
        for col_idx, text in sorted(carry, key=lambda x: x[0]):
            _ensure_len(row_vals, col_idx)
            row_vals[col_idx] = text

        # 2) 将当前行单元格按 colstart 或“第一个可用空位”插入（遇到 colspan 占用多个列）
        cursor = 0
        for cell in raw_row:
            text = str(cell.get("text", "")).strip()
            rs = cell.get("rowspan") or 1
            cs = cell.get("colspan") or 1
            try:
                rs = int(rs) if int(rs) > 0 else 1
            except Exception:
                rs = 1
            try:
                cs = int(cs) if int(cs) > 0 else 1
            except Exception:
                cs = 1

            # 优先使用提供的 colstart（兼容 colStart 作为回退）
            declared_col_start = cell.get("colstart")
            if declared_col_start is None:
                declared_col_start = cell.get("colStart")
            start_c: Optional[int] = None
            if isinstance(declared_col_start, int) and declared_col_start >= 0:
                start_c = declared_col_start
                _ensure_len(row_vals, start_c)
                # 若冲突（被占用），则就近右移
                while start_c < len(row_vals) and row_vals[start_c] is not None:
                    start_c += 1
            else:
                # 找到第一个空位作为起始列
                while True:
                    if cursor >= len(row_vals):
                        row_vals.append(None)
                        break
                    if row_vals[cursor] is None:
                        break
                    cursor += 1
                start_c = cursor

            # 在当前行填充 colspan 列
            for k in range(cs):
                idx = start_c + k
                _ensure_len(row_vals, idx)
                row_vals[idx] = text

            # 为后续行登记 rowspan 的填充列
            if rs > 1:
                for rr in range(1, rs):
                    target_r = r_idx + rr
                    lst = pending.setdefault(target_r, [])
                    for k in range(cs):
                        lst.append((start_c + k, text))

            cursor = max(cursor, start_c + cs)

        # 将 None 转为空字符串
        grid.append([v if v is not None else "" for v in row_vals])

    # 3) 对齐至统一列数
    max_cols = max((len(r) for r in grid), default=0)
    for r in grid:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))
    return grid

def build_two_dim_table(element: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    if element.get("type") != "table":
        return None
    rows = element.get("rows")
    if not rows and element.get("html"):
        try:
            rows = parse_html_table_rows(element.get("html") or "")
        except Exception:
            rows = []
    if not rows or not isinstance(rows, list):
        return None

    grid = _expand_table_rows_with_spans(rows)
    if not grid or len(grid) < 2:
        return None

    header = [c.strip() for c in grid[0]]
    # 找到“项目/科目”列索引（若不存在，默认第 0 列）
    stub_idx = 0
    for idx, h in enumerate(header):
        if h in {"项目", "科目", "目 录", "目錄", "Item", "科 目"}:
            stub_idx = idx
            break

    # 顶层键为各列标题（除去 stub 列）
    col_headers: List[Tuple[int, str]] = []
    for j, h in enumerate(header):
        if j == stub_idx:
            continue
        if not h:
            # 避免空表头
            h = f"列{j+1}"
        col_headers.append((j, h))

    result: Dict[str, Dict[str, Any]] = {}
    for j, h in col_headers:
        result[h] = {}

    for r in grid[1:]:
        if not r or stub_idx >= len(r):
            continue
        row_key = (r[stub_idx] or "").strip()
        if not row_key:
            continue
        for j, h in col_headers:
            if j >= len(r):
                continue
            val = r[j]
            result[h][row_key] = _normalize_numeric(val)

    # 若结果全空则返回 None
    if all((not v) for v in result.values()):
        return None
    return result

def build_table_2d_for_items(items: List[Dict[str, Any]]) -> None:
    for element in items or []:
        if element.get("type") != "table":
            continue
        if not _is_financial_three_statement(element.get("table_type")):
            continue
        try:
            data2d = build_two_dim_table(element)
            if data2d is not None:
                element["two_d_data"] = data2d
        except Exception:
            # 忽略单表构建失败
            pass

# ---------------- 文档类型判断 ----------------

def is_financial_report_name(name: Optional[str]) -> bool:
    """基于文件名判断是否为财报类文档（年报/季报/半年度等）。
    
    用途：决定是否启用标题层级分配和稳定ID功能。
    """
    if not name:
        return False
    try:
        base_name = Path(name).stem
    except Exception:
        base_name = str(name)
    return any(kw in base_name for kw in _FIN_REPORT_KEYWORDS)

def convert_from_content_list(items: List[Dict[str, Any]], pdf_info: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """将 content_list.json 的条目映射到目标结构。
    
    Args:
        items: content_list 中的元素列表
        pdf_info: 从 middle_json 中提取的 pdf_info，包含每页的尺寸信息
    """
    # 如果有 pdf_info，从中提取标题层级信息（用于 LLM 标题分级）
    title_levels_map = {}
    if pdf_info:
        total_titles_in_pdf_info = 0
        titles_with_level = 0
        for page in pdf_info:
            page_idx = page.get('page_idx', 0)
            para_blocks = page.get('para_blocks', [])
            title_index_in_page = 0
            for block in para_blocks:
                if block.get('type') == 'title':
                    total_titles_in_pdf_info += 1
                    # 用 (page_idx, title_index_in_page) 作为键
                    if 'level' in block:
                        title_levels_map[(page_idx, title_index_in_page)] = block['level']
                        titles_with_level += 1
                    title_index_in_page += 1
        
        logger.info(f"从 pdf_info 中找到 {total_titles_in_pdf_info} 个标题，其中 {titles_with_level} 个有 LLM 层级信息")
        if titles_with_level > 0:
            logger.info(f"标题层级映射样例: {dict(list(title_levels_map.items())[:5])}")
    
    results: List[Dict[str, Any]] = []
    # 跟踪每页的标题索引
    title_counter_per_page = {}
    
    # 调试：打印第一个元素的结构（INFO 级别以便在日志中查看）
    if items and len(items) > 0:
        first_item = items[0]
        logger.info(f"content_list 第一个元素的字段: {list(first_item.keys()) if isinstance(first_item, dict) else type(first_item)}")
        logger.info(f"content_list 第一个元素内容: {str(first_item)[:500]}")
    
    for it in items:
        if not isinstance(it, dict):
            continue
        # 兼容不同的字段名：MinerU 可能使用 page 或 page_idx
        page_idx = it.get('page_idx')
        if page_idx is None:
            page_idx = it.get('page')
            # 如果是 page（1-indexed），转换为 page_idx（0-indexed）
            if isinstance(page_idx, int) and page_idx > 0:
                page_idx = page_idx - 1
        
        t = it.get('type')
        
        # 兼容不同的字段名：MinerU 可能使用 text, content, value
        text = it.get('text')
        if text is None:
            content_val = it.get('content')
            if isinstance(content_val, str):
                text = content_val
            else:
                text = _extract_text_from_nested_content(content_val)
        if text is None:
            text = it.get('value')
        if not isinstance(text, str):
            text = _extract_text_from_nested_content(text)
        
        text_level = it.get('text_level')
        if text_level is None:
            # 兼容 level 字段
            text_level = it.get('level')
        
        # 兼容不同的字段名：bbox 或 bBox
        bbox = it.get('bbox')
        if bbox is None:
            bbox = it.get('bBox')
            # 如果 bBox 是 dict 格式，转换为 list 格式
            if isinstance(bbox, dict):
                bbox = [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
        entry: Dict[str, Any] = {}
        if isinstance(page_idx, int):
            entry['page'] = page_idx + 1
        if t:
            # 忽略 header 和 footer 类型，不进行转换
            if t in ('header', 'footer', 'page_header', 'page_footer', 'page_number'):
                continue
            if t in ('text', 'title'):
                entry['type'] = 'text'

                is_potential_title = (t == 'title') or (text_level is not None)
                
                if is_potential_title and isinstance(page_idx, int):
                    # 获取当前页面的标题索引
                    if page_idx not in title_counter_per_page:
                        title_counter_per_page[page_idx] = 0
                    title_idx = title_counter_per_page[page_idx]
                    title_counter_per_page[page_idx] += 1
                    
                    # 从映射中获取层级
                    if title_levels_map:
                        llm_level = title_levels_map.get((page_idx, title_idx))
                        if llm_level is not None and llm_level > 0:
                            entry['headingLevel'] = int(llm_level)
                            text_preview = text[:30] if isinstance(text, str) else ""
                            logger.debug(f"标题 '{text_preview}' (page={page_idx}, idx={title_idx}) 设置层级为 {llm_level}")
                
                # 处理原有的 text_level（兼容性）
                if text_level is not None:
                    try:
                        lvl = int(text_level)
                        if lvl >= 0:
                            entry['textLevel'] = lvl
                    except Exception:
                        pass
            elif t == 'list':
                # 保留 list 类型并处理 list_items
                entry['type'] = 'list'
                sub_type = it.get('sub_type')
                if sub_type:
                    entry['sub_type'] = sub_type
                list_items = _extract_list_items_from_nested(it.get('list_items'))
                if not list_items:
                    list_items = _extract_list_items_from_nested(it.get('content'))
                if list_items:
                    entry['list_items'] = list_items
                    # 生成 markdown 格式的列表文本
                    entry['md'] = '\n'.join(list_items)
                    entry['value'] = '\n'.join(list_items)
            elif t == 'code':
                # 处理 code_body
                entry['type'] = 'text'
                entry['value'] = it.get('code_body')
                entry['md'] = it.get('code_body')
            else:
                entry['type'] = t
                # 兜底：将未见过/不在 Schema 的类型统一归一为 text
                try:
                    _allowed_types = {"text", "table", "image", "list"}
                    if entry.get('type') not in _allowed_types:
                        entry['type'] = 'text'
                except Exception:
                    entry['type'] = 'text'
        if isinstance(bbox, list) and len(bbox) == 4:
            # 使用 convert_bbox_from_normalized 将归一化坐标转换为实际坐标
            # 根据 page_idx 获取对应页面的尺寸
            page_width, page_height = 595, 842  # 默认 A4 尺寸
            if pdf_info and isinstance(page_idx, int) and 0 <= page_idx < len(pdf_info):
                page_size = pdf_info[page_idx].get('page_size')
                if page_size and isinstance(page_size, list) and len(page_size) == 2:
                    page_width, page_height = page_size[0], page_size[1]
            entry['bBox'] = convert_bbox_from_normalized(bbox, page_width, page_height)
        # 对于非 list 类型，使用 text 字段
        if isinstance(text, str) and text.strip() and entry.get('type') != 'list':
            entry['value'] = text
            entry['md'] = text
        if entry.get('type') == 'text' and not _has_required_text_payload(entry):
            continue
        if entry.get('type') == 'table':
            table_content = it.get('content') if isinstance(it.get('content'), dict) else {}
            table_body = it.get('table_body')
            if not (isinstance(table_body, str) and table_body.strip()):
                content_html = table_content.get('html')
                if isinstance(content_html, str):
                    table_body = content_html

            img_path = it.get('img_path')
            if not (isinstance(img_path, str) and img_path.strip()):
                image_source = table_content.get('image_source')
                if isinstance(image_source, dict):
                    img_path = image_source.get('path')

            table_caption = it.get('table_caption')
            if not table_caption:
                table_caption = _extract_string_list_from_nested(table_content.get('table_caption'))

            table_footnote = it.get('table_footnote')
            if not table_footnote:
                table_footnote = _extract_string_list_from_nested(table_content.get('table_footnote'))
            
            has_table_body = isinstance(table_body, str) and table_body.strip() != ""
            has_img = isinstance(img_path, str) and img_path.strip() != ""
            if not has_table_body and not has_img:
                continue
            ensure_table_defaults(entry)
            if has_table_body:
                entry['html'] = table_body
                ensure_rows_from_html(entry)
                _ensure_normalized_table_fields(entry)
            
            # 保留表格标题、脚注和图片路径信息
            if table_caption:
                entry['table_caption'] = table_caption
            if table_footnote:
                entry['table_footnote'] = table_footnote
            if img_path:
                entry['img_path'] = img_path
        # image 类型：保留图像相关字段
        if entry.get('type') == 'image':
            image_content = it.get('content') if isinstance(it.get('content'), dict) else {}
            img_path = it.get('img_path')
            if not (isinstance(img_path, str) and img_path.strip()):
                image_source = image_content.get('image_source')
                if isinstance(image_source, dict):
                    img_path = image_source.get('path')
            if isinstance(img_path, str) and img_path.strip():
                entry['img_path'] = img_path
            if 'image_caption' in it:
                entry['image_caption'] = it.get('image_caption')
            if 'image_footnote' in it:
                entry['image_footnote'] = it.get('image_footnote')
        if entry:
            results.append(entry)
    return results

def convert_from_model(model_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 model.json 做兜底转换，保留其已有的表格字段。"""
    results: List[Dict[str, Any]] = []
    flat_items: List[Dict[str, Any]] = []
    for it in model_items:
        if isinstance(it, list):
            for sub in it:
                if isinstance(sub, dict):
                    flat_items.append(sub)
            continue
        if isinstance(it, dict):
            flat_items.append(it)

    for it in flat_items:
        entry: Dict[str, Any] = {}
        for k in ['page', 'type', 'value', 'md', 'bBox', 'rows', 'html', 'perfectTable', 'table_caption', 'table_footnote', 'img_path', 'image_caption', 'image_footnote', 'sub_type', 'list_items']:
            if k in it:
                entry[k] = it[k]
        if entry.get('type') == 'page_number':
            continue
        if 'value' not in entry and isinstance(it.get('content'), str):
            entry['value'] = it['content']
        if 'md' not in entry and isinstance(it.get('content'), str):
            entry['md'] = it['content']
        if 'type' not in entry and ('rows' in it or 'html' in it):
            entry['type'] = 'table'
        if 'bbox' in it and 'bBox' not in entry:
            bbox_val = it.get('bbox')
            if isinstance(bbox_val, dict):
                entry['bBox'] = bbox_val
            elif isinstance(bbox_val, list) and len(bbox_val) == 4:
                entry['bBox'] = convert_bbox_from_normalized(bbox_val)
        # 兜底：模型输出中的未知类型一律归一为 text（除 table/image/list 外）
        try:
            _allowed_types = {"text", "table", "image", "list"}
            if entry.get('type') not in _allowed_types:
                entry['type'] = 'text'
        except Exception:
            entry['type'] = 'text'
        if entry.get('type') == 'text':
            if not _has_required_text_payload(entry):
                continue
        normalize_bbox(entry, default_origin='BOTTOMLEFT')
        if entry.get('type') == 'table':
            ensure_table_defaults(entry)
        if entry:
            results.append(entry)
    return results

def build_model_tables_index(model_items: List[Dict[str, Any]]) -> Tuple[DefaultDict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    tables_by_page: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    global_tables: List[Dict[str, Any]] = []
    
    # 处理嵌套列表结构：model_output 可能是 [[...]] 而非 [...]
    flat_items = model_items
    if model_items and isinstance(model_items[0], list):
        # 展平嵌套列表
        flat_items = []
        for sub in model_items:
            if isinstance(sub, list):
                flat_items.extend(sub)
            elif isinstance(sub, dict):
                flat_items.append(sub)
    
    for it in flat_items:
        if not isinstance(it, dict):
            continue
        is_table_like = (it.get('type') == 'table') or ('rows' in it) or ('html' in it)
        if not is_table_like:
            continue
        page = it.get('page')
        if isinstance(page, int):
            tables_by_page[page].append(it)
        else:
            global_tables.append(it)
    return tables_by_page, global_tables

def enrich_tables_with_model(results: List[Dict[str, Any]], model_items: List[Dict[str, Any]]) -> None:
    if not results or not model_items:
        return
    tables_by_page, global_tables = build_model_tables_index(model_items)
    page_to_idx: Dict[int, int] = {}
    global_idx = 0
    for entry in results:
        if entry.get('type') != 'table':
            continue
        page = entry.get('page')
        model_table = None
        if isinstance(page, int) and page in tables_by_page:
            lst = tables_by_page[page]
            pos = page_to_idx.get(page, 0)
            if pos < len(lst):
                model_table = lst[pos]
                page_to_idx[page] = pos + 1
        if model_table is None and global_idx < len(global_tables):
            model_table = global_tables[global_idx]
            global_idx += 1
        if model_table is None:
            continue
        for k in ['rows', 'html', 'perfectTable', 'md', 'table_caption', 'table_footnote', 'img_path', 'sub_type', 'list_items']:
            if (k not in entry or not entry.get(k)) and (k in model_table and model_table[k]):
                entry[k] = model_table[k]
        if 'bBox' in model_table and isinstance(model_table['bBox'], dict):
            entry['bBox'] = model_table['bBox']
        elif 'bbox' in model_table and isinstance(model_table['bbox'], dict):
            entry['bBox'] = model_table['bbox']
        normalize_bbox(entry, default_origin='BOTTOMLEFT')
        if entry.get('type') == 'table':
            ensure_rows_from_html(entry)
            _ensure_normalized_table_fields(entry)

def _pick_payload_container(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict): return {}
    results = d.get('results')
    if isinstance(results, dict) and results:
        try:
            first_key = next(iter(results))
            node = results.get(first_key)
            if isinstance(node, dict): return node
        except Exception: pass
    for _k, v in d.items():
        if isinstance(v, dict) and any(x in v for x in ['content_list', 'model_output', 'middle_json']):
            return v
    return d

def _safe_parse_json_str(maybe_str: Any) -> Any:
    if isinstance(maybe_str, str):
        try: return json.loads(maybe_str)
        except Exception: return None
    return maybe_str

def _normalize_content_list_items(raw_items: Any) -> Optional[List[Dict[str, Any]]]:
    """兼容 MinerU 不同版本的 content_list 结构，统一为一维 dict 列表。

    - 2.x 常见结构: List[Dict]
    - 3.x 某些返回: List[List[Dict]]（按页分组）
    """
    if not isinstance(raw_items, list):
        return None
    if not raw_items:
        return []

    if all(isinstance(it, dict) for it in raw_items):
        return raw_items

    flattened: List[Dict[str, Any]] = []
    for page_idx, page_items in enumerate(raw_items):
        if isinstance(page_items, dict):
            flattened.append(page_items)
            continue
        if not isinstance(page_items, list):
            continue

        for item in page_items:
            if not isinstance(item, dict):
                continue
            if "page_idx" not in item and "page" not in item:
                patched = dict(item)
                patched["page_idx"] = page_idx
                flattened.append(patched)
            else:
                flattened.append(item)

    return flattened if flattened else []

def _safe_get_mineru_table_html(table_block: Dict[str, Any]) -> str:
    try:
        blocks = table_block.get("blocks") or []
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "table_body":
                continue
            lines = blk.get("lines") or []
            if not isinstance(lines, list) or not lines:
                continue
            first_line = lines[0] if isinstance(lines[0], dict) else {}
            spans = first_line.get("spans") or []
            if not isinstance(spans, list) or not spans:
                continue
            first_span = spans[0] if isinstance(spans[0], dict) else {}
            html = first_span.get("html")
            return html if isinstance(html, str) else ""
    except Exception:
        return ""
    return ""

def _mineru_table_block_is_deleted(table_block: Dict[str, Any]) -> bool:
    try:
        blocks = table_block.get("blocks") or []
        if not isinstance(blocks, list) or not blocks:
            return False
        # mineru_parsers.merge_table 会在被合并的 table 的每个 block 上打 lines_deleted=True 并清空 lines
        if any(isinstance(b, dict) and b.get("lines_deleted") is True for b in blocks):
            return True
        # 兜底：全部 lines 为空也视为被删
        if all(isinstance(b, dict) and not (b.get("lines") or []) for b in blocks):
            return True
    except Exception:
        return False
    return False


def _mineru_text_from_block(block: Dict[str, Any]) -> str:
    parts: list[str] = []
    try:
        for line in block.get("lines") or []:
            if not isinstance(line, dict):
                continue
            for span in line.get("spans") or []:
                if isinstance(span, dict):
                    text = span.get("content")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
    except Exception:
        return ""
    return " ".join(parts).strip()


def _mineru_block_is_ignorable_before_first_table(block: Dict[str, Any]) -> bool:
    if not isinstance(block, dict):
        return True
    block_type = block.get("type")
    if block_type in {"header", "footer", "page_number", "discarded"}:
        return True
    if block_type not in {"title", "text"}:
        return False
    text = re.sub(r"\s+", " ", _mineru_text_from_block(block)).strip().lower()
    if not text:
        return True
    if "continued" in text or "cont." in text:
        return True
    if "registration no." in text or "incorporated in" in text:
        return True
    if len(text) <= 4 and text.isdigit():
        return True
    return False


def _mineru_page_has_continuation_signal_before_table(para_blocks: Any, table_idx: int) -> bool:
    if not isinstance(para_blocks, list) or table_idx is None:
        return False
    for block in para_blocks[:table_idx]:
        if not isinstance(block, dict) or block.get("type") not in {"title", "text", "table_caption"}:
            continue
        text = re.sub(r"\s+", " ", _mineru_text_from_block(block)).strip().lower()
        if "continued" in text or "cont." in text:
            return True
    return False

def _recompute_converted_table_fields(entry: Dict[str, Any]) -> None:
    if not isinstance(entry, dict) or entry.get("type") != "table":
        return
    ensure_table_defaults(entry)
    # html 已更新；强制重算 rows/md
    entry["rows"] = []
    entry["md"] = ""
    ensure_rows_from_html(entry)
    _ensure_normalized_table_fields(entry)

def _cell_text_for_signature(cell: Any) -> str:
    if not isinstance(cell, dict):
        return str(cell or "").strip()
    for key in ("text", "md", "value", "content"):
        v = cell.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _row_signature_for_merge(row: Any) -> tuple[str, ...]:
    cells: list[Any] = []
    if isinstance(row, list):
        cells = row
    elif isinstance(row, dict) and isinstance(row.get("cells"), list):
        cells = row.get("cells") or []
    normalized = [re.sub(r"\s+", "", _cell_text_for_signature(c)) for c in cells]
    while normalized and not normalized[-1]:
        normalized.pop()
    return tuple(normalized)

def _ensure_row_cells_source_page_no(rows: Any, default_source_page_no: Optional[int]) -> None:
    """为 rows 中缺失 source_page_no 的单元格补齐物理页码。"""
    if not isinstance(rows, list) or not isinstance(default_source_page_no, int):
        return
    for row in rows:
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, dict) and not isinstance(cell.get("source_page_no"), int):
                cell["source_page_no"] = default_source_page_no

def _extract_row_cell_source_page_nos(row: Any, fallback_source_page_no: Optional[int]) -> list[Optional[int]]:
    out: list[Optional[int]] = []
    if not isinstance(row, list):
        return out
    for cell in row:
        if isinstance(cell, dict) and isinstance(cell.get("source_page_no"), int):
            out.append(cell.get("source_page_no"))
        else:
            out.append(fallback_source_page_no if isinstance(fallback_source_page_no, int) else None)
    return out

def _inject_source_page_no_into_parsed_rows(
    parsed_rows: Any,
    source_rows: Any,
    current_page_no: Optional[int],
) -> list[list[dict[str, Any]]]:
    """
    根据 source_rows 将 source_page_no 回填到 parsed_rows。
    规则：
    1) 若 source_rows 对应位置存在 source_page_no，则使用该值；
    2) 若 parsed_rows 单元格本身已有 source_page_no，则保持不变（严格不覆盖）；
    3) 仅对缺失 source_page_no 的单元格，用 current_page_no 作为兜底补齐。
    """
    parsed_norm: list[list[dict[str, Any]]] = [r for r in (parsed_rows or []) if isinstance(r, list)]
    source_norm: list[list[dict[str, Any]]] = [r for r in (source_rows or []) if isinstance(r, list)]

    output_rows: list[list[dict[str, Any]]] = []

    # 源 rows 构建行签名到 source_page_no 数组的映射
    sig_to_source_pages: dict[tuple[str, ...], list[list[Optional[int]]]] = defaultdict(list)
    for src_row in source_norm:
        sig = _row_signature_for_merge(src_row)
        if sig:
            sig_to_source_pages[sig].append(_extract_row_cell_source_page_nos(src_row, None))

    for row in parsed_norm:
        sig = _row_signature_for_merge(row)
        # 默认使用 current_page_no
        page_nos = []
        if sig in sig_to_source_pages and sig_to_source_pages[sig]:
            # 从 source_rows 对应行获取 source_page_no 分布
            page_nos = sig_to_source_pages[sig].pop(0)
        out_row: list[dict[str, Any]] = []
        for c_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            new_cell = dict(cell)
            # 优先用映射；其次保留现有值；最后仅补缺失
            mapped_page_no: Optional[int] = None
            if page_nos and c_idx < len(page_nos) and isinstance(page_nos[c_idx], int):
                mapped_page_no = page_nos[c_idx]
            if isinstance(mapped_page_no, int):
                new_cell["source_page_no"] = mapped_page_no
            elif not isinstance(new_cell.get("source_page_no"), int) and isinstance(current_page_no, int):
                new_cell["source_page_no"] = current_page_no
            out_row.append(new_cell)
        output_rows.append(out_row)
    return output_rows


_HTML_PARSE_FIELDS = frozenset({"text", "rowspan", "colspan", "rowstart", "colstart", "source_page_no"})


def _merge_coordinate_fields_into_parsed_rows(
    preserved_rows: Any,
    candidate_rows: Any,
) -> list[list[dict[str, Any]]]:
    """以 candidate_rows 的结构为准，从 preserved_rows 中按文本匹配回填坐标字段。

    parse_html_table_rows 生成的新单元格仅包含 text/rowspan/colspan/rowstart/colstart，
    不含 bBox/cellTraceabilitySource 等溯源坐标。本函数按行签名 + 文本匹配，
    将 preserved_rows 中对应单元格的坐标字段复制到 candidate_rows，避免坐标丢失。
    """
    if not isinstance(preserved_rows, list) or not isinstance(candidate_rows, list):
        return candidate_rows

    # 构建 preserved 中每个单元格的文本 → 坐标字段映射
    def _coord_fields(cell: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in cell.items() if k not in _HTML_PARSE_FIELDS}

    # 先尝试按行签名匹配，不行再按逐单元格文本匹配
    for cand_row in candidate_rows:
        if not isinstance(cand_row, list):
            continue
        cand_sig = _row_signature_for_merge(cand_row)
        best_preserved_row: Optional[list[dict[str, Any]]] = None
        for prev_row in preserved_rows:
            if not isinstance(prev_row, list):
                continue
            if _row_signature_for_merge(prev_row) == cand_sig:
                best_preserved_row = prev_row
                break
        if best_preserved_row is None:
            continue

        # 按位置一对一复制坐标字段
        for ci, cand_cell in enumerate(cand_row):
            if not isinstance(cand_cell, dict):
                continue
            if ci < len(best_preserved_row) and isinstance(best_preserved_row[ci], dict):
                coords = _coord_fields(best_preserved_row[ci])
                cand_cell.update(coords)

        # 若有剩余 preserved 单元格（HTML 解析可能丢失某些列），
        # 按文本匹配补回遗漏的坐标
        if len(best_preserved_row) > len(cand_row):
            text_to_coord: dict[str, dict[str, Any]] = {}
            for pc in best_preserved_row[len(cand_row):]:
                if isinstance(pc, dict) and pc.get("text"):
                    text_to_coord[pc["text"].strip()] = _coord_fields(pc)
            for cand_cell in cand_row:
                if isinstance(cand_cell, dict) and cand_cell.get("text"):
                    key = cand_cell["text"].strip()
                    if key in text_to_coord and not any(
                        k in cand_cell for k in text_to_coord[key]
                    ):
                        cand_cell.update(text_to_coord[key])

    return candidate_rows


def _merge_rows_preserving_cells(prev_rows: Any, next_rows: Any) -> list[list[dict[str, Any]]]:
    """合并跨页表格 rows，尽量保留 OCR/模型注入的单元格坐标等字段。"""
    if not isinstance(prev_rows, list) or not isinstance(next_rows, list):
        return prev_rows if isinstance(prev_rows, list) else []

    prev_norm: list[list[dict[str, Any]]] = [r for r in prev_rows if isinstance(r, list)]
    next_norm: list[list[dict[str, Any]]] = [r for r in next_rows if isinstance(r, list)]
    if not prev_norm:
        return next_norm
    if not next_norm:
        return prev_norm

    prefix_len = 0
    max_prefix = min(5, len(prev_norm), len(next_norm))
    while prefix_len < max_prefix:
        sig_a = _row_signature_for_merge(prev_norm[prefix_len])
        sig_b = _row_signature_for_merge(next_norm[prefix_len])
        if sig_a and sig_a == sig_b:
            prefix_len += 1
            continue
        break

    return prev_norm + next_norm[prefix_len:]

def _merge_str_field(dst: Dict[str, Any], src: Dict[str, Any], field: str, sep: str = "\n") -> None:
    try:
        a = dst.get(field)
        b = src.get(field)
        a = a if isinstance(a, str) else ""
        b = b if isinstance(b, str) else ""
        if not b.strip():
            return
        if not a.strip():
            dst[field] = b
            return
        if b.strip() in a:
            return
        dst[field] = a.rstrip() + sep + b.lstrip()
    except Exception:
        return

def _merge_table_in_converted_item(
    converted_items: List[Dict[str, Any]],
    pdf_info: Optional[List[Dict[str, Any]]],
) -> None:
    """按 pdf_info 的 lines_deleted 标记，在 converted_items 内执行跨页表合并。

    规则：
    - 若第 N 页首个 para_block 是被删除(table continuation)，则视为与 N-1 页末表跨页合并；
    - 使用“按行拼接 + 前缀去重”方式合并 rows，保持二维行结构，支持连续多页链式合并；
    - 使用 pdf_info 中上一页末表的 merged html 回写目标表，并重算 md；
    - 删除 N 页续表片段，避免后续重复处理。
    """
    if not isinstance(converted_items, list) or not isinstance(pdf_info, list):
        return

    # page(1-index) -> converted_items 中 table 的索引（按出现顺序）
    page_to_table_indices: DefaultDict[int, List[int]] = defaultdict(list)
    for idx, it in enumerate(converted_items):
        if not isinstance(it, dict) or it.get("type") != "table":
            continue
        page_no = it.get("page")
        if isinstance(page_no, int):
            page_to_table_indices[page_no].append(idx)

    merged_pairs = 0
    removed_indices: List[int] = []

    for page_idx in range(len(pdf_info) - 1, 0, -1):
        page_obj = pdf_info[page_idx]
        if not isinstance(page_obj, dict):
            continue
        para_blocks = page_obj.get("para_blocks") or []
        if not isinstance(para_blocks, list) or not para_blocks:
            continue

        table_block_idx = None
        for idx, block in enumerate(para_blocks):
            if isinstance(block, dict) and block.get("type") == "table":
                if all(_mineru_block_is_ignorable_before_first_table(prev) for prev in para_blocks[:idx]) or _mineru_page_has_continuation_signal_before_table(para_blocks, idx):
                    table_block_idx = idx
                break
        if table_block_idx is None:
            continue
        first_block = para_blocks[table_block_idx]
        if not isinstance(first_block, dict) or first_block.get("type") != "table":
            continue
        if not _mineru_table_block_is_deleted(first_block):
            continue

        current_page_no = page_idx + 1
        prev_page_no = current_page_no - 1

        cur_table_idxs = page_to_table_indices.get(current_page_no) or []
        prev_table_idxs = page_to_table_indices.get(prev_page_no) or []
        if not cur_table_idxs or not prev_table_idxs:
            continue

        cur_idx = cur_table_idxs[0]      # 当前页首表（续表）
        prev_idx = prev_table_idxs[-1]   # 上一页末表（被并入目标）

        cur_item = converted_items[cur_idx]
        prev_item = converted_items[prev_idx]
        if not isinstance(cur_item, dict) or not isinstance(prev_item, dict):
            continue
        if cur_item.get("type") != "table" or prev_item.get("type") != "table":
            continue

        ensure_table_defaults(prev_item)
        ensure_table_defaults(cur_item)

        _merge_str_field(prev_item, cur_item, "table_footnote", sep="\n")
        prev_rows = prev_item.get("rows")
        cur_rows = cur_item.get("rows")
        if not isinstance(prev_rows, list):
            prev_rows = []
            prev_item["rows"] = prev_rows
        if not isinstance(cur_rows, list):
            cur_rows = []

        # 按行合并并去除重叠前缀，保持行结构，避免把多页续表压扁到单行。
        merged_rows = _merge_rows_preserving_cells(prev_rows, deepcopy(cur_rows))
        if merged_rows:
            prev_item["rows"] = merged_rows
            merged_pairs += 1

        # 从 pdf_info 的上一页末尾 table 读取合并后的 html，回写到目标表并重算 rows/md。
        prev_page_obj = pdf_info[page_idx - 1]
        merged_html = ""
        if isinstance(prev_page_obj, dict):
            prev_para_blocks = prev_page_obj.get("para_blocks") or []
            if isinstance(prev_para_blocks, list) and prev_para_blocks:
                for blk in reversed(prev_para_blocks):
                    if isinstance(blk, dict) and blk.get("type") == "table":
                        merged_html = _safe_get_mineru_table_html(blk)
                        break

        if isinstance(merged_html, str) and merged_html.strip():
            prev_item["html"] = merged_html
            try:
                ensure_table_defaults(prev_item)
                html_parsed_rows = parse_html_table_rows(merged_html, source_page_no=prev_page_no)
                merged_rows_for_target = prev_item.get("rows")
                html_rows_with_source_page_no = _inject_source_page_no_into_parsed_rows(
                    html_parsed_rows,
                    merged_rows_for_target,
                    current_page_no=prev_page_no,
                )
                if html_parsed_rows:
                    candidate_rows = html_rows_with_source_page_no or html_parsed_rows
                    if isinstance(merged_rows_for_target, list) and merged_rows_for_target:
                        # 以 HTML 解析的行结构为准，但将 merged_rows 中的坐标字段回填到 candidate_rows
                        # 避免 HTML 重新解析导致 bBox/cellTraceabilitySource 等溯源坐标丢失
                        candidate_rows = _merge_coordinate_fields_into_parsed_rows(
                            merged_rows_for_target,
                            candidate_rows,
                        )
                    prev_item["rows"] = candidate_rows

                prev_item["md"] = ""
                _ensure_normalized_table_fields(prev_item)
            except Exception:
                _recompute_converted_table_fields(prev_item)

        # 当前页续表置空，稍后统一删除（倒序去重删除，避免索引位移）
        cur_item["rows"] = []
        cur_item["html"] = ""
        cur_item["md"] = ""
        removed_indices.append(cur_idx)

    if removed_indices:
        for idx in sorted(set(removed_indices), reverse=True):
            try:
                converted_items.pop(idx)
            except Exception:
                continue

    if merged_pairs or removed_indices:
        logger.info(
            f"跨页表合并(converted_items)：合并 {merged_pairs} 处，删除续表 {len(set(removed_indices))} 个"
        )

def apply_cross_page_table_merge_from_pdf_info(
    converted_items: List[Dict[str, Any]],
    pdf_info: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """基于 middle_json['pdf_info'] 的 para_blocks，对 converted json 中的跨页表进行合并。

    复用 mineru_parsers.merge_table 的判断/合并逻辑：
    - 若第 N 页首个 para_block 为 table 且被判定为续表，则合并进第 N-1 页末尾 table，并删除第 N 页对应 table。
    """
    if not converted_items or not isinstance(converted_items, list):
        return converted_items
    if not pdf_info or not isinstance(pdf_info, list):
        return converted_items

    try:
        from .mineru_parsers import merge_table as _merge_table_in_pdf_info
    except Exception as e:
        logger.warning(f"跨页表合并：导入 mineru_parsers.merge_table 失败，跳过。err={e}")
        return converted_items

    try:
        _merge_table_in_pdf_info(pdf_info)
    except Exception as e:
        logger.warning(f"跨页表合并：pdf_info merge_table 执行失败，跳过。err={e}")
        return converted_items

    # 打标统计：用于排查“续表未被识别”为 deleted 的情况
    deleted_first_table_pages = 0
    deleted_table_blocks = 0
    for page_obj in pdf_info:
        if not isinstance(page_obj, dict):
            continue
        para_blocks = page_obj.get("para_blocks") or []
        if not isinstance(para_blocks, list) or not para_blocks:
            continue
        first_block = None
        for blk_idx, blk in enumerate(para_blocks):
            if isinstance(blk, dict) and blk.get("type") == "table":
                if all(_mineru_block_is_ignorable_before_first_table(prev) for prev in para_blocks[:blk_idx]) or _mineru_page_has_continuation_signal_before_table(para_blocks, blk_idx):
                    first_block = blk
                break
        if isinstance(first_block, dict) and _mineru_table_block_is_deleted(first_block):
            deleted_first_table_pages += 1
        for blk in para_blocks:
            if isinstance(blk, dict) and blk.get("type") == "table" and _mineru_table_block_is_deleted(blk):
                deleted_table_blocks += 1
    logger.info(
        "跨页表合并：merge_table 打标统计 first_table_deleted_pages={} deleted_table_blocks={} total_pages={}".format(
            deleted_first_table_pages, deleted_table_blocks, len(pdf_info)
        )
    )
    try:
        _merge_table_in_converted_item(converted_items, pdf_info)
    except Exception as e:
        logger.warning(f"跨页表合并：converted_items merge_table according to pdf_info 执行失败，跳过。err={e}")
        return converted_items

    return converted_items

def _convert_mineru_payload_to_items(
    *,
    items: Optional[List[Dict[str, Any]]],
    model_items: List[Dict[str, Any]],
    pdf_info: Optional[List[Dict[str, Any]]],
    file_id: Any,
) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []

    if items:
        try:
            _t0 = time.perf_counter()
            data = convert_from_content_list(items, pdf_info=pdf_info)
            _dt = time.perf_counter() - _t0
            logger.info(f"内容转换完成，用时 {_dt:.2f}s，items={len(items)}")
            if model_items:
                _t1 = time.perf_counter()
                enrich_tables_with_model(data, model_items)
                _dt2 = time.perf_counter() - _t1
                logger.info(f"表格模型信息融合完成，用时 {_dt2:.2f}s，model_items={len(model_items)}")
        except Exception as e:
            import traceback
            logger.error(f"convert_from_content_list 失败: {e}")
            logger.error(f"异常堆栈:\n{traceback.format_exc()}")
            data = []

    if (not data) and model_items:
        try:
            _t0 = time.perf_counter()
            data = convert_from_model(model_items)
            _dt = time.perf_counter() - _t0
            logger.info(f"从模型输出构建 items 完成，用时 {_dt:.2f}s，model_items={len(model_items)}")
        except Exception:
            data = []

    if data and file_id is not None:
        data = attach_file_id(data, file_id)
    # 表头与标准二维结构检测：在异步服务路径中执行；此同步函数不主动触发，避免与事件循环冲突
    return data


async def convert_mineru_response_async(
    miner_u_data: Any, 
    file_id: Any = None,
    use_llm_heading: bool = False,
) -> List[Dict[str, Any]]:
    """异步将 MinerU 响应对象转换为统一 items 列表。"""
    data: List[Dict[str, Any]] = []
    if miner_u_data is None:
        return data
    if isinstance(miner_u_data, list):
        return miner_u_data
    if not isinstance(miner_u_data, dict):
        return data

    payload = _pick_payload_container(miner_u_data)

    raw_content_list = payload.get('content_list') if isinstance(payload, dict) else None
    raw_model_output = payload.get('model_output') if isinstance(payload, dict) else None
    raw_middle_json = payload.get('middle_json') if isinstance(payload, dict) else None

    items = _safe_parse_json_str(raw_content_list)
    model_items = _safe_parse_json_str(raw_model_output)
    middle_json = _safe_parse_json_str(raw_middle_json)

    items = _normalize_content_list_items(items)
    model_items = model_items if isinstance(model_items, list) else []

    pdf_info = None
    if isinstance(middle_json, dict):
        pdf_info = middle_json.get('pdf_info')
        if not isinstance(pdf_info, list):
            pdf_info = None

    if use_llm_heading and pdf_info:
        try:
            logger.info(f"开始 LLM 分级... (页数={len(pdf_info)})")
            _t0 = time.perf_counter()
            title_count_before = sum(
                1 for page in pdf_info
                for block in page.get('para_blocks', [])
                if block.get('type') == 'title'
            )
            logger.info(f" pdf_info 中找到 {title_count_before} 个标题块")

            await llm_aided_title(pdf_info)

            title_with_level = sum(
                1 for page in pdf_info
                for block in page.get('para_blocks', [])
                if block.get('type') == 'title' and 'level' in block
            )
            _dt = time.perf_counter() - _t0
            logger.info(f"LLM 分级完成，用时 {_dt:.2f}s，设置层级 {title_with_level}/{title_count_before}")
        except Exception as e:
            import traceback
            logger.error(f"LLM 分级失败: {e}")
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")

    return _convert_mineru_payload_to_items(
        items=items,
        model_items=model_items,
        pdf_info=pdf_info,
        file_id=file_id,
    )