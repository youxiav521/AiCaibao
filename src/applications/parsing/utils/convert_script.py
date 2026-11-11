import os
import json
import argparse
from html import escape
from typing import Any, Dict, List, Optional


def _safe_get_first_bbox_and_page(prov_list: Optional[List[Dict[str, Any]]]) -> (Optional[int], Optional[Dict[str, Any]]):
    """
    从 `prov_list` 中安全提取第一个元素的页码与边界框信息。

    参数:
        prov_list: Docling 溯源列表，元素通常包含 "page_no" 与 "bbox" 字段。

    返回:
        (page_no, bbox): 若列表为空或字段缺失，返回 (None, None)。
    """
    if not prov_list:
        return None, None
    first = prov_list[0] or {}
    return first.get("page_no"), first.get("bbox")


def _build_table_rows_from_grid(grid: List[List[Dict[str, Any]]]) -> List[List[str]]:
    """
    将表格网格数据转换为二维字符串行列表。

    参数:
        grid: 二维网格，每个单元为字典，包含 "text" 等字段。

    返回:
        rows: 二维字符串数组，仅保留单元格的文本值。
    """
    rows: List[List[str]] = []
    for row_cells in grid:
        row_values: List[str] = []
        for cell in row_cells:
            if not isinstance(cell, dict):
                row_values.append("")
                continue
            row_values.append(str(cell.get("text", "") or ""))
        rows.append(row_values)
    return rows


def _build_cells_from_grid(grid: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """
    将表格网格数据规范化为二维单元格字典数组（直接映射，不做推断计算）。

    参数:
        grid: 二维网格，每个单元为字典，包含 "text"、"bbox"、"row_span"、"col_span" 等。

    返回:
        cells_rows: 二维数组。每个单元包含: text/row_span/col_span/bBox。
    """
    cells_rows: List[List[Dict[str, Any]]] = []
    for row_cells in grid:
        row_list: List[Dict[str, Any]] = []
        for cell in row_cells:
            if not isinstance(cell, dict):
                row_list.append({"text": "", "row_span": 1, "col_span": 1, "bBox": None})
                continue
            text_value = str(cell.get("text", "") or "")
            # 优先使用单元格自身的 bbox；若缺失再回退到 prov 中的 bbox
            bbox = cell.get("bbox") or _safe_get_first_bbox_and_page(cell.get("prov"))[1]
            row_span = int(cell.get("row_span", 1) or 1)
            col_span = int(cell.get("col_span", 1) or 1)
            row_list.append({
                "text": text_value,
                "row_span": row_span,
                "col_span": col_span,
                "bBox": bbox or None,
            })
        cells_rows.append(row_list)
    return cells_rows


def _build_table_rows_from_cells(cells: List[Dict[str, Any]], num_rows: int, num_cols: int) -> List[List[str]]:
    """
    根据带有定位信息的单元格列表还原出二维文本网格。

    参数:
        cells: 单元格列表，包含起始行列偏移与 "text" 字段（如 start_row_offset_idx/start_col_offset_idx）。
        num_rows: 预期的行数。
        num_cols: 预期的列数。

    返回:
        grid: 二维字符串数组，在指定起始位置填入文本，其余为空字符串。
    """
    grid: List[List[str]] = [["" for _ in range(max(num_cols, 0))] for _ in range(max(num_rows, 0))]
    for cell in cells:
        start_row = int(cell.get("start_row_offset_idx", 0) or 0)
        start_col = int(cell.get("start_col_offset_idx", 0) or 0)
        text_value = str(cell.get("text", "") or "")
        if 0 <= start_row < len(grid) and 0 <= start_col < len(grid[start_row]):
            grid[start_row][start_col] = text_value
    return grid


def _build_cells_from_table_cells(cells: List[Dict[str, Any]], num_rows: int, num_cols: int) -> List[List[Dict[str, Any]]]:
    """
    根据 table_cells 构建二维单元格字典网格（直接映射，不做推断计算）。

    参数:
        cells: 单元格列表，包含文本、起始位置、跨行跨列、bbox/溯源等信息。
        num_rows: 预期的行数。
        num_cols: 预期的列数。

    返回:
        grid: 二维数组，每个位置为包含 text/row_span/col_span/bBox 的字典。
    """
    grid: List[List[Dict[str, Any]]] = [[{"text": "", "row_span": 1, "col_span": 1, "bBox": None} for _ in range(max(num_cols, 0))] for _ in range(max(num_rows, 0))]
    for cell in cells:
        try:
            start_row = int(cell.get("start_row_offset_idx", 0) or 0)
            start_col = int(cell.get("start_col_offset_idx", 0) or 0)
            text_value = str(cell.get("text", "") or "")
            # 优先单元格自身 bbox，回退到 prov 中的 bbox
            bbox = cell.get("bbox") or _safe_get_first_bbox_and_page(cell.get("prov"))[1]
            row_span = int(cell.get("row_span", 1) or 1)
            col_span = int(cell.get("col_span", 1) or 1)
            entry = {
                "text": text_value,
                "row_span": row_span,
                "col_span": col_span,
                "bBox": bbox or None,
            }
            if 0 <= start_row < len(grid) and 0 <= start_col < len(grid[start_row]):
                grid[start_row][start_col] = entry
        except Exception:
            continue
    return grid


def _render_markdown_table(rows: List[List[str]]) -> str:
    """
    将二维文本行数据渲染为简易 Markdown 表格。

    规则:
        - 若无行，返回空字符串。
        - 若仅一行，则将其视为数据行（无表头），直接用竖线拼接。
        - 若两行及以上，首行视作表头并生成分隔行，其余为数据行。
    """
    if not rows:
        return ""
    def _join_line(line: List[str]) -> str:
        return "| " + " | ".join((str(v) if v is not None else "") for v in line) + " |"
    if len(rows) == 1:
        return _join_line(rows[0])
    header = _join_line(rows[0])
    sep = "| " + " | ".join(["---" for _ in rows[0]]) + " |"
    body_lines = [_join_line(r) for r in rows[1:]]
    return "\n".join([header, sep, *body_lines])


def _build_md_for_text(label: str, text_value: str, formatting: Optional[Dict[str, Any]] = None, marker: Optional[str] = None) -> str:
    """
    按 label 类型“手动拼接”文本项的 md 字段。

    约定映射（尽量保守，未识别类型回退为原文）：
        - heading_1/2/3/... 或 title/heading/section_header/header -> 井号标题
        - list_item -> 列表项（使用 "- ")
        - quote -> 引用（"> ")
        - code_block/code -> 代码块（``` 包裹）
        - paragraph/inline/unspecified/其他 -> 原文
    """
    value = str(text_value or "")
    if not label:
        return value
    label_lower = str(label).lower()

    # heading_N
    if label_lower.startswith("heading_"):
        try:
            level = int(label_lower.split("_", 1)[1])
            level = max(1, min(level, 6))
        except Exception:
            level = 2
        return f"{'#' * level} {value}".strip()

    # generic headings
    if label_lower in {"title", "heading", "section_header", "header"}:
        return f"## {value}".strip()

    # list item
    if label_lower == "list_item":
        prefix = "- "
        # 如果 docling 提供了 marker，尽量保留
        if marker:
            cleaned_marker = str(marker).strip()
            if cleaned_marker:
                prefix = f"{cleaned_marker} ".strip() + " "
        return f"{prefix}{value}".strip()

    # quote
    if label_lower in {"quote", "blockquote"}:
        return f"> {value}".strip()

    # code
    if label_lower in {"code", "code_block"}:
        return f"```\n{value}\n```"

    # 默认：段落/内联/未指定
    return value


def _render_html_table(rows: List[List[str]]) -> str:
    """
    将二维文本行数据渲染为简单的 HTML 表格字符串。

    参数:
        rows: 二维字符串数组。

    返回:
        html: 由 <table>/<tr>/<td> 组成的 HTML 片段，已对文本进行转义。
    """
    html_lines: List[str] = ["<table>"]
    for row in rows:
        html_lines.append("  <tr>")
        for value in row:
            html_lines.append(f"    <td>{escape(value)}</td>")
        html_lines.append("  </tr>")
    html_lines.append("</table>")
    return "".join(html_lines)


def _is_perfect_table(data: Dict[str, Any], rows: List[List[str]]) -> bool:
    """
    判断表格是否“完美”：
    - 每行列数一致且与 num_rows/num_cols 一致（若提供）。
    - 不存在跨行/跨列单元格。
    - 不存在空文本单元格。

    参数:
        data: 表格结构元数据，包含 grid/table_cells/num_rows/num_cols 等。
        rows: 由表格数据生成的二维文本数组。

    返回:
        bool: 若满足上述条件则为 True，否则为 False。
    """
    if not rows:
        return False
    expected_cols = len(rows[0])
    if expected_cols == 0:
        return False
    for r in rows:
        if len(r) != expected_cols:
            return False
    num_rows = int((data.get("num_rows") or 0))
    num_cols = int((data.get("num_cols") or 0))
    if num_rows and num_rows != len(rows):
        return False
    if num_cols and num_cols != expected_cols:
        return False
    any_span = False
    for source in (data.get("grid") or []):
        for cell in source:
            if isinstance(cell, dict) and (int(cell.get("row_span", 1)) > 1 or int(cell.get("col_span", 1)) > 1):
                any_span = True
                break
        if any_span:
            break
    if not any_span and data.get("table_cells"):
        for cell in data.get("table_cells"):
            if int(cell.get("row_span", 1)) > 1 or int(cell.get("col_span", 1)) > 1:
                any_span = True
                break
    if any_span:
        return False
    for r in rows:
        for v in r:
            if v is None or (isinstance(v, str) and v.strip() == ""):
                return False
    return True


def convert_docling_json(input_path: str, output_path: str) -> None:
    """
    将 Docling 解析输出 JSON 转换为统一的 items 列表格式，并写入文件。

    转换规则:
        - 文本: 输出包含 page/type/value/md/bBox/prov 的文本项。
        - 表格: 解析 grid/table_cells，生成 rows/html/isPerfectTable/bBox/cells 等字段。

    参数:
        input_path: 输入的 Docling JSON 文件路径。
        output_path: 输出的目标 JSON 文件路径。
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {input_path}: {e}")
        return

    items: List[Dict[str, Any]] = []

    texts = data.get("texts") or []
    for text_obj in texts:
        prov_list = text_obj.get("prov") or []
        page_no, bbox = _safe_get_first_bbox_and_page(prov_list)
        value_text = text_obj.get("text") or text_obj.get("orig") or ""
        label_value = text_obj.get("label") or "text"
        # 过滤页眉页脚
        if str(label_value).lower() in {"page_header", "page_footer"}:
            continue
        # 过滤合并后的表格片段
        if str(label_value).lower() == "table_fragment_merged":
            continue
        md_value = _build_md_for_text(label_value, value_text, text_obj.get("formatting"), text_obj.get("marker"))
        item: Dict[str, Any] = {
            "page": int(page_no) if page_no is not None else None,
            "type": label_value,
            "value": value_text,
            "md": md_value,
            "bBox": bbox or None,
        }
        items.append(item)

    tables = data.get("tables") or []
    for table_obj in tables:
        prov_list = table_obj.get("prov") or []
        page_no, bbox = _safe_get_first_bbox_and_page(prov_list)
        label_value = table_obj.get("label") or "table"
        # 过滤页眉页脚
        if str(label_value).lower() in {"page_header", "page_footer"}:
            continue
        # 过滤合并后的表格片段
        if str(label_value).lower() == "table_fragment_merged":
            continue
        data_block = table_obj.get("data") or {}
        rows_text: List[List[str]]
        cells_grid: List[List[Dict[str, Any]]] = []
        if data_block.get("grid"):
            rows_text = _build_table_rows_from_grid(data_block.get("grid"))
            try:
                cells_grid = _build_cells_from_grid(data_block.get("grid"))
            except Exception:
                cells_grid = []
        elif data_block.get("table_cells"):
            rows_text = _build_table_rows_from_cells(
                data_block.get("table_cells") or [],
                int(data_block.get("num_rows") or 0),
                int(data_block.get("num_cols") or 0),
            )
            try:
                cells_grid = _build_cells_from_table_cells(
                    data_block.get("table_cells") or [],
                    int(data_block.get("num_rows") or 0),
                    int(data_block.get("num_cols") or 0),
                )
            except Exception:
                cells_grid = []
        else:
            rows_text = []
            cells_grid = []

        html_table = _render_html_table(rows_text) if rows_text else "<table></table>"
        perfect = _is_perfect_table(data_block, rows_text) if rows_text else False
        md_table = _render_markdown_table(rows_text) if rows_text else ""

        item = {
            "page": int(page_no) if page_no is not None else None,
            "type": label_value,
            "rows": cells_grid,
            "html": html_table,
            "isPerfectTable": bool(perfect),
            "bBox": bbox or None,
            "md": md_table,
        }
        items.append(item)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Successfully converted file and saved to {output_path}")


    # ------------------------------------------------------------
    # 结果层重排（不改 Docling）：只改 data.body.children 的顺序
    # ------------------------------------------------------------
def reorder_body_children(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """对导出的 Docling dict 执行阅读顺序重排（按页内阅读顺序）：
    规则：
    - 先按 prov.page_no（1-based）分组。
    - 每页内根据坐标原点进行排序：
      - coord_origin == TOPLEFT: 按 t 升序，再按 l 升序。
      - coord_origin == BOTTOMLEFT: 按 t 降序（实现为 -t 升序），再按 l 升序。
      - 若缺失 coord_origin，则按 TOPLEFT 处理。
    - 特例：若 TABLE 紧挨着其上方最近的 SECTION_HEADER（垂直间距 < 阈值），
      则将该 TABLE 放在该 SECTION_HEADER 之后（使用标准化后的 top 进行比较）。
    - PAGE_HEADER 固定在页首；PAGE_FOOTER 固定在页末。
    """
    try:
        body = data.get("body") or {}
        children = list(body.get("children") or [])
        if not children:
            return data
    except Exception:
        return data
    def _resolve_ref(root: Dict[str, Any], ref: str) -> Optional[Dict[str, Any]]:
        try:
            if not isinstance(ref, str) or not ref.startswith("#/"):
                return None
            node: Any = root
            for key in ref[2:].split("/"):
                if isinstance(node, dict):
                    node = node.get(key)
                elif isinstance(node, list):
                    idx = int(key)
                    node = node[idx] if 0 <= idx < len(node) else None
                else:
                    return None
                if node is None:
                    return None
            if isinstance(node, dict):
                return node
            return None
        except Exception:
            return None
    def _normalize_label(label: Any) -> str:
        try:
            return str(label or "").strip().upper().replace(" ", "_")
        except Exception:
            return ""
    def _extract_coord_and_bbox(obj: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """优先从 prov[0].bbox 取坐标与原点；若没有则回退到 obj.bbox。"""
        # 1) 从 prov[0].bbox 读取
        try:
            prov = obj.get("prov") or []
            if isinstance(prov, list) and prov:
                p0 = prov[0]
                if isinstance(p0, dict):
                    pb = p0.get("bbox")
                    if isinstance(pb, dict):
                        origin = str(pb.get("coord_origin") or "TOPLEFT").upper()
                        return origin, pb
        except Exception:
            pass
        # 2) 回退到对象级 bbox
        try:
            obb = obj.get("bbox") or obj.get("bounds") or obj.get("box")
            if isinstance(obb, dict):
                origin = str(obb.get("coord_origin") or "TOPLEFT").upper()
                return origin, obb
        except Exception:
            pass
        return None, None
    def _standardized_top_left(obj: Dict[str, Any]) -> tuple[float, float, str]:
        """返回用于排序/比较的标准化 top 与 left 以及确定的 coord_origin。
        - 若 coord_origin == TOPLEFT: top_key = t（或 top/y/y0）
        - 若 coord_origin == BOTTOMLEFT: top_key = -t（或 -top/-y/-y0）
        - left 一律取 l/left/x/x0 的数值（不翻转）
        - 若无法解析，返回 (inf, inf, "UNKNOWN")
        """
        origin, bbox = _extract_coord_and_bbox(obj)
        if not isinstance(bbox, dict):
            return float("inf"), float("inf"), str(origin or "UNKNOWN")
        # 支持多种键名
        def _pick(keys: list[str]) -> Optional[float]:
            for k in keys:
                if k in bbox:
                    try:
                        v = float(bbox.get(k))
                        return v
                    except Exception:
                        continue
            return None
        # y-top 与 x-left 的候选键
        t_val = _pick(["t", "top", "y", "y0"])  # 优先使用 t/top，没有则 y/y0
        l_val = _pick(["l", "left", "x", "x0"])  # 优先使用 l/left，没有则 x/x0
        if t_val is None or l_val is None:
            return float("inf"), float("inf"), str(origin or "UNKNOWN")
        origin_norm = str((origin or "TOPLEFT")).upper()
        if origin_norm == "BOTTOMLEFT":
            top_key = -t_val
        else:
            # 默认 TOPLEFT
            top_key = t_val
        return top_key, l_val, origin_norm
    # 收集 children 的度量
    collected: list[Dict[str, Any]] = []
    for idx, child in enumerate(children):
        try:
            ref = child.get("$ref") if isinstance(child, dict) else None
            if not ref:
                continue
            obj = _resolve_ref(data, ref)
            if not obj:
                continue
            label = _normalize_label(obj.get("label"))
            page_no = self._get_first_page_from_prov(obj.get("prov") or [])
            if page_no is None:
                page_no = 10**9  # 无页信息的放到最后，保持稳定
            top_key, left, origin = _standardized_top_left(obj)
            collected.append(
                {
                    "ref": ref,
                    "obj": obj,
                    "label": label,
                    "page": int(page_no),
                    "top": float(top_key),
                    "left": float(left),
                    "coord_origin": origin,
                    "orig_idx": idx,
                }
            )
        except Exception:
            continue
    if not collected:
        return data
    # 分页
    page_to_items: Dict[int, list[Dict[str, Any]]] = {}
    for item in collected:
        page_to_items.setdefault(item["page"], []).append(item)
    # 排序阈值（像素/点）
    attach_threshold = 30.0
    def _sort_key(x: Dict[str, Any]):
        return (x["top"], x["left"], x["orig_idx"])
    new_children: list[Dict[str, Any]] = []
    for page in sorted(page_to_items.keys()):
        items = page_to_items[page]
        # 页眉/页脚分离
        headers = [i for i in items if i["label"] == "PAGE_HEADER"]
        footers = [i for i in items if i["label"] == "PAGE_FOOTER"]
        middle = [i for i in items if i not in headers and i not in footers]
        headers.sort(key=_sort_key)
        footers.sort(key=_sort_key)
        middle.sort(key=_sort_key)
        # 特例：表格靠近上方 SECTION_HEADER 时，插入到其后
        ordered_middle: list[Dict[str, Any]] = []
        last_section_idx: Optional[int] = None
        for rec in middle:
            if rec["label"] == "SECTION_HEADER":
                ordered_middle.append(rec)
                last_section_idx = len(ordered_middle) - 1
                continue
            if rec["label"] == "TABLE" and last_section_idx is not None:
                section_rec = ordered_middle[last_section_idx]
                if (rec["top"] - section_rec["top"]) <= attach_threshold:
                    insert_at = last_section_idx + 1
                    ordered_middle.insert(insert_at, rec)
                    last_section_idx = last_section_idx  # 保持不变
                    continue
            ordered_middle.append(rec)
        # 合并为本页顺序：页眉 → 中间 → 页脚
        page_refs = [
            {"$ref": r["ref"]} for r in (headers + ordered_middle + footers)
        ]
        new_children.extend(page_refs)
    # 回写
    data.setdefault("body", {})["children"] = new_children
    return data






# 测试用
def main() -> None:
    """
    命令行入口：解析参数并调用转换函数。

    行为:
        - 允许通过 --input/--output 指定输入输出路径。
        - 未指定时，使用项目下 storage 目录的默认示例路径。
    """
    parser = argparse.ArgumentParser(description="Convert Docling JSON to flattened items format")
    parser.add_argument("--input", dest="input_path", type=str, default=None, help="Path to Docling JSON")
    parser.add_argument("--output", dest="output_path", type=str, default=None, help="Path to save converted JSON")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_input = os.path.join(project_root, "storage", "docling_output", "4-企业征信报告_docling.json")
    default_output_dir = os.path.join(project_root, "storage", "converted_output")
    os.makedirs(default_output_dir, exist_ok=True)
    default_output = os.path.join(default_output_dir, "4-企业征信报告_converted.json")

    input_path = args.input_path or default_input
    output_path = args.output_path or default_output
    convert_docling_json(input_path, output_path)





if __name__ == "__main__":
    main()