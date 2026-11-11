"""根据表格总坐标（bBox）调整每个单元格的坐标，使单元格坐标落在表格 bBox 范围内。

步骤：
1. 计算比例：找出整个 rows 中 r 和 b 的最大值，每个单元格的 l/r 占 r_max 的比例、t/b 占 b_max 的比例。
2. 变换：每个单元格的坐标按表格 bBox 线性映射，
   l_new = table.l + (table.r - table.l) * (cell.l / r_max)，r/t/b 同理。
"""


def _get_bbox_cell(cell):
    """返回单元格的 bBox 字典，若无则返回 None。"""
    if not isinstance(cell, dict):
        return None
    return cell.get("bBox")


def _get_bbox_table(table):
    """返回表格的 bBox 字典，需包含 l/t/r/b。"""
    bbox = table.get("bBox")
    if not bbox or not isinstance(bbox, dict):
        return None
    for k in ("l", "t", "r", "b"):
        if k not in bbox:
            return None
    return bbox


def adjust_table_cell_coords(table):
    """
    对单个 table 元素，根据 table.bBox 调整 table.rows 中每个单元格的 bBox。

    - 先求所有单元格中 r 的最大值 r_max、b 的最大值 b_max；
    - 再对每个单元格：用 l/r_max, r/r_max 作为横向比例，t/b_max, b/b_max 作为纵向比例；
    - 最后：cell.l = table.l + (table.r - table.l) * (cell.l / r_max)，r/t/b 同理。
    """
    rows = table.get("rows", [])
    if not rows:
        return

    table_bbox = _get_bbox_table(table)
    if not table_bbox:
        return

    tbl_l = float(table_bbox["l"])
    tbl_t = float(table_bbox["t"])
    tbl_r = float(table_bbox["r"])
    tbl_b = float(table_bbox["b"])

    # 收集所有单元格 bBox，并求 r_max、b_max
    all_cells = []
    for row in rows:
        if not isinstance(row, list):
            continue
        for cell in row:
            bbox = _get_bbox_cell(cell)
            if bbox is not None:
                all_cells.append((cell, bbox))

    if not all_cells:
        return

    r_max = max(box.get("r", 0) for _, box in all_cells)
    b_max = max(box.get("b", 0) for _, box in all_cells)
    if r_max <= 0 or b_max <= 0:
        return

    tbl_w = tbl_r - tbl_l
    tbl_h = tbl_b - tbl_t

    for cell, bbox in all_cells:
        l_ = float(bbox.get("l", 0))
        t_ = float(bbox.get("t", 0))
        r_ = float(bbox.get("r", 0))
        b_ = float(bbox.get("b", 0))

        l_ratio = l_ / r_max
        r_ratio = r_ / r_max
        t_ratio = t_ / b_max
        b_ratio = b_ / b_max

        new_l = tbl_l + tbl_w * l_ratio
        new_r = tbl_l + tbl_w * r_ratio
        new_t = tbl_t + tbl_h * t_ratio
        new_b = tbl_t + tbl_h * b_ratio

        cell["bBox"] = {
            "l": round(new_l, 2),
            "t": round(new_t, 2),
            "r": round(new_r, 2),
            "b": round(new_b, 2),
        }


def execute(items):
    """
    对列表中的每个 type=table 元素执行单元格坐标调整（就地修改）。

    - items: 与 pipeline 中一致的列表，可含 text、table 等元素
    """
    if not items:
        return items
    if not isinstance(items, list):
        items = [items]
    for el in items:
        if isinstance(el, dict) and el.get("type") == "table":
            adjust_table_cell_coords(el)
    return items
