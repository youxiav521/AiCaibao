"""推理缺失单元格坐标：有框直接用，空格从同列借 l,r、同行借 t,b。"""


def _flatten_cells(table):
    """按 row-major 顺序展开 cell，返回 (cell, rowstart, colstart)。"""
    out = []
    for ri, row in enumerate(table.get("rows", [])):
        if not isinstance(row, list):
            continue
        for ci, cell in enumerate(row):
            if isinstance(cell, dict):
                out.append((cell, int(cell.get("rowstart", ri) or ri), int(cell.get("colstart", ci) or ci)))
    return out


def _find_same_col_lr(grid, r, c, n_rows):
    """同列中最近有 bbox 的格子，取其 l,r。"""
    for rr in range(r - 1, -1, -1):
        if grid[rr][c] and len(grid[rr][c].get("coordinate", [])) >= 4:
            coord = grid[rr][c]["coordinate"]
            return coord[0], coord[2]
    for rr in range(r + 1, n_rows):
        if grid[rr][c] and len(grid[rr][c].get("coordinate", [])) >= 4:
            coord = grid[rr][c]["coordinate"]
            return coord[0], coord[2]
    return 0.0, 0.0


def _find_same_row_tb(grid, r, c, n_cols):
    """同行中最近有 bbox 的格子，取其 t,b。"""
    for cc in range(c - 1, -1, -1):
        if grid[r][cc] and len(grid[r][cc].get("coordinate", [])) >= 4:
            coord = grid[r][cc]["coordinate"]
            return coord[1], coord[3]
    for cc in range(c + 1, n_cols):
        if grid[r][cc] and len(grid[r][cc].get("coordinate", [])) >= 4:
            coord = grid[r][cc]["coordinate"]
            return coord[1], coord[3]
    return 0.0, 0.0


def infer_missing_boxes(grid, table, gap_x=None, gap_y=None):
    """
    有框直接用；空格从同列找 l,r、同行找 t,b 填入。
    grid 来自 utils.infer_grid_from_boxes（y1/x1 分组），grid[r][c]=box 或 None。
    """
    cells = _flatten_cells(table)
    if not cells or not grid:
        return []

    n_rows, n_cols = len(grid), len(grid[0]) if grid else 0
    if n_rows <= 0 or n_cols <= 0:
        return []

    out = []
    for cell, rs, cs in cells:
        rs = max(0, min(n_rows - 1, rs))
        cs = max(0, min(n_cols - 1, cs))
        b = grid[rs][cs]
        if b and len(b.get("coordinate", [])) >= 4:
            out.append(b)
        else:
            l, r = _find_same_col_lr(grid, rs, cs, n_rows)
            t, b = _find_same_row_tb(grid, rs, cs, n_cols)
            out.append({
                "coordinate": [l, t, r, b],
                "label": "cell", "cls_id": 0, "score": 0.0,
            })
    return out
