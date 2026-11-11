"""公共工具：过滤排序 boxes"""


def filter_and_sort_boxes(boxes, row_tol=8.0):
    """过滤有效 coordinate，按行分组(y1 差<=row_tol)，行内按 x1 排序，从上到下从左到右"""
    filtered = [b for b in boxes if len(b.get("coordinate", [])) >= 4]
    if not filtered:
        return []
    filtered.sort(key=lambda b: (b["coordinate"][1], b["coordinate"][0]))
    rows, cur, anchor_y = [], [], None
    for b in filtered:
        y1 = b["coordinate"][1]
        if anchor_y is None or abs(y1 - anchor_y) <= row_tol:
            cur.append(b)
        else:
            if cur:
                rows.append(sorted(cur, key=lambda b: b["coordinate"][0]))
            cur = [b]
            anchor_y = y1  # 新行以第一个 box 的 y1 为锚点，避免链式漂移
        if anchor_y is None:
            anchor_y = y1
    if cur:
        rows.append(sorted(cur, key=lambda b: b["coordinate"][0]))
    return [b for row in rows for b in row]


# def table_sig(table):
#     """(行数, 单元格数)"""
#     rows = table.get("rows", [])
#     return len(rows), sum(len(r) for r in rows)


def table_grid_shape(table):
    """从 table.rows 的 rowstart/colstart 得到 (n_rows, n_cols)。"""
    n_rows, n_cols = 0, 0
    for row in table.get("rows", []):
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, dict):
                r = int(cell.get("rowstart") or 0)
                c = int(cell.get("colstart") or 0)
                n_rows = max(n_rows, r + 1)
                n_cols = max(n_cols, c + 1)
    return n_rows, n_cols


def _group_by_tol(values, tol):
    """按容差分组，返回每组代表值（中位）及组边界。values 已排序。"""
    if not values:
        return [], []
    groups, cur = [], [values[0]]
    for v in values[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    if cur:
        groups.append(cur)
    reps = [sorted(g)[len(g) // 2] for g in groups]
    return groups, reps


def infer_grid_from_boxes(boxes, row_tol=8.0, col_tol=5.0):
    from statistics import median, mode as _mode
    from collections import Counter

    filtered = [b for b in boxes if len(b.get("coordinate", [])) >= 4]
    if not filtered:
        return 0, 0, []

    # ── Step 1: 按 y1 分行 ──
    y1_sorted = sorted(b["coordinate"][1] for b in filtered)
    _, row_reps = _group_by_tol(y1_sorted, row_tol)
    n_rows = len(row_reps)
    if n_rows == 0:
        return 0, 0, []

    # 把 box 分配到行
    row_buckets: list[list] = [[] for _ in range(n_rows)]
    for b in filtered:
        y1 = b["coordinate"][1]
        r = min(range(n_rows), key=lambda i: abs(y1 - row_reps[i]))
        row_buckets[r].append(b)

    # ── Step 2: 每行内部按 center_x 聚类 ──
    def _center_x(b):
        return (b["coordinate"][0] + b["coordinate"][2]) / 2.0

    # 自适应列容差：取所有框宽度中位数的 30%，至少 col_tol
    widths = [b["coordinate"][2] - b["coordinate"][0] for b in filtered]
    adaptive_col_tol = max(col_tol, median(widths) * 0.30) if widths else col_tol

    row_col_counts: list[int] = []
    row_cx_groups: list[list[list[float]]] = []  # row_cx_groups[r] = [[cx1,cx2,...], [cx3,...], ...]
    for bucket in row_buckets:
        if not bucket:
            row_col_counts.append(0)
            row_cx_groups.append([])
            continue
        cxs = sorted(_center_x(b) for b in bucket)
        groups, reps = _group_by_tol(cxs, adaptive_col_tol)
        row_col_counts.append(len(reps))
        row_cx_groups.append(groups)

    # ── Step 3: 投票确定全局列数 ──
    nonzero_counts = [c for c in row_col_counts if c > 0]
    if not nonzero_counts:
        return 0, 0, []

    count_freq = Counter(nonzero_counts)
    global_n_cols = count_freq.most_common(1)[0][0]

    # 收集所有"列数==全局列数"的行的 center_x，合并出全局列模板
    all_col_cxs: list[list[float]] = [[] for _ in range(global_n_cols)]
    for r, bucket in enumerate(row_buckets):
        if row_col_counts[r] != global_n_cols:
            continue
        cxs_sorted = sorted(_center_x(b) for b in bucket)
        groups, _ = _group_by_tol(cxs_sorted, adaptive_col_tol)
        if len(groups) != global_n_cols:
            continue
        for ci, grp in enumerate(groups):
            all_col_cxs[ci].extend(grp)

    if not any(all_col_cxs):
        # fallback: 用全部 center_x 做全局聚类
        all_cx = sorted(_center_x(b) for b in filtered)
        _, col_reps = _group_by_tol(all_cx, adaptive_col_tol)
        global_n_cols = len(col_reps)
    else:
        col_reps = [median(sorted(cxs)) if cxs else 0.0 for cxs in all_col_cxs]

    n_cols = global_n_cols
    if n_cols == 0:
        return 0, 0, []

    # ── Step 4: 分配 box 到 (row, col) ──
    grid = [[None] * n_cols for _ in range(n_rows)]
    for b in filtered:
        y1 = b["coordinate"][1]
        cx = _center_x(b)
        r = min(range(n_rows), key=lambda i: abs(y1 - row_reps[i]))
        c = min(range(n_cols), key=lambda i: abs(cx - col_reps[i]))
        # 如果格子已被占用，保留 score 更高的
        if grid[r][c] is not None:
            old_score = grid[r][c].get("score", 0) or 0
            new_score = b.get("score", 0) or 0
            if new_score <= old_score:
                continue
        grid[r][c] = b

    return n_rows, n_cols, grid


def coord_to_bbox(coord):
    # """[x1,y1,x2,y2] -> {l,t,r,b,coordOrigin}"""
    if len(coord) < 4:
        return {}
    return {"l": round(coord[0], 2), "t": round(coord[1], 2), "r": round(coord[2], 2), "b": round(coord[3], 2), "coordOrigin": "TOPLEFT"}
