from __future__ import annotations

import copy
import re
from typing import Any, Optional

from src.applications.parsing.service.three_statements.cells import (
    get_raw_cell_col_span,
    get_raw_cell_col_start,
    normalize_raw_rows,
)
from src.applications.parsing.service.three_statements.headers import (
    STATEMENT_AUXILIARY_HEADER_TOKENS,
    STATEMENT_NOTE_PATTERNS,
    identify_auxiliary_column_indices,
)


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_PURE_AMOUNT_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _looks_like_pure_numeric_text(text: str) -> bool:
    """判断单元格文本是否"纯数字风格金额"（含千分位、括号负号、占位符）。"""
    raw = (text or "").strip()
    if not raw:
        return True  # 空白视为可数字列的占位
    if raw in ("-", "--", "---", "—", "——", "———"):
        return True
    normalized = (
        raw.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("(", "-")
        .replace(")", "")
        .replace("（", "-")
        .replace("）", "")
    )
    return bool(_PURE_AMOUNT_RE.fullmatch(normalized))


def _identify_blank_header_columns(
    header_cells: list[list[Any]],
    total_cols: int,
    existing_aux: set[int],
    data_rows: list[list[Any]],
) -> set[int]:
    """识别表头空白的非首列：

    - 表头空白（``header_cells`` 多行展开后该列所有 cell 文本均为空）；
    - 首列（科目列）必保留；
    - 该列数据行不全为纯数字时才删除（避免误删合法金额列）。
    """
    extra_aux: set[int] = set()
    if total_cols <= 0:
        return extra_aux
    # 按 colStart/colSpan 把 header_cells 展开成 col -> [文本片段]
    col_texts: dict[int, list[str]] = {c: [] for c in range(total_cols)}
    for row in header_cells or []:
        if not isinstance(row, list):
            continue
        cursor = 0
        for cell in row:
            if not isinstance(cell, dict):
                cursor += 1
                continue
            cstart = cell.get("colStart")
            if cstart is None:
                cstart = cell.get("colstart")
            if isinstance(cstart, int):
                cursor = cstart
            cspan = max(_safe_int(cell.get("colspan", cell.get("colSpan", 1)), 1), 1)
            text = (cell.get("text") or "").strip()
            for offset in range(cspan):
                idx = cursor + offset
                if idx < total_cols and text:
                    col_texts[idx].append(text)
            cursor += cspan
    for col_idx in range(1, total_cols):
        if col_idx in existing_aux:
            continue
        if col_texts[col_idx]:
            continue
        # 表头空白：再看数据行是否全是数字风格
        all_numeric = True
        any_data = False
        for row in data_rows:
            if not isinstance(row, list):
                continue
            data_cell = None
            cursor = 0
            for c in row:
                col_span = max(_safe_int(c.get("colspan", 1), 1), 1) if isinstance(c, dict) else 1
                if cursor <= col_idx < cursor + col_span:
                    data_cell = c
                    break
                cursor += col_span
            data_text = ""
            if isinstance(data_cell, dict):
                data_text = (data_cell.get("text") or "").strip()
            elif isinstance(data_cell, str):
                data_text = data_cell.strip()
            if not data_text:
                continue
            any_data = True
            if not _looks_like_pure_numeric_text(data_text):
                all_numeric = False
                break
        # 数据行存在非数字内容 → 视为辅助列（脚注引用、行号等）
        if any_data and not all_numeric:
            extra_aux.add(col_idx)
        elif not any_data:
            # 数据全空：表头也空 → 删除（无价值列）
            extra_aux.add(col_idx)
    return extra_aux


def _filter_header_cells_by_kept_indices(
    header_cells: list[list[Any]],
    kept_col_indices: list[int],
    total_cols: int,
) -> list[list[Any]]:
    """按列索引过滤 ``headerCells`` 多行表头，保留 kept_col_indices 中的列，并重排 colStart/colSpan。

    展开 colSpan 时仅保留落在 kept 中的列段；连续命中相同 cell 的列段合并为新 colSpan。
    """
    if not header_cells or not kept_col_indices:
        return []
    new_idx_by_old = {old: new for new, old in enumerate(kept_col_indices)}
    out: list[list[Any]] = []
    for row in header_cells:
        if not isinstance(row, list):
            continue
        # 展开为 col_cursor -> source cell
        cell_per_col: dict[int, dict[str, Any]] = {}
        col_cursor = 0
        for cell in row:
            if not isinstance(cell, dict):
                col_cursor += 1
                continue
            cstart = cell.get("colStart")
            if cstart is None:
                cstart = cell.get("colstart")
            if isinstance(cstart, int):
                col_cursor = cstart
            cspan = max(_safe_int(cell.get("colspan", cell.get("colSpan", 1)), 1), 1)
            for offset in range(cspan):
                if col_cursor + offset < total_cols:
                    cell_per_col[col_cursor + offset] = cell
            col_cursor += cspan

        new_row: list[dict[str, Any]] = []
        run_cell: Optional[dict[str, Any]] = None
        run_start = 0
        run_count = 0

        def _flush() -> None:
            nonlocal run_cell, run_start, run_count
            if run_cell is None or run_count <= 0:
                return
            new_cell = copy.deepcopy(run_cell)
            new_cell["colStart"] = run_start
            new_cell["colSpan"] = run_count
            new_cell["colstart"] = run_start
            new_cell["colspan"] = run_count
            new_row.append(new_cell)
            run_cell = None
            run_count = 0

        for new_idx, old_idx in enumerate(kept_col_indices):
            source = cell_per_col.get(old_idx)
            if source is None:
                _flush()
                new_row.append({
                    "text": "",
                    "colStart": new_idx,
                    "colSpan": 1,
                    "colstart": new_idx,
                    "colspan": 1,
                })
                continue
            if run_cell is None:
                run_cell = source
                run_start = new_idx
                run_count = 1
                continue
            if source is run_cell:
                run_count += 1
                continue
            _flush()
            run_cell = source
            run_start = new_idx
            run_count = 1
        _flush()
        out.append(new_row)
    return out


def _row_text_for_col(row: list[Any], col_idx: int) -> str:
    """取 row 中"展开后逻辑列"col_idx 处的文本（考虑 colspan）。"""
    if not isinstance(row, list):
        return ""
    cursor = 0
    for cell in row:
        if not isinstance(cell, dict):
            if cursor == col_idx:
                return ""
            cursor += 1
            continue
        cspan = max(_safe_int(cell.get("colspan", cell.get("colSpan", 1)), 1), 1)
        if cursor <= col_idx < cursor + cspan:
            return (cell.get("text") or "").strip()
        cursor += cspan
    return ""


def _is_date_like_header_text(text: str) -> bool:
    """文本看起来像日期/年份（用于在多行表头中挑出"日期行"）。"""
    raw = (text or "").strip()
    if not raw:
        return False
    # 中文常见
    if re.search(r"\d{4}\s*[年/.\-]\s*\d{1,2}", raw):
        return True
    if re.search(r"\d{4}年度", raw):
        return True
    # ISO / 欧式
    if re.search(r"\d{4}[\-./]\d{1,2}[\-./]\d{1,2}", raw):
        return True
    if re.search(r"\d{1,2}[\-./]\d{1,2}[\-./]\d{2,4}", raw):
        return True
    # 单纯 4 位年份
    if re.fullmatch(r"(19|20)\d{2}", raw):
        return True
    # 含 4 位年份（即使混入单位/币种等其他文字，例如 "2024RM'000" / "FY2024" /
    # "31 December 2024 RM'000"）。用边界判断避免误命中"2024-01-01"前缀冲突。
    if re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", raw):
        return True
    return False


_PERIOD_HEADER_KEYWORDS = ("期初", "期末", "年初", "年末")


def _is_period_like_header_text(text: str) -> bool:
    """文本含期初/期末/年初/年末等会计期间关键词。"""
    raw = (text or "").strip()
    if not raw:
        return False
    return any(kw in raw for kw in _PERIOD_HEADER_KEYWORDS)


def _select_header_row_for_display(
    header_cells: list[list[Any]],
    total_cols: int,
) -> Optional[list[Any]]:
    """从 cleaned_header_cells 选最合适的展示行：

    1. 从尾部往上扫，第一行"至少 2 列含日期"或"至少 2 列含期初/期末/年初/年末"的行即选中；
    2. 都不满足 -> 取最后一行（已剥离章节标题，最贴近数据）。
    """
    if not header_cells:
        return None
    for row in reversed(header_cells):
        if not isinstance(row, list):
            continue
        date_hits = 0
        period_hits = 0
        for col_idx in range(total_cols):
            text = _row_text_for_col(row, col_idx)
            if _is_date_like_header_text(text):
                date_hits += 1
            if _is_period_like_header_text(text):
                period_hits += 1
        if date_hits >= 2 or period_hits >= 2:
            return row
    for row in reversed(header_cells):
        if isinstance(row, list) and row:
            return row
    return None


def _expand_row_to_total_cols(row: list[Any], total_cols: int) -> list[dict[str, Any]]:
    """将一行 cells 展开/对齐到 total_cols 列，每列一个 cell（合并段拆开）。"""
    out: list[dict[str, Any]] = []
    cursor = 0
    for cell in row:
        if cursor >= total_cols:
            break
        if not isinstance(cell, dict):
            out.append({"text": str(cell or ""), "colStart": cursor, "colSpan": 1,
                        "colstart": cursor, "colspan": 1})
            cursor += 1
            continue
        cstart = cell.get("colStart")
        if cstart is None:
            cstart = cell.get("colstart")
        if isinstance(cstart, int):
            while cursor < cstart and cursor < total_cols:
                out.append({"text": "", "colStart": cursor, "colSpan": 1,
                            "colstart": cursor, "colspan": 1})
                cursor += 1
        cspan = max(_safe_int(cell.get("colspan", cell.get("colSpan", 1)), 1), 1)
        text = (cell.get("text") or "").strip()
        for offset in range(cspan):
            if cursor >= total_cols:
                break
            new_cell = {
                "text": text if offset == 0 else "",
                "colStart": cursor,
                "colSpan": 1,
                "colstart": cursor,
                "colspan": 1,
            }
            # 保留单元格上的其它元数据（rowstart/source_page_no 等）
            for k, v in cell.items():
                if k in ("text", "colStart", "colSpan", "colstart", "colspan"):
                    continue
                new_cell.setdefault(k, v)
            out.append(new_cell)
            cursor += 1
    while cursor < total_cols:
        out.append({"text": "", "colStart": cursor, "colSpan": 1,
                    "colstart": cursor, "colspan": 1})
        cursor += 1
    return out


def prepend_clean_statement_header_row(
    rows: Any,
    table_el: dict[str, Any],
    *,
    fallback_page_at: Optional[int],
    locale: Optional[str] = None,
) -> Optional[list[list[Any]]]:
    if not isinstance(table_el, dict):
        return None
    if table_el.get("type") not in ("table", "excel_table"):
        return None
    if not table_el.get("hasHeader"):
        return None

    header_cells = table_el.get("headerCells")
    if not header_cells or not isinstance(header_cells, list):
        return None

    src_rows_raw = rows if isinstance(rows, list) else []
    if len(src_rows_raw) < len(header_cells):
        return None

    num_header_rows = len(header_cells)
    src_rows = rows if isinstance(rows, list) else []
    data_start = num_header_rows if len(src_rows) >= num_header_rows else 0
    data_rows_raw = src_rows[data_start:]
    data_col_count = 0
    for row in data_rows_raw[:10]:
        if isinstance(row, list):
            c = 0
            for cell in row:
                sp = get_raw_cell_col_span(cell)
                cs = get_raw_cell_col_start(cell)
                start = cs if cs is not None else c
                data_col_count = max(data_col_count, start + sp)
                c = start + sp
    header_col_count = 0
    if header_cells[0]:
        c = 0
        for cell in header_cells[0]:
            sp = get_raw_cell_col_span(cell)
            cs = get_raw_cell_col_start(cell)
            start = cs if cs is not None else c
            header_col_count = max(header_col_count, start + sp)
            c = start + sp
    total_cols = data_col_count if data_col_count > 0 else header_col_count
    if total_cols == 0:
        return None

    cleaned_header_cells: list[list[Any]] = []
    section_header_rowstarts: set[int] = set()
    last_header_row_idx = len(header_cells) - 1
    for row_idx, row in enumerate(header_cells):
        keep_last_header_row = row_idx == last_header_row_idx
        if (
            not keep_last_header_row
            and isinstance(row, list)
            and len(row) == 1
            and isinstance(row[0], dict)
            and max(_safe_int(row[0].get("colspan", 1), 1), 1) >= total_cols
        ):
            row_start = row[0].get("rowstart")
            if isinstance(row_start, int):
                section_header_rowstarts.add(row_start)
            continue
        cleaned_header_cells.append(row)

    # 从尾部向上剔除连续的"单 cell 覆盖全列"伪表头行（章节标题被 LLM 误纳入 headerCells）。
    # 遇到第一行不符合即停止，避免误伤上方真实表头。
    while cleaned_header_cells:
        tail = cleaned_header_cells[-1]
        if (
            isinstance(tail, list)
            and len(tail) == 1
            and isinstance(tail[0], dict)
            and max(_safe_int(tail[0].get("colspan", 1), 1), 1) >= total_cols
        ):
            row_start = tail[0].get("rowstart")
            if isinstance(row_start, int):
                section_header_rowstarts.add(row_start)
            cleaned_header_cells.pop()
            continue
        break

    if not cleaned_header_cells:
        return None

    # 数据行：跳过被表头覆盖的行，得到真正的"数据起点"
    src_rows = rows if isinstance(rows, list) else []
    if not src_rows:
        return None

    hc_max_rowstart = -1
    for hc_row in header_cells:
        for cell in hc_row:
            if isinstance(cell, dict):
                row_start = cell.get("rowstart")
                if isinstance(row_start, int) and row_start not in section_header_rowstarts:
                    hc_max_rowstart = max(hc_max_rowstart, row_start)

    skip_count = 0
    for row in src_rows:
        if not isinstance(row, list) or not row:
            skip_count += 1
            continue
        first_cell = row[0] if isinstance(row[0], dict) else {}
        row_start = first_cell.get("rowstart")
        if row_start is not None and isinstance(row_start, int) and row_start <= hc_max_rowstart:
            skip_count += 1
        else:
            break

    if skip_count == 0:
        skip_count = 1

    data_rows = src_rows[skip_count:]
    raw_data_rows = normalize_raw_rows(data_rows)

    # 识别辅助列（NOTE/序号/附注 + 表头空白且数据非纯数字）
    aux_col_indices = identify_auxiliary_column_indices(cleaned_header_cells, total_cols)
    aux_col_indices |= _identify_blank_header_columns(
        cleaned_header_cells, total_cols, aux_col_indices, raw_data_rows
    )
    kept_col_indices = [index for index in range(total_cols) if index not in aux_col_indices]
    header_cols_with_cells: set[int] = set()
    for hc_row in cleaned_header_cells:
        if not isinstance(hc_row, list):
            continue
        for cell in hc_row:
            cs = get_raw_cell_col_start(cell)
            sp = get_raw_cell_col_span(cell)
            start = cs if cs is not None else 0
            for c in range(start, start + sp):
                header_cols_with_cells.add(c)
    # 同时从数据行收集列号，防止 dual caliber split 后 header colStart 重新编号
    # 导致稀疏 data 列被误判为幽灵列
    for dr in raw_data_rows:
        if not isinstance(dr, list):
            continue
        for cell in dr:
            if not isinstance(cell, dict):
                continue
            cs = get_raw_cell_col_start(cell)
            sp = get_raw_cell_col_span(cell)
            start = cs if cs is not None else 0
            for c in range(start, start + sp):
                if c < total_cols:
                    header_cols_with_cells.add(c)
    kept_col_indices = [c for c in kept_col_indices if c in header_cols_with_cells]
    if len(kept_col_indices) < 2:
        return None

    # 同步切片 cleaned_header_cells（仅保留 kept 列）
    cleaned_header_cells = _filter_header_cells_by_kept_indices(
        cleaned_header_cells, kept_col_indices, total_cols
    )

    # 选展示行：优先含日期最多的行；若都无日期 → 最后一行
    new_total_cols = len(kept_col_indices)
    chosen_row = _select_header_row_for_display(cleaned_header_cells, new_total_cols)
    if chosen_row is None:
        return None
    display_row = _expand_row_to_total_cols(chosen_row, new_total_cols)
    if display_row and not (display_row[0].get("text") or "").strip():
        if any((cell.get("text") or "").strip() for cell in display_row[1:]):
            display_row[0]["text"] = "Item" if ((locale or "").strip().lower().startswith("en")) else "项目"

    # 数据行同步切片：按列范围切片，正确处理 colSpan>1 的合并单元格
    kept_set = set(kept_col_indices)
    new_index_by_old = {
        old_index: new_index
        for new_index, old_index in enumerate(kept_col_indices)
    }
    filtered_data_rows: list[list[Any]] = []
    for row in raw_data_rows:
        if not isinstance(row, list):
            filtered_data_rows.append(row)
            continue
        kept_cells: list[Any] = []
        col_cursor = 0
        for cell in row:
            col_span = get_raw_cell_col_span(cell)
            col_start_val = get_raw_cell_col_start(cell)
            start = col_start_val if col_start_val is not None else col_cursor
            end = start + col_span
            cell_cols = list(range(start, end))
            kept_cols = [c for c in cell_cols if c in kept_set]
            if kept_cols:
                new_cell = dict(cell) if isinstance(cell, dict) else {"text": str(cell or "")}
                new_start = new_index_by_old[kept_cols[0]]
                new_cell["colstart"] = new_start
                new_cell["colStart"] = new_start
                new_cell["colspan"] = len(kept_cols)
                new_cell["colSpan"] = len(kept_cols)
                kept_cells.append(new_cell)
            col_cursor = end
        filtered_data_rows.append(kept_cells)

    final_rows = [display_row] + filtered_data_rows

    # 回写 cleaned_header_cells 到 table_el：用于下游推断口径/类型/GAAP/日期
    table_el["cleanedHeaderCells"] = cleaned_header_cells

    return final_rows


def drop_note_column_by_first_row(rows: Any) -> Optional[list[list[Any]]]:
    """当 headerCells 缺失时的兜底：基于 rows[0] 删除以下列：
    1. 文本是 NOTE/Note/notes/附注/备注（大小写不敏感）的列；
    2. 文本是 行次/行数/序号/编号/注释/说明 等辅助列表头的列；
    3. 文本为空白的非首列（首列保留作为科目列）。
    删除后重排 colstart。

    返回新的 rows；若未发现可删列或 rows 为空，则返回 None（调用方保留原 rows）。
    """
    raw_rows = normalize_raw_rows(rows)
    if not raw_rows:
        return None

    first_row = raw_rows[0]
    if not isinstance(first_row, list) or not first_row:
        return None

    def _cell_col_range(cell: Any, col_cursor: int) -> tuple[int, int]:
        col_span = get_raw_cell_col_span(cell)
        col_start_val = get_raw_cell_col_start(cell)
        start = col_start_val if col_start_val is not None else col_cursor
        return start, start + col_span

    drop_col_indices: list[int] = []
    col_cursor = 0
    for cell in first_row:
        start, end = _cell_col_range(cell, col_cursor)
        col_span = end - start
        text = ""
        if isinstance(cell, dict):
            text = (cell.get("text") or "").strip()
        text_lower = text.lower()
        if text and (
            text_lower in STATEMENT_NOTE_PATTERNS
            or text in STATEMENT_AUXILIARY_HEADER_TOKENS
            or text.startswith(("附注", "注释", "备注"))
        ):
            for offset in range(col_span):
                drop_col_indices.append(start + offset)
        elif not text and start > 0:
            for offset in range(col_span):
                drop_col_indices.append(start + offset)
        col_cursor = end

    # 数据值检测：非首列中存在连续 5 行每次涨 1 且值 <= 100 的，视为行次列
    total_cols = 0
    for row in raw_rows:
        if not isinstance(row, list):
            continue
        c = 0
        for cell in row:
            s, e = _cell_col_range(cell, c)
            total_cols = max(total_cols, e)
            c = e
    drop_set_preview = set(drop_col_indices)
    for col_idx in range(1, total_cols):
        if col_idx in drop_set_preview:
            continue
        consecutive = 0
        prev_val: Optional[int] = None
        for row in raw_rows[1:]:
            if not isinstance(row, list):
                consecutive = 0
                prev_val = None
                continue
            cell_val: Optional[str] = None
            c = 0
            for cell in row:
                s, e = _cell_col_range(cell, c)
                if s <= col_idx < e:
                    if isinstance(cell, dict):
                        cell_val = (cell.get("text") or "").strip()
                    break
                c = e
            if not cell_val:
                consecutive = 0
                prev_val = None
                continue
            try:
                int_val = int(cell_val)
            except ValueError:
                consecutive = 0
                prev_val = None
                continue
            if int_val > 100:
                consecutive = 0
                prev_val = None
                continue
            if prev_val is not None and int_val == prev_val + 1:
                consecutive += 1
                if consecutive >= 5:
                    drop_col_indices.append(col_idx)
                    break
            else:
                consecutive = 1
            prev_val = int_val

    if not drop_col_indices:
        return None

    drop_set = set(drop_col_indices)

    new_rows: list[list[Any]] = []
    for row in raw_rows:
        if not isinstance(row, list):
            new_rows.append(row)
            continue
        kept_cells: list[Any] = []
        col_cursor = 0
        for cell in row:
            start, end = _cell_col_range(cell, col_cursor)
            col_span = end - start
            cell_cols = list(range(start, end))
            kept_cols = [c for c in cell_cols if c not in drop_set]
            if kept_cols:
                if len(kept_cols) == col_span:
                    kept_cells.append(cell)
                else:
                    new_cell = dict(cell) if isinstance(cell, dict) else {"text": str(cell or "")}
                    new_cell["colspan"] = len(kept_cols)
                    kept_cells.append(new_cell)
            col_cursor = end
        new_rows.append(kept_cells)

    # 重排 colstart：保留原始相对位置，只扣除被删列的偏移
    sorted_drop = sorted(drop_set)
    for row in new_rows:
        if not isinstance(row, list):
            continue
        fallback_col_start = 0
        for cell in row:
            if not isinstance(cell, dict):
                fallback_col_start += 1
                continue
            col_span = get_raw_cell_col_span(cell)
            col_start_val = get_raw_cell_col_start(cell)
            if col_start_val is not None:
                shift = sum(1 for d in sorted_drop if d < col_start_val)
                new_col_start = col_start_val - shift
            else:
                new_col_start = fallback_col_start
            if "colstart" in cell:
                cell["colstart"] = new_col_start
            if "colStart" in cell:
                cell["colStart"] = new_col_start
            fallback_col_start = new_col_start + col_span

    return new_rows
