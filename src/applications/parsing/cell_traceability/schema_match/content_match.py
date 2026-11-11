"""内容匹配兜底：当结构不匹配时，调用 /predict/rec 获取 OCR 文本，与 input_json rows 做行级+行内 DP 对齐回填 bBox。"""

import asyncio
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from loguru import logger

from src.utils import json_utils
from .utils import coord_to_bbox, filter_and_sort_boxes

_OCR_BASE = os.getenv("CELL_LOCATION_API_BASE", "http://10.192.72.11:8082/api/v1/predict/model")
_OCR_REC_URL = os.getenv(
    "OCR_DETECT_REC_URL",
    _OCR_BASE.replace("/predict/model", "/predict/rec") if "/predict/model" in _OCR_BASE else _OCR_BASE + "/predict/rec",
)
_OCR_TIMEOUT = float(os.getenv("OCR_DETECT_TIMEOUT", "120"))

# ── 持久化连接池 ──────────────────────────────────────────────────────────────
# 复用 TCP 连接，避免每次请求新建/销毁 httpx.AsyncClient 的开销。
# 注意：AsyncClient 绑定创建它的 event loop，必须在应用关闭前显式 aclose()，
# 否则 GC finalizer 在 loop 已关闭后调用 aclose() 会抛 RuntimeError: Event loop is closed。
# close_shared_client() 已注册到 src/main.py lifespan 的 finally 块。
_shared_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    """惰性创建并复用全局 AsyncClient；若已关闭则重建。"""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=_OCR_TIMEOUT,
            limits=httpx.Limits(max_connections=20, keepalive_expiry=30),
        )
    return _shared_client


async def close_shared_client() -> None:
    """应用关闭时显式释放连接池，避免 loop 关闭后 GC finalizer 报错。"""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None

_STRIP_RE = re.compile(r"[\s\u3000,，.。、;；:：!！?？(（)）\[\]【】{}\-—_/\\\"'`~～@#$%^&*+=<>《》|]")
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _normalize(text: Optional[str]) -> str:
    """统一文本：去空格/标点，'-' / '—' / '一' 视为空。"""
    if text is None:
        return ""
    t = str(text).strip()
    if t in {"-", "—", "一"}:
        return ""
    return _STRIP_RE.sub("", t)


def _normalize_number(text: Optional[str]) -> str:
    """数字专用归一化：仅移除空白和千分位逗号，保留数字符号和小数点。"""
    if text is None:
        return ""
    t = str(text).strip().replace("，", ",").replace(" ", "").replace("\u3000", "")
    if t in {"-", "—", "一"}:
        return ""
    t = t.replace(",", "")
    return t


def _is_numeric_like(text: Optional[str]) -> bool:
    t = _normalize_number(text)
    return bool(t) and bool(_NUM_RE.fullmatch(t))


async def _call_predict_rec(
    image_path: str,
) -> List[Dict[str, Any]]:
    """调用 /api/v1/predict/rec，返回 [{value, coordinate}, ...]。"""
    p = Path(image_path)
    if not p.is_file():
        logger.warning("content_match: 图片不存在 | path={}", image_path)
        return []

    raw = await asyncio.to_thread(p.read_bytes)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    files = {"file": (p.name, raw, mime)}

    try:
        client = await _get_client()
        resp = await client.post(_OCR_REC_URL, files=files)
        resp.raise_for_status()
        body = json_utils.response_json(resp)
    except Exception:
        logger.exception("content_match: 调用 predict/rec 失败 | url={}", _OCR_REC_URL)
        return []

    return body.get("data", []) if isinstance(body, dict) else []


def _group_rec_rows(rec_results: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """按 y1 分组，把 rec 结果聚成若干行。用锚点比较避免链式漂移。"""
    rows: List[List[Dict[str, Any]]] = []
    filtered = [r for r in rec_results if isinstance(r, dict) and len(r.get("coordinate", [])) >= 4]
    if not filtered:
        return rows

    filtered = sorted(filtered, key=lambda r: (float(r["coordinate"][1]), float(r["coordinate"][0])))
    heights = [float(r["coordinate"][3]) - float(r["coordinate"][1]) for r in filtered]
    tol = max(5.0, median(heights) * 0.35) if heights else 5.0

    cur: List[Dict[str, Any]] = []
    anchor_y: Optional[float] = None
    for r in filtered:
        y1 = float(r["coordinate"][1])
        if anchor_y is None or abs(y1 - anchor_y) <= tol:
            cur.append(r)
            if anchor_y is None:
                anchor_y = y1  # 新行以第一个元素的 y1 为锚点，避免链式漂移
        else:
            rows.append(sorted(cur, key=lambda x: float(x["coordinate"][0])))
            cur = [r]
            anchor_y = y1
    if cur:
        rows.append(sorted(cur, key=lambda x: float(x["coordinate"][0])))
    return rows


def _match_score(cell_text: Optional[str], rec_text: Optional[str]) -> float:
    """单个 cell 与 rec 文本的匹配分数。数字必须完全一致；文本允许高相似度。"""
    cell_norm = _normalize(cell_text)
    rec_norm = _normalize(rec_text)

    if cell_norm == "" and rec_norm == "":
        return 3.0
    if cell_norm == "" or rec_norm == "":
        return -4.0

    cell_is_num = _is_numeric_like(cell_text)
    rec_is_num = _is_numeric_like(rec_text)
    if cell_is_num or rec_is_num:
        if cell_is_num and rec_is_num:
            if _normalize_number(cell_text) == _normalize_number(rec_text):
                return 10.0
            # OCR 常见小错（空格插入、字符混淆）允许高相似度近似匹配
            ratio = SequenceMatcher(None, _normalize_number(cell_text), _normalize_number(rec_text)).ratio()
            if ratio >= 0.95:
                return 6.0
        return -10.0

    if cell_norm == rec_norm:
        return 10.0
    if len(cell_norm) >= 4 and (cell_norm in rec_norm or rec_norm in cell_norm):
        return 6.0

    ratio = SequenceMatcher(None, cell_norm, rec_norm).ratio()
    if ratio >= 0.95:
        return 7.0
    if ratio >= 0.88:
        return 4.0
    return -6.0


def _align_row(
    table_row: Sequence[Dict[str, Any]],
    rec_row: Sequence[Dict[str, Any]],
    skip_cell_penalty: float = -2.0,
    skip_rec_penalty: float = -2.0,
) -> Tuple[float, List[Tuple[int, int]]]:
    """行内 DP 对齐，返回最佳分数及匹配对 (cell_idx, rec_idx)。"""
    n, m = len(table_row), len(rec_row)
    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    bt: List[List[Tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + skip_cell_penalty
        bt[i][0] = ("skip_cell", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + skip_rec_penalty
        bt[0][j] = ("skip_rec", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cell = table_row[i - 1]
            rec = rec_row[j - 1]
            s = _match_score(cell.get("text"), rec.get("value"))

            candidates = [
                (dp[i - 1][j - 1] + s, ("match", i - 1, j - 1)),
                (dp[i - 1][j] + skip_cell_penalty, ("skip_cell", i - 1, j)),
                (dp[i][j - 1] + skip_rec_penalty, ("skip_rec", i, j - 1)),
            ]
            best_score, best_bt = max(candidates, key=lambda x: x[0])
            dp[i][j] = best_score
            bt[i][j] = best_bt

    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = bt[i][j]
        if step is None:
            break
        kind, pi, pj = step
        if kind == "match":
            if _match_score(table_row[i - 1].get("text"), rec_row[j - 1].get("value")) > 0:
                pairs.append((i - 1, j - 1))
            i, j = pi, pj
        elif kind == "skip_cell":
            i, j = pi, pj
        else:
            i, j = pi, pj
    pairs.reverse()
    return dp[n][m], pairs


def _row_anchor_texts(table_row: Sequence[Dict[str, Any]]) -> List[str]:
    texts = []
    for cell in table_row:
        t = _normalize(cell.get("text"))
        if t:
            texts.append(t)
    return texts


def _row_pair_score(table_row: Sequence[Dict[str, Any]], rec_row: Sequence[Dict[str, Any]]) -> float:
    """表格行与 OCR 行的匹配分数。"""
    table_non_empty = len([1 for c in table_row if _normalize(c.get("text"))])
    rec_non_empty = len([1 for r in rec_row if _normalize(r.get("value"))])

    # 双方都是空行/极短行：给 0 分而非负分，避免财务表分隔空行拖垮整体对齐
    if table_non_empty == 0 and rec_non_empty == 0:
        return 0.0

    score, pairs = _align_row(table_row, rec_row)
    if not pairs:
        return -8.0

    overlap_bonus = min(len(pairs), max(table_non_empty, rec_non_empty, 1)) * 1.2
    return score + overlap_bonus


def _align_rows(
    table_rows: Sequence[Sequence[Dict[str, Any]]],
    rec_rows: Sequence[Sequence[Dict[str, Any]]],
    skip_table_penalty: float = -3.0,
    skip_rec_penalty: float = -3.0,
) -> List[Tuple[int, int]]:
    """行级 DP 对齐，返回 (table_row_idx, rec_row_idx)。"""
    n, m = len(table_rows), len(rec_rows)
    row_score = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            row_score[i][j] = _row_pair_score(table_rows[i], rec_rows[j])

    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    bt: List[List[Tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + skip_table_penalty
        bt[i][0] = ("skip_table", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + skip_rec_penalty
        bt[0][j] = ("skip_rec", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                (dp[i - 1][j - 1] + row_score[i - 1][j - 1], ("match", i - 1, j - 1)),
                (dp[i - 1][j] + skip_table_penalty, ("skip_table", i - 1, j)),
                (dp[i][j - 1] + skip_rec_penalty, ("skip_rec", i, j - 1)),
            ]
            best_score, best_bt = max(candidates, key=lambda x: x[0])
            dp[i][j] = best_score
            bt[i][j] = best_bt

    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = bt[i][j]
        if step is None:
            break
        kind, pi, pj = step
        if kind == "match" and row_score[i - 1][j - 1] > 0:
            pairs.append((i - 1, j - 1))
        i, j = pi, pj
    pairs.reverse()
    return pairs


def _infer_unfilled(rows: List[List[Dict]], n_rows: int, n_cols: int) -> None:
    """对 rows 中仍无 bBox 的 cell，从同行/同列已有 bBox 的邻居推断。就地修改。"""
    grid: List[List[Optional[Dict[str, Any]]]] = [[None] * n_cols for _ in range(n_rows)]
    for row in rows:
        if not isinstance(row, list):
            continue
        for cell in row:
            if not isinstance(cell, dict):
                continue
            r = int(cell.get("rowstart", 0) or 0)
            c = int(cell.get("colstart", 0) or 0)
            if 0 <= r < n_rows and 0 <= c < n_cols:
                grid[r][c] = cell

    def _col_lr(r: int, c: int) -> Tuple[Optional[float], Optional[float]]:
        for rr in range(r - 1, -1, -1):
            cell = grid[rr][c]
            if cell and cell.get("bBox"):
                bb = cell["bBox"]
                return bb.get("l"), bb.get("r")
        for rr in range(r + 1, n_rows):
            cell = grid[rr][c]
            if cell and cell.get("bBox"):
                bb = cell["bBox"]
                return bb.get("l"), bb.get("r")
        return None, None

    def _row_tb(r: int, c: int) -> Tuple[Optional[float], Optional[float]]:
        for cc in range(c - 1, -1, -1):
            cell = grid[r][cc]
            if cell and cell.get("bBox"):
                bb = cell["bBox"]
                return bb.get("t"), bb.get("b")
        for cc in range(c + 1, n_cols):
            cell = grid[r][cc]
            if cell and cell.get("bBox"):
                bb = cell["bBox"]
                return bb.get("t"), bb.get("b")
        return None, None

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            if cell is None or cell.get("bBox"):
                continue
            l, right = _col_lr(r, c)
            t, b = _row_tb(r, c)
            if l is not None and t is not None and right is not None and b is not None:
                cell["bBox"] = {
                    "l": round(l, 2),
                    "t": round(t, 2),
                    "r": round(right, 2),
                    "b": round(b, 2),
                    "coordOrigin": "TOPLEFT",
                }
                cell["cellTraceabilitySource"] = "inferred"


async def content_match_and_fill(
    image_path: str,
    boxes: List[Dict[str, Any]],
    table: Dict[str, Any],
) -> bool:
    """
    内容匹配兜底：
      1. 对 boxes 排序后调用 /predict/rec 获取 OCR 文本
      2. 将 rec 结果按 y 分组成若干行
      3. 表格行与 OCR 行之间做 DP 对齐
      4. 每一行内部再做 DP 对齐
      5. 数字必须完全一致；匹配成功直接使用 rec.coordinate
      6. 未填充的 cell 通过邻居推断 bBox

    返回 True 表示至少有一个 cell 被填充。
    """
    sorted_boxes = filter_and_sort_boxes(boxes)
    if not sorted_boxes:
        return False

    rec_results = await _call_predict_rec(image_path)
    if not rec_results:
        return False

    table_rows = [row for row in table.get("rows", []) if isinstance(row, list)]
    if not table_rows:
        return False
    rec_rows = _group_rec_rows(rec_results)
    if not rec_rows:
        return False

    row_pairs = _align_rows(table_rows, rec_rows)
    filled_count = 0
    matched_row_count = 0

    for tr_idx, rr_idx in row_pairs:
        table_row = table_rows[tr_idx]
        rec_row = rec_rows[rr_idx]
        score, cell_pairs = _align_row(table_row, rec_row)
        if not cell_pairs:
            continue
        matched_row_count += 1
        for ci, rj in cell_pairs:
            cell = table_row[ci]
            rec = rec_row[rj]
            coord = rec.get("coordinate")
            if isinstance(coord, (list, tuple)) and len(coord) >= 4:
                cell["bBox"] = coord_to_bbox(coord)
                cell["cellTraceabilitySource"] = "matched"
                filled_count += 1

    cells = [cell for row in table_rows for cell in row if isinstance(cell, dict)]
    n_rows = max((int(c.get("rowstart", 0) or 0) for c in cells), default=0) + 1
    n_cols = max((int(c.get("colstart", 0) or 0) for c in cells), default=0) + 1
    _infer_unfilled(table_rows, n_rows, n_cols)

    return filled_count > 0
