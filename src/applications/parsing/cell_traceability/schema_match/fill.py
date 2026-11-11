"""Step3: 将 filter 结果的坐标填入 input_json 中 type=table 的 rows 单元格。

匹配策略：
  1. 按文件名 ID 配对 — 从 filter dict 的 input_path 解析 (file_id, element_id)，
     直接定位 items 中对应。
  2. 结构校验 — 比较 OCR 检测的 grid (n_rows, n_cols) 与 table 行列数，
     一致则走 infer_missing_boxes + 回填。
  3. 结构不匹配 — 调用 content_match 按 OCR 文本与 rows 对齐回填；表级字段 cellTraceabilityMatch=content|content_failed。
"""
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from PIL import Image

from .utils import table_grid_shape, infer_grid_from_boxes, coord_to_bbox
from .infer_coor import infer_missing_boxes
from .content_match import content_match_and_fill


def _rotate_image_to_upright(image_path: str, angle: int) -> Optional[str]:
    """将旋转的图片回正，保存到临时文件，返回临时文件路径。"""
    if angle == 0:
        return None
    try:
        img = Image.open(image_path)
        # PIL rotate 正值=逆时针，OCR 返回的 angle 是需要逆时针旋转的角度
        rotated = img.rotate(angle, expand=True)
        suffix = Path(image_path).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        rotated.save(tmp.name)
        return tmp.name
    except Exception as e:
        logger.warning(f"回正图片失败: {e}, path={image_path}, angle={angle}")
        return None


def _parse_filename_id(input_path_or_stem: str) -> Optional[Tuple[str, int]]:
    """从文件名（如 'abc123_5.png'）解析出 (file_id, element_id)。

    约定格式：{file_id}_{element_id}.{ext}，以下划线分割，最后一段为 element_id。
    """
    stem = Path(input_path_or_stem).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        eid = int(parts[-1])
    except ValueError:
        return None
    file_id = "_".join(parts[:-1])
    return (file_id, eid)


async def execute(filter_results: list[dict], input_json=None, orient_map: dict = None):
    """将 filter 结果的坐标回填到 input_json 中 type=table 的 rows 单元格。"""
    if input_json is None:
        raise ValueError("input_json 必填")
    items = input_json if isinstance(input_json, list) else [input_json]
    if not isinstance(items, list):
        items = [items]

    tables = [(i, el) for i, el in enumerate(items) if isinstance(el, dict) and el.get("type") == "table"]
    if not tables:
        return items

    id2table = {}
    for idx_in_tables, (idx_in_items, tbl) in enumerate(tables):
        fid = tbl.get("file_id")
        eid = tbl.get("element_id")
        if fid is not None and eid is not None:
            id2table[(str(fid), int(eid))] = idx_in_tables

    matches = {}
    # 收集需要 content match 的表格，最后并发处理
    content_match_tasks = []  # [(idx_in_tables, input_path, boxes, tbl, rotated_path)]

    for d in filter_results:
        boxes = d.get("boxes", [])
        input_path = d.get("input_path", "")
        parsed = _parse_filename_id(input_path) if input_path else None
        idx_in_tables = id2table.get(parsed) if parsed else None
        if idx_in_tables is None:
            continue
        idx_in_items, tbl = tables[idx_in_tables]
        stem = Path(input_path).stem if input_path else ""
        if not boxes:
            tbl["cellTraceabilityMatch"] = "skipped_no_boxes"
            continue
        n_rows, n_cols, grid = infer_grid_from_boxes(boxes)

        # 校验：OCR 检测的 grid 结构与 table 行列数是否一致
        table_shape = table_grid_shape(tbl)
        detected_shape = (n_rows, n_cols)
        if detected_shape != table_shape:
            if input_path:
                # 如果图片有旋转，先回正再传给 rec
                rotated_path = None
                if orient_map and stem in orient_map:
                    info = orient_map[stem]
                    rotated_path = _rotate_image_to_upright(input_path, info["angle"])
                content_match_tasks.append((idx_in_tables, input_path, boxes, tbl, rotated_path))
            else:
                tbl["cellTraceabilityMatch"] = "content_failed"
            continue

        boxes_completed = infer_missing_boxes(grid, tbl)
        matches[idx_in_tables] = (stem, boxes_completed)

    # 并发执行所有 content match
    if content_match_tasks:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            content_match_and_fill(rp or ip, bx, tbl)
            for _, ip, bx, tbl, rp in content_match_tasks
        ])
        logger.info(
            "单元格溯源 rec 总耗时: {:.2f}s, 文件数={}",
            time.perf_counter() - t0,
            len(content_match_tasks),
        )
        for (idx_in_tables, _, _, tbl, rp), ok in zip(content_match_tasks, results):
            tbl["cellTraceabilityMatch"] = "content" if ok else "content_failed"
            # 清理临时回正图片
            if rp:
                try:
                    Path(rp).unlink(missing_ok=True)
                except Exception:
                    pass

    # 回填：按 cell 的 (rowstart,colstart) 顺序与 boxes_completed 一一对应
    for idx_in_tables, (name, boxes_completed) in sorted(matches.items()):
        idx_in_items, tbl = tables[idx_in_tables]
        tbl["cellTraceabilityMatch"] = "structure"
        it = iter(boxes_completed)
        for row in tbl.get("rows", []):
            if not isinstance(row, list):
                continue
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                try:
                    b = next(it)
                    if b and len(b.get("coordinate", [])) >= 4:
                        cell["bBox"] = coord_to_bbox(b["coordinate"])
                        cell["cellTraceabilitySource"] = "structure"
                except StopIteration:
                    break

    return items
