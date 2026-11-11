"""流水线：detect -> filter -> fill -> adjust，输入 json 内容，输出填好坐标且按表格 bBox 调整后的 json 内容"""
import asyncio
from pathlib import Path

from loguru import logger

from . import adjust_cell_coords
from . import detect
from . import filter as filter_mod
from . import fill

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")


def _resolve_images(images, images_dir):
    """解析图片路径。仅使用调用方传入的 images 或 images_dir，无默认。"""
    if images:
        return [Path(p) for p in images]
    if images_dir:
        d = Path(images_dir)
        if d.exists():
            paths = []
            for pat in _IMAGE_EXTS:
                paths.extend(d.glob(pat))
            if paths:
                return sorted(paths)
    return []


def _bbox_rotated_to_original(bbox: list, angle: int, orig_w: int, orig_h: int) -> list:
    """将回正图上的像素 bbox 逆旋转回原图坐标。

    cv2/PIL 均为逆时针旋转，逆变换如下
    （px/py 为回正图坐标，ox/oy 为原图坐标）：
      90°:  rot尺寸=(orig_h, orig_w), inv: ox=orig_w-1-py,   oy=px
      180°: rot尺寸=(orig_w, orig_h), inv: ox=orig_w-1-px,   oy=orig_h-1-py
      270°: rot尺寸=(orig_h, orig_w), inv: ox=py,             oy=orig_h-1-px
    """
    if angle == 0:
        return bbox
    x1, y1, x2, y2 = bbox
    if angle == 90:
        def inv(px, py): return (orig_w - 1 - py, px)
    elif angle == 180:
        def inv(px, py): return (orig_w - 1 - px, orig_h - 1 - py)
    elif angle == 270:
        def inv(px, py): return (py, orig_h - 1 - px)
    else:
        return bbox
    corners = [inv(x1, y1), inv(x2, y1), inv(x2, y2), inv(x1, y2)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def _inverse_rotate_cell_bboxes(items: list, orient_map: dict) -> list:
    """将 fill 产出的 cell bBox（回正图像素空间）逆旋转回原图像素空间。"""
    if not orient_map:
        return items
    for el in items:
        if not isinstance(el, dict) or el.get("type") != "table":
            continue
        file_id = el.get("file_id")
        element_id = el.get("element_id")
        if file_id is None or element_id is None:
            continue
        stem = f"{file_id}_{element_id}"
        info = orient_map.get(stem)
        if not info:
            continue
        angle = info["angle"]
        orig_w = info["orig_width"]
        orig_h = info["orig_height"]
        count = 0
        for row in el.get("rows", []):
            if not isinstance(row, list):
                continue
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                bbox = cell.get("bBox")
                if not isinstance(bbox, dict):
                    continue
                l_, t_, r_, b_ = bbox.get("l"), bbox.get("t"), bbox.get("r"), bbox.get("b")
                if None in (l_, t_, r_, b_):
                    continue
                x1, y1, x2, y2 = _bbox_rotated_to_original(
                    [float(l_), float(t_), float(r_), float(b_)], angle, orig_w, orig_h
                )
                cell["bBox"] = {"l": round(x1, 2), "t": round(y1, 2), "r": round(x2, 2), "b": round(y2, 2)}
                count += 1
    return items


async def cell_traceability(images=None, images_dir=None, input_json=None):
    """输入 json 内容（表格），输出填好坐标的 json 内容。"""
    jpgs = _resolve_images(images, images_dir)
    if not jpgs:
        return None
    img_paths = [str(p) for p in jpgs]

    if not input_json or (isinstance(input_json, list) and len(input_json) == 0):
        return None
    tables_in = input_json if isinstance(input_json, list) else [input_json]

    detect_results, orient_map = await detect.execute(img_paths)
    for d in detect_results:
        n = len(d.get("boxes", []))
        name = Path(d.get("input_path", "")).name
        if n == 0:
            logger.warning(f"detect {name} 返回空 boxes，CELL_LOCATION_API_BASE 可能不可达或返回空")
    filter_results = await asyncio.to_thread(filter_mod.execute, detect_results)
    filled = await fill.execute(filter_results, input_json=tables_in, orient_map=orient_map)

    # 逆旋转：将 cell bbox 从回正图像素空间映射回原图像素空间
    if orient_map:
        filled = await asyncio.to_thread(_inverse_rotate_cell_bboxes, filled, orient_map)

    filled = await asyncio.to_thread(adjust_cell_coords.execute, filled)
    tables_processed = sum(1 for el in filled if isinstance(el, dict) and el.get("type") == "table")
    return {"content": filled, "images": img_paths, "tables_processed": tables_processed}
