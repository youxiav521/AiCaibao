"""
ADE Extract VLM 工具函数

从 llm_rule_extract_visualize.py / ocr_guided_extract.py 提取的纯函数，
供 VlmExtractService 使用。两个脚本后续可删除。
"""
import base64
import copy
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # type: ignore
from PIL import Image as PILImage
from PIL import ImageOps

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
COORD_NORM = 1000
MERGE_GAP = 30
MODEL_MAX_DIM = 1024

OCR_BOX_PATTERN = re.compile(
    r"(?P<text>.+?)\((?P<x1>\d+(?:\.\d+)?)\s*,\s*(?P<y1>\d+(?:\.\d+)?)\)\s*,\s*"
    r"\((?P<x2>\d+(?:\.\d+)?)\s*,\s*(?P<y2>\d+(?:\.\d+)?)\)"
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class PageImage:
    page_index: int
    image_format: str
    image_bytes: bytes
    width: int
    height: int
    offset_x: int = 0
    offset_y: int = 0
    # 原始尺寸（PDF 为页面 points，图片为像素）
    # bbox 最终需换算到这个坐标系返回给调用方
    orig_width: float = 0.0
    orig_height: float = 0.0


@dataclass
class OCRItem:
    text: str
    bbox: List[float]  # pixel coords [x1, y1, x2, y2]
    page_index: int


# ---------------------------------------------------------------------------
# 页面渲染
# ---------------------------------------------------------------------------

def render_pages_from_bytes(
    file_bytes: bytes, filename: str, dpi: int = 160, max_pages: int = 5
) -> List[PageImage]:
    """将文件字节渲染为 PageImage 列表。PDF 用 fitz，图片直接读。"""
    suffix = Path(filename).suffix.lower()
    pages: List[PageImage] = []

    if suffix in IMAGE_EXTS:
        with PILImage.open(io.BytesIO(file_bytes)) as im:
            im = ImageOps.exif_transpose(im)
            width, height = im.size
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        pages.append(PageImage(
            page_index=0, image_format="png",
            image_bytes=img_bytes, width=width, height=height,
            orig_width=float(width), orig_height=float(height),
        ))
    else:
        doc = fitz.open(stream=file_bytes, filetype=suffix.lstrip("."))
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        try:
            limit = len(doc) if max_pages <= 0 else min(len(doc), max_pages)
            for page_index in range(limit):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pages.append(PageImage(
                    page_index=page_index, image_format="png",
                    image_bytes=pix.tobytes("png"),
                    width=pix.width, height=pix.height,
                    orig_width=page.rect.width,
                    orig_height=page.rect.height,
                ))
        finally:
            doc.close()

    return pages


# ---------------------------------------------------------------------------
# 图片处理
# ---------------------------------------------------------------------------

def image_bytes_to_data_url(image_bytes: bytes, image_format: str) -> str:
    """将图片字节编码为 data URL。"""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_format};base64,{encoded}"


def resize_image_for_model(
    image_bytes: bytes, width: int, height: int, max_dim: int = MODEL_MAX_DIM
) -> Tuple[bytes, int, int, float, float]:
    """缩放图片使最长边不超过 max_dim。

    Returns (new_bytes, new_w, new_h, scale_x, scale_y)
    scale_x/scale_y 用于将模型坐标映射回原图坐标。
    """
    if max(width, height) <= max_dim:
        return image_bytes, width, height, 1.0, 1.0

    ratio = max_dim / max(width, height)

    doc = fitz.open("png", image_bytes)
    try:
        pdf_bytes = doc.convert_to_pdf()
    finally:
        doc.close()

    page_doc = fitz.open("pdf", pdf_bytes)
    try:
        pdf_page = page_doc[0]
        mat = fitz.Matrix(
            ratio * width / pdf_page.rect.width,
            ratio * height / pdf_page.rect.height,
        )
        pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
        new_bytes = pix.tobytes("png")
        new_w, new_h = pix.width, pix.height
    finally:
        page_doc.close()

    scale_x = width / max(new_w, 1)
    scale_y = height / max(new_h, 1)
    return new_bytes, new_w, new_h, scale_x, scale_y


def build_resized_page_info(
    pages: List[PageImage],
) -> List[Tuple[bytes, str, int, int, float, float]]:
    """对每页做 resize，返回 (img_bytes, fmt, rw, rh, sx, sy) 列表。"""
    result = []
    for page in pages:
        img_bytes, rw, rh, sx, sy = resize_image_for_model(
            page.image_bytes, page.width, page.height
        )
        result.append((img_bytes, page.image_format, rw, rh, sx, sy))
    return result


# ---------------------------------------------------------------------------
# JSON Schema 处理
# ---------------------------------------------------------------------------

def compact_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """精简 JSON Schema，只保留 type/name/description/properties/items。"""
    schema_type = schema.get("type")
    result: Dict[str, Any] = {"type": schema_type}

    name = schema.get("name")
    if name:
        result["name"] = name
    description = schema.get("description")
    if description:
        result["description"] = description

    if schema_type == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            result["properties"] = {
                k: compact_schema(v) for k, v in properties.items() if isinstance(v, dict)
            }
        elif isinstance(schema.get("object"), list):
            children = {}
            for item in schema["object"]:
                if isinstance(item, dict) and item.get("name"):
                    children[str(item["name"])] = compact_schema(item)
            result["properties"] = children
        return result

    if schema_type == "array":
        if isinstance(schema.get("items"), dict):
            result["items"] = compact_schema(schema["items"])
        elif isinstance(schema.get("object"), list):
            children = {}
            for item in schema["object"]:
                if isinstance(item, dict) and item.get("name"):
                    children[str(item["name"])] = compact_schema(item)
            result["items"] = {"type": "object", "properties": children}
        return result

    return result


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------

def get_json_text(content: Any) -> str:
    """从模型响应中提取文本内容。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def try_parse_json(text: str) -> Dict[str, Any]:
    """安全解析 JSON，支持 fenced code block 和宽松匹配。"""
    text = text.strip()
    if not text:
        raise ValueError("模型返回为空")

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        obj = json.loads(match.group(0))

    if not isinstance(obj, dict):
        raise ValueError("模型返回不是 JSON 对象")
    return obj


# ---------------------------------------------------------------------------
# OCR 结果解析
# ---------------------------------------------------------------------------

def parse_ocr_response(raw_text: str, page: PageImage) -> List[OCRItem]:
    """用 OCR_BOX_PATTERN 解析 OCR 输出文本，返回 OCRItem 列表。

    坐标从 0-1000 归一化空间转换为原图像素坐标。
    """
    items: List[OCRItem] = []
    for m in OCR_BOX_PATTERN.finditer(raw_text):
        text = m.group("text").strip()
        x1 = float(m.group("x1")) / COORD_NORM * page.width
        y1 = float(m.group("y1")) / COORD_NORM * page.height
        x2 = float(m.group("x2")) / COORD_NORM * page.width
        y2 = float(m.group("y2")) / COORD_NORM * page.height
        items.append(OCRItem(text=text, bbox=[x1, y1, x2, y2], page_index=page.page_index))
    return items


# ---------------------------------------------------------------------------
# ocrIndices 分离
# ---------------------------------------------------------------------------

def split_ocr_indices(
    raw: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
    """从模型返回的结果中分离 ocrIndices。

    Returns:
        (extract_output, match_result)
        - extract_output: 去掉 ocrIndices 的标准提取结果
        - match_result: Dict[field_path, List[int]]
    """
    output = copy.deepcopy(raw)
    result_node = output.get("result")
    if not isinstance(result_node, dict):
        result_node = output

    match_result: Dict[str, List[int]] = {}
    _extract_ocr_indices(result_node, [], match_result)

    return output, match_result


def _extract_ocr_indices(
    node: Any,
    path_tokens: List[str],
    match_result: Dict[str, List[int]],
) -> None:
    """递归遍历结果树，提取并移除 ocrIndices 字段。"""
    if node is None:
        return
    if isinstance(node, dict):
        is_leaf = (
            {"value", "pageIndex", "sourceText"}.issubset(node.keys())
            and node.get("value") is not None
            and str(node.get("value", "")).strip()
        )
        if is_leaf:
            field_key = ".".join(path_tokens)
            indices = node.pop("ocrIndices", None)
            if isinstance(indices, list):
                match_result[field_key] = [
                    int(i) for i in indices if isinstance(i, (int, float))
                ]
            return
        for key, v in node.items():
            _extract_ocr_indices(v, path_tokens + [key], match_result)
        return
    if isinstance(node, list):
        for i, item in enumerate(node):
            _extract_ocr_indices(item, path_tokens + [str(i)], match_result)


# ---------------------------------------------------------------------------
# 字段收集（用于溯源）
# ---------------------------------------------------------------------------

def collect_fields_for_grounding(
    node: Any, path_tokens: List[str], out: List[Dict[str, Any]]
) -> None:
    """递归收集非空叶子字段，用于后续 bbox 溯源。"""
    if node is None:
        return
    if isinstance(node, dict):
        if {"value", "pageIndex", "sourceText"}.issubset(node.keys()) and "bbox" not in node:
            val = node.get("value")
            if val is not None and str(val).strip():
                out.append({
                    "field": ".".join(path_tokens),
                    "value": val,
                    "pageIndex": node.get("pageIndex"),
                    "sourceText": node.get("sourceText", ""),
                })
            return
        if {"value", "pageIndex", "bbox", "sourceText"}.issubset(node.keys()):
            val = node.get("value")
            if val is not None and str(val).strip():
                out.append({
                    "field": ".".join(path_tokens),
                    "value": val,
                    "pageIndex": node.get("pageIndex"),
                    "sourceText": node.get("sourceText", ""),
                })
            return
        for key, v in node.items():
            collect_fields_for_grounding(v, path_tokens + [key], out)
        return
    if isinstance(node, list):
        for i, item in enumerate(node):
            collect_fields_for_grounding(item, path_tokens + [str(i)], out)


# ---------------------------------------------------------------------------
# bbox 合并
# ---------------------------------------------------------------------------

def merge_nearby_boxes(
    boxes: List[List[float]], gap: float = MERGE_GAP
) -> List[List[float]]:
    """合并 Y 方向相近的框（同字段多行文本）。"""
    if len(boxes) <= 1:
        return boxes
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[List[float]] = []
    cur = list(sorted_boxes[0])
    for b in sorted_boxes[1:]:
        if b[1] <= cur[3] + gap:
            cur[0] = min(cur[0], b[0])
            cur[1] = min(cur[1], b[1])
            cur[2] = max(cur[2], b[2])
            cur[3] = max(cur[3], b[3])
        else:
            merged.append(cur)
            cur = list(b)
    merged.append(cur)
    return merged


# ---------------------------------------------------------------------------
# 从 match 结果构建 bbox_map
# ---------------------------------------------------------------------------

def build_bbox_from_match(
    fields_to_ground: List[Dict[str, Any]],
    ocr_items: List[OCRItem],
    match_result: Dict[str, List[int]],
) -> Dict[str, List[float]]:
    """从 match 结果构建 bbox_map: field_path -> 合并后的像素 bbox。

    逻辑与 llm_rule_extract_visualize.build_visualization_from_match 一致。
    """
    field_boxes: Dict[str, List[List[float]]] = defaultdict(list)
    field_info: Dict[str, Dict[str, Any]] = {f["field"]: f for f in fields_to_ground}

    # 规范化 match_result 的 key（LLM 可能把 "." 输出成 "。" 等）
    normalized_match: Dict[str, List[int]] = {}
    for key, indices in match_result.items():
        nk = re.sub(r'[。，；：、！？（）\s]', '', key).replace(".", "")
        normalized_match[nk] = indices

    # 规范化 field_info 的 key
    norm_field_info: Dict[str, Dict[str, Any]] = {}
    for f in fields_to_ground:
        nk = re.sub(r'[。，；：、！？（）\s]', '', f["field"]).replace(".", "")
        norm_field_info[nk] = f

    for norm_key, indices in normalized_match.items():
        info = norm_field_info.get(norm_key)
        if not info:
            continue
        original_field = info["field"]
        for idx in indices:
            if 0 <= idx < len(ocr_items):
                field_boxes[original_field].append(ocr_items[idx].bbox)

    bbox_map: Dict[str, List[float]] = {}
    for field_path, boxes in field_boxes.items():
        if not boxes:
            continue
        merged = merge_nearby_boxes(boxes)
        if len(merged) == 1:
            bbox_map[field_path] = merged[0]
        else:
            x1 = min(b[0] for b in merged)
            y1 = min(b[1] for b in merged)
            x2 = max(b[2] for b in merged)
            y2 = max(b[3] for b in merged)
            bbox_map[field_path] = [x1, y1, x2, y2]

    return bbox_map


# ---------------------------------------------------------------------------
# 注入 bbox 到结果树
# ---------------------------------------------------------------------------

def inject_bboxes_into_result(
    node: Any, path_tokens: List[str], bbox_map: Dict[str, List[float]]
) -> None:
    """将 bbox_map 中的坐标注入到提取结果树的叶子节点。"""
    if node is None:
        return
    if isinstance(node, dict):
        is_leaf = (
            {"value", "pageIndex", "sourceText"}.issubset(node.keys())
            and (node.get("value") is not None and str(node.get("value", "")).strip())
        )
        if is_leaf:
            field_key = ".".join(path_tokens)
            if field_key in bbox_map:
                node["bbox"] = bbox_map[field_key]
            else:
                node["bbox"] = None
            return
        for key, v in node.items():
            inject_bboxes_into_result(v, path_tokens + [key], bbox_map)
        return
    if isinstance(node, list):
        for i, item in enumerate(node):
            inject_bboxes_into_result(item, path_tokens + [str(i)], bbox_map)


# ---------------------------------------------------------------------------
# 合并叶子值（去掉 metadata 包装）
# ---------------------------------------------------------------------------

def merge_leaf_values(node: Any) -> Any:
    """递归提取叶子节点的 value，去掉 pageIndex/bbox/sourceText 包装。"""
    if node is None:
        return None
    if isinstance(node, dict):
        if {"value", "pageIndex", "bbox", "sourceText"}.issubset(node.keys()):
            return node.get("value")
        return {k: merge_leaf_values(v) for k, v in node.items()}
    if isinstance(node, list):
        return [merge_leaf_values(item) for item in node]
    return node


# ---------------------------------------------------------------------------
# 方向回正
# ---------------------------------------------------------------------------

def apply_orientation_to_pages(
    pages: List[PageImage], angles: List[int]
) -> List[PageImage]:
    """对每页按角度回正，返回新的 PageImage 列表（宽高随旋转更新）。

    angle 为 0 时直接复用原对象，非 0 时用 PIL rotate(angle, expand=True) 回正。
    """
    rotated: List[PageImage] = []
    for page, angle in zip(pages, angles):
        if angle == 0:
            rotated.append(page)
            continue
        with PILImage.open(io.BytesIO(page.image_bytes)) as im:
            im_rot = im.rotate(angle, expand=True, fillcolor=(255, 255, 255))
            buf = io.BytesIO()
            im_rot.save(buf, format="PNG")
            rotated.append(PageImage(
                page_index=page.page_index,
                image_format="png",
                image_bytes=buf.getvalue(),
                width=im_rot.width,
                height=im_rot.height,
            ))
    return rotated


def bbox_rotated_to_original(
    bbox: List[float], angle: int, orig_w: int, orig_h: int
) -> List[float]:
    """将回正图上的像素 bbox 逆旋转回原图坐标。

    对应 PIL Image.rotate(angle, expand=True) 的逆变换：
      90°:  inv(px, py) = (orig_w-1-py, px)
      180°: inv(px, py) = (orig_w-1-px, orig_h-1-py)
      270°: inv(px, py) = (py, orig_h-1-px)
    对 4 个角点分别变换后取 min/max。
    """
    if angle == 0:
        return bbox
    x1, y1, x2, y2 = bbox
    if angle == 90:
        def inv(px: float, py: float) -> Tuple[float, float]:
            return (orig_w - 1 - py, px)
    elif angle == 180:
        def inv(px: float, py: float) -> Tuple[float, float]:
            return (orig_w - 1 - px, orig_h - 1 - py)
    elif angle == 270:
        def inv(px: float, py: float) -> Tuple[float, float]:
            return (py, orig_h - 1 - px)
    else:
        return bbox
    corners = [inv(x1, y1), inv(x2, y1), inv(x2, y2), inv(x1, y2)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return [min(xs), min(ys), max(xs), max(ys)]
