"""通用文档转图片与多模态图片输入处理工具。"""
from __future__ import annotations

import base64
import io
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import fitz
from loguru import logger
from PIL import Image

from src.utils.soffice import convert_to_pdf

# Qwen-VL 系列视觉编码：每 28x28 像素合并为 1 个视觉 token
VISION_PATCH_SIZE = 28
# 分类：单图视觉 token 预算，超出则等比缩放
CLASSIFY_MAX_VISION_TOKENS = 1536
# 提取：单图视觉 token 预算，超出则等比缩放
EXTRACT_MAX_VISION_TOKENS = 3072


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def estimate_vision_tokens(width: int, height: int, patch: int = VISION_PATCH_SIZE) -> int:
    """按 Qwen-VL patch=28 公式估算图片的视觉 token 数。"""
    if width <= 0 or height <= 0:
        return 0
    return math.ceil(width / patch) * math.ceil(height / patch)


def build_image_data_uri(
    image_path: str,
    max_vision_tokens: int = EXTRACT_MAX_VISION_TOKENS,
) -> str:
    """构建用于多模态模型调用的 data-uri。

    按视觉 token 预算等比缩放图片：token 未超预算则原样编码为 PNG；
    超出则缩到目标像素面积以内并编码为 JPEG（更省字节，token 由像素决定不受影响）。
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    compressed_bytes, mime_type = resize_image_for_token_budget(
        image_bytes=image_bytes,
        max_vision_tokens=max_vision_tokens,
    )
    b64 = base64.b64encode(compressed_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def resize_image_for_token_budget(
    image_bytes: bytes,
    max_vision_tokens: int = EXTRACT_MAX_VISION_TOKENS,
    patch: int = VISION_PATCH_SIZE,
) -> tuple[bytes, str]:
    """按视觉 token 预算等比缩放图片，返回 (字节, mime_type)。

    - token 未超预算：直接返回原始字节，跳过解码+重编码
    - 超出预算：缩到 max_vision_tokens * patch^2 像素面积以内（宽高对齐到 patch），
      编码为 JPEG（q=85）以减小传输字节
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size
        cur_tokens = estimate_vision_tokens(width, height, patch)

        if cur_tokens <= max_vision_tokens:
            fmt = (img.format or "PNG").lower()
            return image_bytes, f"image/{fmt}"

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        target_pixels = max_vision_tokens * patch * patch
        scale = (target_pixels / (width * height)) ** 0.5
        new_width = max(int(width * scale) // patch * patch, patch)
        new_height = max(int(height * scale) // patch * patch, patch)

        resized = img.resize((new_width, new_height), Image.BILINEAR)
        if resized.mode == "RGBA":
            background = Image.new("RGB", resized.size, (255, 255, 255))
            background.paste(resized, mask=resized.split()[3])
            resized = background

        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=85)

    new_tokens = estimate_vision_tokens(new_width, new_height, patch)
    logger.info(
        "按 token 预算缩放图片 | orig={}x{}({}tok) -> new={}x{}({}tok), budget={}",
        width, height, cur_tokens, new_width, new_height, new_tokens, max_vision_tokens,
    )
    return buffer.getvalue(), "image/jpeg"


def convert_to_png(
    input_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 200,
    max_pages: int = 0,
    start_page: int = 0,
) -> List[str]:
    """将 PDF、图片、Word、Excel 等文档转换为 PNG 图片。

    Args:
        max_pages: 最多转换的页数，0 表示不限制。
        start_page: 起始页码（0-based），用于分批转换。仅 PDF 有效。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    input_path_obj = Path(input_path)
    file_ext = input_path_obj.suffix.lower()

    if output_dir is None:
        unique_id = uuid.uuid4().hex[:8]
        output_dir = os.path.join(os.path.dirname(input_path), f"converted_images_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)

    base_name = input_path_obj.stem

    try:
        if file_ext == ".pdf":
            logger.info("开始转换PDF文件 | path={}, start={}, max_pages={}", input_path, start_page, max_pages)
            png_files = _pdf_to_png(input_path, output_dir, base_name, dpi, max_pages, start_page)
        elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            logger.info("开始转换图片文件 | path={}", input_path)
            png_files = _image_to_png(input_path, output_dir, base_name)
        elif file_ext in [".doc", ".docx"]:
            logger.info("开始转换Word文档 | path={}, start={}, max_pages={}", input_path, start_page, max_pages)
            png_files = _word_to_png(input_path, output_dir, base_name, dpi, max_pages, start_page)
        elif file_ext in [".xls", ".xlsx"]:
            logger.info("开始转换Excel文件 | path={}, max_pages={}", input_path, max_pages)
            png_files = _excel_to_png(input_path, output_dir, base_name, dpi, max_pages)
        else:
            raise ValueError(f"不支持的文件类型: {file_ext}")

        logger.info("文件转换完成 | count={}", len(png_files))
        return png_files
    except Exception:
        logger.exception("文件转换失败 | path={}", input_path)
        raise


def _pdf_to_png(
    pdf_path: str,
    output_dir: str,
    base_name: str,
    dpi: int,
    max_pages: int = 0,
    start_page: int = 0,
) -> List[str]:
    png_files = []

    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        start = min(start_page, total_pages)
        end = total_pages if max_pages <= 0 else min(total_pages, start + max_pages)
        for page_num in range(start, end):
            page = pdf_document[page_num]
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            output_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.png")
            pix.save(output_path)
            png_files.append(output_path)
        if end < total_pages:
            logger.info("PDF分批转换 | total={}, rendered=[{},{}), remaining={}", total_pages, start, end, total_pages - end)
        pdf_document.close()
    except Exception:
        logger.exception("PDF转换失败 | path={}", pdf_path)
        raise

    return png_files


def _image_to_png(
    image_path: str,
    output_dir: str,
    base_name: str,
) -> List[str]:
    try:
        img = Image.open(image_path)
        output_path = os.path.join(output_dir, f"{base_name}.png")

        if img.mode in ("RGBA", "LA", "P"):
            img_rgb = img.convert("RGBA")
        else:
            img_rgb = img.convert("RGB")

        img_rgb.save(output_path, "PNG")
        return [output_path]
    except Exception:
        logger.exception("图片转换失败 | path={}", image_path)
        raise


def _office_to_pdf(src_path: str) -> str:
    """soffice 将 Office 文档（Word/Excel）转为 PDF，返回临时 PDF 路径（调用方负责清理）。

    实际转换逻辑（soffice 探测、重试、超时、UserInstallation 隔离）由
    src.utils.soffice.convert_to_pdf 提供；这里负责把生成的 PDF 移到独立的
    临时文件，使调用方仍按“清理单个文件”的契约工作。
    """
    tmp_dir = tempfile.mkdtemp(prefix="office_to_pdf_")
    try:
        pdf_path = convert_to_pdf(src_path, outdir=tmp_dir)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf_path = tmp.name
        shutil.move(str(pdf_path), tmp_pdf_path)
        return tmp_pdf_path
    except Exception:
        logger.exception("Office文档转PDF失败 | path={}", src_path)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _word_to_png(
    word_path: str,
    output_dir: str,
    base_name: str,
    dpi: int,
    max_pages: int = 0,
    start_page: int = 0,
) -> List[str]:
    tmp_pdf_path = _office_to_pdf(word_path)
    try:
        return _pdf_to_png(tmp_pdf_path, output_dir, base_name, dpi, max_pages, start_page)
    finally:
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            try:
                os.remove(tmp_pdf_path)
            except Exception:
                logger.exception("清理临时PDF失败 | path={}", tmp_pdf_path)


def _excel_to_png(
    excel_path: str,
    output_dir: str,
    base_name: str,
    dpi: int,
    max_pages: int = 0,
) -> List[str]:
    """通过 soffice 将 Excel 转为 PDF，再渲染为 PNG，与 Word 路径保持一致。

    相比原先基于 matplotlib 的表格重绘，soffice 能保留原始样式、合并单元格、
    列宽行高与字体，渲染更准确，且无需操心中文字体安装。
    """
    try:
        # soffice 按扩展名识别格式，扩展名与实际不一致时先复制到正确后缀的临时文件
        actual_format = _detect_excel_format(excel_path)
        parse_path, parse_path_is_temp = _ensure_excel_ext_matches(excel_path, actual_format)
        try:
            tmp_pdf_path = _office_to_pdf(parse_path)
        finally:
            if parse_path_is_temp:
                try:
                    os.remove(parse_path)
                except OSError:
                    logger.warning("清理 Excel 临时文件失败 | path={}", parse_path)
        try:
            return _pdf_to_png(tmp_pdf_path, output_dir, base_name, dpi, max_pages, start_page=0)
        finally:
            if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                try:
                    os.remove(tmp_pdf_path)
                except Exception:
                    logger.exception("清理临时PDF失败 | path={}", tmp_pdf_path)
    except Exception:
        logger.exception("Excel转换失败 | path={}", excel_path)
        raise


def _detect_excel_format(excel_path: str) -> str:
    """通过文件头判断 Excel 真实格式，避免扩展名与内容不一致导致的解析错误。"""
    try:
        with open(excel_path, "rb") as f:
            header = f.read(8)
    except OSError:
        logger.exception("读取 Excel 文件头失败 | path={}", excel_path)
        raise

    if header.startswith(b"PK\x03\x04"):
        return "xlsx"
    if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "xls"

    # 兜底：按扩展名推断
    ext = excel_path.lower()
    if ext.endswith(".xlsx"):
        return "xlsx"
    if ext.endswith(".xls"):
        return "xls"
    return "unknown"


def _ensure_excel_ext_matches(excel_path: str, actual_format: str) -> tuple[str, bool]:
    """若文件扩展名与真实格式不一致，复制一份到正确扩展名的临时文件并返回 (path, is_temp)。"""
    if actual_format not in {"xls", "xlsx"}:
        return excel_path, False

    current_ext = os.path.splitext(excel_path)[1].lower()
    expected_ext = f".{actual_format}"
    if current_ext == expected_ext:
        return excel_path, False

    with tempfile.NamedTemporaryFile(suffix=expected_ext, delete=False) as tmp:
        tmp_path = tmp.name
    with open(excel_path, "rb") as src, open(tmp_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    logger.info(
        "Excel 扩展名与实际格式不一致，复制到临时文件 | original={}, actual={}, temp={}",
        excel_path,
        actual_format,
        tmp_path,
    )
    return tmp_path, True
