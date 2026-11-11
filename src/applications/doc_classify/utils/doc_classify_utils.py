"""文档分类工具函数。"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import fitz
from loguru import logger
from PIL import Image

def thumbnail_from_png_bytes(
    png_bytes: bytes,
    thumb_size: str,
) -> Optional[str]:
    """从已转换的 PNG 字节生成缩略图并返回 base64 data-uri。"""
    try:
        parts = thumb_size.split(",")
        if len(parts) != 2:
            logger.warning("thumb 格式错误，期望 '宽,高' | thumb={}", thumb_size)
            return None
        thumb_w, thumb_h = int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, AttributeError) as exc:
        logger.warning("解析 thumb 尺寸失败 | error={}", exc)
        return None

    try:
        img = Image.open(io.BytesIO(png_bytes))
        img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        logger.exception("从 PNG 字节生成缩略图失败")
        return None


def generate_thumbnail_base64(
    file_content: bytes,
    filename: str,
    thumb_size: str,
) -> Optional[str]:
    """对文件内容生成缩略图并返回 base64 data-uri。"""
    try:
        parts = thumb_size.split(",")
        if len(parts) != 2:
            logger.warning("thumb 格式错误，期望 '宽,高' | thumb={}", thumb_size)
            return None
        thumb_w, thumb_h = int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, AttributeError) as exc:
        logger.warning("解析 thumb 尺寸失败 | error={}", exc)
        return None

    file_ext = Path(filename).suffix.lower()

    try:
        if file_ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            img = Image.open(io.BytesIO(file_content))
        elif file_ext == ".pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            try:
                doc = fitz.open(tmp_path)
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                doc.close()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
            finally:
                os.remove(tmp_path)
        elif file_ext in (".xls", ".xlsx", ".doc", ".docx"):
            from src.utils.document_render import convert_to_png
            import shutil

            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            try:
                png_paths = convert_to_png(tmp_path)
                if not png_paths:
                    logger.warning("generate_thumbnail_base64: 转换后无图片 | ext={}", file_ext)
                    return None
                png_dir = os.path.dirname(png_paths[0])
                img = Image.open(png_paths[0])
            finally:
                os.remove(tmp_path)
                if "png_dir" in dir():
                    shutil.rmtree(png_dir, ignore_errors=True)
        else:
            logger.warning("generate_thumbnail_base64: 不支持的文件类型 | ext={}", file_ext)
            return None

        img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        logger.exception("生成缩略图失败 | filename={}", filename)
        return None


__all__ = [
    "generate_thumbnail_base64",
    "thumbnail_from_png_bytes",
]
