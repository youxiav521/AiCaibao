"""
MinerU 产物访问层 — 统一的 MinIO 读写入口

所有 MinerU 解析产物（converted.json 与图片）的存储路径拼接与上传/下载逻辑集中在此模块，
上层业务代码不应自行拼接 object_name、调用 storage_service 或依赖本地目录。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.base.storage_service import (
    StorageError,
    StorageObjectNotFoundError,
    storage_service,
)
from src.utils import json_utils

# MinerU 解析产物所在的 MinIO 桶名
MINERU_OUTPUT_BUCKET = os.getenv("MINERU_OUTPUT_BUCKET", "mineru-output")


def build_mineru_converted_object_name(file_id: str) -> str:
    """构建 converted.json 在 MinIO 中的对象名。

    参数：
        file_id: 文件 ID。

    返回：
        MinIO 对象名字符串，如 ``"abc123/converted.json"``。
    """
    return f"{file_id}/converted.json"


def build_mineru_image_object_name(file_id: str, img_path: str) -> str:
    """构建 MinerU 图片在 MinIO 中的对象名。

    ``img_path`` 是 MinerU 原始产物中图片的路径值（如 ``"images/xxx.jpg"``），
    调用方直接传入该字段即可。

    参数：
        file_id:  文件 ID。
        img_path: MinerU 输出的图片相对路径（如 ``"images/abc.jpg"``）。

    返回：
        MinIO 对象名字符串，如 ``"abc123/images/abc.jpg"``。
    """
    return f"{file_id}/{img_path}"


def build_mineru_result_object_name(file_id: str) -> str:
    return f"{file_id}/mineru.json"


def build_mineru_original_object_name(file_id: str, filename: str) -> str:
    safe_name = Path(filename).name or "unknown"
    return f"{file_id}/original/{safe_name}"


# ---------------------------------------------------------------------------
# 上传助手 — 封装 storage_service.upload()，避免上层直接操作 MinIO
# ---------------------------------------------------------------------------

async def upload_mineru_items(file_id: str, items: Any) -> None:
    """将 MinerU 解析后的 items 列表上传为 ``{file_id}/converted.json``。"""
    await storage_service.upload(
        MINERU_OUTPUT_BUCKET,
        build_mineru_converted_object_name(file_id),
        json_utils.dumps_bytes(items, ensure_ascii=False),
        "application/json",
    )


async def upload_mineru_result(file_id: str, mineru_result: Any) -> None:
    """将 MinerU 兼容结果结构上传为 ``{file_id}/mineru.json``。"""
    await storage_service.upload(
        MINERU_OUTPUT_BUCKET,
        build_mineru_result_object_name(file_id),
        json_utils.dumps_bytes(mineru_result, ensure_ascii=False),
        "application/json",
    )


async def upload_mineru_image(file_id: str, img_path: str, image_bytes: bytes, content_type: str = "application/octet-stream") -> None:
    """将单张 MinerU 解析图片上传到 ``{file_id}/{img_path}``。"""
    await storage_service.upload(
        MINERU_OUTPUT_BUCKET,
        build_mineru_image_object_name(file_id, img_path),
        image_bytes,
        content_type,
    )


async def upload_original_file(
    file_id: str,
    filename: str,
    file_bytes: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """将上传原始文件持久化到 MinIO，返回对象名。"""
    return await storage_service.upload(
        MINERU_OUTPUT_BUCKET,
        build_mineru_original_object_name(file_id, filename),
        file_bytes,
        content_type,
    )


# ---------------------------------------------------------------------------
# 下载助手
# ---------------------------------------------------------------------------

async def load_mineru_items(file_id: str) -> list[dict]:
    """从 MinIO 下载 converted.json 并返回解析后的列表。

    抛出：
        FileNotFoundError: 对象不存在。
        StorageError: 存储层下载失败。
        ValueError: JSON 解析失败或内容不是列表结构。
    """
    object_name = build_mineru_converted_object_name(file_id)
    try:
        data = await storage_service.download(MINERU_OUTPUT_BUCKET, object_name)
    except StorageObjectNotFoundError as exc:
        logger.warning(
            "MinIO 中不存在 MinerU 解析产物 | file_id={} bucket={} object={}",
            file_id, MINERU_OUTPUT_BUCKET, object_name,
        )
        raise FileNotFoundError(object_name) from exc
    except StorageError:
        logger.exception(
            "从 MinIO 加载 MinerU 解析产物失败 | file_id={} bucket={} object={}",
            file_id, MINERU_OUTPUT_BUCKET, object_name,
        )
        raise

    try:
        items = json_utils.loads(data.decode("utf-8"))
    except Exception as exc:
        logger.exception(
            "解析 MinerU converted.json 失败 | file_id={} bucket={} object={}",
            file_id, MINERU_OUTPUT_BUCKET, object_name,
        )
        raise ValueError(f"解析 MinerU converted.json 失败: {object_name}") from exc

    if not isinstance(items, list):
        raise ValueError(f"MinerU converted.json 内容不是列表结构: {object_name}")
    return items


async def load_mineru_result(file_id: str) -> dict:
    """从 MinIO 下载兼容 MinerU 结果结构（mineru.json）。

    抛出：
        FileNotFoundError: 对象不存在。
        StorageError: 存储层下载失败。
        ValueError: JSON 解析失败或内容不是对象结构。
    """
    object_name = build_mineru_result_object_name(file_id)
    try:
        data = await storage_service.download(MINERU_OUTPUT_BUCKET, object_name)
    except StorageObjectNotFoundError as exc:
        logger.warning(
            "MinIO 中不存在 MinerU 兼容结果 | file_id={} bucket={} object={}",
            file_id, MINERU_OUTPUT_BUCKET, object_name,
        )
        raise FileNotFoundError(object_name) from exc
    except StorageError:
        logger.exception(
            "从 MinIO 加载 MinerU 兼容结果失败 | file_id={} bucket={} object={}",
            file_id, MINERU_OUTPUT_BUCKET, object_name,
        )
        raise

    try:
        payload = json_utils.loads(data.decode("utf-8"))
    except Exception as exc:
        logger.exception(
            "解析 MinerU 兼容结果失败 | file_id={} bucket={} object={}",
            file_id, MINERU_OUTPUT_BUCKET, object_name,
        )
        raise ValueError(f"解析 MinerU 兼容结果失败: {object_name}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"MinerU 兼容结果不是对象结构: {object_name}")
    return payload


async def download_mineru_image(file_id: str, img_path: str) -> bytes:
    """从 MinIO 下载 MinerU 图片，返回原始字节。

    抛出：
        StorageError / StorageObjectNotFoundError: 下载失败时直接向上抛出。
    """
    object_name = build_mineru_image_object_name(file_id, img_path)
    return await storage_service.download(MINERU_OUTPUT_BUCKET, object_name)


async def load_original_pdf_bytes(file_id: str) -> Optional[bytes]:
    """下载上传时保存的原始 PDF；不存在或不是 PDF 时返回 None。"""
    object_names = await storage_service.list_objects(
        MINERU_OUTPUT_BUCKET,
        prefix=f"{file_id}/original/",
        recursive=True,
    )
    pdf_names = [name for name in object_names if name.lower().endswith(".pdf")]
    if not pdf_names:
        return None
    data = await storage_service.download(MINERU_OUTPUT_BUCKET, pdf_names[0])
    return data if data.startswith(b"%PDF") else None


# ---------------------------------------------------------------------------
# 共享解析工具 — 从 mineru_result 中提取页面尺寸
# ---------------------------------------------------------------------------

def extract_pdf_info_from_mineru_result(
    mineru_result: Any,
    *,
    file_id: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """从 MinerU 兼容结果结构中提取 ``pdf_info``（每页尺寸列表）。

    统一的 ``mineru_result.results.{file_id}.middle_json.pdf_info`` 导航逻辑。
    如果 *file_id* 未传或在 results 中找不到，回退到 ``results`` 的第一个值。

    返回：
        ``pdf_info`` 列表，提取失败时返回 ``None``。
    """
    if not isinstance(mineru_result, dict):
        return None

    results = mineru_result.get("results", {})
    if not isinstance(results, dict) or not results:
        return None

    file_data = results.get(file_id) if file_id else None
    if file_data is None:
        file_data = next(iter(results.values()))
    if not isinstance(file_data, dict):
        return None

    middle_json_str = file_data.get("middle_json")
    if not middle_json_str:
        return None

    try:
        middle_json = (
            json_utils.loads(middle_json_str)
            if isinstance(middle_json_str, str)
            else middle_json_str
        )
    except Exception:
        return None

    if not isinstance(middle_json, dict):
        return None

    pdf_info = middle_json.get("pdf_info")
    return pdf_info if isinstance(pdf_info, list) else None
