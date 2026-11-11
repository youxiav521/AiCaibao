from __future__ import annotations

import base64
import io
import mimetypes
from typing import Any, Mapping, Optional, Sequence

from loguru import logger
from PIL import Image

from src.agents.content_generate_agent.utils import ainvoke_llm_with_messages
from src.applications.parsing.utils.mineru_storage import (
    build_mineru_image_object_name,
    download_mineru_image,
)
from src.applications.rag.schemas import (
    ChildChunk,
    ChunkFields,
    GrandchildChunk,
)

from ..utils.chunking_helpers import hash_id
import asyncio


MAX_DATA_URI_BYTES = 10 * 1024 * 1024
DEFAULT_TARGET_IMAGE_BYTES = 7 * 1024 * 1024
_MIN_IMAGE_SIDE = 768
_IMAGE_SUMMARY_PROMPT = """# Role
你是一个专业的 RAG (检索增强生成) 系统索引专家，擅长将多模态信息转化为高价值、低噪音的检索文本。

# Task
请对上传的图片进行深度解析，并生成一段专为“向量检索”优化的摘要。

# Constraints (严格遵守)
1. 禁止废话：严禁使用“这是一张...”、“图片显示了...”、“图中包含...”等引导词。
2. 高密度词汇：直接输出核心事实、关键词和实体信息。
3. 结构化输出：按照以下三个维度进行总结，但不要输出标题，直接将内容融合成一段。
4. 语言要求：使用中文。

# Content Dimensions
1. 文本与数字 (OCR)：精准提取图片中出现的机构名称、日期、金额、单据编号等核心文本。
2. 核心实体与属性：明确图片类别及其核心特征。
3. 意图与状态：描述该图片代表的业务动作或事实状态。

# Example
[北京市海淀区市场监督管理局]行政公章，红色圆形，中心五角星。日期：2021年08月12日。行政审批/文件认证状态。用途：政府机关公文效力凭证、业务准入证明。关键词：海淀区、市监局、盖章认证、20210812。

# Start Processing
请根据上述规则处理当前图片："""


def compress_image_bytes_for_llm(
    image_bytes: bytes,
    *,
    target_image_bytes: int = DEFAULT_TARGET_IMAGE_BYTES,
) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

        working = image.copy()
        scale = 1.0

        for _ in range(8):
            buffer = io.BytesIO()
            working.save(buffer, format="PNG", optimize=True)
            current_bytes = buffer.getvalue()
            if len(current_bytes) <= target_image_bytes:
                return current_bytes

            width, height = working.size
            if min(width, height) <= _MIN_IMAGE_SIDE:
                return current_bytes

            scale *= 0.85
            new_width = max(int(image.width * scale), _MIN_IMAGE_SIDE)
            new_height = max(int(image.height * scale), _MIN_IMAGE_SIDE)
            if (new_width, new_height) == working.size:
                return current_bytes
            working = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        working.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        texts = [str(item or "").strip() for item in value]
        return [text for text in texts if text]
    text = str(value or "").strip()
    return [text] if text else []


def build_image_fallback_text(item: Mapping[str, Any] | None) -> str:
    source = dict(item or {})
    texts = _normalize_text_list(source.get("image_caption"))
    texts.extend(_normalize_text_list(source.get("image_footnote")))
    return "\n".join(texts).strip()


async def download_image_bytes_from_storage(
    img_path: str | None,
    *,
    file_id: str,
) -> Optional[bytes]:
    """从 MinIO 下载 MinerU 图片，返回原始字节。

    下载失败时记录 warning 日志并返回 None，不向上抛出异常。
    """
    normalized = str(img_path or "").strip()
    if not normalized:
        return None
    try:
        return await download_mineru_image(file_id, normalized)
    except Exception as exc:
        logger.warning(
            "从 MinIO 下载 MinerU 图片失败: file_id={}, img_path={}, error={}",
            file_id, normalized, exc,
        )
        return None


def resolve_image_storage_path(
    img_path: str | None,
    *,
    file_id: str,
) -> Optional[str]:
    """解析图片在 MinIO 存储中的对象路径。

    返回 ``"{file_id}/{img_path}"`` 格式的对象名，供存储后端定位图片。
    """
    normalized = str(img_path or "").strip()
    if not normalized:
        return None
    return build_mineru_image_object_name(file_id, normalized)


def build_image_data_uri(
    image_bytes: bytes,
    *,
    mime_type: str = "image/png",
    max_data_uri_bytes: int = MAX_DATA_URI_BYTES,
    target_image_bytes: int = DEFAULT_TARGET_IMAGE_BYTES,
) -> str:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    if len(image_base64) <= max_data_uri_bytes:
        return f"data:{mime_type};base64,{image_base64}"

    compressed_bytes = compress_image_bytes_for_llm(
        image_bytes=image_bytes,
        target_image_bytes=target_image_bytes,
    )
    compressed_base64 = base64.b64encode(compressed_bytes).decode("utf-8")
    if len(compressed_base64) > max_data_uri_bytes:
        raise ValueError(
            f"图片压缩后仍超过模型限制，base64大小={len(compressed_base64)}，限制={max_data_uri_bytes}"
        )
    return f"data:image/png;base64,{compressed_base64}"


async def generate_image_summary(image_bytes: bytes, mime_type: str = "image/png") -> str:
    data_uri = build_image_data_uri(image_bytes, mime_type=mime_type)
    messages = [
        (
            "user",
            [
                {"type": "text", "text": _IMAGE_SUMMARY_PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        )
    ]
    return str(
        await ainvoke_llm_with_messages(messages=messages, model_id="image_summary")
        or ""
    ).strip()


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def _coerce_sort_number(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _representative_child_sort_key(
    child: ChildChunk,
) -> tuple[str]:
    child_id = child.child_id
    return (child_id,)


def _build_candidate_children_by_element_id(
    children: Sequence[ChildChunk],
) -> dict[str, list[ChildChunk]]:
    candidate_children: dict[str, list[ChildChunk]] = {}
    for child in children or []:
        source_element_ids = child.source_element_ids
        for source_element_id in source_element_ids:
            normalized_element_id = _coerce_int(source_element_id)
            if normalized_element_id is None:
                continue
            candidate_children.setdefault(str(normalized_element_id), []).append(child)
    return candidate_children


async def build_image_summary_grandchildren(
    items: Sequence[Mapping[str, Any]],
    *,
    children: Sequence[ChildChunk],
    file_id: str,
    folder_id: str | None = None,
    file_name: str | None = None,
) -> list[GrandchildChunk]:
    if not items:
        return []

    grandchildren: list[GrandchildChunk] = []
    candidate_children_by_element_id = _build_candidate_children_by_element_id(children)

    async def build_grandchild(item):
        if (
            not isinstance(item, Mapping)
            or str(item.get("type") or "").strip().lower() != "image"
        ):
            return None

        element_id = _coerce_int(item.get("element_id"))
        if element_id is None:
            logger.warning(
                "图片元素缺少合法 element_id，已跳过: file_id={}, item={}",
                file_id,
                dict(item),
            )
            return None

        candidate_children = candidate_children_by_element_id.get(str(element_id)) or []
        if not candidate_children:
            logger.warning(
                "图片元素未找到可绑定的 child，已跳过: file_id={}, element_id={}",
                file_id,
                element_id,
            )
            return None

        representative_child = sorted(candidate_children, key=_representative_child_sort_key)[0]
        child_metadata = representative_child
        child_id = representative_child.child_id.strip()
        parent_id = representative_child.parent_id.strip()
        if not child_id:
            logger.warning(
                "图片元素绑定的 child 缺少 child_id，已跳过: file_id={}, element_id={}",
                file_id,
                element_id,
            )
            return None

        raw_img_path = str(item.get("img_path") or "").strip()
        image_storage_path = resolve_image_storage_path(raw_img_path, file_id=file_id)
        image_bytes = await download_image_bytes_from_storage(
            raw_img_path,
            file_id=file_id,
        )
        summary_text = ""

        if image_bytes is not None:
            try:
                mime_type, _ = mimetypes.guess_type(raw_img_path)
                if not mime_type:
                    mime_type = "image/png"
                summary_text = await generate_image_summary(
                    image_bytes,
                    mime_type=mime_type,
                )
            except Exception as exc:
                logger.warning(
                    "图片摘要生成失败，准备回退到 caption/footnote: file_id={}, element_id={}, error={}",
                    file_id,
                    element_id,
                    exc,
                )
        else:
            logger.warning(
                "MinIO 中未找到图片对象，准备回退到 caption/footnote: file_id={}, element_id={}, img_path={}",
                file_id,
                element_id,
                raw_img_path,
            )

        if not summary_text:
            summary_text = build_image_fallback_text(item)
        if not summary_text:
            logger.warning(
                "图片既无摘要也无可用回退文本，已跳过: file_id={}, element_id={}",
                file_id,
                element_id,
            )
            return None

        grandchild_seed = "|".join(
            [
                str(file_id),
                "image",
                str(element_id),
                child_id,
                summary_text or raw_img_path,
            ]
        )
        grandchild_id = f"{file_id}:g:{hash_id(grandchild_seed, length=16)}:{child_id}"
        grandchild = GrandchildChunk(
            grandchild_id=grandchild_id,
            parent_id=parent_id,
            child_id=child_id,
            text=summary_text,
            file_id=str(file_id),
            folder_id=str(folder_id or child_metadata.folder_id or ""),
            file_name=str(file_name or child_metadata.file_name or ""),
            type="image",
            element_id=element_id,
            source_element_ids=[element_id],
            image_storage_path=image_storage_path,
        )
        grandchildren.append(grandchild)
        return grandchild

    
    tasks = [build_grandchild(item) for item in items]
    await asyncio.gather(*tasks)
    
    return grandchildren
