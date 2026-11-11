from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.applications.parsing.utils.mineru_storage import (
    MINERU_OUTPUT_BUCKET,
    build_mineru_image_object_name,
    load_mineru_items,
)
from src.base.storage_service import storage_service

from ..schemas import SearchType
from .vectorstore import VectorStoresService

_IMAGE_QUERY_LIMIT = 3
_IMAGE_URL_EXPIRES_S = 300


@dataclass(frozen=True)
class ImageQueryItem:
    file_id: str
    file_name: str
    element_id: int
    page: Optional[int]
    score: int
    image_url: str
    summary: str


@dataclass(frozen=True)
class ImageSearchCandidate:
    """图片语义检索候选项，供 HTTP 图片检索和内容生成图片题复用。"""

    file_id: str
    element_id: int
    image_storage_path: str
    score: float
    file_name: str = ""
    page: Optional[int] = None
    summary: str = ""


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def _score(metadata: dict) -> float:
    try:
        return float(metadata.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


async def _resolve_legacy_image_storage_path(
    file_id: str,
    element_id: int,
    *,
    path_index_by_file: dict[str, dict[int, str]],
) -> str:
    """为旧向量索引按 MinerU ``converted.json`` 回查图片对象路径。"""
    if file_id not in path_index_by_file:
        try:
            items = await load_mineru_items(file_id)
        except Exception as exc:
            logger.warning("加载 MinerU 图片路径失败，跳过旧索引兜底 | file_id={} error={}", file_id, exc)
            path_index_by_file[file_id] = {}
        else:
            path_index_by_file[file_id] = {
                item_element_id: build_mineru_image_object_name(file_id, img_path)
                for item in items
                if str(item.get("type") or "").strip().lower() == "image"
                and (item_element_id := _as_int(item.get("element_id"))) is not None
                and (img_path := str(item.get("img_path") or "").strip())
            }
    return path_index_by_file[file_id].get(element_id, "")


async def search_image_candidates(
    question: str,
    *,
    top_k: int,
    file_ids: Optional[list[str]] = None,
    folder_id: Optional[str] = None,
) -> list[ImageSearchCandidate]:
    """语义检索并按 ``(file_id, element_id)`` 去重后的图片候选项。"""
    limit = max(1, int(top_k))
    retriever = VectorStoresService.get_grandchildren_retriever(
        top_k=limit,
        file_ids=file_ids,
        folder_id=folder_id,
        search_type=SearchType.IMAGE.value,
        # 图片摘要任务直接写入向量库，历史图片记录的 doc_kind 可能为空。
        doc_kinds=["chunk", ""],
    )
    documents = await retriever.aget_relevant_documents(question)
    best_by_image: dict[tuple[str, int], tuple[object, dict]] = {}
    path_index_by_file: dict[str, dict[int, str]] = {}
    for document in documents:
        metadata = dict(getattr(document, "metadata", {}) or {})
        file_id = str(metadata.get("file_id") or "").strip()
        element_id = _as_int(metadata.get("element_id"))
        path = str(metadata.get("image_storage_path") or "").strip()
        if str(metadata.get("type") or "").strip().lower() != "image" or not file_id or element_id is None:
            continue
        if not path:
            path = await _resolve_legacy_image_storage_path(
                file_id,
                element_id,
                path_index_by_file=path_index_by_file,
            )
            if not path:
                continue
            metadata["image_storage_path"] = path
        key = (file_id, element_id)
        previous = best_by_image.get(key)
        if previous is None or _score(metadata) > _score(previous[1]):
            best_by_image[key] = (document, metadata)

    ordered = sorted(best_by_image.values(), key=lambda entry: _score(entry[1]), reverse=True)
    return [
        ImageSearchCandidate(
            file_id=str(metadata["file_id"]),
            element_id=int(metadata["element_id"]),
            image_storage_path=str(metadata["image_storage_path"]),
            score=_score(metadata),
            file_name=str(metadata.get("file_name") or ""),
            page=_as_int(metadata.get("page")),
            summary=str(getattr(document, "page_content", "") or ""),
        )
        for document, metadata in ordered[:limit]
    ]


async def load_image_as_data_uri(candidate: ImageSearchCandidate) -> str:
    """从 MinIO 加载图片原图，并转换为可直接放入 ``img.src`` 的 Data URI。"""
    image_bytes = await storage_service.download(
        MINERU_OUTPUT_BUCKET,
        candidate.image_storage_path,
    )
    mime_type = mimetypes.guess_type(candidate.image_storage_path)[0] or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


async def query_images(
    question: str,
    *,
    top_k: int = _IMAGE_QUERY_LIMIT,
    file_ids: Optional[list[str]] = None,
    folder_id: Optional[str] = None,
) -> list[ImageQueryItem]:
    """语义检索图片孙切片，返回最多三张可直接展示的原图临时 URL。"""
    limit = max(1, min(int(top_k), _IMAGE_QUERY_LIMIT))
    candidates = await search_image_candidates(
        question,
        top_k=limit * 5,
        file_ids=file_ids,
        folder_id=folder_id,
    )
    results: list[ImageQueryItem] = []
    for candidate in candidates:
        if len(results) >= limit:
            break
        path = candidate.image_storage_path
        try:
            if not await storage_service.exists(MINERU_OUTPUT_BUCKET, path):
                continue
            image_url = await storage_service.presigned_url(MINERU_OUTPUT_BUCKET, path, expires_s=_IMAGE_URL_EXPIRES_S)
        except Exception as exc:
            logger.warning("图片原图不可访问，跳过候选 | file_id={} element_id={} error={}", candidate.file_id, candidate.element_id, exc)
            continue
        results.append(ImageQueryItem(
            file_id=candidate.file_id,
            file_name=candidate.file_name,
            element_id=candidate.element_id,
            page=candidate.page,
            score=int(round(candidate.score * 1000)),
            image_url=image_url,
            summary=candidate.summary,
        ))
    return results
