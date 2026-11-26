"""
文档分类API路由
"""
from fastapi import APIRouter, HTTPException, Body, Header
from typing import Optional
from loguru import logger
import asyncio
from pathlib import Path
from minio import Minio

from ..schemas.doc_classify_schema import (
    DocClassifyRequest,
    DocClassifyMinioResponse,
    ClassifyItem,
)
from src.applications.doc_classify.services.doc_classify_services import DocClassifyService
from src.applications.doc_classify.utils.doc_classify_utils import generate_thumbnail_base64, thumbnail_from_png_bytes

# 路由
router = APIRouter(
    prefix="/api/doc",
    tags=["文档分类"]
)

classify_service = DocClassifyService()


def _download_file_from_minio(
    server_url: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_id: str
) -> bytes:
    try:
        # 自动去掉开头的 bucket 前缀
        if file_id.startswith(f"{bucket_name}/"):
            object_name = file_id[len(bucket_name) + 1:]
        else:
            object_name = file_id

        logger.info("最终下载路径 | bucket={}, object={}", bucket_name, object_name)

        client = Minio(
            server_url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        response = client.get_object(bucket_name, object_name)
        file_content = response.read()
        response.close()
        response.release_conn()

        logger.info("文件下载成功 | bytes={}", len(file_content))
        return file_content

    except Exception as e:
        logger.exception("MinIO下载失败 | bucket={}, file_id={}", bucket_name, file_id)
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")


def _parse_accept_language(accept_language: str | None) -> bool:
    if not accept_language:
        return False
    primary_tag = accept_language.split(",")[0].strip().split(";")[0].strip()
    return primary_tag == "en_US"


@router.post("/classify_minio", response_model=DocClassifyMinioResponse)
async def classify_document_from_minio(
    request: DocClassifyRequest = Body(...),
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    try:
        # 组装分类项：classify 为分类名数组，classify_prmpt 按下标一一对应提示词
        classify_items: list[ClassifyItem] = []
        classify_prompts = request.classify_prmpt or []
        for idx, name in enumerate(request.classify):
            prompt = classify_prompts[idx] if idx < len(classify_prompts) else None
            classify_items.append(ClassifyItem(name=name, prompt=prompt))

        category_names = [item.name for item in classify_items]
        if not category_names:
            raise HTTPException(status_code=400, detail="classify不能为空")
        if not request.files:
            raise HTTPException(status_code=400, detail="files不能为空")

        # 正确获取桶名
        bucket_name = next(iter(request.files.keys()))
        file_list = request.files[bucket_name]
        file_info = file_list[0]

        # 下载文件（正确调用）
        file_content = await asyncio.to_thread(
            _download_file_from_minio,
            request.minio.server_url,
            request.minio.access_key,
            request.minio.secret_key,
            bucket_name,
            file_info.fileId
        )

        # 如果指定了 assign_classify，跳过大模型判断，直接返回指定分类 + 缩略图
        if request.assign_classify:
            logger.info("使用指定分类，跳过大模型判断 | assign_classify={}", request.assign_classify)
            if not request.thumb:
                raise HTTPException(status_code=400, detail="thumb不能为空")

            thumb_str = ""
            fallback = await asyncio.to_thread(
                generate_thumbnail_base64,
                file_content,
                file_info.fileName,
                request.thumb,
            )
            if fallback:
                thumb_str = fallback

            return DocClassifyMinioResponse(
                classify=request.assign_classify,
                fileId=file_info.fileId,
                fileName=file_info.fileName,
                thumb=thumb_str
            )

        # 原实现：classify_items 直接取 request.classify（当时 classify 为对象数组）
        # classify_items = request.classify

        file_ext = Path(file_info.fileName).suffix.lower()
        is_text_file = file_ext in {'.txt', '.csv', '.md', '.log'}

        use_english = _parse_accept_language(accept_language)

        result = await classify_service.classify(
            file_content=file_content,
            filename=file_info.fileName,
            classify_items=classify_items,
            max_pages=3,
            use_english=use_english,
        )

        # 缩略图，只返回第一页 base64
        if not request.thumb:
            raise HTTPException(status_code=400, detail="thumb不能为空")

        thumb_str = ""
        if not is_text_file:
            pages_bytes = result.get("pages_bytes", [])
            if pages_bytes:
                first_page = pages_bytes[0]
                t = await asyncio.to_thread(thumbnail_from_png_bytes, first_page, request.thumb)
                if t:
                    thumb_str = t
            else:
                fallback = await asyncio.to_thread(
                    generate_thumbnail_base64,
                    file_content,
                    file_info.fileName,
                    request.thumb,
                )
                if fallback:
                    thumb_str = fallback

        return DocClassifyMinioResponse(
            classify=result["cls"],
            fileId=file_info.fileId,
            fileName=file_info.fileName,
            thumb=thumb_str
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("接口异常")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")
