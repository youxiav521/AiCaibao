"""
图片类型识别与印章标记路由
"""
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.applications.image_classify.schema.image_classify_schema import (
    SealTaggingRequest,
    SealTaggingResult,
)
from src.applications.image_classify.services.image_classify_service import (
    ImageClassifyService,
)

router = APIRouter(prefix="/api/parsing/mineru", tags=["图片类型识别"])


@router.post(
    "/seal-tagging",
    summary="对 MinerU 解析结果中的 image 元素进行印章识别并标记 type",
    response_model=SealTaggingResult,
)
async def seal_tagging(request: SealTaggingRequest) -> SealTaggingResult:
    """
    输入 file_id，加载 MinerU 解析结果，将所有 image 元素的图片发送到
    /api/v1/predict/layout 接口进行类型识别，识别为印章的元素 type 改为 "seal"。

    Args:
        request: 包含 file_id 的请求。

    Returns:
        处理后的 items 列表及统计信息。
    """
    try:
        svc = ImageClassifyService()
        result = await svc.tag_seals(request.file_id)
        return SealTaggingResult(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"解析结果不存在: {e}")
    except Exception as e:
        logger.exception("印章标记失败 | file_id={}", request.file_id)
        raise HTTPException(status_code=500, detail=f"印章标记失败: {e}")
