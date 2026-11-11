"""文档标签匹配API路由"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..schemas.tag_schema import TagRequest, TagResponse
from ..service.tag_service import TagService

# 创建路由
router = APIRouter(
    prefix="/api/doc",
    tags=["doc-tag"]
)

# 实例化服务（可以考虑使用依赖注入优化）
tag_service = TagService()


@router.post("/tags", response_model=TagResponse, summary="文档标签匹配")
async def match_document_tags(request: TagRequest) -> TagResponse:
    """
    为文档内容匹配相关标签
    
    - **tags**: 待匹配的标签列表
    - **docContent**: 解析后的文档内容列表（原始JSON）
    
    返回命中的标签ID列表（逗号分隔）
    """
    try:
        logger.info(f"接收标签匹配请求 - 标签数: {len(request.tags)}")
        
        # 调用服务层
        response = await tag_service.match_tags(request)
        
        return response
        
    except Exception as e:
        logger.error(f"标签匹配API异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"标签匹配失败: {str(e)}")
