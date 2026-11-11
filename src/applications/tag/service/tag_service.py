"""文档标签匹配服务层"""

from typing import List
from loguru import logger

from ..schemas.tag_schema import TagRequest, TagResponse
from src.agents.doc_tag_agent.tag_agent import DocTagAgent
from src.applications.parsing.entity.converted_entity import ConvertedDocument


class TagService:
    """标签匹配服务"""
    
    def __init__(self):
        """初始化服务"""
        self.agent = DocTagAgent()
    
    async def match_tags(self, request: TagRequest) -> TagResponse:
        """
        执行标签匹配
        
        Args:
            request: 标签匹配请求
            
        Returns:
            TagResponse: 匹配结果
        """
        try:
            logger.info(f"接收标签匹配请求 - 标签数: {len(request.tags)}")
            
            # 使用ConvertedDocument解析文档内容（可选，用于验证格式）
            try:
                doc = ConvertedDocument.from_json_data(request.docContent)
                logger.info(f"文档解析成功 - 元素数: {len(doc.items)}, "
                           f"文本: {len(doc.texts())}, 标题: {len(doc.headings())}, 表格: {len(doc.tables())}")
            except Exception as e:
                logger.warning(f"文档格式解析警告（将继续处理）: {e}")
            
            # 转换为Agent所需格式
            tags = [{"tagId": tag.tagId, "tagName": tag.tagName} for tag in request.tags]
            # 直接传递原始JSON数据
            doc_content = request.docContent
            
            # 调用Agent执行匹配
            result = await self.agent.match_tags(
                tags=tags,
                doc_content=doc_content
            )
            
            if not result.get("success", False):
                logger.error(f"标签匹配失败: {result.get('error', '未知错误')}")
                return TagResponse(
                    tagIds="",
                )
            
            # 提取匹配的标签ID
            matched_tags = result.get("matched_tags", [])
            tag_ids_str = ",".join(matched_tags) if matched_tags else ""
            
            # 提取匹配说明
            explanations = result.get("explanations", [])
            explanation_str = "\n".join(explanations) if explanations else None
            
            logger.info(f"标签匹配完成 - 匹配到 {len(matched_tags)} 个标签: {tag_ids_str}")
            
            return TagResponse(
                tagIds=tag_ids_str,
                explanation=explanation_str,
            )
            
        except Exception as e:
            logger.error(f"标签匹配服务异常: {e}", exc_info=True)
            return TagResponse(
                tagIds="",
            )
