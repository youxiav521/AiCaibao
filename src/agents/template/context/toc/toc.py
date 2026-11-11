from typing import List
import asyncio
import traceback
import concurrent.futures
from loguru import logger
from src.agents.template.context.abc import BaseNode
from src.agents.template.context.toc.document_context import Toc, TocInfo
from src.agents.template.llm_utils import abatch_toc_extraction

class TocExtractor(BaseNode):
    """文档目录"""
    
    def _aggregate_tocs(self, all_tocs: List[List[Toc]]) -> List[Toc]:
        """
        整合所有文本块的目录结构，并**保证输出顺序与原文出现顺序一致**。
        """
        if not all_tocs:
            return []

        # 稳定展开：先按 chunk 顺序，再按 chunk 内顺序
        ordered_items: List[tuple[int, int, Toc]] = []
        for chunk_idx, chunk_tocs in enumerate(all_tocs):
            if not chunk_tocs:
                continue
            for toc_idx, toc in enumerate(chunk_tocs):
                ordered_items.append((chunk_idx, toc_idx, toc))

        # 稳定去重：基于（文本内容、层级）去重，保留首次出现项
        seen: set[tuple[str, int]] = set()
        unique_ordered_items: List[tuple[int, int, Toc]] = []
        for chunk_idx, toc_idx, toc in ordered_items:
            key = (toc.text.strip(), toc.level)
            if key in seen:
                continue
            seen.add(key)
            unique_ordered_items.append((chunk_idx, toc_idx, toc))

        # 明确按“原文出现顺序”排序（即使未来上游并发实现变动，也能兜底）
        unique_ordered_items.sort(key=lambda x: (x[0], x[1]))
        return [toc for _, __, toc in unique_ordered_items]
    
    async def toc_level_async(self, text: str, model_id: str = "model_bailian_80b") -> list[TocInfo]:
        """
        异步批量提取带层级的文档目录结构（使用原始文本，类似 toc_level 但使用 abatch）
        Args:
            text: 原始文档文本
            model_id: 模型ID，默认为 "model_bailian_80b"
        Returns:
            TocInfo 对象列表，包含提取的目录结构
        """
        logger.info(f'开始异步结构提取（使用原始文本）...')
        
        try:
            # 获取文本块
            chunks = self.get_chunks(text, chunk_size=20000)
            # 提取文本块内容
            text_chunks = [chunk.page_content for chunk in chunks]
            
            logger.info(f'文本分块完成，共 {len(text_chunks)} 个块')
            
            # 调用异步批量提取方法（返回每个块的结果）
            chunk_toc_results = await abatch_toc_extraction(
                text_chunks=text_chunks,
                model_id=model_id,
            )
            
            # 聚合所有块的目录结构（类似 TocAggregation 的逻辑）
            aggregated_tocs = self._aggregate_tocs(chunk_toc_results)
            
            # 转换为 TocInfo 格式，并过滤无效标题
            rs = []
            for i, toc in enumerate(aggregated_tocs):
                # 清理文本（去除 Markdown 标记）
                cleaned_text = toc.text.replace('*', '').strip()
                cleaned_text = cleaned_text.lstrip('#').strip()
                
                # 过滤无效标题：排除过长的文本（可能是段落）
                if len(cleaned_text) > 50:
                    logger.debug(f'跳过过长文本（可能是段落）: "{cleaned_text[:50]}..." (长度: {len(cleaned_text)})')
                    continue
                
                # 过滤空文本
                if not cleaned_text:
                    logger.debug(f'跳过空文本')
                    continue
                
                # 判断是否有子节点（下一个节点的 level 大于当前节点）
                has_children = False
                if i < len(aggregated_tocs) - 1:
                    has_children = aggregated_tocs[i + 1].level > toc.level
                
                rs.append(TocInfo(
                    text=cleaned_text,
                    level=toc.level,
                    id=len(rs) + 1,  # 使用实际添加的数量作为ID
                    has_children=has_children
                ))
            
            logger.info(f'提取完成，共 {len(rs)} 个目录项')
            return rs
        except Exception as e:
            logger.error(f'目录结构提取失败: {str(e)}')
            logger.error(f'错误堆栈:\n{traceback.format_exc()}')
            # 返回空列表，避免后续处理出错
            return []
    
    def toc_level_sync(self, text: str, model_id: str = "model_bailian_80b") -> list[TocInfo]:
        """
        同步批量提取带层级的文档目录结构（使用原始文本）
        这是 toc_level_async 的同步包装方法
        
        Args:
            text: 原始文档文本
            model_id: 模型ID，默认为 "model_bailian_80b"
        Returns:
            TocInfo 对象列表，包含提取的目录结构
        """
        def run_in_thread():
            """在线程中运行异步代码，创建新的事件循环"""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.toc_level_async(text, model_id))
            finally:
                new_loop.close()
        
        try:
            # 尝试获取当前运行中的事件循环
            loop = asyncio.get_running_loop()
            # 如果事件循环正在运行，使用线程池来运行异步代码，避免事件循环冲突
            logger.info('检测到运行中的事件循环，使用线程池执行异步代码')
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except RuntimeError:
            # 如果没有运行中的事件循环，直接使用 asyncio.run
            logger.info('未检测到运行中的事件循环，直接使用 asyncio.run')
            return asyncio.run(self.toc_level_async(text, model_id))
        except Exception as e:
            logger.error(f'toc_level_sync 执行失败: {str(e)}')
            logger.error(f'错误堆栈:\n{traceback.format_exc()}')
            # 返回空列表，避免后续处理出错
            return [] 
    