from __future__ import annotations

from typing import List, Dict, Any, Optional

import time


from ..config import TestConfig, create_default_configs
from ..input_models import DocumentItems, Item, extract_text_yusys_ai, extract_text_only_table
from ..utils import (
    apply_prompt_template,
    ainvoke_llm_with_structured_output,
    abatch_llm_with_structured_output,
    log_error_type_and_message,
)
from .block_manager import BlockManager
from src.base.langfuse_observe import trace_span, trace_llm_call

from .state import SelectedContent, SelectedPiece, BlockSelectionResponse, BlockSelectionResponseSimple, Piece, BlockPieceSelection
from loguru import logger


class LLMRetriever:
    """
    LLM-based retriever that evaluates piece relevance per block.
    """

    def __init__(
        self,
        *,
        test_config: Optional[TestConfig] = None,
        model_id: str = "model_for_content_generation_backup",
        isTableRetrieval: bool = False,
        answer_lang: str = "zh_CN",
    ) -> None:
        # Fallback to defaults if not provided
        if test_config is None:
            default_test_config = create_default_configs()
            test_config = default_test_config

        self.test_config: TestConfig = test_config
        self.model_id: str = model_id
        self.isTableRetrieval: bool = isTableRetrieval
        self.answer_lang: str = answer_lang
        # 将文档切分相关的依赖合并到检索器内部，便于对外隐藏实现细节
        self.block_manager = BlockManager()
        logger.info(
            f"LLMRetriever 初始化: model_id={self.model_id}, "
            f"isTableRetrieval={self.isTableRetrieval}, answer_lang={self.answer_lang}"
        )

    # ----------------------- Structured Output Methods -------------------------
    @trace_span(name="LLMRetriever._create_block_selection_prompt_structured")
    def _create_block_selection_prompt_structured(
        self, question: str, block_pieces: List[Piece], block_index: int, simple: bool = False
    ) -> str:
        """Create the selection prompt for structured JSON output."""
        # 构建文档片段内容
        pieces_content = ""
        for i, piece in enumerate(block_pieces, 1):
            pieces_content += f"[{i}] {piece.text}\n\n"
        
        state = {
            "block_index": block_index,
            "total_pieces": len(block_pieces),
            "pieces_content": pieces_content,
            "question": question
        }
        
        return apply_prompt_template(
            "block_selection_structured" if not simple else "block_selection_structured_simple",
            state,
            locale=self.answer_lang,
        )

    @trace_llm_call(name="LLMRetriever._call_llm_structured")
    async def _call_llm_structured(self, prompt: str, simple: bool = False) -> BlockSelectionResponse | BlockSelectionResponseSimple:
        """Call the LLM using structured output."""
        try:    
            # 使用结构化输出调用模型
            response = await ainvoke_llm_with_structured_output(
                prompt=prompt,
                model_id=self.model_id,
                response_type=BlockSelectionResponse if not simple else BlockSelectionResponseSimple
            )
            return response
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="结构化输出LLM调用失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            # 返回空响应而不是抛出异常
            return BlockSelectionResponse(selected_pieces=[]) if not simple else BlockSelectionResponseSimple(selected_pieces=[])

    @staticmethod
    @trace_span(name="LLMRetriever._parse_structured_response")
    def _parse_structured_response(
        response: BlockSelectionResponse | BlockSelectionResponseSimple, block_pieces: List[Piece], block_index: int
    ) -> List[SelectedPiece]:
        """Parse structured response into SelectedPiece list."""
        selected_pieces: List[SelectedPiece] = []
        
        try:
            for selection in response.selected_pieces:
                # 验证选择的piece_index是否有效
                if 1 <= selection.piece_index <= len(block_pieces):
                    piece = block_pieces[selection.piece_index - 1]
                    
                    # 提取来源信息
                    eids = [s.element_id for s in piece.sources if s.element_id is not None]
                    fids = [s.file_id for s in piece.sources if s.file_id is not None]
                    # 确认文件id是否统一，若不统一则抛出异常
                    if len(fids) == 0 or len(set(fids)) != 1:
                        logger.error(f"Block {block_index} 的pieces中file_id存在错误: {fids}")
                        raise ValueError(f"Block {block_index} 的pieces中file_id存在错误: {fids}")
                    fid = fids[0]
                    
                    selected_piece = SelectedPiece(
                        doc_id=f"block_{block_index}_piece_{piece.piece_index}",
                        element_ids=eids,
                        file_id=fid,
                        relevance_score=selection.relevance_score,
                        selection_reason=selection.selection_reason if isinstance(selection, BlockPieceSelection) else "", 
                        source_stage="document",
                    )
                    selected_pieces.append(selected_piece)
                else:
                    logger.error(f"无效的piece_index: {selection.piece_index}, 总pieces数: {len(block_pieces)}")
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="解析结构化响应时出错",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
        
        return selected_pieces

    @trace_span(name="LLMRetriever.evaluate_block_pieces_structured_simple")
    async def evaluate_block_pieces_structured_simple(
        self, question: str, block_pieces: List[Piece], block_index: int
    ) -> List[SelectedPiece]:
        """使用结构化输出评估文档片段的相关性（简单版，不包含选择理由）"""
        prompt = self._create_block_selection_prompt_structured(question, block_pieces, block_index, simple=True)
        response = await self._call_llm_structured(prompt, simple=True)
        return self._parse_structured_response(response, block_pieces, block_index)

    @trace_span(name="LLMRetriever.evaluate_block_pieces_structured")
    async def evaluate_block_pieces_structured(
        self, question: str, block_pieces: List[Piece], block_index: int, simple: bool = False
    ) -> List[SelectedPiece]:
        """使用结构化输出评估文档片段的相关性（推荐使用）"""
        prompt = self._create_block_selection_prompt_structured(question, block_pieces, block_index, simple=simple)
        response = await self._call_llm_structured(prompt, simple=simple)
        return self._parse_structured_response(response, block_pieces, block_index)

    @trace_span(name="LLMRetriever.run_batch")
    async def run_batch(
        self,
        content_list: List[Item],
        questions: List[str],
        question_ids: List[str],
        simple: bool = False,
    ) -> List[SelectedContent]:
        """
        对提供的内容和问题执行批量检索，支持多个问题。

        参数
        ----------
        content_list: List[Item]
            强类型内容项列表（已在上层完成校验与转换），必须提供，将在内部切分为 blocks/pieces。
        questions: List[str]
            问题列表。必须提供。
        question_ids: List[str]
            问题ID列表。必须提供。  
            question_ids与questions一一对应，问题ID列表必须与问题列表长度相同。
            
        返回值
        -------
        List[Dict[str, Any]]
            返回每个问题的检索结果列表，这里返回的是全量的结果，没有进行top_k筛选
        """
        t0 = time.perf_counter()
        
        # 确定要处理的问题列表
        active_questions = questions or []
        if not active_questions:
            logger.warning("未提供问题列表，跳过检索流程")
            return []
        
        logger.info(f"开始处理 {len(active_questions)} 个问题")
        for i, q in enumerate(active_questions):
            logger.info(f"  问题 {i+1}: {q}")

        
        # 读取并构建 blocks/pieces
        # 根据 isTableRetrieval 参数选择文本提取函数
        text_extractor = extract_text_only_table if self.isTableRetrieval else extract_text_yusys_ai
        blocks = self.block_manager.build_block_yusys_ai(
            content_list,
            max_token_per_block=self.test_config.max_tokens_per_block,
            text_extractor=text_extractor,
        )
        blocks_with_pieces = self.block_manager.arrange_blocks(
            blocks,
            min_token_per_piece=self.test_config.min_tokens_per_piece,
        )
        logger.info(f"接收到 {len(blocks_with_pieces)} 个 blocks")

        # 存储每个问题的结果（按索引对齐，避免相同 question_text 合并）
        question_results: List[List[SelectedPiece]] = [[] for _ in active_questions]
        total_blocks = len(blocks_with_pieces)

        # 外层循环：遍历每个block
        for i, block in enumerate(blocks_with_pieces):
            if not block.pieces:
                logger.warning(f"Block {block.block_index} 没有pieces，跳过")
                continue
            logger.info(f"\n评估进度: {i+1}/{total_blocks} - Block {block.block_index}")
            logger.info(f"  Block {block.block_index} 包含 {len(block.pieces)} 个pieces")

            # 内层循环（批量版）：为同一 block 构造所有问题的 prompts，一次 batch 提交
            prompts = []
            for question_idx, current_question in enumerate(active_questions):
                logger.info(f"    处理问题 {question_idx+1}/{len(active_questions)}: {current_question[:50]}...")
                prompt = self._create_block_selection_prompt_structured(
                    current_question, block.pieces, block.block_index, simple=simple
                )
                prompts.append(prompt)

            if not prompts:
                logger.error("未找到问题列表，已跳过检索流程")
                continue

            try:
                responses = await abatch_llm_with_structured_output(
                    prompts, self.model_id, BlockSelectionResponseSimple if simple else BlockSelectionResponse
                )
            except Exception as e:
                log_error_type_and_message(
                    logger=logger,
                    context="批量结构化输出LLM调用失败",
                    e=e,
                    exc_info=False,
                    max_message_chars=200,
                )
                responses = [BlockSelectionResponseSimple(selected_pieces=[]) if simple else BlockSelectionResponse(selected_pieces=[])] * len(prompts)

            # 将每个问题对应的响应解析并写回 question_results
            for question_idx, response in enumerate(responses):
                selected_pieces = self._parse_structured_response(
                    response, block.pieces, block.block_index
                )
                if selected_pieces:
                    logger.info(
                        f"      -> 问题 {question_idx+1}/{len(active_questions)} (Block {block.block_index}): "
                        f"LLM选择了 {len(selected_pieces)} 个相关片段"
                    )
                    question_results[question_idx].extend(selected_pieces)
                else:
                    logger.info(
                        f"      -> 问题 {question_idx+1}/{len(active_questions)} (Block {block.block_index}): "
                        f"LLM未选择任何片段（可能该block中无相关内容）"
                    )

        # 对每个问题的结果进行总结
        all_results: List[SelectedContent] = []
        
        for idx, current_question in enumerate(active_questions):
            logger.info(f"=== 总结问题{question_ids[idx]}: {current_question} ===")
            
            # 获取该问题的所有选中片段
            question_selected_pieces = question_results[idx]

            # 汇总结果 
            question_id = question_ids[idx]
            
            results = SelectedContent(
                question=current_question,
                question_id=question_id,
                selected_pieces=question_selected_pieces,
            )

            logger.info(f"问题: {current_question}：评估了 {total_blocks} 个blocks，LLM选择 {len(question_selected_pieces)} 个片段")

            all_results.append(results)

        total_seconds = time.perf_counter() - t0
        logger.info(f"\n=== 所有问题检索完成 === 总用时 {total_seconds:.3f}s")
        logger.info(f"处理了 {len(active_questions)} 个问题，{total_blocks} 个blocks")

        return all_results


__all__ = ["LLMRetriever"]
