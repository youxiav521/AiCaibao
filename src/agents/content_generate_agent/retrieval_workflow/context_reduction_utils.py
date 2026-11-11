from __future__ import annotations

import asyncio
from typing import Dict, List, Set, Tuple, Optional, Any
from pydantic import BaseModel, Field

from ..input_models import Item, extract_text_yusys_ai
from .state import SelectedContent, SelectedPiece
from ..config import create_default_configs
from loguru import logger
from ..utils import (
    ainvoke_llm_with_structured_output,
    abatch_llm_with_structured_output,
    apply_prompt_template,
    count_tokens,
    log_error_type_and_message,
)

_CFG = create_default_configs()
_CTX_LIMIT = _CFG.context_token_limit

"""
本文件用于“上下文缩减（context reduction）”类操作：在不改变检索阶段产出结构的前提下，
对最终合并后的 SelectedContent 做去重与规整，以降低上下文冗余。

当前场景：
- chunk_index_retrieval / summary_index_retrieval / document_reading_retrieval 三个阶段分别保证“各自不重复”
- 但阶段之间可能产生重复片段（同一 file_id + element_id 被多次选中）

注意：
- 文本内容通过 file_id + element_ids 从 origin_content 回查获取。
  这里的去重策略是：只更新 element_ids，并在 element_ids 被去光时丢弃该 piece（完全重复）。
"""



def _consecutive_runs(sorted_ids: List[int]) -> List[List[int]]:
    """
    将一组“已升序且已去重”的 element_id 按连续性切分为多个 runs。

    示例：
    - 输入 [1,2,3, 7,8, 10] -> 输出 [[1,2,3], [7,8], [10]]
    """
    if not sorted_ids:
        return []
    runs: List[List[int]] = [[sorted_ids[0]]]
    for x in sorted_ids[1:]:
        # 相邻差 1 则视为连续
        if x == runs[-1][-1] + 1:
            runs[-1].append(x)
        else:
            runs.append([x])
    return runs

def dedupe_selected_content_for_single_question(
    selected_content: SelectedContent,
) -> SelectedContent:
    """
    对“单个问题”（一个 SelectedContent）的 selected_pieces 做 element 级去重与连续性规整。

    输入/输出：
    - 输入：一个问题的 SelectedContent（内部含若干 SelectedPiece）
    - 输出：同结构 SelectedContent，但 selected_pieces 可能被丢弃/其 element_ids 可能被更新

    去重目标：
    - 对于同一问题，跨不同 SelectedPiece 之间，去除重复的 (file_id, element_id)

    连续性目标（按你的要求）：
    - 去重后，如果某个 piece 的 element_ids 出现断裂（例如 [1,2,3, 10,11]），
      则按连续性拆分成多个 SelectedPiece（每个 piece 只指向一个连续段）
    - 拆分出的每个 piece 会分配一个不同的 doc_id（在原 doc_id 后追加分段后缀）

    注意：
    - 文本内容通过 file_id + element_ids 从 origin_content 回查获取。
      这里只更新 element_ids，并在 element_ids 被去光时丢弃该 piece（完全重复）。
    """
    question_id = getattr(selected_content, "question_id", "unknown")
    input_pieces_count = len(selected_content.selected_pieces)
    logger.info(f"[去重] 开始处理问题 {question_id}，输入 pieces 数量: {input_pieces_count}")
    
    # used: 用于记录“在当前问题下、每个 file_id 已经使用过哪些 element_id”
    # 注意：这是“逐问题独立”的状态，不会跨问题互相影响。
    used: Dict[str, Set[int]] = {}  # file_id -> used element_ids
    deduped_pieces: List[SelectedPiece] = []
    
    # 统计信息
    discarded_count = 0  # 完全重复被丢弃的pieces数量
    split_count = 0  # 被拆分的pieces数量
    total_runs_created = 0  # 拆分后创建的总runs数量

    for piece_idx, piece in enumerate(selected_content.selected_pieces, start=1):
        fid = piece.file_id
        # file_id 是必填项，不应该缺失

        used.setdefault(fid, set())

        # piece.element_ids 可能包含重复、也可能顺序不稳定：直接去重并排序
        original_ids = sorted(set(piece.element_ids or []))
        if not original_ids:
            # 没有 element_ids 的 piece 是不正确的，警告后丢弃
            logger.warning(f"[去重] piece {piece_idx} (doc_id={piece.doc_id}, file_id={fid}): 无 element_ids，已丢弃")
            discarded_count += 1
            continue

        
        # 去除已经在更早 piece 中出现过的 element_id
        filtered_ids = [eid for eid in original_ids if eid not in used[fid]]

        # 测试用，打印去重前的element_ids和去重后的element_ids
        logger.debug(f"piece {piece_idx}(doc_id={piece.doc_id}, file_id={fid}): original_ids: {original_ids}, filtered_ids: {filtered_ids}")
        
        if not filtered_ids:
            # 全部重复 -> 该 piece 对当前问题不再贡献新 element，直接丢弃
            discarded_count += 1
            logger.info(f"[去重] piece {piece_idx} (doc_id={piece.doc_id}, file_id={fid}): "
                        f"element_ids={original_ids} 全部重复，已丢弃")
            continue

        # 为了可靠判断连续性，使用已去重且已排序的 filtered_ids 做切分
        runs = _consecutive_runs(filtered_ids)
        if not runs:
            logger.warning(f"[去重] piece {piece_idx} (doc_id={piece.doc_id}): "
                          f"filtered_ids={filtered_ids} 无法生成 runs，已跳过")
            continue

        # 如果被拆分成多个runs，记录拆分信息
        if len(runs) > 1:
            split_count += 1
            logger.info(f"[去重] piece {piece_idx} (doc_id={piece.doc_id}, file_id={fid}): "
                        f"element_ids={filtered_ids} 被拆分为 {len(runs)} 个连续段: {runs}"
                        f"移除了 {len(original_ids) - len(filtered_ids)} 个重复element")

        # 将每个连续段拆成一个新的 SelectedPiece，并保证 doc_id 唯一
        # - doc_id 规则：原 doc_id + "_part{n}"
        # - doc_id 是必填项，不应该为空
        if not piece.doc_id:
            logger.error(f"[去重] piece {piece_idx} (file_id={fid}): doc_id 为空，这是不正确的")
            raise ValueError(f"SelectedPiece 的 doc_id 不能为空 (file_id={fid}, element_ids={original_ids})")
        
        base_doc_id = piece.doc_id
        for idx, run in enumerate(runs, start=1):
            # 将当前 run 写入 used，供后续 piece 去重
            used[fid].update(run)

            new_doc_id = f"{base_doc_id}_part{idx}"
            deduped_pieces.append(piece.model_copy(update={"element_ids": run, "doc_id": new_doc_id}))
            total_runs_created += 1

    output_pieces_count = len(deduped_pieces)
    logger.info(f"[去重] 问题 {question_id} 处理完成: "
                f"输入 {input_pieces_count} 个 pieces -> 输出 {output_pieces_count} 个 pieces, "
                f"丢弃 {discarded_count} 个完全重复的 pieces, "
                f"拆分 {split_count} 个 pieces 为 {total_runs_created} 个连续段")
    
    return selected_content.model_copy(update={"selected_pieces": deduped_pieces})


def dedupe_selected_content_list(
    contents: List[SelectedContent],
) -> List[SelectedContent]:
    """
    对多个问题的 SelectedContent 批量执行去重（逐问题独立）。

    典型调用点：在各检索阶段 merge 之后、进入 judge/生成答案之前。
    """
    logger.info(f"[去重] 开始批量处理 {len(contents)} 个问题的 SelectedContent")
    result = [dedupe_selected_content_for_single_question(c) for c in contents]
    logger.info(f"[去重] 批量处理完成，共处理 {len(result)} 个问题")
    return result


def rename_selected_content_for_single_question(
    selected_content: SelectedContent,
) -> SelectedContent:
    """为单个问题的 SelectedContent 的 selected_pieces 重新分配归一化 doc_id。

    重命名规则：doc1, doc2, doc3, ...（从 1 开始，无下划线）
    """
    question_id = getattr(selected_content, "question_id", "unknown")
    pieces_count = len(selected_content.selected_pieces)
    logger.info(f"[重命名] 开始处理问题 {question_id}，共 {pieces_count} 个 pieces")
    
    renamed_pieces: List[SelectedPiece] = []
    
    for idx, piece in enumerate(selected_content.selected_pieces, start=1):
        new_doc_id = f"doc{idx}"
        old_doc_id = piece.doc_id
        renamed_pieces.append(piece.model_copy(update={"doc_id": new_doc_id}))
        logger.debug(f"[重命名] piece {idx}: {old_doc_id} -> {new_doc_id}")
    
    logger.info(f"[重命名] 问题 {question_id} 处理完成，共重命名 {len(renamed_pieces)} 个 pieces")
    
    return selected_content.model_copy(update={"selected_pieces": renamed_pieces})


def sort_and_rename_retrieval_result(
    retrieval_result: SelectedContent,
) -> SelectedContent:
    """
    统一处理检索结果的排序和重命名。
    
    对检索结果进行以下处理：
    1. 按 relevance_score 从高到低排序（对于后续如果截断的话尽量保留高的片段）
    2. 重新分配 doc_id（统一格式：doc_1, doc_2, ...）
    
    Args:
        retrieval_result: 检索结果对象
        
    Returns:
        处理后的检索结果对象
        
    Note:
        - score排序实际上并不是很靠谱，但是只能暂时如此，之后需要思考一下是否要废除截断策略
    """
    if not retrieval_result or not retrieval_result.selected_pieces:
        return retrieval_result
    
    # 按score从高到低排序
    retrieval_result.selected_pieces.sort(
        key=lambda x: (x.relevance_score or 0), reverse=True
    )
    
    # 重新rename
    retrieval_result = rename_selected_content_for_single_question(retrieval_result)
    
    return retrieval_result


def build_item_key_to_item(origin_content: List[Item]) -> Dict[Tuple[str, int], Item]:
    """
    构建 (file_id, element_id) -> Item 的映射字典，用于快速回查。

    Args:
        origin_content: 原始内容列表（Item 对象列表）

    Returns:
        Dict[Tuple[str, int], Item]: 以 (file_id, element_id) 为键的映射字典
    """
    logger.debug(f"[回查] 开始构建 item_map，origin_content 数量: {len(origin_content)}")
    item_map: Dict[Tuple[str, int], Item] = {}
    skipped_count = 0
    duplicate_count = 0
    
    for item in origin_content:
        file_id = str(getattr(item, "file_id", ""))
        element_id = getattr(item, "element_id", None)
        if file_id and element_id is not None:
            key = (file_id, element_id)
            # 如果同一个 key 出现多次，保留最后一个（通常不会发生，但为了健壮性）
            if key in item_map:
                duplicate_count += 1
                logger.error(f"[回查] 检测到重复的 key: (file_id={file_id}, element_id={element_id})，将覆盖之前的项")
            item_map[key] = item
        else:
            skipped_count += 1
            logger.error(f"[回查] 出现无效项: (file_id={file_id}, element_id={element_id})")

    return item_map


def resolve_piece_text(
    piece: SelectedPiece,
    item_map: Dict[Tuple[str, int], Item],
) -> str:
    """
    根据 SelectedPiece 的 file_id 和 element_ids，从 item_map 中回查并拼接文本。

    按 element_ids 的顺序（保持原顺序）查找对应的 Item，使用 extract_text_yusys_ai 提取文本，
    然后用换行符连接。

    Args:
        piece: SelectedPiece 对象，包含 file_id 和 element_ids
        item_map: (file_id, element_id) -> Item 的映射字典

    Returns:
        str: 拼接后的文本内容。如果找不到任何匹配的 Item，返回空字符串。
    """
    # 优先使用内嵌的原始文本（用于 data_sources 直接注入的上下文）
    if getattr(piece, "raw_text", None):
        logger.info(f"[回查] 使用内嵌的原始文本: doc_id={piece.doc_id}, raw_text={piece.raw_text}")
        return piece.raw_text

    if not piece.file_id or not piece.element_ids:
        logger.error("警告：SelectedPiece 的 file_id 或 element_ids 为空，且未提供 raw_text，已跳过。")
        return ""

    texts: List[str] = []
    for element_id in piece.element_ids:
        key = (str(piece.file_id), element_id)
        item = item_map.get(key)
        if item:
            text = extract_text_yusys_ai(item)
            if text:
                texts.append(text)

    return "\n".join(texts)


class ContentFilterResult(BaseModel):
    """
    内容筛选结果的结构化输出模型
    """
    relevant_doc_ids: List[str] = Field(
        default_factory=list,
        description="对回答问题有帮助的文档片段 doc_id 列表。只包含真正相关的片段，不要为了凑数而选择不相关的"
    )


async def filter_selected_content_for_single_question(
    selected_content: SelectedContent,
    origin_content: List[Item],
    model_id: str = "model_for_content_generation_backup",
    context_token_limit: int = _CTX_LIMIT,
) -> SelectedContent:
    """
    使用 LLM 对单个问题的 SelectedContent 的 selected_pieces 进行筛选，只保留对回答问题有帮助的 SelectedPiece。

    输入/输出：
    - 输入：一个问题的 SelectedContent（内部含若干 SelectedPiece）、原始内容列表、模型ID、上下文token限制
    - 输出：同结构 SelectedContent，但 selected_pieces 只包含 LLM 认为对回答问题有帮助的片段

    筛选逻辑：
    - 使用 LLM 评估每个 piece 与问题的相关性
    - 只保留 LLM 认为对回答问题有帮助的 pieces
    - 如果所有 pieces 都不相关，返回空的 selected_pieces 列表
    - 当 pieces 总长度超过 context_token_limit 时，会分成多批处理，每批分别调用 LLM，然后合并结果

    注意：
    - 需要提供 origin_content 以便回查文本内容
    - context_token_limit 用于控制每批的最大 token 数，超出时会自动分批处理
    """
    question_id = selected_content.question_id
    question = selected_content.question
    input_pieces_count = len(selected_content.selected_pieces)
    logger.info(f"[筛选] 开始处理问题 {question_id}，输入 pieces 数量: {input_pieces_count}")
    
    # 如果没有 pieces，直接返回
    if not selected_content.selected_pieces:
        logger.warning(f"[筛选] 问题 {question_id} 没有 pieces，直接返回")
        return selected_content
    
    # 构建 item_map 用于回查文本
    item_map = build_item_key_to_item(origin_content)
    
    # 计算基础 prompt 的 token 数（不包含 pieces_content 部分）
    base_prompt = apply_prompt_template("content_filter", {
        "question": question,
        "total_pieces": 0,
        "pieces_content": ""
    })
    base_prompt_tokens = count_tokens(base_prompt)
    
    # 计算每批可用的 token 数
    # 每批的 token 数 = base_prompt_tokens + pieces_content_tokens
    # 需要确保每批不超过 context_token_limit
    available_tokens_per_batch = context_token_limit - base_prompt_tokens
    if available_tokens_per_batch <= 0:
        logger.error(f"[筛选] 基础 prompt token 数 ({base_prompt_tokens}) 超过限制 ({context_token_limit})，无法处理")
        return selected_content
    
    # 在遍历 pieces 的同时构建内容、计算 token 并分批
    batches: List[List[Tuple[str, str, SelectedPiece]]] = []
    current_batch: List[Tuple[str, str, SelectedPiece]] = []
    current_batch_tokens = 0
    doc_id_to_piece: Dict[str, SelectedPiece] = {}
    valid_pieces_count = 0
    
    for piece in selected_content.selected_pieces:
        if not piece.doc_id:
            logger.error(f"[筛选] 发现 doc_id 为空的 piece，已跳过")
            continue
        
        # 通过回查机制获取文本
        piece_text = resolve_piece_text(piece, item_map)
        if not piece_text:
            logger.error(f"[筛选] piece (doc_id={piece.doc_id}) 无法获取文本内容，已跳过")
            continue
        
        # 构建片段描述
        piece_info = f"文档ID: {piece.doc_id}\n"
        piece_info += f"内容: {piece_text}\n"
        
        # 计算当前 piece 的 token 数
        piece_tokens = count_tokens(piece_info)
        
        # 如果当前 piece 单独就超过限制，记录警告并跳过
        if piece_tokens > available_tokens_per_batch:
            logger.error(f"[筛选] piece {piece.doc_id} 的 token 数 ({piece_tokens}) 超过单批限制 ({available_tokens_per_batch})，将直接被忽略")
            continue
        
        # 如果加入当前 piece 会超过限制，开始新批次
        if current_batch and (current_batch_tokens + piece_tokens > available_tokens_per_batch):
            batches.append(current_batch)
            current_batch = []
            current_batch_tokens = 0
        
        # 添加到当前批次
        current_batch.append((piece_info, piece.doc_id, piece))
        current_batch_tokens += piece_tokens
        doc_id_to_piece[piece.doc_id] = piece
        valid_pieces_count += 1
    
    # 添加最后一批
    if current_batch:
        batches.append(current_batch)
    
    if valid_pieces_count == 0:
        logger.error(f"[筛选] 问题 {question_id} 没有有效的 pieces 内容，返回空列表")
        return selected_content.model_copy(update={"selected_pieces": []})
    
    logger.info(f"[筛选] 问题 {question_id} 将 {valid_pieces_count} 个 pieces 分成 {len(batches)} 批处理")
    
    # 构建所有批次的 prompts 和元数据
    batch_prompts: List[str] = []
    batch_metadata: List[Dict[str, Any]] = []  # 存储每批的 doc_ids 和 batch_idx
    
    for batch_idx, batch in enumerate(batches, start=1):
        batch_pieces_content = "\n\n".join([piece_info for piece_info, _, _ in batch])
        batch_doc_ids = [doc_id for _, doc_id, _ in batch]
        
        # 使用 prompt 模板
        prompt = apply_prompt_template("content_filter", {
            "question": question,
            "total_pieces": len(batch),
            "pieces_content": batch_pieces_content
        })
        
        batch_prompts.append(prompt)
        batch_metadata.append({
            "batch_idx": batch_idx,
            "doc_ids": batch_doc_ids,
            "batch_size": len(batch)
        })
    
    # 批量调用 LLM
    all_filtered_doc_ids: Set[str] = set()
    
    try:
        logger.info(f"[筛选] 开始批量调用 LLM，共 {len(batch_prompts)} 个批次")
        results: List[ContentFilterResult] = await abatch_llm_with_structured_output(
            prompts=batch_prompts,
            model_id=model_id,
            response_type=ContentFilterResult
        )
        
        # 处理每批的结果
        for result, metadata in zip(results, batch_metadata):
            batch_idx = metadata["batch_idx"]
            batch_doc_ids = metadata["doc_ids"]
            batch_size = metadata["batch_size"]
            
            # 验证返回的 doc_id 是否都在当前批次的 pieces 中
            valid_doc_ids = set(batch_doc_ids)
            filtered_doc_ids = [
                doc_id for doc_id in result.relevant_doc_ids 
                if doc_id in valid_doc_ids
            ]
            
            # 如果模型返回了不存在的 doc_id，记录警告
            invalid_doc_ids = [
                doc_id for doc_id in result.relevant_doc_ids 
                if doc_id not in valid_doc_ids
            ]
            if invalid_doc_ids:
                logger.error(f"[筛选] 模型返回了不存在的 doc_id: {invalid_doc_ids}，已过滤")
            
            # 添加到总结果中
            all_filtered_doc_ids.update(filtered_doc_ids)
            logger.info(f"[筛选] 第 {batch_idx} 批：从 {batch_size} 个 pieces 中筛选出 {len(filtered_doc_ids)} 个相关 pieces")
            
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="[筛选] 批量 LLM 调用失败，将保留所有批次的 pieces",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        # 如果批量调用失败，保留所有批次的 pieces
        for metadata in batch_metadata:
            all_filtered_doc_ids.update(metadata["doc_ids"])
    
    # 根据筛选后的 doc_id 列表构建新的 pieces
    filtered_pieces = [
        doc_id_to_piece[doc_id] 
        for doc_id in all_filtered_doc_ids
        if doc_id in doc_id_to_piece
    ]
    
    output_pieces_count = len(filtered_pieces)
    logger.info(f"[筛选] 问题 {question_id} 处理完成: "
                f"输入 {input_pieces_count} 个 pieces -> 输出 {output_pieces_count} 个 pieces, "
                f"筛选掉 {input_pieces_count - output_pieces_count} 个不相关的 pieces")
    
    return selected_content.model_copy(update={"selected_pieces": filtered_pieces})


async def filter_long_context_if_needed(
    selected_content: List[SelectedContent],
    question_ids: Optional[List[str]] = None,
    origin_content: Optional[List[Item]] = None,
    model_id: str = "model_for_content_generation_backup",
    token_threshold: Optional[int] = None,
) -> List[SelectedContent]:
    """
    扫描所有 question_ids 对应的 SelectedContent，对于检测到的任何总 token 数大于
    token_threshold 的调用 filter_selected_content_for_single_question 进行筛选。
    
    Args:
        selected_content: SelectedContent 列表
        question_ids: 问题ID列表（可选，如果为 None 则检查所有 selected_content）
        origin_content: 原始内容列表，用于回查文本
        model_id: 模型ID，用于筛选调用
        token_threshold: token 阈值，默认 50000
    
    Returns:
        更新后的 SelectedContent 列表
    """
    if not selected_content or not origin_content:
        return selected_content
    
    # 获取需要检查的问题ID列表
    question_ids_to_check: List[str] = []
    if question_ids:
        question_ids_to_check = question_ids
    else:
        # 如果没有提供问题列表，检查所有 selected_content
        question_ids_to_check = [content.question_id for content in selected_content]
        
    # 构建 item_map 用于回查文本和计算 token
    item_map = build_item_key_to_item(origin_content)
    
    # 构建 question_id 到 SelectedContent 的映射
    question_id_to_content: Dict[str, SelectedContent] = {}
    for content in selected_content:
            question_id_to_content[content.question_id] = content
    
    # 需要筛选的 SelectedContent 列表
    contents_to_filter: List[SelectedContent] = []
    question_ids_to_filter: List[str] = []
    
    if token_threshold is None:
        token_threshold = _CTX_LIMIT

    # 扫描所有问题对应的 SelectedContent
    for qid in question_ids_to_check:
        content = question_id_to_content.get(qid)
        if not content:
            logger.warning(f"[上下文筛选] 问题 {qid} 对应的 SelectedContent 不存在，已跳过")
            continue
        
        # 计算该问题的总 token 数
        total_tokens = 0
        for piece in content.selected_pieces:
            piece_text = resolve_piece_text(piece, item_map)
            if piece_text:
                # 计算 piece 的 token 数（包括 doc_id 等元信息）
                piece_info = f"文档ID: {piece.doc_id}\n内容: {piece_text}\n"
                total_tokens += count_tokens(piece_info)
        
        # 如果超过阈值，添加到筛选列表
        if total_tokens > token_threshold:
            logger.info(f"[上下文筛选] 问题 {qid} 的上下文 token 数 ({total_tokens}) 超过阈值 ({token_threshold})，将进行筛选")
            contents_to_filter.append(content)
            question_ids_to_filter.append(qid)
        else: # TODO:暂时INFO级别，之后修改为DEBUG级别
            logger.info(f"[上下文筛选] 问题 {qid} 的上下文 token 数 ({total_tokens}) 未超过阈值 ({token_threshold})，已跳过")
    
    if not contents_to_filter:
        logger.info(f"[上下文筛选] 没有需要筛选的问题（阈值: {token_threshold}）")
        return selected_content
    
    logger.info(f"[上下文筛选] 开始并发筛选 {len(contents_to_filter)} 个问题的 SelectedContent")
    
    # ========== 优化：并发调用 filter_selected_content_for_single_question ==========
    # 创建并发任务
    tasks = [
        filter_selected_content_for_single_question(
            selected_content=content,
            origin_content=origin_content,
            model_id=model_id,
            context_token_limit=token_threshold,
        )
        for content in contents_to_filter
    ]
    
    # 并发执行所有筛选任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    filtered_contents: List[SelectedContent] = []
    filtered_cnt = 0
    for idx, result in enumerate(results):
        content = contents_to_filter[idx]
        if isinstance(result, Exception):
            logger.error(f"[上下文筛选] 筛选问题 {content.question_id} 失败: {result}，保留原始内容", exc_info=True)
            filtered_contents.append(content)
        else:
            filtered_contents.append(result)
            filtered_cnt += 1
    
    # 构建更新后的 selected_content 列表
    # 将筛选后的内容替换原内容，其他内容保持不变
    updated_selected_content: List[SelectedContent] = []
    filtered_content_map = {c.question_id: c for c in filtered_contents}
    
    for content in selected_content:
        qid = getattr(content, "question_id", None)
        if qid in filtered_content_map:
            updated_selected_content.append(filtered_content_map[qid])
        else:
            updated_selected_content.append(content)
    
    logger.info(f"[上下文筛选] 完成筛选，共处理 {len(contents_to_filter)} 个问题，对 {filtered_cnt} 个问题进行了筛选法上下文压缩")
    
    return updated_selected_content
