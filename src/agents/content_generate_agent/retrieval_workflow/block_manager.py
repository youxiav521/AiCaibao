from typing import List, Dict, Any, Optional, Tuple, Callable
from ....utils.langfuse_tracing import trace_span
from ..input_models import Item, DocumentItems, extract_text_yusys_ai
from ..utils import count_tokens, get_default_tokenizer
from .state import Block, BlockWithPieces, Piece, Source
from loguru import logger

class BlockManager:
    """管理文本块构建和切分的类"""
    
    def __init__(self):
        """
        初始化BlockManager，使用默认tokenizer
        """
        pass
    
    # 已过时
    def extract_text_minerU(self, item: Dict[str, Any]) -> Optional[str]:
        """
        提取文本，目前只支持minerU型json格式的text和table类型
        
        Args:
            item: 包含文本内容的字典项
            
        Returns:
            提取的文本内容，如果无法提取则返回None
        """
        if not isinstance(item, dict):
            logger.warning("警告：遇到非dict条目，已跳过。")
            return None

        item_type = item.get("type")

        if item_type == "text":
            txt = item.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt
            else:
                logger.warning("警告：type=text 但不包含有效的 text 字段，已跳过。")
                return None

        if item_type == "table":
            body = item.get("table_body")
            if isinstance(body, str) and body.strip():
                return body
            else:
                logger.warning("警告：type=table 但不包含有效的 table_body 字段，已跳过。")
                return None

    
    def _build_blocks_for_file(self, data: List[Item], max_token_per_block: int, start_block_index: int = 0, text_extractor: Optional[Callable[[Item], Optional[str]]] = None) -> List[Block]:
        '''
        为单个文件构建blocks（原有逻辑）
        
        Args:
            data: 单个文件的Item列表
            max_token_per_block: 每个block的最大token数
            start_block_index: 起始block索引
            text_extractor: 文本提取函数，如果为None则使用默认的extract_text_yusys_ai
            
        Returns:
            构建的blocks列表
        '''
        # 如果没有提供text_extractor，使用默认的extract_text_yusys_ai
        if text_extractor is None:
            text_extractor = extract_text_yusys_ai
            
        blocks: List[Block] = []
        current_texts: List[str] = []
        current_sources: List[Source] = []  # align with current_texts
        current_tokens = 0

        for idx, item in enumerate(data):
            item_text = text_extractor(item)
            if not item_text:
                continue

            item_tokens = count_tokens(item_text)

            # 若单个Item本身超过 max_token_per_block，则该Item独立成块
            if item_tokens > max_token_per_block:
                if current_texts:
                    blocks.append(Block(
                        block_index=start_block_index + len(blocks),
                        token_count=current_tokens,
                        text_full=current_texts,
                        sources_full=current_sources,
                    ))
                    current_texts = []
                    current_sources = []
                    current_tokens = 0
                blocks.append(Block(
                    block_index=start_block_index + len(blocks),
                    token_count=item_tokens,
                    text_full=[item_text],
                    sources_full=[Source(
                        element_id=getattr(item, "element_id", None),
                        file_id=getattr(item, "file_id", None),
                    )],
                ))
                continue

            # 尝试将该Item加入当前块；若会超限，则先结束当前块，再以该Item作为新块的第一条
            if current_tokens + item_tokens > max_token_per_block:
                if current_texts:
                    blocks.append(Block(
                        block_index=start_block_index + len(blocks),
                        token_count=current_tokens,
                        text_full=current_texts,
                        sources_full=current_sources,
                    ))
                # 新块从该Item开始
                current_texts = [item_text]
                current_sources = [Source(
                    element_id=getattr(item, "element_id", None),
                    file_id=getattr(item, "file_id", None),
                )]
                current_tokens = item_tokens
            else:
                current_texts.append(item_text)
                current_sources.append(Source(
                    element_id=getattr(item, "element_id", None),
                    file_id=getattr(item, "file_id", None),
                ))
                current_tokens += item_tokens

        # 收尾：有残留则落盘
        if current_texts:
            blocks.append(Block(
                block_index=start_block_index + len(blocks),
                token_count=current_tokens,
                text_full=current_texts,
                sources_full=current_sources,
            ))
        
        return blocks

    @trace_span(name="build_block_yusys_ai")
    def build_block_yusys_ai(self, data: List[Item], max_token_per_block: int, text_extractor: Optional[Callable[[Item], Optional[str]]] = None) -> List[Block]:
        '''
        从满足yusys ai文本解析格式json格式的文本串data中得到初步的block结构
        方案1：按文件分组，为每个文件单独构建blocks，避免跨文件合并

        Args:
            data: `input_models.DocumentItems` 解析后的 `Item` 列表；若传入原始 dict 列表，将自动校验并转换
            max_token_per_block: 每个block的最大token数
            text_extractor: 文本提取函数，如果为None则使用默认的extract_text_yusys_ai
            
        Returns:
            构建的blocks列表
        '''
        # 按 file_id 分组，保持稳定顺序
        file_order: List[Optional[str]] = []
        file_to_items: Dict[Optional[str], List[Item]] = {}

        for item in data:
            fid = getattr(item, "file_id", None)
            if fid not in file_to_items:
                file_to_items[fid] = []
                file_order.append(fid)
            file_to_items[fid].append(item)

        all_blocks: List[Block] = []
        next_block_index = 0

        # 为每个文件单独构建blocks
        for fid in file_order:
            items = file_to_items.get(fid, [])
            if not items:
                logger.warning(f"警告：文件 {fid} 没有内容，已跳过")
                continue
                
            file_blocks = self._build_blocks_for_file(items, max_token_per_block, next_block_index, text_extractor)
            all_blocks.extend(file_blocks)
            next_block_index += len(file_blocks)

        return all_blocks
    
    

    @trace_span(name="arrange_blocks")
    def arrange_blocks(self, blocks: List[Block], min_token_per_piece: int) -> List[BlockWithPieces]:
        '''
        将每个block分为若干个piece，每个的token数不小于min_token_per_piece参数
        
        Args:
            blocks: 要切分的blocks列表
            min_token_per_piece: 每个piece的最小token数
            
        Returns:
            包含pieces的blocks列表
        '''
        # 处理每个block，将texts转换为pieces
        blocks_with_pieces: List[BlockWithPieces] = []
        total_pieces = 0

        for block in blocks:
            block_idx = block.block_index
            texts = block.text_full  # 这是一个字符串数组
            sources = block.sources_full  # 与 texts 对齐的来源信息
            
            if not isinstance(texts, list):
                logger.warning(f"警告：block {block_idx} 的 text 字段不是数组，跳过")
                continue
            
            logger.info(f"处理 block {block_idx}，包含 {len(texts)} 个 texts")
            
            # 将texts转换为pieces
            pieces: List[Piece] = []
            current_piece_texts = []
            current_piece_sources: List[Source] = []
            current_piece_text = ""
            current_piece_tokens = 0
            
            for idx, text in enumerate(texts):
                if not isinstance(text, str):
                    logger.warning(f"警告：block {block_idx} 中存在非字符串text，跳过")
                    continue
                src = sources[idx]
                
                text_tokens = count_tokens(text)
                
                # 如果单个text就超过最小要求，直接作为独立piece
                if text_tokens >= min_token_per_piece:
                    # 先保存之前的piece（如果有的话）
                    if current_piece_texts:
                        piece_index = len(pieces)
                        pieces.append(Piece(
                            piece_index=piece_index,
                            token_count=current_piece_tokens,
                            text=current_piece_text,
                            texts=current_piece_texts,
                            sources=current_piece_sources,
                        ))
                        current_piece_texts = []
                        current_piece_sources = []
                        current_piece_text = ""
                        current_piece_tokens = 0
                    
                    # 当前text作为独立piece
                    piece_index = len(pieces)   
                    pieces.append(Piece(
                        piece_index=piece_index,
                        token_count=text_tokens,
                        text=text,
                        texts=[text],
                        sources=[src],
                    ))
                    continue
                
                # 尝试将当前text添加到当前piece
                # 尝试将当前text加入当前piece；若当前piece已经达到token数要求先保存后，本text作为新piece第一块
                if current_piece_tokens >= min_token_per_piece:  
                    # 当前piece已达到合理大小，保存并开始新的piece
                    if current_piece_texts:
                        piece_index = len(pieces)   
                        pieces.append(Piece(
                            piece_index=piece_index,
                            token_count=current_piece_tokens,
                            text=current_piece_text,
                            texts=current_piece_texts,
                            sources=current_piece_sources,
                        ))
                    # 开始新的piece
                    current_piece_texts = [text]
                    current_piece_sources = [src]
                    current_piece_text = text
                    current_piece_tokens = text_tokens
                else:
                    # 当前piece未达到合理大小，尝试加入当前piece        
                    current_piece_texts.append(text)
                    current_piece_sources.append(src)
                    current_piece_text = "\n\n".join(current_piece_texts)
                    current_piece_tokens += text_tokens
                
            # 处理block末尾的piece
            if current_piece_texts:
                piece_index = len(pieces)   
                pieces.append(Piece(
                    piece_index=piece_index,
                    token_count=current_piece_tokens,
                    text=current_piece_text,
                    texts=current_piece_texts,
                    sources=current_piece_sources,
                ))
            
            # 构建新的block结构，保持原有字段，添加pieces
            new_block = BlockWithPieces(
                block_index=block.block_index,
                token_count=block.token_count,
                pieces=pieces,
                pieces_count=len(pieces)
            )
            
            blocks_with_pieces.append(new_block)
            total_pieces += len(pieces)
            logger.info(f"  -> 生成 {len(pieces)} 个 pieces")

        return blocks_with_pieces
