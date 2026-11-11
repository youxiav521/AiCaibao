from typing import List, Dict, Any
from pathlib import Path
import logging
from src.applications.parsing.utils.ragflow_parsers import get_text, num_tokens_from_string

logger = logging.getLogger(__name__)

class TxtParsingService:
    """TXT 文件解析服务"""

    def __init__(self, chunk_token_num=128):
        self.chunk_token_num = chunk_token_num

    def parse_to_items(self, file_path: str) -> List[Dict[str, Any]]:
        """
        将 txt 文件解析为元素列表（文本块）。
        
        Args:
            file_path: txt 文件的绝对路径。
            
        Returns:
            表示解析后元素的字典列表。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        # 加载文件内容
        content = get_text(str(path))
        
        # 按行分割以追踪行号
        lines = content.splitlines(keepends=True)
        
        items: List[Dict[str, Any]] = []
        current_chunk_lines = []
        current_chunk_tokens = 0
        start_line_idx = 0
        
        for i, line in enumerate(lines):
            line_tokens = num_tokens_from_string(line)
            
            # 如果添加当前行会超过限制且有待处理内容，输出当前块
            if current_chunk_lines and (current_chunk_tokens + line_tokens > self.chunk_token_num):
                chunk_text = "".join(current_chunk_lines)
                end_line_idx = start_line_idx + len(current_chunk_lines) - 1
                
                if chunk_text.strip():
                    items.append({
                        "type": "text",
                        "value": chunk_text,
                        "md": chunk_text,
                        "startRow": start_line_idx + 1,
                        "endRow": end_line_idx + 1
                    })
                
                # 重置
                current_chunk_lines = []
                current_chunk_tokens = 0
                start_line_idx = i
                
            current_chunk_lines.append(line)
            current_chunk_tokens += line_tokens
            
        # 输出剩余内容
        if current_chunk_lines:
            chunk_text = "".join(current_chunk_lines)
            end_line_idx = start_line_idx + len(current_chunk_lines) - 1
            if chunk_text.strip():
                items.append({
                    "type": "text",
                    "value": chunk_text,
                    "md": chunk_text,
                    "startRow": start_line_idx + 1,
                    "endRow": end_line_idx + 1
                })
                
        return items
