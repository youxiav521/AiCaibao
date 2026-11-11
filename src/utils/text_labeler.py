"""文本标签化和提取工具类"""

import re
from pathlib import Path
from typing import List, Union


class TextLabeler:
    """文本标签化和提取工具类
    
    用于将文本内容转换成带标签的文本，以及从标签化的文本中按照标签ID抽取特定内容。
    """

    @staticmethod
    def label_content(content: str, prefix: str = "doc") -> str:
        """将文本内容转换为带标签的文本
        
        Args:
            content: 原始文本内容
            prefix: 标签前缀，默认为"doc"
            
        Returns:
            str: 标签化后的文本内容
        """
        lines = content.split("\n")
        content_lines = []
        line_counter = 1

        for line in lines:
            if line.strip():  # 如果行不为空（忽略空白字符）
                label = f"{prefix}_{line_counter}"
                content_lines.append(f"<{label}>{line}</{label}>")
                line_counter += 1
            else:
                content_lines.append(line)  # 保持空行

        return "\n".join(content_lines)

    @staticmethod
    def label_file(file_path: Union[str, Path], prefix: str = "doc") -> str:
        """读取文件并转换为带标签的文本
        
        Args:
            file_path: 文件路径
            prefix: 标签前缀，默认为"doc"
            
        Returns:
            str: 标签化后的文本内容
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        
        return TextLabeler.label_content(raw_content, prefix)

    @staticmethod
    def extract_by_ids(ids: List[str], labeled_text: str) -> str:
        """从标签化的文本中提取指定ID的内容
        
        Args:
            ids: 要提取的标签ID列表
            labeled_text: 标签化的文本内容
            
        Returns:
            str: 提取出的文本内容，多个内容用换行符连接
        """
        extracted_content = []
        
        for id in ids:
            # 提取标签之间的文本内容
            pattern = f"<{id}>(.*?)</{id}>"
            matches = re.findall(pattern, labeled_text, re.DOTALL)
            if matches:
                for match in matches:
                    content = match.strip()
                    extracted_content.append(content)

        return "\n".join(extracted_content)

    @staticmethod
    def extract_by_id(id: str, labeled_text: str) -> str:
        """从标签化的文本中提取单个ID的内容
        
        Args:
            id: 要提取的标签ID
            labeled_text: 标签化的文本内容
            
        Returns:
            str: 提取出的文本内容，如果没有找到则返回空字符串
        """
        pattern = f"<{id}>(.*?)</{id}>"
        match = re.search(pattern, labeled_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def get_all_ids(labeled_text: str) -> List[str]:
        """获取标签化文本中的所有标签ID
        
        Args:
            labeled_text: 标签化的文本内容
            
        Returns:
            List[str]: 所有标签ID的列表
        """
        pattern = r"<([^>]+)>.*?</\1>"
        matches = re.findall(pattern, labeled_text, re.DOTALL)
        return list(set(matches))  # 去重并返回
