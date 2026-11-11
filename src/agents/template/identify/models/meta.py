"""
元数据模型定义
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any

@dataclass
class TableMetadata:
    """表格元数据"""
    df: Any  # pandas DataFrame
    text: str = ''
    text_overlap: str = ''
    intent: Optional[tuple] = None  # 表格意图信息
    position: tuple = None

@dataclass
class ParagraphMetadata:
    """文本段落元数据"""
    text: str  # 段落文本内容

@dataclass
class SNMetadata:
    """带编号文档块"""
    position: tuple = None 
    content: Union[ParagraphMetadata, TableMetadata] = None

@dataclass
class TocInfoSN:
    """目录SN信息"""
    text: str = ''
    level: int = 0
    id: str = ''
    has_children: bool = False
    sn: int = 0

@dataclass
class DocumentMetadata:
    """SN标注文档元数据"""
    file_path: Optional[str] = None
    file_path_html: Optional[str] = None
    sn_metadatas: List[SNMetadata] = field(default_factory=list)
    sn_count: Optional[int] = 0
    sn_texts: Optional[str] = None
    file_path_html_out: Optional[str] = None
    toc: List[TocInfoSN] = field(default_factory=list)
