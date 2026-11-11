"""
docxtool 模块
文档加载工具模块，提供 Word 和 HTML 文档的加载功能
"""
from src.agents.template.identify.docxtool.docx_loader import DocxLoader
from src.agents.template.identify.docxtool.html_loader import HtmlLoader
from src.agents.template.identify.docxtool.docx_sn_marker import DocxSNMarker
from src.agents.template.identify.docxtool.docx_q_marker import DocxQMarker

__all__ = [
    "DocxLoader",
    "HtmlLoader",
    "DocxSNMarker",
    "DocxQMarker",
]
