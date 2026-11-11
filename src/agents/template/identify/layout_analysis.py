"""
版面分析模块
重新实现 LayoutAnalysis 和 base_analyze 方法
"""
import re
import pandas as pd
import numpy as np
from loguru import logger
from src.agents.template.identify.models.meta import DocumentMetadata, ParagraphMetadata, SNMetadata, TableMetadata, TocInfoSN
from src.agents.template.identify.docxtool import DocxLoader, HtmlLoader, DocxSNMarker
from src.agents.template.identify.utils import extract_text_between_strings
from src.agents.template.identify.utils import get_sn_int, to_sn
from src.agents.template.identify.table_intent import TableIntentRecognition


class LayoutAnalysis:
    """
    模板版面分析 - 重新实现版本
    """

    def __init__(self, cfg=None):
        """
        初始化版面分析器
        
        Args:
            cfg: 配置参数
        """
        self.cfg = cfg
        self.docx_loader = DocxLoader()
        self.html_loader = HtmlLoader()
        self.tbl_intent_recognition = TableIntentRecognition()
        self.metadata = None

    def base_analyze(self, file_path: str) -> DocumentMetadata:
        """
        执行文档版面分析
        Args:
            file_path: 文档路径
            use_toc: 是否使用目录识别
        Returns:
            DocumentMetadata: 文档元数据
        """
        logger.info(f"开始版面分析: {file_path}")

        self.metadata = DocumentMetadata()

        # 使用 DocxSNMarker 进行 SN 标注
        sn_marker = DocxSNMarker()
        self.metadata.file_path = sn_marker.marked(file_path)
        self.metadata.sn_count = sn_marker.counter
        self.metadata.sn_texts = sn_marker.v_docx_sn_detail

        logger.info(f"SN标注完成, SN数量: {self.metadata.sn_count}")

        # 转换为 Markdown
        f_md = self.docx_loader.to_markdown(self.metadata.file_path)
        
        # 转换为 HTML
        self.metadata.file_path_html = self.docx_loader.to_html(self.metadata.file_path)

        # 提取表格
        f_html_tbs = self.html_loader.extract_tables(self.metadata.file_path_html)
        f_html_tbs_df = self.html_loader.extract_tables_from_file_with_pandas(self.metadata.file_path_html)

        # 构建 SN 元数据
        self.metadata.sn_metadatas = []
        c_sn_start = 1

        m_tbls = []
        for i in range(len(f_html_tbs)):
            t = f_html_tbs[i]
            tdf = f_html_tbs_df[i]

            if tdf.shape[0] < 2:
                continue

            # 创建表格元数据
            tm = TableMetadata(
                df=tdf,
                text=t,
                position=(get_sn_int(t), get_sn_int(t, -1))
            )

            # 计算文本重叠
            start_pos = tm.position[0]
            if start_pos > 30:
                tm.text_overlap = ''.join([
                    t[1] for t in self.metadata.sn_texts[start_pos - 30:start_pos - 1]
                ])

            # 分析表格意图
            tm.intent = self.tbl_intent_recognition.analysis_table_intent(
                tdf.replace(to_replace=r'#\d+', value='', regex=True)
            )
            m_tbls.append(tm)

            # 处理 SN 元数据
            if tm.position[0] > c_sn_start:
                out = extract_text_between_strings(f_md, to_sn(c_sn_start), to_sn(tm.position[0]))
                if out:
                    self.metadata.sn_metadatas.append(
                        SNMetadata((c_sn_start, tm.position[0] - 1), ParagraphMetadata(out))
                    )

            self.metadata.sn_metadatas.append(SNMetadata(tm.position, tm))
            c_sn_start = tm.position[1] + 1

        # 处理剩余的 SN
        if c_sn_start < len(self.metadata.sn_texts):
            out = extract_text_between_strings(
                f_md, 
                to_sn(c_sn_start), 
                to_sn(len(self.metadata.sn_texts))
            )
            self.metadata.sn_metadatas.append(
                SNMetadata((c_sn_start, len(self.metadata.sn_texts)), ParagraphMetadata(out))
            )

        logger.info(f"版面分析完成, SN总数: {self.metadata.sn_count}, 表格数: {len(m_tbls)}")

        return self.metadata

