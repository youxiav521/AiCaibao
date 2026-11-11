"""
Docx Q 标记模块
根据识别的 SN 编号，在 marked 文件中标记问题位置
在对应文本两端加上 <q></q> 标记，并去除所有 SN 标记（#XX）
保持原有样式不变
"""
import os
import re
from loguru import logger
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.shared import RGBColor, Pt

from src.agents.template.identify.models.sn_items import SNItem


class DocxQMarker:
    """Docx Q 标记器"""
    
    def __init__(self):
        """初始化标记器"""
        self.doc = None
        self.output_path = None
        # 保存 SN 编号到段落索引的映射
        self.sn_to_para_index = {}
        
    def mark(self, marked_file_path: str, sn_items_list: list[SNItem], output_dir: str = None) -> str:
        """
        根据 SN 编号在 marked 文件中标记问题位置
        Args:
            marked_file_path: marked 文件路径（带 SN 编号的文档）
            sn_items: SN 项列表，每个项包含 start_sn 和 end_sn 字段
            output_dir: 输出目录，默认为 .tmp_llmops/doc/tmplateVars/identified
        Returns:
            str: 标记后的文档保存路径
        """
        # 构建输出路径
        if output_dir is None:
            output_dir = os.path.join(".tmp_llmops", "doc", "tmplateVars", "identified")
        os.makedirs(output_dir, exist_ok=True)
        
        file_name = os.path.basename(marked_file_path)
        output_path = os.path.join(output_dir, f"[identified]-{file_name}")
        
        logger.info(f'开始 Q 标记: {marked_file_path}')
        logger.info(f'输出文件: {output_path}')
        
        # 加载文档
        self.doc = Document(marked_file_path)
        
        # 第一遍扫描：建立 SN 编号到段落索引的映射
        self._build_sn_index()
        
        # 收集所有需要标记的 SN 范围
        sn_ranges = self._collect_sn_ranges(sn_items_list)
        
        # 第二遍处理：遍历所有段落，清除 SN 标签并添加 Q 标签
        self._process_all_elements(sn_ranges)
        
        # 保存文件
        self.doc.save(output_path)
        
        logger.info(f'Q 标记完成')
        return output_path
    
    def _build_sn_index(self):
        """建立 SN 编号到段落索引的映射"""
        self.sn_to_para_index = {}
        elements = list(self.doc.element.body)
        
        for i, element in enumerate(elements):
            if isinstance(element, CT_P):
                para = Paragraph(element, self.doc)
                text = para.text
                # 检查是否以 SN 编号开头（如 #1, #2）
                match = re.match(r'^#(\d+)\s*', text)
                if match:
                    sn_num = match.group(1)
                    self.sn_to_para_index[sn_num] = ('para', i)
                    
            elif isinstance(element, CT_Tbl):
                # 表格中的每个单元格也是一个段落
                table = Table(element, self.doc)
                for row_idx, row in enumerate(table.rows):
                    for cell_idx, cell in enumerate(row.cells):
                        for para_idx, para in enumerate(cell.paragraphs):
                            text = para.text
                            match = re.match(r'^#(\d+)\s*', text)
                            if match:
                                sn_num = match.group(1)
                                # 记录表格中的位置
                                self.sn_to_para_index[sn_num] = ('table_cell', i, row_idx, cell_idx, para_idx)
        
        logger.info(f'SN 索引建立完成，共 {len(self.sn_to_para_index)} 个 SN 编号')
    
    def _collect_sn_ranges(self, sn_items_list: list[SNItem]) -> list[dict]:
        """
        收集所有需要处理的 SN 范围
        Returns:
            list[dict]: 每个元素包含 {'start': int, 'end': int}
        """
        ranges = []
        
        for item in sn_items_list:
            start_sn = getattr(item, 'start_sn', None)
            end_sn = getattr(item, 'end_sn', None)
            
            if not start_sn or not end_sn:
                continue
            
            try:
                start_num = int(start_sn.lstrip('#'))
                end_num = int(end_sn.lstrip('#'))
            except ValueError:
                continue
            
            ranges.append({
                'start': start_num,
                'end': end_num
            })
        
        return ranges
    
    def _process_all_elements(self, sn_ranges: list[dict]):
        """
        遍历所有元素，清除 SN 标签并添加 Q 标签
        """
        elements = list(self.doc.element.body)
        
        for i, element in enumerate(elements):
            if isinstance(element, CT_P):
                para = Paragraph(element, self.doc)
                self._process_paragraph_clear_sn(para, sn_ranges)
            elif isinstance(element, CT_Tbl):
                table = Table(element, self.doc)
                self._process_table_clear_sn(table, sn_ranges)
        
        logger.info(f'所有 SN 标签清除完成')
    
    def _process_paragraph_clear_sn(self, para: Paragraph, sn_ranges: list[dict]):
        """处理段落，去除 SN 标签，并根据需要添加 Q 标签"""
        text = para.text
        if not text:
            return
        
        # 检查是否以 SN 编号开头
        match = re.match(r'^#(\d+)\s*(.*)', text)
        if not match:
            return
        
        sn_num = int(match.group(1))
        remaining_text = match.group(2)
        
        # 判断这个 SN 是否需要添加 Q 标签
        # 在 start_sn 段落前添加 <q>，在 end_sn 段落后添加 </q>
        q_prefix = False
        q_suffix = False
        
        for range_info in sn_ranges:
            start = range_info['start']
            end = range_info['end']
            
            if sn_num == start:
                q_prefix = True
            if sn_num == end:
                q_suffix = True
        
        # 处理段落，保留 SN 标签
        self._replace_paragraph_with_q_and_keep_sn(para, sn_num, remaining_text, q_prefix, q_suffix)
    
    def _process_table_clear_sn(self, table: Table, sn_ranges: list[dict]):
        """处理表格，去除 SN 标签"""
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    self._process_paragraph_clear_sn(para, sn_ranges)
                
                # 处理嵌套表格
                for nested_table in cell.tables:
                    self._process_table_clear_sn(nested_table, sn_ranges)
    
    def _replace_paragraph_with_q_and_keep_sn(self, para: Paragraph, sn_num: int, remaining_text: str, add_q_prefix: bool = False, add_q_suffix: bool = False):
        """
        去除 SN 标签，并添加 Q 标签（可选）
        """
        # 保存原有段落样式信息
        para_format = {
            'alignment': para.alignment,
            'spacing_before': para.paragraph_format.space_before,
            'spacing_after': para.paragraph_format.space_after,
            'line_spacing': para.paragraph_format.line_spacing,
        }

        # 保存原有 runs 的样式信息
        original_runs_info = []
        for run in para.runs:
            run_info = {
                'text': run.text,
                'bold': run.font.bold,
                'italic': run.font.italic,
                'underline': run.font.underline,
                'color': getattr(run.font.color, 'rgb', None),
                'size': run.font.size,
                'name': run.font.name
            }
            original_runs_info.append(run_info)

        # 清空段落
        para.clear()

        # 恢复段落格式
        if para_format['alignment']:
            para.alignment = para_format['alignment']
        if para_format['spacing_before']:
            para.paragraph_format.space_before = para_format['spacing_before']
        if para_format['spacing_after']:
            para.paragraph_format.space_after = para_format['spacing_after']
        if para_format['line_spacing']:
            para.paragraph_format.line_spacing = para_format['line_spacing']

        # 获取文本的样式（使用原有的文本样式，保持与原始一致）
        text_style = self._get_text_style(original_runs_info)

        # 添加前缀 <q>
        if add_q_prefix:
            q_run = para.add_run("<q>")
            self._apply_run_style(q_run, text_style)

        # 添加剩余文本，保留原始文本样式（去除 SN 标签 #XX）
        if remaining_text:
            text_run = para.add_run(f" {remaining_text}")
            self._apply_run_style(text_run, text_style)

        # 添加后缀 </q>
        if add_q_suffix:
            q_run = para.add_run("</q>")
            self._apply_run_style(q_run, text_style)
    
    def _get_sn_style(self, original_runs_info: list) -> dict:
        """获取 SN 编号的样式（使用第一个 run 的样式）"""
        if original_runs_info:
            run_info = original_runs_info[0]
            return {
                'bold': run_info.get('bold'),
                'italic': run_info.get('italic'),
                'underline': run_info.get('underline'),
                'color': run_info.get('color'),
                'size': run_info.get('size'),
                'name': run_info.get('name')
            }
        return {
            'bold': False,
            'italic': False,
            'underline': False,
            'color': None,
            'size': None,
            'name': None
        }
    
    def _get_text_style(self, original_runs_info: list) -> dict:
        """获取文本内容的样式（使用最后一个非空 run 的样式，或默认样式）"""
        # 优先使用最后一个非空 run 的样式（通常是文本内容）
        for run_info in reversed(original_runs_info):
            if run_info.get('text') and run_info['text'].strip():
                return {
                    'bold': run_info.get('bold'),
                    'italic': run_info.get('italic'),
                    'underline': run_info.get('underline'),
                    'color': run_info.get('color'),
                    'size': run_info.get('size'),
                    'name': run_info.get('name')
                }
        # 如果没有找到，返回默认样式
        return {
            'bold': False,
            'italic': False,
            'underline': False,
            'color': None,
            'size': None,
            'name': None
        }
    
    def _get_q_style(self, original_runs_info: list) -> dict:
        """获取 Q 标签的样式（使用原始文本的样式）"""
        if original_runs_info:
            # 使用第一个非空 run 的样式
            for run_info in original_runs_info:
                if run_info['text']:
                    return {
                        'bold': run_info.get('bold'),
                        'italic': run_info.get('italic'),
                        'underline': run_info.get('underline'),
                        'color': run_info.get('color'),
                        'size': run_info.get('size'),
                        'name': run_info.get('name')
                    }
        # 默认样式
        return {
            'bold': False,
            'italic': False,
            'underline': False,
            'color': None,
            'size': None,
            'name': None
        }
    
    def _apply_run_style(self, run, style: dict):
        """应用样式到 run"""
        if style.get('bold') is not None:
            run.font.bold = style['bold']
        if style.get('italic') is not None:
            run.font.italic = style['italic']
        if style.get('underline') is not None:
            run.font.underline = style['underline']
        if style.get('color'):
            run.font.color.rgb = RGBColor(*style['color'])
        if style.get('size'):
            run.font.size = style['size']
        if style.get('name'):
            run.font.name = style['name']


def main():
    """测试函数"""
    # 测试文件路径
    marked_file = r".tmp_llmops\doc\tmplateVars\[marked]-[简化]中国建设银行公司类客户调查评价报告（制造业模板）.docx"
    
    if not os.path.exists(marked_file):
        print(f"Marked 文件不存在: {marked_file}")
        return
    
    # 模拟 SN 项数据（从 LLM 识别结果）
    # 使用新的 start_sn 和 end_sn 格式
    class MockSNItem:
        def __init__(self, answer_type, start_sn, end_sn):
            self.answer_type = answer_type
            self.start_sn = start_sn
            self.end_sn = end_sn
    
    # 示例：测试 default 类型和 dynamic_table 类型
    sn_items = [
        MockSNItem("default", "#3", "#3"),
        MockSNItem("default", "#5", "#7"),
        MockSNItem("dynamic_table", "#10", "#15"),
    ]
    
    marker = DocxQMarker()
    output_path = marker.mark(marked_file, sn_items)
    
    print(f"标记完成!")
    print(f"输出文件: {output_path}")


if __name__ == '__main__':
    main()
