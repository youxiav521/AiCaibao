"""
Docx SN 标注模块
给 Word 文档的段落和表格添加 SN 编号
"""
import os
import re
from loguru import logger
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import _Cell, Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl


class DocxSNMarker:
    """Docx SN 标注器"""
    
    def __init__(self):
        """初始化标注器"""
        self.counter = 0
        self.v_docx_sn_detail = []  # [(#编号, 文本)]
        self.doc = None
    
    def marked(self, docx_file: str) -> str:
        """
        给 docx 文档标注序号，返回标注好的文档路径
        
        Args:
            docx_file: 输入的 docx 文件路径
            
        Returns:
            str: 标注后的文档保存路径
        """
        # 构建输出文件路径
        file_name = os.path.basename(docx_file)
        output_dir = os.path.join(".tmp_llmops", "doc", "tmplateVars")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"[marked]-{file_name}")
        
        logger.info(f'开始 SN 标注: {docx_file}')
        logger.info(f'输出文件: {output_file}')
        
        # 重置计数器
        self.counter = 0
        self.v_docx_sn_detail = []
        
        # 加载文档
        self.doc = Document(docx_file)
        
        # 处理文档元素
        elements = list(self.doc.element.body)
        self._mark_elements(elements)
        
        # 保存文件
        self.doc.save(output_file)
        
        logger.info(f'SN 标注完成，共标注 {self.counter} 个元素')
        return output_file
    
    def _mark_elements(self, elements):
        """处理文档中的所有元素（与 DocxColorSN.mark_element 一致）"""
        for i in range(len(elements)):
            element = elements[i]
            if isinstance(element, CT_P):
                p = Paragraph(element, self.doc)
                self._set_background(p)
            elif isinstance(element, CT_Tbl):
                self._process_table(element)
    
    def _set_background(self, para):
        """
        设置段落背景并添加 SN 编号（与 DocxColorSN.set_background 一致）
        
        Args:
            para: 段落或表格单元格对象
        """
        # 跳过已标注的段落（以 #数字 开头）
        if re.match(r'^#\d+', para.text):
            return
        
        if isinstance(para, _Cell):
            # 单元格中嵌套表格
            for ele in para._element.inner_content_elements:
                if isinstance(ele, CT_P):
                    self._set_background(Paragraph(ele, self.doc))
                elif isinstance(ele, CT_Tbl):
                    self._process_table(ele)
        elif isinstance(para, Paragraph):
            # 添加 SN 编号
            self._format_text(para)
    
    def _format_text(self, p: Paragraph):
        """
        格式化文本，添加 SN 编号（与 DocxColorSN._format_text 一致）
        
        Args:
            p: 段落对象
        """
        self.counter += 1
        
        # 定义前缀样式
        prefix_style = {
            'bold': False,
            'color': (255, 0, 0),  # 红色
            'size': 10,
            'italic': False
        }
        _sn = f'#{self.counter}'
        self.v_docx_sn_detail.append((_sn, p.text))
        self._add_styled_prefix_to_existing_paragraph(p, prefix_text=_sn, prefix_style=prefix_style)
    
    def _add_styled_prefix_to_existing_paragraph(self, paragraph, prefix_text="123", prefix_style=None):
        """
        给现有段落添加带样式的前缀，保持原有文本样式不变
        （与 DocxColorSN.add_styled_prefix_to_existing_paragraph 一致）
        
        Args:
            paragraph: 段落对象
            prefix_text: 前缀文本
            prefix_style: 前缀样式字典
        """
        from docx.shared import RGBColor, Pt
        
        # 保存原有文本和样式信息
        original_runs_info = []
        for run in paragraph.runs:
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
        paragraph.clear()
        
        # 添加前缀文本
        prefix_run = paragraph.add_run(prefix_text)
        
        # 应用前缀样式
        if prefix_style:
            if prefix_style.get('bold') is not None:
                prefix_run.font.bold = prefix_style['bold']
            if prefix_style.get('italic') is not None:
                prefix_run.font.italic = prefix_style['italic']
            if prefix_style.get('underline') is not None:
                prefix_run.font.underline = prefix_style['underline']
            if prefix_style.get('color'):
                prefix_run.font.color.rgb = RGBColor(*prefix_style['color'])
            if prefix_style.get('size'):
                prefix_run.font.size = Pt(prefix_style['size'])
            if prefix_style.get('font_name'):
                prefix_run.font.name = prefix_style['font_name']
        
        # 添加分隔符
        paragraph.add_run(" ")
        
        # 恢复原有文本和样式
        for run_info in original_runs_info:
            original_run = paragraph.add_run(run_info['text'])
            original_run.font.bold = run_info['bold']
            original_run.font.italic = run_info['italic']
            original_run.font.underline = run_info['underline']
            if run_info['color']:
                original_run.font.color.rgb = run_info['color']
            if run_info['size']:
                original_run.font.size = run_info['size']
            if run_info['name']:
                original_run.font.name = run_info['name']
    
    def _process_table(self, element):
        """
        处理表格（与 DocxColorSN.process_table 一致）
        
        Args:
            element: 表格元素
        """
        table = Table(element, self.doc)
        table.autofit = True
        
        for row in table.rows:
            for cell in row.cells:
                self._set_background(cell)


def main():
    """测试函数"""
    test_file = r"storage\template\兴业银行模板.docx"
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    marker = DocxSNMarker()
    output_path = marker.marked(test_file)
    
    print(f"标注完成!")
    print(f"输出文件: {output_path}")
    print(f"SN 总数: {marker.counter}")
    print(f"前5个 SN 详情:")
    for sn, text in marker.v_docx_sn_detail[:5]:
        print(f"  {sn}: {text[:50]}...")


if __name__ == '__main__':
    main()
