"""
Docx 加载器模块
将 Word 文档转换为 Markdown 和 HTML
"""
import os
import time
from loguru import logger
from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
import mammoth

class DocxLoader:
    """Docx 文档加载器"""
    
    def __init__(self):
        """初始化 DocxLoader"""
        pass
    
    def to_markdown(self, file_path: str) -> str:
        """
        将 Word 文档转换为 Markdown
        
        Args:
            file_path: Word 文件路径
            
        Returns:
            str: Markdown 内容
        """
        logger.info(f"开始转换 Word 为 Markdown: {file_path}")
        
        # 使用 markitdown 转换
        from markitdown import MarkItDown
        md = MarkItDown(enable_plugins=False)
        result = md.convert(file_path)
        
        logger.info(f"Word 转 Markdown 完成")
        return result.text_content
    
    def to_text(self, file_path: str) -> str:
        """
        将 Word 文档转换为纯文本
        
        Args:
            file_path: Word 文件路径
            
        Returns:
            str: 文本内容
        """
        import io
        
        doc = Document(file_path)
        elements = list(doc.element.body)
        output = io.StringIO()
        
        for e in elements:
            if isinstance(e, CT_P):
                p = Paragraph(e, doc)
                output.write(p.text + "\n")
            elif isinstance(e, CT_Tbl):
                table = Table(e, doc)
                for row in table.rows:
                    for cell in row.cells:
                        output.write(cell.text + " ")
                    output.write("\n")
        
        result = output.getvalue()
        output.close()
        return result
    
    def to_html(self, file_path: str) -> str:
        """
        将 Word 文档转换为 HTML
        
        Args:
            file_path: Word 文件路径
            
        Returns:
            str: HTML 文件路径
        """
        st = time.time()
        
        result = mammoth.convert_to_html(file_path)
        
        # 构建完整的 HTML 文档
        html_with_styles = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Llmops Gen Document</title>
</head>
<body>
{result.value}
</body>
</html>"""
        
        # 保存 HTML 文件 - 使用 os.path.splitext 替代 file_ext
        ext = os.path.splitext(file_path)[1]
        output_path = file_path.replace(ext, '.html')
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_with_styles)
        
        logger.info(f"docx转html耗时: {time.time() - st}, 保存路径: {output_path}")
        return output_path
    
    def get_tables_markdown(self, file_path: str) -> list:
        """
        从 Word 文档提取表格并转为 Markdown
        
        Args:
            file_path: Word 文件路径
            
        Returns:
            list: Markdown 表格列表
        """
        from docx.oxml.table import CT_Tbl
        from docx.table import Table
        
        doc = Document(file_path)
        md_tables = []
        
        for element in doc.element.body:
            if isinstance(element, CT_Tbl):
                table = Table(element, doc)
                if not table.rows:
                    continue
                
                # 处理表头
                header_cells = table.rows[0].cells
                header_values = [cell.text.strip().replace('\n', '<br>') for cell in header_cells]
                md_table = ['| ' + ' | '.join(header_values) + ' |']
                md_table.append('| ' + ' | '.join(['---' for _ in header_values]) + ' |')
                
                # 处理数据行
                for row in table.rows[1:]:
                    cell_values = [cell.text.strip().replace('\n', '<br>') for cell in row.cells]
                    # 确保列数一致
                    while len(cell_values) < len(header_values):
                        cell_values.append('')
                    md_table.append('| ' + ' | '.join(cell_values) + ' |')
                
                md_tables.append('\n'.join(md_table))
        
        return md_tables


def main():
    """测试函数"""
    # 测试文件路径
    test_file = r"storage\template\兴业银行模板.docx"
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    loader = DocxLoader()
    
    # 测试转 HTML
    html_path = loader.to_html(test_file)
    print(f"HTML 路径: {html_path}")
    
    # 测试转 Markdown
    # md_content = loader.to_markdown(test_file)
    # print(f"Markdown 内容: {md_content[:500]}...")
    
    # 测试提取表格
    tables = loader.get_tables_markdown(test_file)
    print(f"提取表格数量: {len(tables)}")
    if tables:
        print(f"第一个表格:\n{tables[0]}")


if __name__ == '__main__':
    main()
