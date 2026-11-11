from spire.doc import *
from loguru import logger


def docx_to_html(file_path, target_dir):
    """ 将docx文档转换为html文档 """

    # 创建Document对象
    document = Document()
    file_name = os.path.basename(file_path)
    # 目标文件名称
    target_file_name = os.path.splitext(file_name)[0] + ".html"
    target_file_path = os.path.join(target_dir, target_file_name)
    # 加载一个docx文档
    document.LoadFromFile(file_path)
    # 设置：图片嵌入 HTML 中
    document.HtmlExportOptions.ImageEmbedded = True
    # 设置：样式写入 HTML 中
    document.HtmlExportOptions.CssStyleSheetType = CssStyleSheetType.Internal

    # 保存为HTML文件
    document.SaveToFile(target_file_path, FileFormat.Html)
    logger.info(f"html文件保存成功：{target_file_path}")
    document.Close()
    return target_file_path


def html_to_docx(file_path, target_dir):
    """ 将html文档转换为ocx文档 """
    # 创建Document对象
    document = Document()
    file_name = os.path.basename(file_path)
    # 目标文件名称
    target_file_name = os.path.splitext(file_name)[0] + ".docx"
    target_file_path = os.path.join(target_dir, target_file_name)
    # 加载一个HTML文件
    document.LoadFromFile(file_path, FileFormat.Html, XHTMLValidationType.none)
    # 将HTML文件保存为.docx格式
    document.SaveToFile(target_file_path, FileFormat.Docx2016)
    logger.info(f"docx文件保存成功：{target_file_path}")
    document.Close()
    return target_file_path