"""
HTML 加载器模块
从 HTML 文件提取内容和表格
"""
from typing import List
import pandas as pd
from loguru import logger
from src.agents.template.identify.utils import read_file_content

class HtmlLoader:
    """HTML 文档加载器"""
    
    def __init__(self):
        """初始化 HtmlLoader"""
        pass
    
    def extract_tables(self, file_path: str) -> List[str]:
        """
        从 HTML 文件中提取 HTML 表格代码片段列表
        
        Args:
            file_path (str): HTML 文件路径
            
        Returns:
            List[str]: HTML 表格代码片段列表
        """
        from bs4 import BeautifulSoup
        
        try:
            content = read_file_content(file_path)
            # 使用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # 查找所有 <table> 元素
            tables = soup.find_all('table')
            
            # 将每个表格转换为 HTML 字符串
            table_html_list = [str(table) for table in tables]
            
            logger.info(f"成功从 {file_path} 提取了 {len(table_html_list)} 个 HTML 表格片段")
            return table_html_list
            
        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return []
        except Exception as e:
            logger.error(f"解析 HTML 文件 {file_path} 时出错: {e}")
            return []
    
    def extract_tables_from_file_with_pandas(self, file_path: str) -> List[pd.DataFrame]:
        """
        使用 pandas.read_html 从文件中提取表格
        
        Args:
            file_path (str): HTML 文件路径
            
        Returns:
            List[pd.DataFrame]: DataFrame 列表
        """
        try:
            dataframes = pd.read_html(file_path)
            logger.info(f"成功从 {file_path} 提取了 {len(dataframes)} 个表格")
            return dataframes
        except Exception as e:
            logger.error(f"解析 HTML 文件时出错: {e}")
            return []
    
    def extract_text(self, file_path: str) -> str:
        """
        从 HTML 文件提取纯文本内容
        
        Args:
            file_path: HTML 文件路径
            
        Returns:
            str: 纯文本内容
        """
        from bs4 import BeautifulSoup
        
        try:
            content = read_file_content(file_path)
            soup = BeautifulSoup(content, 'html.parser')
            
            # 移除 script 和 style 标签
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 获取文本
            text = soup.get_text()
            
            # 清理空白字符
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"提取文本时出错: {e}")
            return ""
    
    def extract_tables_with_pandas_from_html(self, html_content: str) -> List[pd.DataFrame]:
        """
        从 HTML 内容字符串提取表格
        
        Args:
            html_content: HTML 内容字符串
            
        Returns:
            List[pd.DataFrame]: DataFrame 列表
        """
        try:
            dataframes = pd.read_html(html_content)
            logger.info(f"从 HTML 内容提取了 {len(dataframes)} 个表格")
            return dataframes
        except Exception as e:
            logger.error(f"解析 HTML 内容时出错: {e}")
            return []


def main():
    """测试函数"""
    # 测试文件路径
    test_file = r".tmp_llmops/doc/tmplateVars/兴业银行模板.html"
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    loader = HtmlLoader()
    
    # 测试提取表格 (HTML 片段)
    tables = loader.extract_tables(test_file)
    print(f"提取表格数量: {len(tables)}")
    
    # 测试提取表格 (DataFrame)
    dfs = loader.extract_tables_from_file_with_pandas(test_file)
    print(f"DataFrame 数量: {len(dfs)}")
    if dfs:
        print(f"第一个表格形状: {dfs[0].shape}")


if __name__ == '__main__':
    import os
    main()
