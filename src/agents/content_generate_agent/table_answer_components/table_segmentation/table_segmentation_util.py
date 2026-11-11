import copy
from bs4 import BeautifulSoup, Tag, NavigableString
from src.agents.content_generate_agent.table_answer_components.models import Tables


def _copy_tag_to_soup(tag: Tag, target_soup: BeautifulSoup) -> Tag:
    """将 tag 复制到 target_soup 中，返回新的 tag"""
    if not tag or not tag.name:
        return None
    
    # 将标签转为字符串，然后用 target_soup 解析并提取
    # 这样可以确保所有内容（包括文本）都被正确复制
    tag_str = str(tag)
    # 使用 target_soup 的解析能力来解析标签字符串
    temp_soup = BeautifulSoup(tag_str, "html.parser")
    source_tag = temp_soup.find(tag.name)
    
    if not source_tag:
        return None
    
    # 创建新标签并复制属性
    new_tag = target_soup.new_tag(tag.name, attrs=dict(tag.attrs) if tag.attrs else {})
    
    # 复制所有子节点（包括文本和标签）
    for child in source_tag.children:
        if hasattr(child, 'name') and child.name:
            # 子元素是标签，递归复制
            new_child = _copy_tag_to_soup(child, target_soup)
            if new_child:
                new_tag.append(new_child)
        else:
            # 子元素是文本（NavigableString），需要创建 NavigableString 对象
            if isinstance(child, NavigableString):
                # 如果是 NavigableString，直接创建新的 NavigableString
                child_text = str(child)
                if child_text.strip():
                    new_string = NavigableString(child_text)
                    new_tag.append(new_string)
            else:
                # 其他情况，转换为字符串后创建 NavigableString
                child_text = str(child)
                if child_text.strip():
                    new_string = NavigableString(child_text)
                    new_tag.append(new_string)
    
    return new_tag


async def split_html_table_by_table_rows(
    original_html: str,
    tables: Tables,
) -> list:
    """
    按照tables中的TableRows把原始的html拆分为多个子表格的html字符串
    
    逻辑：
    1. 从原表格中提取所有的 tr 行
    2. 对每个 TableRows，根据其 row_numbers（基于1的行号）从原表格中提取对应的行
    3. 将每个 TableRows 的所有行组成一个新的子表格
    
    Args:
        original_html: 原始table的HTML（包含<table> ... </table>）
        tables: Tables对象，包含TableRows列表，每个TableRows的row_numbers指定要提取的行号（基于1，第一行为1）

    Returns:
        拆分后的子表格HTML字符串列表，每个元素对应一个TableRows，为完整的table片段（最简单的形式，无thead/tbody）
    """
    # 拆分失败或未获得行号分组时，直接返回原始表格，避免后续流程无表可用
    if not original_html:
        return []
    if not tables or not getattr(tables, "table_list", None):
        return [original_html]

    result = []

    # 解析原始html
    soup = BeautifulSoup(original_html, "html.parser")
    table_tag = soup.find("table")
    if not table_tag:
        return []

    # 获取原表格中所有的 tr 标签（忽略 thead/tbody 结构，直接获取所有 tr）
    all_trs = table_tag.find_all("tr", recursive=True)
    if not all_trs:
        return []

    # 遍历每个 TableRows，每个 TableRows 对应一个子表格
    for table_rows in tables.table_list:
        row_numbers = table_rows.row_numbers  # 基于1的行号列表，例如 [1, 2, 3] 表示提取第1、2、3行
        
        # 创建新的表格
        new_soup = BeautifulSoup('<table></table>', "html.parser")
        new_table = new_soup.find("table")
        if new_table is None:
            new_table = new_soup.new_tag("table")
            new_soup.append(new_table)
        
        # 按照 row_numbers 的顺序，从原表格中提取对应的行并添加到新表格中
        for row_num in row_numbers:
            idx = row_num - 1  # 转换为0-based索引（row_num=1 -> idx=0）
            
            # 检查行号是否有效
            if idx < 0 or idx >= len(all_trs):
                continue
            
            # 从原表格中获取对应的 tr 行
            original_tr = all_trs[idx]
            
            # 复制 tr 到新表格中
            new_tr = _copy_tag_to_soup(original_tr, new_soup)
            if new_tr:
                new_table.append(new_tr)
        
        # 将生成的子表格添加到结果列表
        result.append(str(new_table))
    
    return result

# ====== main 测试 ======
import asyncio
import json
import sys
from pathlib import Path

def _test_main():
    # 从当前文件向上6级到项目根目录（table_segmentation -> table_answer_components -> content_generate_agent -> agents -> src -> 项目根）
    project_root = Path(__file__).resolve().parents[6]
    # 添加项目根目录到 Python 路径，以便导入 src 模块
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.agents.content_generate_agent.table_answer_components.models import TableRows, Tables
    # 使用测试数据
    test_html = """
    <table>
        <tr><th>A</th><th>B</th></tr>
        <tr><td>1</td><td>x</td></tr>
        <tr><td>2</td><td>y</td></tr>
        <tr><td>3</td><td>z</td></tr>
        <tr><td>5</td><td>os</td></tr>
    </table>
    """
    
    groups = Tables(table_list=[
        TableRows(row_numbers=[1, 2, 4, 5]),  
        TableRows(row_numbers=[1, 3, 4])    
    ])
    print("使用内置测试数据")
    
    results = asyncio.run(split_html_table_by_table_rows(test_html, groups))
    print(f"\n共生成 {len(results)} 个子表格:\n")
    for idx, html_part in enumerate(results, 1):
        print(f"\n=== 子表格 {idx} ===")
        print(html_part)
        print("-" * 50)

if __name__ == "__main__":
    _test_main()