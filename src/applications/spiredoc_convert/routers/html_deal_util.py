import os
import re
from bs4 import BeautifulSoup, Tag
from loguru import logger


def remove_spire_warning(soup: BeautifulSoup) -> None:
    """删除 Spire.Doc 注释段落。"""
    for span in soup.find_all('span'):
        if 'Evaluation Warning' in span.get_text():
            span.find_parent('p').decompose()
            break


def fix_angle_brackets(soup: BeautifulSoup) -> None:
    """将 html 文本内容中的 &lt;q&gt; 转义符还原为 <q>。"""
    for element in soup.find_all(string=True):
        if '&lt;' in element or '&gt;' in element:
            fixed = element.replace('&lt;', '<').replace('&gt;', '>')
            element.replace_with(fixed)


def merge_html_files(html_dir: str, output_path: str) -> None:
    """
    将目录下所有 html 文件的 body 内容合并到第一个 html 文件中。

    Args:
        html_dir: 包含 html 文件的目录路径
        output_path: 合并后输出的 html 文件路径
    """
    # 按文件名数字顺序排序
    html_files = sorted(
        [f for f in os.listdir(html_dir) if f.endswith('.html')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    if not html_files:
        raise FileNotFoundError(f"目录下没有 html 文件: {html_dir}")

    # 读取第一个 html 文件作为基准
    base_path = os.path.join(html_dir, html_files[0])
    with open(base_path, 'r', encoding='utf-8') as f:
        base_soup = BeautifulSoup(f.read(), 'html.parser')

    # 删除基准文件中的注释
    remove_spire_warning(base_soup)
    # 修复 html 符号
    # fix_angle_brackets(base_soup)
    base_body = base_soup.find('body')

    # 遍历其余 html 文件，提取 body 内容拼接到基准文件
    for filename in html_files[1:]:
        file_path = os.path.join(html_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # 删除基准文件中的注释
        remove_spire_warning(soup)
        # 修复 html 符号
        # fix_angle_brackets(soup)
        body = soup.find('body')
        if body:
            for element in body.children:
                base_body.append(element)

    # 输出合并后的 html 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        # f.write(str(base_soup.prettify(formatter=None)))
        f.write(str(base_soup))

    logger.info(f"✅ 合并完成: {output_path}")


def get_first_level_elements(body):
    first_level_divs = []
    non_div_buffer = []

    def flush_buffer():
        if non_div_buffer:
            # 创建一个虚拟 div 容器包裹非 div 元素
            wrapper = BeautifulSoup('<div class="non-div-wrapper"></div>', "html.parser").div
            for el in non_div_buffer:
                wrapper.append(el.__copy__())
            first_level_divs.append(wrapper)
            non_div_buffer.clear()

    for tag in body.children:
        if not isinstance(tag, Tag):  # 跳过纯文本节点
            continue
        if tag.name == "div":
            flush_buffer()  # 先把缓冲区的非 div 打包
            first_level_divs.append(tag)
        else:
            non_div_buffer.append(tag)

    flush_buffer()  # 处理末尾残留的非 div 元素
    return first_level_divs


def split_html_files(input_path: str, output_dir: str) -> None:
    """
        按照HTML中的<body><div>拆分为多个html文件，完整保留原文档样式。
        核心思路：以原文档为模板，拆分 body 内容，而非新建空白文档。
        """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到文件: {input_path}")

    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        base_soup = BeautifulSoup(f.read(), 'html.parser')

    # 2. 提取原始 <head>（保持完整，含注释、meta、style 等）
    original_head = base_soup.find("head")
    head_str = str(original_head) if original_head else "<head></head>"

    # 3. 找到 <body> 下的第一级 <div>
    body = base_soup.find("body")
    if body is None:
        raise FileNotFoundError("⚠️  未找到 <body> 标签，退出。")
        return

    # first_level_divs = [tag for tag in body.children if getattr(tag, "name", None) == "div"]
    first_level_divs = get_first_level_elements(body)

    # 4. 逐个写出独立 HTML 文件
    for idx, div in enumerate(first_level_divs, start=1):
        # 为输出文件起个有意义的名字（优先用 id 属性）
        filename = f"div_{idx}.html"
        filepath = os.path.join(output_dir, filename)

        # 组装完整 HTML
        output_html = (
            "<!DOCTYPE html>\n"
            "<html>\n"
            f"{head_str}\n"
            "<body>\n"
            f"{str(div)}\n"
            "</body>\n"
            "</html>"
        )

        # 美化输出（可选）
        # pretty = BeautifulSoup(output_html, "html.parser").prettify()

        with open(filepath, "w", encoding="utf-8") as out:
            out.write(output_html)

        logger.info(f"✅  已写出：{filepath}")

    logger.info(f"\n共拆分 {len(first_level_divs)} 个 <div>，文件保存在 [{output_dir}/] 目录。")


if __name__ == '__main__':
    # 使用示例
    split_html_files(
        input_path=r"E:\tmp\spiredoc\merge_test.html",
        output_dir=r"E:\tmp\spiredoc\split"
    )
