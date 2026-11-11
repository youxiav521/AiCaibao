import os
import copy
import zipfile
from lxml import etree
from docx import Document
from docx.oxml.ns import qn
from loguru import logger


def split_docx_by_page_breaks(input_path: str, output_dir: str) -> None:
    """
    按分页符拆分 docx，完整保留原文档样式。
    核心思路：以原文档为模板，替换 body 内容，而非新建空白文档。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到文件: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 解析原文档，切割内容块 ──────────────────────────────
    doc = Document(input_path)
    body = doc.element.body

    sections_content = []
    current_chunk = []

    for child in body:
        if child.tag == qn('w:p'):
            has_page_break = False
            for br in child.findall('.//' + qn('w:br')):
                if br.get(qn('w:type')) == 'page':
                    has_page_break = True
                    break

            if has_page_break:
                # 把分页符所在段落（去掉 br 后）也加入当前块
                clean_para = copy.deepcopy(child)
                for br in clean_para.findall('.//' + qn('w:br')):
                    if br.get(qn('w:type')) == 'page':
                        br.getparent().remove(br)
                current_chunk.append(clean_para)
                sections_content.append(current_chunk)
                current_chunk = []
            else:
                current_chunk.append(child)
        else:
            # sectPr 等其他元素：跳过顶层 sectPr，保留其他
            if child.tag != qn('w:sectPr'):
                current_chunk.append(child)

    if current_chunk:
        sections_content.append(current_chunk)

    sections_content = [c for c in sections_content if c]

    # ── 2. 读取原文档的 ZIP 二进制内容（用于复制样式文件）──────
    with open(input_path, 'rb') as f:
        original_zip_bytes = f.read()

    # ── 3. 为每个内容块生成独立 docx ──────────────────────────
    for idx, chunk in enumerate(sections_content, start=1):
        out_path = os.path.join(output_dir, f"section_{idx:02d}.docx")

        # 3a. 以原文档 ZIP 为基础复制（保留所有样式文件）
        with open(out_path, 'wb') as f:
            f.write(original_zip_bytes)

        # 3b. 打开副本，替换 document.xml 中的 body 内容
        _replace_body_in_docx(out_path, chunk)

        logger.info(f"✅ 已生成: {out_path}")


def _replace_body_in_docx(docx_path: str, chunk: list) -> None:
    """
    在已存在的 docx 文件中，用 chunk 替换 word/document.xml 的 body 内容。
    使用临时文件避免 zipfile 不支持原地修改的限制。
    """
    tmp_path = docx_path + ".tmp"

    with zipfile.ZipFile(docx_path, 'r') as zin, \
            zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():
            data = zin.read(item.filename)

            if item.filename == 'word/document.xml':
                # 解析原 document.xml
                root = etree.fromstring(data)
                body = root.find('.//' + qn('w:body'))

                # 清除原有 body 内容（保留最后的 sectPr）
                sect_pr = body.find(qn('w:sectPr'))
                for child in list(body):
                    body.remove(child)

                # 写入新内容（深拷贝避免跨文档引用污染）
                for elem in chunk:
                    body.append(copy.deepcopy(elem))

                # 补充 sectPr（必须是 body 最后一个子元素）
                if sect_pr is not None:
                    body.append(copy.deepcopy(sect_pr))
                else:
                    etree.SubElement(body, qn('w:sectPr'))

                data = etree.tostring(root, xml_declaration=True,
                                      encoding='UTF-8', standalone=True)

            zout.writestr(item, data)

    # 替换原文件
    os.remove(docx_path)
    os.replace(tmp_path, docx_path)


def merge_docx_files(merge_list, output_dir):
    from docx import Document
    from docxcompose.composer import Composer

    # 将多个word文档，合并为一个
    output_path = os.path.join(output_dir, "merged.docx")

    # 以第一个文档为基础
    master = Document(merge_list[0])
    composer = Composer(master)

    # 依次追加后续文档
    for item in merge_list[1:]:
        doc = Document(item)
        composer.append(doc)

    composer.save(output_path)
    logger.info(f"✅ docx文档合并完成: {output_path}")
    return output_path


# ================= 使用示例 =================
if __name__ == "__main__":
    INPUT_FILE = r"E:\知识库问答\V2新版\福建农商模板\分页符-模板\福清农商_调查报告模板_标注版本V1.0.docx"  # 替换为你的文件路径
    OUTPUT_FOLDER = "/tmp/spiredoc/style/"  # 替换为输出目录
    split_docx_by_page_breaks(INPUT_FILE, OUTPUT_FOLDER)
