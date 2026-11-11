import os
import shutil
from fastapi import Request, Response, HTTPException, APIRouter
from fastapi import UploadFile, File, Form
import uuid  # 用于生成唯一标识符
import asyncio
from docx import Document
from loguru import logger
from .file_split_util import split_docx_by_page_breaks, merge_docx_files
from .file_convert_util import docx_to_html, html_to_docx
from .html_deal_util import merge_html_files, split_html_files
from .add_breakpage import read_docx_in_order

BASE_DIR = "/tmp/spiredoc/"
ADD_BREAKPAGE = True
os.makedirs(BASE_DIR, exist_ok=True)

spiredoc_router = APIRouter(prefix="/api/spiredoc", tags=["SpireDoc文档格式转换"])


@spiredoc_router.post("/word-to-html", summary="Word转HTML")
async def convert_word(file: UploadFile = File(..., description="上传的Word文件")):
    """接收Word文件，返回HTML内容"""
    try:
        # 验证文件类型
        if not file.filename.lower().endswith(('.doc', '.docx')):
            raise HTTPException(status_code=400, detail="仅支持Word文件(.doc/.docx)")

        # 生成临时文件路径
        unique_id = str(uuid.uuid4())
        uuid_dir = os.path.join(BASE_DIR, unique_id)
        # docx 目录
        os.makedirs(uuid_dir, exist_ok=True)
        # subdocx 拆分子文档目录
        subdocx_dir = os.path.join(uuid_dir, "subdocx")
        os.makedirs(subdocx_dir, exist_ok=True)
        # html 转换目录
        html_dir = os.path.join(uuid_dir, "html")
        os.makedirs(html_dir, exist_ok=True)
        word_path = os.path.join(uuid_dir, f"{unique_id}.docx")
        html_merge_path = os.path.join(uuid_dir, f"{unique_id}.html")

        # 保存上传文件
        with open(word_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        try:
            deal_word_path = word_path
            # 添加分页符
            if ADD_BREAKPAGE:
                deal_word_path = read_docx_in_order(word_path)

            # 将docx文档按照分页符进行拆分
            split_docx_by_page_breaks(deal_word_path, subdocx_dir)
            # 按照文件名顺序，读取 word 文件
            for filename in sorted(os.listdir(subdocx_dir), key=lambda x: int(x.split('_')[1].split('.')[0])):
                subdocx_path = os.path.join(subdocx_dir, filename)
                docx_to_html(subdocx_path, html_dir)
            # 读取html目录下的所有文件, 合并成一个html文件
            merge_html_files(html_dir, html_merge_path)
            # 读取并返回内容
            with open(html_merge_path, 'rb') as f:
                html_content = f.read()
            return Response(
                content=html_content,
                media_type="text/html; charset=utf-8"
            )
        finally:
            # 先判断目录是否存在，避免报错
            if os.path.exists(uuid_dir) and os.path.isdir(uuid_dir):
                shutil.rmtree(uuid_dir)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")


@spiredoc_router.post("/html-to-word", summary="HTML转Word")
async def convert_html(file: UploadFile = File(..., description="上传的HTML文件")):
    """接收HTML文件，返回DOCX内容"""
    try:
        # 验证文件类型
        if not file.filename.lower().endswith('.html'):
            raise HTTPException(status_code=400, detail="仅支持HTML文件(.html)")

        # 生成临时文件路径
        unique_id = str(uuid.uuid4())
        uuid_dir = os.path.join(BASE_DIR, unique_id)
        # docx 目录
        os.makedirs(uuid_dir, exist_ok=True)
        # subdocx 拆分子文档目录
        subdocx_dir = os.path.join(uuid_dir, "subdocx")
        os.makedirs(subdocx_dir, exist_ok=True)
        # html 转换目录
        html_dir = os.path.join(uuid_dir, "html")
        os.makedirs(html_dir, exist_ok=True)
        word_path = os.path.join(uuid_dir, f"{unique_id}.docx")
        html_merge_path = os.path.join(uuid_dir, f"{unique_id}.html")

        # 保存上传文件
        with open(html_merge_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        try:
            merge_list = []
            # 将HTML文件按照<body><div>拆分
            split_html_files(html_merge_path, html_dir)
            # 按照文件名顺序，读取 html 文件
            for filename in sorted(os.listdir(html_dir), key=lambda x: int(x.split('_')[1].split('.')[0])):
                subhtml_path = os.path.join(html_dir, filename)
                sub_docx_path = html_to_docx(subhtml_path, subdocx_dir)
                merge_list.append(sub_docx_path)

            for item in merge_list:
                # 删除文档开始第一段的标志 ”spire.doc警告“
                doc = Document(item)
                # 判断第一段是否包含 Spire.Doc 警告
                if doc.paragraphs and "Evaluation Warning" in doc.paragraphs[0].text:
                    doc._body._body.remove(doc.paragraphs[0]._element)
                doc.save(item)

            # 将多个word文档，合并为一个
            merge_docx_path = merge_docx_files(merge_list, uuid_dir)

            # 读取并返回内容
            with open(merge_docx_path, 'rb') as f:
                docx_content = f.read()
            return Response(
                content=docx_content,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={
                    "Content-Disposition": "attachment; filename=merged.docx"
                }
            )

        finally:
            # 先判断目录是否存在，避免报错
            if os.path.exists(uuid_dir) and os.path.isdir(uuid_dir):
                shutil.rmtree(uuid_dir)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")


def tst_convert_word():
    import io
    input_path = r"E:\知识库问答\V2新版\福建农商模板\分页符-模板\福清农商_调查报告模板_标注版本V1.0.docx"
    input_path = r"E:\知识库问答\V2新版\福建农商模板\分页符-模板\test文档.docx"
    input_path = r"E:\知识库问答\V2新版\福建农商模板\分页符-模板\test0520.docx"
    # 包装成本地 UploadFile 对象
    with open(input_path, "rb") as f:
        # 构造 FastAPI UploadFile
        upload_file = UploadFile(
            filename=os.path.basename(input_path),
            file=io.BytesIO(f.read())
        )

        # 调用异步函数
        asyncio.run(convert_word(upload_file))


def tst_convert_html():
    import io
    input_path = r"E:\tmp\spiredoc\merge_test.html"
    # 包装成本地 UploadFile 对象
    with open(input_path, "rb") as f:
        # 构造 FastAPI UploadFile
        upload_file = UploadFile(
            filename=os.path.basename(input_path),
            file=io.BytesIO(f.read())
        )

        # 调用异步函数
        asyncio.run(convert_html(upload_file))


if __name__ == '__main__':
    tst_convert_word()
    # tst_convert_html()
