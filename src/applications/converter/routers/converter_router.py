from fastapi import Request,Response, HTTPException, APIRouter
import subprocess
import os
import json
import glob
import time
import asyncio
from loguru import logger
from pydantic import BaseModel, Field
from fastapi import UploadFile, File,Form
import uuid  # 用于生成唯一标识符
import tempfile
from src.agents.data_analyst_agent.file_content_analysis_agent.content_classify_agent import classify_document_type_tool
from src.agents.data_analyst_agent.file_content_analysis_agent.content_extract_agent import extract_content_to_json_tool
from src.applications.parsing.service.parse_minerU_service import MinerUParsingService
from urllib.parse import quote
from fastapi import Request
from fastapi.responses import StreamingResponse

# BASE_DIR = "/tmp/word2html/"  # 原代码：Linux 风格路径，Windows 下 soffice 解析不到
BASE_DIR = os.path.join(tempfile.gettempdir(), "word2html")
os.makedirs(BASE_DIR, exist_ok=True)

import shutil

SOFFICE = shutil.which("soffice") or r"C:\Program Files\LibreOffice\program\soffice.exe"

converter_router = APIRouter(prefix="/api/convert", tags=["文档解析"])

def convert_to_html(input_path: str, output_dir: str) -> bytes:
    """转换文件为HTML，使用指定输出目录"""
    # 执行转换 linux
    # result = subprocess.run(
    #     [
    #         "soffice",
    #         "--headless",
    #         "--convert-to", "html",
    #         "--outdir", output_dir,
    #         input_path
    #     ],
    #     capture_output=True,
    #     text=True,
    #     timeout=30
    # )

    result = subprocess.run(
        [
            SOFFICE,
            "--headless",
            "--convert-to", "html",
            "--outdir", output_dir,
            input_path
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        raise Exception(f"转换失败: {result.stderr.strip()}")
    
    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 查找对应的HTML文件（考虑可能的扩展名大小写）
    html_patterns = [
        os.path.join(output_dir, f"{input_filename}.html"),
        os.path.join(output_dir, f"{input_filename}.HTML")
    ]
    
    html_file = None
    for pattern in html_patterns:
        if os.path.exists(pattern):
            html_file = pattern
            break
    
    if not html_file:
        # 如果没找到，尝试模糊查找（应对可能的格式转换差异）
        html_files = glob.glob(os.path.join(output_dir, f"{input_filename}*.html")) + \
                     glob.glob(os.path.join(output_dir, f"{input_filename}*.HTML"))
        if html_files:
            html_file = html_files[0]
    
    if not html_file:
        raise Exception(f"未找到转换后的HTML文件: {input_filename}")
    
    # 读取并返回内容
    with open(html_file, 'rb') as f:
        content = f.read()
    
    # 删除转换后的HTML文件
    if os.path.exists(html_file):
        os.remove(html_file)
        
    return content


def convert_to_pdf(input_path: str, output_dir: str) -> bytes:
    """转换文件为PDF，使用指定输出目录"""
    # 执行转换
    result = subprocess.run(
        [
            # "soffice",  # 原代码
            SOFFICE,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", output_dir,
            input_path
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        raise Exception(f"转换失败: {result.stderr.strip()}")
    
    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 查找对应的HTML文件（考虑可能的扩展名大小写）
    html_patterns = [
        os.path.join(output_dir, f"{input_filename}.pdf"),
        os.path.join(output_dir, f"{input_filename}.PDF")
    ]
    
    html_file = None
    for pattern in html_patterns:
        if os.path.exists(pattern):
            html_file = pattern
            break
    
    if not html_file:
        # 如果没找到，尝试模糊查找（应对可能的格式转换差异）
        html_files = glob.glob(os.path.join(output_dir, f"{input_filename}*.pdf")) + \
                     glob.glob(os.path.join(output_dir, f"{input_filename}*.PDF"))
        if html_files:
            html_file = html_files[0]
    
    if not html_file:
        raise Exception(f"未找到转换后的PDF文件: {input_filename}")
    
    # 读取并返回内容
    with open(html_file, 'rb') as f:
        content = f.read()
    
    # 删除转换后的HTML文件
    if os.path.exists(html_file):
        os.remove(html_file)
        
    return content

@converter_router.post("/word-to-html", summary="Word转HTML")
async def convert_word(file: UploadFile = File(..., description="上传的Word文件")):
    """接收Word文件，返回HTML内容"""
    try:
        # 验证文件类型
        if not file.filename.lower().endswith(('.doc', '.docx')):
            raise HTTPException(status_code=400, detail="仅支持Word文件(.doc/.docx)")
        
        # 生成临时文件路径
        unique_id = str(uuid.uuid4())
        word_path = os.path.join(BASE_DIR, f"{unique_id}.docx")
        
        # 保存上传文件
        with open(word_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        try:
            # 转换为HTML
            html_content = await asyncio.to_thread(convert_to_html, word_path, BASE_DIR)
            return Response(
                content=html_content,
                media_type="text/html; charset=utf-8"
            )
        finally:
            # 清理临时文件
            if os.path.exists(word_path):
                os.remove(word_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")

class DocRequest(BaseModel):
    """输入参数"""
    prompt: str = Field(None, description="提示词")
    fileContent: str = Field(None, description="文件内容")
  
@converter_router.post("/pdf", summary="文件转PDF，返回pdf文件流")
async def convert_pdf(
    file: UploadFile = File(..., description="上传的文件")):

    """接收文件，返回PDF内容"""
    try:
       
        # 生成临时文件路径
        name = file.filename
        unique_id = str(uuid.uuid4())
        suffix = os.path.splitext(name)[1]
        word_path = os.path.join(BASE_DIR, f"{unique_id}{suffix}")
        
        # 保存上传文件
        with open(word_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        try:
            # 转换为PDF
            # pdf_content = convert_to_pdf(word_path, BASE_DIR)   
            pdf_content = await asyncio.to_thread(
                convert_to_pdf, word_path, BASE_DIR
            )

            filename = os.path.splitext(name)[0] + ".pdf"
            quoted_filename = quote(filename)
            content_disposition = f"attachment; filename*=UTF-8''{quoted_filename}"

            return Response(
                content=pdf_content,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": content_disposition
                }
            )
        finally:
            # 清理临时文件
            if os.path.exists(word_path):
                os.remove(word_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")
  
@converter_router.post("/excel/to/pdf", summary="Excel转PDF，返回pdf文件流")
async def convert_excel_to_pdf(
    file: UploadFile = File(..., description="上传的文件")):

    """接收Excel文件，返回PDF内容"""
    try:
       
        # 生成临时文件路径
        name = file.filename
        excel_path = os.path.join(BASE_DIR, name)
        
        # 保存上传文件
        with open(excel_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        try:
            # 转换为PDF
            # pdf_content = convert_to_pdf(excel_path, BASE_DIR)   
            pdf_content = await asyncio.to_thread(
                excel_convert_to_pdf, excel_path, BASE_DIR
            )

            filename = os.path.splitext(name)[0] + ".pdf"
            quoted_filename = quote(filename)
            content_disposition = f"attachment; filename*=UTF-8''{quoted_filename}"

            return Response(
                content=pdf_content,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": content_disposition
                }
            )
        finally:
            # 清理临时文件
            if os.path.exists(excel_path):
                os.remove(excel_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")

def excel_convert_to_pdf(input_path: str, output_dir: str) -> bytes:
    """转换文件为PDF，使用指定输出目录"""
    # 执行转换
    result = subprocess.run(
        [
            # "soffice",  # 原代码
            SOFFICE,
            "--headless",
            "--convert-to",
            # 核心分页+适配参数：宽适配一页、高自动分页
            "pdf:calc_pdf_Export:FitWidth=1;SinglePageSheets=false;PageSize=A4;PrintAllSheets=true;ExportPrintRange=EntireSheet",
            "--outdir", output_dir,
            input_path
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        raise Exception(f"转换失败: {result.stderr.strip()}")
    
    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 查找对应的HTML文件（考虑可能的扩展名大小写）
    html_patterns = [
        os.path.join(output_dir, f"{input_filename}.pdf"),
        os.path.join(output_dir, f"{input_filename}.PDF")
    ]
    
    html_file = None
    for pattern in html_patterns:
        if os.path.exists(pattern):
            html_file = pattern
            break
    
    if not html_file:
        # 如果没找到，尝试模糊查找（应对可能的格式转换差异）
        html_files = glob.glob(os.path.join(output_dir, f"{input_filename}*.pdf")) + \
                     glob.glob(os.path.join(output_dir, f"{input_filename}*.PDF"))
        if html_files:
            html_file = html_files[0]
    
    if not html_file:
        raise Exception(f"未找到转换后的PDF文件: {input_filename}")
    
    # 读取并返回内容
    with open(html_file, 'rb') as f:
        content = f.read()
    
    # 删除转换后的HTML文件
    if os.path.exists(html_file):
        os.remove(html_file)
        
    return content

@converter_router.post("/classify", summary="根据输入的文件内容和提示词 返回文件所属类型")
def classify(param: DocRequest) -> dict: 
    re = classify_document_type_tool(param.prompt,param.fileContent)
    return {"data": re}

@converter_router.post("/extract", summary="根据输入的文件内容和提示词 返回结构化内容")
def extract(param: DocRequest) -> dict: 
    re = extract_content_to_json_tool(param.prompt,param.fileContent)
    return {"data": re}

@converter_router.post("/read", summary="读取文件内容")
async def read_content(
    file: UploadFile = File(..., description="上传的文件"),
    filename: str = Form(None, description="原始文件名")) -> str:
    
    try:
        # 生成临时文件路径
        file_path = os.path.join(BASE_DIR, file.filename)
        
        # 保存上传文件
        start_time = time.time()
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        duration = time.time() - start_time
        logger.info(f"保存上传文件耗时: {duration:.2f} 秒")
        start_time = 0.0
 
        try:
            # 转换为字符串
            start_time = time.time()
            service = MinerUParsingService()
            parsing_content = await service.parse_to_json_async(file_path)
            duration = time.time() - start_time
            logger.info(f"解析文件内容耗时: {duration:.2f} 秒")
            start_time = time.time()
            parsing_content = json.dumps(parsing_content, ensure_ascii=False)
            duration = time.time() - start_time
            logger.info(f"转换为字符串耗时: {duration:.2f} 秒")
            
            return Response(
                content=parsing_content,
                media_type="text/html; charset=utf-8"
            )
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")

@converter_router.get("/sse")
async def sse_endpoint(request: Request):
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )

# 事件流生成器
async def event_generator(request: Request):
    counter = 0
    while True:
        # 如果客户端断开，request.is_disconnected 会变为 True
        if await request.is_disconnected():
            break
        counter += 1
        payload = {"time": time.strftime("%H:%M:%S"), "counter": counter}
        yield sse_message(payload, event="tick")
        await asyncio.sleep(1)

# 生成 SSE 格式的数据
def sse_message(data: str, event: str = None, id: str = None) -> str:
    msg = f"data: {json.dumps(data)}\n"
    if event:
        msg = f"event: {event}\n{msg}"
    if id:
        msg = f"id: {id}\n{msg}"
    return msg + "\n"