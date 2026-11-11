from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import re
from loguru import logger
import hashlib
from typing import Any, Dict, List
import aiofiles
import shutil
from src.base.storage_service import storage_service

TEMPLATE_PARSE_BUCKET = "template-parse"

def to_html(file_path: str):
    import mammoth
    return mammoth.convert_to_html(file_path).value

def to_int(text: str)-> int:
    """提取字符中第一个数字。"""
    if not text:
        return 0
    import re
    match = re.search(r'-?\d+', text)
    if match:
        return int(match.group())
    return 0
def to_json(obj: Any):
    return json.dumps(obj, ensure_ascii=False)
def file_name(file_path: str) -> str:
    """Return the base name of a file without extension."""
    path = Path(file_path)
    return path.stem

def text_md5(text: str) -> str:
    """Return the MD5 hash (hex) of the provided text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def file_md5(file_path: str, chunk_size: int = 8192) -> str:
    """Return the MD5 hash (hex) of the file content."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

async def write_file(files)->str:
    """上传文件到 MinIO，返回 file_id"""
    # 先读取文件内容到内存
    file_content = await files.read()
    
    # 计算 file_id
    md5 = hashlib.md5()
    md5.update(file_content)
    file_id = md5.hexdigest()
    
    # 构造 MinIO 路径
    minio_path = f"upload/{file_id}/{files.filename}"
    
    # 上传到 MinIO
    await storage_service.upload(
        bucket=TEMPLATE_PARSE_BUCKET,
        object_name=minio_path,
        data=file_content,
        content_type=files.content_type or "application/octet-stream"
    )
    
    logger.info(f"文件已上传到 MinIO: {TEMPLATE_PARSE_BUCKET}/{minio_path}, file_id: {file_id}")
    return file_id

async def file_md5_async(file_path: str, chunk_size: int = 8192) -> str:
    """Return the MD5 hash (hex) of the file content (async version)."""
    md5 = hashlib.md5()
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()

async def read_file(id)->str:
    """从 MinIO 下载文件到本地临时目录，返回临时文件路径"""
    # 先列出 bucket 下 upload/ 开头的文件，找到匹配 file_id 的文件
    all_objects = await storage_service.list_objects(
        bucket=TEMPLATE_PARSE_BUCKET,
        prefix="upload/",
        recursive=True
    )
    
    # 查找包含 file_id 的对象
    target_object = None
    for obj in all_objects:
        if f"/{id}/" in obj:
            target_object = obj
            break
    
    if not target_object:
        raise FileNotFoundError(f"No files found for id: {id} in MinIO")
    
    # 下载文件到本地临时目录
    file_content = await storage_service.download(
        bucket=TEMPLATE_PARSE_BUCKET,
        object_name=target_object
    )
    
    # 提取文件名
    filename = Path(target_object).name
    
    # 保存到本地临时目录
    temp_dir = tmp_path(f'dataset/download/{id}')
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / filename
    
    async with aiofiles.open(temp_file_path, "wb") as f:
        await f.write(file_content)
    
    logger.info(f"文件已从 MinIO 下载: {TEMPLATE_PARSE_BUCKET}/{target_object} -> {temp_file_path}")
    return str(temp_file_path)

async def write_jsonl(jsonl_path:str,datas:list):
    async with aiofiles.open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in datas:
            payload = _make_jsonable(obj)
            await f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

def _make_jsonable(obj: Any) -> Any:
    """Convert common object types (BaseModel, dataclass, etc.) to JSON-able data."""
    try:
        from pydantic import BaseModel  # local import to avoid hard dependency at module load
    except ImportError:  # pragma: no cover
            BaseModel = tuple()  # type: ignore

    if isinstance(obj, BaseModel):
            return obj.model_dump(exclude_none=True)
    if is_dataclass(obj):
            return asdict(obj)
    if hasattr(obj, "dict"):
            return obj.dict()
    return obj


def tmp_path(relative_path:str):
    dataset_dir = Path(__file__).resolve().parents[3] / ".tmp_llmops" / relative_path
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir

def calculate_similarity(str1, str2):
    """
    计算两个字符串之间的相似度（简化版本）
    可以替换为更复杂的算法如编辑距离、余弦相似度等
    """
    if not str1 or not str2:
        return 0
        
    # 简单的相似度计算方法：公共子序列长度 / 最长字符串长度
    common_chars = set(str1) & set(str2)
    max_len = max(len(str1), len(str2))
    
    if max_len == 0:
        return 0
        
    return len(common_chars) / max_len
def is_empty(text:str):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if not text:
        return True
    text = text.strip()
    if  len(text)==0 or text in ['None','无']:
        return True
    if bool(re.fullmatch(r'[\….。]+[:：]?', text)):
        return True
    return  False
def openai_api():
    return (os.environ.get("BASE_MODEL", "qwen3-next-80b-a3b-instruct"),os.environ.get("BASE_URL", "http://10.192.72.11:7003/v1"),os.environ.get("API_KEY", "sk-1"))
def remove_q_tags(text: str) -> str:
    # 匹配 <q_数字> 或 </q_数字> 或 </q>
    return re.sub(r'</?q(?:_\d+)?>', '', text)
def split_by_q_tags(text: str) -> List[Dict[str, str]]:
    """按照 <q_数字> 和 </q_数字> 标签拆分文本"""
    pattern = r'<q_(\d+)>(.*?)</q_\1>'
    matches = re.findall(pattern, text, re.DOTALL)

    id_texts = [
        {"id": f"q_{q_num}", "content": content}
        for q_num, content in matches
    ]
    rs = []
    for id_text in id_texts:
        s=text.split(f'<{id_text["id"]}>')
        rs.append({"id": id_text["id"], "content": 
        f'{remove_q_tags(s[0])[-1000:]}<{id_text["id"]}>{id_text["content"]}</{id_text["id"]}>'})
    
    return rs
