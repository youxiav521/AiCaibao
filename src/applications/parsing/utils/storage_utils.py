import os
import secrets
import string
import tempfile
import uuid
from pathlib import Path

from fastapi import UploadFile

from src.applications.parsing.utils.mineru_storage import upload_original_file

def generate_file_id() -> str:
    """生成文件ID：UUID + 10位随机字符串"""
    # 生成UUID（去掉横线）
    uuid_part = str(uuid.uuid4()).replace('-', '')
    
    # 生成10位随机字符串（字母+数字）
    alphabet = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(alphabet) for _ in range(10))
    
    # 组合：UUID(32位) + 随机字符串(10位) = 42位
    file_id = uuid_part + random_part
    
    return file_id


async def upload_file_to_minio_temp(file: UploadFile) -> tuple[str, str, str]:
    """
    将上传文件持久化到 MinIO，并生成解析器使用的短期本地临时文件。

    返回：
        (file_id, file_path, original_filename)
    """
    file_id = generate_file_id()
    filename = file.filename or "unknown"
    ext = Path(filename).suffix

    file.file.seek(0)
    data = await file.read()

    await upload_original_file(
        file_id,
        filename,
        data,
        file.content_type or "application/octet-stream",
    )

    fd, file_path = tempfile.mkstemp(prefix=f"{file_id}_", suffix=ext)
    try:
        with os.fdopen(fd, "wb") as buffer:
            buffer.write(data)
    except Exception:
        os.close(fd)
        raise

    return file_id, file_path, filename
