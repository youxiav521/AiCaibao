"""
VLM 文件输入处理工具。
"""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import HTTPException
from loguru import logger


def _read_field(source: Any, field_name: str) -> Any:
    if isinstance(source, dict):
        return source.get(field_name)
    return getattr(source, field_name, None)


def _download_file_from_minio(
    server_url: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_id: str,
) -> bytes:
    from minio import Minio

    try:
        object_name = file_id[len(bucket_name) + 1:] if file_id.startswith(f"{bucket_name}/") else file_id
        logger.info("最终下载路径 | bucket={}, object={}", bucket_name, object_name)

        client = Minio(
            server_url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        response = client.get_object(bucket_name, object_name)
        try:
            file_content = response.read()
        finally:
            response.close()
            response.release_conn()

        logger.info("文件下载成功 | size={}", len(file_content))
        return file_content
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(exc)}") from exc


async def download_files_from_minio(
    minio_config: Any,
    files: Mapping[str, Sequence[Any]],
) -> list[str]:
    """从 MinIO 下载文件到临时路径，返回临时文件路径列表（不做转图）。"""
    if not minio_config or not files:
        raise HTTPException(status_code=400, detail="vlm模式下必须提供minio配置和files文件信息")

    server_url = _read_field(minio_config, "server_url")
    access_key = _read_field(minio_config, "access_key")
    secret_key = _read_field(minio_config, "secret_key")
    if not server_url or not access_key or not secret_key:
        raise HTTPException(status_code=400, detail="minio配置缺少server_url、access_key或secret_key")

    file_paths: list[str] = []

    for bucket_name, file_list in files.items():
        for file_info in file_list:
            file_id = _read_field(file_info, "fileId")
            file_name = _read_field(file_info, "fileName")
            if not file_id or not file_name:
                raise HTTPException(status_code=400, detail="files中的fileId和fileName不能为空")

            file_content = await asyncio.to_thread(
                _download_file_from_minio,
                server_url,
                access_key,
                secret_key,
                bucket_name,
                file_id,
            )
            ext = Path(file_name).suffix.lower()
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            try:
                tmp.write(file_content)
            finally:
                tmp.close()
            file_paths.append(tmp.name)

    if not file_paths:
        raise HTTPException(status_code=400, detail="MinIO文件下载结果为空，请检查files参数")

    logger.info("MinIO文件下载完成 | count={}", len(file_paths))
    return file_paths
