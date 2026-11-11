from loguru import logger
from minio import Minio
from fastapi import HTTPException

def download_file_from_minio(
    server_url: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_id: str
) -> bytes:
    try:
        # 自动去掉开头的 bucket 前缀
        if file_id.startswith(f"{bucket_name}/"):
            object_name = file_id[len(bucket_name) + 1:]
        else:
            object_name = file_id

        logger.info(f"✅ 最终下载路径 -> 桶: {bucket_name}, 文件: {object_name}")

        client = Minio(
            server_url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        response = client.get_object(bucket_name, object_name)
        file_content = response.read()
        response.close()
        response.release_conn()

        logger.info(f"✅ 文件下载成功，大小: {len(file_content)} 字节")
        return file_content

    except Exception as e:
        logger.error(f"❌ MinIO下载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")