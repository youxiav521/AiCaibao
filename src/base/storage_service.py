"""
StorageService — 对象存储服务

屏蔽底层对象存储实现，提供统一的文件管理 API。
调用方无需感知具体存储引擎，所有操作均为 async。

使用示例:
    from src.base.storage_service import storage_service

    # 上传（建议路径格式：{domain}/{yyyy_mm_dd}/{file_name}）
    path = await storage_service.upload("yusys", "finance/2024_01_01/report.pdf", data, "application/pdf")

    # 下载
    content = await storage_service.download("yusys", "finance/2024_01_01/report.pdf")

    # 检查存在
    ok = await storage_service.exists("yusys", "finance/2024_01_01/report.pdf")

    # 列出文件
    names = await storage_service.list_objects("yusys", prefix="finance/")

    # 预签名下载链接（默认有效期 1 小时）
    url = await storage_service.presigned_url("yusys", "finance/2024_01_01/report.pdf")

    # 删除
    await storage_service.delete("yusys", "finance/2024_01_01/report.pdf")
"""

import asyncio
import io
from datetime import timedelta

from minio.commonconfig import ENABLED, Filter
from minio.error import S3Error
from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule

from loguru import logger

from src.config import config


class StorageError(Exception):
    """对象存储操作异常，屏蔽底层实现细节。"""


class StorageObjectNotFoundError(StorageError):
    """对象不存在。"""


class StorageService:
    """
    对象存储服务（懒加载单例客户端）。

    所有公开方法返回纯 Python 类型，底层 SDK 不对外暴露。
    客户端在首次 IO 调用时初始化，Gunicorn fork 后各 worker 独立持有。
    """

    _client = None

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @classmethod
    def _get_client(cls):
        """返回懒加载的底层客户端实例。"""
        if cls._client is None:
            from minio import Minio  # 本地导入，隔离底层依赖
            from urllib3 import PoolManager
            cfg = config.minio
            http_client = PoolManager(
                num_pools=50,
                maxsize=50,
            )
            cls._client = Minio(
                cfg.endpoint,
                access_key=cfg.access_key,
                secret_key=cfg.secret_key,
                secure=cfg.secure,
                http_client=http_client,
            )
            logger.info(
                "StorageService | 客户端已初始化 | endpoint={} secure={}",
                cfg.endpoint, cfg.secure,
            )
            # 连通性测试
            try:
                cls._client.list_buckets()
                logger.info("StorageService | 连通性测试成功 | endpoint={}", cfg.endpoint)
            except Exception as exc:
                logger.error("StorageService | 连通性测试失败 | endpoint={} err={}", cfg.endpoint, exc)
                raise StorageError(f"Minio 连通性测试失败: {exc}") from exc
        return cls._client

    @classmethod
    def _ensure_bucket(cls, client, bucket: str) -> None:
        if not client.bucket_exists(bucket):
            try:
                client.make_bucket(bucket)
                logger.info("StorageService | 桶已创建 | bucket={}", bucket)
            except S3Error as exc:
                if exc.code == "BucketAlreadyOwnedByYou":
                    logger.info("StorageService | 桶已存在（并发创建） | bucket={}", bucket)
                    return
                raise

    # ── 公开 API ──────────────────────────────────────────────────────────────

    async def upload(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        上传文件到指定桶，桶不存在时自动创建。

        Args:
            bucket:       桶名
            object_name:  对象路径（建议格式：{domain}/{yyyy_mm_dd}/{file_name}，如 "finance/2024_01_01/report.pdf"）
            data:         文件内容
            content_type: MIME 类型

        Returns:
            object_name（可作为后续操作的句柄）

        Raises:
            StorageError: 上传失败
        """
        def _do() -> str:
            try:
                client = self._get_client()
                self._ensure_bucket(client, bucket)
                client.put_object(
                    bucket,
                    object_name,
                    io.BytesIO(data),
                    length=len(data),
                    content_type=content_type,
                )
                return object_name
            except Exception as exc:
                logger.error(
                    "StorageService | 上传失败 | bucket={} object={} err={}",
                    bucket, object_name, exc,
                )
                raise StorageError(f"上传失败: {exc}") from exc

        return await asyncio.to_thread(_do)

    async def upload_stream(
        self,
        bucket: str,
        object_name: str,
        stream,  # BinaryIO：可 read() 的二进制文件流
        *,
        content_type: str = "application/octet-stream",
        part_size: int = 10 * 1024 * 1024,  # 10MB，MinIO 分块上传最小要求
    ) -> str:
        """
        流式上传文件到指定桶，桶不存在时自动创建。

        与 upload() 不同，本方法直接接收文件流对象，底层走 MinIO 分块上传
        （length=-1 + part_size），适用于大文件（≥500MB）场景，避免一次性
        把整个文件读进内存导致 OOM。

        Args:
            bucket:       桶名
            object_name:  对象路径（建议格式：{domain}/{yyyy_mm_dd}/{file_name}，如 "finance/2024_01_01/report.pdf"）
            stream:       可 read() 的二进制文件流（如 Starlette UploadFile.file 或 open(path, 'rb')）
            content_type: MIME 类型
            part_size:    分块大小（字节），MinIO 要求 ≥ 5MB，默认 10MB

        Returns:
            object_name（可作为后续操作的句柄）

        Raises:
            StorageError: 上传失败
        """
        def _do() -> str:
            try:
                client = self._get_client()
                self._ensure_bucket(client, bucket)
                client.put_object(
                    bucket,
                    object_name,
                    stream,
                    length=-1,
                    part_size=part_size,
                    content_type=content_type,
                )
                return object_name
            except Exception as exc:
                logger.error(
                    "StorageService | 流式上传失败 | bucket={} object={} err={}",
                    bucket, object_name, exc,
                )
                raise StorageError(f"上传失败: {exc}") from exc

        return await asyncio.to_thread(_do)

    async def download(self, bucket: str, object_name: str) -> bytes:
        """
        下载文件，返回原始字节。

        Raises:
            StorageError: 文件不存在或下载失败
        """
        def _do() -> bytes:
            try:
                client = self._get_client()
                response = client.get_object(bucket, object_name)
                try:
                    content = response.read()
                finally:
                    response.close()
                    response.release_conn()
                return content
            except S3Error as exc:
                if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                    logger.debug(
                        "StorageService | 对象不存在 | bucket={} object={}",
                        bucket, object_name,
                    )
                    raise StorageObjectNotFoundError(f"对象不存在: {bucket}/{object_name}") from exc
                logger.error(
                    "StorageService | 下载失败 | bucket={} object={} err={}",
                    bucket, object_name, exc,
                )
                raise StorageError(f"下载失败: {exc}") from exc
            except Exception as exc:
                logger.error(
                    "StorageService | 下载失败 | bucket={} object={} err={}",
                    bucket, object_name, exc,
                )
                raise StorageError(f"下载失败: {exc}") from exc

        return await asyncio.to_thread(_do)

    async def delete(self, bucket: str, object_name: str) -> None:
        """
        删除指定对象。对象不存在时静默成功。

        Raises:
            StorageError: 删除失败（权限不足等）
        """
        def _do() -> None:
            try:
                client = self._get_client()
                client.remove_object(bucket, object_name)
                logger.info(
                    "StorageService | 删除成功 | bucket={} object={}",
                    bucket, object_name,
                )
            except Exception as exc:
                logger.error(
                    "StorageService | 删除失败 | bucket={} object={} err={}",
                    bucket, object_name, exc,
                )
                raise StorageError(f"删除失败: {exc}") from exc

        await asyncio.to_thread(_do)

    async def exists(self, bucket: str, object_name: str) -> bool:
        """
        检查对象是否存在。

        Returns:
            True 表示存在，False 表示不存在
        """
        def _do() -> bool:
            try:
                client = self._get_client()
                client.stat_object(bucket, object_name)
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_do)

    async def presigned_url(
        self,
        bucket: str,
        object_name: str,
        *,
        expires_s: int = 300,
    ) -> str:
        """生成对象下载用的短时预签名 URL。"""
        if expires_s <= 0:
            raise ValueError("expires_s 必须大于 0")

        def _do() -> str:
            try:
                return self._get_client().presigned_get_object(
                    bucket,
                    object_name,
                    expires=timedelta(seconds=expires_s),
                )
            except Exception as exc:
                logger.error(
                    "StorageService | 生成预签名URL失败 | bucket={} object={} err={}",
                    bucket,
                    object_name,
                    exc,
                )
                raise StorageError(f"生成预签名URL失败: {exc}") from exc

        return await asyncio.to_thread(_do)

    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[str]:
        """
        列出桶内对象名称列表。

        Args:
            bucket:    桶名
            prefix:    前缀过滤（如 "docs/"），默认不过滤
            recursive: 是否递归子目录，默认 True

        Returns:
            对象名称列表（不含桶名前缀）

        Raises:
            StorageError: 查询失败
        """
        def _do() -> list[str]:
            try:
                client = self._get_client()
                objects = client.list_objects(bucket, prefix=prefix, recursive=recursive)
                return [obj.object_name for obj in objects if obj.object_name is not None]
            except Exception as exc:
                logger.error(
                    "StorageService | 列表查询失败 | bucket={} prefix={} err={}",
                    bucket, prefix, exc,
                )
                raise StorageError(f"列表查询失败: {exc}") from exc

        return await asyncio.to_thread(_do)

    async def ensure_bucket(self, bucket: str) -> None:
        """
        确保桶存在，不存在则创建。

        Raises:
            StorageError: 创建失败
        """
        def _do() -> None:
            try:
                client = self._get_client()
                self._ensure_bucket(client, bucket)
            except Exception as exc:
                logger.error(
                    "StorageService | 桶操作失败 | bucket={} err={}",
                    bucket, exc,
                )
                raise StorageError(f"桶操作失败: {exc}") from exc

        await asyncio.to_thread(_do)

    async def set_bucket_lifecycle(self, bucket: str, *, expiration_days: int) -> None:
        """
        设置桶的对象生命周期过期规则。

        Raises:
            StorageError: 设置生命周期失败
        """
        def _do() -> None:
            try:
                client = self._get_client()
                self._ensure_bucket(client, bucket)
                lifecycle_config = LifecycleConfig(
                    [
                        Rule(
                            ENABLED,
                            rule_filter=Filter(prefix=""),
                            rule_id="expire-all-objects",
                            expiration=Expiration(days=expiration_days),
                        ),
                    ],
                )
                client.set_bucket_lifecycle(bucket, lifecycle_config)
                logger.info(
                    "StorageService | 桶生命周期已设置 | bucket={} expiration_days={}",
                    bucket, expiration_days,
                )
            except Exception as exc:
                logger.error(
                    "StorageService | 桶生命周期设置失败 | bucket={} expiration_days={} err={}",
                    bucket, expiration_days, exc,
                )
                raise StorageError(f"桶生命周期设置失败: {exc}") from exc

        await asyncio.to_thread(_do)


# 模块级单例，直接 import 使用
storage_service = StorageService()
