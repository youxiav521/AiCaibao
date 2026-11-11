"""应用配置模块"""
from typing import ClassVar, Optional, Set, Any

from loguru import logger
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.crypt.decryptor_registry import decrypt_value


# ── 嵌套子配置模型 ────────────────────────────────────────────────────────────

class ServerConfig(BaseModel):
    name: str = "yusys-ai"
    version: str = "1.0.0"


class LoggerConfig(BaseModel):
    level: str = "INFO"


class SwaggerConfig(BaseModel):
    enabled: bool = True


class NacosConfig(BaseModel):
    enabled: bool = False
    serverAddr: str = "localhost:8848"
    namespace: str = "public"
    group: str = "DEFAULT_GROUP"
    instanceIp: Optional[str] = None
    instancePort: int = 9999
    username: Optional[str] = None
    password: Optional[str] = None


class EventLoopMonitorConfig(BaseModel):
    enabled: bool = True
    threshold_ms: int = 200
    poll_ms: int = 50
    stack_depth: int = 25


class DefaultExecutorMonitorConfig(BaseModel):
    enabled: bool = True
    max_workers: Optional[int] = None
    saturation_threshold_ms: int = 0
    emit_queue_metrics: bool = True


class MonitorConfig(BaseModel):
    enabled: bool = False
    event_loop_monitor: EventLoopMonitorConfig = Field(default_factory=EventLoopMonitorConfig)
    default_executor_monitor: DefaultExecutorMonitorConfig = Field(default_factory=DefaultExecutorMonitorConfig)


class PrometheusConfig(BaseModel):
    enabled: bool = False


class TracingConfig(BaseModel):
    """yusys_tracing 独立 Agent tracing 配置（不影响 Langfuse）。"""

    enabled: bool = False
    backend: str = "yusys"  # yusys | langfuse
    otlp_endpoint: Optional[str] = None  # 例如 http://localhost:4318
    project_id: str = "yusys-ai"
    service_name: str = "yusys-ai"
    agent_name: str = "yusys-ai"
    environment: str = "development"
    max_body_chars: int = 2000


class MinioConfig(BaseModel):
    """MinIO 对象存储配置。endpoint 格式为 host:port，不含协议前缀。"""
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False
    default_bucket: str = "yusys"


# ── 主配置类 ──────────────────────────────────────────────────────────────────

class Config(BaseSettings):
    """应用配置类 — 字段自动从环境变量 / .env 文件读取。

    嵌套子配置使用 __ 分隔符，例如：
        SERVER__NAME=yusys-ai
        NACOS__ENABLED=false
        MONITOR__EVENT_LOOP_MONITOR__THRESHOLD_MS=200
        MINIO__ENDPOINT=localhost:9000
    """

    model_config = SettingsConfigDict(
        env_file=(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # 服务配置
    server: ServerConfig = Field(default_factory=ServerConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    swagger: SwaggerConfig = Field(default_factory=SwaggerConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    nacos: NacosConfig = Field(default_factory=NacosConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)

    # Redis配置
    REDIS_MODE: str = "standalone"
    REDIS_HOST: str = ""
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    REDIS_SENTINELS: str = ""
    REDIS_SENTINEL_MASTER_NAME: str = "mymaster"
    REDIS_SENTINEL_USERNAME: str = ""
    REDIS_SENTINEL_PASSWORD: str = ""

    # 文件存储配置（已废弃：生产持久化统一使用 MinIO）
    STORAGE_PATH: str = "./storage"

    # 文本处理配置
    MAX_TITLE_LENGTH: int = 100
    MAX_DESCRIPTION_LENGTH: int = 200
    MAX_DESCRIPTION_LINES: int = 10

    # AI配置
    AI_MODEL: str = "qwen3-32b"

    # API配置
    API_PREFIX: str = "/api/policy"

    # 应用信息
    APP_NAME: str = "Policy API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "策略文件上传和解析服务"

    # 加密配置
    CRYPT_TYPE: str = "SM"

    # 不从 env 读取的常量 — ClassVar 使 pydantic 跳过这些字段
    ALLOWED_FILE_EXTENSIONS: ClassVar[Set[str]] = {'.txt', '.pdf', '.docx', '.md'}
    API_TAGS: ClassVar[list] = ["Policy"]

    @model_validator(mode="after")
    def decrypt_encrypted_fields(self) -> "Config":
        """解密所有以 DECRYPT 前缀开头的字段值"""
        crypt_type = self.CRYPT_TYPE.strip().upper()
        self._decrypt_recursive(self, crypt_type)
        return self

    def _decrypt_recursive(self, obj: Any, crypt_type: str) -> None:
        """递归解密对象中的所有字段"""
        if isinstance(obj, BaseModel):
            for field_name, _ in type(obj).model_fields.items():
                current_value = getattr(obj, field_name)
                if isinstance(current_value, str):
                    decrypted = decrypt_value(current_value, crypt_type)
                    if decrypted != current_value:
                        setattr(obj, field_name, decrypted)
                elif isinstance(current_value, BaseModel):
                    self._decrypt_recursive(current_value, crypt_type)

    def ensure_config(self) -> None:
        """确保配置已正确加载"""
        self.ensure_redis_configured()

    def ensure_storage_directory(self) -> None:
        """已废弃：启动时不再创建本地持久化存储目录。"""
        return None

    def ensure_redis_configured(self) -> None:
        """检查Redis配置，未配置则抛出异常终止启动"""
        redis_mode = self.REDIS_MODE.strip().lower()
        if redis_mode not in {"standalone", "sentinel"}:
            logger.error("REDIS_MODE 配置无效 | mode={}", self.REDIS_MODE)
            raise ValueError("REDIS_MODE 仅支持 standalone 或 sentinel")

        if redis_mode == "sentinel":
            if not self.REDIS_SENTINELS:
                logger.error("REDIS_SENTINELS 环境变量未配置，请设置 Sentinel 节点列表")
                raise ValueError("REDIS_SENTINELS 环境变量未配置，请设置 Sentinel 节点列表")
            if not self.REDIS_SENTINEL_MASTER_NAME:
                logger.error("REDIS_SENTINEL_MASTER_NAME 环境变量未配置，请设置 Sentinel master 名称")
                raise ValueError("REDIS_SENTINEL_MASTER_NAME 环境变量未配置，请设置 Sentinel master 名称")
            return

        if not self.REDIS_HOST:
            logger.error("REDIS_HOST 环境变量未配置，请设置 REDIS_HOST 才能启动应用")
            raise ValueError("REDIS_HOST 环境变量未配置，请设置 REDIS_HOST 才能启动应用")


# 全局配置单例 — 模块加载时自动读取 .env / 环境变量
config = Config()
config.ensure_config()
