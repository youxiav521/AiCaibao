from langchain_core.language_models import BaseChatModel
import json
import os
from typing import Optional, Dict, Any, List
from langchain.chat_models import init_chat_model
from loguru import logger
import httpx

# openai SDK 默认超时与连接限制，可通过环境变量覆盖
# OPENAI_TIMEOUT: 总超时（秒），OPENAI_CONNECT_TIMEOUT: 连接建立超时（秒）
# OPENAI_MAX_CONNECTIONS: 最大连接数，OPENAI_MAX_KEEPALIVE_CONNECTIONS: 最大保活连接数
_OPENAI_DEFAULT_TIMEOUT = httpx.Timeout(
    timeout=float(os.getenv("OPENAI_TIMEOUT", "600")),
    connect=float(os.getenv("OPENAI_CONNECT_TIMEOUT", "5.0")),
)
_OPENAI_DEFAULT_LIMITS = httpx.Limits(
    max_connections=int(os.getenv("OPENAI_MAX_CONNECTIONS", "1000")),
    max_keepalive_connections=int(os.getenv("OPENAI_MAX_KEEPALIVE_CONNECTIONS", "100")),
)

from .model_enhancements import RateLimitedChatModel


class Provider:
    def __init__(
        self,
        name: str,
        api_base: str,
        api_key: str,
        extra_body: Optional[Dict[str, Any]] = None,
        http_proxy: Optional[str] = None,
        models: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.api_base = api_base
        self.api_key = api_key
        self.extra_body = extra_body or {}
        self.http_proxy = http_proxy
        self.models = models or []
        self._http_client: Optional[httpx.Client] = None
        self._async_http_client: Optional[httpx.AsyncClient] = None

    def get_http_client(self) -> Optional[httpx.Client]:
        if self.http_proxy and self._http_client is None:
            self._http_client = httpx.Client(
                proxy=self.http_proxy,
                timeout=_OPENAI_DEFAULT_TIMEOUT,
                limits=_OPENAI_DEFAULT_LIMITS,
                follow_redirects=True,
            )
        return self._http_client

    def get_async_http_client(self) -> Optional[httpx.AsyncClient]:
        if self.http_proxy and self._async_http_client is None:
            self._async_http_client = httpx.AsyncClient(
                proxy=self.http_proxy,
                timeout=_OPENAI_DEFAULT_TIMEOUT,
                limits=_OPENAI_DEFAULT_LIMITS,
                follow_redirects=True,
            )
        return self._async_http_client

    def close(self) -> None:
        """释放 httpx 客户端资源，应在应用关闭时调用。"""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        if self._async_http_client is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_http_client.aclose())
                else:
                    loop.run_until_complete(self._async_http_client.aclose())
            except Exception:
                pass
            self._async_http_client = None

    def has_model(self, model_name: str) -> bool:
        for m in self.models:
            if isinstance(m, str):
                if m == model_name:
                    return True
            elif isinstance(m, dict):
                if m.get("name") == model_name:
                    return True
        return False

    def supports_vision(self, model_name: str) -> bool:
        for m in self.models:
            if isinstance(m, dict) and m.get("name") == model_name:
                return m.get("image", False)
        return False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Provider":
        name = config.get("name", "")
        if not name:
            raise ValueError("Provider 配置中缺少 name 字段")
        return cls(
            name=name,
            api_base=config.get("api_base", ""),
            api_key=config.get("api_key", ""),
            extra_body=config.get("extra_body", {}),
            http_proxy=config.get("http_proxy"),
            models=config.get("models", []),
        )


class CustomModel:
    def __init__(
        self,
        key: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.key = key
        self.model = model
        self.provider = provider
        self.extra_body = extra_body or {}

    @classmethod
    def from_config(cls, key: str, config: Any) -> Optional["CustomModel"]:
        if isinstance(config, str):
            return cls(key=key, model=config)
        elif isinstance(config, dict):
            return cls(
                key=key,
                model=config.get("model"),
                provider=config.get("provider"),
                extra_body=config.get("extra_body", {}),
            )
        return None


class Config:
    _config: Dict[str, Any] = {}
    _config_path: str = "config/llm.json"
    _initialized: bool = False

    @classmethod
    def load(cls) -> Dict[str, Any]:
        if cls._initialized:
            return cls._config

        try:
            with open(cls._config_path, "r", encoding="utf-8") as f:
                cls._config = json.load(f)

            providers = cls._config.get("providers", [])
            default_provider = cls._config.get("default_provider", "")

            if not providers:
                raise ValueError("配置文件中没有找到任何 providers")

            # 如果没有设置默认 provider
            if not default_provider:
                if len(providers) == 1:
                    # 只有一个 provider 时，自动设为默认
                    default_provider = providers[0].get("name", "")
                    cls._config["default_provider"] = default_provider
                    logger.info("自动设置默认 provider: {}", default_provider)
                else:
                    # 多个 provider 时，必须指定默认
                    raise ValueError(
                        "配置文件中有多个 providers，但未设置 default_provider"
                    )
            elif not any(p.get("name") == default_provider for p in providers):
                raise ValueError(
                    f"默认 provider '{default_provider}' 未在 providers 中定义"
                )

            logger.info(
                f"Config 加载配置成功 | providers={[p.get('name') for p in providers]}, default_provider={default_provider}"
            )

            cls._initialized = True
            return cls._config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {cls._config_path} 不存在")
        except json.JSONDecodeError:
            raise ValueError("配置文件格式错误")

    @classmethod
    def get_providers(cls) -> List[Dict[str, Any]]:
        return cls.load().get("providers", [])

    @classmethod
    def get_default_provider(cls) -> str:
        return cls.load().get("default_provider", "")

    @classmethod
    def get_default_model(cls) -> str:
        return cls.load().get("default_model", "")

    @classmethod
    def get_default_vl_provider(cls) -> str:
        return cls.load().get("default_vl_provider", "")

    @classmethod
    def get_default_vl_model(cls) -> str:
        return cls.load().get("default_vl_model", "")

    @classmethod
    def resolve_vl_config(cls):
        """解析 VL 配置，返回 (provider_name, model_name)。

        当 default_vl_provider 未设置时回退到 default_provider。
        若未配置任何 VL 字段则返回 ("", "")。
        """
        config = cls.load()
        provider = config.get("default_vl_provider", "") or cls.get_default_provider()
        model = config.get("default_vl_model", "") or cls.get_default_model()
        return provider, model

    @classmethod
    def get_custom_models(cls) -> Dict[str, Any]:
        return cls.load().get("custom_models", {})

    @classmethod
    def get_default_temperature(cls) -> float:
        return cls.load().get("default_temperature", 0.1)

    @classmethod
    def get_default_max_tokens(cls) -> Optional[int]:
        return cls.load().get("default_max_tokens")

    @classmethod
    def get_default_frequency_penalty(cls) -> Optional[float]:
        return cls.load().get("default_frequency_penalty")

    @classmethod
    def get_default_presence_penalty(cls) -> Optional[float]:
        return cls.load().get("default_presence_penalty")

    @classmethod
    def get_provider_limits(cls, provider_name: str) -> Dict[str, Any]:
        """获取 provider 的限流相关配置

        只从 provider 自身配置中读取，无全局默认
        约定: 值为 None / -1 表示关闭该限制，正数表示启用该限制
        """
        for provider in cls.get_providers():
            if provider.get("name") == provider_name:
                return {
                    "concurrency": provider.get("concurrency"),
                    "rps": provider.get("rps"),
                    "rpm": provider.get("rpm"),
                    "max_retries": provider.get("max_retries", 5),
                    "retry_delay": provider.get("retry_delay", 20.0),
                    "max_wait": provider.get("max_wait", 600.0),
                }
        return {}

    @classmethod
    def _is_limit_enabled(cls, value) -> bool:
        """判断某个限制是否启用（None 或 -1 表示关闭，正数表示启用）"""
        if value is None:
            return False
        if isinstance(value, (int, float)) and value <= 0:
            return False
        return True


class ModelFactory:
    _providers: Dict[str, Provider] = {}
    _custom_models: Dict[str, CustomModel] = {}
    _initialized: bool = False
    _logged_missing_keys: set = set()
    _logged_vision_warnings: set = set()

    @classmethod
    def _initialize(cls) -> None:
        if cls._initialized:
            return

        cls._providers = {
            p["name"]: Provider.from_config(p) for p in Config.get_providers()
        }
        cls._custom_models = {
            key: cfg
            for key, cfg in (
                (key, CustomModel.from_config(key, cfg))
                for key, cfg in Config.get_custom_models().items()
            )
            if cfg is not None
        }

        cls._validate_config()

        logger.info(
            f"ModelFactory 初始化完成 | providers={list(cls._providers.keys())}, custom_models={list(cls._custom_models.keys())}"
        )
        cls._initialized = True

    @classmethod
    def _validate_config(cls) -> None:
        default_model = Config.get_default_model()
        default_provider_name = Config.get_default_provider()
        if default_model and default_provider_name:
            provider = cls._providers.get(default_provider_name)
            if provider and provider.models:
                if not provider.has_model(default_model):
                    raise ValueError(
                        f"默认模型 '{default_model}' 不在 provider '{default_provider_name}' 的 models 列表中"
                    )

        vl_model = Config.get_default_vl_model()
        if vl_model:
            vl_provider_name = Config.get_default_vl_provider() or Config.get_default_provider()
            provider = cls._providers.get(vl_provider_name)
            if provider and provider.models:
                if not provider.has_model(vl_model):
                    raise ValueError(
                        f"默认 VL 模型 '{vl_model}' 不在 provider '{vl_provider_name}' 的 models 列表中"
                    )
                if not provider.supports_vision(vl_model):
                    raise ValueError(
                        f"默认 VL 模型 '{vl_model}' 不支持 vision 功能"
                    )

        for key, custom in cls._custom_models.items():
            if custom.provider and custom.model:
                provider = cls._providers.get(custom.provider)
                if provider and provider.models:
                    if not provider.has_model(custom.model):
                        raise ValueError(
                            f"自定义模型 '{key}' 引用的模型 '{custom.model}' 不在 provider '{custom.provider}' 的 models 列表中"
                        )

    @classmethod
    def supports_vision(cls, provider_name: Optional[str] = None, model_name: Optional[str] = None) -> bool:
        cls._initialize()
        provider = cls.get_provider(provider_name)
        model = model_name or Config.get_default_model()
        return provider.supports_vision(model)

    @classmethod
    def get_provider(cls, provider_name: Optional[str] = None) -> Provider:
        cls._initialize()
        name = provider_name or Config.get_default_provider()
        provider = cls._providers.get(name)
        if not provider:
            raise ValueError(
                f"未找到 provider: {name}, 可用的 providers: {list(cls._providers.keys())}"
            )
        return provider

    @classmethod
    def get_model_by_id(
        cls,
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
        overrides=None,
        rate_limited: bool = True,
    ) -> BaseChatModel:
        cls._initialize()

        provider = cls.get_provider(provider_name)
        model = model_name or Config.get_default_model()
        temp = Config.get_default_temperature()

        model_config = {
            "model": model,
            "base_url": provider.api_base,
            "api_key": provider.api_key,
            "model_provider": "openai",
            "temperature": temp,
            # 显式传入 timeout，确保无 http_proxy 时也生效
            "timeout": _OPENAI_DEFAULT_TIMEOUT,
        }

        max_tokens = Config.get_default_max_tokens()
        if max_tokens is not None:
            model_config["max_tokens"] = max_tokens

        freq_penalty = Config.get_default_frequency_penalty()
        if freq_penalty is not None:
            model_config["frequency_penalty"] = freq_penalty

        presence_penalty = Config.get_default_presence_penalty()
        if presence_penalty is not None:
            model_config["presence_penalty"] = presence_penalty

        http_client = provider.get_http_client()
        async_http_client = provider.get_async_http_client()
        if http_client is not None:
            model_config["http_client"] = http_client
        if async_http_client is not None:
            model_config["http_async_client"] = async_http_client

        # 合并 extra_body：overrides 优先级最高
        provider_extra_body = provider.extra_body or {}
        overrides_extra_body = (overrides or {}).get("extra_body", {})
        # 自定义模型可通过 null 删除 provider 层不适用的默认参数。
        merged_extra_body = {
            key: value
            for key, value in {**provider_extra_body, **overrides_extra_body}.items()
            if value is not None
        }
        if merged_extra_body:
            model_config["extra_body"] = merged_extra_body

        # 将 overrides 更新到 model_config 中（但 extra_body 已单独处理）
        if overrides:
            for k, v in overrides.items():
                if k != "extra_body":
                    model_config[k] = v

        try:
            model = init_chat_model(**model_config)
        except Exception as e:
            raise RuntimeError(f"创建模型失败: {e}")

        # 始终用 RateLimitedChatModel 包装，确保 metrics 统计始终可用
        # 注意：无限流配置时，限流器会直接透传，仅保留 metrics 回调功能
        # 独立脚本、冒烟测试可传 rate_limited=False，完全跳过包装
        if rate_limited:
            provider_config = Config.get_provider_limits(provider.name)

            # 约定: None 或 -1 表示关闭该限制，正数表示启用
            concurrency_limit = provider_config.get("concurrency") if Config._is_limit_enabled(provider_config.get("concurrency")) else None
            rps_limit = provider_config.get("rps") if Config._is_limit_enabled(provider_config.get("rps")) else None
            rpm_limit = provider_config.get("rpm") if Config._is_limit_enabled(provider_config.get("rpm")) else None

            # 这里要用 model_name（字符串），不能用 model（对象）
            model_key = f"{provider.name}:{model_name}"
            model = RateLimitedChatModel(
                model=model,
                model_key=model_key,
                provider_name=provider.name,
                concurrency=concurrency_limit,
                rps=rps_limit,
                rpm=rpm_limit,
                max_retries=provider_config.get("max_retries", 5),
                max_wait=provider_config.get("max_wait", 600.0),
                retry_initial_delay=provider_config.get("retry_delay", 20.0),
            )

        return model

    @classmethod
    def get_model(cls, key: Optional[str] = None, vision: bool = False, rate_limited: bool = True, **overrides) -> BaseChatModel:
        cls._initialize()

        custom = cls._custom_models.get(key) if key else None

        if custom and custom.model:
            model_name = custom.model
            provider_name = custom.provider
            merged_overrides = {**overrides}
            if custom.extra_body:
                existing_extra_body = merged_overrides.get("extra_body", {})
                merged_overrides["extra_body"] = {**existing_extra_body, **custom.extra_body}
        else:
            if key:
                if key not in cls._logged_missing_keys:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                    cls._logged_missing_keys.add(key)
                    logger.info("未找到自定义模型配置: {}，使用默认模型", key)

            if vision:
                provider_name, model_name = Config.resolve_vl_config()
            else:
                provider_name = Config.get_default_provider()
                model_name = Config.get_default_model()
            merged_overrides = {**overrides}

        if vision:
            provider = cls.get_provider(provider_name)
            if not provider.supports_vision(model_name):
                if model_name not in cls._logged_vision_warnings:
                    cls._logged_vision_warnings.add(model_name)
                    logger.warning("⚠️ 模型 '{}' 不支持 vision 能力", model_name)

        return cls.get_model_by_id(
            model_name=model_name,
            provider_name=provider_name,
            overrides=merged_overrides,
            rate_limited=rate_limited,
        )

    @classmethod
    def resolved_model_name_for_key(cls, key: Optional[str]) -> str:
        """与 `get_model(key=...)` 一致的解析规则：custom_models 命中则取其 model，否则用 default_model。"""
        cls._initialize()
        custom = cls._custom_models.get(key) if key else None
        if custom and custom.model:
            return custom.model
        return Config.get_default_model()

    @classmethod
    def is_deepseek_model_for_key(cls, key: Optional[str]) -> bool:
        """DeepSeek 兼容网关通常不支持 json_schema 形态的 response_format，结构化输出需走 json_mode。"""
        name = cls.resolved_model_name_for_key(key)
        return "deepseek" in (name or "").lower()

    @classmethod
    def health_check(cls) -> bool:
        default_model = Config.get_default_model()
        if not default_model:
            logger.warning("⚠️ 未配置 default_model，跳过 LLM 健康检查")
            return False

        try:
            # rate_limited=False 避免走 RateLimitedChatModel.invoke 的 async fallback 路径
            # （该路径会创建临时 loop 并通过 ainvoke 调用，导致 @lru_cache 缓存的
            # _AsyncHttpxClientWrapper 绑定到临时 loop；临时 loop 关闭后，
            # GC 触发 __del__ → create_task(aclose()) 在主 loop 中执行 aclose() 时
            # 会因 self._loop 已关闭而抛 RuntimeError: Event loop is closed）
            model = cls.get_model_by_id(
                model_name=default_model,
                overrides={"temperature": 0, "max_tokens": 3},
                rate_limited=False,
            )
            response = model.invoke("Response word 'pong' only without any explanation")
            content = response.content if hasattr(response, "content") else str(response)
            if content and content.strip():
                logger.info(
                    "✅ LLM 健康检查通过 | model={}, response='{}'",
                    default_model,
                    str(content)[:10],
                )
                return True

            logger.warning("⚠️ LLM 健康检查异常 | model={}, response 为空", default_model)
            return False
        except Exception:
            logger.exception("❌ LLM 健康检查失败 | model={}", default_model)
            return False


ModelFactory._initialize()

import threading
threading.Thread(target=ModelFactory.health_check, daemon=True, name="llm-health-check").start()
