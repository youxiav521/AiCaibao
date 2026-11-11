"""测试 ModelFactory 在各种配置场景下的 get_model / get_model_by_id 行为

测试目标：
  - 默认路径（无 key / 无 vision）
  - VL 路径（vision=True）
  - 自定义模型路径（key=...）
  - 参数覆盖（overrides）
  - extra_body 合并优先级
  - supports_vision / get_provider 辅助方法
  - 初始化配置校验
  - provider 级别限流配置（concurrency/rps/rpm）
  - RateLimitedChatModel 包装行为
  - _is_limit_enabled 判断逻辑
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from src.utils.model_factory import ModelFactory, Config


# ─── 辅助 Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singletons():
    """每次测试前重置 ModelFactory 和 Config 的单例状态"""
    ModelFactory._initialized = False
    ModelFactory._providers = {}
    ModelFactory._custom_models = {}
    ModelFactory._logged_missing_keys = set()
    ModelFactory._logged_vision_warnings = set()
    Config._initialized = False
    Config._config = {}
    yield


@pytest.fixture
def mock_chat_model():
    """mock 的 BaseChatModel 实例，替代 init_chat_model 的返回值"""
    return MagicMock(spec=BaseChatModel)


# ─── 测试用配置数据 ──────────────────────────────────────────────────────────

# 基础配置：无任何限流（默认行为，等同于全部设为 None）
BASIC_CONFIG = {
    "providers": [
        {
            "name": "test_provider",
            "api_base": "http://test.base/v1",
            "api_key": "test_key",
            "extra_body": {"provider_option": True},
            "models": [
                {"name": "test-model", "image": True},
                {"name": "test-model-no-image", "image": False},
            ],
            # 未配置任何限流项 → 不包装 RateLimitedChatModel
        }
    ],
    "default_provider": "test_provider",
    "default_model": "test-model",
    "default_temperature": 0.1,
}

# 带有限流配置的 provider（RPM 优先模式）
LIMITER_CONFIG = {
    "providers": [
        {
            "name": "limited_provider",
            "api_base": "http://limited.base/v1",
            "api_key": "test_key",
            "models": [{"name": "limited-model", "image": False}],
            # provider 级别限流配置
            "concurrency": 10,
            "rpm": 100,
            "max_retries": 3,
            "retry_delay": 5.0,
            "max_wait": 60.0,
        },
        {
            "name": "mixed_provider",
            "api_base": "http://mixed.base/v1",
            "api_key": "test_key",
            "models": [{"name": "mixed-model", "image": False}],
            # 只配置 concurrency → 仍会包装
            "concurrency": 5,
        },
        {
            "name": "disabled_provider",
            "api_base": "http://disabled.base/v1",
            "api_key": "test_key",
            "models": [{"name": "disabled-model", "image": False}],
            # -1 表示关闭限制 → 不包装
            "concurrency": -1,
            "rps": -1,
            "rpm": -1,
        },
    ],
    "default_provider": "limited_provider",
    "default_model": "limited-model",
    "default_temperature": 0.1,
}

CUSTOM_MODEL_CONFIG = {
    **BASIC_CONFIG,
    "custom_models": {
        "my-custom": {
            "model": "test-model-no-image",
            "provider": "test_provider",
        },
        "my-custom-extra": {
            "model": "test-model",
            "provider": "test_provider",
            "extra_body": {"custom_option": True},
        },
    },
}

MULTI_PROVIDER_CONFIG = {
    "providers": [
        {
            "name": "provider_a",
            "api_base": "http://a.base/v1",
            "api_key": "key_a",
            "extra_body": {},
            "models": [{"name": "model-a", "image": True}],
        },
        {
            "name": "provider_b",
            "api_base": "http://b.base/v1",
            "api_key": "key_b",
            "extra_body": {},
            "models": [{"name": "model-b", "image": False}],
        },
    ],
    "default_provider": "provider_a",
    "default_model": "model-a",
    "default_temperature": 0.1,
}


def _use_config(config: dict, monkeypatch):
    """用 monkeypatch 将 Config.load 固定返回指定配置"""
    monkeypatch.setattr(Config, "load", lambda: config)


# ═══════════════════════════════════════════════════════════════════════════════
# 默认路径：get_model() / get_model_by_id()
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetModelDefault:
    """get_model() 无 key 无 vision —— 使用 default_provider + default_model"""

    def test_uses_default_provider_and_model(self, monkeypatch, mock_chat_model):
        _use_config(BASIC_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            result = ModelFactory.get_model()

        # 默认始终被 RateLimitedChatModel 包装（确保 metrics 可用）
        assert result is not mock_chat_model
        assert hasattr(result, "_model")
        assert result._model is mock_chat_model
        mock_init.assert_called_once()
        _, kwargs = mock_init.call_args
        assert kwargs["model"] == "test-model"
        assert kwargs["base_url"] == "http://test.base/v1"
        assert kwargs["api_key"] == "test_key"
        assert kwargs["temperature"] == 0.1

    def test_passes_provider_extra_body(self, monkeypatch, mock_chat_model):
        _use_config(BASIC_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model()

        _, kwargs = mock_init.call_args
        assert kwargs["extra_body"] == {"provider_option": True}

    def test_applies_overrides(self, monkeypatch, mock_chat_model):
        _use_config(BASIC_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(temperature=0.9, max_tokens=2048)

        _, kwargs = mock_init.call_args
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 2048

    def test_extra_body_override_merges_correctly(self, monkeypatch, mock_chat_model):
        """override 的 extra_body 与 provider extra_body 合并，同名 key 以 override 为准"""
        config = {
            **BASIC_CONFIG,
            "providers": [
                {
                    **BASIC_CONFIG["providers"][0],
                    "extra_body": {"provider_option": True, "shared_key": "from_provider"},
                }
            ],
        }
        _use_config(config, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(extra_body={"override_key": True, "shared_key": "from_override"})

        _, kwargs = mock_init.call_args
        assert kwargs["extra_body"]["provider_option"] is True
        assert kwargs["extra_body"]["override_key"] is True
        assert kwargs["extra_body"]["shared_key"] == "from_override"

    def test_model_provider_defaults_to_openai(self, monkeypatch, mock_chat_model):
        _use_config(BASIC_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model()

        _, kwargs = mock_init.call_args
        assert kwargs["model_provider"] == "openai"


# ═══════════════════════════════════════════════════════════════════════════════
# VL 路径：get_model(vision=True)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetModelVision:
    """get_model(vision=True) —— 使用 VL 配置或回落至默认"""

    def test_uses_vl_config_when_configured(self, monkeypatch, mock_chat_model):
        config = {
            **BASIC_CONFIG,
            "default_vl_provider": "test_provider",
            "default_vl_model": "test-model",
        }
        _use_config(config, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(vision=True)

        _, kwargs = mock_init.call_args
        assert kwargs["model"] == "test-model"

    def test_falls_back_to_default_when_vl_not_configured(self, monkeypatch, mock_chat_model):
        """未配置 default_vl_provider / default_vl_model 时回落至 default_provider / default_model"""
        _use_config(BASIC_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(vision=True)

        _, kwargs = mock_init.call_args
        assert kwargs["model"] == "test-model"


# ═══════════════════════════════════════════════════════════════════════════════
# key + vision：get_model(key=..., vision=True)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetModelKeyWithVision:
    """get_model(key=..., vision=True) —— key 存在时校验 vision 能力，不存在时回落 VL 默认"""

    def test_key_supports_vision_no_warning(self, monkeypatch, mock_chat_model):
        config = {
            **BASIC_CONFIG,
            "custom_models": {
                "vl-key": {
                    "model": "test-model",
                    "provider": "test_provider",
                },
            },
        }
        _use_config(config, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            with patch("src.utils.model_factory.logger.warning") as mock_warning:
                ModelFactory.get_model(key="vl-key", vision=True)

        _, kwargs = mock_init.call_args
        assert kwargs["model"] == "test-model"
        mock_warning.assert_not_called()

    def test_key_not_supports_vision_logs_warning(self, monkeypatch, mock_chat_model):
        config = {
            **BASIC_CONFIG,
            "custom_models": {
                "non-vl-key": {
                    "model": "test-model-no-image",
                    "provider": "test_provider",
                },
            },
        }
        _use_config(config, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            with patch("src.utils.model_factory.logger.warning") as mock_warning:
                ModelFactory.get_model(key="non-vl-key", vision=True)

        _, kwargs = mock_init.call_args
        assert kwargs["model"] == "test-model-no-image"
        mock_warning.assert_called_once()
        assert "不支持 vision 能力" in mock_warning.call_args[0][0]

    def test_key_missing_falls_back_to_vl_default(self, monkeypatch, mock_chat_model):
        config = {
            **BASIC_CONFIG,
            "default_vl_provider": "test_provider",
            "default_vl_model": "test-model",
        }
        _use_config(config, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(key="nonexistent", vision=True)

        _, kwargs = mock_init.call_args
        assert kwargs["model"] == "test-model"


# ═══════════════════════════════════════════════════════════════════════════════
# 自定义模型路径：get_model(key=...)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetModelCustomKey:
    """get_model(key=...) —— 使用 custom_models 映射"""

    def test_uses_custom_model_when_key_exists(self, monkeypatch, mock_chat_model):
        _use_config(CUSTOM_MODEL_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(key="my-custom")

        _, kwargs = mock_init.call_args
        assert kwargs["model"] == "test-model-no-image"

    def test_falls_back_to_default_when_key_missing(self, monkeypatch, mock_chat_model):
        """不存在的 key 回退到默认模型并记录一条 info 日志"""
        _use_config(CUSTOM_MODEL_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            with patch("src.utils.model_factory.logger.info") as mock_info:
                ModelFactory.get_model(key="nonexistent")

        _, kwargs = mock_init.call_args
        assert kwargs["model"] == "test-model"
        # logger.info 被调用 2 次：初始化完成日志 + 缺失 key 日志
        mock_info.assert_any_call("未找到自定义模型配置: {}，使用默认模型", "nonexistent")

    def test_only_logs_missing_key_once(self, monkeypatch, mock_chat_model):
        """同一个缺失的 key 在整个生命周期中只记录一次日志"""
        _use_config(CUSTOM_MODEL_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model):
            with patch("src.utils.model_factory.logger.info") as mock_info:
                ModelFactory.get_model(key="nonexistent")
                ModelFactory.get_model(key="nonexistent")

        # call_count == 2：初始化完成日志(1) + 缺失 key 日志(1)，第二次调用不再记录
        assert mock_info.call_count == 2

    def test_custom_extra_body_merges_with_provider(self, monkeypatch, mock_chat_model):
        """自定义模型的 extra_body 与 provider extra_body 合并"""
        _use_config(CUSTOM_MODEL_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(key="my-custom-extra")

        _, kwargs = mock_init.call_args
        assert kwargs["extra_body"]["provider_option"] is True
        assert kwargs["extra_body"]["custom_option"] is True

    def test_custom_extra_body_overrides_provider(self, monkeypatch, mock_chat_model):
        """自定义模型的 extra_body 同名 key 覆盖 provider"""
        config = {
            **BASIC_CONFIG,
            "custom_models": {
                "my-override": {
                    "model": "test-model",
                    "provider": "test_provider",
                    "extra_body": {"provider_option": False, "custom_only": True},
                },
            },
        }
        _use_config(config, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(key="my-override")

        _, kwargs = mock_init.call_args
        assert kwargs["extra_body"]["provider_option"] is False
        assert kwargs["extra_body"]["custom_only"] is True

    def test_custom_extra_body_combined_with_override_arg(self, monkeypatch, mock_chat_model):
        """自定义模型的 extra_body 在 merged_overrides 中最后合并，优先级高于 overrides 参数"""
        config = {
            **BASIC_CONFIG,
            "custom_models": {
                "my-stacked": {
                    "model": "test-model",
                    "provider": "test_provider",
                    "extra_body": {"from_custom": True, "shared": "custom"},
                },
            },
        }
        _use_config(config, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            ModelFactory.get_model(key="my-stacked", extra_body={"from_arg": True, "shared": "arg"})

        _, kwargs = mock_init.call_args
        assert kwargs["extra_body"]["from_custom"] is True
        assert kwargs["extra_body"]["from_arg"] is True
        # custom.extra_body 在 merged_overrides 中最后合并，覆盖同名字段
        assert kwargs["extra_body"]["shared"] == "custom"


# ═══════════════════════════════════════════════════════════════════════════════
# supports_vision
# ═══════════════════════════════════════════════════════════════════════════════

class TestSupportsVision:
    """ModelFactory.supports_vision()"""

    def test_vision_supported(self, monkeypatch):
        _use_config(BASIC_CONFIG, monkeypatch)
        assert ModelFactory.supports_vision("test_provider", "test-model") is True

    def test_vision_not_supported(self, monkeypatch):
        _use_config(BASIC_CONFIG, monkeypatch)
        assert ModelFactory.supports_vision("test_provider", "test-model-no-image") is False

    def test_vision_default_model(self, monkeypatch):
        """不传 model_name 时使用 default_model"""
        _use_config(BASIC_CONFIG, monkeypatch)
        assert ModelFactory.supports_vision("test_provider") is True


# ═══════════════════════════════════════════════════════════════════════════════
# get_provider
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetProvider:
    """ModelFactory.get_provider()"""

    def test_get_default_provider(self, monkeypatch):
        _use_config(BASIC_CONFIG, monkeypatch)
        provider = ModelFactory.get_provider()
        assert provider.name == "test_provider"

    def test_get_specific_provider(self, monkeypatch):
        _use_config(MULTI_PROVIDER_CONFIG, monkeypatch)
        provider = ModelFactory.get_provider("provider_b")
        assert provider.name == "provider_b"
        assert provider.api_base == "http://b.base/v1"

    def test_get_nonexistent_provider_raises(self, monkeypatch):
        _use_config(BASIC_CONFIG, monkeypatch)
        with pytest.raises(ValueError, match="未找到 provider"):
            ModelFactory.get_provider("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════════
# 限流配置相关测试：get_provider_limits / _is_limit_enabled
# ═══════════════════════════════════════════════════════════════════════════════

class TestLimiterConfig:
    """限流配置相关测试：provider 级别限流配置读取与判断逻辑"""

    def test_get_provider_limits_reads_all_fields(self, monkeypatch):
        """get_provider_limits 正确读取 provider 配置中的所有限流字段"""
        _use_config(LIMITER_CONFIG, monkeypatch)
        limits = Config.get_provider_limits("limited_provider")

        assert limits["concurrency"] == 10
        assert limits["rpm"] == 100
        assert limits["rps"] is None  # 未配置则为 None
        assert limits["max_retries"] == 3
        assert limits["retry_delay"] == 5.0
        assert limits["max_wait"] == 60.0

    def test_get_provider_limits_returns_empty_for_unknown(self, monkeypatch):
        """未知 provider 返回空 dict"""
        _use_config(LIMITER_CONFIG, monkeypatch)
        limits = Config.get_provider_limits("nonexistent")
        assert limits == {}

    def test_is_limit_enabled_none_is_false(self, monkeypatch):
        """None 表示关闭限制"""
        _use_config(BASIC_CONFIG, monkeypatch)
        assert Config._is_limit_enabled(None) is False

    def test_is_limit_enabled_negative_is_false(self, monkeypatch):
        """-1 或负数表示关闭限制"""
        _use_config(BASIC_CONFIG, monkeypatch)
        assert Config._is_limit_enabled(-1) is False
        assert Config._is_limit_enabled(0) is False

    def test_is_limit_enabled_positive_is_true(self, monkeypatch):
        """正数表示启用限制"""
        _use_config(BASIC_CONFIG, monkeypatch)
        assert Config._is_limit_enabled(10) is True
        assert Config._is_limit_enabled(0.5) is True

    def test_provider_with_no_limits_still_wrapped(self, monkeypatch, mock_chat_model):
        """即使未配置任何限流项，也始终包装 RateLimitedChatModel（确保 metrics 可用）"""
        _use_config(BASIC_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            result = ModelFactory.get_model()

        # 始终被包装（但内部限流逻辑会直接透传）
        assert result is not mock_chat_model
        assert hasattr(result, "_model")
        assert result._model is mock_chat_model
        mock_init.assert_called_once()

    def test_provider_with_disabled_limits_still_wrapped(self, monkeypatch, mock_chat_model):
        """即使所有限流都设为 -1（禁用），也始终包装 RateLimitedChatModel"""
        _use_config(LIMITER_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            result = ModelFactory.get_model_by_id(
                model_name="disabled-model",
                provider_name="disabled_provider",
            )

        # 始终被包装（但内部限流逻辑会直接透传）
        assert result is not mock_chat_model
        assert hasattr(result, "_model")
        assert result._model is mock_chat_model
        mock_init.assert_called_once()

    def test_rate_limited_false_skips_wrapping(self, monkeypatch, mock_chat_model):
        """rate_limited=False 参数跳过限流包装（用于测试/脚本等场景）"""
        _use_config(LIMITER_CONFIG, monkeypatch)
        with patch("src.utils.model_factory.init_chat_model", return_value=mock_chat_model) as mock_init:
            result = ModelFactory.get_model(rate_limited=False)

        # 即使有限流配置，rate_limited=False 也不包装
        assert result is mock_chat_model
        mock_init.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# 初始化校验
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    """ModelFactory._initialize() 阶段的配置校验"""

    def test_missing_default_model_raises(self, monkeypatch):
        config = {**BASIC_CONFIG, "default_model": "nonexistent-model"}
        _use_config(config, monkeypatch)
        with pytest.raises(ValueError, match="默认模型.*不在 provider"):
            ModelFactory._initialize()

    def test_missing_vl_model_raises(self, monkeypatch):
        config = {
            **BASIC_CONFIG,
            "default_vl_provider": "test_provider",
            "default_vl_model": "nonexistent-model",
        }
        _use_config(config, monkeypatch)
        with pytest.raises(ValueError, match="默认 VL 模型.*不在 provider"):
            ModelFactory._initialize()

    def test_vl_model_without_vision_raises(self, monkeypatch):
        """VL 模型在 providers 中存在但 image=False 时也应报错"""
        config = {
            **BASIC_CONFIG,
            "default_vl_provider": "test_provider",
            "default_vl_model": "test-model-no-image",
        }
        _use_config(config, monkeypatch)
        with pytest.raises(ValueError, match="不支持 vision"):
            ModelFactory._initialize()

    def test_custom_model_bad_ref_raises(self, monkeypatch):
        """custom_models 引用的模型不在 provider 的 models 列表中"""
        config = {
            **BASIC_CONFIG,
            "custom_models": {
                "bad-ref": {
                    "model": "nonexistent-model",
                    "provider": "test_provider",
                },
            },
        }
        _use_config(config, monkeypatch)
        with pytest.raises(ValueError, match="自定义模型.*不在 provider"):
            ModelFactory._initialize()

    def test_no_vl_validation_when_vl_not_configured(self, monkeypatch):
        """未配置 default_vl_model 时不进行 VL 校验，即使默认模型不支持 vision 也不报错"""
        config = {
            **BASIC_CONFIG,
            "default_model": "test-model-no-image",
        }
        _use_config(config, monkeypatch)
        ModelFactory._initialize()
