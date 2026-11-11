import os
from dataclasses import dataclass, replace

from loguru import logger

# 【临时测试】强制所有问题都进入 document_reading_retrival 阶段（chunk/summary 后不标记"内容充足"，一律进入 document_reading）
FORCE_ALL_TO_DOCUMENT_READING = False  # 测试完后请改回 False

# 注意：检索阶段配置已移除，现在通过每个问题的 tools 字段来配置检索方式

@dataclass
class TestConfig:
    max_tokens_per_block: int
    min_tokens_per_piece: int
    context_token_limit: int
    # 是否启用摘要 (summary_index) 检索阶段；关闭时 chunk 后直接结束，剩余问题用当前上下文回答
    enable_summary_index_retrieval: bool
    # 是否启用全文 (document_reading) 检索阶段；关闭时 summary 后直接结束，剩余问题用当前上下文回答
    enable_document_reading_retrieval: bool
    # 是否使用 filter_long_context_if_needed 进行上下文筛选（超长截断）；关闭时不做该筛选
    enable_filter_long_context: bool
    # coding 的最大上下文长度
    coding_context_token_limit: int
    # 代码生成最大重试次数
    coding_max_retry_limit: int
    # coding 优先的 Model ID，可选，默认为 None
    coding_model_id: str | None = None
    # reporter 中当 type 为 text 时使用的 Model ID，可选，默认为 None （为None时reporter会使用默认模型）
    reporter_text_model_id: str | None = None
    # reporter 中当 type 为 text-json-citation 时使用的 Model ID，可选，默认为 None （为None时reporter会使用默认模型）
    reporter_text_json_citation_model_id: str | None = None

DEFAULT_TEST_CONFIG = TestConfig(
    max_tokens_per_block=80_000, 
    min_tokens_per_piece=1_000,
    context_token_limit=80_000, # 120k token 限制，留一些余量
    enable_summary_index_retrieval=True,
    enable_document_reading_retrieval=False,
    enable_filter_long_context=False,
    coding_model_id="data_analyst_agent", # 代码生成优先的 Model ID，默认为 model_for_coding
    reporter_text_model_id="reporter_text_model", # reporter 中 type 为 text 时使用的微调模型，默认为 None （为None时reporter会使用默认模型）
    reporter_text_json_citation_model_id="reporter_text_json_citation_model", # reporter 中 type 为 text-json-citation 时使用的微调模型，默认为 None （为None时reporter会使用默认模型）
    coding_context_token_limit=200_000, #250k token 限制时建议最大上下文长度为 200k，代码生成较消耗 token，预留可以更大
    coding_max_retry_limit=10, # 代码生成时最大重试次数，每次代码生成通常会有 1k-5ktoken，应该结合预留 token 进行设置
)


def _get_env_int(name: str) -> int | None:
    """读取整型环境变量；未设置/空值返回 None；无法解析返回 None（并写 warning）。"""
    raw = os.getenv(name)
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        logger.warning("无效的 {}={}，将忽略该覆盖值", name, raw)
        return None


def _apply_env_int_override(cfg: TestConfig, *, env_name: str, field_name: str) -> TestConfig:
    """若 env_name 可解析为 int，则覆盖 cfg 对应 field_name。"""
    v = _get_env_int(env_name)
    if v is None:
        return cfg
    logger.info("配置覆盖：{} -> {}={}", env_name, field_name, v)
    return replace(cfg, **{field_name: v})


def _get_env_bool(name: str) -> bool | None:
    """读取布尔环境变量；未设置/空值返回 None；无法解析返回 None（并写 warning）。"""
    raw = os.getenv(name)
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    if s in truthy:
        return True
    if s in falsy:
        return False
    logger.warning("无效的 {}={}，将忽略该覆盖值", name, raw)
    return None


def _apply_env_bool_override(cfg: TestConfig, *, env_name: str, field_name: str) -> TestConfig:
    """若 env_name 可解析为 bool，则覆盖 cfg 对应 field_name。"""
    v = _get_env_bool(env_name)
    if v is None:
        return cfg
    logger.info("配置覆盖：{} -> {}={}", env_name, field_name, v)
    return replace(cfg, **{field_name: v})


def _apply_env_str_override(cfg: TestConfig, *, env_name: str, field_name: str) -> TestConfig:
    """若 env_name 为非空字符串，则覆盖 cfg 对应 field_name。"""
    raw = os.getenv(env_name)
    if raw is None:
        return cfg
    v = str(raw).strip()
    if not v:
        return cfg
    logger.info("配置覆盖：{} -> {}={}", env_name, field_name, v)
    return replace(cfg, **{field_name: v})


def create_default_configs() -> TestConfig:
    """
    创建默认配置。

    `context_token_limit` 可由环境变量 `CONTENT_GENERATE_CONTEXT_TOKEN_LIMIT` 覆盖
    （整数）。未设置或为空时使用 `DEFAULT_TEST_CONFIG` 中的默认值。

    `coding_context_token_limit` 可由环境变量 `CONTENT_GENERATE_CODING_CONTEXT_TOKEN_LIMIT` 覆盖
    （整数）。未设置或为空时使用 `DEFAULT_TEST_CONFIG` 中的默认值。

    `coding_model_id` 可由环境变量 `CONTENT_GENERATE_CODING_MODEL_ID` 覆盖（字符串）。

    `coding_max_retry_limit` 可由环境变量 `CONTENT_GENERATE_CODING_MAX_RETRY_LIMIT` 覆盖（整数）。

    `enable_summary_index_retrieval` 可由环境变量
    `CONTENT_GENERATE_ENABLE_SUMMARY_INDEX_RETRIEVAL` 覆盖（布尔）。

    `enable_document_reading_retrieval` 可由环境变量
    `CONTENT_GENERATE_ENABLE_DOCUMENT_READING_RETRIEVAL` 覆盖（布尔）。

    `reporter_text_model_id` 可由环境变量 `CONTENT_GENERATE_REPORTER_TEXT_MODEL_ID` 覆盖（字符串）。

    `reporter_text_json_citation_model_id` 可由环境变量 `CONTENT_GENERATE_REPORTER_TEXT_JSON_CITATION_MODEL_ID` 覆盖（字符串）。
    """
    try:
        cfg = DEFAULT_TEST_CONFIG
        cfg = _apply_env_int_override(
            cfg, env_name="CONTENT_GENERATE_CONTEXT_TOKEN_LIMIT", field_name="context_token_limit"
        )
        cfg = _apply_env_int_override(
            cfg,
            env_name="CONTENT_GENERATE_CODING_CONTEXT_TOKEN_LIMIT",
            field_name="coding_context_token_limit",
        )
        cfg = _apply_env_str_override(
            cfg,
            env_name="CONTENT_GENERATE_CODING_MODEL_ID",
            field_name="coding_model_id",
        )
        cfg = _apply_env_str_override(
            cfg,
            env_name="CONTENT_GENERATE_REPORTER_TEXT_MODEL_ID",
            field_name="reporter_text_model_id",
        )
        cfg = _apply_env_str_override(
            cfg,
            env_name="CONTENT_GENERATE_REPORTER_TEXT_JSON_CITATION_MODEL_ID",
            field_name="reporter_text_json_citation_model_id",
        )
        cfg = _apply_env_int_override(
            cfg,
            env_name="CONTENT_GENERATE_CODING_MAX_RETRY_LIMIT",
            field_name="coding_max_retry_limit",
        )
        cfg = _apply_env_bool_override(
            cfg,
            env_name="CONTENT_GENERATE_ENABLE_SUMMARY_INDEX_RETRIEVAL",
            field_name="enable_summary_index_retrieval",
        )
        cfg = _apply_env_bool_override(
            cfg,
            env_name="CONTENT_GENERATE_ENABLE_DOCUMENT_READING_RETRIEVAL",
            field_name="enable_document_reading_retrieval",
        )
        return cfg
    except Exception:
        logger.exception("create_default_configs 环境变量覆盖失败，回退到默认配置")
        return DEFAULT_TEST_CONFIG
