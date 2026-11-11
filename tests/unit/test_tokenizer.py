# -*- coding: utf-8 -*-
"""src/utils/tokenizer.py 的单元测试。

验证目标
--------
1. 缓存目录解析正确指向 <project_root>/tiktoken_cache/
2. tiktoken_cache/ 缺失时 Tokenizer 构造立即抛出 RuntimeError（fail-fast）
3. tokenizer(text) / count / limit / ensure 行为正确
4. 通过 content_generate_agent.utils 的封装函数仍可正常使用（向后兼容）
5. ragflow_parsers 的 num_tokens_from_string 通过 Tokenizer 驱动
"""

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_exists() -> bool:
    from src.utils.tokenizer import _resolve_cache_dir
    try:
        return _resolve_cache_dir() is not None
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# 缓存目录解析
# ---------------------------------------------------------------------------

class TestResolveCacheDir:
    def test_returns_path_when_exists(self):
        from src.utils.tokenizer import _resolve_cache_dir
        # tests/unit/test_tokenizer.py → parents[2] = project_root
        project_root = Path(__file__).resolve().parents[2]
        expected = project_root / "tiktoken_cache"
        if expected.is_dir():
            result = _resolve_cache_dir()
            assert isinstance(result, Path)
            assert result == expected
        else:
            with pytest.raises(RuntimeError, match="tiktoken_cache"):
                _resolve_cache_dir()

    def test_raises_when_cache_missing(self, monkeypatch, tmp_path):
        """缓存目录不存在时应抛出 RuntimeError。"""
        from src.utils import tokenizer as tok_mod
        # 临时修改 __file__ 路径，使 parents[2] 指向一个不存在 tiktoken_cache 的目录
        monkeypatch.setattr(tok_mod, "__file__", str(tmp_path / "src" / "utils" / "tokenizer.py"))
        with pytest.raises(RuntimeError, match="tiktoken_cache/ 目录不存在"):
            tok_mod._resolve_cache_dir()


# ---------------------------------------------------------------------------
# _require_local_cache — fail-fast 行为
# ---------------------------------------------------------------------------

class TestRequireLocalCache:
    def test_raises_when_cache_missing_and_no_env(self, monkeypatch, tmp_path):
        """缓存目录不存在且无环境变量时，应立即抛出 RuntimeError。"""
        from src.utils import tokenizer as tok_mod
        monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
        monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
        # 临时修改 __file__ 路径，使 parents[2] 指向一个不存在 tiktoken_cache 的目录
        monkeypatch.setattr(tok_mod, "__file__", str(tmp_path / "src" / "utils" / "tokenizer.py"))
        with pytest.raises(RuntimeError, match="tiktoken_cache/ 目录不存在"):
            tok_mod._require_local_cache()

    def test_no_raise_when_env_set(self, monkeypatch):
        """TIKTOKEN_CACHE_DIR 已由外部设置时不应抛出，也不应覆盖。"""
        from src.utils.tokenizer import _require_local_cache
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "/some/external/cache")
        _require_local_cache()  # 不应抛异常
        assert os.environ["TIKTOKEN_CACHE_DIR"] == "/some/external/cache"

    def test_sets_env_from_local_dir(self, monkeypatch):
        """本地缓存存在时，应将 TIKTOKEN_CACHE_DIR 设置为该路径。"""
        if not _cache_exists():
            pytest.skip("本地 tiktoken_cache/ 不存在")
        from src.utils.tokenizer import _require_local_cache, _resolve_cache_dir
        monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
        monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
        _require_local_cache()
        assert os.environ.get("TIKTOKEN_CACHE_DIR") == str(_resolve_cache_dir())


# ---------------------------------------------------------------------------
# Tokenizer 构造
# ---------------------------------------------------------------------------

class TestTokenizerInit:
    def test_loads_successfully(self):
        """cl100k_base 编码应成功加载。"""
        if not _cache_exists():
            pytest.skip("本地 tiktoken_cache/ 不存在")
        from src.utils.tokenizer import Tokenizer
        tok = Tokenizer()
        assert repr(tok) == "Tokenizer('cl100k_base')"

    def test_raises_when_cache_dir_missing(self, monkeypatch, tmp_path):
        """tiktoken_cache/ 目录不存在时，构造应立即抛出 RuntimeError。"""
        from src.utils import tokenizer as tok_mod
        monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
        monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
        # 临时修改 __file__ 路径，使 parents[2] 指向一个不存在 tiktoken_cache 的目录
        monkeypatch.setattr(tok_mod, "__file__", str(tmp_path / "src" / "utils" / "tokenizer.py"))
        with pytest.raises(RuntimeError, match="tiktoken_cache/ 目录不存在"):
            tok_mod.Tokenizer()


# ---------------------------------------------------------------------------
# Tokenizer.__call__ / count
# ---------------------------------------------------------------------------

class TestTokenizerCount:
    @pytest.fixture(scope="class")
    def tok(self):
        if not _cache_exists():
            pytest.skip("本地 tiktoken_cache/ 不存在")
        from src.utils.tokenizer import Tokenizer
        return Tokenizer()

    def test_call_returns_int(self, tok):
        result = tok("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_count_equals_call(self, tok):
        text = "一些测试文本"
        assert tok.count(text) == tok(text)

    def test_empty_string(self, tok):
        assert tok("") == 0

    def test_longer_text_more_tokens(self, tok):
        assert tok("Hi") < tok("Hello, world! This is a longer sentence.")


# ---------------------------------------------------------------------------
# Tokenizer.limit
# ---------------------------------------------------------------------------

class TestTokenizerLimit:
    @pytest.fixture(scope="class")
    def tok(self):
        if not _cache_exists():
            pytest.skip("本地 tiktoken_cache/ 不存在")
        from src.utils.tokenizer import Tokenizer
        return Tokenizer()

    def test_short_text_unchanged(self, tok):
        text = "short text"
        assert tok.limit(text, max_tokens=1000) == text

    def test_long_text_truncated(self, tok):
        long = "token " * 2000
        result = tok.limit(long, max_tokens=100)
        assert len(result) < len(long)

    def test_default_suffix_appended(self, tok):
        result = tok.limit("token " * 2000, max_tokens=50)
        assert "截断" in result

    def test_custom_suffix(self, tok):
        result = tok.limit("word " * 2000, max_tokens=50, suffix="[CUT]")
        assert result.endswith("[CUT]")

    def test_empty_suffix(self, tok):
        result = tok.limit("word " * 2000, max_tokens=50, suffix="")
        assert "截断" not in result

    def test_result_within_token_limit(self, tok):
        """截断后的 token 数（不含 suffix）应 <= max_tokens。"""
        max_tokens = 100
        result = tok.limit("word " * 2000, max_tokens=max_tokens, suffix="")
        assert tok(result) <= max_tokens


# ---------------------------------------------------------------------------
# Tokenizer.ensure
# ---------------------------------------------------------------------------

class TestTokenizerEnsure:
    @pytest.fixture(scope="class")
    def tok(self):
        if not _cache_exists():
            pytest.skip("本地 tiktoken_cache/ 不存在")
        from src.utils.tokenizer import Tokenizer
        return Tokenizer()

    def test_short_text_returned_as_is(self, tok):
        text = "Hello world"
        assert tok.ensure(text, max_tokens=30_000) == text

    def test_long_text_truncated(self, tok):
        long = "token " * 50_000
        result = tok.ensure(long, max_tokens=500)
        assert len(result) < len(long)

    def test_no_exception_on_truncation(self, tok):
        tok.ensure("token " * 50_000, max_tokens=100)  # should not raise


# ---------------------------------------------------------------------------
# 向后兼容：content_generate_agent/utils.py 的封装函数
# ---------------------------------------------------------------------------

class TestBackwardCompatContent:
    def test_count_tokens(self):
        from src.agents.content_generate_agent.utils import count_tokens
        assert count_tokens("Hello") > 0

    def test_count_tokens_with_tokenizer_arg(self):
        """传入 Tokenizer 实例时应使用它，而非模块默认值。"""
        if not _cache_exists():
            pytest.skip("本地 tiktoken_cache/ 不存在")
        from src.utils.tokenizer import Tokenizer
        from src.agents.content_generate_agent.utils import count_tokens
        tok = Tokenizer()
        assert count_tokens("hello world", tok) > 0

    def test_truncate_to_token_limit(self):
        from src.agents.content_generate_agent.utils import truncate_to_token_limit
        result = truncate_to_token_limit("hello world", max_tokens=1000)
        assert result == "hello world"

    def test_ensure_within_token_limit(self):
        from src.agents.content_generate_agent.utils import ensure_within_token_limit
        result = ensure_within_token_limit("hello", max_tokens=30_000)
        assert result == "hello"

    def test_get_default_tokenizer_returns_tokenizer(self):
        from src.utils.tokenizer import Tokenizer
        from src.agents.content_generate_agent.utils import get_default_tokenizer
        tok = get_default_tokenizer()
        assert isinstance(tok, Tokenizer)

    def test_default_max_tokens(self):
        from src.agents.content_generate_agent.utils import DEFAULT_MAX_TOKENS
        assert DEFAULT_MAX_TOKENS == 30_000


# ---------------------------------------------------------------------------
# 向后兼容：ragflow_parsers 的 num_tokens_from_string
# ---------------------------------------------------------------------------

class TestBackwardCompatRagflow:
    def test_num_tokens_from_string(self):
        from src.applications.parsing.utils.ragflow_parsers import num_tokens_from_string
        result = num_tokens_from_string("Hello, RAGFlow!")
        assert isinstance(result, int)
        assert result > 0

    def test_module_level_tokenizer_is_tokenizer_instance(self):
        from src.utils.tokenizer import Tokenizer
        from src.applications.parsing.utils import ragflow_parsers
        assert isinstance(ragflow_parsers._tokenizer, Tokenizer)
