from __future__ import annotations

import os
from pathlib import Path

import tiktoken as _tiktoken
from loguru import logger

DEFAULT_MAX_TOKENS: int = 30_000
DEFAULT_ENCODING: str = "cl100k_base"
DEFAULT_QWEN_TOKENIZER_PATH: str = "config/tokenizers/qwen3_5/tokenizer.json"


def resolve_encoding_for_model(model_name: str | None) -> str:
    """
    Resolve tokenizer encoding by model family.

    Priority:
    1) TOKENIZER_ENCODING_NAME env
    2) model heuristic (Qwen3/Qwen3.5/Qwen3-max -> o200k_base)
    3) cl100k_base
    """
    forced = os.environ.get("TOKENIZER_ENCODING_NAME")
    if forced:
        return forced

    name = (model_name or "").lower()
    if "qwen3" in name or "qwen-3" in name or "qwen3.5" in name or "qwen3-max" in name:
        return "o200k_base"
    return DEFAULT_ENCODING


def _resolve_cache_dir() -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "tiktoken_cache"
    if cache_dir.is_dir():
        return cache_dir

    logger.error("{} 不存在或不是文件夹", str(cache_dir))
    raise RuntimeError(
        "tiktoken_cache/ 目录不存在，无法加载 tokenizer。\n"
        "请确认已将离线缓存随项目一起部署，路径应为 <project_root>/tiktoken_cache/。\n"
        "若使用外部缓存，请设置环境变量 TIKTOKEN_CACHE_DIR=<路径>。"
    )


def _require_local_cache() -> None:
    if os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR"):
        return

    cache_dir = _resolve_cache_dir()
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    logger.info("TIKTOKEN_CACHE_DIR 已设置为本地缓存: {}", str(cache_dir))


class Tokenizer:
    """Token counting / truncation utility based on tiktoken."""

    def __init__(self, encoding_name: str | None = None, model_name: str | None = None) -> None:
        self._mode = "tiktoken"
        self._encoding_name = encoding_name or resolve_encoding_for_model(model_name)
        self._qwen_tokenizer_path = None
        self._qwen_tokenizer = self._try_init_qwen_official(model_name=model_name)
        if self._qwen_tokenizer is not None:
            self._mode = "qwen_official"
            logger.info("Tokenizer 'qwen_official' 已就绪 | path={}", self._qwen_tokenizer_path)
            return

        _require_local_cache()
        self._enc = self._load_encoding(self._encoding_name)
        logger.info("Tokenizer '{}' 已就绪", self._encoding_name)

    def _try_init_qwen_official(self, model_name: str | None = None):
        """
        Try loading Qwen official tokenizer.json via `tokenizers`.
        Enabled when:
        - TOKENIZER_IMPL is not set to `tiktoken`
        - and model_name looks like qwen OR TOKENIZER_IMPL explicitly `qwen_official`
        """
        impl = (os.environ.get("TOKENIZER_IMPL") or "").strip().lower()
        if impl == "tiktoken":
            return None

        model_lc = (model_name or "").lower()
        prefer_qwen = impl == "qwen_official" or "qwen" in model_lc or bool(os.environ.get("QWEN_TOKENIZER_JSON_PATH"))
        if not prefer_qwen:
            return None

        try:
            from tokenizers import Tokenizer as HFTokenizer  # type: ignore
        except Exception:
            logger.warning("tokenizers 未安装，无法使用 Qwen 官方 tokenizer，回退到 tiktoken")
            return None

        project_root = Path(__file__).resolve().parents[2]
        candidates = []
        env_path = os.environ.get("QWEN_TOKENIZER_JSON_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())
        candidates.append(project_root / DEFAULT_QWEN_TOKENIZER_PATH)

        for path in candidates:
            p = path if path.is_absolute() else (project_root / path)
            if p.exists() and p.is_file():
                try:
                    tok = HFTokenizer.from_file(str(p))
                    self._qwen_tokenizer_path = str(p)
                    return tok
                except Exception as e:
                    logger.warning("加载 Qwen tokenizer 失败: {} | error={}", p, e)
                    continue

        logger.warning(
            "未找到可用 Qwen tokenizer.json，已回退 tiktoken。可设置 QWEN_TOKENIZER_JSON_PATH，或放在 {}",
            project_root / DEFAULT_QWEN_TOKENIZER_PATH,
        )
        return None

    @staticmethod
    def _cleanup_tiktoken_cache() -> None:
        """Pre-load housekeeping: clear .tmp leftovers that break Windows rename."""
        cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
        if not cache_dir or not os.path.isdir(cache_dir):
            return
        for name in os.listdir(cache_dir):
            if name.endswith(".tmp"):
                tmp_path = os.path.join(cache_dir, name)
                try:
                    os.remove(tmp_path)
                    logger.warning("清理 tiktoken 残留临时文件: {}", tmp_path)
                except OSError:
                    logger.exception("清理 tiktoken 临时文件失败: {}", tmp_path)

    def _load_encoding(self, encoding_name: str):
        self._cleanup_tiktoken_cache()
        try:
            return _tiktoken.get_encoding(encoding_name)
        except FileExistsError:
            # Windows: os.rename 目标已存在会失败。删除旧缓存后重试一次，
            # 让 tiktoken 重新下载并写入完整文件。
            cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
            if cache_dir and os.path.isdir(cache_dir):
                for name in os.listdir(cache_dir):
                    if not name.endswith(".tmp"):
                        continue
                    base = name[:-4]
                    if "." in base:
                        base = base.split(".", 1)[0]
                    stale = os.path.join(cache_dir, base)
                    if os.path.exists(stale):
                        try:
                            os.remove(stale)
                            logger.warning("删除损坏的 tiktoken 缓存以触发重下载: {}", stale)
                        except OSError:
                            logger.exception("删除 tiktoken 缓存失败: {}", stale)
            self._cleanup_tiktoken_cache()
            try:
                return _tiktoken.get_encoding(encoding_name)
            except Exception:
                if encoding_name != DEFAULT_ENCODING:
                    logger.warning(
                        "Tokenizer encoding {} 加载失败，回退到 {}",
                        encoding_name,
                        DEFAULT_ENCODING,
                    )
                    self._encoding_name = DEFAULT_ENCODING
                    return _tiktoken.get_encoding(DEFAULT_ENCODING)
                raise RuntimeError(f"无法从本地 tiktoken_cache/ 加载编码: {encoding_name}")
        except Exception:
            if encoding_name != DEFAULT_ENCODING:
                logger.warning(
                    "Tokenizer encoding {} 加载失败，回退到 {}",
                    encoding_name,
                    DEFAULT_ENCODING,
                )
                self._encoding_name = DEFAULT_ENCODING
                return _tiktoken.get_encoding(DEFAULT_ENCODING)
            raise RuntimeError(f"无法从本地 tiktoken_cache/ 加载编码: {encoding_name}")

    def __call__(self, text: str) -> int:
        if self._mode == "qwen_official":
            return len(self._qwen_tokenizer.encode(text).ids)
        # 文档内容可能含 <|endoftext|> 等特殊 token 字面量，按普通文本编码
        return len(self._enc.encode(text, disallowed_special=()))

    def count(self, text: str) -> int:
        return self(text)

    def limit(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "...(文本过长，已截断)",
    ) -> str:
        if self._mode == "qwen_official":
            tokens = self._qwen_tokenizer.encode(text).ids
            if len(tokens) <= max_tokens:
                return text
            return self._qwen_tokenizer.decode(tokens[:max_tokens]) + suffix

        tokens = self._enc.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        return self._enc.decode(tokens[:max_tokens]) + suffix

    def ensure(
        self,
        text: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        suffix: str = "...(文本过长，已截断)",
    ) -> str:
        n = self(text)
        if n > max_tokens:
            truncated = self.limit(text, max_tokens, suffix)
            logger.warning(
                "文本超过 token 限制 ({} tokens)，原始 {} tokens，已截断",
                max_tokens,
                n,
            )
            return truncated
        return text

    def __repr__(self) -> str:
        if self._mode == "qwen_official":
            return f"Tokenizer('qwen_official', path='{self._qwen_tokenizer_path}')"
        return f"Tokenizer('{self._encoding_name}')"
