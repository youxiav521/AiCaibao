"""繁体中文 → 简体中文转换工具

使用 opencc-python-reimplemented 实现繁→简转换。
全局单例，线程安全（OpenCC 内部只读查表）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from loguru import logger

try:
    from opencc import OpenCC

    _cc_t2s: Optional[OpenCC] = OpenCC("t2s")
except Exception as e:  # pragma: no cover
    logger.warning("opencc-python-reimplemented 不可用，繁→简转换将跳过: {}", e)
    _cc_t2s = None


def traditional_to_simplified(text: str) -> str:
    """将繁体中文文本转换为简体中文，非中文字符不受影响。

    若 opencc 未安装或初始化失败，原样返回。
    """
    if not text or _cc_t2s is None:
        return text
    return _cc_t2s.convert(text)


@lru_cache(maxsize=1)
def _build_sample_chars() -> set[str]:
    """高频繁体字采样集（用于快速检测）。"""
    samples = (
        "國際與於這個從來說對進種經動還現開發點報設計書認證費資產負債權損應營業務員會計審議價處學習環體質數據認識調節項運輸"
    )
    simplified = traditional_to_simplified(samples)
    return {t for t, s in zip(samples, simplified) if t != s}


def is_traditional_chinese(text: str, threshold: float = 0.03) -> bool:
    """抽样检测文本是否以繁体中文为主。

    通过对比转换前后差异字符占比判断（阈值默认 3%）。
    """
    if not text or _cc_t2s is None:
        return False
    sample_chars = _build_sample_chars()
    cjk_count = 0
    trad_hit = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            cjk_count += 1
            if ch in sample_chars:
                trad_hit += 1
    if cjk_count == 0:
        return False
    return trad_hit / cjk_count > threshold
