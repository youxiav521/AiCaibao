from __future__ import annotations

import re


def approx_token_len(text: str) -> int:
    """估算中英混合文本的 token 数量。"""
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    total = len(text)
    non_cn = max(total - chinese_chars, 0)
    return int(chinese_chars / 1.6 + non_cn / 4.2)


__all__ = ["approx_token_len"]
