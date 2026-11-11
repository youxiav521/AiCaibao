"""
ADE Extract — LLM 模式工具函数

包含：
  - chunk 窗口切割（LangExtract 风格）
  - 多窗口结果合并
"""
import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.utils.tokenizer import Tokenizer


# ------------------------------------------------------------------ #
# 窗口切割
# ------------------------------------------------------------------ #

def locate_chunks_in_markdown(
    markdown: str,
    chunks: List[Dict],
) -> List[Tuple[int, int, str]]:
    """
    在 markdown 中定位每个 chunk 的起止位置。

    markdown 已经是 id+type+text 的拼接文本，每个 chunk 的 markdown 片段
    是全文 markdown 的子串，按顺序查找即可。

    Returns:
        List[(start_pos, end_pos, chunk_id)]，按 chunk 顺序排列
    """
    spans: List[Tuple[int, int, str]] = []
    search_from = 0

    for chunk in chunks:
        chunk_id = chunk.get("id", "")
        chunk_md = chunk.get("markdown", "")
        if not chunk_md:
            continue

        pos = markdown.find(chunk_md, search_from)
        if pos == -1:
            pos = markdown.find(chunk_md)

        if pos != -1:
            end = pos + len(chunk_md)
            spans.append((pos, end, chunk_id))
            search_from = end
        else:
            logger.warning("chunk id={} 在 markdown 中未找到，跳过", chunk_id)

    return spans


def build_chunk_windows(
    markdown: str,
    chunks: List[Dict],
    max_tokens: int,
    tok: Tokenizer,
) -> List[str]:
    """
    按 chunk id 边界切割 markdown，LangExtract 风格滑动窗口。

    策略：
    - 在现成 markdown 中按 chunk id 定位每个 chunk 的起止位置
    - 贪心聚合连续 chunk，累计 token 达到阈值时切窗
    - 下一窗口包含上一窗口最后一个 chunk 的内容作为 previous context（overlap）

    Returns:
        List[str]，每个元素是一个窗口的 markdown 文本
    """
    if not chunks:
        return [markdown]

    chunk_spans = locate_chunks_in_markdown(markdown, chunks)
    if not chunk_spans:
        return [markdown]

    windows: List[str] = []
    i = 0
    prev_overlap_text: str = ""

    while i < len(chunk_spans):
        window_chunks: List[Tuple[int, int, str]] = []
        window_tokens = tok.count(prev_overlap_text)

        while i < len(chunk_spans):
            start, end, chunk_id = chunk_spans[i]
            chunk_text = markdown[start:end]
            chunk_tokens = tok.count(chunk_text)

            if window_tokens + chunk_tokens > max_tokens and window_chunks:
                break

            window_chunks.append((start, end, chunk_id))
            window_tokens += chunk_tokens
            i += 1

        if not window_chunks:
            # 单个 chunk 自身超限，兜底强制纳入
            start, end, chunk_id = chunk_spans[i]
            window_chunks.append((start, end, chunk_id))
            i += 1

        main_start = window_chunks[0][0]
        main_end = window_chunks[-1][1]
        main_text = markdown[main_start:main_end]

        if prev_overlap_text:
            window_text = (
                f"[Previous Context]\n{prev_overlap_text}\n\n"
                f"[Main Window]\n{main_text}"
            )
        else:
            window_text = main_text

        windows.append(window_text)

        last_start, last_end, _ = window_chunks[-1]
        prev_overlap_text = markdown[last_start:last_end]

    return windows if windows else [markdown]


# ------------------------------------------------------------------ #
# 多窗口结果合并
# ------------------------------------------------------------------ #

def merge_raw_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个窗口的 LLM 原始返回结果。

    合并规则：
    - 有值覆盖空值（null / None）
    - 两者都有值时，前一段（先出现）的值优先保留
    - 数组类型：拼接所有元素（去重）
    """
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    merged: Dict[str, Any] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        _merge_into(merged, result)
    return merged


def _merge_into(base: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    """递归将 incoming 合并进 base。"""
    for k, v in incoming.items():
        if k.startswith("_"):
            continue

        if k not in base:
            base[k] = v
            continue

        base_v = base[k]

        # 叶子节点：{value, chunkId}
        if isinstance(base_v, dict) and isinstance(v, dict):
            if "value" in base_v or "value" in v:
                base_val = base_v.get("value") if isinstance(base_v, dict) else base_v
                inc_val = v.get("value") if isinstance(v, dict) else v
                if _is_empty(base_val) and not _is_empty(inc_val):
                    # 有值覆盖空值
                    base[k] = v
                # 都有值：保留前一段（base 不变）
                # base_val 为空且 inc_val 也为空：不变
            else:
                # 普通嵌套对象，递归合并
                _merge_into(base_v, v)
            continue

        # 数组：拼接（去重）
        if isinstance(base_v, list) and isinstance(v, list):
            seen = {_stable_key(item) for item in base_v}
            for item in v:
                key = _stable_key(item)
                if key not in seen:
                    base_v.append(item)
                    seen.add(key)
            continue

        # 标量：有值覆盖空值，都有值保留前一段
        if _is_empty(base_v) and not _is_empty(v):
            base[k] = v


def _is_empty(val: Any) -> bool:
    """判断值是否为空（None、空字符串、空列表、空字典）。"""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, (list, dict)) and len(val) == 0:
        return True
    return False


def _stable_key(item: Any) -> str:
    """生成列表元素的去重 key。"""
    try:
        return json.dumps(item, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(item)
