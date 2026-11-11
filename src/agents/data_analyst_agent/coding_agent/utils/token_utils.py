"""
文档列表的 Token 计算与按 token 划分工具。

用于在调用 run_full_analysis_optional 等流程前，根据 token 上限对 doc_list 进行划分，
避免单次传入内容超过模型限制；划分时保证不拆开单个 document。
"""

from typing import List

from src.agents.data_analyst_agent.coding_agent.models.document_model import Document
from src.agents.content_generate_agent.utils import count_tokens
from loguru import logger


# 默认 token 上限（与常见模型上下文长度对齐，可按需修改）
DEFAULT_TOKEN_LIMIT = 100000
# 默认每个划分块的 token 目标值（单块不超过此值，便于留余量给 prompt 等）
DEFAULT_PARTITION_TOKEN_VALUE = 90000


def count_document_tokens(doc: "Document") -> int:
    """
    计算单个 Document 的 token 数（用于划分时统计，不拆开单个 doc）。

    将 title、text、table 拼接成字符串后计算 token，与下游实际使用方式一致。
    """
    content = f"{doc.title or ''}\n{doc.text or ''}\n{doc.table or ''}"
    try:
        return count_tokens(content)
    except Exception as e:
        logger.warning(f"使用 content_generate_agent.count_tokens 失败: {e}，将使用字符数近似")
        return len(content)


def count_doc_list_tokens(doc_list: List["Document"]) -> int:
    """
    计算 doc_list 中所有 Document 的总 token 数。

    Args:
        doc_list: Document 列表（可为 typing 中的 Document 或 dict，本模块按 Document 使用）。

    Returns:
        总 token 数。
    """
    total = 0
    for doc in doc_list or []:
        content = f"{getattr(doc, 'title', '') or ''}\n{getattr(doc, 'text', '') or ''}\n{getattr(doc, 'table', '') or ''}"
        total += count_tokens(content)
    return total


def split_doc_list_by_token_limit(
    doc_list: List["Document"],
    token_limit: int = DEFAULT_TOKEN_LIMIT,
    partition_token_value: int = DEFAULT_PARTITION_TOKEN_VALUE,
) -> List[List["Document"]]:
    """
    按 token 上限与划分值将 doc_list 划分为多块，且不拆开单个 document。

    当 doc_list 总 token 数不超过 token_limit 时，返回 [doc_list]；
    超过时，按每块不超过 partition_token_value 的规则，将文档整份地分配到各块中
    （即一个 document 只属于一块，不会被截断）。
    """
    if not doc_list:
        return []

    total_tokens = count_doc_list_tokens(doc_list)
    if total_tokens <= token_limit:
        return [doc_list]

    # 计算每个 document 的 token 数，用于按块聚合
    doc_tokens: List[int] = []
    for doc in doc_list:
        content = f"{getattr(doc, 'title', '') or ''}\n{getattr(doc, 'text', '') or ''}\n{getattr(doc, 'table', '') or ''}"
        doc_tokens.append(count_tokens(content))

    chunks: List[List["Document"]] = []
    current_chunk: List["Document"] = []
    current_tokens = 0

    for doc, tokens in zip(doc_list, doc_tokens):
        # 若当前块加上本 doc 会超过划分值，且当前块非空，则先结束当前块
        if current_chunk and (current_tokens + tokens > partition_token_value):
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        # 若单个 doc 就超过划分值，仍单独成块（不拆开 document）
        current_chunk.append(doc)
        current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk)

    logger.info(
        f"doc_list 总 token 数 {total_tokens} 超过上限 {token_limit}，"
        f"已按 partition_token_value={partition_token_value} 划分为 {len(chunks)} 块"
    )
    return chunks
