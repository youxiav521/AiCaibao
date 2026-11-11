"""RAG MCP 服务：把 POST /api/rag/query 包装为两个 MCP 工具。

工具：
- get_doc_context_by_id：按文件 id 拉全量（对应 kb_use.md 方式 A）
- retrieve_from_kb：基于用户查询 + 文件范围做 RAG（对应方式 B）

同进程内挂载，不走 HTTP 跳跃；最终 URL = /api/rag/mcp。
"""
from __future__ import annotations

from fastmcp import FastMCP

from src.applications.rag.schemas import (
    BatchQueryRequest,
    QuestionItem,
)
from src.applications.rag.services.query_service import run_batch_query

mcp = FastMCP(
    name="rag_mcp",
    instructions=(
        "RAG 检索工具集。get_doc_context_by_id 拉取指定文件全部上下文，"
        "retrieve_from_kb 基于用户问题在指定文件范围（或全库）内做 RAG。"
    ),
)


def _build_request(
    query: str,
    file_ids: list[str],
    top_k: int,
) -> BatchQueryRequest:
    """构造内部 BatchQueryRequest，question_id 统一为 'q1'。"""
    return BatchQueryRequest(
        questions=[QuestionItem(question_id="q1", question_text=query)],
        top_k=top_k,
        file_ids=file_ids or None,
        folder_id=None,
    )


def _to_chunk_dicts(items) -> list[dict]:
    """把 BatchQueryItem 列表规整为 kb_use.md §3.3 约定的内部结构。"""
    out: list[dict] = []
    for idx, it in enumerate(items, 1):
        out.append(
            {
                "id": f"{idx:04d}",
                "doc_id": it.file_id,
                "chunk_id": it.element_ids or [],
                "content": it.text or "",
                "source": it.file_name or "",
            }
        )
    return out


@mcp.tool(
    name="get_doc_context_by_id",
    description=(
        "按 file_id 拉取该文件的全部 chunk 上下文。"
        "适用于：列出文件全部内容、做总结/翻译/抽取。"
    ),
    tags={"rag", "full"},
)
async def get_doc_context_by_id(file_id: str) -> list[dict]:
    """通过文件 id 拉取文档或图片的完整内容。"""
    req = _build_request(query="全部", file_ids=[file_id], top_k=100)
    items = await run_batch_query(req)
    return _to_chunk_dicts(items)


@mcp.tool(
    name="retrieve_from_kb",
    description=(
        "在指定 file_ids 范围内对用户问题做 RAG 检索。"
        "file_ids 为空时退化为全库检索。"
    ),
    tags={"rag", "search"},
)
async def retrieve_from_kb(
    query: str,
    file_ids: list[str] = [],
    top_k: int = 15,
) -> list[dict]:
    """基于用户问题在指定文件范围（或全库）内做 RAG 检索。"""
    req = _build_request(query=query, file_ids=file_ids, top_k=top_k)
    items = await run_batch_query(req)
    return _to_chunk_dicts(items)


# 暴露给 FastAPI 挂载；最终 URL = /api/rag/mcp
#
# stateless_http=True 关键意义：
# - 每个请求创建独立 transport，跨 worker / 跨连接无 session 绑定
# - 适配 gunicorn 多 worker（docker/dev/docker-compose.yml 用 -w 6）部署
# - 关闭 GET 长连接（streamable_http 在 stateless 模式下不支持 server-initiated）
# - POST + DELETE 仍走完整 MCP 协议（initialize / tools/list / tools/call 等）
mcp_app = mcp.http_app(path="/mcp", transport="streamable-http", stateless_http=True)
