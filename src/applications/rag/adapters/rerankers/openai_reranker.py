from __future__ import annotations

from typing import Any, Optional

from loguru import logger
from openai import AsyncOpenAI

from src.applications.rag.config.rag_settings import RAG_SETTINGS
from src.applications.rag.core.interfaces import Reranker

QWEN_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN_RERANK_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
QWEN_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class OpenAIReranker(Reranker):
    """基于 OpenAI 兼容 rerank 接口的异步重排实现。"""

    def __init__(self, *, timeout_seconds: int = 20) -> None:
        self.model = RAG_SETTINGS.rerank_model
        self.endpoint_path = _normalize_endpoint_path(
            RAG_SETTINGS.rerank_endpoint_path,
            RAG_SETTINGS.rerank_base_url,
        )
        self.qwen_template_enabled = _is_qwen_reranker(self.model)
        self._timeout_seconds = timeout_seconds
        # 注意：AsyncOpenAI 不能在 __init__ 中创建。
        # openai 库的 AsyncAPIClient.__del__ 会调用
        # asyncio.get_running_loop().create_task(self.aclose())，在 uvicorn --reload
        # 模式下旧 loop 关闭后会抛 RuntimeError: Event loop is closed。
        # 改为首次调用 arerank 时懒加载。
        self._client: Optional[AsyncOpenAI] = None

        logger.info(
            "Reranker 初始化: model={}, base_url={}, qwen_template_enabled={}",
            self.model,
            RAG_SETTINGS.rerank_base_url,
            self.qwen_template_enabled,
        )

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=RAG_SETTINGS.rerank_api_key or "dummy",
                base_url=RAG_SETTINGS.rerank_base_url,
                timeout=self._timeout_seconds,
            )
        return self._client

    async def aclose(self) -> None:
        """显式关闭底层 AsyncOpenAI 客户端，避免 GC 时 __del__ 触发 create_task 报错。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        logger.info("Reranker 请求: model={}, top_k={}, docs_count={}", self.model, top_k, len(documents))

        try:
            payload = self._build_payload(query, documents, top_k)
            data: dict[str, Any] = await self.client.post(
                self.endpoint_path,
                cast_to=dict[str, Any],
                body=payload,
            )
        except Exception:
            logger.exception("Reranker 调用失败")
            return _fallback_results(len(documents))

        results = [_parse_result(item) for item in _response_items(data)]
        results = [item for item in results if item is not None]
        logger.info("Reranker 返回: results_count={}", len(results))
        return results or _fallback_results(len(documents))

    def _build_payload(
        self,
        query: str,
        documents: list[str],
        top_k: int | None,
    ) -> dict[str, Any]:
        request_query, request_documents = self._build_request_inputs(query, documents)
        payload: dict[str, Any] = {
            "model": self.model,
            "query": request_query,
            "documents": request_documents,
        }
        if top_k is not None and top_k > 0:
            payload["top_n"] = top_k
        return payload

    def _build_request_inputs(self, query: str, documents: list[str]) -> tuple[str, list[str]]:
        if not self.qwen_template_enabled:
            return query, documents

        request_query = f"{QWEN_RERANK_PREFIX}<Instruct>: {QWEN_RERANK_INSTRUCTION}\n<Query>: {query}\n"
        request_documents = [f"<Document>: {document}{QWEN_RERANK_SUFFIX}" for document in documents]
        return request_query, request_documents


def _normalize_endpoint_path(endpoint_path: str, base_url: str | None) -> str:
    path = f"/{endpoint_path.strip('/')}"
    if (not base_url or base_url.rstrip("/").endswith("/v1")) and path.startswith("/v1/"):
        return path.removeprefix("/v1")
    return path


def _is_qwen_reranker(model: str) -> bool:
    model_name = model.lower()
    return "qwen" in model_name and "rerank" in model_name


def _response_items(data: dict[str, Any]) -> list[Any]:
    items = data.get("results") or data.get("data") or []
    return items if isinstance(items, list) else []


def _parse_result(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or item.get("index") is None:
        return None

    score = item.get("relevance_score")
    if score is None:
        score = item.get("score")

    return {
        "index": int(item["index"]),
        "score": float(score) if score is not None else None,
    }


def _fallback_results(count: int) -> list[dict[str, Any]]:
    return [{"index": index, "score": None} for index in range(count)]
