from __future__ import annotations

from typing import Any, Protocol

class Embedder(Protocol):
    """将原始文本编码为稠密向量。"""

    async def aembed_texts(self, texts: list[str], *, stopwatch_task: Any | None = None) -> list[list[float]]:
        """批量编码文本。

        Args:
            texts: 待编码的原始文本列表。
            stopwatch_task: 可选的父级计时任务，用于记录内部耗时。

        Returns:
            与 ``texts`` 一一对应的稠密向量列表。
        """
        ...

    async def aembed_query(self, text: str) -> list[float]:
        """编码单条查询文本。

        Args:
            text: 查询文本。

        Returns:
            查询对应的稠密向量。
        """
        ...

    async def aclose(self) -> None:
        """释放 embedder 关联的异步客户端资源。"""
        ...


class Reranker(Protocol):
    """根据查询对候选文档进行重排。"""

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """异步按相关性重排候选文档。

        Args:
            query: 用户查询。
            documents: 待比较的候选文档列表。
            top_k: 可选的结果上限。

        Returns:
            重排后的结果项列表。每个结果项至少应包含
            ``index`` 和 ``score`` 字段。
        """
        ...
