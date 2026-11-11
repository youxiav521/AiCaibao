from __future__ import annotations

from typing import Optional, List
import asyncio
from .tree_builder import Node
from ..input_models import extract_text_yusys_ai
from ..utils import apply_prompt_template, ainvoke_llm_with_prompt, abatch_llm_with_prompt
from ..utils import ensure_within_token_limit, count_tokens
from ..config import create_default_configs

_CFG = create_default_configs()
_CTX_LIMIT = _CFG.context_token_limit
# 摘要树生成专用总预算，低于模型上限，以规避百炼侧 token 计数差异导致超 129024
SUMMARY_CTX_LIMIT = 50_000
from ....utils.langfuse_tracing import trace_span
from loguru import logger

class SummaryGenerator:
    
    def __init__(self, root: Node, question: Optional[str] = None, extra_instruction: str = ""):
        self.root = root

    # @trace_span(name="SummaryGenerator.get_node_str_content")
    def get_node_str_content(self, node: Node) -> str:
        parts: list[str] = []
        for it in node.content:
            text = extract_text_yusys_ai(it)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    # _post_order_summarize 已废弃，请使用 run_async 或 run_batch

    # 同步方式已废弃，请使用 run_async 或 run_batch
    # 保留此方法仅用于向后兼容，内部使用异步实现
    @trace_span(name="SummaryGenerator.run")
    def run(self):
        import asyncio
        async def _run_async():
            await self.run_async()
        asyncio.run(_run_async())
        return

    # @trace_span(name="SummaryGenerator._post_order_get_leafs")
    # 异步方式进行summarize的辅助方法: 先得到所有叶子节点
    def _post_order_get_leafs(self, leafs: List[Node], node: Node):
        # 非叶子节点
        if node.childs: 
            for child in node.childs:
                # 递归收集叶子节点
                self._post_order_get_leafs(leafs, child)
            return
        # 叶子节点：添加到leafs之后统一处理
        leafs.append(node)
        return 

    # 异步方式调用，并将summary填入字段
    @trace_span(name="SummaryGenerator._llm_summarize_async")
    async def _llm_summarize_async(self, node: Node):
        """
        使用 Prompt 模板与统一 LLM 调用工具进行摘要。
        总 prompt token 预算：先计算模板固定部分 token 开销，再用 max_total - overhead 得到可用于 context_text 的预算。
        """
        content = self.get_node_str_content(node)

        # 总 prompt token 预算：先计算模板固定部分（仅指令，不含 context_text）
        max_total = SUMMARY_CTX_LIMIT
        template_prompt = apply_prompt_template(
            "summary_generation",
            {"context_text": ""},
        )
        template_overhead = count_tokens(template_prompt)
        available_tokens = max_total - template_overhead
        
        if available_tokens <= 0:
            logger.error(
                "SummaryGenerator._llm_summarize_async: 模板 token 数({})已超过总预算({})，将不附加任何上下文内容。",
                template_overhead,
                max_total,
            )
            processed_content = ""
        else:
            # 检查并在“可用于上下文的 token 预算”内截断
            processed_content = ensure_within_token_limit(content, max_tokens=available_tokens)

        prompt = apply_prompt_template(
            "summary_generation",
            {
                "context_text": processed_content,
            },
        )
        summary = await ainvoke_llm_with_prompt(prompt)
        node.summary = summary
        return 

        
    # 异步方法对外接口
    @trace_span(name="SummaryGenerator.run_async")
    def run_async(self)-> None:

        leafs: List[Node] = []
        self._post_order_get_leafs(leafs = leafs, node = self.root)

        # 在新线函数运行事件循环
        async def main():
            tasks = []
            for leaf_node in leafs:
                task = self._llm_summarize_async(leaf_node)
                tasks.append(task)
            return await asyncio.gather(*tasks)
        asyncio.run(main())

        return 


    # batch方式调用（已改用异步方法）
    @trace_span(name="SummaryGenerator._llm_summarize_batch")
    async def _llm_summarize_batch(self, nodes: List[Node]):
        """
        使用 Prompt 模板与统一 LLM 调用工具进行摘要（异步版本）。
        总 prompt token 预算：先计算模板固定部分 token 开销，再用 max_total - overhead 得到可用于 context_text 的预算。
        """
        prompts = []

        # 总 prompt token 预算：模板固定部分（仅指令，不含 context_text），所有节点共享
        max_total = SUMMARY_CTX_LIMIT
        template_prompt = apply_prompt_template(
            "summary_generation",
            {"context_text": ""},
        )
        template_overhead = count_tokens(template_prompt)
        available_tokens = max_total - template_overhead
        logger.info(
            "摘要树(_llm_summarize_batch): 模板 token 开销={}, 可用于 context_text 的 token 预算={}",
            template_overhead,
            available_tokens,
        )
        if available_tokens <= 0:
            logger.error(
                "SummaryGenerator._llm_summarize_batch: 模板 token 数({})已超过总预算({})，将不附加任何上下文内容。",
                template_overhead,
                max_total,
            )

        for node in nodes:
            content = self.get_node_str_content(node)

            if available_tokens <= 0:
                processed_content = ""
            else:
                # 检查并在“可用于上下文的 token 预算”内截断
                processed_content = ensure_within_token_limit(content, max_tokens=available_tokens)

            prompt = apply_prompt_template(
                "summary_generation",
                {
                    "context_text": processed_content,
                },
            )
            prompts.append(prompt)
        summaries = await abatch_llm_with_prompt(prompts)

        ## 生成的summary长度应与nodes数组长度相同 测试用
        if len(nodes) != len(summaries):
            logger.error(
                f"_llm_summarize_batch: nodes count ({len(nodes)}) != summaries count ({len(summaries)}); extra items will be ignored."
            )

        for node, summary in zip(nodes, summaries):
            node.summary = summary
        return 
    
    # batch方法对外接口
    @trace_span(name="SummaryGenerator.run_batch")
    async def run_batch(self)-> None:

        leafs: List[Node] = []
        self._post_order_get_leafs(leafs = leafs, node = self.root)
        await self._llm_summarize_batch(leafs)


        return 