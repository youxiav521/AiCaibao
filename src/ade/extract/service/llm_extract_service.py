"""
ADE 结构化提取 Service — LLM 模式

流程：
  1. HTTP 调用 parse 接口 → 获取 markdown + chunks + grounding
  2. 判断 markdown 是否超过 token 阈值
     - 未超限：直接一次提取
     - 超限：按 chunk id 边界切割 markdown，LangExtract 风格滑动窗口
       - 每个窗口 = 主窗口 markdown + 上一窗口最后一个 chunk 的内容（overlap）
  3. 对每条 ExtractRule，多窗口结果递归合并
  4. chunkId 溯源：用 LLM 返回的 chunkId 查 grounding bbox
  5. 组装 ExtractionItem 返回
"""
import json
import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from src.ade.extract.schema.extract_schema import (
    ExtractRule,
    ExtractionItem,
    FieldMeta,
    FieldReference,
    ModelEnum,
)
from src.ade.extract.utils.llm_utils import build_chunk_windows, merge_raw_results
from src.utils.model_factory import ModelFactory
from src.utils.tokenizer import Tokenizer

# parse 接口地址，从环境变量读取（base url，不含具体路径）
ADE_PARSE_API_URL = os.getenv(
    "ADE_PARSE_API_URL", "http://10.192.72.11:8000/openapi/ade"
)

# 异步解析轮询间隔与超时（秒）
_PARSE_POLL_INTERVAL = float(os.getenv("ADE_PARSE_POLL_INTERVAL", "10"))
_PARSE_POLL_TIMEOUT = float(os.getenv("ADE_PARSE_POLL_TIMEOUT", "1200"))

# 单次 LLM 调用最大 token 数（主窗口内容上限）
_MAX_WINDOW_TOKENS = int(os.getenv("ADE_EXTRACT_MAX_WINDOW_TOKENS", "20000"))

# LLM model_id 映射
_MODEL_ID_MAP = {
    ModelEnum.yusys_llm: "doc_extract",
}

# 延迟初始化 tokenizer（避免 worker 启动时加载失败）
_tokenizer: Optional[Tokenizer] = None


def _get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    return _tokenizer


# ---------- parse 结果内部数据结构 ----------

class _ParseResult:
    """parse 接口返回的内部数据结构"""
    def __init__(self, markdown: str, chunks: List[Dict], grounding: Dict):
        self.markdown = markdown
        self.chunks = chunks  # List[{id, markdown, type, ...}]
        self.grounding = grounding  # {chunk_id: {box, page, type, ...}}


class LlmExtractService:
    """ADE 结构化提取服务 — LLM 模式"""

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    # 公开入口
    # ------------------------------------------------------------------ #
    async def extract(
        self,
        file_bytes: bytes,
        filename: str,
        rules: List[ExtractRule],
        model: Optional[ModelEnum] = None,
    ) -> ExtractionItem:
        """
        LLM 模式提取主入口。

        Returns:
            ExtractionItem，extraction 以 extractName 为 key 聚合多规则结果
        """
        parse_result = await self._call_parse_api(file_bytes, filename)

        logger.info(
            "parse 完成: markdown 长度={}, chunks={}, grounding keys={}",
            len(parse_result.markdown),
            len(parse_result.chunks),
            len(parse_result.grounding),
        )

        merged_extraction: Dict[str, Any] = {}
        merged_metadata: Dict[str, Any] = {}

        for rule in rules:
            extraction, metadata = await self._extract_rule(parse_result, rule, model)
            merged_extraction[rule.extractName] = extraction
            merged_metadata[rule.extractName] = metadata

        return ExtractionItem(
            extraction=merged_extraction,
            extraction_metadata=merged_metadata,
        )

    # ------------------------------------------------------------------ #
    # 调用 parse 接口（异步任务模式）
    # ------------------------------------------------------------------ #
    async def _call_parse_api(self, file_bytes: bytes, filename: str) -> _ParseResult:
        """
        调用 ADE parse 异步接口，流程：
          1. POST /upload-file          → fileId
          2. POST /submit-parse-job     → jobId
          3. 轮询 GET /parse/jobs/{jobId} 直到 completed / failed
        """
        async with httpx.AsyncClient(timeout=120) as client:
            # 1. 上传文件
            upload_resp = await client.post(
                f"{ADE_PARSE_API_URL}/upload-file",
                files={"file": (filename, file_bytes)},
            )
            upload_resp.raise_for_status()
            file_id = upload_resp.json()["data"]["fileId"]
            logger.info("parse upload-file 完成: fileId={}", file_id)

            # 2. 提交异步解析任务
            submit_resp = await client.post(
                f"{ADE_PARSE_API_URL}/submit-parse-job",
                json={"fileId": file_id},
            )
            submit_resp.raise_for_status()
            job_id = submit_resp.json()["data"]["jobId"]
            logger.info("parse submit-parse-job 完成: jobId={}", job_id)

        # 3. 轮询任务状态（独立 client，超时设长）
        elapsed = 0.0
        async with httpx.AsyncClient(timeout=30) as poll_client:
            while elapsed < _PARSE_POLL_TIMEOUT:
                await asyncio.sleep(_PARSE_POLL_INTERVAL)
                elapsed += _PARSE_POLL_INTERVAL

                status_resp = await poll_client.get(
                    f"{ADE_PARSE_API_URL}/parse/jobs/{job_id}"
                )
                status_resp.raise_for_status()
                body = status_resp.json()
                job_status = (body.get("data") or {}).get("status", "")

                logger.debug("parse 轮询: jobId={} status={} elapsed={:.0f}s", job_id, job_status, elapsed)

                if job_status == "completed":
                    parse_result = (body.get("data") or {}).get("parse_result") or {}
                    markdown = parse_result.get("markdown", "")
                    chunks = parse_result.get("chunks", [])
                    grounding = parse_result.get("grounding") or {}
                    logger.info(
                        "parse 完成: jobId={} markdown长度={} chunks={} grounding keys={}",
                        job_id, len(markdown), len(chunks), len(grounding),
                    )
                    return _ParseResult(markdown=markdown, chunks=chunks, grounding=grounding)

                if job_status == "failed":
                    failure_reason = (body.get("data") or {}).get("failure_reason", "未知错误")
                    raise RuntimeError(f"parse 任务失败: jobId={job_id} reason={failure_reason}")

        raise TimeoutError(f"parse 任务超时（>{_PARSE_POLL_TIMEOUT}s）: jobId={job_id}")

    # ------------------------------------------------------------------ #
    # 单条规则提取（含窗口切割）
    # ------------------------------------------------------------------ #
    async def _extract_rule(
        self,
        parse_result: _ParseResult,
        rule: ExtractRule,
        model: Optional[ModelEnum],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """对单条规则做提取，超长时自动切窗口并合并。"""
        tok = _get_tokenizer()
        total_tokens = tok.count(parse_result.markdown)

        if total_tokens <= _MAX_WINDOW_TOKENS or not parse_result.chunks:
            raw = await self._llm_extract(parse_result.markdown, rule, model)
            return self._build_extraction_and_metadata(raw, parse_result.grounding)

        windows = build_chunk_windows(
            parse_result.markdown,
            parse_result.chunks,
            _MAX_WINDOW_TOKENS,
            tok,
        )
        logger.info("长文档切割: rule={} windows={}", rule.extractName, len(windows))

        window_results: List[Dict[str, Any]] = []
        for win_md in windows:
            raw = await self._llm_extract(win_md, rule, model)
            window_results.append(raw)

        merged_raw = merge_raw_results(window_results)
        return self._build_extraction_and_metadata(merged_raw, parse_result.grounding)

    # ------------------------------------------------------------------ #
    # LLM 提取
    # ------------------------------------------------------------------ #
    async def _llm_extract(
        self,
        markdown: str,
        rule: ExtractRule,
        model: Optional[ModelEnum] = None,
    ) -> Dict[str, Any]:
        """调用 LLM 按 jsonSchema 提取字段值 + chunkId"""
        model_id = _MODEL_ID_MAP.get(model, "doc_extract")
        llm = ModelFactory.get_model(key=model_id)

        prompt_parts = [f"## JSON Schema\n```json\n{rule.jsonSchema}\n```"]
        if rule.extractPrompt:
            prompt_parts.append(f"## 补充说明\n{rule.extractPrompt}")
        prompt_parts.append(
            f"## 文档内容\n{markdown}\n\n"
            "文档内容格式说明：每个语义块格式如下：\n"
            "- 普通文本块：`chunk_id:{chunk_id}|chunk_type:{chunk_type}` 换行后接正文内容\n"
            "- 表格块：`chunk_id:{chunk_id}|chunk_type:table` 换行后接 `<{cell_id}|{cell_text}><{cell_id}|{cell_text}>...`（每行一条，单元格依次排列）\n"
            "每个叶子字段请返回 {\"value\": 字段值, \"chunkId\": 包含该字段值的块的 chunk_id}，"
            "chunk_id 即 `chunk_id:xxx|chunk_type:yyy` 中 `chunk_id:` 后面的部分；chunkId 不能为 null，必须填写。\n"
            "未找到的字段整体返回 null。"
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(prompt_parts)),
        ]

        max_retries = 3
        initial_delay = 15.0
        for attempt in range(max_retries + 1):
            try:
                response = await llm.ainvoke(messages)
                break
            except Exception as e:
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(
                        "LLM 调用失败（第 {} 次），{}s 后重试: {}", attempt + 1, delay, e
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("LLM 调用失败，已重试 {} 次: {}", max_retries, e)
                    raise

        raw_text = response.content if hasattr(response, "content") else str(response)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("LLM 返回非 JSON: {}", raw_text[:200])
            return {"_raw": raw_text}

    # ------------------------------------------------------------------ #
    # 提取结果 + 溯源 metadata 构建
    # ------------------------------------------------------------------ #
    def _build_extraction_and_metadata(
        self,
        raw_extracted: Dict[str, Any],
        grounding: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        递归遍历 LLM 返回结果：
        - 叶子节点 {value, chunkId} → extraction 存纯 value，metadata 存 FieldMeta
        - 非叶子节点递归处理，metadata 与 extraction 保持同构嵌套
        """
        extraction, meta_tree = self._walk(raw_extracted, grounding)
        if not isinstance(extraction, dict):
            extraction = {"_raw": extraction}
        if not isinstance(meta_tree, dict):
            meta_tree = {}
        return extraction, meta_tree

    def _walk(
        self,
        node: Any,
        grounding: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """递归处理节点，返回 (展开后的纯值, 同构 metadata 节点)。"""
        if node is None:
            return None, None

        if isinstance(node, dict):
            # 叶子节点：{value, chunkId}
            if "value" in node and "chunkId" in node:
                value = node["value"]
                chunk_id = node.get("chunkId")
                bbox: Optional[List[float]] = None
                page_index: Optional[int] = None
                if chunk_id:
                    g = grounding.get(str(chunk_id))
                    if isinstance(g, dict):
                        box = g.get("box") or {}
                        if box:
                            bbox = [
                                box.get("left", 0),
                                box.get("top", 0),
                                box.get("right", 0),
                                box.get("bottom", 0),
                            ]
                        if "page" in g:
                            page_index = g["page"] + 1
                    elif hasattr(g, "box"):
                        b = g.box
                        bbox = [b.left, b.top, b.right, b.bottom]
                        page_index = g.page + 1
                meta = FieldMeta(
                    value=value,
                    references=FieldReference(bbox=bbox, pageIndex=page_index),
                )
                return value, meta

            result = {}
            meta_dict = {}
            for k, v in node.items():
                if k.startswith("_"):
                    continue
                child_val, child_meta = self._walk(v, grounding)
                result[k] = child_val
                meta_dict[k] = child_meta
            return result, meta_dict

        if isinstance(node, list):
            result_list = []
            meta_list = []
            for item in node:
                child_val, child_meta = self._walk(item, grounding)
                result_list.append(child_val)
                meta_list.append(child_meta)
            return result_list, meta_list

        return node, None


# ------------------------------------------------------------------ #
# 系统提示词
# ------------------------------------------------------------------ #
_SYSTEM_PROMPT = """你是一个专业的文档信息提取助手。请根据给定的 JSON Schema 从文档内容中提取结构化信息。

## 文档格式
文档内容由多个语义块组成，每个块有两种格式：

- 普通文本块：`chunk_id:{chunk_id}|chunk_type:{chunk_type}` 换行后接正文内容
- 表格块：`chunk_id:{chunk_id}|chunk_type:table` 换行后接 `<{cell_id}|{cell_text}><{cell_id}|{cell_text}>...`（每行一条，单元格依次排列）

`chunk_id` 是该块的唯一标识，用于溯源。

## 提取规则
1. 必须以 JSON Schema 为准，抽取其中定义的所有字段
2. 参考字段的 description 理解含义和提取方式
3. extractPrompt 只是补充说明，不限制抽取字段范围
4. 严格保持 JSON Schema 的嵌套层级，禁止省略或合并
5. 无法提取的字段使用 null
6. 每个叶子字段必须返回 {"value": 字段值, "chunkId": 包含该字段值的块的 chunk_id}
7. chunkId 取 `chunk_id:xxx|chunk_type:yyy` 中 `chunk_id:` 后面的部分，不能为 null
8. 若文档包含 [Previous Context] 段落，该段仅用于理解上下文，优先从 [Main Window] 中抽取字段

## 输出要求
- 输出必须是有效 JSON，不要包含任何其他文字
- 直接输出结构化结果对象，键为字段名，值为提取结果"""
