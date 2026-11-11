from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from openai import OpenAI
import dotenv

dotenv.load_dotenv()



def _normalize_text(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _get_block_text(it: Dict[str, Any]) -> str:
    return _normalize_text(it.get("md")) or _normalize_text(it.get("value"))


def _is_heading(it: Dict[str, Any]) -> Tuple[bool, int]:
    level = it.get("headingLevel")
    try:
        if isinstance(level, str) and level.strip().isdigit():
            level = int(level.strip())
    except Exception:
        level = None
    return (it.get("type") == "text" and isinstance(level, int), int(level) if isinstance(level, int) else 0)


def _match_keywords(text: str, keywords: List[str]) -> bool:
    return bool(text) and any(kw in text for kw in keywords)


def _collect_subtree(items: List[Dict[str, Any]], start_idx: int, start_level: int, max_blocks: int = 300) -> List[int]:
    picked: List[int] = [start_idx]
    i = start_idx + 1
    while i < len(items) and len(picked) < max_blocks:
        it = items[i]
        is_head, lv = _is_heading(it)
        if is_head and lv <= start_level:
            break
        picked.append(i)
        i += 1
    return picked


def _recall_by_keywords(items: List[Dict[str, Any]], keywords: List[str], max_hits: int = 8, max_per_hit: int = 200) -> List[int]:
    indices: List[int] = []
    for idx, it in enumerate(items):
        text = _get_block_text(it)
        is_head, lv = _is_heading(it)
        if is_head and _match_keywords(text, keywords):
            indices.extend(_collect_subtree(items, idx, lv, max_per_hit))
            if len(indices) >= max_hits * max_per_hit:
                break
    if not indices:
        for idx, it in enumerate(items):
            text = _get_block_text(it)
            if _match_keywords(text, keywords):
                indices.append(idx)
                if len(indices) >= max_hits:
                    break
    seen = set()
    deduped: List[int] = []
    for i in indices:
        if i not in seen:
            deduped.append(i)
            seen.add(i)
    return deduped


def _make_elements(items: List[Dict[str, Any]], idxs: List[int], max_chars: int = 1600) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    for i in idxs:
        it = items[i]
        eid = i + 1
        page = it.get("page")
        text = _get_block_text(it).strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        el: Dict[str, Any] = {"element_id": eid, "md": text}
        if isinstance(page, int):
            el["page"] = page
        if "type" in it:
            el["type"] = it["type"]
        if "table_footnote" in it:
            el["table_footnote"] = it["table_footnote"]
        elements.append(el)
    return elements


def _call_llm_json(model_id: str, prompt_text: str) -> Dict[str, Any]:
    """
    使用 ModelFactory 调用 LLM 并返回 JSON 结果
    """
    from src.utils.model_factory import ModelFactory
    from langchain_core.prompts import ChatPromptTemplate
    
    model = ModelFactory.get_model(key=model_id, temperature=0.01)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是资深财报分析专家。"),
        ("human", "{prompt}")
    ])
    
    try:
        chain = prompt_template | model
        resp = chain.invoke({"prompt": prompt_text})
        content = resp.content if hasattr(resp, "content") else str(resp)
        
        if isinstance(content, str):
            # 移除 <think> 标签内容
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.strip()
            
            # 尝试提取 markdown 代码块中的 JSON
            json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 尝试提取任意代码块中的 JSON
            json_match = re.search(r"```\n?(.*?)\n?```", content, re.DOTALL)
            if json_match:
                potential_json = json_match.group(1).strip()
                if potential_json.startswith('{') or potential_json.startswith('['):
                    return json.loads(potential_json)
            
            # 直接尝试解析内容
            return json.loads(content)
        return {}
    except Exception as e:
        logger.exception("[financial_meta] LLM 调用失败: {}", str(e)[:300])
        return {}


class BaseSectionExtractor(ABC):
    def __init__(self, *, items: List[Dict[str, Any]], indices: List[int], model_id: str) -> None:
        self.items = items
        self.indices = indices
        self.model_id = model_id

    @classmethod
    def section_key(cls) -> Optional[str]:
        return None

    @classmethod
    def recall_keywords(cls) -> Optional[List[str]]:
        return None

    @classmethod
    def recall_params(cls) -> Dict[str, int]:
        return {}

    @classmethod
    def recall_indices(cls, items: List[Dict[str, Any]]) -> Optional[List[int]]:
        return None

    @abstractmethod
    def title(self) -> str:
        ...

    @abstractmethod
    def schema_hint(self) -> Dict[str, Any]:
        ...

    def extra_rules(self) -> Optional[str]:
        return None

    def examples(self) -> Optional[str]:
        return None

    @abstractmethod
    def parse(self, raw: Dict[str, Any]) -> Any:
        ...

    def _make_prompt(self, elements: List[Dict[str, Any]]) -> str:
        rules = (
            "你是一位资深、严谨的财报分析专家。你的任务是根据提供的JSON片段`elements`，严格、精确地抽取出指定信息。\n"
            "核心规则：\n"
            "1. **绝不臆造**：所有`value`的取值都必须直接来源于`elements`中的文本。如果找不到对应信息，必须返回`null`。\n"
            "2. **精确溯源**：`source_element_id`是关键！对于每一个提取出的`value`，必须提供其最直接、最相关的来源`element_id`。如果一个值是由多个elements综合得出的，选择最关键的那个element_id。\n"
            "3. **格式严格**：仅输出符合下面“JSON模式”的JSON对象，不要包含任何额外的解释、注释或markdown标记。\n"
            "4. **日期标准化**：所有日期必须统一为 YYYY-MM-DD 格式。\n"
        )
        if self.extra_rules():
            rules += self.extra_rules() + "\n"
        lines: List[str] = []
        lines.append(rules)
        lines.append(f"==== 任务：{self.title()} ====")
        lines.append("【上下文信息】 `elements` (JSON 数组, 每项含 element_id/page/md):")
        lines.append(json.dumps(elements, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("【输出格式】 本组 JSON 模式如下（注意键名为中文，且仅限以下键）：")
        lines.append(json.dumps(self.schema_hint(), ensure_ascii=False, indent=2))
        if self.examples():
            lines.append("")
            lines.append("【参考示例】")
            lines.append(self.examples() or "")
        lines.append("\n请严格按照上述规则和格式，开始你的工作。")
        return "\n".join(lines)

    def extract(self) -> Any:
        elements = _make_elements(self.items, self.indices, max_chars=self._max_chars())
        logger.info("[financial_meta] Prompt({}) 构造完成: elements={}", self.title(), len(elements))
        prompt = self._make_prompt(elements)
        raw = _call_llm_json(self.model_id, prompt)
        return self.parse(raw)

    def _max_chars(self) -> int:
        return 1600


