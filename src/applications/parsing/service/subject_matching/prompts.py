from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SubjectPromptContext:
    report_name: str
    source_texts: list[str]


class SubjectMatchPromptProvider(Protocol):
    name: str

    def build_normalization_prompt(self, context: SubjectPromptContext) -> str:
        ...


class ChinaSubjectPromptProvider:
    name = "china"

    def build_normalization_prompt(self, context: SubjectPromptContext) -> str:
        text_list = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(context.source_texts))
        return f"""你是中国会计准则（CAS）专家。以下科目名称来自「{context.report_name}」，可能使用了旧准则术语、行业简称或非标准表述。

科目列表：
{text_list}

请逐条判断：
- 如果该科目使用了旧准则术语或非标准表述，且存在对应的现代标准名称，返回标准名称
- 如果已经是标准/通用名称，或者无法确定，请省略该条

请以JSON格式返回：{{"mappings": [{{"idx": 编号, "standard": "标准名称"}}]}}
只返回JSON，不要返回其他内容。"""


class InternationalSubjectPromptProvider:
    name = "international"

    def build_normalization_prompt(self, context: SubjectPromptContext) -> str:
        text_list = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(context.source_texts))
        return f"""You are a financial reporting expert familiar with IFRS and common international annual report terminology.

Report: {context.report_name}

Line items extracted from a financial statement table:
{text_list}

Task:
- Normalize only clear accounting terminology variants to common IFRS-style line item names.
- Preserve the accounting meaning, direction, and classification.
- Do not translate company-specific labels unless there is a clear standard equivalent.
- Omit items that are already standard or uncertain.

Return JSON only in this exact format:
{{"mappings": [{{"idx": number, "standard": "normalized line item"}}]}}"""


def choose_subject_prompt_provider(strategy: str | None = None) -> SubjectMatchPromptProvider:
    key = (strategy or "").strip().lower()
    if key in {"international", "intl", "ifrs", "us-gaap", "us_gaap"}:
        return InternationalSubjectPromptProvider()
    return ChinaSubjectPromptProvider()
