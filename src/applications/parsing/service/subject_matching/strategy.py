from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class FinancialStatementLanguage(str, Enum):
    ZH = "zh"
    EN = "en"


class SubjectTextRules(Protocol):
    family_suffixes_dedup: tuple[str, ...]
    aggregate_suffixes: tuple[str, ...]
    net_type_suffixes: tuple[str, ...]
    summary_suffixes: tuple[str, ...]
    cashflow_section_keywords: tuple[str, ...]
    eps_keywords: tuple[str, ...]
    profit_keywords: tuple[str, ...]

    def normalize_subject_text(self, text: str) -> str: ...

    def deep_normalize(self, text: str) -> str: ...

    def template_text_key(self, text: str) -> str: ...

    def has_trailing_heading_punctuation(self, text: str) -> bool: ...

    def is_hard_non_subject_text(self, text: str) -> bool: ...

    def is_soft_non_subject_text(self, text: str) -> bool: ...

    def looks_like_section_heading(self, text: str) -> bool: ...

    def deterministic_match_key(self, text: str) -> str | None: ...

    def classify_family_pair(self, deep_a: str, deep_b: str) -> str | None: ...


@dataclass(frozen=True)
class SubjectMatchStrategy:
    language: FinancialStatementLanguage
    text_rules: SubjectTextRules


def text_has_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def text_has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))
