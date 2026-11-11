from __future__ import annotations

from typing import Iterable

from src.applications.parsing.service.subject_matching.en.text_rules import EnglishSubjectTextRules
from src.applications.parsing.service.subject_matching.strategy import (
    FinancialStatementLanguage,
    SubjectMatchStrategy,
    text_has_latin_letters,
)
from src.applications.parsing.service.subject_matching.zh.text_rules import ChineseSubjectTextRules

ZH_STRATEGY = SubjectMatchStrategy(
    language=FinancialStatementLanguage.ZH,
    text_rules=ChineseSubjectTextRules(),
)
EN_STRATEGY = SubjectMatchStrategy(
    language=FinancialStatementLanguage.EN,
    text_rules=EnglishSubjectTextRules(),
)

_EN_REPORT_HINTS: tuple[str, ...] = (
    "statement of financial position",
    "balance sheet",
    "statement of profit or loss",
    "income statement",
    "statement of comprehensive income",
    "statement of cash flows",
    "cash flow statement",
)


def _iter_non_empty(values: Iterable[str | None]) -> Iterable[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            yield value.strip()


def detect_subject_match_language(
    *,
    report_names: Iterable[str | None],
    table_names: Iterable[str | None],
    subject_names: Iterable[str | None],
    cell_texts: Iterable[str | None],
) -> FinancialStatementLanguage:
    """Detect the matching strategy language.

    Defaults to Chinese to preserve the existing domestic behavior unless there is
    clear English evidence from report/table names, template subjects, or cells.
    """
    report_and_table_texts = list(_iter_non_empty([*report_names, *table_names]))
    for text in report_and_table_texts:
        lowered = text.lower()
        if any(hint in lowered for hint in _EN_REPORT_HINTS):
            return FinancialStatementLanguage.EN

    sample_texts = [*report_and_table_texts]
    sample_texts.extend(list(_iter_non_empty(subject_names))[:80])
    sample_texts.extend(list(_iter_non_empty(cell_texts))[:120])
    if not sample_texts:
        return FinancialStatementLanguage.ZH

    latin_count = 0
    for text in sample_texts:
        if text_has_latin_letters(text):
            latin_count += 1

    if latin_count >= max(3, len(sample_texts) // 3):
        return FinancialStatementLanguage.EN
    return FinancialStatementLanguage.ZH


def get_subject_match_strategy(language: FinancialStatementLanguage) -> SubjectMatchStrategy:
    if language == FinancialStatementLanguage.EN:
        return EN_STRATEGY
    return ZH_STRATEGY
