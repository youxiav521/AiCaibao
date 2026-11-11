from __future__ import annotations

import re
from typing import Any, Optional

from src.applications.parsing.service.three_statements.strategies import choose_statement_strategy

_EN_CONTENT_HINTS: tuple[str, ...] = (
    "statement of financial position",
    "statement of profit or loss",
    "statement of comprehensive income",
    "statement of cash flows",
    "cash flow statement",
    "balance sheet",
    "income statement",
    "profit or loss",
    "total assets",
    "total liabilities",
    "total equity",
)


def _detect_items_language(items: list[Any]) -> str:
    """Scan items for English vs Chinese financial content. Returns 'en' or 'zh'."""
    en_hits = 0
    zh_hits = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        value_text = str(item.get("value") or "").lower()
        md_text = str(item.get("md") or "").lower()
        text = f"{value_text} {md_text}"
        if not text.strip():
            continue
        en_hits += sum(1 for hint in _EN_CONTENT_HINTS if hint in text)
        if re.search(r"[\u4e00-\u9fff]", text):
            zh_hits += 1
    if en_hits > zh_hits and en_hits >= 3:
        return "en"
    return "zh"


def extract_three_statements_with_strategy(
    items: list[Any],
    *,
    locale: Optional[str] = None,
    strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Strategy-aware three-statement extraction entry point.

    The domestic path delegates to the existing stable extractor. The
    international branch is intentionally isolated here so future IFRS/English
    rules can evolve without changing the domestic API behavior.

    Language auto-detection: when ``locale`` (from browser Accept-Language)
    and the document content language disagree, the document content wins.
    """
    doc_lang = _detect_items_language(items)
    if doc_lang == "en" and (not locale or not locale.strip().lower().startswith("en")):
        locale = "en-US"
    chosen = choose_statement_strategy(strategy, locale=locale)

    from src.applications.parsing.service.three_statements_frontend_service import (
        extract_three_statements_frontend,
    )

    if chosen.name == "china":
        return extract_three_statements_frontend(items, locale=locale)

    return extract_three_statements_frontend(items, locale=locale, statement_strategy=chosen)
