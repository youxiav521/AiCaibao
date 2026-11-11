from __future__ import annotations

import re

EN_SUBJECT_HEADER_KEYWORDS: frozenset[str] = frozenset({
    "item", "items", "description", "particulars", "account", "line item",
})
EN_AMOUNT_HEADER_KEYWORDS: frozenset[str] = frozenset({
    "amount", "balance", "carrying amount", "book value",
})
EN_AMOUNT_HEADER_RE = re.compile(
    r"(?:amount|balance|carrying\s+amount|book\s+value|\d{4}\s*(?:[/-]|$)|RM|USD|EUR|GBP|CNY|MYR)",
    re.IGNORECASE,
)
