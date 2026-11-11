from __future__ import annotations

import re

from src.applications.parsing.service.subject_matching.strategy import text_has_cjk

EN_AGGREGATE_SUFFIXES: tuple[str, ...] = (
    "net", "net increase", "net decrease", "net movement", "net change",
)
EN_NET_TYPE_SUFFIXES: tuple[str, ...] = (
    "net", "net loss", "net income", "net gain", "net expense",
)
EN_SUMMARY_SUFFIXES: tuple[str, ...] = (
    "total", "subtotal", "sub-total",
)
EN_CASHFLOW_SECTION_KEYWORDS: tuple[str, ...] = (
    "cash flows from operating activities",
    "cash flow from operating activities",
    "cash flows from investing activities",
    "cash flow from investing activities",
    "cash flows from financing activities",
    "cash flow from financing activities",
)
EN_EPS_KEYWORDS: tuple[str, ...] = (
    "earnings per share", "eps", "basic earnings per share", "diluted earnings per share",
)
EN_PROFIT_KEYWORDS: tuple[str, ...] = (
    "profit for the year", "profit before tax", "profit after tax",
    "net profit", "net income", "operating profit",
)
EN_SECTION_HEADINGS: frozenset[str] = frozenset({
    "assets", "current assets", "non-current assets", "noncurrent assets",
    "liabilities", "current liabilities", "non-current liabilities", "noncurrent liabilities",
    "equity", "shareholders equity", "shareholders' equity", "owners equity", "owners' equity",
    "income", "expenses", "operating activities", "investing activities", "financing activities",
})
EN_SKIP_EXACT: frozenset[str] = frozenset({
    "item", "items", "description", "particulars", "account", "line item", "note", "notes",
    "amount", "balance", "remarks", "rm", "rm'000", "rm000", "usd", "usd'000",
    "group", "company", "bank", "fund", "parent", "consolidated",
    "2024", "2023", "2022", "2021", "2020",
    "assets", "liabilities", "equity", "total assets", "total liabilities", "total equity",
})
EN_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\(?\d+[\),.]?$"),
    re.compile(r"^[\d,.'\-—\s%()]+$"),
    re.compile(r"^(?:rm|usd|eur|gbp|myr|cny)\s*'?\s*(?:000|million|thousand|bn|m)?$", re.I),
    re.compile(r"^as\s+at\s+\d{1,2}\s+[a-z]+\s+\d{4}$", re.I),
    re.compile(r"^for\s+the\s+(?:financial\s+)?year\s+ended\s+\d{1,2}\s+[a-z]+\s+\d{4}$", re.I),
)
EN_FAMILY_SUFFIXES_DEDUP: tuple[str, ...] = (
    "net", "gross", "allowance", "impairment", "provision", "accumulated depreciation",
    "depreciation", "amortisation", "amortization", "total", "subtotal", "sub-total",
)


def normalize_english_subject_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"^[\s\-–—•*]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;:])$", r"\1", text)
    text = text.strip()
    if text.endswith(":") and text.lower().rstrip(":") in EN_SECTION_HEADINGS:
        return text[:-1].strip()
    return text


def deep_normalize_english(text: str) -> str:
    text = normalize_english_subject_text(text).lower()
    text = text.replace("&", "and")
    text = re.sub(r"\bthe\b", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def english_template_text_key(text: str) -> str:
    return deep_normalize_english(text)


def has_english_trailing_heading_punctuation(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped.endswith(":"):
        return False
    return stripped[:-1].strip().lower() in EN_SECTION_HEADINGS


def classify_english_family_pair(deep_a: str, deep_b: str) -> str | None:
    if not deep_a or not deep_b:
        return None
    if deep_a == deep_b:
        return "identical"
    if deep_a.startswith(deep_b) or deep_b.startswith(deep_a):
        return "prefix"
    return None


class EnglishSubjectTextRules:
    family_suffixes_dedup = EN_FAMILY_SUFFIXES_DEDUP
    aggregate_suffixes = EN_AGGREGATE_SUFFIXES
    net_type_suffixes = EN_NET_TYPE_SUFFIXES
    summary_suffixes = EN_SUMMARY_SUFFIXES
    cashflow_section_keywords = EN_CASHFLOW_SECTION_KEYWORDS
    eps_keywords = EN_EPS_KEYWORDS
    profit_keywords = EN_PROFIT_KEYWORDS

    def normalize_subject_text(self, text: str) -> str:
        return normalize_english_subject_text(text)

    def deep_normalize(self, text: str) -> str:
        return deep_normalize_english(text)

    def template_text_key(self, text: str) -> str:
        return english_template_text_key(text)

    def has_trailing_heading_punctuation(self, text: str) -> bool:
        return has_english_trailing_heading_punctuation(text)

    def is_hard_non_subject_text(self, text: str) -> bool:
        if not text:
            return True
        normalized = normalize_english_subject_text(text)
        compact = normalized.lower().strip(" .;:")
        if not compact:
            return True
        if compact in EN_SKIP_EXACT:
            return True
        if text_has_cjk(normalized):
            return False
        return any(pattern.match(compact) for pattern in EN_NOISE_PATTERNS)

    def is_soft_non_subject_text(self, text: str) -> bool:
        normalized = normalize_english_subject_text(text).lower().strip(" .;:")
        if not normalized:
            return False
        return normalized.startswith(("total ", "subtotal ", "sub-total "))

    def looks_like_section_heading(self, text: str) -> bool:
        normalized = normalize_english_subject_text(text).lower().strip(" .;:")
        if not normalized:
            return False
        return normalized in EN_SECTION_HEADINGS

    def deterministic_match_key(self, text: str) -> str | None:
        return None

    def classify_family_pair(self, deep_a: str, deep_b: str) -> str | None:
        return classify_english_family_pair(deep_a, deep_b)
