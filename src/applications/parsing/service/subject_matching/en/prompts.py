from __future__ import annotations


def build_review_matches_prompt(
    *,
    report_name: str,
    statement_label: str,
    pairs_text: str,
) -> str:
    return f"""You are a financial reporting expert familiar with IFRS and common annual-report statement formats.

Statement type: {statement_label or report_name}

The following pairs were automatically matched. The left side is the line item text extracted from the statement table, and the right side is the matched template subject:
{pairs_text}

Review each pair and identify only clearly wrong matches. Reject a pair only when:
1. The accounting meaning is materially different.
2. Direction, maturity, or classification is mutually exclusive, such as receivable vs payable, current vs non-current, asset vs liability, operating vs investing vs financing.
3. A parent line item is matched to an unrelated child line item, or the reverse.
4. The match crosses different statement families.

Important:
- Reject only obvious errors.
- Do not reject exact matches or generally accepted synonyms.
- If all matches are reasonable, return an empty list.

Return JSON only: {{"reject": [indexes], "reasons": {{"index": "reason"}}}}
"""


def build_normalize_batch_prompt(*, report_name: str, text_list: str) -> str:
    return f"""You are a financial reporting expert familiar with IFRS and common English annual-report terminology.
The following line items come from "{report_name}" and may contain abbreviations, legacy terms, or non-standard wording.

Line items:
{text_list}

For each item:
- If the item has a clear modern/common standard name with the same accounting meaning, return that standard name.
- If it is already standard/common, or if uncertain, omit it.
- Do not translate into Chinese.
- Do not change the accounting meaning.
- Do not normalize section headings, dates, units, or column headers.

Examples:
- Trade and other receivables, net -> Trade and other receivables
- Cash and bank balances -> Cash and cash equivalents only if the template context clearly uses that term for the same concept
- PPE -> Property, plant and equipment

Return JSON only: {{"mappings": [{{"idx": number, "standard": "standard name"}}]}}
"""
