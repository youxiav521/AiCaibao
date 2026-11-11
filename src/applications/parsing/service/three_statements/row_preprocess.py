"""Row-level preprocessing for financial statement tables.

Handles two structural issues that arise from PDF parsing:
1. Continuation rows: a subject name spans two PDF lines, producing a row with
   text in the subject column but empty amount columns, followed by a row whose
   first cell is the tail fragment (often starting lowercase) with amounts.
2. Missing spaces: when multi-line text is concatenated without a separator,
   producing tokens like "reclassifiedsubsequently" or "frominsurance".
"""
from __future__ import annotations

import re
from typing import Any

from src.applications.parsing.service.three_statements.cells import (
    get_raw_cell_text,
    normalize_raw_rows,
)
from src.applications.parsing.service.three_statements.headers import looks_like_amount_text

_LOWERCASE_CONTINUATION_RE = re.compile(r"^[a-z]")
_CAMEL_GLUE_RE = re.compile(r"([a-z])([A-Z])")


def _row_has_amounts(cells: list[Any]) -> bool:
    """Check if any cell beyond the first contains an amount-like value."""
    for cell in cells[1:]:
        text = get_raw_cell_text(cell).strip()
        if text and looks_like_amount_text(text):
            return True
    return False


def _row_is_subject_only(cells: list[Any]) -> bool:
    """A row where the first cell has text but no other cell has amount data."""
    if not cells:
        return False
    first_text = get_raw_cell_text(cells[0]).strip()
    if not first_text:
        return False
    if looks_like_amount_text(first_text):
        return False
    return not _row_has_amounts(cells)


def _next_row_is_continuation(current_cells: list[Any], next_cells: list[Any]) -> bool:
    """Determine if next_row is a continuation of current_row using structural signals.

    Signals (no keywords/whitelists):
    - next row's first cell starts with a lowercase letter (sentence continuation)
    - next row has amounts in non-first columns (it carries the data)
    - current row has NO amounts (it's just the subject fragment)
    """
    if not next_cells:
        return False
    next_first = get_raw_cell_text(next_cells[0]).strip()
    if not next_first:
        return False
    if not _LOWERCASE_CONTINUATION_RE.match(next_first):
        return False
    if not _row_has_amounts(next_cells):
        return False
    return True


def _merge_two_rows(upper_cells: list[Any], lower_cells: list[Any]) -> list[Any]:
    """Merge two rows: concatenate first-cell text, take amounts from lower row."""
    if not upper_cells or not lower_cells:
        return lower_cells or upper_cells or []

    merged = []
    upper_first_text = get_raw_cell_text(upper_cells[0]).strip()
    lower_first_text = get_raw_cell_text(lower_cells[0]).strip()

    separator = " " if upper_first_text and not upper_first_text.endswith((" ", "-")) else ""
    combined_text = f"{upper_first_text}{separator}{lower_first_text}"

    first_cell = dict(lower_cells[0]) if isinstance(lower_cells[0], dict) else {"text": ""}
    first_cell["text"] = combined_text
    merged.append(first_cell)

    for i in range(1, max(len(upper_cells), len(lower_cells))):
        if i < len(lower_cells):
            lower_text = get_raw_cell_text(lower_cells[i]).strip()
            if lower_text:
                merged.append(lower_cells[i])
            elif i < len(upper_cells):
                merged.append(upper_cells[i])
            else:
                merged.append(lower_cells[i])
        elif i < len(upper_cells):
            merged.append(upper_cells[i])

    return merged


def merge_continuation_rows(rows: list[Any]) -> list[Any]:
    """Merge rows where a subject name was split across two PDF lines.

    Detection is purely structural:
    - Current row: has text in col-0 but no amounts in other columns
    - Next row: first cell starts with lowercase letter AND has amounts
    """
    raw = normalize_raw_rows(rows)
    if not raw or len(raw) < 2:
        return rows

    result_indices: list[int] = []
    skip_next = False

    for i in range(len(raw)):
        if skip_next:
            skip_next = False
            continue

        if i + 1 < len(raw) and _row_is_subject_only(raw[i]) and _next_row_is_continuation(raw[i], raw[i + 1]):
            merged_row = _merge_two_rows(raw[i], raw[i + 1])
            result_indices.append(i)
            if isinstance(rows[i], list):
                rows[i] = merged_row
            elif isinstance(rows[i], dict):
                rows[i]["cells"] = merged_row
            skip_next = True
        else:
            result_indices.append(i)

    return [rows[i] for i in result_indices]


def _is_glued_word_boundary(text: str, pos: int) -> bool:
    """Check if position `pos` in text is a likely missing-space boundary.

    Uses character-level features:
    - A lowercase letter followed by an uppercase letter (camelCase glue)
    - A sequence of >=3 lowercase chars followed by another >=3 lowercase sequence
      where the boundary creates two valid-length tokens
    """
    if pos <= 0 or pos >= len(text):
        return False
    before = text[pos - 1]
    after = text[pos]
    if before.islower() and after.isupper():
        return True
    return False


def fix_missing_spaces_in_text(text: str) -> str:
    """Fix concatenated words where a space was lost during PDF line merging.

    Strategy: detect lowercase→uppercase transitions within a single token
    (no whitespace around it) and insert a space. This handles cases like:
    - "reclassifiedsubsequently" → "reclassified subsequently"  (via camelCase detection)
    - "frominsurance" → "from insurance" (via camelCase detection won't catch this)

    For the second pattern, we use a statistical approach: if splitting at a
    lowercase→lowercase boundary produces two segments each >=3 chars where
    the left segment ends a common English morpheme, insert a space.
    """
    if not text or not any(c.isalpha() for c in text):
        return text

    result = _CAMEL_GLUE_RE.sub(r"\1 \2", text)

    result = _fix_lowercase_glue(result)

    return result


_COMMON_SUFFIXES = re.compile(
    r"(ed|ing|tion|sion|ment|ness|ance|ence|ity|ous|ive|able|ible|ful|less|ly|al|er|or|ar|"
    r"from|with|into|upon|over|under|after|before)$",
    re.IGNORECASE,
)

_VOWELS = set("aeiouAEIOU")


def _looks_like_word(s: str) -> bool:
    """Check if a string looks like a plausible English word using structural features.
    
    A plausible word has:
    - At least one vowel
    - Not all consonants or all vowels
    - Reasonable consonant cluster patterns
    """
    if len(s) < 3:
        return False
    vowel_count = sum(1 for c in s if c in _VOWELS)
    if vowel_count == 0:
        return False
    if vowel_count == len(s):
        return False
    consonant_run = 0
    max_consonant_run = 0
    for c in s:
        if c.isalpha() and c not in _VOWELS:
            consonant_run += 1
            max_consonant_run = max(max_consonant_run, consonant_run)
        else:
            consonant_run = 0
    if max_consonant_run > 4:
        return False
    return True


def _fix_lowercase_glue(text: str) -> str:
    """Fix cases where two lowercase words are glued without a space.

    Only operates on tokens (whitespace-delimited segments) that are unusually
    long (>12 chars) and can be split into two valid-looking segments.
    """
    tokens = text.split()
    fixed_tokens = []
    for token in tokens:
        if len(token) > 12 and token.isalpha() and token[0].islower():
            split = _try_split_glued_token(token)
            if split:
                fixed_tokens.append(split)
            else:
                fixed_tokens.append(token)
        else:
            fixed_tokens.append(token)
    return " ".join(fixed_tokens)


def _try_split_glued_token(token: str) -> str | None:
    """Try to split a glued token into two words at the best boundary.

    Uses structural features of both halves:
    - Both halves must look like plausible words (vowel/consonant structure)
    - Left half ending with a common morpheme suffix gets bonus score
    - Right half starting with a consonant (typical word start) gets bonus
    - Penalize splits where right half starts with unusual patterns
    """
    best_pos = -1
    best_score = 0

    for pos in range(3, len(token) - 2):
        left = token[:pos]
        right = token[pos:]
        if len(left) < 3 or len(right) < 3:
            continue

        if not _looks_like_word(left) or not _looks_like_word(right):
            continue

        score = 1.0

        if _COMMON_SUFFIXES.search(left):
            score += 3.0

        if right[0].islower() and len(right) >= 4:
            score += 0.5

        if right[0] not in _VOWELS and right[0].islower():
            score += 1.0
        elif right[0] in _VOWELS:
            score -= 0.5

        if len(left) >= 4 and len(right) >= 6:
            score += 0.5

        if score > best_score:
            best_score = score
            best_pos = pos

    if best_score >= 2 and best_pos > 0:
        return f"{token[:best_pos]} {token[best_pos:]}"
    return None


def fix_missing_spaces_in_rows(rows: list[Any]) -> list[Any]:
    """Apply missing-space fix to all text cells in the given rows."""
    raw = normalize_raw_rows(rows)
    if not raw:
        return rows

    for i, row in enumerate(raw):
        for cell in row:
            if not isinstance(cell, dict):
                continue
            text = get_raw_cell_text(cell).strip()
            if not text or looks_like_amount_text(text):
                continue
            fixed = fix_missing_spaces_in_text(text)
            if fixed != text:
                cell["text"] = fixed

    return rows
