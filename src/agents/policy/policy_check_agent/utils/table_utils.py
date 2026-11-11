from __future__ import annotations

import re
from typing import List

from .text_utils import normalize_whitespace


def parse_markdown_table(markdown_text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in (markdown_text or "").splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        cells = [normalize_whitespace(cell) for cell in stripped.strip("|").split("|")]
        if not any(cells):
            continue
        if all(re.fullmatch(r"[-:\s]+", cell or "") for cell in cells):
            continue
        rows.append(cells)
    return rows


def is_form_like_table(rows: List[List[str]]) -> bool:
    if not rows:
        return False
    non_empty_rows = [row for row in rows if any(cell for cell in row)]
    short_rows = sum(1 for row in non_empty_rows if len([cell for cell in row if cell]) <= 2)
    return bool(non_empty_rows) and short_rows / len(non_empty_rows) >= 0.6


def row_to_text(row: List[str]) -> str:
    values = [normalize_whitespace(value) for value in row if normalize_whitespace(value)]
    return " | ".join(values)


def kv_pairs_from_rows(rows: List[List[str]]) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for row in rows:
        values = [normalize_whitespace(value) for value in row if normalize_whitespace(value)]
        if len(values) < 2:
            continue
        if len(values) == 2:
            pairs.append((values[0], values[1]))
            continue
        for index in range(0, len(values) - 1, 2):
            pairs.append((values[index], values[index + 1]))
    return pairs
