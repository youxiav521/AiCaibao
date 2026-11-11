from __future__ import annotations

from typing import Any


def _table_blob_lower(table_el: dict[str, Any]) -> str:
    md = str(table_el.get("md") or "").strip()
    if md:
        return md.lower()
    rows = table_el.get("rows")
    if not isinstance(rows, list):
        return ""
    parts: list[str] = []
    for row in rows:
        cells = row if isinstance(row, list) else row.get("cells", []) if isinstance(row, dict) else []
        if not isinstance(cells, list):
            continue
        parts.append(" ".join(str(c.get("text") or "") for c in cells if isinstance(c, dict)))
    return "\n".join(parts).lower()


def should_merge_international_balance_sheet_halves(previous_candidate: dict[str, Any], next_candidate: dict[str, Any]) -> bool:
    """International BS continuation rule.

    IFRS annual reports often split Statement of Financial Position into an
    assets block and an equity/liabilities block on the same page or adjacent
    pages. Unlike the domestic default path, the international strategy should
    merge these blocks into one logical balance sheet candidate.
    """
    previous_label = str(previous_candidate.get("label") or "")
    next_label = str(next_candidate.get("label") or "")
    if "资产负债表" not in previous_label or "资产负债表" not in next_label:
        return False
    if previous_label != next_label:
        return False

    previous_idx = previous_candidate.get("expanded_idx")
    next_idx = next_candidate.get("expanded_idx")
    if not isinstance(previous_idx, int) or not isinstance(next_idx, int):
        return False
    if next_idx != previous_idx + 1:
        return False

    previous_table = previous_candidate.get("table") if isinstance(previous_candidate.get("table"), dict) else {}
    next_table = next_candidate.get("table") if isinstance(next_candidate.get("table"), dict) else {}
    previous_blob = _table_blob_lower(previous_table)
    next_blob = _table_blob_lower(next_table)

    if "total assets" not in previous_blob:
        return False
    if "total equity and liabilities" in previous_blob:
        return False
    return (
        "total equity and liabilities" in next_blob
        or "total liabilities and shareholders' equity" in next_blob
        or "total equity and liability" in next_blob
    )
