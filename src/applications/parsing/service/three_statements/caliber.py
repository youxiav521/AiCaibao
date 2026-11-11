from __future__ import annotations

from typing import Any, Optional


def finalize_caliber_label_for_output(
    table_caliber: Optional[str], category_for_inference: Optional[str]
) -> str:
    if table_caliber == "合并":
        return "合并"
    if table_caliber == "单体":
        return "单体"
    if isinstance(category_for_inference, str):
        if category_for_inference.startswith("合并"):
            return "合并"
        if category_for_inference.startswith("母公司"):
            return "单体"
    return "单体"


def coerce_cell_caliber_to_binary(rows: list[Any], fallback_caliber: str) -> None:
    if fallback_caliber not in ("合并", "单体"):
        fallback_caliber = "单体"
    for row in rows or []:
        if not isinstance(row, list):
            continue
        for cell in row:
            if not isinstance(cell, dict):
                continue
            if cell.get("caliberLabel") in ("合并", "单体"):
                continue
            cell["caliberLabel"] = fallback_caliber
