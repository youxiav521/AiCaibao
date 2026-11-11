from __future__ import annotations

import re
from typing import Any, Optional

from src.applications.parsing.service.three_statements.cells import get_raw_cell_text

SUBJECT_LIKE_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")
AMOUNT_RE = re.compile(r"^-?[\d,]+\.?\d*$")
FINANCIAL_CATEGORY_RE = re.compile(
    r"流动资产|非流动资产|流动负债|非流动负债|所有者权益|股东权益|"
    r"资产总计|负债总计|负债合计|"
    r"营业收入|营业成本|营业利润|利润总额|净利润|营业总收入|营业总成本|"
    r"经营活动|投资活动|筹资活动|现金及现金等价物|"
    r"^[一二三四五六七八九十]+[、.\s]"
)

TITLE_ROW_KEYWORDS: tuple[str, ...] = (
    "编制单位", "法定代表人", "财务负责人", "负责人",
    "主管会计", "金额单位", "单位：", "单位:", "币种",
    "有限公司", "股份公司", "有限责任公司",
    "资产负债表", "利润表", "现金流量表", "利润分配表",
    "所有者权益变动表",
)
TITLE_ROW_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\d{4}\s*[年/.\-]\s*\d{1,2}\s*[月/.\-]\s*\d{1,2}"),
    re.compile(r"^[A-Z]{2,}\s*\(\s*[A-Z]{2,}\s*\)"),
    re.compile(r"^表\s*\d"),
)
ENTITY_NAME_RE = re.compile(r"银行|集团|控股|证券|保险|信托|基金|投资|资本")
COLUMN_HEADER_KWS: frozenset[str] = frozenset({
    "项目", "附注", "期末", "期初", "本期", "上期", "本年", "上年",
    "调整前", "调整后", "余额", "金额", "行次", "序号",
})
STATEMENT_AUXILIARY_HEADER_TOKENS: tuple[str, ...] = (
    "行次", "行数", "序号", "编号", "附注", "注释", "备注", "说明",
)
STATEMENT_NOTE_PATTERNS: set[str] = {"note", "notes", "备注", "附注"}


def normalize_header_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def looks_like_subject(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    return bool(SUBJECT_LIKE_RE.search(value))


def looks_like_amount_text(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if text in ("-", "--", "---", "—", "——", "———"):
        return True
    normalized = text.replace(",", "").replace("，", "").replace("(", "-").replace(")", "")
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", normalized))


def looks_like_sequence_number_text(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"\d{1,3}", compact):
        return True
    return bool(re.fullmatch(r"[一二三四五六七八九十]{1,3}[、.)．]?", compact))


def looks_like_date_header_text(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    return bool(
        re.search(r"\d{4}年\d{1,2}月(\d{1,2}日)?", compact)
        or re.search(r"\d{4}[./-]\d{1,2}([./-]\d{1,2})?", compact)
        or re.search(r"\d{4}年度", compact)
    )


def is_header_like_row(cells: list[Any]) -> bool:
    non_empty_texts: list[str] = []
    for cell in cells:
        text = get_raw_cell_text(cell).strip()
        if not text:
            continue
        if AMOUNT_RE.match(text):
            return False
        non_empty_texts.append(text)

    if not non_empty_texts:
        return True

    if len(non_empty_texts) <= 2:
        for text in non_empty_texts:
            if FINANCIAL_CATEGORY_RE.search(text):
                return False

    return True


def cell_grid_header_text_blob_lower(
    cell_grid: list[list[Optional[dict[str, Any]]]], header_row_count: int
) -> str:
    parts: list[str] = []
    for row_index in range(min(max(header_row_count, 0), len(cell_grid))):
        for cell in cell_grid[row_index]:
            text = get_raw_cell_text(cell or {}).strip()
            if text:
                parts.append(text.lower())
    return " ".join(parts)


def is_group_with_bank_or_company_dual_scope_table(
    cell_grid: list[list[Optional[dict[str, Any]]]], header_row_count: int
) -> bool:
    blob = cell_grid_header_text_blob_lower(cell_grid, header_row_count)
    if "group" not in blob:
        # 国际场景：Consolidated + Standalone 并列也不应被 two-block-split 拆开
        if "consolidated" in blob and "standalone" in blob:
            return True
        return False
    if re.search(r"\bbank\b", blob):
        return True
    if re.search(r"\bcompany\b", blob):
        return True
    return False


def ensure_keep_columns_include_all_dual_scope_amounts(
    cell_grid: list[list[Optional[dict[str, Any]]]],
    *,
    header_row_count: int,
    keep_column_indices: list[int],
    total_columns: int,
) -> list[int]:
    if total_columns > 16:
        return keep_column_indices
    if not is_group_with_bank_or_company_dual_scope_table(cell_grid, header_row_count):
        return keep_column_indices
    return list(range(total_columns))


def guess_header_row_count(cell_grid: list[list[Optional[dict[str, Any]]]], *, max_rows: int = 3) -> int:
    if not cell_grid:
        return 0

    count = 0
    for row in cell_grid[: max(max_rows, 0)]:
        cells = [cell for cell in row if cell is not None]
        if not cells:
            continue
        if is_header_like_row(cells):
            count += 1
            continue
        break
    return count


def is_title_row(cells: list[Any]) -> bool:
    texts = [get_raw_cell_text(cell).strip() for cell in cells]
    non_empty = [text for text in texts if text]
    if not non_empty:
        return True

    if non_empty and all(looks_like_sequence_number_text(text) for text in non_empty):
        return True

    has_amount = any(looks_like_amount_text(text) for text in non_empty)
    combined = " ".join(non_empty)
    compact = combined.replace(" ", "")

    col_hdr_hits = sum(1 for text in non_empty if any(keyword in text for keyword in COLUMN_HEADER_KWS))
    if not has_amount and col_hdr_hits >= 2:
        return False

    if any(keyword in compact for keyword in TITLE_ROW_KEYWORDS):
        return True
    if any(pattern.search(compact) for pattern in TITLE_ROW_PATTERNS):
        return True
    if not has_amount and len(non_empty) <= 2:
        for text in non_empty:
            if len(text) >= 3 and ENTITY_NAME_RE.search(text):
                return True
    if len(non_empty) == 1 and len(non_empty[0]) > 4:
        if not AMOUNT_RE.match(non_empty[0]) and not looks_like_subject(non_empty[0]):
            return True
    return False


def is_auxiliary_statement_column_header(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    lower = compact.lower().strip(".．")
    if lower in {"no", "number"}:
        return True
    if compact in {"单位", "单位元", "金额单位", "币种"}:
        return True
    return any(token in compact for token in STATEMENT_AUXILIARY_HEADER_TOKENS)


def identify_auxiliary_column_indices(header_cells: list[list[Any]], total_cols: int) -> set[int]:
    grid: list[list[str]] = [[""] * total_cols for _ in range(len(header_cells))]
    occupied: list[list[bool]] = [[False] * total_cols for _ in range(len(header_cells))]
    for row_idx, row in enumerate(header_cells):
        col_cursor = 0
        for cell in row:
            if not isinstance(cell, dict):
                col_cursor += 1
                continue
            cstart = cell.get("colStart")
            if cstart is None:
                cstart = cell.get("colstart")
            if isinstance(cstart, int):
                col_cursor = cstart
            while col_cursor < total_cols and occupied[row_idx][col_cursor]:
                col_cursor += 1
            if col_cursor >= total_cols:
                break
            text = (cell.get("text") or "").strip()
            row_span = max(int(cell.get("rowspan", 1) or 1), 1)
            col_span = max(int(cell.get("colspan", 1) or 1), 1)
            for row_index in range(row_idx, min(row_idx + row_span, len(header_cells))):
                for col_index in range(col_cursor, min(col_cursor + col_span, total_cols)):
                    grid[row_index][col_index] = text
                    occupied[row_index][col_index] = True
            col_cursor += col_span

    aux_indices: set[int] = set()
    for col_idx in range(total_cols):
        col_texts = [grid[row_index][col_idx].strip() for row_index in range(len(header_cells)) if grid[row_index][col_idx].strip()]
        if not col_texts:
            continue
        combined = "".join(col_texts)
        if any(token in combined for token in STATEMENT_AUXILIARY_HEADER_TOKENS):
            aux_indices.add(col_idx)
            continue
        if any(text.lower() in STATEMENT_NOTE_PATTERNS for text in col_texts):
            aux_indices.add(col_idx)
            continue
        # 百分比占比列（如「金額|％」子列中的 ％ 列）作为辅助列删除。
        # 用精确匹配单个符号，避免「毛利率%」等含 % 的科目列被误删。
        if any(text in {"％", "%"} for text in col_texts):
            aux_indices.add(col_idx)
            continue
    return aux_indices


def is_statement_date_fragment(text: str) -> bool:
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    return any(
        re.fullmatch(pattern, compact)
        for pattern in (
            r"\d{4}年(\d{1,2}月(\d{1,2}日)?)?",
            r"\d{1,2}月(\d{1,2}日)?",
            r"\d{1,2}日",
            r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}",
            r"\d{4}[-/.]\d{1,2}",
            r"\d{1,2}[-/.]\d{1,2}",
        )
    )


def is_statement_meta_fragment(text: str) -> bool:
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    if any(keyword in compact for keyword in ("编制单位", "单位：", "单位:", "金额单位", "币种")):
        return True
    if re.search(r"会企\s*0?[1-9]\s*表", compact):
        return True
    if is_statement_date_fragment(compact):
        return True
    return False

