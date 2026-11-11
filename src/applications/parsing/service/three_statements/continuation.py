from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from src.applications.parsing.service.three_statements.cells import (
    find_raw_cell_covering_col,
    get_page_at,
    get_raw_cell_text,
    get_text,
    infer_table_total_columns,
    normalize_raw_rows,
    to_int,
)
from src.applications.parsing.service.three_statements.statement_content import (
    BASE_TYPE_START_KEYWORDS,
    content_matches_base_type,
    content_matches_other_base_type,
)

logger = logging.getLogger(__name__)
MAX_ABSOLUTE_PAGE_GAP = 4
_CASHFLOW_CONTINUATION_SECTIONS = (
    "经营活动产生的现金流量",
    "投资活动产生的现金流量",
    "筹资活动产生的现金流量",
)
_NUMBERED_CONTINUATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "资产负债表": (
        "流动资产",
        "非流动资产",
        "资产总计",
        "流动负债",
        "非流动负债",
        "负债合计",
        "所有者权益",
        "股东权益",
    ),
    "利润表": (
        "营业收入",
        "营业成本",
        "营业利润",
        "利润总额",
        "净利润",
        "其他综合收益",
        "综合收益总额",
        "每股收益",
        "债权投资信用减值准备",
        "金融资产重分类",
    ),
    "现金流量表": (
        "经营活动",
        "投资活动",
        "筹资活动",
        "收到的现金",
        "支付的现金",
        "现金流量净额",
        "现金及现金等价物",
    ),
}
_STATEMENT_PERIOD_RE = re.compile(
    r"(?:19|20)\d{2}(?:年\d{1,2}月\d{1,2}日)?",
    re.IGNORECASE,
)

_NUMERIC_TEXT_RE = re.compile(
    r"^[-+]?[\d,]+(\.\d+)?%?$"
    r"|^[（(][-]?[\d,]+(\.\d+)?%?[)）]$"
    r"|^-+$"
)


def _is_numeric_text(text: str) -> bool:
    t = (text or "").strip().replace(" ", "").replace("\u00a0", "")
    if not t:
        return False
    return bool(_NUMERIC_TEXT_RE.match(t))


def _is_blank_data_row(row: list[Any]) -> bool:
    if not isinstance(row, list) or not row:
        return True
    for idx, cell in enumerate(row):
        if idx == 0:
            continue
        if get_raw_cell_text(cell):
            return False
    return True


def _shrink_rowspans_for_removed_rows(
    raw_rows: list[list[Any]],
    removed_row_indices: set[int],
) -> None:
    """删除行后同步收缩覆盖被删行的跨行单元格。

    位于第 i 行、rowspan=r 的单元格覆盖第 i..i+r-1 行；若其中某行被删除，
    rowspan 必须相应减 1，否则残留的幽灵 rowSpan 会让下游
    align_table_rows 生成错误占位，把数据列挤出表格右边界。
    """
    if not removed_row_indices:
        return
    for row_idx, row in enumerate(raw_rows):
        if row_idx in removed_row_indices:
            continue
        if not isinstance(row, list):
            continue
        for cell in row:
            if not isinstance(cell, dict):
                continue
            row_span = to_int(
                cell.get("rowspan") or cell.get("rowSpan") or cell.get("row_span")
            )
            if row_span is None or row_span <= 1:
                continue
            removed_inside = sum(
                1
                for removed_idx in removed_row_indices
                if row_idx < removed_idx <= row_idx + row_span - 1
            )
            if not removed_inside:
                continue
            new_span = max(row_span - removed_inside, 1)
            for key in ("rowspan", "rowSpan", "row_span"):
                if key in cell:
                    cell[key] = new_span


def strip_residual_header_rows(rows: list[Any]) -> list[Any]:
    raw_rows = normalize_raw_rows(rows)
    if len(raw_rows) < 3:
        return rows if isinstance(rows, list) else raw_rows

    total_cols = infer_table_total_columns(raw_rows)
    if total_cols < 2:
        return rows if isinstance(rows, list) else raw_rows

    numeric_col_indices: set[int] = set()
    for col_idx in range(1, total_cols):
        total_non_empty = 0
        n_num = 0
        for row in raw_rows:
            if not isinstance(row, list):
                continue
            cell = find_raw_cell_covering_col(row, col_idx)
            text = get_raw_cell_text(cell) if cell is not None else ""
            if not text:
                continue
            total_non_empty += 1
            if _is_numeric_text(text):
                n_num += 1
        if total_non_empty >= 3 and n_num / total_non_empty >= 0.6:
            numeric_col_indices.add(col_idx)

    if not numeric_col_indices:
        return rows if isinstance(rows, list) else raw_rows

    kept_rows: list[Any] = []
    removed_count = 0
    removed_row_indices: set[int] = set()
    for row_idx, row in enumerate(raw_rows):
        if row_idx == 0:
            kept_rows.append(row)
            continue
        if not isinstance(row, list):
            kept_rows.append(row)
            continue
        if _is_blank_data_row(row):
            kept_rows.append(row)
            continue
        intrusion_count = 0
        for col_idx in numeric_col_indices:
            cell = find_raw_cell_covering_col(row, col_idx)
            text = get_raw_cell_text(cell) if cell is not None else ""
            if text and not _is_numeric_text(text):
                intrusion_count += 1
        if numeric_col_indices and intrusion_count / len(numeric_col_indices) >= 0.5:
            removed_count += 1
            removed_row_indices.add(row_idx)
            logger.debug(
                "strip_residual_header_rows: removed row_idx=%d intrusions=%d/%d",
                row_idx, intrusion_count, len(numeric_col_indices),
            )
            continue
        kept_rows.append(row)

    if removed_count > 0:
        _shrink_rowspans_for_removed_rows(raw_rows, removed_row_indices)
        logger.debug(
            "strip_residual_header_rows: removed %d residual header rows (numeric_cols=%s)",
            removed_count, sorted(numeric_col_indices),
        )
    return kept_rows

NormalizeStatementLabel = Callable[[str], Optional[str]]
FindNearestStatementTitle = Callable[..., tuple[Optional[dict[str, Any]], Optional[str]]]


def base_statement_type(label: str) -> str:
    if "所有者权益变动" in label:
        return "所有者权益变动表"
    for base in ("资产负债表", "利润表", "现金流量表"):
        if base in label:
            return base
    return label


def row_signature(row: Any) -> str:
    raw = normalize_raw_rows([row]) if row else []
    if not raw or not raw[0]:
        return ""
    parts = []
    for cell in raw[0]:
        text = get_raw_cell_text(cell).strip()
        if text:
            parts.append(text)
    return "|".join(parts)


def deduplicate_rows(rows: list[Any]) -> list[Any]:
    if not rows:
        return rows
    seen: set[str] = set()
    deduped: list[Any] = []
    dup_count = 0
    for row in rows:
        signature = row_signature(row)
        if not signature:
            deduped.append(row)
            continue
        if signature in seen:
            dup_count += 1
            continue
        seen.add(signature)
        deduped.append(row)
    if dup_count > 0:
        logger.debug("Row deduplication removed %d duplicate rows", dup_count)
    return deduped


def statement_scope(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    if "合并" in label:
        return "合并"
    if "母公司" in label or ("母" in label and "公司" in label):
        return "母公司"
    return None


def _nearest_previous_table_index(items: list[Any], idx: int) -> Optional[int]:
    for previous_idx in range(idx - 1, -1, -1):
        item = items[previous_idx]
        if isinstance(item, dict) and item.get("type") in {"table", "excel_table"}:
            return previous_idx
    return None


def _compact_row_label(row: list[Any]) -> str:
    for cell in row:
        text = re.sub(r"\s+", "", get_raw_cell_text(cell))
        if text:
            return re.sub(r"[：:，,。.]", "", text).lower()
    return ""


def _canonical_amount(text: str) -> Optional[str]:
    compact = re.sub(r"[\s,，]", "", text or "")
    if not compact or compact in {"-", "—", "–"}:
        return None
    negative = compact.startswith("(") and compact.endswith(")")
    if negative:
        compact = compact[1:-1]
    compact = compact.replace("−", "-").replace("－", "-")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
        return None
    if negative and not compact.startswith("-"):
        compact = f"-{compact}"
    return compact.lstrip("+")


def _row_amounts(row: list[Any]) -> tuple[str, ...]:
    values: list[str] = []
    found_label = False
    for cell in row:
        text = get_raw_cell_text(cell).strip()
        if not found_label:
            if text:
                found_label = True
            continue
        amount = _canonical_amount(text)
        if amount is not None:
            values.append(amount)
    return tuple(values)


def _is_asset_total_label(label: str) -> bool:
    return label in {"资产总计", "资产合计", "totalassets"}


def _is_liabilities_and_equity_total_label(label: str) -> bool:
    if label in {
        "负债和股东权益总计",
        "负债及股东权益总计",
        "负债和所有者权益总计",
        "负债及所有者权益总计",
        "权益和负债总计",
        "权益及负债总计",
        "totalequityandliabilities",
        "totalliabilitiesandshareholdersequity",
        "totalliabilitiesandstockholdersequity",
    }:
        return True
    return (
        "负债" in label
        and ("股东权益" in label or "所有者权益" in label)
        and label.endswith(("总计", "合计"))
    )


def _find_total_amounts(
    rows: list[Any],
    label_predicate: Callable[[str], bool],
) -> Optional[tuple[str, ...]]:
    for row in reversed(normalize_raw_rows(rows)):
        label = _compact_row_label(row)
        if not label_predicate(label):
            continue
        amounts = _row_amounts(row)
        return amounts or None
    return None


def _statement_periods(rows: list[Any]) -> tuple[str, ...]:
    periods: list[str] = []
    for row in normalize_raw_rows(rows)[:3]:
        for cell in row:
            compact = re.sub(r"\s+", "", get_raw_cell_text(cell))
            periods.extend(match.group(0) for match in _STATEMENT_PERIOD_RE.finditer(compact))
    return tuple(periods)


def _is_balance_sheet_equation_continuation(
    previous: Optional[dict[str, Any]],
    candidate: dict[str, Any],
) -> bool:
    """Use the balance-sheet equation as strong evidence across adjacent pages."""
    if not previous:
        return False
    previous_page = get_page_at(previous)
    candidate_page = get_page_at(candidate)
    if (
        previous_page is None
        or candidate_page is None
        or candidate_page != previous_page + 1
    ):
        return False

    previous_rows = previous.get("rows")
    candidate_rows = candidate.get("rows")
    if not isinstance(previous_rows, list) or not isinstance(candidate_rows, list):
        return False

    previous_normalized_rows = normalize_raw_rows(previous_rows)
    candidate_normalized_rows = normalize_raw_rows(candidate_rows)
    asset_totals = _find_total_amounts(
        previous_normalized_rows[-6:],
        _is_asset_total_label,
    )
    liability_equity_totals = _find_total_amounts(
        candidate_normalized_rows,
        _is_liabilities_and_equity_total_label,
    )
    if not asset_totals or asset_totals != liability_equity_totals:
        return False
    if _find_total_amounts(candidate_normalized_rows, _is_asset_total_label):
        return False
    if not any(
        "负债" in _compact_row_label(row) or "权益" in _compact_row_label(row)
        for row in candidate_normalized_rows[:6]
    ):
        return False

    previous_periods = _statement_periods(previous_rows)
    candidate_periods = _statement_periods(candidate_rows)
    if previous_periods and candidate_periods and previous_periods != candidate_periods:
        return False
    return True


def _looks_like_fresh_statement_table(item: dict[str, Any], base_type: str) -> bool:
    """Return whether a table restarts a statement instead of continuing one."""
    start_keywords = BASE_TYPE_START_KEYWORDS.get(base_type, ())
    rows = normalize_raw_rows(item.get("rows"))[:6]
    if not start_keywords or not rows:
        return False

    leading_texts = [
        "".join(get_raw_cell_text(cell).replace(" ", "") for cell in row)
        for row in rows
    ]
    has_header = any(
        "项目" in text or text.startswith("项") or "particulars" in text.lower()
        for text in leading_texts[:4]
    )
    return has_header and any(
        any(keyword in text for keyword in start_keywords)
        for text in leading_texts
    )


def _leading_ordinal(text: str) -> Optional[int]:
    match = re.match(r"^\s*(\d+)\s*[.．、]", text or "")
    return int(match.group(1)) if match else None


def _table_text(item: dict[str, Any]) -> str:
    table_md = item.get("md")
    if isinstance(table_md, str) and table_md.strip():
        return table_md
    return "\n".join(
        " | ".join(get_raw_cell_text(cell) for cell in row)
        for row in normalize_raw_rows(item.get("rows"))
    )


def _numbered_continuation_content_matches(table_text: str, base_type: str) -> bool:
    compact = "".join((table_text or "").split())
    keywords = _NUMBERED_CONTINUATION_KEYWORDS.get(base_type, ())
    primary_score = sum(1 for keyword in keywords if keyword in compact)
    if primary_score < 2:
        return False
    return all(
        primary_score
        > sum(1 for keyword in other_keywords if keyword in compact)
        for other_base, other_keywords in _NUMBERED_CONTINUATION_KEYWORDS.items()
        if other_base != base_type
    )


def _numbered_table_rows_continue(
    previous: dict[str, Any],
    current: dict[str, Any],
    base_type: str,
) -> bool:
    """Treat sequential row numbers as supporting, not sufficient, evidence."""
    current_rows = normalize_raw_rows(current.get("rows"))
    if not current_rows:
        return False
    if get_page_at(current) != (get_page_at(previous) or 0) + 1:
        return False

    previous_rows = normalize_raw_rows(previous.get("rows"))
    if not previous_rows or abs(len(previous_rows[-1]) - len(current_rows[0])) > 1:
        return False
    previous_text = get_raw_cell_text(previous_rows[-1][0]) if previous_rows[-1] else ""
    current_text = get_raw_cell_text(current_rows[0][0]) if current_rows[0] else ""
    previous_ordinal = _leading_ordinal(previous_text)
    current_ordinal = _leading_ordinal(current_text)
    return bool(
        previous_ordinal is not None
        and current_ordinal is not None
        and current_ordinal == previous_ordinal + 1
        and _numbered_continuation_content_matches(_table_text(current), base_type)
    )


def _table_bbox(item: dict[str, Any]) -> Optional[tuple[float, float, float, float]]:
    bbox = item.get("bBox") or item.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        left = float(bbox["l"])
        top = float(bbox["t"])
        right = float(bbox["r"])
        bottom = float(bbox["b"])
    except (KeyError, TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _is_same_page_vertical_cashflow_continuation(
    items: list[Any],
    previous_idx: int,
    candidate_idx: int,
    primary_base: str,
    *,
    normalize_statement_label: NormalizeStatementLabel,
) -> bool:
    """Recognize one cash-flow statement split into vertically stacked table blocks."""
    if primary_base != "现金流量表":
        return False
    previous = items[previous_idx] if 0 <= previous_idx < len(items) else None
    candidate = items[candidate_idx] if 0 <= candidate_idx < len(items) else None
    if not isinstance(previous, dict) or not isinstance(candidate, dict):
        return False
    if get_page_at(previous) != get_page_at(candidate):
        return False

    candidate_label = normalize_statement_label(str(candidate.get("table_type") or ""))
    if candidate_label and base_statement_type(candidate_label) != primary_base:
        return False

    previous_bbox = _table_bbox(previous)
    candidate_bbox = _table_bbox(candidate)
    if previous_bbox is None or candidate_bbox is None:
        return False
    prev_left, prev_top, prev_right, prev_bottom = previous_bbox
    curr_left, curr_top, curr_right, _ = candidate_bbox
    horizontal_overlap = min(prev_right, curr_right) - max(prev_left, curr_left)
    min_width = min(prev_right - prev_left, curr_right - curr_left)
    if horizontal_overlap <= 0 or horizontal_overlap / min_width < 0.8:
        return False
    if curr_top < prev_bottom - 2 or curr_top <= prev_top:
        return False

    between_texts = [
        get_text(item).replace(" ", "")
        for item in items[previous_idx + 1 : candidate_idx]
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    if any("现金流量表" in text for text in between_texts):
        return False
    return any(
        section in text
        for text in between_texts
        for section in _CASHFLOW_CONTINUATION_SECTIONS
    )


def get_continuation_evidence(
    items: list[Any],
    previous_idx: int,
    candidate_idx: int,
    primary_base: str,
    *,
    normalize_statement_label: NormalizeStatementLabel,
) -> Optional[str]:
    """Return the strongest shared classification/merge continuation evidence."""
    previous = items[previous_idx] if 0 <= previous_idx < len(items) else None
    candidate = items[candidate_idx] if 0 <= candidate_idx < len(items) else None
    if not isinstance(previous, dict) or not isinstance(candidate, dict):
        return None
    if _looks_like_fresh_statement_table(candidate, primary_base):
        return None

    intervening_labels = [
        normalize_statement_label(get_text(item))
        for item in items[previous_idx + 1 : candidate_idx]
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    if any(intervening_labels):
        return None

    if (
        primary_base == "资产负债表"
        and _is_balance_sheet_equation_continuation(previous, candidate)
    ):
        return "balance_sheet_equation"

    if _is_same_page_vertical_cashflow_continuation(
        items,
        previous_idx,
        candidate_idx,
        primary_base,
        normalize_statement_label=normalize_statement_label,
    ):
        return "same_page_cashflow_section"

    previous_page = get_page_at(previous)
    candidate_page = get_page_at(candidate)
    if (
        previous_page is None
        or candidate_page is None
        or candidate_page != previous_page + 1
    ):
        return None

    table_text = _table_text(candidate)
    if content_matches_other_base_type(table_text, primary_base):
        return None
    if content_matches_base_type(table_text, primary_base):
        return "statement_content"
    if _numbered_table_rows_continue(previous, candidate, primary_base):
        return "numbered_statement_content"
    return None


def repair_statement_continuation_labels(
    items: list[Any],
    *,
    normalize_statement_label: NormalizeStatementLabel,
) -> int:
    """Correct classifier output only when adjacent-table evidence is decisive."""
    table_indices = [
        index
        for index, item in enumerate(items)
        if isinstance(item, dict) and item.get("type") in {"table", "excel_table"}
    ]
    changed = 0
    for previous_idx, candidate_idx in zip(table_indices, table_indices[1:]):
        previous = items[previous_idx]
        candidate = items[candidate_idx]
        previous_label = normalize_statement_label(str(previous.get("table_type") or ""))
        if not previous_label:
            continue
        primary_base = base_statement_type(previous_label)
        if primary_base not in {"资产负债表", "利润表", "现金流量表"}:
            continue

        evidence = get_continuation_evidence(
            items,
            previous_idx,
            candidate_idx,
            primary_base,
            normalize_statement_label=normalize_statement_label,
        )
        if not evidence:
            continue

        candidate_label = normalize_statement_label(str(candidate.get("table_type") or ""))
        candidate_base = base_statement_type(candidate_label) if candidate_label else None
        previous_scope = statement_scope(previous_label)
        candidate_scope = statement_scope(candidate_label)

        if candidate_label:
            if candidate_base != primary_base and evidence not in {
                "balance_sheet_equation",
                "statement_content",
                "numbered_statement_content",
            }:
                continue
            if (
                candidate_base == primary_base
                and previous_scope
                and candidate_scope
                and previous_scope != candidate_scope
                and evidence not in {
                    "balance_sheet_equation",
                    "same_page_cashflow_section",
                }
            ):
                continue
            if candidate_label == previous_label:
                continue

        candidate["table_type"] = previous_label
        candidate["table_type_is_full_statement"] = False
        candidate["table_type_evidence"] = f"continuation:{evidence}"
        changed += 1
    return changed


def is_safe_continuation(
    items: list[Any],
    idx: int,
    primary_base: str,
    primary_label: str,
    *,
    normalize_statement_label: NormalizeStatementLabel,
    find_nearest_statement_title: FindNearestStatementTitle,
) -> bool:
    item = items[idx]
    primary_scope = statement_scope(primary_label)
    has_positive_evidence = False
    previous_idx = _nearest_previous_table_index(items, idx)
    continuation_evidence = (
        get_continuation_evidence(
            items,
            previous_idx,
            idx,
            primary_base,
            normalize_statement_label=normalize_statement_label,
        )
        if previous_idx is not None
        else None
    )
    balance_sheet_equation_continuation = (
        continuation_evidence == "balance_sheet_equation"
    )

    table_type_text = item.get("table_type")
    if table_type_text:
        candidate_normalized = normalize_statement_label(table_type_text)
        if candidate_normalized:
            if (
                base_statement_type(candidate_normalized) != primary_base
                and continuation_evidence not in {
                    "balance_sheet_equation",
                    "statement_content",
                    "numbered_statement_content",
                }
            ):
                return False
            has_positive_evidence = True
            candidate_scope = statement_scope(candidate_normalized)
            if (
                primary_scope
                and candidate_scope
                and candidate_scope != primary_scope
                and not balance_sheet_equation_continuation
            ):
                return False

    _, nearest_label = find_nearest_statement_title(items, table_index=idx)
    if nearest_label:
        if base_statement_type(nearest_label) != primary_base:
            return False
        nearest_scope = statement_scope(nearest_label)
        if (
            primary_scope
            and nearest_scope
            and nearest_scope != primary_scope
            and not balance_sheet_equation_continuation
        ):
            return False
        has_positive_evidence = True

    if balance_sheet_equation_continuation:
        return True

    if has_positive_evidence:
        return True

    return continuation_evidence is not None


def collect_continuation_rows_forward(
    items: list[Any],
    primary_idx: int,
    primary_label: str,
    primary_page: Optional[int],
    primary_base: str,
    *,
    normalize_statement_label: NormalizeStatementLabel,
    find_nearest_statement_title: FindNearestStatementTitle,
    max_page_gap: int = 2,
    max_look: int = 60,
    max_tables: int = 8,
) -> tuple[list[Any], list[int]]:
    extra_rows: list[Any] = []
    collected_indices: list[int] = []
    found = 0
    last_page = primary_page
    previous_table_idx = primary_idx

    for index in range(primary_idx + 1, min(len(items), primary_idx + max_look)):
        item = items[index]
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            new_label = normalize_statement_label(get_text(item))
            if new_label and base_statement_type(new_label) != primary_base:
                break
            continue
        if item.get("type") not in {"table", "excel_table"}:
            continue
        candidate_table_type = item.get("table_type")
        if candidate_table_type:
            candidate_normalized = normalize_statement_label(candidate_table_type)
            if (
                candidate_normalized
                and base_statement_type(candidate_normalized) != primary_base
                and get_continuation_evidence(
                    items,
                    previous_table_idx,
                    index,
                    primary_base,
                    normalize_statement_label=normalize_statement_label,
                )
                not in {
                    "balance_sheet_equation",
                    "statement_content",
                    "numbered_statement_content",
                }
            ):
                break
        item_page = get_page_at(item)
        same_page_vertical_continuation = _is_same_page_vertical_cashflow_continuation(
            items,
            previous_table_idx,
            index,
            primary_base,
            normalize_statement_label=normalize_statement_label,
        )
        if item_page is not None and primary_page is not None:
            if item_page <= primary_page and not same_page_vertical_continuation:
                continue
            if item_page - primary_page > MAX_ABSOLUTE_PAGE_GAP:
                break
        if item_page is not None and last_page is not None:
            if item_page - last_page > max_page_gap:
                break
        if _looks_like_fresh_statement_table(item, primary_base):
            logger.debug(
                "Forward continuation stopped for %r: idx=%d restarts a statement",
                primary_label,
                index,
            )
            break
        if (
            not same_page_vertical_continuation
            and not is_safe_continuation(
                items,
                index,
                primary_base,
                primary_label,
                normalize_statement_label=normalize_statement_label,
                find_nearest_statement_title=find_nearest_statement_title,
            )
        ):
            break
        item_rows = item.get("rows")
        if isinstance(item_rows, list) and item_rows:
            extra_rows.extend(item_rows)
            collected_indices.append(index)
            found += 1
            if item_page is not None:
                last_page = item_page
            previous_table_idx = index
            logger.debug("Forward continuation for %r: idx=%d, page=%s, rows=%d", primary_label, index, item_page, len(item_rows))
        if found >= max_tables:
            break
    return extra_rows, collected_indices


def collect_continuation_rows_backward(
    items: list[Any],
    primary_idx: int,
    primary_label: str,
    primary_page: Optional[int],
    primary_base: str,
    *,
    normalize_statement_label: NormalizeStatementLabel,
    find_nearest_statement_title: FindNearestStatementTitle,
    max_page_gap: int = 2,
    max_look: int = 60,
    max_tables: int = 8,
) -> tuple[list[Any], list[int]]:
    fragments: list[tuple[int, list[Any]]] = []
    found = 0
    last_page = primary_page
    for index in range(primary_idx - 1, max(-1, primary_idx - max_look - 1), -1):
        item = items[index]
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            new_label = normalize_statement_label(get_text(item))
            if new_label and base_statement_type(new_label) != primary_base:
                break
            continue
        if item.get("type") not in {"table", "excel_table"}:
            continue
        candidate_table_type = item.get("table_type")
        if candidate_table_type:
            candidate_normalized = normalize_statement_label(candidate_table_type)
            if candidate_normalized and base_statement_type(candidate_normalized) != primary_base:
                break
        item_page = get_page_at(item)
        if item_page is not None and primary_page is not None:
            if item_page >= primary_page:
                continue
            if primary_page - item_page > MAX_ABSOLUTE_PAGE_GAP:
                break
        if item_page is not None and last_page is not None:
            if last_page - item_page > max_page_gap:
                break
        if not is_safe_continuation(
            items,
            index,
            primary_base,
            primary_label,
            normalize_statement_label=normalize_statement_label,
            find_nearest_statement_title=find_nearest_statement_title,
        ):
            break
        item_rows = item.get("rows")
        if isinstance(item_rows, list) and item_rows:
            fragments.append((index, list(item_rows)))
            found += 1
            if item_page is not None:
                last_page = item_page
            logger.debug("Backward continuation for %r: idx=%d, page=%s, rows=%d", primary_label, index, item_page, len(item_rows))
        if found >= max_tables:
            break
    fragments.reverse()
    result: list[Any] = []
    collected_indices: list[int] = []
    for index, rows in fragments:
        result.extend(rows)
        collected_indices.append(index)
    return result, collected_indices


def collect_continuation_rows(
    items: list[Any],
    primary_idx: int,
    primary_label: str,
    *,
    normalize_statement_label: NormalizeStatementLabel,
    find_nearest_statement_title: FindNearestStatementTitle,
    max_page_gap: int = 2,
    max_look: int = 60,
    max_tables: int = 8,
) -> tuple[list[Any], list[Any], list[int]]:
    if primary_idx < 0 or primary_idx >= len(items):
        return [], [], []
    primary_table = items[primary_idx] if isinstance(items[primary_idx], dict) else {}
    primary_page = get_page_at(primary_table)
    primary_base = base_statement_type(primary_label)
    preceding, pre_indices = collect_continuation_rows_backward(
        items,
        primary_idx,
        primary_label,
        primary_page,
        primary_base,
        normalize_statement_label=normalize_statement_label,
        find_nearest_statement_title=find_nearest_statement_title,
        max_page_gap=max_page_gap,
        max_look=max_look,
        max_tables=max_tables,
    )
    following, fwd_indices = collect_continuation_rows_forward(
        items,
        primary_idx,
        primary_label,
        primary_page,
        primary_base,
        normalize_statement_label=normalize_statement_label,
        find_nearest_statement_title=find_nearest_statement_title,
        max_page_gap=max_page_gap,
        max_look=max_look,
        max_tables=max_tables,
    )
    return preceding, following, pre_indices + fwd_indices
