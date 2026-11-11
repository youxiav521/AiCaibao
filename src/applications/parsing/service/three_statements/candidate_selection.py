from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Optional

from src.applications.parsing.service.three_statements.labels import get_base_type

logger = logging.getLogger(__name__)


def estimate_table_rows_count(table: dict[str, Any]) -> int:
    rows = table.get("rows")
    if isinstance(rows, list):
        return len(rows)
    return 0


def pick_better_statement_candidate(
    prev: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    new_full = new.get("is_full", False)
    prev_full = prev.get("is_full", False)
    new_page = new.get("page")
    prev_page = prev.get("page")
    new_rows = new.get("rows_count", 0)
    prev_rows = prev.get("rows_count", 0)

    if new_full and not prev_full:
        return new
    if not new_full and prev_full:
        return prev
    if new_page is not None and prev_page is not None and new_page != prev_page:
        if new_page < prev_page and new_rows >= prev_rows * 0.5:
            return new
        if prev_page < new_page and prev_rows >= new_rows * 0.5:
            return prev
        if new_rows > prev_rows:
            return new
    elif new_rows > prev_rows:
        return new
    return prev


def update_best_candidate(
    best: dict[str, dict[str, Any]], label: str, candidate: dict[str, Any]
) -> None:
    prev = best.get(label)
    if prev is None:
        best[label] = candidate
        logger.debug(
            "Statement candidate registered: label=%s, page=%s, rows=%s, is_full=%s",
            label, candidate.get("page"), candidate.get("rows_count"), candidate.get("is_full"),
        )
        return

    winner = pick_better_statement_candidate(prev, candidate)
    should_replace = winner is candidate
    new_full = candidate.get("is_full", False)
    prev_full = prev.get("is_full", False)
    new_page = candidate.get("page")
    prev_page = prev.get("page")
    new_rows = candidate.get("rows_count", 0)
    prev_rows = prev.get("rows_count", 0)
    reason = ""
    if should_replace:
        if new_full and not prev_full:
            reason = "new candidate passed content validation"
        elif new_page is not None and prev_page is not None and new_page != prev_page:
            if new_page < prev_page and new_rows >= prev_rows * 0.5:
                reason = f"earlier page ({new_page} vs {prev_page}), sufficient rows ({new_rows} vs {prev_rows})"
            elif new_rows > prev_rows:
                reason = f"more rows ({new_rows} vs {prev_rows})"
        elif new_rows > prev_rows:
            reason = f"more rows ({new_rows} vs {prev_rows})"

    action = "replaced" if should_replace else "kept existing"
    logger.debug(
        "Candidate %s for %r: %s | new=(page=%s,rows=%s,full=%s) prev=(page=%s,rows=%s,full=%s)",
        action, label, reason or "no advantage",
        new_page, new_rows, new_full,
        prev_page, prev_rows, prev_full,
    )
    if should_replace:
        best[label] = candidate


def push_three_statement_candidate(
    label_to_candidates: defaultdict[str, list[dict[str, Any]]],
    siblings: list[dict[str, Any]],
    label: str,
    candidate: dict[str, Any],
    *,
    locale: Optional[str],
    statement_strategy: Any = None,
    looks_like_independent_balance_sheet_halves: Callable[[dict[str, Any], dict[str, Any]], bool],
    can_merge_adjacent_balance_sheet_halves: Callable[[dict[str, Any], dict[str, Any]], bool],
    merge_balance_sheet_split_candidates: Callable[..., dict[str, Any]],
    looks_like_independent_same_page_tables: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> None:
    candidate_list = label_to_candidates[label]
    base_type = get_base_type(label)
    if base_type == "资产负债表" and candidate_list:
        prev_bs = candidate_list[-1]
        if statement_strategy is not None and getattr(statement_strategy, "name", "") != "china":
            should_merge = statement_strategy.should_merge_balance_sheet_halves(prev_bs, candidate)
            if should_merge:
                candidate_list[-1] = merge_balance_sheet_split_candidates(prev_bs, candidate, locale=locale)
                return
        if looks_like_independent_balance_sheet_halves(prev_bs, candidate):
            siblings.append(candidate)
            return
        if can_merge_adjacent_balance_sheet_halves(prev_bs, candidate):
            candidate_list[-1] = merge_balance_sheet_split_candidates(prev_bs, candidate, locale=locale)
            return

    if candidate_list:
        prev_same = candidate_list[-1]
        if looks_like_independent_same_page_tables(prev_same, candidate):
            siblings.append(candidate)
            return

    expanded_index = candidate.get("expanded_idx")
    if expanded_index is not None:
        for index, exist in enumerate(candidate_list):
            if exist.get("expanded_idx") == expanded_index:
                candidate_list[index] = pick_better_statement_candidate(exist, candidate)
                return
    candidate_list.append(candidate)


def infer_scope_for_promotion(
    candidate: dict[str, Any],
    *,
    table_caption_joined: Callable[[dict[str, Any]], str],
) -> Optional[str]:
    merge_tokens = ("合并", "consolidated")
    standalone_tokens = ("母公司", "单体", "standalone", "parent only", "parent company")
    table = candidate.get("table") if isinstance(candidate.get("table"), dict) else {}
    blob_parts = [
        str(candidate.get("label") or ""),
        str(table.get("table_type") or ""),
        str(table.get("md") or ""),
        table_caption_joined(table),
    ]
    blob = "\n".join(part for part in blob_parts if part).lower()
    if any(token in blob for token in standalone_tokens):
        return "母公司"
    if any(token in blob for token in merge_tokens):
        return "合并"
    return None


def promote_no_prefix_in_multi(
    label_to_candidates: defaultdict[str, list[dict[str, Any]]],
    *,
    no_prefix_statement_labels: tuple[str, ...],
    merged_statement_labels: tuple[str, ...],
    table_caption_joined: Callable[[dict[str, Any]], str],
) -> None:
    for no_prefix, merged in zip(no_prefix_statement_labels, merged_statement_labels):
        queue = list(label_to_candidates.get(no_prefix) or [])
        if not queue:
            continue
        label_to_candidates[no_prefix] = []
        for candidate in queue:
            scope = infer_scope_for_promotion(candidate, table_caption_joined=table_caption_joined)
            if scope == "合并":
                target = merged
            elif scope == "母公司":
                target = f"母公司{no_prefix}"
            else:
                label_to_candidates[no_prefix].append(candidate)
                continue
            if label_to_candidates.get(target):
                candidate["label"] = no_prefix
                label_to_candidates[no_prefix].append(candidate)
                continue
            candidate["label"] = target
            label_to_candidates[target].append(candidate)
