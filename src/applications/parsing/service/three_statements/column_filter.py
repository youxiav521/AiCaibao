from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.applications.parsing.service.three_statements.cells import get_raw_cell_text, to_int
from src.applications.parsing.service.three_statements.headers import (
    STATEMENT_NOTE_PATTERNS,
    looks_like_amount_text,
    looks_like_date_header_text,
    looks_like_sequence_number_text,
    looks_like_subject,
    normalize_header_text,
)

logger = logging.getLogger(__name__)


def call_table_column_filter_llm(prompt_text: str) -> dict[str, Any]:
    try:
        from src.applications.parsing.extractor.base import _call_llm_json

        model_id = "model_for_table_header_aided"
        if not model_id:
            return {}
        return _call_llm_json(model_id, prompt_text)
    except Exception as exc:
        logger.warning("Column filter LLM call failed: %s", exc)
        return {}


def build_table_column_profiles(
    cell_grid: list[list[Optional[dict[str, Any]]]],
    *,
    header_row_count: int = 3,
    sample_row_count: int = 6,
) -> list[dict[str, Any]]:
    if not cell_grid or not cell_grid[0]:
        return []

    total_columns = len(cell_grid[0])
    actual_header_row_count = min(max(header_row_count, 0), len(cell_grid))
    effective_sample_count = max(sample_row_count, 1) * (2 if actual_header_row_count == 0 else 1)
    last_sample_row_index = min(len(cell_grid), actual_header_row_count + effective_sample_count)

    column_profiles: list[dict[str, Any]] = []
    for column_index in range(total_columns):
        header_parts: list[str] = []
        for row_index in range(actual_header_row_count):
            cell_text = normalize_header_text(get_raw_cell_text(cell_grid[row_index][column_index]))
            if cell_text and (not header_parts or header_parts[-1] != cell_text):
                header_parts.append(cell_text)

        header_text = " ".join(header_parts).strip()
        sample_texts: list[str] = []
        for row_index in range(actual_header_row_count, last_sample_row_index):
            sample_cell_text = get_raw_cell_text(cell_grid[row_index][column_index])
            if sample_cell_text:
                sample_texts.append(sample_cell_text)

        column_profiles.append(
            {
                "index": column_index,
                "header": header_text,
                "samples": sample_texts,
            }
        )

    return column_profiles


def build_table_column_filter_prompt(column_profiles: list[dict[str, Any]]) -> str:
    has_any_header = any(profile.get("header") for profile in column_profiles)
    no_header_hint = ""
    if not has_any_header:
        no_header_hint = (
            "\n注意：该表格没有表头行，所有 header 字段均为空。"
            "请完全根据 samples 内容来判断列的用途：\n"
            "- 如果某列 samples 中主要是中文科目名称（如'流动资产合计''营业收入''净利润'等），则为科目列；\n"
            "- 如果某列 samples 中主要是数字金额（如'156,331,445.85'），则为时间金额列；\n"
            "- 如果某列 samples 中主要是连续整数（如'1''2''3'），则为行次/序号列，应排除。\n"
        )

    return (
        "你是一个财务报表“三大表”(资产负债表/利润表/现金流量表)的表格列筛选器。\n"
        "我会给你每一列的列索引、表头文本(可能为空)以及该列若干样例单元格内容。\n"
        + no_header_hint +
        "\n"
        "任务：判断每一列是否应该保留。只保留两类列：\n"
        "1) 科目列：项目/科目/Item/名称等（描述行含义的那一列）\n"
        "2) 时间金额列：年份、期末/期初、本期/上期、本年/上年、金额/余额等（对应数值的列）\n"
        "\n"
        "需要排除的常见列：行次/序号/编号/No、附注/备注/说明/Note/Notes、单位 等辅助列。\n"
        "\n"
        "输出要求：只输出 JSON 对象，格式严格为：\n"
        "{\"keep_columns\": [0, 1, 2]}\n"
        "其中 keep_columns 为需要保留的列索引（从 0 开始），按升序且不重复。\n"
        "如果无法判断或不确定，请输出：{\"keep_columns\": []}。\n"
        "\n"
        "columns:\n"
        f"{json.dumps(column_profiles, ensure_ascii=False, indent=2)}\n"
    )


def choose_keep_column_indices_by_llm(
    cell_grid: list[list[Optional[dict[str, Any]]]],
    *,
    header_row_count: int = 3,
) -> Optional[list[int]]:
    if not cell_grid or not cell_grid[0]:
        return None

    column_profiles = build_table_column_profiles(cell_grid, header_row_count=header_row_count)
    if not column_profiles:
        return None

    prompt_text = build_table_column_filter_prompt(column_profiles)
    llm_result = call_table_column_filter_llm(prompt_text)
    if not isinstance(llm_result, dict):
        return None

    keep_columns_raw = llm_result.get("keep_columns")
    if not isinstance(keep_columns_raw, list) or not keep_columns_raw:
        return None

    total_columns = len(cell_grid[0])
    keep_column_indices: list[int] = []
    for value in keep_columns_raw:
        column_index = to_int(value)
        if column_index is None:
            continue
        if 0 <= column_index < total_columns:
            keep_column_indices.append(column_index)

    keep_column_indices = sorted(set(keep_column_indices))
    if len(keep_column_indices) < 2:
        return None
    return keep_column_indices


def choose_keep_column_indices_by_rule(
    cell_grid: list[list[Optional[dict[str, Any]]]],
    *,
    header_row_count: int,
) -> Optional[list[int]]:
    if not cell_grid or not cell_grid[0]:
        return None

    total_columns = len(cell_grid[0])
    start_row = min(max(header_row_count, 0), len(cell_grid) - 1)
    subject_scores: list[float] = [0.0] * total_columns
    amount_scores: list[float] = [0.0] * total_columns
    sequence_scores: list[float] = [0.0] * total_columns
    auxiliary_header_penalties: list[float] = [0.0] * total_columns

    for column_index in range(total_columns):
        header_texts: list[str] = []
        for row_index in range(min(header_row_count, len(cell_grid))):
            text = get_raw_cell_text(cell_grid[row_index][column_index]).strip()
            if text:
                header_texts.append(text)

        if header_texts:
            joined_header = " ".join(header_texts)
            if any(token in joined_header for token in ("行次", "序号", "编号", "No", "NO", "附注", "注释", "备注", "说明")):
                auxiliary_header_penalties[column_index] += 5.0
            elif any(text.strip().lower() in STATEMENT_NOTE_PATTERNS for text in header_texts):
                auxiliary_header_penalties[column_index] += 5.0
            if all(looks_like_sequence_number_text(text) for text in header_texts):
                sequence_scores[column_index] += 3.0
            if any(looks_like_date_header_text(text) for text in header_texts):
                amount_scores[column_index] += 1.5

    sample_rows = cell_grid[start_row:] or cell_grid
    for row in sample_rows:
        for column_index in range(total_columns):
            text = get_raw_cell_text(row[column_index]).strip()
            if not text:
                continue
            if looks_like_amount_text(text):
                amount_scores[column_index] += 1.0
            elif looks_like_sequence_number_text(text):
                sequence_scores[column_index] += 1.0
            elif looks_like_subject(text):
                subject_scores[column_index] += 1.0

    subject_candidates = [
        index
        for index in range(total_columns)
        if subject_scores[index] > 0
        and subject_scores[index] >= amount_scores[index]
        and sequence_scores[index] <= max(subject_scores[index], 1.0)
        and auxiliary_header_penalties[index] < 5.0
    ]
    if not subject_candidates:
        return None

    subject_index = max(
        subject_candidates,
        key=lambda index: (
            subject_scores[index] - amount_scores[index] - sequence_scores[index] - auxiliary_header_penalties[index],
            -index,
        ),
    )

    amount_candidates = [
        index
        for index in range(total_columns)
        if index != subject_index
        and amount_scores[index] > 0
        and sequence_scores[index] < amount_scores[index] + 1.0
        and auxiliary_header_penalties[index] < 5.0
    ]
    if not amount_candidates:
        return None

    preferred_amounts = [index for index in amount_candidates if index > subject_index]
    if preferred_amounts:
        amount_candidates = preferred_amounts

    amount_candidates = sorted(
        amount_candidates,
        key=lambda index: (
            -(amount_scores[index] - sequence_scores[index] - auxiliary_header_penalties[index]),
            index,
        ),
    )
    keep_column_indices = sorted(set([subject_index] + amount_candidates[:16]))
    return keep_column_indices if len(keep_column_indices) >= 2 else None
