from __future__ import annotations

import argparse
import sys
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.applications.parsing.utils.ragflow_parsers import num_tokens_from_string
from src.utils import json_utils


SLICE_TYPES: tuple[str, ...] = ("parents", "children", "grandchildren")


@dataclass(frozen=True)
class ChunkMetrics:
    char_count: int
    token_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 converted_output 中三种切片的字符数与 token 数分布"
    )
    parser.add_argument(
        "--base-dir",
        default="storage/mineru_output/converted_output",
        help="converted_output 根目录，默认: storage/mineru_output/converted_output",
    )
    return parser.parse_args()


def _safe_int_median(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _safe_float_mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json_utils.load(file)


def _extract_chunk_texts(payload: Any, *, path: Path) -> list[str]:
    if not isinstance(payload, list):
        logger.warning("跳过非列表 JSON 文件: path={}", path)
        return []

    texts: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            logger.warning("跳过非 dict 切片项: path={}, index={}", path, index)
            continue
        text = item.get("text")
        if text is None:
            logger.warning("切片项缺少 text 字段: path={}, index={}", path, index)
            continue
        texts.append(str(text))
    return texts


def _collect_metrics(dir_path: Path) -> tuple[list[ChunkMetrics], int]:
    metrics: list[ChunkMetrics] = []
    file_count = 0

    if not dir_path.exists():
        logger.warning("目录不存在，跳过统计: path={}", dir_path)
        return metrics, file_count

    for json_path in sorted(dir_path.glob("*.json")):
        file_count += 1
        try:
            payload = _load_json(json_path)
        except Exception:
            logger.exception("读取 JSON 失败: path={}", json_path)
            continue

        for text in _extract_chunk_texts(payload, path=json_path):
            metrics.append(
                ChunkMetrics(
                    char_count=len(text),
                    token_count=num_tokens_from_string(text),
                )
            )

    return metrics, file_count


def _format_stats(slice_type: str, metrics: list[ChunkMetrics], file_count: int) -> str:
    char_values = [item.char_count for item in metrics]
    token_values = [item.token_count for item in metrics]

    return (
        f"{slice_type:<14}"
        f"{file_count:>8}"
        f"{len(metrics):>10}"
        f"{(min(char_values) if char_values else 0):>12}"
        f"{_safe_int_median(char_values):>12.1f}"
        f"{_safe_float_mean(char_values):>12.1f}"
        f"{(max(char_values) if char_values else 0):>12}"
        f"{(min(token_values) if token_values else 0):>12}"
        f"{_safe_int_median(token_values):>12.1f}"
        f"{_safe_float_mean(token_values):>12.1f}"
        f"{(max(token_values) if token_values else 0):>12}"
    )


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    total_metrics: list[ChunkMetrics] = []
    total_files = 0

    logger.info("开始统计切片大小: base_dir={}", base_dir)
    logger.info(
        "{:<14}{:>8}{:>10}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "slice_type",
        "files",
        "chunks",
        "char_min",
        "char_med",
        "char_avg",
        "char_max",
        "tok_min",
        "tok_med",
        "tok_avg",
        "tok_max",
    )

    for slice_type in SLICE_TYPES:
        dir_path = base_dir / slice_type
        metrics, file_count = _collect_metrics(dir_path)
        total_metrics.extend(metrics)
        total_files += file_count
        logger.info("{}", _format_stats(slice_type, metrics, file_count))
    logger.info("{}", _format_stats("total", total_metrics, total_files))


if __name__ == "__main__":
    main()
