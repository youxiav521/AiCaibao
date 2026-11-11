"""单次知识图谱离线评估脚本。

指标仅以 Gold 为准；Silver 只用于标记可能由 OCR 错误导致的差异。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from loguru import logger

_PROJECT_DIR = Path(__file__).resolve().parents[4]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.ade.knowledge_graph.evaluate.metrics import run_evaluation
from src.ade.knowledge_graph.evaluate.parser import parse_graph
from src.ade.knowledge_graph.evaluate.report import (
    render_confidence_markdown,
    render_html,
    render_markdown,
)
from src.ade.knowledge_graph.evaluate.schema import EvaluationOptions


_REPORT_NAME = "single_kg_evaluation_report.md"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="单次知识图谱离线评估")
    parser.add_argument("--gold", type=Path, default=script_dir / "gold.txt", help="Gold 图谱")
    parser.add_argument(
        "--target",
        type=Path,
        default=script_dir / "preprocess_results" / "1.txt",
        help="待测图谱",
    )
    parser.add_argument(
        "--silver", type=Path, default=script_dir / "silver.txt", help="Silver 图谱"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "single_kg_report" / _REPORT_NAME,
        help="Markdown 报告路径；无后缀时视为输出目录",
    )
    parser.add_argument(
        "--gold-format", choices=["auto", "txt", "json"], default="auto"
    )
    parser.add_argument(
        "--target-format", choices=["auto", "txt", "json"], default="auto"
    )
    parser.add_argument(
        "--silver-format", choices=["auto", "txt", "json"], default="auto"
    )
    parser.add_argument("--include-attributes", action="store_true", help="是否评估属性")
    parser.add_argument(
        "--list-limit", type=int, default=0, help="普通条目最大列举数量，0 表示全部"
    )
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.list_limit < 0:
        raise SystemExit("--list-limit 不能小于 0")
    if not args.gold.exists():
        raise SystemExit(f"Gold 文件不存在: {args.gold}")
    if not args.target.exists():
        raise SystemExit(f"Target 文件不存在: {args.target}")

    silver_path = args.silver.resolve()
    silver = None
    if silver_path.exists():
        silver = parse_graph(silver_path, args.silver_format)
    else:
        logger.warning("Silver 文件不存在，已跳过 OCR 标记 | path={}", silver_path)

    result = run_evaluation(
        args.gold.resolve(),
        args.target.resolve(),
        gold_format=args.gold_format,
        target_format=args.target_format,
        options=EvaluationOptions(
            include_attributes=args.include_attributes,
            list_limit=args.list_limit,
        ),
    )
    markdown_report = render_markdown(
        result,
        silver_path=silver_path,
        silver=silver,
    )
    markdown_path = _output_path(args.output).resolve()
    html_path = markdown_path.with_suffix(".html")
    confidence_markdown_path = markdown_path.parent / "confidence_report.md"
    confidence_html_path = confidence_markdown_path.with_suffix(".html")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report, encoding="utf-8")
    html_path.write_text(render_html(markdown_report), encoding="utf-8")
    confidence_report = render_confidence_markdown(
        result,
        silver_path=silver_path,
        silver=silver,
    )
    confidence_markdown_path.write_text(confidence_report, encoding="utf-8")
    confidence_html_path.write_text(render_html(confidence_report), encoding="utf-8")
    logger.info(
        "单次知识图谱评估报告已生成 | markdown={}, html={}, confidence_markdown={}, confidence_html={}",
        markdown_path,
        html_path,
        confidence_markdown_path,
        confidence_html_path,
    )
    return markdown_path, html_path, confidence_markdown_path, confidence_html_path


def _output_path(output: Path) -> Path:
    return output if output.suffix else output / _REPORT_NAME


def main(argv: Sequence[str] | None = None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
