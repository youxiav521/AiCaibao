"""多次知识图谱离线评估脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from loguru import logger


_PROJECT_DIR = Path(__file__).resolve().parents[4]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.ade.knowledge_graph.evaluate.multi import (
    discover_run_paths,
    render_multi_markdown,
    run_multi_evaluation,
)
from src.ade.knowledge_graph.evaluate.report import render_html


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="多次知识图谱离线评估")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "preprocess_results",
        help="1.txt ... n.txt 所在目录",
    )
    parser.add_argument(
        "--gold", type=Path, default=script_dir / "gold.txt", help="Gold 图谱"
    )
    parser.add_argument(
        "--silver", type=Path, default=script_dir / "silver.txt", help="Silver 图谱"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "multi_kg_report" / "multi_kg_evaluation_report.md",
        help="Markdown 报告路径",
    )
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> tuple[Path, Path]:
    result = run_multi_evaluation(
        args.gold.resolve(),
        args.silver.resolve(),
        discover_run_paths(args.input_dir.resolve()),
    )
    markdown_report = render_multi_markdown(result)
    markdown_path = args.output.resolve()
    html_path = markdown_path.with_suffix(".html")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report, encoding="utf-8")
    html_path.write_text(render_html(markdown_report), encoding="utf-8")
    logger.info(
        "多次知识图谱评估报告已生成 | markdown={}, html={}, runs={}",
        markdown_path,
        html_path,
        result.run_count,
    )
    return markdown_path, html_path


def main(argv: Sequence[str] | None = None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
