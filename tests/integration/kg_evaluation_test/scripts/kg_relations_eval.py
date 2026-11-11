"""按关系类别汇总多次知识图谱提取结果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from loguru import logger


_PROJECT_DIR = Path(__file__).resolve().parents[4]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.ade.knowledge_graph.client.prompt import RELATION_TYPE_LIST, RELATION_TYPES_TABLE
from src.ade.knowledge_graph.evaluate.multi import (
    discover_run_paths,
    parse_relation_schema,
    render_relation_type_markdown,
    run_relation_type_evaluation,
)
from src.ade.knowledge_graph.evaluate.report import render_html


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="多次知识图谱关系分类评估")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="使用前 N 个结果；默认使用 preprocess_results 中的全部 TXT",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "preprocess_results",
        help="1.txt ... n.txt 所在目录",
    )
    parser.add_argument("--gold", type=Path, default=script_dir / "gold.txt", help="Gold 图谱")
    parser.add_argument(
        "--silver", type=Path, default=script_dir / "silver.txt", help="Silver 图谱"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "relations_eval_report" / "relations_evaluation_report.md",
        help="Markdown 报告路径；HTML 使用相同文件名",
    )
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> tuple[Path, Path]:
    run_paths = discover_run_paths(args.input_dir.resolve())
    if args.runs is not None:
        if args.runs < 1:
            raise SystemExit("--runs 必须大于等于 1")
        if args.runs > len(run_paths):
            raise SystemExit(f"--runs={args.runs} 超过可用文件数 {len(run_paths)}")
        run_paths = run_paths[: args.runs]
    markdown_path = args.output.resolve()
    if markdown_path.suffix.lower() != ".md":
        raise SystemExit("--output 必须是 .md 文件路径")

    schema = parse_relation_schema(RELATION_TYPES_TABLE, RELATION_TYPE_LIST)
    result = run_relation_type_evaluation(
        args.gold.resolve(),
        args.silver.resolve(),
        run_paths,
        schema,
    )
    markdown = render_relation_type_markdown(result)
    html_path = markdown_path.with_suffix(".html")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(render_html(markdown), encoding="utf-8")
    logger.info(
        "关系分类评估报告已生成 | markdown={}, html={}, runs={}",
        markdown_path,
        html_path,
        result.run_count,
    )
    return markdown_path, html_path


def main(argv: Sequence[str] | None = None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
