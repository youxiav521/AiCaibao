import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.content_generate_agent.table_answer_components.table_segmentation.table_segmentation import (
    asegment_html_table_into_relations,
)


DATA_DIR = PROJECT_ROOT / "storage" / "table_seg_temp"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_INPUT_HTML = DATA_DIR / "table_sample.html"
DEFAULT_OUTPUT = DATA_DIR / "save_temp.json"


async def _run_once(table_html: str, model_id: str) -> Dict[str, Any]:
    """调用测试方法并返回结构化结果。"""
    tables = await asegment_html_table_into_relations(table_html, model_id=model_id)
    return {
        "tables": tables.model_dump(),
    }


async def _amain(input_html_path: Path, output_path: Path, model_id: str) -> None:
    if not input_html_path.exists():
        raise FileNotFoundError(f"输入 HTML 文件不存在: {input_html_path}")

    logger.info(f"读取 HTML: {input_html_path}")
    table_html = input_html_path.read_text(encoding="utf-8")
    if not table_html.strip():
        raise ValueError(f"HTML 文件为空: {input_html_path}")

    try:
        result = await _run_once(table_html, model_id)
    except Exception as e:
        logger.error(f"处理 HTML 失败: {e}")
        result = {
            "error": str(e),
            "tables": {"table_list": []},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"写入结果: {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("完成")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test table segmentation (HTML 输入)")
    parser.add_argument("--input_html", type=str, default=str(DEFAULT_INPUT_HTML), help="输入 HTML 文件路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出保存路径")
    parser.add_argument("--model_id", type=str, default="model_for_data_analyst", help="模型ID")
    args = parser.parse_args()

    asyncio.run(_amain(Path(args.input_html), Path(args.output), args.model_id))


if __name__ == "__main__":
    main()


