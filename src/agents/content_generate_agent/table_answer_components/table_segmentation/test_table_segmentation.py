import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.content_generate_agent.input_models import TableItem
from src.agents.content_generate_agent.table_answer_components.table_segmentation.table_segmentation import (
    asegment_table_into_relations,
    asegment_table_into_relations_with_html,
)


DATA_DIR = PROJECT_ROOT / "storage" / "table_seg"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_INPUT = DATA_DIR / "table_elements_test_input.json"
DEFAULT_OUTPUT = DATA_DIR / "save_max.json"


async def _run_once(table: TableItem, model_id: str) -> Dict[str, Any]:
    """Run segmentation for a single TableItem and return combined result."""
    tables = await asegment_table_into_relations(table, model_id=model_id)
    # tables_html = await asegment_table_into_relations_with_html(table, model_id=model_id)

    return {
        "page": table.page,
        "element_id": getattr(table, "element_id", None),
        "file_id": getattr(table, "file_id", None),
        "tables": tables.model_dump(),
        # "tables_with_html": tables_html.model_dump(),
    }


async def _amain(input_path: Path, output_path: Path, model_id: str) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    logger.info(f"读取输入: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # 仅选择 type == "table" 的项，并强类型化为 TableItem
    tables: List[TableItem] = []
    for obj in data:
        if isinstance(obj, dict) and obj.get("type") == "table":
            try:
                tables.append(TableItem.model_validate(obj))
            except Exception as e:
                logger.error(f"TableItem 校验失败，跳过该条: {e}")
                continue

    logger.info(f"共读取到 {len(tables)} 个表格条目，开始分组...")

    results: List[Dict[str, Any]] = []
    for idx, t in enumerate(tables, 1):
        try:
            logger.info(f"处理表格 {idx}/{len(tables)}: page={t.page}, element_id={getattr(t, 'element_id', None)}")
            combined = await _run_once(t, model_id)
            results.append(combined)
        except Exception as e:
            logger.error(f"处理表格 {idx} 失败: {e}")
            results.append({
                "page": t.page,
                "element_id": getattr(t, "element_id", None),
                "file_id": getattr(t, "file_id", None),
                "error": str(e),
                "tables": {"table_list": []},
                # "tables_with_html": {"table_list": []},
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"写入结果: {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("完成")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test table segmentation")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="输入JSON路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出保存路径")
    parser.add_argument("--model_id", type=str, default="model_for_data_analyst", help="模型ID")
    args = parser.parse_args()

    asyncio.run(_amain(Path(args.input), Path(args.output), args.model_id))


if __name__ == "__main__":
    main()
