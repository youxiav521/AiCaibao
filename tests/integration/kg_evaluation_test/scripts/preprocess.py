"""清洗单个或一批知识图谱 JSON，并生成便于人工审阅的表格文本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[4]
KG_EVALUATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = KG_EVALUATION_DIR / "origin_kgs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "preprocess_results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.json_utils import dump, load  # noqa: E402


CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}


def render_attributes(attributes: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, raw_value in attributes.items():
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        parts.extend(
            f"{key}={value}"
            for value in values
            if value is not None and str(value).strip()
        )
    return ", ".join(parts)


def has_reference_confidence(payload: dict[str, Any]) -> bool:
    return any(
        "confidence" in reference
        for collection_name in ("entities", "relations")
        for item in payload[collection_name]
        for reference in item.get("references", [])
        if isinstance(reference, dict)
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预处理知识图谱 JSON")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="原始 JSON 文件或目录",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="预处理结果目录"
    )
    return parser.parse_args(argv)


def clean_graph(payload: Any, source_path: Path) -> dict[str, Any]:
    """原地删除评估无关的大字段，并返回清洗后的图谱。"""
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {source_path}")

    for collection_name in ("entities", "relations"):
        collection = payload.get(collection_name)
        if not isinstance(collection, list):
            raise ValueError(f"缺少 {collection_name} 数组: {source_path}")
        for index, item in enumerate(collection, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{collection_name}[{index}] 必须是对象: {source_path}"
                )
            item.pop("embed", None)

    if has_reference_confidence(payload):
        for collection_name in ("entities", "relations"):
            for item in payload[collection_name]:
                confidences = (
                    reference.get("confidence")
                    for reference in item.get("references", [])
                    if isinstance(reference, dict)
                )
                item["confidence"] = max(
                    (
                        confidence
                        for confidence in confidences
                        if isinstance(confidence, str)
                        and confidence in CONFIDENCE_RANK
                    ),
                    key=CONFIDENCE_RANK.get,
                    default=None,
                )

    payload.pop("files", None)
    return payload


def render_table(payload: dict[str, Any]) -> str:
    entities = payload["entities"]
    relations = payload["relations"]
    show_confidence = has_reference_confidence(payload)
    entity_header = (
        f"{'序号':>4s}  {'name':30s}  {'type':10s}  {'type_en':15s}  "
        f"{'attributes'}  {'confidence':10s}"
        if show_confidence
        else f"{'序号':>4s}  {'name':30s}  {'type':10s}  {'type_en':15s}  {'attributes'}"
    )
    lines = [
        "=" * 80,
        f"总实体: {len(entities)}  总关系: {len(relations)}",
        "=" * 80,
        "",
        "━" * 80,
        "  ENTITIES",
        "━" * 80,
        entity_header,
        "━" * 80,
    ]

    for index, entity in enumerate(entities, start=1):
        attributes = entity.get("attributes") or {}
        if not isinstance(attributes, dict):
            raise ValueError(f"entities[{index}].attributes 必须是对象")
        attributes_text = render_attributes(attributes)
        entity_text = (
            f"{index:4d}  {entity['name']:30s}  {entity['type']:10s}  "
            f"{entity['type_en']:15s}  "
        )
        entity_text += attributes_text
        if show_confidence:
            entity_text += f"  {entity.get('confidence') or '':10s}"
        lines.append(entity_text)

    relation_header = (
        f"{'序号':>4s}  {'from_entity':30s}  {'type':12s}  {'type_en':15s}  "
        f"{'to_entity'}  {'attributes'}  {'confidence':10s}"
        if show_confidence
        else (
            f"{'序号':>4s}  {'from_entity':30s}  {'type':12s}  "
            f"{'type_en':15s}  {'to_entity'}  {'attributes'}"
        )
    )

    lines.extend(
        [
            "",
            "━" * 80,
            "  RELATIONS",
            "━" * 80,
            relation_header,
            "━" * 80,
        ]
    )
    for index, relation in enumerate(relations, start=1):
        attributes = relation.get("attributes") or {}
        if not isinstance(attributes, dict):
            raise ValueError(f"relations[{index}].attributes 必须是对象")
        relation_text = (
            f"{index:4d}  {relation['from_entity']:30s}  {relation['type']:12s}  "
            f"{relation.get('type_en', ''):15s}  "
        )
        relation_text += f"{relation['to_entity']}  {render_attributes(attributes)}"
        if show_confidence:
            relation_text += f"  {relation.get('confidence') or '':10s}"
        lines.append(relation_text)
    lines.append("━" * 80)
    return "\n".join(lines)


def preprocess(input_path: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"输入文件必须是 JSON: {input_path}")
        input_paths = [input_path]
    elif input_path.is_dir():
        input_paths = sorted(input_path.glob("*.json"), key=lambda path: path.name)
        if not input_paths:
            raise FileNotFoundError(f"输入目录中没有 JSON 文件: {input_path}")
    else:
        raise ValueError(f"不支持的输入路径: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[Path, Path]] = []
    total_entities = 0
    total_relations = 0

    for input_path in input_paths:
        payload = clean_graph(load(input_path), input_path)
        table = render_table(payload)
        json_path = output_dir / input_path.name
        txt_path = output_dir / f"{input_path.stem}.txt"

        dump(payload, json_path, indent=2)
        txt_path.write_text(table, encoding="utf-8")

        entity_count = len(payload["entities"])
        relation_count = len(payload["relations"])
        total_entities += entity_count
        total_relations += relation_count
        outputs.append((json_path, txt_path))
        logger.info(
            "预处理完成 | input={}, entities={}, relations={}, json={}, txt={}",
            input_path,
            entity_count,
            relation_count,
            json_path,
            txt_path,
        )

    logger.info(
        "预处理汇总 | files={}, entities={}, relations={}, output_dir={}",
        len(outputs),
        total_entities,
        total_relations,
        output_dir,
    )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preprocess(args.input, args.output_dir)


if __name__ == "__main__":
    main()
