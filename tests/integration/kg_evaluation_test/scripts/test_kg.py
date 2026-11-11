"""知识图谱构建测试脚本

从 converted/ 读取所有 converted.json，调用完整的图谱构建流程，
最终验证图谱已保存到 MinIO，并将结果导出到本地。

用法:
    .venv/bin/python tests/integration/kg_evaluation_test/scripts/test_kg.py
    .venv/bin/python tests/integration/kg_evaluation_test/scripts/test_kg.py --runs 10
    .venv/bin/python tests/integration/kg_evaluation_test/scripts/test_kg.py --input-dir /path/to/converted
"""

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

from loguru import logger


# ── 路径解析（必须在 import 业务模块前执行）───────────
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# 加载 .env
def _load_env():
    env_local = _PROJECT_DIR / ".env.local"
    env_file = _PROJECT_DIR / ".env"
    env_path = env_local if env_local.exists() else env_file
    if not env_path.exists():
        logger.warning("未找到环境配置文件，预期路径: {} 或 {}", env_local, env_file)
        return False

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)

    logger.info("已加载环境配置: {}", env_path)
    return True


# ── 数据路径 ─────────────────────────
_CONVERTED_DIR = _PROJECT_DIR / "tests" / "integration" / "kg_evaluation_test" / "ocr_converted" / "礼光"
_ORIGIN_KGS_DIR = _PROJECT_DIR / "tests" / "integration" / "kg_evaluation_test" / "origin_kgs"
_ORIGIN_RESPONSE_DIR = _PROJECT_DIR / "tests" / "integration" / "kg_evaluation_test" / "origin_response"
_LOG_PATH = _ORIGIN_KGS_DIR / "log.txt"
_FOLDER_ID = "TASK-XX"
_FOLDER_NAME = _CONVERTED_DIR.name + "信贷尽调项目"


def parse_args():
    parser = argparse.ArgumentParser(description="生成一个或多个知识图谱")
    parser.add_argument("--runs", type=int, default=1, help="串行生成次数（默认: 1）")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_CONVERTED_DIR,
        help=f"converted 根目录（默认: {_CONVERTED_DIR}）",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必须大于等于 1")
    return args


def discover_files(converted_dir: Path) -> list[dict]:
    """递归扫描 JSON 文件，返回 FileInput 的 dict 列表。"""
    from src.ade.knowledge_graph.utils.preprocess_file import converted_json_to_fileinput

    converted_dir = converted_dir.expanduser().resolve()
    if not converted_dir.is_dir():
        logger.error("测试数据目录不存在: {}", converted_dir)
        sys.exit(1)

    json_paths = sorted(converted_dir.rglob("*.json"))
    if not json_paths:
        logger.error("未找到任何 JSON 文件: {}", converted_dir)
        sys.exit(1)

    files = []
    for path in json_paths:
        file_name = path.parent.name if path.name == "converted.json" else path.stem
        file_input = converted_json_to_fileinput(
            path,
            file_id=file_name,
            file_name=file_name,
        )
        files.append(file_input.model_dump())
    return files


def build_output_paths(runs: int) -> list[Path]:
    _ORIGIN_KGS_DIR.mkdir(parents=True, exist_ok=True)
    existing_numbers = [
        int(path.stem)
        for path in _ORIGIN_KGS_DIR.glob("*.json")
        if path.stem.isdigit()
    ]
    start = max(existing_numbers, default=0) + 1
    return [_ORIGIN_KGS_DIR / f"{start + index}.json" for index in range(runs)]


def _piece_index(piece) -> int:
    prefix = f"piece_{piece.file_id}_"
    suffix = piece.id[len(prefix):] if piece.id.startswith(prefix) else ""
    return int(suffix) if suffix.isdigit() else 0


def _write_raw_responses(raw_responses: list[tuple], files: list, output_path: Path) -> Path:
    file_order = {file.id: index for index, file in enumerate(files)}
    ordered_responses = sorted(
        raw_responses,
        key=lambda item: (
            file_order.get(item[0].file_id, len(file_order)),
            _piece_index(item[0]),
            item[0].id,
        ),
    )
    sections = [
        f"===== file_name={piece.file_name} | piece_id={piece.id} =====\n{raw_response}"
        for piece, raw_response in ordered_responses
    ]

    _ORIGIN_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    response_path = _ORIGIN_RESPONSE_DIR / f"{output_path.stem}.txt"
    response_path.write_text("\n\n".join(sections), encoding="utf-8")
    return response_path


async def generate_kg(files, output_path: Path):
    """生成、验证并导出一个知识图谱。"""
    logger.info("构建图谱...")
    from src.ade.knowledge_graph.service.knowledge_graph_service import KnowledgeGraphService

    raw_responses = []
    graph_id = f"kg_test_{uuid.uuid4().hex[:12]}"
    try:
        graph = await KnowledgeGraphService.generate(
            files=files,
            folder_id=_FOLDER_ID,
            folder_name=_FOLDER_NAME,
            graph_id=graph_id,
            raw_response_callback=lambda piece, response: raw_responses.append((piece, response)),
        )
    finally:
        response_path = _write_raw_responses(raw_responses, files, output_path)
        logger.info("      LLM诊断响应已保存: {}", response_path)

    logger.info("      图谱构建完成")
    logger.info("      graph_id : {}", graph.id)
    logger.info("      entities : {}", len(graph.entities))
    logger.info("      relations: {}", len(graph.relations))

    if graph.entities:
        logger.info("\n      ── 实体列表 ──")
        for i, entity in enumerate(graph.entities[:20], 1):
            logger.info("      {:3d}. {:25s} type={}", i, entity.name, entity.type)
        if len(graph.entities) > 20:
            logger.info("      ... 共 {} 个实体", len(graph.entities))

    if graph.relations:
        logger.info("\n      ── 关系列表 ──")
        for i, relation in enumerate(graph.relations[:20], 1):
            logger.info(
                "      {:3d}. {:20s} ─[{}]─> {}",
                i,
                relation.from_entity,
                relation.type,
                relation.to_entity,
            )
        if len(graph.relations) > 20:
            logger.info("      ... 共 {} 个关系", len(graph.relations))

    logger.info("\n验证 MinIO 持久化...")
    from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService

    saved = await GraphStorageService.exists(_FOLDER_ID)
    logger.info("      图谱文件存在 : {}", saved)
    assert saved, f"MinIO 中未找到图谱: folder_id={_FOLDER_ID}"

    restored = await GraphStorageService.get(_FOLDER_ID)
    assert restored is not None, "从 MinIO 读取图谱失败"
    assert len(restored.entities) == len(graph.entities), "实体数量不一致"
    assert len(restored.relations) == len(graph.relations), "关系数量不一致"
    logger.info(
        "      读回验证通过: entities={} relations={}",
        len(restored.entities),
        len(restored.relations),
    )

    logger.info("\n导出本地 JSON...")
    from src.utils.json_utils import dump

    dump(restored.model_dump(by_alias=True), output_path, indent=2)
    logger.info("      本地文件已保存: {}", output_path)

async def run_batch(runs: int, input_dir: Path):
    """串行生成指定数量的知识图谱。"""
    logger.info("\n{}", "=" * 60)
    logger.info("  知识图谱构建测试")
    logger.info("{}\n", "=" * 60)

    logger.info("扫描测试数据...")
    files_data = discover_files(input_dir)
    logger.info("      共发现 {} 份文档\n", len(files_data))

    from src.ade.knowledge_graph.schema.kg_schema import FileInput
    from src.utils.model_factory import ModelFactory

    files = [FileInput(**data) for data in files_data]
    output_paths = build_output_paths(runs)

    try:
        for index, output_path in enumerate(output_paths, 1):
            logger.info("\n{}", "-" * 60)
            logger.info("  运行 {}/{} | 输出: {}", index, runs, output_path)
            logger.info("{}\n", "-" * 60)
            await generate_kg(files, output_path)
    finally:
        logger.info("\n清理资源...")
        # 显式关闭所有 Provider 的 httpx 客户端（在事件循环关闭前）
        for provider in ModelFactory._providers.values():
            provider.close()
        logger.info("      httpx 客户端已清理")

    logger.info("\n{}", "=" * 60)
    logger.info("  ✅ 知识图谱构建测试通过，共生成 {} 个图谱", runs)
    logger.info("{}\n", "=" * 60)


def main():
    _ORIGIN_KGS_DIR.mkdir(parents=True, exist_ok=True)
    log_sink = logger.add(_LOG_PATH, mode="w", encoding="utf-8")
    try:
        _load_env()
        args = parse_args()
        asyncio.run(run_batch(args.runs, args.input_dir))
    finally:
        logger.remove(log_sink)


if __name__ == "__main__":
    main()
