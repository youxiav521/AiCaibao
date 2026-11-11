"""知识图谱构建测试脚本

从 tests/integration/kg_test/converted/ 读取所有 converted.json，调用完整的图谱构建流程，
最终验证图谱已保存到 MinIO 并打印统计信息。

用法:
    .venv/Scripts/python tests/integration/kg_test/scripts/test_kg.py
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path


# ── 路径解析（必须在 import 业务模块前执行） ────────────
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# 加载 .env
def _load_env():
    env_local = _PROJECT_DIR / ".env.local"
    env_file = _PROJECT_DIR / ".env"
    env_path = env_local if env_local.exists() else env_file
    if not env_path.exists():
        print(f"[WARN] 未找到环境配置文件，预期路径: {env_local} 或 {env_file}")
        return False

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)

    print(f"[INFO] 已加载环境配置: {env_path}")
    return True

_load_env()

# ── 数据路径 ──────────────────────────────────────────
_CONVERTED_DIR = _PROJECT_DIR / "tests" / "integration" / "kg_test" / "converted"


def discover_files() -> list[dict]:
    """扫描 converted/ 下所有子目录的 converted.json，返回 FileInput 的 dict 列表。"""
    from src.ade.knowledge_graph.utils.preprocess_file import converted_json_to_fileinput

    if not _CONVERTED_DIR.exists():
        print(f"[ERROR] 测试数据目录不存在: {_CONVERTED_DIR}")
        sys.exit(1)

    json_paths = sorted(_CONVERTED_DIR.rglob("converted.json"))
    if not json_paths:
        print(f"[ERROR] converted/ 下未找到任何 converted.json 文件")
        sys.exit(1)

    files = []
    for jp in json_paths:
        fi = converted_json_to_fileinput(jp)
        files.append(fi.model_dump())

    return files


async def test_kg():
    """主测试流程"""
    print(f"\n{'='*60}")
    print(f"  知识图谱构建测试")
    print(f"{'='*60}\n")

    # ── 1. 发现测试文件 ────────────────────────────────────
    print("[1/4] 扫描测试数据...")
    files_data = discover_files()
    print(f"      共发现 {len(files_data)} 份文档\n")

    # ── 2. 创建 FileInput 并调用 generate ──────────────────
    print("[2/4] 构建图谱...")
    from src.ade.knowledge_graph.schema.kg_schema import FileInput
    from src.ade.knowledge_graph.service.knowledge_graph_service import KnowledgeGraphService

    files = [FileInput(**f) for f in files_data]
    folder_id = "TASK-XX"
    folder_name = "测试知识库"
    graph_id = f"kg_test_{uuid.uuid4().hex[:12]}"

    graph = await KnowledgeGraphService.generate(
        files=files,
        folder_id=folder_id,
        folder_name=folder_name,
        graph_id=graph_id,
    )

    print(f"      图谱构建完成")
    print(f"      graph_id : {graph.id}")
    print(f"      entities : {len(graph.entities)}")
    print(f"      relations: {len(graph.relations)}")

    # 打印实体摘要
    if graph.entities:
        print(f"\n      ── 实体列表 ──")
        for i, e in enumerate(graph.entities[:20], 1):
            print(f"      {i:3d}. {e.name:25s} type={e.type}")
        if len(graph.entities) > 20:
            print(f"      ... 共 {len(graph.entities)} 个实体")

    # 打印关系摘要
    if graph.relations:
        print(f"\n      ── 关系列表 ──")
        for i, r in enumerate(graph.relations[:20], 1):
            print(f"      {i:3d}. {r.from_entity:20s} ─[{r.type}]─> {r.to_entity}")
        if len(graph.relations) > 20:
            print(f"      ... 共 {len(graph.relations)} 个关系")

    # ── 3. 验证 MinIO 持久化 ────────────────────────────────
    print(f"\n[3/4] 验证 MinIO 持久化...")
    from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService

    saved = await GraphStorageService.exists(folder_id)
    print(f"      图谱文件存在 : {saved}")
    assert saved, f"MinIO 中未找到图谱: folder_id={folder_id}"

    restored = await GraphStorageService.get(folder_id)
    assert restored is not None, "从 MinIO 读取图谱失败"
    assert len(restored.entities) == len(graph.entities), "实体数量不一致"
    assert len(restored.relations) == len(graph.relations), "关系数量不一致"
    print(f"      读回验证通过: entities={len(restored.entities)} relations={len(restored.relations)}")

    # ── 4. 清理资源 ─────────────────────────────────────────────
    print(f"\n[4/4] 清理资源...")
    from src.utils.model_factory import ModelFactory
    # 显式关闭所有 Provider 的 httpx 客户端（在事件循环关闭前）
    for provider in ModelFactory._providers.values():
        provider.close()
    print(f"      httpx 客户端已清理")

    print(f"\n{'='*60}")
    print(f"  ✅ 知识图谱构建测试通过")
    print(f"{'='*60}\n")
    return graph


def main():
    asyncio.run(test_kg())


if __name__ == "__main__":
    main()
