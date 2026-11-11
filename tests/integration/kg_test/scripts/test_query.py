"""知识图谱检索测试脚本

基于已构建的知识图谱（通过 test_kg.py 生成），测试检索与问答流程。
前置条件：已运行 test_kg.py，MinIO 中存在 folderId 对应的图谱。

用法:
    .venv/Scripts/python tests/integration/kg_test/scripts/test_query.py
    .venv/Scripts/python tests/integration/kg_test/scripts/test_query.py --folder-id my_folder
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


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


async def test_query(folder_id: str, queries: list[str]):
    from loguru import logger
    import sys
    from src.ade.knowledge_graph.service.query_service import QueryService, RELATION_TYPE_TRIGGERS
    from src.ade.knowledge_graph.service.query_service_v2 import QueryServiceV2
    from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService


    # 先移除默认的 handler，确保我们完全控制
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sink=sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
        level="DEBUG"
    )

    # context 保存目录 - 测试脚本同级目录
    script_dir = Path(__file__).resolve().parent
    context_dir = script_dir / "100queries"
    context_dir.mkdir(exist_ok=True)
    
    # 添加 loguru sink 保存日志到 terminal.txt
    terminal_file = context_dir / "terminal.txt"
    logger.add(
        sink=terminal_file,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=False,
        level="DEBUG"
    )

    print(f"\n{'='*60}")
    print(f"  知识图谱检索测试")
    print(f"{'='*60}\n")

    graph = await GraphStorageService.get(folder_id)
    if graph is None:
        print(f"[ERROR] 知识库 {folder_id} 下不存在图谱，请先运行 test_kg.py 构建图谱")
        sys.exit(1)

    print(f"  folderId  : {folder_id}")
    print(f"  entities  : {len(graph.entities)}")
    print(f"  relations : {len(graph.relations)}")
    print(f"[INFO] Context 保存目录: {context_dir}\n")

    # 测试所有查询（并发，上限 10）
    sem = asyncio.Semaphore(10)

    async def run_one(i: int, query_text: str):
        async with sem:
            result, context = await QueryServiceV2.query(
                folder_id=folder_id,
                query_text=query_text,
                return_context=True,
            )
        return i, query_text, result.answer, context

    results = await asyncio.gather(*[run_one(i, q) for i, q in enumerate(queries, 1)])

    # 按序号排序后写入单个文件
    results.sort(key=lambda x: x[0])
    result_file = context_dir / "all_results.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        for i, query_text, answer, context in results:
            f.write(f"[{i}/{len(queries)}] {query_text}\n")
            f.write(f"Answer: {answer}\n")
            f.write(f"{'-'*60}\n")
            print(f"   [{i}/{len(queries)}] {query_text[:30]}...")

    print(f"\n[INFO] 结果已保存: {result_file}")
        
    print(f"{'='*60}")
    print(f"  ✅ 知识图谱检索测试完成")
    print(f"{'='*60}\n")  
    print(f"[INFO] 日志已保存到: {terminal_file}")


def main():
    parser = argparse.ArgumentParser(description="知识图谱检索测试")
    parser.add_argument("--folder-id", default="TASK-XX", help="知识库ID（默认:TASK-XX）")
    args = parser.parse_args()

    queries_file = Path(__file__).resolve().parent / "queries.txt"
    if queries_file.exists():
        with open(queries_file, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        print(f"[INFO] 从 {queries_file} 加载了 {len(queries)} 条查询")
    else:
        print(f"[WARN] queries.txt 不存在: {queries_file}")
    asyncio.run(test_query(args.folder_id, queries))


if __name__ == "__main__":
    main()
