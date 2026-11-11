"""知识图谱 E2E 测试脚本

测试链路: FastAPI Router → TaskManager → ARQ Worker → MinIO
验证完整的异步任务调度链路，涵盖 Router 入参/出参、Worker 消费、状态流转、结果存储。

前提:
  1. Redis 运行于 localhost:6379（或环境变量 REDIS_HOST/REDIS_PORT 配置的地址）
  2. MinIO 运行于 localhost:9000（或环境变量配置的地址）
  3. Uvicorn 已启动: uv run uvicorn src.main:app --port 6300
  4. Worker 已启动: uv run arq src.worker.WorkerSettings

用法:
    uv run python tests/integration/kg_test/scripts/test_kg_e2e.py

或带参数指定 API 地址:
    uv run python tests/integration/kg_test/scripts/test_kg_e2e.py --base-url http://localhost:6300
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

# ── 路径解析（必须在 import 业务模块前执行） ────────────
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# 加载 .env（在 import config.py 前设好环境变量）
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

# ── 测试数据路径 ──────────────────────────────────────
_DATA_DIR = _PROJECT_DIR / "tests" / "integration" / "kg_test" / "converted"


def discover_test_docs(max_docs: int = 2) -> list[dict]:
    if not _DATA_DIR.exists():
        print(f"[ERROR] 测试数据目录不存在: {_DATA_DIR}")
        sys.exit(1)

    all_dirs = sorted(d for d in _DATA_DIR.iterdir() if d.is_dir())
    selected = all_dirs[:max_docs]

    files = []
    for doc_dir in selected:
        json_path = doc_dir / "converted.json"
        if not json_path.exists():
            continue

        with open(json_path, encoding="utf-8") as f:
            items = json.load(f)

        elements = []
        for item in items:
            md = item.get("md", "")
            if not md.strip():
                continue
            elem_type = item.get("type", "text")
            rows = item.get("rows") if elem_type == "table" else None
            elements.append({
                "element_id": item["element_id"],
                "type": elem_type,
                "md": md,
                "rows": rows,
            })

        doc_id = f"e2e_{doc_dir.name}"
        files.append({"id": doc_id, "name": doc_dir.name, "elements": elements})
        print(f"  [文档] {doc_dir.name:30s} | elements={len(elements):3d} | chars={sum(len(e['md']) for e in elements)}")

    if not files:
        print(f"[ERROR] {_DATA_DIR} 下未找到 converted.json")
        sys.exit(1)

    return files


async def create_job(base_url: str, files: list[dict]) -> str:
    """调用 POST /api/kg/jobs 创建知识图谱生成任务"""
    print(f"\n{'─' * 60}")
    print("  [步骤 1/4] 创建知识图谱生成任务")
    print(f"{'─' * 60}")

    payload = {
        "folderName": "E2E测试知识库",
        "folderId": f"e2e_{uuid.uuid4().hex[:8]}",
        "files": files,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{base_url}/api/kg/jobs", json=payload)

    assert resp.status_code == 200, f"创建任务失败: HTTP {resp.status_code} {resp.text}"
    data = resp.json()
    assert data["code"] == "0", f"创建任务失败: code={data['code']} message={data.get('message')}"

    job_id = data["data"]["jobId"]
    print(f"  任务已创建 | job_id={job_id}")
    return job_id


async def poll_until_done(base_url: str, job_id: str, timeout: int = 600) -> dict:
    """轮询 GET /api/kg/jobs/{job_id} 直到任务完成或超时

    Returns:
        dict: 任务详情数据（包含 graphId、summary 等）
    """
    print(f"\n{'─' * 60}")
    print("  [步骤 2/4] 轮询等待任务完成")
    print(f"{'─' * 60}")
    print(f"  超时设置: {timeout}s — LLM 提取可能需要较长时间，请耐心等待")

    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=10) as client:
        while time.time() < deadline:
            resp = await client.get(f"{base_url}/api/kg/jobs/{job_id}")
            assert resp.status_code == 200, (
                f"查询任务状态失败: HTTP {resp.status_code} {resp.text[:500]}"
            )

            data = resp.json()
            assert data["code"] == "0", f"查询任务状态失败: {data}"

            detail = data["data"]
            status = detail["status"]
            elapsed = int(time.time() - (deadline - timeout))
            print(f"  ⏳ 等待中... (已过 {elapsed}s) 状态: {status}")

            if status == "success":
                print(f"\n  ✅ 任务成功完成 | 耗时约 {elapsed}s")
                return detail

            if status == "failed":
                reason = detail.get("failureReason", "未知错误")
                raise RuntimeError(f"任务失败: {reason}")

            await asyncio.sleep(3)

    raise TimeoutError(f"任务 {job_id} 在 {timeout}s 内未完成")


async def verify_graph(base_url: str, folder_id: str) -> None:
    """调用 GET /api/kg/{folderId} 获取并验证图谱数据"""
    print(f"\n{'─' * 60}")
    print("  [步骤 3/4] 获取并验证图谱数据")
    print(f"{'─' * 60}")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{base_url}/api/kg/{folder_id}")

    assert resp.status_code == 200, f"获取图谱失败: HTTP {resp.status_code}"
    data = resp.json()
    assert data["code"] == "0", f"获取图谱失败: {data}"

    graph_data = data["data"]
    entity_count = graph_data["totalEntities"]
    relation_count = graph_data["totalRelations"]
    entities = graph_data.get("entities", [])
    relations = graph_data.get("relations", [])

    print(f"  图谱验证通过:")
    print(f"    folder_id  : {folder_id}")
    print(f"    entities   : {entity_count}")
    print(f"    relations  : {relation_count}")

    assert entity_count > 0, "图谱中没有任何实体"
    assert relation_count > 0, "图谱中没有任何关系"

    # 打印实体摘要
    if entities:
        print(f"\n    ── 实体列表（前 10 个）──")
        for i, e in enumerate(entities[:10], 1):
            print(f"    {i:3d}. {e['name']:25s} type={e['type']}")
        if len(entities) > 10:
            print(f"    ... 共 {entity_count} 个实体")

    # 打印关系摘要
    if relations:
        print(f"\n    ── 关系列表（前 10 个）──")
        for i, r in enumerate(relations[:10], 1):
            print(f"    {i:3d}. {r['from_entity']:20s} ─[{r['type']}]─> {r['to_entity']}")
        if len(relations) > 10:
            print(f"    ... 共 {relation_count} 个关系")


async def cleanup(job_id: str, folder_id: str, base_url: str) -> None:
    """清理测试产生的数据：MinIO 图谱文件 + Redis 任务记录"""
    print(f"\n{'─' * 60}")
    print("  [步骤 4/4] 清理测试数据")
    print(f"{'─' * 60}")

    # ── 4a. 删除 MinIO 中的图谱文件 ──────────────────
    try:
        from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService

        existed = await GraphStorageService.exists(folder_id)
        if existed:
            await GraphStorageService.delete(folder_id)
            still_exists = await GraphStorageService.exists(folder_id)
            assert not still_exists, "MinIO 图谱删除失败"
            print(f"  ✅ MinIO 图谱已删除: {folder_id}")
        else:
            print(f"  ℹ️  MinIO 中不存在此图谱，无需清理")
    except Exception as e:
        print(f"  ⚠️  MinIO 清理异常: {e}")

    # ── 4b. 删除 Redis 中的任务记录 ──────────────────
    try:
        from src.base.task_manager import TaskManager

        record = await TaskManager.get(job_id)
        if record is not None:
            await TaskManager.delete(job_id, task_type="KnowledgeGraph", remove_arq_keys=True)
            still_exists = await TaskManager.get(job_id)
            assert still_exists is None, "Redis 任务记录删除失败"
            print(f"  ✅ Redis 任务记录已删除: {job_id}")
        else:
            print(f"  ℹ️  Redis 中不存在此任务记录，无需清理")
    except Exception as e:
        print(f"  ⚠️  Redis 清理异常: {e}")


async def run_e2e(base_url: str = "http://localhost:6300", max_docs: int = 2) -> None:
    """E2E 测试主流程"""
    graph_id = None
    job_id = None
    folder_id = None

    try:
        print(f"{'=' * 60}")
        print(f"  知识图谱 E2E 测试")
        print(f"  API 地址: {base_url}")
        print(f"{'=' * 60}")

        # ── 1. 发现测试数据 ────────────────────────────
        print(f"\n{'─' * 60}")
        print("  [准备] 扫描测试数据...")
        print(f"{'─' * 60}")
        files = discover_test_docs(max_docs=max_docs)
        print(f"  共使用 {len(files)} 份文档")

        # ── 2. 创建任务 ────────────────────────────────
        job_id = await create_job(base_url, files)

        # ── 3. 轮询等待 ────────────────────────────────
        detail = await poll_until_done(base_url, job_id)
        returned_graph_id = detail.get("graphId")
        assert returned_graph_id == job_id, f"graph_id 不一致: 预期 {job_id}, 实际 {returned_graph_id}"
        graph_id = returned_graph_id
        metadata = detail.get("metadata") or {}
        folder_id = metadata.get("folderId")
        assert folder_id, f"任务详情未返回 folderId: {detail}"
        summary = detail.get("summary", {})
        print(f"  graph_id  : {returned_graph_id}")
        print(f"  folder_id : {folder_id}")
        print(f"  entities  : {summary.get('entities', 'N/A')}")
        print(f"  relations : {summary.get('relations', 'N/A')}")

        # ── 4. 验证图谱 ────────────────────────────────
        await verify_graph(base_url, folder_id)

        # ── 全部通过 ───────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  ✅ 知识图谱 E2E 测试全部通过")
        print(f"{'=' * 60}\n")

    except Exception:
        print(f"\n{'=' * 60}")
        print(f"  ❌ 知识图谱 E2E 测试失败")
        print(f"{'=' * 60}\n")
        raise

    finally:
        # ── 5. 清理（无论成功失败都执行） ────────────────
        if job_id or folder_id:
            await cleanup(job_id or "", folder_id or "", base_url)


def main():
    parser = argparse.ArgumentParser(description="知识图谱 E2E 测试")
    parser.add_argument(
        "--base-url",
        default="http://localhost:6300",
        help="FastAPI 服务地址（默认 http://localhost:6300）",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=2,
        help="E2E 测试使用的文档数量（默认 2，建议不超过 3 以节省时间）",
    )
    args = parser.parse_args()

    asyncio.run(run_e2e(base_url=args.base_url, max_docs=args.max_docs))


if __name__ == "__main__":
    main()
