"""文档结构批量召回集成测试。

通过内存 ASGI transport 调用 /api/rag/structure/recall-batch，
验证结构选择准确性。

用法:
    .venv/bin/python -m pytest tests/integration/test_structure_recall.py -v
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import pytest
from httpx import AsyncClient, ASGITransport

from fastapi import FastAPI

DEFAULT_FILE_IDS = [
    "219e9fddd5a740718c3e8cf3d7212dafJdG6qnU2jM",
    "5b6d2a2fde3a45bda39dcead5349a2c0jGaE1sv87j",
    "ae993b4446d546a99ca6824d7acc16eepomBg7sh09",
    "ea76e9c1b9e249f89259e7006a42617c0VAvJDykiE",
]

pytestmark = pytest.mark.asyncio(loop_scope="session")


def make_question(entry: dict, fmt: str) -> str:
    if fmt == "file_name_title":
        return f"{entry['file_name']} {entry['title']}"
    return entry["title"]


@pytest.fixture(scope="module")
def structure_recall_app(load_env):
    app = FastAPI()
    from src.applications.rag.routers.structure_recall_router import router
    app.include_router(router, prefix="/api/rag")
    return app


@pytest.fixture(scope="module")
def flat_titles(project_root):
    data_path = project_root / "storage" / "flattened_titles.json"
    if not data_path.exists():
        pytest.skip(f"数据文件不存在: {data_path}")
    titles = json.loads(data_path.read_text(encoding="utf-8"))
    limit = int(os.environ.get("STRUCTURE_RECALL_TEST_LIMIT", "0")) or 0
    if limit > 0:
        titles = titles[:limit]
    return titles


async def _call_batch_api(
    client: AsyncClient,
    questions: List[dict],
    file_ids: List[str],
    max_tokens: int = 200000,
    max_structure_num: int = 20,
) -> List[dict]:
    payload = {
        "questions": [
            {"questionId": q["id"], "questionText": q["text"]}
            for q in questions
        ],
        "fileIds": file_ids,
        "maxTokens": max_tokens,
        "maxStructureNum": max_structure_num,
    }
    resp = await client.post(
        "/api/rag/structure/recall-batch",
        json=payload,
        timeout=300,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"
    data = resp.json()
    assert isinstance(data, list), f"响应格式异常，期望数组，实际: {type(data).__name__}"
    return data


async def _evaluate_and_assert(
    client: AsyncClient,
    flat_titles: List[dict],
    file_ids: List[str],
    question_fmt: str,
    min_accuracy: float,
) -> dict:
    total = len(flat_titles)
    questions = [
        {"id": f"test_{entry['index']}", "text": make_question(entry, question_fmt)}
        for entry in flat_titles
    ]

    t_start = time.monotonic()
    sections = await _call_batch_api(client, questions, file_ids)
    t_elapsed = time.monotonic() - t_start
    print(f"\n批量请求耗时: {t_elapsed:.1f}s | 返回 section 总数: {len(sections)}")

    sections_by_question: Dict[str, List[dict]] = {}
    for s in sections:
        qid = s.get("questionId", "")
        sections_by_question.setdefault(qid, []).append(s)

    correct = 0
    errors = 0
    wrong_details: List[dict] = []

    for i, entry in enumerate(flat_titles):
        expected_title = entry["title"]
        expected_file = entry["file_name"]
        question_id = questions[i]["id"]
        question_sections = sections_by_question.get(question_id, [])

        if not question_sections:
            wrong_details.append({
                "index": entry["index"],
                "question": make_question(entry, question_fmt),
                "file_name": expected_file,
                "title": expected_title,
                "total_sections": 0,
                "matched_file_names": [],
            })
            continue

        matched = [
            s for s in question_sections
            if s.get("fileName") == expected_file
            and (s.get("text") or "").find(expected_title) != -1
        ]
        if matched:
            correct += 1
        else:
            wrong_details.append({
                "index": entry["index"],
                "question": make_question(entry, question_fmt),
                "file_name": expected_file,
                "title": expected_title,
                "total_sections": len(question_sections),
                "matched_file_names": list({s.get("fileName") for s in question_sections}),
            })

    total_cases = total - errors
    accuracy = correct / total_cases if total_cases > 0 else 0.0

    metrics = {
        "total_titles": total,
        "total_cases": total_cases,
        "errors": errors,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "elapsed": round(t_elapsed, 1),
    }

    print(f"\n结构选择准确性测试结果（batch API）")
    print(f"  测试数量:      {metrics['total_titles']}")
    print(f"  正确/失败:     {metrics['correct']} / {metrics['errors']}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  耗时:          {metrics['elapsed']:.1f}s")

    if wrong_details:
        print(f"\n未正确命中案例 ({len(wrong_details)}条):")
        for w in wrong_details[:10]:
            print(f"  - [{w['file_name']}] {w['title'][:60]}")
            print(f"    返回文件: {w['matched_file_names']}")
        if len(wrong_details) > 10:
            print(f"  ... 还有 {len(wrong_details) - 10} 条")

    assert accuracy >= min_accuracy, (
        f"Accuracy {accuracy:.4f} 低于最低要求 {min_accuracy}"
    )

    return metrics


@pytest.mark.integration
async def test_structure_recall_accuracy(
    structure_recall_app,
    flat_titles,
):
    """测试文档结构批量召回准确率，验证至少达到最低阈值。"""
    if not flat_titles:
        pytest.skip("无测试数据")

    async with AsyncClient(
        transport=ASGITransport(app=structure_recall_app),
        base_url="http://testserver",
    ) as client:
        await _evaluate_and_assert(
            client=client,
            flat_titles=flat_titles,
            file_ids=DEFAULT_FILE_IDS,
            question_fmt="file_name_title",
            min_accuracy=0.3,
        )


@pytest.mark.integration
async def test_structure_recall_question_format_title_only(
    structure_recall_app,
    flat_titles,
):
    """使用仅标题格式的问题进行召回测试。

    仅给标题不给文件名时，LLM 无法区分多个年报中同名的章节
    （如两个年报都有"董事会决议……"），fileName 匹配率天然偏低。
    此用例仅验证通路正常，不做 accuracy 硬断言。
    """
    if not flat_titles:
        pytest.skip("无测试数据")

    async with AsyncClient(
        transport=ASGITransport(app=structure_recall_app),
        base_url="http://testserver",
    ) as client:
        metrics = await _evaluate_and_assert(
            client=client,
            flat_titles=flat_titles,
            file_ids=DEFAULT_FILE_IDS,
            question_fmt="title",
            min_accuracy=0.0,
        )
        print(f"title-only accuracy: {metrics['accuracy']} (预期偏低，跨多年报同名章节无法区分)")
