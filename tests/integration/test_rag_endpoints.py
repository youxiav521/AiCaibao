"""
RAG API 集成测试
测试RAG的删除、索引和查询功能
"""
import json
import pytest
import time
from pathlib import Path
from httpx import AsyncClient, ASGITransport

# 常量定义
TEST_FILE_ID = "1ca24f083e494b55b384d92d2432611dWwTg26VHUK"
TEST_FOLDER_ID = "test_folder_001"
TEST_FILENAME = "燕塘乳业2022年年度报告.pdf"
QUERY_QUESTION = "广东燕塘乳业股份有限公司实际控制人"
EXPECTED_ANSWER_KEYWORDS = ["广东农垦", "广东省农垦集团公司"]

@pytest.fixture(scope="module")
def rag_app(load_env):
    """包含RAG路由的FastAPI应用"""
    from fastapi import FastAPI
    from src.applications.rag.routers.rag_router import router as rag_router

    app = FastAPI()
    app.include_router(rag_router, prefix="/api/rag")
    return app

@pytest.fixture(scope="module")
def test_parsed_data(project_root):
    """加载测试用的PDF解析数据"""
    file_path = project_root / "tests" / "files" / "test_pdf_parsed.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.mark.asyncio
async def test_rag_workflow(rag_app, test_parsed_data):
    """测试完整的RAG工作流：删除 -> 索引 -> 查询"""
    async with AsyncClient(
        transport=ASGITransport(app=rag_app),
        base_url="http://testserver"
    ) as client:
        # 步骤1：调用删除接口，确保测试环境干净
        start_time = time.time()
        delete_response = await client.post(
            "/api/rag/doc/delete",
            json={"fileId": TEST_FILE_ID}
        )
        delete_duration = time.time() - start_time
        print(f"删除接口耗时: {delete_duration:.1f}秒")
        assert delete_response.status_code == 200
        delete_result = delete_response.json()
        assert delete_result["file_id"] == TEST_FILE_ID
        # deleted可以是True或False，取决于之前是否有数据

        # 步骤2：调用索引接口，上传解析后的PDF数据
        start_time = time.time()
        index_response = await client.post(
            "/api/rag/index",
            json={
                "docParseContent": test_parsed_data,
                "fileName": TEST_FILENAME,
                "docFolderId": TEST_FOLDER_ID
            }
        )
        index_duration = time.time() - start_time
        print(f"索引接口耗时: {index_duration:.1f}秒")
        assert index_response.status_code == 200
        index_result = index_response.json()
        assert index_result["success"] == True

        # 等待索引完成（根据实际情况调整等待时间）
        import asyncio
        await asyncio.sleep(2)

        # 步骤3：调用查询接口，验证能正确检索到信息
        start_time = time.time()
        query_response = await client.post(
            "/api/rag/query",
            json={
                "questions": [
                    {
                        "questionId": "test_001",
                        "questionText": QUERY_QUESTION
                    }
                ],
                "topK": 5,
                "folderId": TEST_FOLDER_ID
            }
        )
        query_duration = time.time() - start_time
        print(f"查询接口耗时: {query_duration:.1f}秒")
        assert query_response.status_code == 200
        query_results = query_response.json()

        # 验证查询结果非空
        assert len(query_results) > 0, "查询结果为空，未能检索到相关信息"

        # 验证查询结果包含预期的关键词
        found_expected = False
        for result in query_results:
            # 每个问题的结果包含text字段
            content = result.get("text", "")
            if any(keyword in content for keyword in EXPECTED_ANSWER_KEYWORDS):
                found_expected = True
                break

        assert found_expected, f"查询结果中未找到预期的关键词：{EXPECTED_ANSWER_KEYWORDS}"
        print(f"✓ 成功查询到实际控制人信息：{'、'.join(EXPECTED_ANSWER_KEYWORDS)}")
