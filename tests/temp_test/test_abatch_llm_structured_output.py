import asyncio
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from langchain_core.prompts import ChatPromptTemplate

from src.agents.content_generate_agent.utils import _with_structured_output_compat
from src.utils.model_factory import ModelFactory


class TestResponse(BaseModel):
    answer: str = Field(..., description="模型回答")
    confidence: float = Field(..., description="0到1之间的置信度")


def build_chain(model_id: str, model=None):
    if model is None:
        model = ModelFactory.get_model(key=model_id, rate_limited=False)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Return a valid json object only."),
        ("human", "{prompt}"),
    ])
    return prompt_template | _with_structured_output_compat(model, model_id, TestResponse)


async def test_single_invoke(model_id: str):
    logger.info("=== 测试1: 单次 ainvoke ===")
    chain = build_chain(model_id)
    t0 = time.monotonic()
    result = await chain.ainvoke({"prompt": "请用一句话回答：1+1等于多少？并给出0到1之间的置信度。"})
    elapsed = time.monotonic() - t0
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    logger.info("耗时: {:.2f}s", elapsed)


async def test_abatch(model_id: str):
    logger.info("=== 测试2: abatch 并发5个请求 ===")
    chain = build_chain(model_id)
    prompts = [
        "请用一句话回答：1+1等于多少？并给出0到1之间的置信度。",
        "请用一句话回答：中国的首都是哪里？并给出0到1之间的置信度。",
        "请用一句话回答：水的化学式是什么？并给出0到1之间的置信度。",
        "请用一句话回答：地球离太阳大约多远？并给出0到1之间的置信度。",
        "请用一句话回答：光速大约是多少？并给出0到1之间的置信度。",
    ]
    inputs = [{"prompt": p} for p in prompts]
    t0 = time.monotonic()
    results = await chain.abatch(inputs)
    elapsed = time.monotonic() - t0
    for i, r in enumerate(results, 1):
        print(f"[{i}] {json.dumps(r.model_dump(), ensure_ascii=False)}")
    logger.info("abatch 5请求总耗时: {:.2f}s (平均: {:.2f}s/个)", elapsed, elapsed / len(results))


async def test_timeout(model_id: str):
    logger.info("=== 测试3: 超时控制 (10分钟) ===")
    chain = build_chain(model_id)
    try:
        t0 = time.monotonic()
        result = await asyncio.wait_for(
            chain.ainvoke({"prompt": "请用一句话回答：太阳系有几大行星？并给出0到1之间的置信度。"}),
            timeout=600,
        )
        elapsed = time.monotonic() - t0
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        logger.info("耗时: {:.2f}s", elapsed)
    except asyncio.TimeoutError:
        logger.error("LLM 调用超时 (600s)")


async def main():
    model_id = os.getenv("MODEL_ID", "parse_template")
    logger.info("start, model_id={}", model_id)

    await test_single_invoke(model_id)
    await test_abatch(model_id)
    await test_timeout(model_id)

    logger.info("all tests done")


if __name__ == "__main__":
    asyncio.run(main())
