"""测试 LLM Metrics 功能"""

import asyncio
import pytest

from src.utils.model_enhancements import (
    MetricsContext,
    metrics_context_session,
    MetricsCallbackHandler,
    MetricsResult,
    ModelKeyMetrics,
)


@pytest.mark.asyncio
async def test_metrics_context_basic():
    """测试 MetricsContext 基本功能"""
    async with metrics_context_session(auto_print=False) as ctx:
        assert isinstance(ctx, MetricsContext)
        result = ctx.calculate()
        assert isinstance(result, MetricsResult)
        assert result.total_calls == 0
        assert result.total_errors == 0


@pytest.mark.asyncio
async def test_metrics_callback_handler():
    """测试 MetricsCallbackHandler 初始化"""
    handler = MetricsCallbackHandler(
        model_key="test_model",
        provider="bailian",
    )
    assert handler.model_key == "test_model"
    assert handler.provider == "bailian"


@pytest.mark.asyncio
async def test_metrics_result_to_dict():
    """测试 MetricsResult 转换为 dict"""
    result = MetricsResult(
        total_calls=10,
        total_errors=2,
        total_duration_ms=1000.0,
        total_tokens_in=1000,
        total_tokens_out=500,
        total_queue_wait_ms=500.0,
        total_retry_count=5,
        avg_duration_ms=100.0,
        avg_queue_wait_ms=50.0,
        avg_retry_count=0.5,
        overall_error_rate=0.2,
        by_model_key={
            "model1": ModelKeyMetrics(
                call_count=5,
                error_count=1,
                total_duration_ms=500.0,
                total_tokens_in=500,
                total_tokens_out=250,
                total_queue_wait_ms=250.0,
                total_retry_count=3,
            ),
            "model2": ModelKeyMetrics(
                call_count=5,
                error_count=1,
                total_duration_ms=500.0,
                total_tokens_in=500,
                total_tokens_out=250,
                total_queue_wait_ms=250.0,
                total_retry_count=2,
            ),
        },
    )

    result_dict = result.to_dict()
    assert "summary" in result_dict
    assert "by_model_key" in result_dict
    assert result_dict["summary"]["total_calls"] == 10
    assert result_dict["summary"]["total_errors"] == 2


@pytest.mark.asyncio
async def test_model_key_metrics_calculated_properties():
    """测试 ModelKeyMetrics 的计算属性"""
    metrics = ModelKeyMetrics(
        call_count=10,
        error_count=2,
        total_duration_ms=1000.0,
        total_tokens_in=1000,
        total_tokens_out=500,
        total_queue_wait_ms=300.0,
        total_retry_count=5,
    )

    assert metrics.avg_duration_ms == 100.0
    assert metrics.error_rate == 0.2
    assert metrics.avg_queue_wait_ms == 30.0


@pytest.mark.asyncio
async def test_metrics_pretty_print():
    """测试 MetricsResult 格式化输出"""
    result = MetricsResult(
        total_calls=10,
        total_errors=2,
        total_duration_ms=1000.0,
        total_tokens_in=1000,
        total_tokens_out=500,
        total_queue_wait_ms=500.0,
        total_retry_count=5,
        avg_duration_ms=100.0,
        avg_queue_wait_ms=50.0,
        avg_retry_count=0.5,
        overall_error_rate=0.2,
        by_model_key={},
    )

    output = result.pretty_print()
    assert isinstance(output, str)
    assert "LLM 调用统计报告" in output
    assert "总调用次数: 10" in output
    assert "错误次数: 2" in output


if __name__ == "__main__":
    asyncio.run(test_metrics_context_basic())
    asyncio.run(test_metrics_callback_handler())
    asyncio.run(test_metrics_result_to_dict())
    asyncio.run(test_model_key_metrics_calculated_properties())
    asyncio.run(test_metrics_pretty_print())
    print("所有测试通过!")
