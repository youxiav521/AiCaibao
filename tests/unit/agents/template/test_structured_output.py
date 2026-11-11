"""测试结构化输出能否正确返回 QuestionAnalysis 格式的 JSON。
默认使用 model_qwen37_max，需先指定 config 路径。

用法:
  .venv/bin/python -m tests.unit.agents.template.test_structured_output --config docker/local/llm.json
  .venv/bin/python -m tests.unit.agents.template.test_structured_output --config docker/local/llm.json --live
"""
import json
import sys
from pathlib import Path

import src.utils.model_factory as mf

# ── 解析命令行参数 ──────────────────────────────────────────────────────
config_path = None
live_mode = False
model_id = "model_qwen37_max"

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--live":
        live_mode = True
    elif args[i] == "--config" and i + 1 < len(args):
        config_path = args[i + 1]
        i += 1
    elif args[i] == "--model" and i + 1 < len(args):
        model_id = args[i + 1]
        i += 1
    i += 1

# ── 覆盖配置路径（在 ModelFactory 初始化之前）──────────────────────────
if config_path:
    resolved = str(Path.cwd() / config_path)
    mf.Config._config_path = resolved
    print(f"[config] 使用配置: {resolved}")
else:
    print(f"[config] 使用默认配置: config/llm.json")

from src.utils.model_factory import ModelFactory
from src.agents.template.llm_utils import QuestionAnalysis


def test_resolve_model_path():
    """测试各 model_id 的解析路径。"""
    cases = [
        (model_id,               f"指定测试模型"),
        ("model_bailian_80b",    "默认（不在 custom_models，走 default_model）"),
        (None,                   "默认模型"),
    ]
    for mid, desc in cases:
        name = ModelFactory.resolved_model_name_for_key(mid)
        is_ds = ModelFactory.is_deepseek_model_for_key(mid)
        path = "json_mode" if is_ds else "json_schema"
        print(f"[{path:>12}] model_id={str(mid):30s} → {name:30s}  ({desc})")

    print()
    print(f"→ llm_utils.py 调用链（使用 {model_id}）：")
    name = ModelFactory.resolved_model_name_for_key(model_id)
    is_ds = ModelFactory.is_deepseek_model_for_key(model_id)
    path = "json_mode" if is_ds else "json_schema"
    print(f"   解析为 '{name}' → 走 {path}")
    if not is_ds:
        print(f"   发送 response_format={{'type': 'json_schema', ...}}")
        print(f"   ✅ vLLM guided decoding 保证纯 JSON 输出")
    print()


def test_question_analysis_schema():
    schema = QuestionAnalysis.model_json_schema()
    print("QuestionAnalysis JSON Schema:")
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    print()


def test_live_api_call(mid: str):
    import asyncio
    from src.agents.content_generate_agent.utils import ainvoke_llm_with_structured_output

    name = ModelFactory.resolved_model_name_for_key(mid)
    is_ds = ModelFactory.is_deepseek_model_for_key(mid)
    path = "json_schema" if not is_ds else "json_mode"

    print(f"  model_id:  {mid}")
    print(f"  解析为:    {name}")
    print(f"  走路径:    {path}")
    print()

    test_prompt = "分析这段文本：2024年第一季度营收1000万元，同比增长20%，净利润率15%。"

    async def _run():
        return await ainvoke_llm_with_structured_output(
            prompt=test_prompt,
            model_id=mid,
            response_type=QuestionAnalysis,
        )

    result = asyncio.run(_run())
    print(f"[OK] 返回类型: {type(result).__name__}")
    print(f"[OK] 返回内容:")
    print(result.model_dump_json(indent=2, ensure_ascii=False))

    assert isinstance(result, QuestionAnalysis), f"❌ 期望 QuestionAnalysis，实际 {type(result)}"
    assert result.type in ("text", "table", "financial_table"), f"❌ type 不合法: {result.type}"
    assert result.description, "❌ description 为空"
    assert result.prompt, "❌ prompt 为空"
    print()
    print("✅ 全部校验通过：模型返回了合法的 QuestionAnalysis 对象")


def inspect_config():
    print(f"default_model:       {ModelFactory.resolved_model_name_for_key(None)}")
    custom_names = ModelFactory._custom_models
    if custom_names:
        print(f"custom_models keys:  {list(custom_names.keys())}")
    else:
        print("custom_models:      <空>")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("当前 LLM 配置概览")
    print("=" * 70)
    inspect_config()

    print("=" * 70)
    print("1. 模型解析路径检查")
    print("=" * 70)
    test_resolve_model_path()

    print("=" * 70)
    print("2. QuestionAnalysis Schema")
    print("=" * 70)
    test_question_analysis_schema()

    if live_mode:
        print("=" * 70)
        print(f"3. 真实 API 调用测试 (model_id={model_id})")
        print("=" * 70)
        test_live_api_call(model_id)
    else:
        print("=" * 70)
        print("3. 真实 API 调用（跳过）")
        print("=" * 70)
        print()
        print("添加 --live 执行真实调用：")
        print(f"  .venv/bin/python -m tests.unit.agents.template.test_structured_output --config docker/local/llm.json --live")
