from pathlib import Path

from src.ade.knowledge_graph.evaluate.report import render_html, render_markdown
from src.ade.knowledge_graph.evaluate.schema import (
    AttributeEvalResult,
    EvaluationGraph,
    EvaluationResult,
    MetricResult,
)


def test_render_html_keeps_details_closed_by_default() -> None:
    report = render_html(
        "# 报告\n\n<details>\n<summary>明细</summary>\n\n内容\n\n</details>",
        title="KG Query V3 手工评测报告",
    )

    assert report.startswith("<!doctype html>")
    assert "<title>KG Query V3 手工评测报告</title>" in report
    assert "<h1>报告</h1>" in report
    assert "<details>" in report
    assert "<details open" not in report


def test_render_markdown_collapses_only_nonempty_regular_items() -> None:
    entity_metrics = MetricResult(
        tp=1,
        fp=0,
        fn=0,
        correct={("测试企业", "ENTERPRISE")},
    )
    relation_metrics = MetricResult(
        tp=1,
        fp=0,
        fn=0,
        correct={("测试企业", "关联关系", "关联企业")},
    )
    key_relation_metrics = MetricResult(
        tp=1,
        fp=0,
        fn=0,
        correct={("测试企业", "主借款人", "测试项目")},
    )
    result = EvaluationResult(
        gold_path=Path("gold.txt"),
        target_path=Path("target.txt"),
        gold=EvaluationGraph(),
        target=EvaluationGraph(),
        entity_all=entity_metrics,
        entity_non_isolated=entity_metrics,
        relation=relation_metrics,
        attribute=AttributeEvalResult(
            metrics=MetricResult(tp=0, fp=0, fn=0),
            value_wrong=[
                (("测试企业", "ENTERPRISE", "企业证件号"), {"gold"}, {"target"})
            ],
        ),
        key_relation_total=key_relation_metrics,
        key_relations=[],
    )

    markdown = render_markdown(result)
    regular_section, remainder = markdown.split("## 7. 重点关系评估", maxsplit=1)
    key_relation_section = remainder.split("## 8. 解析 Warning", maxsplit=1)[0]

    assert "<summary><strong>正确条目 (1)</strong></summary>" in regular_section
    assert "<summary><strong>值错误属性 (1)</strong></summary>" in regular_section
    assert "### 缺失条目（Gold 有、Target 无） (0)\n\n无" in regular_section
    assert "<details>" not in key_relation_section
    assert "### 正确项 (1)\n\n- (测试企业, 主借款人, 测试项目)" in key_relation_section
