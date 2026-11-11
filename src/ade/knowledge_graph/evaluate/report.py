import html
from pathlib import Path
from typing import Callable

import markdown

from src.ade.knowledge_graph.evaluate.metrics import entity_key, relation_triples
from src.ade.knowledge_graph.evaluate.schema import (
    KEY_RELATION_TYPES,
    AttributeAtom,
    AttributeKey,
    ConfidenceDetectionResult,
    ConfidenceEvaluationResult,
    EntityKey,
    EvaluationGraph,
    EvaluationResult,
    MetricResult,
    RelationAttributeAtom,
    RelationAttributeKey,
    RelationTriple,
)


def render_html(
    markdown_text: str,
    *,
    title: str = "知识图谱评估报告",
) -> str:
    body = markdown.markdown(markdown_text, extensions=["tables", "md_in_html"])
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
body {{
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 24px;
  color: #1f2328;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.6;
}}
table {{ width: 100%; border-collapse: collapse; display: block; overflow-x: auto; }}
th, td {{ padding: 6px 12px; border: 1px solid #d0d7de; text-align: left; }}
th {{ background: #f6f8fa; }}
details {{ margin: 16px 0; }}
summary {{ cursor: pointer; }}
code {{ padding: 2px 5px; border-radius: 4px; background: #eff1f3; }}
@media (max-width: 640px) {{ body {{ padding: 20px 12px; }} }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_markdown(
    result: EvaluationResult,
    reference_label: str = "Gold",
    target_label: str = "Target",
    *,
    silver_path: Path | None = None,
    silver: EvaluationGraph | None = None,
) -> str:
    silver_entities = {entity_key(entity) for entity in silver.entities} if silver else set()
    silver_relations = relation_triples(silver.relations) if silver else set()
    lines: list[str] = [
        "# 知识图谱评估报告",
        "",
        "## 1. 输入文件",
        "",
        f"- {reference_label}: `{result.gold_path}`",
        f"- {target_label}: `{result.target_path}`",
        *([f"- Silver: `{silver_path}`"] if silver_path else []),
        f"- {reference_label} 解析: 实体 {len(result.gold.entities)} 个, 关系 {len(result.gold.relations)} 条",
        f"- {target_label} 解析: 实体 {len(result.target.entities)} 个, 关系 {len(result.target.relations)} 条",
        *(
            [f"- Silver 解析: 实体 {len(silver.entities)} 个, 关系 {len(silver.relations)} 条"]
            if silver
            else []
        ),
        "",
        "## 2. 总览",
        "",
        "| 评估项 | TP | FP | FN | Accuracy | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    overview_rows: list[tuple[str, MetricResult]] = [
        ("实体", result.entity_all),
        ("关系", result.relation),
        ("重点关系（合计）", result.key_relation_total),
    ]
    if result.attribute is not None:
        overview_rows.append(("实体属性", result.attribute.metrics))
    else:
        lines.append("| 实体属性 | - | - | - | 已跳过 | 已跳过 | 已跳过 | 已跳过 |")
    if result.relation_attribute is not None:
        overview_rows.append(("关系属性", result.relation_attribute.metrics))
    else:
        lines.append("| 关系属性 | - | - | - | 已跳过 | 已跳过 | 已跳过 | 已跳过 |")

    for label, metrics in overview_rows:
        lines.append(
            f"| {label} | {metrics.tp} | {metrics.fp} | {metrics.fn} | "
            f"{metrics.accuracy:.4f} | {metrics.precision:.4f} | "
            f"{metrics.recall:.4f} | {metrics.f1:.4f} |"
        )
    lines.append("")

    for section_title, metrics, formatter, ocr_items in [
        ("## 3. 实体抽取评估", result.entity_all, _format_entity, silver_entities),
        ("## 4. 关系抽取评估", result.relation, _format_relation, silver_relations),
    ]:
        lines.append(section_title)
        lines.append("")
        lines.extend(_format_metric_table(metrics))
        lines.append("")
        lines.extend(
            _render_item_list(
                "正确条目",
                metrics.correct,
                formatter,
                result.options.list_limit,
                collapsible=True,
            )
        )
        lines.extend(
            _render_item_list(
                f"缺失条目（{reference_label} 有、{target_label} 无）",
                metrics.missing,
                formatter,
                result.options.list_limit,
                collapsible=True,
            )
        )
        lines.extend(
            _render_item_list(
                f"多余条目（{target_label} 有、{reference_label} 无）",
                metrics.extra,
                formatter,
                result.options.list_limit,
                collapsible=True,
                ocr_items=ocr_items,
            )
        )

    lines.extend(_render_attribute_section(result))
    lines.extend(_render_key_relation_section(result, silver_relations))
    lines.extend(
        _render_warnings(
            result,
            reference_label,
            target_label,
            silver_path=silver_path,
            silver=silver,
        )
    )
    return "\n".join(lines)


def render_confidence_markdown(
    result: EvaluationResult,
    *,
    silver_path: Path | None = None,
    silver: EvaluationGraph | None = None,
) -> str:
    silver_entities = {entity_key(entity) for entity in silver.entities} if silver else set()
    silver_relations = relation_triples(silver.relations) if silver else set()
    lines = [
        "# 知识图谱置信度评估报告",
        "",
        "## 1. 输入与统计口径",
        "",
        f"- Gold: `{result.gold_path}`",
        f"- Target: `{result.target_path}`",
        *([f"- Silver: `{silver_path}`"] if silver_path else []),
        "- 只过滤 Target，Gold 保持不变。",
        "- 删除实体时，同时删除所有涉及该实体的关系。",
        "- 未标注置信度的条目保留。",
        "- 真实错误指 Target 中不在 Gold 的条目。",
        "",
    ]
    lines.extend(
        _render_confidence_evaluation(
            "## 2. 删除 Low 后", result.confidence_exclude_low
        )
    )
    lines.extend(
        _render_confidence_evaluation(
            "## 3. 删除 Low 和 Medium 后", result.confidence_high_only
        )
    )
    lines.extend(
        _render_confidence_details(
            result.confidence_high_only,
            silver_entities,
            silver_relations,
        )
    )
    lines.extend(
        [
            "## 4. Low/Medium 识错能力",
            "",
            "| 评估项 | 真实错误 | Low/Medium 条目 | 命中错误 | Precision | Recall | F1 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for label, detection in [
        ("实体", result.confidence_detection.entity),
        ("关系", result.confidence_detection.relation),
        ("综合", result.confidence_detection.overall),
    ]:
        lines.append(_format_confidence_detection_row(label, detection))
    warnings = [
        *(f"[Gold] {warning}" for warning in result.gold.warnings),
        *(f"[Target] {warning}" for warning in result.target.warnings),
        *((f"[Silver] {warning}" for warning in silver.warnings) if silver else ()),
    ]
    if silver_path is not None and silver is None:
        warnings.append(f"[Silver] 文件不存在，已跳过 OCR 标记: {silver_path}")
    lines.extend(["", "## 5. 解析 Warning", ""])
    lines.extend((f"- {warning}" for warning in warnings) if warnings else ["无"])
    lines.append("")
    return "\n".join(lines)


def _render_confidence_evaluation(
    title: str, evaluation: ConfidenceEvaluationResult
) -> list[str]:
    lines = [
        title,
        "",
        "| 评估项 | Accuracy | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, metrics in [
        ("实体", evaluation.entity),
        ("关系", evaluation.relation),
        ("重点关系", evaluation.key_relation),
    ]:
        lines.append(
            f"| {label} | {metrics.accuracy:.4f} | {metrics.precision:.4f} | "
            f"{metrics.recall:.4f} | {metrics.f1:.4f} |"
        )
    lines.append("")
    return lines


def _render_confidence_details(
    evaluation: ConfidenceEvaluationResult,
    silver_entities: set[EntityKey],
    silver_relations: set[RelationTriple],
) -> list[str]:
    lines: list[str] = []
    for label, metrics, formatter, ocr_items, sort_key in [
        ("实体", evaluation.entity, _format_entity, silver_entities, None),
        ("关系", evaluation.relation, _format_relation, silver_relations, None),
        (
            "重点关系",
            evaluation.key_relation,
            _format_relation,
            silver_relations,
            _key_relation_sort_key,
        ),
    ]:
        lines.extend(
            _render_item_list(
                f"{label}正确条目",
                metrics.correct,
                formatter,
                0,
                collapsible=label in {"实体", "关系"},
                sort_key=sort_key,
            )
        )
        lines.extend(
            _render_item_list(
                f"{label}缺失条目",
                metrics.missing,
                formatter,
                0,
                collapsible=label in {"实体", "关系"},
                sort_key=sort_key,
            )
        )
        lines.extend(
            _render_item_list(
                f"{label}多余条目",
                metrics.extra,
                formatter,
                0,
                collapsible=label in {"实体", "关系"},
                ocr_items=ocr_items,
                sort_key=sort_key,
            )
        )
    return lines


def _format_confidence_detection_row(
    label: str, detection: ConfidenceDetectionResult
) -> str:
    return (
        f"| {label} | {detection.true_errors} | {detection.flagged} | "
        f"{detection.flagged_errors} | {detection.precision:.4f} | "
        f"{detection.recall:.4f} | {detection.f1:.4f} |"
    )


def _render_attribute_section(result: EvaluationResult) -> list[str]:
    lines = ["## 5. 属性抽取评估", ""]
    if result.attribute is None and result.relation_attribute is None:
        return [*lines, "属性评估：已跳过", ""]

    if result.attribute is not None:
        lines.extend(["### 5.1 实体属性", ""])
        attr = result.attribute
        lines.extend(_format_metric_table(attr.metrics))
        lines.append("")
        for title, items, formatter, limit in [
            ("正确实体属性", attr.metrics.correct, _format_attribute, result.options.list_limit),
            ("缺失实体属性", attr.metrics.missing, _format_attribute, result.options.list_limit),
            ("多余实体属性", attr.metrics.extra, _format_attribute, result.options.list_limit),
            ("值错误实体属性", attr.value_wrong, _format_value_wrong_attribute, 0),
        ]:
            lines.extend(
                _render_item_list(
                    title,
                    items,
                    formatter,
                    limit,
                    collapsible=True,
                )
            )

    if result.relation_attribute is not None:
        lines.extend(["### 5.2 关系属性", ""])
        attr = result.relation_attribute
        lines.extend(_format_metric_table(attr.metrics))
        lines.append("")
        for title, items, formatter, limit in [
            (
                "正确关系属性",
                attr.metrics.correct,
                _format_relation_attribute,
                result.options.list_limit,
            ),
            (
                "缺失关系属性",
                attr.metrics.missing,
                _format_relation_attribute,
                result.options.list_limit,
            ),
            (
                "多余关系属性",
                attr.metrics.extra,
                _format_relation_attribute,
                result.options.list_limit,
            ),
            (
                "值错误关系属性",
                attr.value_wrong,
                _format_value_wrong_relation_attribute,
                0,
            ),
        ]:
            lines.extend(
                _render_item_list(
                    title,
                    items,
                    formatter,
                    limit,
                    collapsible=True,
                )
            )
    return lines


def _render_key_relation_section(
    result: EvaluationResult,
    silver_relations: set[RelationTriple],
) -> list[str]:
    lines = ["## 6. 重点关系评估", ""]
    lines.append(f"关系类型按以下顺序排列：{list(KEY_RELATION_TYPES)}。")
    lines.append("")
    lines.extend(_format_metric_table(result.key_relation_total))
    lines.append("")
    lines.extend(
        _render_item_list(
            "正确项",
            result.key_relation_total.correct,
            _format_relation,
            0,
            sort_key=_key_relation_sort_key,
        )
    )
    lines.extend(
        _render_item_list(
            "缺失项",
            result.key_relation_total.missing,
            _format_relation,
            0,
            sort_key=_key_relation_sort_key,
        )
    )
    lines.extend(
        _render_item_list(
            "多余项",
            result.key_relation_total.extra,
            _format_relation,
            0,
            ocr_items=silver_relations,
            sort_key=_key_relation_sort_key,
        )
    )
    return lines


def _render_warnings(
    result: EvaluationResult,
    reference_label: str,
    target_label: str,
    *,
    silver_path: Path | None,
    silver: EvaluationGraph | None,
) -> list[str]:
    all_warnings = [
        *(f"[{reference_label}] {w}" for w in result.gold.warnings),
        *(f"[{target_label}] {w}" for w in result.target.warnings),
        *((f"[Silver] {w}" for w in silver.warnings) if silver else ()),
    ]
    if silver_path is not None and silver is None:
        all_warnings.append(f"[Silver] 文件不存在，已跳过 OCR 标记: {silver_path}")
    lines = ["## 7. 解析 Warning", ""]
    if not all_warnings:
        lines.append("无")
    else:
        lines.extend(f"- {warning}" for warning in all_warnings)
    lines.append("")
    return lines


def _format_metric_table(metrics: MetricResult) -> list[str]:
    lines = [
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| TP | {metrics.tp} |",
        f"| FP | {metrics.fp} |",
        f"| FN | {metrics.fn} |",
        f"| Accuracy | {metrics.accuracy:.4f} |",
        f"| Precision | {metrics.precision:.4f} |",
        f"| Recall | {metrics.recall:.4f} |",
        f"| F1 | {metrics.f1:.4f} |",
    ]
    for note in metrics.zero_denominator_notes():
        lines.append(f"- {note}")
    return lines


def _render_item_list(
    title: str,
    items: set | list,
    formatter: Callable,
    limit: int,
    *,
    collapsible: bool = False,
    ocr_items: set | None = None,
    sort_key: Callable | None = None,
) -> list[str]:
    heading = f"{title} ({len(items)})"
    if not items:
        return [f"### {heading}", "", "无", ""]

    lines = (
        [
            '<details markdown="1">',
            f"<summary><strong>{heading}</strong></summary>",
            "",
        ]
        if collapsible
        else [f"### {heading}", ""]
    )
    sorted_items = sorted(items, key=sort_key or str)
    shown_items = sorted_items if limit <= 0 else sorted_items[:limit]
    lines.extend(
        f"- {formatter(item)}"
        f"{' — ⚠️（OCR错误导致）' if ocr_items and item in ocr_items else ''}"
        for item in shown_items
    )
    if 0 < limit < len(sorted_items):
        lines.append(f"- ... 还有 {len(sorted_items) - limit} 条未列出")
    lines.extend(["", "</details>", ""] if collapsible else [""])
    return lines


def _format_entity(item: EntityKey) -> str:
    return f"({item[0]}, {item[1]})"


def _format_relation(item: RelationTriple) -> str:
    return f"({item[0]}, {item[1]}, {item[2]})"


def _key_relation_sort_key(item: RelationTriple) -> tuple[int, str]:
    return KEY_RELATION_TYPES.index(item[1]), str(item)


def _format_attribute(item: AttributeAtom) -> str:
    return f"({item[0]}, {item[1]}, {item[2]}={item[3]})"


def _format_value_wrong_attribute(
    item: tuple[AttributeKey, set[str], set[str]],
) -> str:
    (name, type_en, attr_key), gold_values, target_values = item
    return (
        f"({name}, {type_en}, {attr_key}): "
        f"gold=`{', '.join(sorted(gold_values))}`, "
        f"target=`{', '.join(sorted(target_values))}`"
    )


def _format_relation_attribute(item: RelationAttributeAtom) -> str:
    return f"({item[0]}, {item[1]}, {item[2]}, {item[3]}={item[4]})"


def _format_value_wrong_relation_attribute(
    item: tuple[RelationAttributeKey, set[str], set[str]],
) -> str:
    (from_entity, relation_type, to_entity, attr_key), gold_values, target_values = item
    return (
        f"({from_entity}, {relation_type}, {to_entity}, {attr_key}): "
        f"gold=`{', '.join(sorted(gold_values))}`, "
        f"target=`{', '.join(sorted(target_values))}`"
    )
