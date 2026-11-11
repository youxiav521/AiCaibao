from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, cast

from src.ade.knowledge_graph.evaluate.metrics import (
    abnormal_key_relations,
    evaluate_entities,
    evaluate_key_relations,
    evaluate_relations,
    evaluate_sets,
    relation_triples,
)
from src.ade.knowledge_graph.evaluate.parser import parse_graph
from src.ade.knowledge_graph.evaluate.schema import (
    EXPECTED_TO_ENTITY,
    KEY_RELATION_TYPES,
    EvaluationGraph,
    MetricResult,
    RelationTriple,
)


EvaluationItem = tuple[str, ...]
ItemFormatter = Callable[[EvaluationItem], str]
RelationSchema = dict[str, tuple[frozenset[str], frozenset[str]]]


@dataclass(frozen=True)
class RunEvaluation:
    run: int
    path: Path
    metrics: MetricResult

    @property
    def candidates(self) -> set[EvaluationItem]:
        return cast(set[EvaluationItem], self.metrics.correct | self.metrics.extra)


@dataclass(frozen=True)
class CategoryEvaluation:
    gold: set[EvaluationItem]
    silver: set[EvaluationItem]
    runs: tuple[RunEvaluation, ...]
    frequency_items: dict[int, set[EvaluationItem]]


@dataclass(frozen=True)
class MultiEvaluationResult:
    gold_path: Path
    silver_path: Path
    entities: CategoryEvaluation
    relations: CategoryEvaluation
    core_relations: CategoryEvaluation
    abnormal_core_relations: CategoryEvaluation
    warnings: tuple[tuple[str, str], ...]

    @property
    def run_count(self) -> int:
        return len(self.entities.runs)


@dataclass(frozen=True)
class RelationTypeEvaluation:
    name: str
    defined_in_prompt: bool
    gold: set[RelationTriple]
    silver: set[RelationTriple]
    counts: Counter[RelationTriple]
    invalid_counts: Counter[RelationTriple]
    metrics: MetricResult
    filtered_metrics: MetricResult

    @property
    def schema_fp_count(self) -> int:
        return sum(
            count for relation, count in self.invalid_counts.items() if relation not in self.gold
        )

    @property
    def schema_fp_ratio(self) -> float:
        return self.schema_fp_count / self.metrics.fp if self.metrics.fp else 0.0


@dataclass(frozen=True)
class RelationTypeEvaluationResult:
    gold_path: Path
    silver_path: Path
    runs: tuple[tuple[int, Path], ...]
    relation_schema: RelationSchema
    overall: RelationTypeEvaluation
    categories: tuple[RelationTypeEvaluation, ...]
    warnings: tuple[tuple[str, str], ...]

    @property
    def run_count(self) -> int:
        return len(self.runs)


def discover_run_paths(input_dir: Path) -> list[tuple[int, Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"输入路径必须是目录: {input_dir}")

    run_paths = [
        (int(path.stem), path)
        for path in input_dir.glob("*.txt")
        if path.stem.isdigit()
    ]
    return _validate_run_paths(run_paths)


def run_multi_evaluation(
    gold_path: Path,
    silver_path: Path,
    run_paths: Sequence[tuple[int, Path]],
) -> MultiEvaluationResult:
    _require_file(gold_path, "Gold")
    _require_file(silver_path, "Silver")
    ordered_runs = _validate_run_paths(run_paths)

    gold = parse_graph(gold_path)
    silver = parse_graph(silver_path)
    parsed_runs = [(run, path, parse_graph(path)) for run, path in ordered_runs]

    entity_gold = evaluate_entities(gold, gold)
    entity_silver = evaluate_entities(silver, silver)
    relation_gold = evaluate_relations(gold, gold)
    relation_silver = evaluate_relations(silver, silver)
    core_gold = evaluate_key_relations(gold, gold)
    core_silver = evaluate_key_relations(silver, silver)
    abnormal_gold = abnormal_key_relations(gold)
    abnormal_silver = abnormal_key_relations(silver)

    entity_runs: list[RunEvaluation] = []
    relation_runs: list[RunEvaluation] = []
    core_runs: list[RunEvaluation] = []
    abnormal_runs: list[RunEvaluation] = []
    warnings: list[tuple[str, str]] = [
        *(("Gold", warning) for warning in gold.warnings),
        *(("Silver", warning) for warning in silver.warnings),
    ]

    for run, path, graph in parsed_runs:
        entity_runs.append(
            RunEvaluation(run, path, evaluate_entities(gold, graph))
        )
        relation_runs.append(RunEvaluation(run, path, evaluate_relations(gold, graph)))
        core_metrics = evaluate_key_relations(gold, graph)
        abnormal_target = abnormal_key_relations(graph)
        core_runs.append(RunEvaluation(run, path, core_metrics))
        abnormal_runs.append(
            RunEvaluation(run, path, evaluate_sets(abnormal_gold, abnormal_target))
        )
        warnings.extend((f"Run {run}", warning) for warning in graph.warnings)

    return MultiEvaluationResult(
        gold_path=gold_path,
        silver_path=silver_path,
        entities=_build_category(entity_gold, entity_silver, entity_runs),
        relations=_build_category(relation_gold, relation_silver, relation_runs),
        core_relations=_build_category(core_gold, core_silver, core_runs),
        abnormal_core_relations=_build_category(
            evaluate_sets(abnormal_gold, abnormal_gold),
            evaluate_sets(abnormal_silver, abnormal_silver),
            abnormal_runs,
        ),
        warnings=tuple(warnings),
    )


def parse_relation_schema(
    relation_types_table: str,
    relation_types: Sequence[str],
) -> RelationSchema:
    schema: RelationSchema = {}
    for line in relation_types_table.splitlines():
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) < 5 or not columns[0].isdigit():
            continue
        relation_type, from_types, to_types = columns[1:4]
        schema[relation_type] = (
            _split_entity_types(from_types),
            frozenset({"项目"})
            if to_types == "{folder_name}"
            else _split_entity_types(to_types),
        )

    expected, actual = set(relation_types), set(schema)
    if actual != expected:
        raise ValueError(
            f"关系 schema 与 RELATION_TYPE_LIST 不一致: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    if any(not from_types or not to_types for from_types, to_types in schema.values()):
        raise ValueError("关系 schema 中存在空的实体类型")
    return schema


def run_relation_type_evaluation(
    gold_path: Path,
    silver_path: Path,
    run_paths: Sequence[tuple[int, Path]],
    relation_schema: RelationSchema,
) -> RelationTypeEvaluationResult:
    _require_file(gold_path, "Gold")
    _require_file(silver_path, "Silver")
    ordered_runs = _validate_run_paths(run_paths)

    gold_graph = parse_graph(gold_path)
    silver_graph = parse_graph(silver_path)
    gold = relation_triples(gold_graph.relations)
    silver = relation_triples(silver_graph.relations)
    counts: Counter[RelationTriple] = Counter()
    invalid_counts: Counter[RelationTriple] = Counter()
    warnings: list[tuple[str, str]] = [
        *(("Gold", warning) for warning in gold_graph.warnings),
        *(("Silver", warning) for warning in silver_graph.warnings),
    ]

    for run, path in ordered_runs:
        graph = parse_graph(path)
        entity_types: dict[str, set[str]] = {}
        for entity in graph.entities:
            entity_types.setdefault(entity.name, set()).add(entity.type)
        for relation in relation_triples(graph.relations):
            counts[relation] += 1
            if _violates_relation_schema(relation, entity_types, relation_schema):
                invalid_counts[relation] += 1
        warnings.extend((f"Run {run}", warning) for warning in graph.warnings)

    run_count = len(ordered_runs)
    overall = _build_relation_type_result(
        "所有关系", True, gold, silver, counts, invalid_counts, run_count
    )
    relation_types = [
        *relation_schema,
        *sorted({relation[1] for relation in gold | set(counts)} - set(relation_schema)),
    ]
    categories = tuple(
        _build_relation_type_result(
            relation_type,
            relation_type in relation_schema,
            {relation for relation in gold if relation[1] == relation_type},
            {relation for relation in silver if relation[1] == relation_type},
            Counter(
                {
                    relation: count
                    for relation, count in counts.items()
                    if relation[1] == relation_type
                }
            ),
            Counter(
                {
                    relation: count
                    for relation, count in invalid_counts.items()
                    if relation[1] == relation_type
                }
            ),
            run_count,
        )
        for relation_type in relation_types
    )
    return RelationTypeEvaluationResult(
        gold_path=gold_path,
        silver_path=silver_path,
        runs=tuple(ordered_runs),
        relation_schema=relation_schema,
        overall=overall,
        categories=categories,
        warnings=tuple(warnings),
    )


def render_relation_type_markdown(result: RelationTypeEvaluationResult) -> str:
    lines = [
        "# 多次知识图谱关系分类评估报告",
        "",
        "## Prompt 关系 Schema",
        "",
        "| 序号 | 关系类型 | 允许头实体类型 | 允许尾实体类型 |",
        "| ---: | --- | --- | --- |",
        *(
            f"| {index} | {relation_type} | {_format_entity_types(from_types)} | "
            f"{_format_entity_types(to_types)} |"
            for index, (relation_type, (from_types, to_types)) in enumerate(
                result.relation_schema.items(), start=1
            )
        ),
        "",
        "## 输入与统计口径",
        "",
        f"- Gold: `{result.gold_path}`",
        f"- Silver: `{result.silver_path}`",
        f"- Run 数量: {result.run_count}",
        *(f"- Run {run}: `{path}`" for run, path in result.runs),
        "- 同一关系在单个 run 内只计一次；Gold 中每条关系期望出现 n 次。",
        "- Accuracy = TP / (TP + FP + FN)。",
        "- ⚠️ 表示 Gold 中没有但 Silver 中存在；"
        "‼️ 表示至少一次出现违反 schema。",
        "",
    ]
    defined_categories = tuple(
        category for category in result.categories if category.defined_in_prompt
    )
    abnormal_categories = tuple(
        category for category in result.categories if not category.defined_in_prompt
    )
    lines.extend(_render_error_ranking(defined_categories))
    lines.extend(_render_relation_type_section(0, result.overall, compare_schema=True))
    for index, category in enumerate(defined_categories, start=1):
        lines.extend(_render_relation_type_section(index, category, compare_schema=False))
    lines.extend(_render_abnormal_relations(len(defined_categories) + 1, abnormal_categories))
    if result.warnings:
        lines.extend(["## 解析 Warning", ""])
        lines.extend(f"- [{label}] {warning}" for label, warning in result.warnings)
        lines.append("")
    return "\n".join(lines)


def _split_entity_types(text: str) -> frozenset[str]:
    return frozenset(item.strip() for item in text.split("/") if item.strip())


def _format_entity_types(entity_types: frozenset[str]) -> str:
    preferred_order = ("企业主体", "自然人", "项目")
    ordered = [item for item in preferred_order if item in entity_types]
    ordered.extend(sorted(entity_types - set(ordered)))
    return "/".join(ordered)


def _violates_relation_schema(
    relation: RelationTriple,
    entity_types: dict[str, set[str]],
    relation_schema: RelationSchema,
) -> bool:
    allowed = relation_schema.get(relation[1])
    if allowed is None:
        return True
    from_types, to_types = allowed
    return not (
        entity_types.get(relation[0], set()) & from_types
        and entity_types.get(relation[2], set()) & to_types
    )


def _build_relation_type_result(
    name: str,
    defined_in_prompt: bool,
    gold: set[RelationTriple],
    silver: set[RelationTriple],
    counts: Counter[RelationTriple],
    invalid_counts: Counter[RelationTriple],
    run_count: int,
) -> RelationTypeEvaluation:
    valid_counts = Counter(
        {
            relation: count - invalid_counts[relation]
            for relation, count in counts.items()
            if count > invalid_counts[relation]
        }
    )
    return RelationTypeEvaluation(
        name=name,
        defined_in_prompt=defined_in_prompt,
        gold=gold,
        silver=silver,
        counts=counts,
        invalid_counts=invalid_counts,
        metrics=_metrics_from_occurrences(gold, counts, run_count),
        filtered_metrics=_metrics_from_occurrences(gold, valid_counts, run_count),
    )


def _metrics_from_occurrences(
    gold: set[RelationTriple],
    counts: Counter[RelationTriple],
    run_count: int,
) -> MetricResult:
    observed = {relation for relation, count in counts.items() if count > 0}
    correct = gold & observed
    extra = observed - gold
    tp = sum(counts[relation] for relation in correct)
    return MetricResult(
        tp=tp,
        fp=sum(counts[relation] for relation in extra),
        fn=run_count * len(gold) - tp,
        correct=correct,
        missing=gold - observed,
        extra=extra,
    )


def _render_error_ranking(
    categories: Sequence[RelationTypeEvaluation],
) -> list[str]:
    ranked = sorted(
        (category for category in categories if category.defined_in_prompt and category.gold),
        key=lambda category: (category.metrics.f1, category.name),
    )
    lines = [
        "## 原始易错类别汇总（F1 升序）",
        "",
        "| 关系类别 | TP | FP | FN | Accuracy | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {category.name} | {category.metrics.tp} | {category.metrics.fp} | "
        f"{category.metrics.fn} | {category.metrics.accuracy:.4f} | "
        f"{category.metrics.precision:.4f} | {category.metrics.recall:.4f} | "
        f"{category.metrics.f1:.4f} |"
        for category in ranked
    )
    if not ranked:
        lines.append("| 无可评估类别 | - | - | - | - | - | - | - |")
    lines.append("")
    return lines


def _render_relation_type_section(
    index: int,
    category: RelationTypeEvaluation,
    *,
    compare_schema: bool,
) -> list[str]:
    before, after = category.metrics, category.filtered_metrics
    lines = [f"## {index}. {category.name}", ""]
    if compare_schema:
        lines.extend(
            [
                "| 指标 | 当前 | 排除违反 schema 后 |",
                "| :--- | ---: | ---: |",
                f"| TP | {before.tp} | {after.tp} |",
                f"| FP | {before.fp} | {after.fp} |",
                f"| FN | {before.fn} | {after.fn} |",
                f"| Accuracy | {before.accuracy:.4f} | {after.accuracy:.4f} |",
                f"| Precision | {before.precision:.4f} | {after.precision:.4f} |",
                f"| Recall | {before.recall:.4f} | {after.recall:.4f} |",
                f"| F1 | {before.f1:.4f} | {after.f1:.4f} |",
                f"| 违反 schema 的 FP 比例 | {category.schema_fp_ratio:.2%} | 0.00% |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| 指标 | 值 |",
                "| :--- | ---: |",
                f"| TP | {before.tp} |",
                f"| FP | {before.fp} |",
                f"| FN | {before.fn} |",
                f"| Accuracy | {before.accuracy:.4f} |",
                f"| Precision | {before.precision:.4f} |",
                f"| Recall | {before.recall:.4f} |",
                f"| F1 | {before.f1:.4f} |",
                "",
            ]
        )
    lines.extend(_render_relation_items("正确项", before.correct, category, show_count=True))
    lines.extend(_render_relation_items("缺失项", before.missing, category, show_count=False))
    lines.extend(_render_relation_items("多余项", before.extra, category, show_count=True))
    return lines


def _render_relation_items(
    title: str,
    items: set,
    category: RelationTypeEvaluation,
    *,
    show_count: bool,
) -> list[str]:
    heading = f"{title}（{len(items)}）"
    lines = [
        '<details markdown="1">',
        f"<summary><strong>{heading}</strong></summary>",
        "",
    ]
    if not items:
        return [*lines, "无", "", "</details>", ""]
    sorted_items = sorted(
        items,
        key=(lambda relation: (-category.counts[relation], str(relation)))
        if show_count
        else str,
    )
    for relation in sorted_items:
        suffix = f" — 出现 {category.counts[relation]} 次" if show_count else ""
        markers: list[str] = []
        if category.invalid_counts[relation]:
            markers.append("‼️")
        if relation not in category.gold and relation in category.silver:
            markers.append("⚠️")
        if markers:
            suffix += f" {' '.join(markers)}"
        lines.append(f"- {_format_relation(relation)}{suffix}")
    lines.extend(["", "</details>", ""])
    return lines


def _render_abnormal_relations(
    index: int,
    categories: Sequence[RelationTypeEvaluation],
) -> list[str]:
    counts: Counter[RelationTriple] = Counter()
    invalid_counts: Counter[RelationTriple] = Counter()
    gold: set[RelationTriple] = set()
    silver: set[RelationTriple] = set()
    for category in categories:
        counts.update(category.counts)
        invalid_counts.update(category.invalid_counts)
        gold.update(category.gold)
        silver.update(category.silver)

    lines = [
        f"## {index}. 异常关系",
        "",
        '<details markdown="1">',
        f"<summary><strong>Prompt 未定义关系（{len(counts)}）</strong></summary>",
        "",
    ]
    if not counts:
        return [*lines, "无", "", "</details>", ""]
    for relation in sorted(counts, key=lambda item: (-counts[item], str(item))):
        markers = ["‼️"] if invalid_counts[relation] else []
        if relation not in gold and relation in silver:
            markers.append("⚠️")
        marker_text = f" {' '.join(markers)}" if markers else ""
        lines.append(
            f"- {_format_relation(relation)} — 出现 {counts[relation]} 次{marker_text}"
        )
    lines.extend(["", "</details>", ""])
    return lines


def render_multi_markdown(result: MultiEvaluationResult) -> str:
    lines = ["# 多次知识图谱评估报告", ""]
    lines.extend(_render_frequency_table("## 1. 实体逐频次准确率", result.entities))
    lines.extend(_render_threshold_table("## 2. 实体累计阈值准确率", result.entities))
    lines.extend(_render_run_table("## 3. 实体逐 run 性能", result.entities))
    lines.extend(_render_frequency_table("## 4. 关系逐频次准确率", result.relations))
    lines.extend(_render_threshold_table("## 5. 关系累计阈值准确率", result.relations))
    lines.extend(_render_run_table("## 6. 关系逐 run 性能", result.relations))

    lines.extend(
        [
            "## 7. 输入与统计口径",
            "",
            f"- Gold: `{result.gold_path}`",
            f"- Silver: `{result.silver_path}`",
            f"- Run 数量: {result.run_count}",
            *(f"- Run {run.run}: `{run.path}`" for run in result.entities.runs),
            "- 同一词条在单个 run 内只计一次。",
            "- correctness 仅以 Gold 为准；Silver 仅用于标注 OCR 错误。",
            "- Accuracy = TP / (TP + FP + FN)。",
            "",
        ]
    )
    lines.extend(
        _render_frequency_details(
            "## 8. 实体频次明细", result.entities, _format_entity
        )
    )
    lines.extend(
        _render_frequency_details(
            "## 9. 关系频次明细", result.relations, _format_relation
        )
    )
    lines.extend(
        [
            "## 10. 核心关系",
            "",
            f"核心关系类型：{list(KEY_RELATION_TYPES)}；仅统计 `to_entity == {EXPECTED_TO_ENTITY}` 的关系。",
            "",
        ]
    )
    lines.extend(
        _render_frequency_table("### 10.1 核心关系逐频次准确率", result.core_relations)
    )
    lines.extend(
        _render_threshold_table("### 10.2 核心关系累计阈值准确率", result.core_relations)
    )
    lines.extend(
        _render_frequency_details(
            "### 10.3 核心关系频次明细",
            result.core_relations,
            _format_relation,
        )
    )
    lines.extend(
        _render_frequency_details(
            "### 10.4 异常核心关系（不计入指标）",
            result.abnormal_core_relations,
            _format_relation,
        )
    )
    lines.extend(["## 11. 解析 Warning", ""])
    lines.extend(
        [f"- [{label}] {warning}" for label, warning in result.warnings]
        if result.warnings
        else ["无"]
    )
    lines.append("")
    return "\n".join(lines)


def _validate_run_paths(
    run_paths: Sequence[tuple[int, Path]],
) -> list[tuple[int, Path]]:
    if not run_paths:
        raise ValueError("输入目录中没有数字命名的 TXT 图谱文件")

    by_run: dict[int, Path] = {}
    for run, path in run_paths:
        if run in by_run:
            raise ValueError(f"Run 编号重复: {run}")
        _require_file(path, f"Run {run}")
        by_run[run] = path

    expected = list(range(1, max(by_run) + 1))
    actual = sorted(by_run)
    if actual != expected:
        raise ValueError(f"Run 编号必须从 1 连续到 n: actual={actual}")
    return [(run, by_run[run]) for run in expected]


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} 文件不存在: {path}")


def _build_category(
    gold_metrics: MetricResult,
    silver_metrics: MetricResult,
    runs: Sequence[RunEvaluation],
) -> CategoryEvaluation:
    gold = cast(set[EvaluationItem], set(gold_metrics.correct))
    silver = cast(set[EvaluationItem], set(silver_metrics.correct))
    frequencies = Counter(item for run in runs for item in run.candidates)
    frequency_items = {
        frequency: {item for item, count in frequencies.items() if count == frequency}
        for frequency in range(len(runs), 0, -1)
    }
    return CategoryEvaluation(gold, silver, tuple(runs), frequency_items)


def _render_frequency_table(title: str, category: CategoryEvaluation) -> list[str]:
    lines = [
        title,
        "",
        "| frequency | candidates | correct in gold | precision-like correctness |",
        "| --- | --- | --- | --- |",
    ]
    for frequency in range(len(category.runs), 0, -1):
        candidates = category.frequency_items[frequency]
        correct = len(candidates & category.gold)
        lines.append(
            f"| {frequency} | {len(candidates)} | {correct} | {_percentage(correct, len(candidates))} |"
        )
    lines.append("")
    return lines


def _render_threshold_table(title: str, category: CategoryEvaluation) -> list[str]:
    lines = [
        title,
        "",
        "| frequency | candidates | correct in gold | precision-like correctness |",
        "| --- | --- | --- | --- |",
    ]
    candidates: set[EvaluationItem] = set()
    for threshold in range(len(category.runs), 0, -1):
        candidates.update(category.frequency_items[threshold])
        correct = len(candidates & category.gold)
        lines.append(
            f"| k≥{threshold} | {len(candidates)} | {correct} | {_percentage(correct, len(candidates))} |"
        )
    lines.append("")
    return lines


def _render_run_table(title: str, category: CategoryEvaluation) -> list[str]:
    lines = [
        title,
        "",
        "| run | precision | recall | F1 | Accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in category.runs:
        metrics = run.metrics
        lines.append(
            f"| {run.run} | {metrics.precision:.4f} | {metrics.recall:.4f} | "
            f"{metrics.f1:.4f} | {metrics.accuracy:.4f} |"
        )
    lines.append("")
    return lines


def _render_frequency_details(
    title: str,
    category: CategoryEvaluation,
    formatter: ItemFormatter,
) -> list[str]:
    lines = [title, ""]
    for frequency in range(len(category.runs), 0, -1):
        items = category.frequency_items[frequency]
        heading = f"频次 {frequency} ({len(items)})"
        if not items:
            lines.extend([f"#### {heading}", "", "无", ""])
            continue
        lines.extend(
            [
                '<details markdown="1">',
                f"<summary><strong>{heading}</strong></summary>",
                "",
            ]
        )
        for item in sorted(items, key=str):
            lines.append(f"- {formatter(item)} — {_gold_status(item, category)}")
        lines.extend(["", "</details>", ""])
    return lines


def _gold_status(item: EvaluationItem, category: CategoryEvaluation) -> str:
    if item in category.gold:
        return "✅"
    if item in category.silver:
        return "⚠️（OCR错误导致）"
    return "❌"


def _percentage(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator * 100:.1f}%" if denominator else "0.0%"


def _format_entity(item: EvaluationItem) -> str:
    return f"({item[0]}, {item[1]})"


def _format_relation(item: EvaluationItem) -> str:
    return f"({item[0]}, {item[1]}, {item[2]})"
