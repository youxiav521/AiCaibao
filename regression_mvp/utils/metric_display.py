from __future__ import annotations

from typing import Any


SEVERITY_LABELS = {
    "blocker": "阻塞问题",
    "critical": "严重差异",
    "major": "主要差异",
    "minor": "轻微差异",
    "info": "提示信息",
}

DIFF_KIND_LABELS = {
    "added_field": "新增字段",
    "removed_field": "缺失字段",
    "type_changed": "类型变化",
    "changed": "值变化",
    "text_changed": "文本变化",
    "float_changed": "数值波动",
    "length_changed": "数量变化",
    "added_item": "新增元素",
    "removed_item": "缺失元素",
    "added_row": "新增表格行",
    "removed_row": "缺失表格行",
    "table_split_merged": "跨页表格拆分/合并",
    "table_layout_changed": "跨页续表布局变化",
    "extra_field": "新增字段",
    "missing_field": "缺失字段",
    "value_changed": "值变化",
}


def key_metric_rows(endpoint: str, result_metrics: dict[str, Any]) -> list[tuple[str, str]]:
    rows = _common_rows(result_metrics)

    if endpoint in {"file_read", "mineru_to_json", "mineru_upload_async", "ade_jobs_async"}:
        return rows

    if endpoint == "financial_metadata":
        return _financial_metadata_rows(result_metrics)
    if endpoint == "financial_three_statements":
        return _financial_three_statements_rows(result_metrics)
    if endpoint == "financial_org_tables":
        return _financial_org_tables_rows(result_metrics)
    if endpoint == "financial_merge_tables":
        return _financial_merge_tables_rows(result_metrics)
    if endpoint == "financial_subject_match":
        return _financial_subject_match_rows(result_metrics)

    return rows


def diff_metric_rows(diff_metrics: dict[str, Any]) -> list[tuple[str, str]]:
    total = int(diff_metrics.get("total") or 0)
    rows: list[tuple[str, str]] = [("差异总数", str(total))]
    if total == 0:
        rows.append(("结论", "未发现差异"))
        return rows

    severity = diff_metrics.get("by_severity")
    if isinstance(severity, dict) and severity:
        rows.extend(_severity_rows(severity))

    kinds = diff_metrics.get("by_kind")
    if isinstance(kinds, dict) and kinds:
        rows.append(("差异类型分布", _fmt_diff_kinds(kinds)))

    fields = diff_metrics.get("top_fields")
    if isinstance(fields, dict) and fields:
        rows.append(("高频差异字段", _fmt_mapping(fields)))

    sample_paths = diff_metrics.get("sample_paths")
    if isinstance(sample_paths, list) and sample_paths:
        rows.append(("优先排查路径", _fmt_list(sample_paths, limit=5)))

    return rows


def _severity_rows(severity: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key in ("blocker", "critical", "major", "minor", "info"):
        count = int(severity.get(key) or 0)
        if count:
            rows.append((SEVERITY_LABELS.get(key, key), str(count)))
    return rows


def _fmt_diff_kinds(kinds: dict[str, Any]) -> str:
    parts = []
    for key, count in kinds.items():
        label = DIFF_KIND_LABELS.get(str(key), str(key))
        parts.append(f"{label}: {count}")
    return "；".join(parts) or "-"


def _common_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("解析元素数", _fmt(metrics.get("item_count"))),
        ("页数", _fmt(metrics.get("page_count"))),
        ("文本块数", _fmt(metrics.get("text_count"))),
        ("表格数", _fmt(metrics.get("table_count"))),
        ("图片数", _fmt(metrics.get("image_count"))),
        ("空文本比例", _fmt_percent(metrics.get("empty_text_ratio"))),
        ("最大表格行数", _fmt(metrics.get("max_table_rows"))),
    ]


def _financial_metadata_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    data = metrics.get("financial_metadata")
    if not isinstance(data, dict):
        return _common_rows(metrics)
    return [
        ("元数据字段数", _fmt(data.get("metadata_field_count"))),
        ("非空元数据字段数", _fmt(data.get("metadata_non_empty_field_count"))),
        ("带溯源字段数", _fmt(data.get("metadata_source_element_count"))),
        ("三大表数量", _fmt(data.get("three_statements_count"))),
        ("变动说明数量", _fmt(data.get("change_explanation_count"))),
        ("识别企业全称", _fmt_bool(data.get("has_company_full_name"))),
        ("识别当前期数", _fmt_bool(data.get("has_current_period"))),
        ("识别审计信息", _fmt_bool(data.get("has_audit"))),
        ("缺失核心字段", _fmt_list(data.get("missing_core_fields"))),
    ]


def _financial_three_statements_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    data = metrics.get("financial_three_statements")
    if not isinstance(data, dict):
        return _common_rows(metrics)
    return [
        ("财务报表数量", _fmt(data.get("statement_count"))),
        ("报表名称", _fmt_list(data.get("statement_names"))),
        ("识别页码", _fmt_list(data.get("pages"))),
        ("币种/单位", _fmt_list(data.get("currency_units"))),
        ("总行数", _fmt(data.get("total_rows"))),
        ("最大表格行数", _fmt(data.get("max_rows"))),
        ("有 rows 的表数", _fmt(data.get("tables_with_rows"))),
        ("有 lines 的表数", _fmt(data.get("tables_with_lines"))),
    ]


def _financial_org_tables_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    data = metrics.get("financial_org_tables")
    if not isinstance(data, dict):
        return _common_rows(metrics)
    return [
        ("原始表格数", _fmt(data.get("org_table_count"))),
        ("财报表格数", _fmt(data.get("financial_statement_table_count"))),
        ("总行数", _fmt(data.get("total_rows"))),
        ("最大表格行数", _fmt(data.get("max_rows"))),
        ("有坐标表格数", _fmt(data.get("tables_with_bbox"))),
        ("坐标覆盖率", _fmt_percent(data.get("bbox_coverage_ratio"))),
        ("表格名称", _fmt_list(data.get("table_names"))),
        ("页码", _fmt_list(data.get("pages"))),
    ]


def _financial_merge_tables_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    data = metrics.get("financial_merge_tables")
    if not isinstance(data, dict):
        return _common_rows(metrics)
    return [
        ("合并请求数量", _fmt(data.get("merge_count"))),
        ("源表数量", _fmt(data.get("source_id_count"))),
        ("合并后表格数", _fmt(data.get("output_table_count"))),
        ("合并后总行数", _fmt(data.get("output_total_rows"))),
        ("合并表名称", _fmt_list(data.get("output_table_names"))),
        ("源表 ID", _fmt_list(data.get("source_ids"))),
    ]


def _financial_subject_match_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    data = metrics.get("financial_subject_match")
    if not isinstance(data, dict):
        return _common_rows(metrics)
    return [
        ("行数", _fmt(data.get("row_count"))),
        ("单元格数", _fmt(data.get("cell_count"))),
        ("已匹配科目数", _fmt(data.get("matched_subject_count"))),
        ("匹配率", _fmt_percent(data.get("matched_subject_rate"))),
        ("未匹配单元格数", _fmt(data.get("empty_subject_count"))),
        ("唯一科目数", _fmt(data.get("unique_subject_count"))),
        ("匹配科目 ID", _fmt_list(data.get("matched_subject_ids"))),
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "-"


def _fmt_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_list(value: Any, limit: int = 8) -> str:
    if not isinstance(value, list) or not value:
        return "-"
    items = [str(item) for item in value[:limit]]
    suffix = f" 等 {len(value)} 项" if len(value) > limit else ""
    return "、".join(items) + suffix


def _fmt_mapping(value: dict[str, Any]) -> str:
    return "；".join(f"{key}: {child}" for key, child in value.items()) or "-"
