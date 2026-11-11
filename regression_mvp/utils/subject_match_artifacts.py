from __future__ import annotations

import html
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


_STATEMENT_TYPES = (
    ("资产负债表", "资产负债表"),
    ("利润表", "利润表"),
    ("现金流量表", "现金流量表"),
)
_YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")


def write_subject_match_table_artifact(
    actual_rows: Any,
    expected_rows: Any,
    output_dir: Path,
    *,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    if not isinstance(actual_rows, list):
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    template_output_path = output_dir / "subject_match_table.html"
    source_output_path = output_dir / "subject_match_source_table.html"
    template_output_path.write_text(
        _render_html(
            actual_rows,
            expected_rows if isinstance(expected_rows, list) else None,
            request_payload,
        ),
        encoding="utf-8",
    )
    source_output_path.write_text(
        _render_html(
            actual_rows,
            expected_rows if isinstance(expected_rows, list) else None,
            request_payload,
            view_mode="source",
        ),
        encoding="utf-8",
    )
    return {
        "subject_match_table": str(template_output_path),
        "subject_match_source_table": str(source_output_path),
    }


def _render_html(
    actual_rows: list[Any],
    expected_rows: list[Any] | None,
    request_payload: dict[str, Any] | None,
    *,
    view_mode: str = "template",
) -> str:
    source_driven = view_mode == "source"
    subject_names = _subject_name_map(request_payload)
    expected_cells = _cells_by_id(expected_rows or [])
    table_views, source_table_count = _financial_table_views(actual_rows, request_payload)
    _fill_missing_amount_headers(table_views)

    sections = "".join(
        _table_section(
            view,
            subject_names,
            expected_cells,
            index,
            source_driven=source_driven,
        )
        for index, view in enumerate(table_views, start=1)
    )
    if not sections:
        sections = (
            '<div class="empty"><strong>未找到三大财报表格</strong>'
            "<span>当前结果中没有可识别的资产负债表、利润表或现金流量表。</span></div>"
        )

    unmatched_source_rows = sum(view["unmatched_source_rows"] for view in table_views)
    filtered_count = max(source_table_count - len(table_views), 0)
    if source_driven:
        page_title = "科目匹配源文件预览"
        hint = "按源文件页码和科目顺序展示三大表，仅呈现文件中实际解析出的科目。"
        source_rows = sum(len(view["rows"]) for view in table_views)
        matched_rows = sum(view["matched_source_rows"] for view in table_views)
        summary_metrics = (
            (len(table_views), "财报表格"),
            (source_rows, "源文件展示行"),
            (matched_rows, "已匹配源文件科目"),
            (unmatched_source_rows, "源文件未匹配科目"),
            (filtered_count, "已过滤其他表格"),
        )
    else:
        page_title = "科目匹配完整模板预览"
        hint = "按源文件页码排列三大表，并以对应口径的完整模板顺序展示。未匹配模板行留空，源文件未匹配科目单独列出。"
        template_rows = sum(len(view["template_subjects"]) for view in table_views)
        matched_rows = sum(view["matched_template_rows"] for view in table_views)
        result_rows = sum(len(view["display_entries"]) for view in table_views)
        summary_metrics = (
            (len(table_views), "财报表格"),
            (template_rows, "模板科目"),
            (matched_rows, "已命中模板科目"),
            (unmatched_source_rows, "源文件未匹配科目"),
            (result_rows, "展示行数"),
            (filtered_count, "已过滤其他表格"),
        )
    summary_html = "".join(
        f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>'
        for value, label in summary_metrics
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
    :root {{ color-scheme:light; --line:#d7dee8; --muted:#667085; --amber:#a15c00; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:#17202a; font:14px/1.55 "Segoe UI","Microsoft YaHei",Arial,sans-serif; background:#f5f7fa; }}
    .page {{ width:min(1480px, calc(100% - 32px)); margin:24px auto 48px; }}
    h1 {{ margin:0; font-size:24px; }}
    .hint {{ margin:6px 0 0; color:var(--muted); }}
    .summary {{ display:flex; flex-wrap:wrap; gap:24px; margin:20px 0 24px; padding:14px 18px; border:1px solid var(--line); background:#fff; }}
    .metric {{ display:grid; gap:2px; min-width:110px; }}
    .metric strong {{ font-size:20px; font-variant-numeric:tabular-nums; }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    .statement {{ margin:0 0 26px; scroll-margin-top:16px; }}
    .statement-head {{ display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between; gap:8px 18px; margin:0 0 8px; }}
    .statement-title {{ margin:0; font-size:18px; }}
    .statement-meta {{ display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); font-size:13px; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); background:#fff; }}
    table {{ width:100%; min-width:760px; border-collapse:collapse; table-layout:fixed; }}
    th, td {{ padding:9px 12px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); vertical-align:middle; overflow-wrap:anywhere; }}
    th:last-child, td:last-child {{ border-right:0; }}
    tbody tr:last-child td {{ border-bottom:0; }}
    th {{ position:sticky; top:0; z-index:1; background:#eef2f7; color:#344054; text-align:left; font-weight:600; }}
    th.template, td.template {{ width:27%; background:#f3f7fc; }}
    th.source, td.source {{ width:29%; }}
    th.amount, td.amount {{ width:22%; text-align:right; font-variant-numeric:tabular-nums; }}
    td.template {{ color:#344054; }}
    td.source {{ color:#101828; }}
    tr:nth-child(even) td:not(.template) {{ background:#fafbfc; }}
    tr.section-row td {{ background:#f8fafc; color:#475467; font-weight:600; }}
    tr.source-unmatched td.template {{ color:var(--amber); background:#fff6e8; }}
    tr.template-unmatched td {{ color:#667085; }}
    tr.template-unmatched td.template {{ color:#344054; background:#f3f7fc; }}
    tr.template-unmatched td.source {{ background:#f8fafc; }}
    tr.template-outside td.template {{ color:var(--amber); background:#fff6e8; }}
    tr.source-group td {{ padding:8px 12px; background:#fff6e8; color:#7a4b00; font-weight:600; }}
    tr.changed td {{ box-shadow:inset 3px 0 #f79009; }}
    .unmatched-label {{ display:inline-block; color:var(--amber); font-size:12px; font-weight:600; }}
    .status-note {{ display:block; margin-top:2px; color:var(--amber); font-size:11px; font-weight:600; }}
    .empty {{ display:grid; gap:6px; padding:28px; border:1px solid var(--line); background:#fff; color:var(--muted); }}
    .empty strong {{ color:#344054; font-size:16px; }}
    @media (max-width:760px) {{
      .page {{ width:calc(100% - 20px); margin-top:14px; }}
      .summary {{ gap:14px; }}
      .metric {{ min-width:90px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>{page_title}</h1>
    <p class="hint">{hint}</p>
    <div class="summary">
      {summary_html}
    </div>
    {sections}
  </main>
</body>
</html>
"""


def _financial_table_views(
    actual_rows: list[Any],
    request_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    rows_by_prefix: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for row in actual_rows:
        if not _is_cell_row(row):
            continue
        prefix = _cell_prefix(row[0].get("cellId"))
        if prefix:
            rows_by_prefix[prefix].append(row)

    source_tables = _request_tables(request_payload)
    if not source_tables:
        source_tables = _tables_from_result(actual_rows)

    views: list[dict[str, Any]] = []
    for source_index, source_table in enumerate(source_tables):
        report_name = _clean(source_table.get("reportName"))
        statement_type = _statement_type(report_name)
        if not statement_type:
            continue

        prefixes = _table_prefixes(source_table)
        rows: list[list[dict[str, Any]]] = []
        for prefix in prefixes:
            rows.extend(rows_by_prefix.get(prefix, []))
        rows = _rows_for_source_table(rows, source_table, report_name)
        if not rows and isinstance(source_table.get("_resultRows"), list):
            rows = source_table["_resultRows"]

        rows.sort(key=_row_sort_key)
        data_rows, amount_headers = _data_rows_and_headers(rows)
        page_numbers = sorted(
            {
                page
                for row in rows
                for cell in row
                if _clean(cell.get("text"))
                if (page := _page_number(cell.get("page") or cell.get("tablePageAt"))) is not None
            }
        )
        source_page = _page_number(source_table.get("pageAt"))
        if not page_numbers and source_page is not None:
            page_numbers = [source_page]

        caliber = _table_caliber(rows, report_name)
        template_subjects = _template_subjects_for_table(
            request_payload,
            source_table,
            report_name,
            caliber,
        )
        display_entries = _build_display_entries(data_rows, template_subjects)
        template_ids = {
            _clean(subject.get("subjectId"))
            for subject in template_subjects
            if _clean(subject.get("subjectId"))
        }
        views.append(
            {
                "source_index": source_index,
                "report_name": report_name,
                "statement_type": statement_type,
                "caliber": caliber,
                "pages": page_numbers,
                "rows": data_rows,
                "template_subjects": template_subjects,
                "display_entries": display_entries,
                "amount_headers": amount_headers,
                "amount_count": max(
                    (_amount_count(row) for row in data_rows),
                    default=len(amount_headers),
                ),
                "matched_template_rows": sum(
                    1 for entry in display_entries if entry["kind"] == "matched"
                ),
                "matched_source_rows": sum(
                    1 for row in data_rows if _row_subject_id(row)
                ),
                "unmatched_source_rows": sum(
                    1
                    for entry in display_entries
                    if entry["kind"] == "source_only"
                    and _row_has_amounts(entry["row"])
                ),
                "template_ids": template_ids,
            }
        )

    views.sort(key=lambda view: (_first_page(view["pages"]), view["source_index"]))
    return views, len(source_tables)


def _table_section(
    view: dict[str, Any],
    subject_names: dict[str, str],
    expected_cells: dict[str, dict[str, Any]],
    index: int,
    *,
    source_driven: bool = False,
) -> str:
    amount_count = max(int(view["amount_count"]), 1)
    headers = list(view["amount_headers"])
    while len(headers) < amount_count:
        headers.append(_fallback_amount_header(len(headers), amount_count))

    header_html = "".join(
        f'<th class="amount">{html.escape(label)}</th>' for label in headers[:amount_count]
    )
    row_parts: list[str] = []
    source_group_added = False
    entries = (
        [
            {"kind": "source_driven", "template_subject": None, "row": row}
            for row in view["rows"]
        ]
        if source_driven
        else view["display_entries"]
    )
    for entry in entries:
        if (
            entry["kind"] == "source_only"
            and view["template_subjects"]
            and not source_group_added
        ):
            row_parts.append(
                f'<tr class="source-group"><td colspan="{amount_count + 2}">'
                "源文件中未匹配到当前模板的科目</td></tr>"
            )
            source_group_added = True
        row_parts.append(
            _financial_row_html(
                entry["row"],
                amount_count,
                subject_names,
                expected_cells,
                template_subject=entry["template_subject"],
                entry_kind=entry["kind"],
                template_ids=view["template_ids"],
            )
        )
    rows_html = "".join(row_parts)
    if not rows_html:
        rows_html = (
            f'<tr><td colspan="{amount_count + 2}" class="source">'
            "该财报表格没有返回可展示的科目行。</td></tr>"
        )

    page_label = _page_label(view["pages"])
    meta = [f"报表类型：{view['statement_type']}"]
    if view["caliber"]:
        meta.append(f"口径：{view['caliber']}")
    if page_label:
        meta.append(page_label)
    if source_driven or not view["template_subjects"]:
        meta.append(f"源文件已匹配 {view['matched_source_rows']} / {len(view['rows'])} 行")
    else:
        meta.append(
            f"模板命中 {view['matched_template_rows']} / "
            f"{len(view['template_subjects'])} 项"
        )
        if view["unmatched_source_rows"]:
            meta.append(f"源文件未匹配 {view['unmatched_source_rows']} 行")
    meta_html = "".join(f"<span>{html.escape(item)}</span>" for item in meta)

    return f"""
    <section class="statement" id="statement-{index}">
      <div class="statement-head">
        <h2 class="statement-title">{html.escape(view['report_name'] or view['statement_type'])}</h2>
        <div class="statement-meta">{meta_html}</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th class="template">模板科目</th><th class="source">源文件科目</th>{header_html}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </section>
    """


def _financial_row_html(
    row: list[dict[str, Any]] | None,
    amount_count: int,
    subject_names: dict[str, str],
    expected_cells: dict[str, dict[str, Any]],
    *,
    template_subject: dict[str, Any] | None = None,
    entry_kind: str = "source_only",
    template_ids: set[str] | None = None,
) -> str:
    source_text = ""
    row_subject_id = ""
    values: list[str] = []
    if row:
        source_index = _source_cell_index(row)
        source_cell = row[source_index]
        source_text = _clean(source_cell.get("text"))
        row_subject_id = _clean(source_cell.get("subjectId")) or _row_subject_id(row)
        values = [_clean(cell.get("text")) for cell in row[source_index + 1 :]]
    values.extend([""] * (amount_count - len(values)))

    has_values = any(values)
    subject_id = _clean((template_subject or {}).get("subjectId")) or row_subject_id
    template_name = (
        _clean((template_subject or {}).get("subjectName"))
        or subject_names.get(subject_id, "")
    )
    row_classes: list[str] = []
    if entry_kind == "template_only":
        row_classes.append("template-unmatched")
    elif not row_subject_id and not has_values:
        row_classes.append("section-row")
    elif not row_subject_id:
        row_classes.append("source-unmatched")
    elif entry_kind == "source_only":
        row_classes.append("template-outside")
    if row and _row_changed(row, expected_cells):
        row_classes.append("changed")

    template_title = f' title="模板科目 ID：{html.escape(subject_id)}"' if subject_id else ""
    if template_name:
        template_html = html.escape(template_name)
    elif row_subject_id:
        template_html = '<span class="unmatched-label">模板名称未载入</span>'
    elif has_values:
        template_html = '<span class="unmatched-label">文件科目未匹配模板</span>'
    else:
        template_html = ""

    if entry_kind == "source_only" and row_subject_id:
        status = (
            "该模板科目已被其他源文件行占用"
            if row_subject_id in (template_ids or set())
            else "匹配结果不属于当前表对应模板"
        )
        template_html += f'<span class="status-note">{status}</span>'

    source_html = "" if entry_kind == "template_only" else html.escape(source_text)

    amount_html = "".join(
        f'<td class="amount">{html.escape(value)}</td>' for value in values[:amount_count]
    )
    class_attr = f' class="{" ".join(row_classes)}"' if row_classes else ""
    return (
        f"<tr{class_attr}><td class=\"template\"{template_title}>{template_html}</td>"
        f'<td class="source">{source_html}</td>{amount_html}</tr>'
    )


def _request_tables(request_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(request_payload, dict):
        return []
    tables = request_payload.get("tables")
    return [table for table in tables if isinstance(table, dict)] if isinstance(tables, list) else []


def _template_subjects_for_table(
    request_payload: dict[str, Any] | None,
    source_table: dict[str, Any],
    report_name: str,
    caliber: str,
) -> list[dict[str, Any]]:
    if not isinstance(request_payload, dict):
        return []
    report_list = request_payload.get("reportList")
    if not isinstance(report_list, list):
        return []
    reports = [report for report in report_list if isinstance(report, dict)]

    source_report_id = _clean(source_table.get("reportId"))
    if source_report_id:
        exact_id_reports = [
            report
            for report in reports
            if _clean(report.get("reportId")) == source_report_id
        ]
        if len(exact_id_reports) == 1:
            return _report_subjects(exact_id_reports[0])

    statement_type = _statement_type(report_name)
    candidates = [
        report
        for report in reports
        if _statement_type(_clean(report.get("reportName"))) == statement_type
    ]
    normalized_caliber = _normalize_caliber_label(caliber)
    if normalized_caliber:
        caliber_candidates = [
            report
            for report in candidates
            if _report_caliber_label(report) == normalized_caliber
        ]
        if len(caliber_candidates) == 1:
            return _report_subjects(caliber_candidates[0])
    if len(candidates) == 1:
        return _report_subjects(candidates[0])
    return []


def _report_subjects(report: dict[str, Any]) -> list[dict[str, Any]]:
    subject_list = report.get("subjectList")
    if not isinstance(subject_list, list):
        return []
    return [
        subject
        for subject in subject_list
        if isinstance(subject, dict)
        and (_clean(subject.get("subjectId")) or _clean(subject.get("subjectName")))
    ]


def _report_caliber_label(report: dict[str, Any]) -> str:
    return _normalize_caliber_label(report.get("reportCaliber"))


def _normalize_caliber_label(value: Any) -> str:
    normalized = _clean(value)
    if normalized in {"2", "合并", "consolidated"}:
        return "合并"
    if normalized in {"1", "单体", "母公司", "standalone", "parent"}:
        return "单体"
    return ""


def _build_display_entries(
    rows: list[list[dict[str, Any]]],
    template_subjects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not template_subjects:
        return [
            {"kind": "source_only", "template_subject": None, "row": row}
            for row in rows
        ]

    row_indexes_by_subject: dict[str, deque[int]] = defaultdict(deque)
    for row_index, row in enumerate(rows):
        subject_id = _row_subject_id(row)
        if subject_id:
            row_indexes_by_subject[subject_id].append(row_index)

    consumed_indexes: set[int] = set()
    entries: list[dict[str, Any]] = []
    for subject in template_subjects:
        subject_id = _clean(subject.get("subjectId"))
        row_index = (
            row_indexes_by_subject[subject_id].popleft()
            if subject_id and row_indexes_by_subject[subject_id]
            else None
        )
        if row_index is None:
            entries.append(
                {
                    "kind": "template_only",
                    "template_subject": subject,
                    "row": None,
                }
            )
            continue
        consumed_indexes.add(row_index)
        entries.append(
            {
                "kind": "matched",
                "template_subject": subject,
                "row": rows[row_index],
            }
        )

    entries.extend(
        {
            "kind": "source_only",
            "template_subject": None,
            "row": row,
        }
        for row_index, row in enumerate(rows)
        if row_index not in consumed_indexes
    )
    return entries


def _tables_from_result(actual_rows: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for row in actual_rows:
        if not _is_cell_row(row):
            continue
        prefix = _cell_prefix(row[0].get("cellId"))
        if prefix:
            grouped[prefix].append(row)

    tables = []
    for rows in grouped.values():
        first_cell = rows[0][0]
        tables.append(
            {
                "reportName": first_cell.get("reportName"),
                "pageAt": first_cell.get("tablePageAt") or first_cell.get("page"),
                "_resultRows": rows,
            }
        )
    return tables


def _data_rows_and_headers(
    rows: list[list[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    if not rows:
        return [], []

    data_rows = []
    headers: list[str] = []
    for row in rows:
        row_headers = _amount_header_labels(row)
        if row_headers:
            if len(row_headers) > len(headers):
                headers = row_headers
            continue
        data_rows.append(row)
    return data_rows, headers


def _fill_missing_amount_headers(views: list[dict[str, Any]]) -> None:
    headers_by_table: dict[tuple[str, str], list[str]] = {}
    headers_by_caliber: dict[str, list[str]] = {}
    headers_by_statement: dict[str, list[str]] = {}
    for view in views:
        headers = view["amount_headers"]
        if not headers:
            continue
        headers_by_table.setdefault((view["statement_type"], view["caliber"]), headers)
        if view["caliber"]:
            headers_by_caliber.setdefault(view["caliber"], headers)
        headers_by_statement.setdefault(view["statement_type"], headers)

    for view in views:
        if view["amount_headers"]:
            continue
        view["amount_headers"] = (
            headers_by_table.get((view["statement_type"], view["caliber"]))
            or headers_by_statement.get(view["statement_type"])
            or headers_by_caliber.get(view["caliber"])
            or []
        )


def _amount_header_labels(row: list[dict[str, Any]]) -> list[str]:
    if len(row) < 2 or _row_subject_id(row):
        return []
    texts = [_clean(cell.get("text")) for cell in row]
    header_indexes = [index for index, value in enumerate(texts) if _is_period_header(value)]
    if not header_indexes:
        return []
    start = header_indexes[0]
    candidates = texts[start:]
    if not all(not value or _is_period_header(value) for value in candidates):
        return []
    return [value for value in candidates if value]


def _is_period_header(value: str) -> bool:
    return bool(_YEAR_PATTERN.search(value)) or value in {
        "本期",
        "上期",
        "期末",
        "期初",
        "本年",
        "上年",
    }


def _source_cell_index(row: list[dict[str, Any]]) -> int:
    for index, cell in enumerate(row):
        if _clean(cell.get("subjectId")):
            return index
    for index, cell in enumerate(row):
        text = _clean(cell.get("text"))
        if text and not _looks_like_amount(text):
            return index
    return 0


def _looks_like_amount(value: str) -> bool:
    compact = value.replace(",", "").replace(" ", "")
    if compact in {"-", "--", "—", "–"}:
        return True
    compact = compact.strip("()（）%")
    try:
        float(compact)
    except ValueError:
        return False
    return True


def _amount_count(row: list[dict[str, Any]]) -> int:
    source_index = _source_cell_index(row)
    amount_count = 0
    for index, cell in enumerate(row[source_index + 1 :], start=1):
        if _clean(cell.get("text")):
            amount_count = index
    return amount_count


def _row_has_amounts(row: list[dict[str, Any]]) -> bool:
    source_index = _source_cell_index(row)
    return any(_clean(cell.get("text")) for cell in row[source_index + 1 :])


def _row_subject_id(row: list[dict[str, Any]]) -> str:
    for cell in row:
        subject_id = _clean(cell.get("subjectId"))
        if subject_id:
            return subject_id
    return ""


def _row_changed(
    row: list[dict[str, Any]],
    expected_cells: dict[str, dict[str, Any]],
) -> bool:
    if not expected_cells:
        return False
    comparable_fields = ("text", "subjectId", "statementType", "accountingYear", "caliberLabel")
    for cell in row:
        cell_id = _clean(cell.get("cellId"))
        expected = expected_cells.get(cell_id)
        if expected is None:
            return True
        if any(cell.get(field) != expected.get(field) for field in comparable_fields):
            return True
    return False


def _cells_by_id(rows: list[Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if not _is_cell_row(row):
            continue
        for cell in row:
            cell_id = _clean(cell.get("cellId"))
            if cell_id:
                result[cell_id] = cell
    return result


def _subject_name_map(request_payload: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(request_payload, dict):
        return {}
    report_list = request_payload.get("reportList")
    if not isinstance(report_list, list):
        return {}

    result: dict[str, str] = {}
    for report in report_list:
        if not isinstance(report, dict):
            continue
        subject_list = report.get("subjectList")
        if not isinstance(subject_list, list):
            continue
        for subject in subject_list:
            if not isinstance(subject, dict):
                continue
            subject_id = _clean(subject.get("subjectId"))
            subject_name = _clean(subject.get("subjectName"))
            if subject_id and subject_name:
                result[subject_id] = subject_name
    return result


def _statement_type(report_name: str) -> str:
    for marker, label in _STATEMENT_TYPES:
        if marker in report_name:
            return label
    return ""


def _table_caliber(rows: list[list[dict[str, Any]]], report_name: str) -> str:
    values = [
        _clean(cell.get("caliberLabel"))
        for row in rows
        for cell in row
        if _clean(cell.get("caliberLabel"))
    ]
    if values:
        return _normalize_caliber_label(Counter(values).most_common(1)[0][0])
    if report_name.startswith("合并"):
        return "合并"
    if report_name.startswith(("单体", "母公司")):
        return "单体"
    return ""


def _table_prefixes(source_table: dict[str, Any]) -> list[str]:
    rows = source_table.get("rows")
    if not isinstance(rows, list):
        return []
    prefixes = []
    for row in rows:
        if not _is_cell_row(row):
            continue
        for cell in row:
            prefix = _cell_prefix(cell.get("cellId"))
            if prefix and prefix not in prefixes:
                prefixes.append(prefix)
    return prefixes


def _rows_for_source_table(
    rows: list[list[dict[str, Any]]],
    source_table: dict[str, Any],
    report_name: str,
) -> list[list[dict[str, Any]]]:
    if not rows:
        return rows

    source_report_id = _clean(source_table.get("reportId"))
    if source_report_id:
        id_matches = [
            row
            for row in rows
            if any(
                _clean(cell.get("reportId")) == source_report_id
                for cell in row
            )
        ]
        if id_matches:
            return id_matches

    if report_name:
        name_matches = [
            row
            for row in rows
            if any(
                _clean(cell.get("reportName")) == report_name
                for cell in row
            )
        ]
        if name_matches:
            return name_matches

    source_caliber = _table_caliber([], report_name)
    if source_caliber:
        caliber_matches = [
            row
            for row in rows
            if _table_caliber([row], "") == source_caliber
        ]
        if caliber_matches:
            return caliber_matches
    return rows


def _cell_prefix(cell_id: Any) -> str:
    value = _clean(cell_id)
    return value.rsplit("_r", 1)[0] if "_r" in value else ""


def _row_sort_key(row: list[dict[str, Any]]) -> tuple[int, int, str]:
    first_cell = row[0]
    page = _page_number(first_cell.get("page") or first_cell.get("tablePageAt"))
    cell_id = _clean(first_cell.get("cellId"))
    row_number = 10**9
    if "_r" in cell_id:
        suffix = cell_id.rsplit("_r", 1)[1].split("_", 1)[0]
        if suffix.isdigit():
            row_number = int(suffix)
    return page if page is not None else 10**9, row_number, cell_id


def _page_number(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_page(pages: list[int]) -> int:
    return pages[0] if pages else 10**9


def _page_label(pages: list[int]) -> str:
    if not pages:
        return ""
    if len(pages) == 1:
        return f"源文件第 {pages[0]} 页"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"源文件第 {pages[0]}-{pages[-1]} 页"
    return "源文件第 " + "、".join(str(page) for page in pages) + " 页"


def _fallback_amount_header(index: int, amount_count: int) -> str:
    if amount_count == 2:
        return ("本期", "上期")[index]
    return f"金额 {index + 1}"


def _is_cell_row(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(cell, dict) for cell in value)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
