from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def write_three_statements_artifacts(
    expected_response: Any,
    actual_response: Any,
    output_dir: Path,
    case_name: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{case_name}.statement_diff_summary.md"
    grid_path = output_dir / f"{case_name}.statement_grid_preview.html"

    summary_path.write_text(
        _summary_markdown(expected_response, actual_response, case_name),
        encoding="utf-8",
    )
    grid_path.write_text(
        _grid_html(expected_response, actual_response, case_name),
        encoding="utf-8",
    )
    return {
        "statement_diff_summary": str(summary_path),
        "statement_grid_preview": str(grid_path),
    }


def write_merge_tables_artifact(
    org_tables: Any,
    merged_response: Any,
    merge_request: Any,
    output_dir: Path,
    case_name: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / f"{case_name}.merge_tables_preview.html"
    preview_path.write_text(
        _merge_tables_html(org_tables, merged_response, merge_request, case_name),
        encoding="utf-8",
    )
    return {"merge_tables_preview": str(preview_path)}


def _merge_tables_html(org_tables: Any, merged_response: Any, merge_request: Any, case_name: str) -> str:
    org_table_list = org_tables if isinstance(org_tables, list) else []
    merged_tables = merged_response if isinstance(merged_response, list) else []
    source_ids = _merge_source_ids(merge_request)
    source_tables = _selected_source_tables(org_table_list, merge_request, source_ids)

    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Merge Tables Preview - {html.escape(case_name)}</title>",
        "<style>",
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#111827;background:#f9fafb}",
        "h1{font-size:22px;margin:0 0 16px}",
        "h2{font-size:18px;margin:28px 0 10px;border-bottom:1px solid #d1d5db;padding-bottom:6px}",
        "h3{font-size:15px;margin:14px 0 8px}",
        ".meta{font-size:12px;color:#4b5563;margin:4px 0 10px}",
        ".source-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;align-items:start}",
        ".panel{background:#fff;border:1px solid #d1d5db;padding:12px;overflow:auto}",
        ".merged{border-color:#2563eb}",
        "table{border-collapse:collapse;width:max-content;min-width:100%;font-size:12px}",
        "td,th{border:1px solid #d1d5db;padding:4px 6px;vertical-align:top;max-width:280px;word-break:break-word}",
        ".row-label{position:sticky;left:0;background:#f3f4f6;color:#374151;font-weight:bold}",
        ".badge{display:inline-block;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe;padding:2px 6px;margin-right:4px;font-size:12px}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Merge Tables Preview - {html.escape(case_name)}</h1>",
        "<div class=\"meta\">Request sourceIds (original element ids): "
        + "".join(f"<span class=\"badge\">{html.escape(source_id)}</span>" for source_id in source_ids)
        + "</div>",
        f"<div class=\"meta\">Merge text: {html.escape(_merge_text(merge_request))}</div>",
        "<h2>Before Merge: Source Tables</h2>",
    ]

    if not source_tables:
        parts.append("<p class=\"meta\">No source tables matched the merge request sourceIds.</p>")
    else:
        parts.append("<div class=\"source-list\">")
        for index, table in enumerate(source_tables, start=1):
            parts.append(_plain_table_panel(f"Source {index}", table))
        parts.append("</div>")

    parts.append("<h2>After Merge: Result Tables</h2>")
    if not merged_tables:
        parts.append("<p class=\"meta\">No merged table returned.</p>")
    else:
        for index, table in enumerate(merged_tables, start=1):
            parts.append(_plain_table_panel(f"Merged {index}", table, extra_class="merged"))

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def _plain_table_panel(title: str, table_obj: dict[str, Any], *, extra_class: str = "") -> str:
    rows = _rows(table_obj)
    parts = [
        f"<section class=\"panel {html.escape(extra_class)}\">",
        f"<h3>{html.escape(title)}: {html.escape(_table_name(table_obj))}</h3>",
        "<div class=\"meta\">"
        f"id={html.escape(str(table_obj.get('id') or ''))}; "
        f"primaryElementId={html.escape(str(table_obj.get('primaryElementId') or ''))}; "
        f"mergeIds={html.escape(str(table_obj.get('mergeIds') or ''))}; "
        f"currencyUnit={html.escape(str(table_obj.get('currencyUnit') or ''))}; "
        f"pageAt={html.escape(str(table_obj.get('pageAt') or ''))}; "
        f"rows={len(rows)}"
        "</div>",
        "<table>",
    ]
    for row_index, row in enumerate(rows):
        parts.append("<tr>")
        parts.append(f"<td class=\"row-label\">{row_index}</td>")
        for cell in row:
            parts.append(f"<td>{html.escape(_cell_text(cell))}</td>")
        parts.append("</tr>")
    parts.extend(["</table>", "</section>"])
    return "\n".join(parts)


def _summary_markdown(expected_response: Any, actual_response: Any, case_name: str) -> str:
    expected_tables = expected_response if isinstance(expected_response, list) else []
    actual_tables = actual_response if isinstance(actual_response, list) else []
    first_mismatch = _first_mismatch(expected_tables, actual_tables)

    lines = [
        f"# Three Statements Diff Summary - {case_name}",
        "",
        "## Overview",
        "",
        f"- expected_statement_count: {len(expected_tables)}",
        f"- actual_statement_count: {len(actual_tables)}",
        "",
        "| index | expected_name | actual_name | expected_rows | actual_rows | first_row_mismatch |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]

    max_tables = max(len(expected_tables), len(actual_tables))
    for table_index in range(max_tables):
        expected_table = _safe_table(expected_tables, table_index)
        actual_table = _safe_table(actual_tables, table_index)
        expected_rows = _rows(expected_table)
        actual_rows = _rows(actual_table)
        row_mismatch = _first_row_mismatch(expected_rows, actual_rows)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(table_index),
                    _md(_table_name(expected_table)),
                    _md(_table_name(actual_table)),
                    str(len(expected_rows)),
                    str(len(actual_rows)),
                    "" if row_mismatch is None else str(row_mismatch),
                ]
            )
            + " |"
        )

    lines.extend(["", "## First Cell Mismatch", ""])
    if first_mismatch is None:
        lines.append("No row/cell text mismatch found.")
    else:
        lines.extend(
            [
                f"- table_index: {first_mismatch['table_index']}",
                f"- expected_table: `{_md(first_mismatch['expected_table'])}`",
                f"- actual_table: `{_md(first_mismatch['actual_table'])}`",
                f"- row_index: {first_mismatch['row_index']}",
                f"- col_index: {first_mismatch['col_index']}",
                "",
                "### Expected Row",
                "",
                "```text",
                first_mismatch["expected_row_text"],
                "```",
                "",
                "### Actual Row",
                "",
                "```text",
                first_mismatch["actual_row_text"],
                "```",
                "",
                "### Expected Cell",
                "",
                "```text",
                first_mismatch["expected_cell_text"],
                "```",
                "",
                "### Actual Cell",
                "",
                "```text",
                first_mismatch["actual_cell_text"],
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _grid_html(expected_response: Any, actual_response: Any, case_name: str) -> str:
    expected_tables = expected_response if isinstance(expected_response, list) else []
    actual_tables = actual_response if isinstance(actual_response, list) else []
    max_tables = max(len(expected_tables), len(actual_tables))

    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Three Statements Grid Preview - {html.escape(case_name)}</title>",
        "<style>",
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#111827;background:#f9fafb}",
        "h1{font-size:22px;margin:0 0 16px}",
        "h2{font-size:18px;margin:28px 0 8px}",
        ".pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}",
        ".panel{background:#fff;border:1px solid #d1d5db;padding:12px;overflow:auto}",
        ".meta{font-size:12px;color:#6b7280;margin-bottom:8px}",
        "table{border-collapse:collapse;width:max-content;min-width:100%;font-size:12px}",
        "td,th{border:1px solid #d1d5db;padding:4px 6px;vertical-align:top;max-width:280px;word-break:break-word}",
        ".diff{background:#fee2e2;border-color:#ef4444}",
        ".missing{background:#fef3c7;border-color:#f59e0b}",
        ".same{background:#ecfdf5}",
        ".row-label{position:sticky;left:0;background:#f3f4f6;color:#374151;font-weight:bold}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Three Statements Grid Preview - {html.escape(case_name)}</h1>",
        "<p class=\"meta\">Cells are compared by extracted text. Red means text differs, yellow means one side is missing.</p>",
    ]

    for table_index in range(max_tables):
        expected_table = _safe_table(expected_tables, table_index)
        actual_table = _safe_table(actual_tables, table_index)
        expected_rows = _rows(expected_table)
        actual_rows = _rows(actual_table)
        parts.append(f"<h2>Table {table_index}: {html.escape(_table_name(expected_table))} / {html.escape(_table_name(actual_table))}</h2>")
        parts.append("<div class=\"pair\">")
        parts.append(_table_panel("Expected", expected_table, expected_rows, actual_rows))
        parts.append(_table_panel("Actual", actual_table, actual_rows, expected_rows))
        parts.append("</div>")

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def _table_panel(title: str, table_obj: dict[str, Any], rows: list[list[Any]], other_rows: list[list[Any]]) -> str:
    parts = [
        "<section class=\"panel\">",
        f"<h3>{html.escape(title)}</h3>",
        "<div class=\"meta\">"
        f"name={html.escape(_table_name(table_obj))}; "
        f"currencyUnit={html.escape(str(table_obj.get('currencyUnit') or ''))}; "
        f"pageAt={html.escape(str(table_obj.get('pageAt') or ''))}; "
        f"rows={len(rows)}"
        "</div>",
        "<table>",
    ]
    max_rows = max(len(rows), len(other_rows))
    for row_index in range(max_rows):
        row = rows[row_index] if row_index < len(rows) else []
        other_row = other_rows[row_index] if row_index < len(other_rows) else []
        max_cols = max(len(row), len(other_row))
        parts.append("<tr>")
        parts.append(f"<td class=\"row-label\">{row_index}</td>")
        for col_index in range(max_cols):
            text = _cell_text(row[col_index]) if col_index < len(row) else ""
            other_text = _cell_text(other_row[col_index]) if col_index < len(other_row) else ""
            css_class = _cell_class(text, other_text, col_index < len(row), col_index < len(other_row))
            parts.append(
                f"<td class=\"{css_class}\" title=\"row={row_index}, col={col_index}\">"
                f"{html.escape(text)}"
                "</td>"
            )
        parts.append("</tr>")
    parts.extend(["</table>", "</section>"])
    return "\n".join(parts)


def _first_mismatch(expected_tables: list[Any], actual_tables: list[Any]) -> dict[str, Any] | None:
    max_tables = max(len(expected_tables), len(actual_tables))
    for table_index in range(max_tables):
        expected_table = _safe_table(expected_tables, table_index)
        actual_table = _safe_table(actual_tables, table_index)
        expected_rows = _rows(expected_table)
        actual_rows = _rows(actual_table)
        max_rows = max(len(expected_rows), len(actual_rows))
        for row_index in range(max_rows):
            expected_row = expected_rows[row_index] if row_index < len(expected_rows) else []
            actual_row = actual_rows[row_index] if row_index < len(actual_rows) else []
            max_cols = max(len(expected_row), len(actual_row))
            for col_index in range(max_cols):
                expected_text = _cell_text(expected_row[col_index]) if col_index < len(expected_row) else ""
                actual_text = _cell_text(actual_row[col_index]) if col_index < len(actual_row) else ""
                if expected_text != actual_text:
                    return {
                        "table_index": table_index,
                        "expected_table": _table_name(expected_table),
                        "actual_table": _table_name(actual_table),
                        "row_index": row_index,
                        "col_index": col_index,
                        "expected_row_text": " | ".join(_cell_text(cell) for cell in expected_row),
                        "actual_row_text": " | ".join(_cell_text(cell) for cell in actual_row),
                        "expected_cell_text": expected_text,
                        "actual_cell_text": actual_text,
                    }
    return None


def _first_row_mismatch(expected_rows: list[list[Any]], actual_rows: list[list[Any]]) -> int | None:
    max_rows = max(len(expected_rows), len(actual_rows))
    for row_index in range(max_rows):
        expected_row = expected_rows[row_index] if row_index < len(expected_rows) else []
        actual_row = actual_rows[row_index] if row_index < len(actual_rows) else []
        if [_cell_text(cell) for cell in expected_row] != [_cell_text(cell) for cell in actual_row]:
            return row_index
    return None


def _safe_table(tables: list[Any], index: int) -> dict[str, Any]:
    if index >= len(tables) or not isinstance(tables[index], dict):
        return {}
    return tables[index]


def _rows(table_obj: dict[str, Any]) -> list[list[Any]]:
    rows = table_obj.get("rows")
    if not isinstance(rows, list):
        return []
    return [row if isinstance(row, list) else [] for row in rows]


def _table_name(table_obj: dict[str, Any]) -> str:
    return str(table_obj.get("text") or table_obj.get("category") or "<missing>")


def _cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("text") or "")
    return "" if cell is None else str(cell)


def _cell_class(text: str, other_text: str, has_cell: bool, other_has_cell: bool) -> str:
    if not has_cell or not other_has_cell:
        return "missing"
    if text != other_text:
        return "diff"
    return "same"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _merge_source_ids(merge_request: Any) -> list[str]:
    first_merge = _first_merge(merge_request)
    source_ids = first_merge.get("sourceIds") if isinstance(first_merge, dict) else []
    if not isinstance(source_ids, list):
        return []
    return [str(value) for value in source_ids]


def _merge_text(merge_request: Any) -> str:
    first_merge = _first_merge(merge_request)
    if not isinstance(first_merge, dict):
        return ""
    return str(first_merge.get("text") or "")


def _first_merge(merge_request: Any) -> dict[str, Any]:
    if not isinstance(merge_request, dict):
        return {}
    merges = merge_request.get("merges")
    if not isinstance(merges, list) or not merges or not isinstance(merges[0], dict):
        return {}
    return merges[0]


def _selected_source_tables(
    org_tables: list[Any],
    merge_request: Any,
    source_ids: list[str],
) -> list[dict[str, Any]]:
    debug = merge_request.get("debug") if isinstance(merge_request, dict) else None
    selected_debug = debug.get("selected_org_tables") if isinstance(debug, dict) else None
    selected_ids = []
    if isinstance(selected_debug, list):
        for item in selected_debug:
            if isinstance(item, dict) and item.get("id") is not None:
                selected_ids.append(str(item["id"]))

    selected_tables: list[dict[str, Any]] = []
    for table in org_tables:
        if not isinstance(table, dict):
            continue
        logical_id = str(table.get("id") or "")
        primary_id = str(table.get("primaryElementId") or table.get("element_id") or "")
        if logical_id in selected_ids or primary_id in source_ids:
            selected_tables.append(table)
    return selected_tables
