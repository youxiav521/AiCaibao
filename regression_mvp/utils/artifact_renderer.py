from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


def render_content_preview(items: list[Any], markdown_path: Path, html_path: Path, *, title: str = "Parsed Content Preview") -> None:
    markdown = _items_to_markdown(items, title)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_content_html(markdown, items, title), encoding="utf-8")


def render_overlay_preview(
    input_file: Path,
    items: list[Any],
    output_dir: Path,
    *,
    output_name: str = "overlay.html",
    title: str = "Coordinate Overlay Preview",
    highlight_indexes: set[int] | None = None,
    diff_detail: list[dict[str, Any]] | None = None,
    comparison_items: list[Any] | None = None,
    highlight_side: str = "actual",
    extra_pages: set[int] | None = None,
    include_missing_cell_search: bool = True,
    ignore_rule_url: str | None = None,
) -> Path | None:
    if input_file.suffix.lower() != ".pdf" or not input_file.is_file():
        return None

    import fitz

    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / output_name

    page_infos: list[dict[str, Any]] = []
    text_search_records: list[dict[str, Any]] = []
    search_terms_by_page = _missing_cell_search_terms(items) if include_missing_cell_search else {}
    highlighted_pages = _highlighted_pages(items, highlight_indexes or set())
    if highlight_side == "actual":
        highlighted_pages.update(_expected_missing_pages(comparison_items, diff_detail or []))
    highlighted_pages.update(_valid_page_numbers(extra_pages))
    with fitz.open(input_file) as doc:
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            if highlighted_pages and page_no not in highlighted_pages:
                continue
            image_name = f"page_{page_no:03d}.png"
            image_path = pages_dir / image_name
            if image_path.is_file():
                try:
                    pix = fitz.Pixmap(str(image_path))
                except RuntimeError:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                    pix.save(image_path)
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                pix.save(image_path)
            page_infos.append(
                {
                    "page": page_no,
                    "pdf_width": float(page.rect.width),
                    "pdf_height": float(page.rect.height),
                    "image_width": pix.width,
                    "image_height": pix.height,
                    "image_path": f"pages/{image_name}",
                }
            )
            text_search_records.extend(_search_missing_cell_boxes(page, page_no, search_terms_by_page.get(page_no, [])))

    overlay_path.write_text(
        _overlay_html(
            page_infos,
            items,
            title=title,
            highlight_indexes=highlight_indexes,
            diff_detail=diff_detail,
            comparison_items=comparison_items,
            highlight_side=highlight_side,
            text_search_records=text_search_records,
            ignore_rule_url=ignore_rule_url,
        ),
        encoding="utf-8",
    )
    return overlay_path


def _valid_page_numbers(pages: set[int] | None) -> set[int]:
    if not pages:
        return set()
    return {page for value in pages if (page := _safe_int(value)) is not None and page > 0}


def _highlighted_pages(items: list[Any], highlight_indexes: set[int]) -> set[int]:
    if not highlight_indexes:
        return set()
    pages: set[int] = set()
    for index in highlight_indexes:
        item = _list_item(items, index - 1)
        if isinstance(item, dict):
            page = _safe_int(_item_page(item))
            if page is not None:
                pages.add(page)
    return pages


def _expected_missing_pages(items: list[Any] | None, diff_detail: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    for diff in diff_detail:
        if _diff_actual_item_index(diff) is not None:
            continue
        expected_item = _list_item(items, _diff_expected_item_index(diff))
        if isinstance(expected_item, dict):
            page = _safe_int(_item_page(expected_item))
            if page is not None:
                pages.add(page)
    return pages


def render_diff_preview(
    expected_items: list[Any] | None,
    actual_items: list[Any] | None,
    diff_detail: list[dict[str, Any]],
    html_path: Path,
) -> None:
    html_path.write_text(_diff_html(expected_items, actual_items, diff_detail), encoding="utf-8")


def render_table_diff_preview(
    expected_items: list[Any] | None,
    actual_items: list[Any] | None,
    diff_detail: list[dict[str, Any]],
    html_path: Path,
) -> None:
    html_path.write_text(_table_diff_html(expected_items, actual_items, diff_detail), encoding="utf-8")


def diff_item_indexes(diff_detail: list[dict[str, Any]]) -> set[int]:
    indexes: set[int] = set()
    for item in diff_detail:
        index = _diff_actual_item_index(item)
        if index is not None:
            indexes.add(index + 1)
    return indexes


def _items_to_markdown(items: list[Any], title: str) -> str:
    lines = [f"# {title}", ""]
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            lines.extend([f"## Item {index}", "", f"```text\n{item}\n```", ""])
            continue

        item_type = _item_type(item)
        page = _item_page(item) or ""
        lines.extend([f"## Item {index} - {item_type} - page {page}", ""])
        if item.get("md"):
            lines.extend([str(item["md"]), ""])
        elif item.get("value"):
            lines.extend([str(item["value"]), ""])
        elif item.get("html"):
            lines.extend([str(item["html"]), ""])
        elif isinstance(item.get("rows"), list):
            lines.extend([_table_markdown(item["rows"]), ""])
    return "\n".join(lines)


def _content_html(markdown: str, items: list[Any], title: str) -> str:
    body_parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;color:#1f2937;line-height:1.5}",
        "table{border-collapse:collapse;margin:12px 0;max-width:100%;overflow:auto}",
        "td,th{border:1px solid #d1d5db;padding:4px 6px;vertical-align:top}",
        ".item{border-top:1px solid #e5e7eb;padding:16px 0}",
        ".meta{color:#6b7280;font-size:12px;margin-bottom:8px}",
        "pre{white-space:pre-wrap;background:#f9fafb;border:1px solid #e5e7eb;padding:12px}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
    ]

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            body_parts.append(f"<section class=\"item\"><div class=\"meta\">Item {index}</div><pre>{html.escape(str(item))}</pre></section>")
            continue

        item_type = html.escape(_item_type(item))
        page = html.escape(str(_item_page(item) or ""))
        title = html.escape(str(item.get("category") or item.get("text") or item.get("id") or ""))
        body_parts.append("<section class=\"item\">")
        body_parts.append(f"<div class=\"meta\">Item {index} | type={item_type} | page={page}{' | ' + title if title else ''}</div>")
        if item.get("html"):
            body_parts.append(str(item["html"]))
        elif item.get("md"):
            body_parts.append(f"<pre>{html.escape(str(item['md']))}</pre>")
        elif item.get("value"):
            body_parts.append(f"<p>{html.escape(str(item['value']))}</p>")
        elif isinstance(item.get("rows"), list):
            body_parts.append(_table_html(item["rows"]))
        body_parts.append("</section>")

    body_parts.extend(
        [
            "<details><summary>Raw Markdown Preview</summary>",
            f"<pre>{html.escape(markdown)}</pre>",
            "</details>",
            "</body></html>",
        ]
    )
    return "\n".join(body_parts)


def _overlay_html(
    page_infos: list[dict[str, Any]],
    items: list[Any],
    *,
    title: str,
    highlight_indexes: set[int] | None = None,
    diff_detail: list[dict[str, Any]] | None = None,
    comparison_items: list[Any] | None = None,
    highlight_side: str = "actual",
    text_search_records: list[dict[str, Any]] | None = None,
    ignore_rule_url: str | None = None,
) -> str:
    highlight_indexes = highlight_indexes or set()
    workbench_groups = _diff_workbench_groups(
        items,
        comparison_items,
        diff_detail or [],
    )
    group_by_actual_index = {
        group["actual_index"] + 1: group["id"]
        for group in workbench_groups
        if group.get("actual_index") is not None
    }
    inspector_templates = _diff_inspector_templates(workbench_groups, ignore_rule_url)
    has_diffs = bool(workbench_groups)
    coordinate_templates, coordinate_item_count = _coordinate_inspector_templates(items)
    has_coordinate_inspector = not has_diffs and coordinate_item_count > 0
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    source_page_notes = _source_page_notes(items)
    anchored_indexes: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        page = _safe_int(_item_page(item))
        bbox = item.get("bBox") or item.get("bbox")
        if page is None or not isinstance(bbox, dict):
            continue
        by_page[page].append(
            {
                "index": index,
                "item": item,
                "bbox": bbox,
                "diff_key": group_by_actual_index.get(index),
                "coordinate_index": index,
            }
        )
        for cell_record in _cell_box_records(index, item):
            cell_record["diff_key"] = None
            cell_record["coordinate_index"] = index
            by_page[cell_record["page"]].append(cell_record)
    for group in workbench_groups:
        if group.get("kind") != "expected_missing":
            continue
        item = group.get("expected_item")
        if not isinstance(item, dict):
            continue
        page = _safe_int(_item_page(item))
        bbox = item.get("bBox") or item.get("bbox")
        if page is None or not isinstance(bbox, dict):
            continue
        by_page[page].append(
            {
                "index": group["expected_index"] + 1,
                "item": item,
                "bbox": bbox,
                "diff_key": group["id"],
                "expected_missing": True,
                "label": f"预期缺失 #{group['expected_index'] + 1}",
            }
        )
    for record in text_search_records or []:
        page = _safe_int(record.get("page"))
        bbox = record.get("bbox")
        if page is None or not isinstance(bbox, dict):
            continue
        record["coordinate_index"] = record.get("index")
        by_page[page].append(record)

    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f3f4f6;color:#111827}",
        "header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #d1d5db;padding:12px 20px}",
        ".page{position:relative;margin:24px auto;background:#fff;box-shadow:0 2px 10px rgba(0,0,0,.12);scroll-margin-top:72px}",
        ".page img{display:block;width:100%;height:auto}",
        ".box{position:absolute;border:2px solid #2563eb;background:rgba(37,99,235,.08);box-sizing:border-box}",
        ".has-diff .box:not(.diff){opacity:.14}",
        ".box.diff{border:5px solid #e11d48;background:rgba(225,29,72,.2);z-index:6;cursor:pointer;box-shadow:0 0 0 4px rgba(225,29,72,.22),0 0 24px rgba(225,29,72,.38)}",
        ".box.diff.expected-missing{border:4px dashed #7c3aed;background:rgba(124,58,237,.18);box-shadow:0 0 0 4px rgba(124,58,237,.2),0 0 24px rgba(124,58,237,.34)}",
        ".box.table{border-color:#dc2626;background:rgba(220,38,38,.08)}",
        ".box.cell{border:1px dashed #f59e0b;background:rgba(245,158,11,.1);z-index:3}",
        ".box.text-search{border:2px solid #7c3aed;background:rgba(124,58,237,.12);z-index:4}",
        ".box.table.diff{border-color:#e11d48;background:rgba(225,29,72,.22)}",
        ".box.image{border-color:#059669;background:rgba(5,150,105,.08)}",
        ".box:hover{background:rgba(250,204,21,.22);z-index:5}",
        ".label{position:absolute;left:0;top:-18px;background:#111827;color:#fff;font-size:11px;padding:1px 4px;white-space:nowrap}",
        ".box.diff .label{background:#e11d48;font-weight:700}",
        ".box.expected-missing .label{background:#6d28d9}",
        ".diff-nav{position:fixed;top:10px;right:14px;z-index:30}.diff-nav-toggle{border:1px solid #be123c;border-radius:6px;background:#fff;color:#9f1239;padding:7px 10px;font:600 13px inherit;cursor:pointer;box-shadow:0 3px 10px rgba(0,0,0,.14)}",
        ".diff-nav-panel{position:absolute;right:0;top:42px;width:min(460px,calc(100vw - 28px));max-height:calc(100vh - 64px);overflow:auto;background:#fff;border:1px solid #fecaca;border-radius:8px;box-shadow:0 18px 48px rgba(0,0,0,.24);padding:12px;display:none}.diff-nav.open .diff-nav-panel{display:block}",
        ".diff-nav-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;color:#7f1d1d}.diff-nav-controls{display:flex;gap:8px;margin-bottom:10px}.diff-nav-controls button{border:1px solid #fecaca;border-radius:5px;background:#fff7f7;color:#9f1239;padding:5px 8px;cursor:pointer}.diff-nav-count{margin-left:auto;color:#667085;font-size:12px}",
        ".diff-nav-list{display:grid;gap:8px;margin-top:10px}.severity-group{border:1px solid #eaecf0;border-radius:6px;background:#fff}.severity-group summary{cursor:pointer;display:flex;align-items:center;justify-content:space-between;padding:7px 8px;font-size:12px;font-weight:700}.severity-group.major summary,.severity-group.critical summary,.severity-group.blocker summary{color:#b42318;background:#fff7f7}.severity-group.minor summary{color:#9a6700;background:#fffaeb}.severity-group.info summary{color:#475467;background:#f8fafc}.severity-count{font-variant-numeric:tabular-nums}.severity-group-body{display:grid;gap:6px;padding:7px}.diff-nav-entry{width:100%;border:1px solid #fecaca;border-radius:6px;background:#fff;text-align:left;padding:7px 8px;cursor:pointer;color:#7f1d1d;font:12px/1.35 inherit}.diff-nav-entry:hover,.diff-nav-entry.active{background:#fff1f2;border-color:#e11d48}.diff-nav-entry small{display:block;color:#667085;margin-top:2px}",
        ".diff-inspector{border:1px solid #fecdca;border-radius:6px;background:#fff8f7;padding:10px}.diff-inspector-empty{margin:0;color:#667085}.diff-inspector h4{margin:0 0 8px;color:#7f1d1d}.inspector-action{margin:10px 0}.compare-exclusion-link{display:inline-flex;align-items:center;min-height:30px;padding:5px 9px;border:1px solid #b45309;border-radius:5px;background:#fffbeb;color:#92400e;font-size:12px;font-weight:700;text-decoration:none}.compare-exclusion-link:hover{background:#fef3c7}.inspector-diff{margin-top:8px;border-top:1px solid #fecdca;padding-top:8px}.inspector-diff summary,.inspector-table summary{cursor:pointer;font-weight:600;color:#7f1d1d}.inspector-meta{margin:6px 0;color:#667085;font-size:12px;overflow-wrap:anywhere}.inspector-values{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.inspector-values h5,.inspector-table-side h5{margin:0 0 5px;font-size:12px;color:#475467}.inspector-values pre{margin:0;max-height:260px;overflow:auto;padding:8px;background:#fff;color:#111827;border:1px solid #eaecf0;white-space:pre-wrap;overflow-wrap:anywhere}.inspector-table{margin-top:10px}.inspector-table-pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.inspector-table .table-scroll{overflow:auto;max-height:380px;border:1px solid #d9dee7;background:#fff}.inspector-table table{border-collapse:collapse;width:max-content;min-width:100%}.inspector-table th,.inspector-table td{border:1px solid #d9dee7;padding:4px 6px;vertical-align:top;max-width:260px;overflow-wrap:anywhere}.inspector-table th,.inspector-table .row-index{background:#f8fafc;color:#475467}.inspector-table td.diff{outline:2px solid #e11d48;background:#fff1f2}.inspector-table td.critical,.inspector-table td.blocker{background:#fee2e2}.inspector-table td.major{background:#fff1f0}.inspector-table td.minor{background:#fef3c7}.inspector-table tr.skipped td{background:#f8fafc;color:#667085;text-align:center;font-style:italic}@media(max-width:760px){.inspector-values,.inspector-table-pair{grid-template-columns:1fr}}",
        ".coordinate-inspector{position:fixed;top:10px;right:14px;z-index:30}.coordinate-inspector-toggle{border:1px solid #2563eb;border-radius:6px;background:#fff;color:#1d4ed8;padding:7px 10px;font:600 13px inherit;cursor:pointer;box-shadow:0 3px 10px rgba(0,0,0,.14)}.coordinate-inspector-panel{position:absolute;right:0;top:42px;width:min(460px,calc(100vw - 28px));max-height:calc(100vh - 64px);overflow:auto;background:#fff;border:1px solid #bfdbfe;border-radius:8px;box-shadow:0 18px 48px rgba(0,0,0,.24);padding:12px;display:none}.coordinate-inspector.open .coordinate-inspector-panel{display:block}.coordinate-inspector-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;color:#1e3a8a}.coordinate-inspector-controls{display:flex;gap:8px;margin-bottom:10px}.coordinate-inspector-controls button{border:1px solid #bfdbfe;border-radius:5px;background:#eff6ff;color:#1d4ed8;padding:5px 8px;cursor:pointer}.coordinate-inspector-count{margin-left:auto;color:#667085;font-size:12px}.coordinate-content{border:1px solid #bfdbfe;border-radius:6px;background:#f8fbff;padding:10px}.coordinate-content h4{margin:0 0 8px;color:#1e3a8a}.coordinate-content h5{margin:0 0 5px;font-size:12px;color:#475467}.coordinate-content pre{margin:0;max-height:360px;overflow:auto;padding:8px;background:#fff;color:#111827;border:1px solid #dbeafe;white-space:pre-wrap;overflow-wrap:anywhere}.coordinate-content .meta{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0;color:#667085;font-size:12px}.coordinate-content .meta span{padding:2px 5px;background:#eff6ff;border-radius:4px}.coordinate-table{margin-top:10px}.coordinate-table summary{cursor:pointer;font-weight:600;color:#1e3a8a}.coordinate-table-scroll{max-height:420px;overflow:auto;border:1px solid #dbeafe;background:#fff;margin-top:8px}.coordinate-table-scroll table{border-collapse:collapse;width:max-content;min-width:100%}.coordinate-table-scroll th,.coordinate-table-scroll td{border:1px solid #dbeafe;padding:4px 6px;vertical-align:top;max-width:260px;overflow-wrap:anywhere}.coordinate-table-scroll th{background:#eff6ff;color:#1e3a8a}.box.selectable{cursor:pointer}.box.selectable.active{outline:5px solid #facc15;outline-offset:4px;z-index:7}",
        ".box.diff.active{outline:5px solid #facc15;outline-offset:4px;animation:diffPulse 1.1s ease-in-out 2}@keyframes diffPulse{50%{box-shadow:0 0 0 9px rgba(250,204,21,.48),0 0 28px rgba(225,29,72,.55)}}",
        ".source-note{position:absolute;left:16px;right:16px;top:16px;z-index:4;background:rgba(255,251,235,.96);border:1px solid #f59e0b;color:#78350f;border-radius:6px;padding:8px 10px;font-size:12px;box-shadow:0 4px 10px rgba(0,0,0,.12)}",
        ".source-note strong{display:block;margin-bottom:4px}.source-note ul{margin:4px 0 0 18px;padding:0}",
        "</style>",
        "</head>",
        f"<body class=\"{'has-diff' if has_diffs else ''}\">",
        f"<header><strong>{html.escape(title)}</strong><span style=\"margin-left:12px;color:#6b7280\">蓝色=文本，红色=表格，橙色=单元格坐标，紫色=PDF 文本兜底，绿色=图片，粗红框=差异项</span></header>",
    ]
    if has_diffs:
        parts.append(_diff_navigation_html(workbench_groups, inspector_templates))
    elif has_coordinate_inspector:
        parts.append(_coordinate_inspector_html(coordinate_item_count, coordinate_templates))

    for page_info in page_infos:
        page_no = int(page_info["page"])
        image_width = float(page_info["image_width"])
        image_height = float(page_info["image_height"])
        pdf_width = float(page_info["pdf_width"])
        pdf_height = float(page_info["pdf_height"])
        scale_x = image_width / pdf_width if pdf_width else 1
        scale_y = image_height / pdf_height if pdf_height else 1

        parts.append(
            f"<section id=\"page-{page_no}\" class=\"page\" style=\"width:{int(image_width)}px\">"
            f"<img src=\"{html.escape(page_info['image_path'])}\" alt=\"page {page_no}\">"
        )
        if source_page_notes.get(page_no):
            parts.append(_source_page_note_html(source_page_notes[page_no]))
        for record in by_page.get(page_no, []):
            item = record["item"]
            bbox = record["bbox"]
            box = _scaled_box(bbox, pdf_height, scale_x, scale_y)
            if box is None:
                continue
            item_type = _item_type(item)
            css_type = item_type if item_type in {"table", "image", "cell", "text-search"} else "text"
            diff_key = record.get("diff_key")
            coordinate_index = record.get("coordinate_index")
            if diff_key:
                css_type = f"{css_type} diff"
            if record.get("expected_missing"):
                css_type = f"{css_type} expected-missing"
            label = str(record.get("label") or f"#{record['index']} {item_type}")
            anchor = ""
            interaction = ""
            if diff_key and diff_key not in anchored_indexes:
                anchor = f'id="diff-item-{diff_key}" '
                interaction = f'data-diff-index="{diff_key}" role="button" tabindex="0" '
                anchored_indexes.add(diff_key)
            elif has_coordinate_inspector and isinstance(coordinate_index, int) and coordinate_index > 0:
                css_type = f"{css_type} selectable"
                if item_type != "cell":
                    css_type = f"{css_type} coordinate-anchor"
                interaction = f'data-coordinate-index="{coordinate_index}" role="button" tabindex="0" '
            parts.append(
                "<div "
                f"{anchor}"
                f"{interaction}"
                f"class=\"box {css_type}\" "
                f"style=\"left:{box['left']:.2f}px;top:{box['top']:.2f}px;width:{box['width']:.2f}px;height:{box['height']:.2f}px\">"
                f"<div class=\"label\">{html.escape(label)}</div>"
                "</div>"
            )
        parts.append("</section>")

    parts.extend(
        [
            _diff_navigation_script() if has_diffs else "",
            _coordinate_inspector_script() if has_coordinate_inspector else "",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(parts)


def _source_page_notes(items: list[Any]) -> dict[int, list[str]]:
    notes: dict[int, list[str]] = defaultdict(list)
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        item_page = _safe_int(_item_page(item))
        rows = item.get("rows")
        if not isinstance(rows, list):
            continue
        samples_by_page: dict[int, list[str]] = defaultdict(list)
        for row in rows:
            if not isinstance(row, list):
                continue
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                page = _safe_int(cell.get("source_page_no") or cell.get("sourcePageNo"))
                if page is None or page == item_page:
                    continue
                text = _cell_text(cell).strip()
                if text:
                    samples_by_page[page].append(text)
        for page, samples in samples_by_page.items():
            unique_samples = _unique_short_texts(samples, limit=3)
            sample_text = "；".join(unique_samples) if unique_samples else "无文本样例"
            notes[page].append(
                f"Item {index} 表格包含来源于本页的单元格，但响应未提供单元格坐标，无法精确画框。样例：{sample_text}"
            )
    return notes


def _missing_cell_search_terms(items: list[Any]) -> dict[int, list[dict[str, Any]]]:
    terms: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for item_index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        item_page = _safe_int(_item_page(item))
        rows = item.get("rows")
        if not isinstance(rows, list):
            continue
        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            for cell_index, cell in enumerate(row):
                if not isinstance(cell, dict):
                    continue
                if isinstance(cell.get("bBox") or cell.get("bbox"), dict):
                    continue
                page = _safe_int(cell.get("source_page_no") or cell.get("sourcePageNo"))
                if page is None or page == item_page:
                    continue
                text = " ".join(_cell_text(cell).split())
                if not _is_searchable_cell_text(text):
                    continue
                dedupe_key = (page, text)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                terms[page].append(
                    {
                        "item_index": item_index,
                        "element_id": item.get("element_id") or item.get("id"),
                        "row": row_index,
                        "col": cell_index,
                        "text": text,
                    }
                )
    return terms


def _search_missing_cell_boxes(page: Any, page_no: int, terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for term in terms[:80]:
        text = str(term.get("text") or "")
        try:
            rects = page.search_for(text)
        except Exception:
            rects = []
        if not rects:
            rects = _search_normalized_words(page, text)
        for rect in rects[:2]:
            records.append(
                {
                    "index": int(term.get("item_index") or 0),
                    "item": {
                        "type": "text-search",
                        "page": page_no,
                        "text": text,
                        "element_id": term.get("element_id"),
                        "row": term.get("row"),
                        "col": term.get("col"),
                    },
                    "bbox": {
                        "l": float(rect.x0),
                        "t": float(rect.y0),
                        "r": float(rect.x1),
                        "b": float(rect.y1),
                    },
                    "page": page_no,
                }
            )
    return records


def _search_normalized_words(page: Any, text: str) -> list[Any]:
    needle = _normalize_search_text(text)
    if not needle:
        return []
    try:
        import fitz

        words = page.get_text("words")
    except Exception:
        return []

    haystack_parts: list[str] = []
    char_rects: list[Any] = []
    for word in words:
        if len(word) < 5:
            continue
        raw = str(word[4] or "")
        normalized = _normalize_search_text(raw)
        if not normalized:
            continue
        rect = fitz.Rect(word[0], word[1], word[2], word[3])
        haystack_parts.append(normalized)
        char_rects.extend(rect for _ in normalized)

    haystack = "".join(haystack_parts)
    start = haystack.find(needle)
    if start < 0:
        return []
    end = start + len(needle)
    matched_rects = char_rects[start:end]
    if not matched_rects:
        return []
    merged = matched_rects[0]
    for rect in matched_rects[1:]:
        merged = merged | rect
    return [merged]


def _normalize_search_text(text: str) -> str:
    replacements = {
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "％": "%",
        "－": "-",
        "—": "-",
        "–": "-",
    }
    normalized = str(text)
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _is_searchable_cell_text(text: str) -> bool:
    if len(text) < 6:
        return False
    return any(ch.isalpha() for ch in text)


def _cell_box_records(item_index: int, item: dict[str, Any]) -> list[dict[str, Any]]:
    rows = item.get("rows")
    if not isinstance(rows, list):
        return []
    records: list[dict[str, Any]] = []
    item_page = _safe_int(_item_page(item))
    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for cell_index, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            bbox = cell.get("bBox") or cell.get("bbox")
            if not isinstance(bbox, dict):
                continue
            page = _safe_int(cell.get("source_page_no") or cell.get("sourcePageNo")) or item_page
            if page is None:
                continue
            records.append(
                {
                    "index": item_index,
                    "item": {
                        "type": "cell",
                        "page": page,
                        "text": _cell_text(cell),
                        "element_id": item.get("element_id") or item.get("id"),
                        "row": row_index,
                        "col": cell_index,
                        "cellTraceabilitySource": cell.get("cellTraceabilitySource"),
                    },
                    "bbox": bbox,
                    "page": page,
                }
            )
    return records


def _source_page_note_html(notes: list[str]) -> str:
    items = "".join(f"<li>{html.escape(note)}</li>" for note in notes[:5])
    overflow = "" if len(notes) <= 5 else f"<li>另有 {len(notes) - 5} 条跨页来源提示。</li>"
    return (
        "<div class=\"source-note\">"
        "<strong>跨页表格来源提示</strong>"
        "<div>本页有表格单元格内容被合并到其他页的表格 item 中，但缺少单元格级 bbox，因此这里只能提示来源，不能画出精确坐标框。</div>"
        f"<ul>{items}{overflow}</ul>"
        "</div>"
    )


def _unique_short_texts(values: list[str], *, limit: int) -> list[str]:
    candidates = values
    if len(values) > limit and limit >= 3:
        candidates = values[: limit - 1] + [values[-1]]
    seen: set[str] = set()
    result: list[str] = []
    for value in candidates:
        text = " ".join(str(value).split())
        if not text or text in seen:
            continue
        seen.add(text)
        if len(text) > 80:
            text = text[:77] + "..."
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _diff_navigation_html(groups: list[dict[str, Any]], inspector_templates: str) -> str:
    severity_groups: list[str] = []
    for severity in ("blocker", "critical", "major", "minor", "info"):
        entries = [group for group in groups if group["severity"] == severity]
        if not entries:
            continue
        buttons = "".join(
            f'<button class="diff-nav-entry" type="button" data-diff-target="diff-item-{group["id"]}">'
            f'<strong>{html.escape(group["label"])}</strong>'
            f'<small>{html.escape(group["summary"])}（{group["diff_count"]} 处）</small>'
            "</button>"
            for group in entries
        )
        open_attr = " open" if severity in {"blocker", "critical", "major"} else ""
        severity_groups.append(
            f'<details class="severity-group {severity}"{open_attr}>'
            f'<summary>{html.escape(_severity_label(severity))}<span class="severity-count">{len(entries)}</span></summary>'
            f'<div class="severity-group-body">{buttons}</div>'
            "</details>"
        )
    return (
        '<nav class="diff-nav" aria-label="差异导航">'
        f'<button class="diff-nav-toggle" type="button" aria-expanded="false">差异导航（{len(groups)}）</button>'
        '<section class="diff-nav-panel">'
        '<div class="diff-nav-title"><strong>差异导航</strong><span>按严重程度分组</span></div>'
        '<div class="diff-nav-controls">'
        '<button type="button" data-diff-action="previous">上一处</button>'
        '<button type="button" data-diff-action="next">下一处</button>'
        '<span class="diff-nav-count">0 / 0</span>'
        "</div>"
        '<section class="diff-inspector"><p class="diff-inspector-empty">点击 PDF 上的差异框，或从下方列表选择一项查看完整对比。</p></section>'
        f"{inspector_templates}"
        f'<section class="diff-nav-list">{"".join(severity_groups)}</section>'
        "</section></nav>"
    )


def _coordinate_inspector_html(item_count: int, templates: str) -> str:
    return (
        '<aside class="coordinate-inspector" aria-label="元素详情">'
        f'<button class="coordinate-inspector-toggle" type="button" aria-expanded="false">元素详情（{item_count}）</button>'
        '<section class="coordinate-inspector-panel">'
        '<div class="coordinate-inspector-title"><strong>元素详情</strong><span>点击坐标框查看完整内容</span></div>'
        '<div class="coordinate-inspector-controls">'
        '<button type="button" data-coordinate-action="previous">上一项</button>'
        '<button type="button" data-coordinate-action="next">下一项</button>'
        '<span class="coordinate-inspector-count">0 / 0</span>'
        "</div>"
        '<section class="coordinate-content"><p class="diff-inspector-empty">点击 PDF 上的坐标框查看该元素的解析内容。</p></section>'
        f"{templates}"
        "</section></aside>"
    )


def _coordinate_inspector_templates(items: list[Any]) -> tuple[str, int]:
    templates: list[str] = []
    count = 0
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        page = _safe_int(_item_page(item))
        bbox = item.get("bBox") or item.get("bbox")
        if page is None or not isinstance(bbox, dict):
            continue
        count += 1
        item_type = _item_type(item)
        title = f"第 {index} 项 · 第 {page} 页 · {item_type}"
        rows = item.get("rows")
        if isinstance(rows, list):
            content = (
                '<details class="coordinate-table" open>'
                f"<summary>解析表格（{len(rows)} 行）</summary>"
                f'<div class="coordinate-table-scroll">{_table_html(rows)}</div>'
                "</details>"
            )
        else:
            text = _item_text(item).strip()
            content = (
                f'<section><h5>解析内容</h5><pre>{html.escape(text or "（该元素没有可展示的文本内容）")}</pre></section>'
            )
        templates.append(
            f'<template id="coordinate-detail-{index}">'
            f'<div class="coordinate-inspector-content"><h4>{html.escape(title)}</h4>'
            f"{_item_meta(item, item)}"
            f"{content}"
            "</div></template>"
        )
    return "".join(templates), count


def _diff_workbench_groups(
    actual_items: list[Any],
    expected_items: list[Any] | None,
    diff_detail: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for diff in diff_detail:
        actual_index = _diff_actual_item_index(diff)
        expected_index = _diff_expected_item_index(diff)
        kind = "actual" if actual_index is not None else "expected_missing"
        item_index = actual_index if actual_index is not None else expected_index
        if item_index is None:
            continue
        group_id = f"{kind}-{item_index + 1}"
        group = groups.get(group_id)
        if group is None:
            actual_item = _list_item(actual_items, actual_index)
            expected_item = _list_item(expected_items, expected_index)
            source_item = actual_item if isinstance(actual_item, dict) else expected_item
            if not isinstance(source_item, dict):
                continue
            item_label = f"第 {item_index + 1} 项 · 第 {_item_page(source_item) or '?'} 页 · {_item_type(source_item)}"
            if kind == "expected_missing":
                item_label = f"预期缺失 · {item_label}"
            group = {
                "id": group_id,
                "kind": kind,
                "actual_index": actual_index,
                "expected_index": expected_index,
                "actual_item": actual_item,
                "expected_item": expected_item,
                "label": item_label,
                "diffs": [],
            }
            groups[group_id] = group
        group["diffs"].append(diff)

    result: list[dict[str, Any]] = []
    for group in groups.values():
        group_diffs = group["diffs"]
        most_severe = min(group_diffs, key=lambda diff: _severity_rank(str(diff.get("severity") or "major")))
        group["severity"] = str(most_severe.get("severity") or "major")
        group["summary"] = _short_diff_note(_diff_note(most_severe))
        group["diff_count"] = len(group_diffs)
        result.append(group)
    return sorted(
        result,
        key=lambda group: (
            _severity_rank(group["severity"]),
            str(
                _item_page(
                    group["actual_item"]
                    if isinstance(group["actual_item"], dict)
                    else group["expected_item"]
                )
                or ""
            ),
            group["id"],
        ),
    )


def _diff_inspector_templates(groups: list[dict[str, Any]], ignore_rule_url: str | None = None) -> str:
    templates: list[str] = []
    for group in groups:
        title = "预期缺失项" if group["kind"] == "expected_missing" else "差异项"
        templates.append(
            f'<template id="diff-detail-{group["id"]}">'
            f'<div class="diff-inspector-content"><h4>{html.escape(title)}：{html.escape(group["label"])}</h4>'
            f'<p class="inspector-meta">本项共有 {group["diff_count"]} 处可定位差异。完整内容支持滚动查看。</p>'
            f'{_compare_exclusion_action_html(group, ignore_rule_url)}'
            f'{_inspector_diff_rows(group["diffs"])}'
            f'{_inspector_table_html(group["expected_item"], group["actual_item"], group["diffs"])}'
            "</div></template>"
        )
    return "".join(templates)


def _compare_exclusion_action_html(group: dict[str, Any], ignore_rule_url: str | None) -> str:
    if not ignore_rule_url:
        return ""
    item = group.get("expected_item")
    if not isinstance(item, dict):
        item = group.get("actual_item")
    if not isinstance(item, dict):
        return ""
    page = _safe_int(_item_page(item))
    if page is None:
        return ""
    params = {
        "page": str(page),
        "item_type": _item_type(item),
        "diff_path": str((group.get("diffs") or [{}])[0].get("path") or ""),
    }
    bbox = item.get("bBox") or item.get("bbox")
    if isinstance(bbox, dict):
        params["bbox"] = json.dumps(bbox, ensure_ascii=False, separators=(",", ":"))
    separator = "&" if "?" in ignore_rule_url else "?"
    href = f"{ignore_rule_url}{separator}{urlencode(params)}"
    return (
        '<p class="inspector-action">'
        f'<a class="compare-exclusion-link" href="{html.escape(href, quote=True)}" target="_blank" rel="noopener">'
        "将此项排除出比较</a></p>"
    )


def _diff_note(diff: dict[str, Any]) -> str:
    expected = _jsonish(diff.get("expected"), 120)
    actual = _jsonish(diff.get("actual"), 120)
    return f"{diff.get('path')}: expected={expected} | actual={actual}"


def _severity_label(value: str) -> str:
    labels = {
        "blocker": "阻塞问题",
        "critical": "严重差异",
        "major": "主要差异",
        "minor": "轻微差异",
        "info": "提示信息",
    }
    return labels.get(value, value)


def _inspector_diff_rows(diffs: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for position, diff in enumerate(diffs, start=1):
        severity = html.escape(str(diff.get("severity") or "major"))
        diff_type = html.escape(_diff_type_label(str(diff.get("type") or "changed")))
        path = html.escape(str(diff.get("path") or ""))
        rows.append(
            '<details class="inspector-diff"'
            f"{' open' if position == 1 else ''}>"
            f"<summary><strong>{severity}</strong> · {diff_type}</summary>"
            f'<div class="inspector-meta"><code>{path}</code></div>'
            '<div class="inspector-values">'
            f'<section><h5>预期答案</h5><pre>{html.escape(_jsonish_full(diff.get("expected")))}</pre></section>'
            f'<section><h5>实际结果</h5><pre>{html.escape(_jsonish_full(diff.get("actual")))}</pre></section>'
            "</div></details>"
        )
    return "".join(rows)


def _diff_type_label(value: str) -> str:
    labels = {
        "table_split_merged": "跨页表格拆分/合并",
    }
    return labels.get(value, value)


def _inspector_table_html(expected_item: Any, actual_item: Any, diffs: list[dict[str, Any]]) -> str:
    if not _has_rows(expected_item) and not _has_rows(actual_item):
        return ""
    expected_marks, actual_marks = _table_cell_marks(diffs)
    expected_rows = expected_item.get("rows") if isinstance(expected_item, dict) else None
    actual_rows = actual_item.get("rows") if isinstance(actual_item, dict) else None
    return (
        '<details class="inspector-table" open>'
        '<summary>受影响表格行对照</summary>'
        '<div class="inspector-table-pair">'
        f'<section class="inspector-table-side"><h5>预期答案</h5>{_diff_table_side(expected_rows, expected_marks)}</section>'
        f'<section class="inspector-table-side"><h5>实际结果</h5>{_diff_table_side(actual_rows, actual_marks)}</section>'
        "</div></details>"
    )


def _short_diff_note(note: str) -> str:
    normalized = " ".join(str(note).split())
    return normalized if len(normalized) <= 140 else normalized[:137] + "..."


def _diff_navigation_script() -> str:
    return """
<script>
(() => {
  const nav = document.querySelector('.diff-nav');
  if (!nav) return;
  const toggle = nav.querySelector('.diff-nav-toggle');
  const entries = Array.from(nav.querySelectorAll('[data-diff-target]'));
  const boxes = entries.map(entry => document.getElementById(entry.dataset.diffTarget)).filter(Boolean);
  const count = nav.querySelector('.diff-nav-count');
  const inspector = nav.querySelector('.diff-inspector');
  let current = -1;
  const openPanel = () => {
    nav.classList.add('open');
    toggle.setAttribute('aria-expanded', 'true');
  };
  const showDetail = box => {
    if (!inspector) return;
    const template = document.getElementById(`diff-detail-${box.dataset.diffIndex || ''}`);
    if (!template || !(template instanceof HTMLTemplateElement)) return;
    inspector.replaceChildren(template.content.cloneNode(true));
  };
  const activate = index => {
    if (!boxes.length) return;
    current = (index + boxes.length) % boxes.length;
    boxes.forEach(box => box.classList.remove('active'));
    entries.forEach(entry => entry.classList.remove('active'));
    const box = boxes[current];
    box.classList.add('active');
    openPanel();
    showDetail(box);
    const entry = entries.find(item => item.dataset.diffTarget === box.id);
    if (entry) entry.classList.add('active');
    if (count) count.textContent = `${current + 1} / ${boxes.length}`;
    box.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };
  toggle.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  entries.forEach(entry => entry.addEventListener('click', () => activate(boxes.findIndex(box => box.id === entry.dataset.diffTarget))));
  boxes.forEach((box, index) => {
    box.addEventListener('click', () => activate(index));
    box.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activate(index);
      }
    });
  });
  nav.querySelector('[data-diff-action="previous"]').addEventListener('click', () => activate(current - 1));
  nav.querySelector('[data-diff-action="next"]').addEventListener('click', () => activate(current + 1));
  if (count) count.textContent = boxes.length ? `0 / ${boxes.length}` : '0 / 0';
})();
</script>
"""


def _coordinate_inspector_script() -> str:
    return """
<script>
(() => {
  const inspector = document.querySelector('.coordinate-inspector');
  if (!inspector) return;
  const toggle = inspector.querySelector('.coordinate-inspector-toggle');
  const content = inspector.querySelector('.coordinate-content');
  const count = inspector.querySelector('.coordinate-inspector-count');
  const boxes = Array.from(document.querySelectorAll('.box[data-coordinate-index]'));
  const anchors = Array.from(document.querySelectorAll('.box.coordinate-anchor[data-coordinate-index]'));
  let current = -1;
  const openPanel = () => {
    inspector.classList.add('open');
    toggle.setAttribute('aria-expanded', 'true');
  };
  const activateKey = (key, shouldScroll) => {
    const template = document.getElementById(`coordinate-detail-${key}`);
    if (!template || !(template instanceof HTMLTemplateElement)) return;
    boxes.forEach(box => box.classList.toggle('active', box.dataset.coordinateIndex === key));
    content.replaceChildren(template.content.cloneNode(true));
    current = anchors.findIndex(anchor => anchor.dataset.coordinateIndex === key);
    if (count) count.textContent = current >= 0 ? `${current + 1} / ${anchors.length}` : `0 / ${anchors.length}`;
    openPanel();
    if (shouldScroll) {
      const anchor = anchors[current];
      if (anchor) anchor.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };
  const activate = index => {
    if (!anchors.length) return;
    const next = (index + anchors.length) % anchors.length;
    activateKey(anchors[next].dataset.coordinateIndex, true);
  };
  toggle.addEventListener('click', () => {
    const open = inspector.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  boxes.forEach(box => {
    const activateBox = () => activateKey(box.dataset.coordinateIndex, false);
    box.addEventListener('click', activateBox);
    box.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activateBox();
      }
    });
  });
  inspector.querySelector('[data-coordinate-action="previous"]').addEventListener('click', () => activate(current - 1));
  inspector.querySelector('[data-coordinate-action="next"]').addEventListener('click', () => activate(current + 1));
  if (count) count.textContent = `0 / ${anchors.length}`;
})();
</script>
"""


def _diff_html(
    expected_items: list[Any] | None,
    actual_items: list[Any] | None,
    diff_detail: list[dict[str, Any]],
) -> str:
    grouped: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for diff in diff_detail:
        grouped[_diff_item_index(diff)].append(diff)

    sections: list[str] = []
    for item_index, diffs in sorted(grouped.items(), key=lambda pair: (-1 if pair[0] is None else pair[0])):
        expected_item = _list_item(expected_items, item_index)
        actual_item = _list_item(actual_items, item_index)
        title = "Global / structure diff" if item_index is None else f"Item {item_index + 1}"
        rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(str(diff.get('path') or ''))}</code></td>"
            f"<td><pre>{html.escape(_jsonish(diff.get('expected')))}</pre></td>"
            f"<td><pre>{html.escape(_jsonish(diff.get('actual')))}</pre></td>"
            "</tr>"
            for diff in diffs
        )
        sections.append(
            "<section class=\"diff-card\">"
            f"<h2>{html.escape(title)} <span>{len(diffs)} diff(s)</span></h2>"
            f"{_item_meta(expected_item, actual_item)}"
            "<div class=\"table-wrap\"><table>"
            "<thead><tr><th>path</th><th>expected</th><th>actual</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table></div>"
            "<div class=\"pair\">"
            f"<div><h3>Expected Item</h3><pre>{html.escape(_jsonish(expected_item, 2200))}</pre></div>"
            f"<div><h3>Actual Item</h3><pre>{html.escape(_jsonish(actual_item, 2200))}</pre></div>"
            "</div>"
            "</section>"
        )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Diff Visual Preview</title>",
            "<style>",
            "body{margin:0;background:#f6f7f9;color:#17202a;font:14px/1.55 'Segoe UI','Microsoft YaHei',Arial,sans-serif}",
            "header{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #d9dee7;padding:16px 24px}",
            "main{max-width:1280px;margin:0 auto;padding:20px 24px 40px}",
            "h1{margin:0;font-size:22px} h2{margin:0 0 10px;font-size:18px} h2 span{color:#667085;font-size:12px;font-weight:400}",
            ".diff-card{margin-bottom:18px;padding:16px;background:#fff;border:1px solid #d9dee7;border-left:4px solid #b42318;border-radius:6px}",
            ".meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;color:#667085}.meta span{padding:2px 8px;background:#eef2f6;border-radius:999px}",
            ".table-wrap{overflow-x:auto} table{width:100%;min-width:760px;border-collapse:collapse} th,td{padding:8px 10px;border-top:1px solid #d9dee7;text-align:left;vertical-align:top} th{background:#f8fafc;color:#475467;font-size:12px}",
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:10px;background:#101828;color:#f9fafb;border-radius:6px;max-height:360px;overflow:auto}",
            ".pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.pair h3{margin:0 0 8px;font-size:14px}",
            "@media(max-width:900px){.pair{grid-template-columns:1fr}main,header{padding-left:14px;padding-right:14px}}",
            "</style>",
            "</head>",
            "<body>",
            f"<header><h1>Diff Visual Preview</h1><div>{len(diff_detail)} differences</div></header>",
            f"<main>{''.join(sections) if sections else '<p>No differences.</p>'}</main>",
            "</body></html>",
        ]
    )


def _table_diff_html(
    expected_items: list[Any] | None,
    actual_items: list[Any] | None,
    diff_detail: list[dict[str, Any]],
) -> str:
    groups: dict[tuple[int | None, int | None], list[dict[str, Any]]] = defaultdict(list)
    for diff in diff_detail:
        actual_index = _diff_actual_item_index(diff)
        expected_index = _diff_item_index(diff)
        expected_item = _list_item(expected_items, expected_index)
        actual_item = _list_item(actual_items, actual_index)
        if _has_rows(expected_item) or _has_rows(actual_item):
            groups[(expected_index, actual_index)].append(diff)

    sections: list[str] = []
    for (expected_index, actual_index), diffs in sorted(
        groups.items(),
        key=lambda pair: (
            pair[0][1] is None,
            pair[0][1] if pair[0][1] is not None else 10**9,
            pair[0][0] if pair[0][0] is not None else 10**9,
        ),
    ):
        expected_item = _list_item(expected_items, expected_index)
        actual_item = _list_item(actual_items, actual_index)
        expected_rows = expected_item.get("rows") if isinstance(expected_item, dict) else None
        actual_rows = actual_item.get("rows") if isinstance(actual_item, dict) else None
        expected_marks, actual_marks = _table_cell_marks(diffs)
        item_label = actual_index + 1 if actual_index is not None else (expected_index + 1 if expected_index is not None else "?")
        sections.append(
            "<section class=\"table-diff-card\">"
            f"<h2>Item {item_label} 表格差异 <span>{len(diffs)} 处</span></h2>"
            f"{_item_meta(expected_item, actual_item)}"
            f"{_table_diff_summary(diffs)}"
            f"{_table_field_diff_html(diffs)}"
            "<div class=\"table-pair\">"
            f"<div><h3>预期答案</h3>{_diff_table_side(expected_rows, expected_marks)}</div>"
            f"<div><h3>实际结果</h3>{_diff_table_side(actual_rows, actual_marks)}</div>"
            "</div>"
            "</section>"
        )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Table Diff Preview</title>",
            "<style>",
            "body{margin:0;background:#f6f7f9;color:#17202a;font:14px/1.55 'Segoe UI','Microsoft YaHei',Arial,sans-serif}",
            "header{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #d9dee7;padding:16px 24px}",
            "main{max-width:1500px;margin:0 auto;padding:20px 24px 40px}",
            "h1{margin:0;font-size:22px} h2{margin:0 0 10px;font-size:18px} h2 span{color:#667085;font-size:12px;font-weight:400}",
            ".table-diff-card{margin-bottom:18px;padding:16px;background:#fff;border:1px solid #d9dee7;border-left:4px solid #b42318;border-radius:6px}",
            ".meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;color:#667085}.meta span{padding:2px 8px;background:#eef2f6;border-radius:999px}",
            ".summary{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px}.badge{display:inline-flex;align-items:center;min-height:22px;padding:2px 8px;border-radius:999px;background:#eef2f6;color:#344054;font-size:12px;font-weight:600}",
            ".critical,.blocker{background:#fde8e7;color:#b42318}.major{background:#fff1f0;color:#b42318}.minor{background:#fff6d7;color:#9a6700}.info{background:#eef2f6;color:#667085}",
            ".field-diffs{margin:10px 0 14px;border:1px solid #fecdca;background:#fff8f7;border-radius:6px}.field-diffs summary{cursor:pointer;padding:8px 10px;font-weight:600;color:#912018}.field-diff{padding:10px;border-top:1px solid #fecdca}.field-diff:first-of-type{border-top:0}.field-diff header{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}.field-diff code{overflow-wrap:anywhere}.field-values{display:grid;grid-template-columns:1fr 1fr;gap:10px}.field-values pre{margin:0;max-height:260px;overflow:auto;padding:8px;background:#fff;border:1px solid #eaecf0;white-space:pre-wrap;overflow-wrap:anywhere}.field-note{margin:0;color:#667085}",
            ".table-pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.table-pair h3{margin:0 0 8px;font-size:14px}",
            ".table-scroll{overflow:auto;max-height:720px;border:1px solid #d9dee7;border-radius:6px;background:#fff}",
            "table{border-collapse:collapse;width:max-content;min-width:100%}th,td{border:1px solid #d9dee7;padding:5px 7px;vertical-align:top;max-width:320px;white-space:normal;overflow-wrap:anywhere}th{position:sticky;top:0;background:#f8fafc;color:#475467;z-index:1}.row-index{position:sticky;left:0;background:#f8fafc;color:#667085;z-index:1;min-width:54px}",
            "td.diff{outline:2px solid #e11d48;background:#fff1f2}td.critical,td.blocker{background:#fee2e2}td.major{background:#fff1f0}td.minor{background:#fef3c7}tr.skipped td{background:#f8fafc;color:#667085;text-align:center;font-style:italic}",
            "@media(max-width:1100px){.table-pair,.field-values{grid-template-columns:1fr}main,header{padding-left:14px;padding-right:14px}}",
            "</style>",
            "</head>",
            "<body>",
            f"<header><h1>表格差异预览</h1><div>{len(groups)} 个表格存在差异，{len(diff_detail)} 条总差异</div></header>",
            f"<main>{''.join(sections) if sections else '<p>没有可渲染的表格差异。</p>'}</main>",
            "</body></html>",
        ]
    )


def _has_rows(item: Any) -> bool:
    return isinstance(item, dict) and isinstance(item.get("rows"), list)


def _table_cell_marks(diffs: list[dict[str, Any]]) -> tuple[dict[tuple[int, int | None], str], dict[tuple[int, int | None], str]]:
    expected_marks: dict[tuple[int, int | None], str] = {}
    actual_marks: dict[tuple[int, int | None], str] = {}
    for diff in diffs:
        parsed = _parse_table_path(diff)
        severity = str(diff.get("severity") or "major")
        if parsed is None:
            continue
        expected_row, actual_row, expected_col, actual_col = parsed
        if expected_row is not None:
            _mark_cell(expected_marks, expected_row, expected_col, severity)
        if actual_row is not None:
            _mark_cell(actual_marks, actual_row, actual_col, severity)
    return expected_marks, actual_marks


def _parse_table_path(diff: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None] | None:
    path = str(diff.get("path") or "")
    row_match = re.search(r"\.rows\[(\+?\d+)(?:~(\+?\d+))?\]", path)
    if not row_match:
        return None
    expected_row = _expected_index_token(row_match.group(1))
    actual_row = _actual_index_token(row_match.group(2)) if row_match.group(2) is not None else _actual_index_token(row_match.group(1))
    col_match = re.search(r"\.rows\[[^\]]+\]\[(\+?\d+)(?:~(\+?\d+))?\]", path)
    expected_col: int | None = None
    actual_col: int | None = None
    if col_match:
        expected_col = _expected_index_token(col_match.group(1))
        actual_col = _actual_index_token(col_match.group(2)) if col_match.group(2) is not None else _actual_index_token(col_match.group(1))
    diff_type = str(diff.get("type") or "")
    if diff_type == "removed_row":
        actual_row = None
        actual_col = None
    elif diff_type == "added_row":
        expected_row = None
        expected_col = None
    return expected_row, actual_row, expected_col, actual_col


def _expected_index_token(value: str | None) -> int | None:
    if value is None or value.startswith("+"):
        return None
    return int(value)


def _actual_index_token(value: str | None) -> int | None:
    if value is None:
        return None
    if value.startswith("+"):
        return int(value[1:])
    return int(value)


def _mark_cell(marks: dict[tuple[int, int | None], str], row: int, col: int | None, severity: str) -> None:
    key = (row, col)
    old = marks.get(key)
    if old is None or _severity_rank(severity) < _severity_rank(old):
        marks[key] = severity


def _severity_rank(severity: str) -> int:
    order = {"blocker": 0, "critical": 1, "major": 2, "minor": 3, "info": 4}
    return order.get(str(severity or "major").lower(), 2)


def _table_diff_summary(diffs: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for diff in diffs:
        counts[str(diff.get("severity") or "major")] += 1
    badges = "".join(
        f"<span class=\"badge {html.escape(severity)}\">{html.escape(severity)} {count}</span>"
        for severity, count in sorted(counts.items(), key=lambda item: _severity_rank(item[0]))
    )
    numeric_count = sum(1 for diff in diffs if diff.get("type") == "financial_numeric_changed")
    numeric = f"<span class=\"badge critical\">数字差异 {numeric_count}</span>" if numeric_count else ""
    return f"<div class=\"summary\">{badges}{numeric}</div>"


def _table_field_diff_html(diffs: list[dict[str, Any]]) -> str:
    field_diffs = [diff for diff in diffs if _parse_table_path(diff) is None]
    if not field_diffs:
        return ""

    rows: list[str] = []
    for diff in field_diffs:
        path = str(diff.get("path") or "")
        field = path.rsplit(".", 1)[-1]
        severity = html.escape(str(diff.get("severity") or "major"))
        raw_diff_type = str(diff.get("type") or "changed")
        diff_type = html.escape(raw_diff_type)
        if raw_diff_type in {"added_item", "removed_item"}:
            detail = "该表格项仅存在于一侧，请直接查看下方左右表格。"
            values = f'<p class="field-note">{detail}</p>'
        elif field in {"rows", "md", "html"}:
            detail = "该字段承载完整表格内容，请直接查看下方左右表格。"
            values = f'<p class="field-note">{detail}</p>'
        else:
            values = (
                '<div class="field-values">'
                f'<div><strong>预期答案</strong><pre>{html.escape(_jsonish_full(diff.get("expected")))}</pre></div>'
                f'<div><strong>实际结果</strong><pre>{html.escape(_jsonish_full(diff.get("actual")))}</pre></div>'
                "</div>"
            )
        rows.append(
            '<article class="field-diff">'
            '<header>'
            f'<span class="badge {severity}">{severity}</span>'
            f'<span class="badge">{diff_type}</span>'
            f'<code>{html.escape(path)}</code>'
            "</header>"
            f"{values}"
            "</article>"
        )
    return (
        '<details class="field-diffs" open>'
        f'<summary>表级字段差异（{len(field_diffs)} 处，未映射到具体单元格）</summary>'
        f'{"".join(rows)}'
        "</details>"
    )


def _jsonish_full(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _diff_table_side(rows: Any, marks: dict[tuple[int, int | None], str]) -> str:
    if not isinstance(rows, list):
        return "<p>没有表格 rows。</p>"
    row_indexes = _visible_table_rows(len(rows), {row for row, _ in marks})
    max_columns = max((len(row) for row in rows if isinstance(row, list)), default=0)
    body = ["<div class=\"table-scroll\"><table><thead><tr><th class=\"row-index\">行</th>"]
    body.extend(f"<th>{col}</th>" for col in range(max_columns))
    body.append("</tr></thead><tbody>")
    previous_index: int | None = None
    for row_index in row_indexes:
        if previous_index is not None and row_index > previous_index + 1:
            body.append(f"<tr class=\"skipped\"><td colspan=\"{max_columns + 1}\">跳过 {row_index - previous_index - 1} 行无差异内容</td></tr>")
        row = rows[row_index] if 0 <= row_index < len(rows) else []
        body.append(f"<tr><td class=\"row-index\">{row_index}</td>")
        for col_index in range(max_columns):
            cell = row[col_index] if isinstance(row, list) and col_index < len(row) else ""
            severity = marks.get((row_index, col_index)) or marks.get((row_index, None))
            class_attr = f" class=\"diff {html.escape(severity)}\"" if severity else ""
            title = f" title=\"{html.escape(severity)}\"" if severity else ""
            body.append(f"<td{class_attr}{title}>{html.escape(_cell_text(cell))}</td>")
        body.append("</tr>")
        previous_index = row_index
    body.append("</tbody></table></div>")
    return "".join(body)


def _visible_table_rows(row_count: int, marked_rows: set[int]) -> list[int]:
    if not marked_rows:
        return list(range(min(row_count, 30)))
    rows: set[int] = set()
    for row in marked_rows:
        for candidate in (row - 1, row, row + 1):
            if 0 <= candidate < row_count:
                rows.add(candidate)
    return sorted(rows)


def _diff_item_index(diff: dict[str, Any]) -> int | None:
    match = re.search(r"items\[(\d+)(?:~\d+)?\]", str(diff.get("path") or ""))
    if not match:
        return None
    return int(match.group(1))


def _diff_actual_item_index(diff: dict[str, Any]) -> int | None:
    if str(diff.get("type") or "") == "removed_item":
        return None
    path = str(diff.get("path") or "")
    match = re.search(r"items\[(\d+)~(\d+)\]", path)
    if match:
        return int(match.group(2))
    match = re.search(r"items\[\+(\d+)\]", path)
    if match:
        return int(match.group(1))
    match = re.search(r"items\[(\d+)\]", path)
    if match:
        return int(match.group(1))
    return None


def _diff_expected_item_index(diff: dict[str, Any]) -> int | None:
    if str(diff.get("type") or "") == "added_item":
        return None
    return _diff_item_index(diff)


def _list_item(items: list[Any] | None, index: int | None) -> Any:
    if items is None or index is None or index < 0 or index >= len(items):
        return None
    return items[index]


def _item_meta(expected_item: Any, actual_item: Any) -> str:
    source = actual_item if isinstance(actual_item, dict) else expected_item
    if not isinstance(source, dict):
        return ""
    values = {
        "type": _item_type(source),
        "page": _item_page(source),
        "element_id": source.get("element_id") or source.get("id"),
    }
    return "<div class=\"meta\">" + "".join(
        f"<span>{html.escape(key)}={html.escape(str(value))}</span>"
        for key, value in values.items()
        if value not in (None, "")
    ) + "</div>"


def _jsonish(value: Any, limit: int = 900) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _item_type(item: dict[str, Any]) -> str:
    if item.get("type"):
        return str(item["type"])
    if isinstance(item.get("rows"), list):
        return "table"
    return "unknown"


def _item_page(item: dict[str, Any]) -> Any:
    return item.get("page") or item.get("pageAt")


def _item_text(item: dict[str, Any]) -> str:
    for key in ("value", "md", "html", "text", "category", "id"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    rows = item.get("rows")
    if isinstance(rows, list):
        cells: list[str] = []
        for row in rows[:6]:
            if isinstance(row, list):
                cells.extend(_cell_text(cell) for cell in row[:8])
        return " | ".join(text for text in cells if text)
    return ""


def _table_markdown(rows: list[Any]) -> str:
    if not rows:
        return ""
    normalized = [_row_texts(row) for row in rows]
    max_columns = max((len(row) for row in normalized), default=0)
    if max_columns == 0:
        return ""
    header = normalized[0] + [""] * (max_columns - len(normalized[0]))
    lines = [
        "| " + " | ".join(_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(max_columns)) + " |",
    ]
    for row in normalized[1:]:
        padded = row + [""] * (max_columns - len(row))
        lines.append("| " + " | ".join(_markdown_cell(cell) for cell in padded) + " |")
    return "\n".join(lines)


def _table_html(rows: list[Any]) -> str:
    if not rows:
        return "<p>No table rows.</p>"
    body: list[str] = ["<div style=\"overflow:auto\"><table>"]
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        body.append("<tr>")
        if isinstance(row, list):
            for cell in row:
                attrs = _cell_span_attrs(cell)
                body.append(f"<{tag}{attrs}>{html.escape(_cell_text(cell))}</{tag}>")
        else:
            body.append(f"<{tag}>{html.escape(str(row))}</{tag}>")
        body.append("</tr>")
    body.append("</table></div>")
    return "".join(body)


def _row_texts(row: Any) -> list[str]:
    if not isinstance(row, list):
        return [str(row)]
    return [_cell_text(cell) for cell in row]


def _cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        value = cell.get("text")
        return "" if value is None else str(value)
    return "" if cell is None else str(cell)


def _cell_span_attrs(cell: Any) -> str:
    if not isinstance(cell, dict):
        return ""
    attrs: list[str] = []
    col_span = _safe_int(cell.get("colSpan") or cell.get("colspan"))
    row_span = _safe_int(cell.get("rowSpan") or cell.get("rowspan"))
    if col_span and col_span > 1:
        attrs.append(f' colspan="{col_span}"')
    if row_span and row_span > 1:
        attrs.append(f' rowspan="{row_span}"')
    return "".join(attrs)


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _scaled_box(bbox: dict[str, Any], page_height: float, scale_x: float, scale_y: float) -> dict[str, float] | None:
    try:
        left = float(bbox["l"])
        right = float(bbox["r"])
        top_raw = float(bbox["t"])
        bottom_raw = float(bbox["b"])
    except (KeyError, TypeError, ValueError):
        return None

    coord_origin = str(bbox.get("coord_origin") or bbox.get("coordOrigin") or "TOPLEFT").upper()
    if coord_origin == "BOTTOMLEFT":
        top = page_height - bottom_raw
        bottom = page_height - top_raw
    else:
        top = top_raw
        bottom = bottom_raw

    return {
        "left": left * scale_x,
        "top": top * scale_y,
        "width": max((right - left) * scale_x, 1),
        "height": max((bottom - top) * scale_y, 1),
    }


def _diff_notes_by_item(diff_detail: list[dict[str, Any]], *, side: str) -> dict[int, list[str]]:
    notes: dict[int, list[str]] = defaultdict(list)
    for diff in diff_detail:
        item_index = _diff_expected_item_index(diff) if side == "expected" else _diff_actual_item_index(diff)
        if item_index is None:
            continue
        expected = _jsonish(diff.get("expected"), 120)
        actual = _jsonish(diff.get("actual"), 120)
        notes[item_index + 1].append(f"{diff.get('path')}: expected={expected} | actual={actual}")
    return notes


def _item_tip(index: int, item: dict[str, Any], diff_notes: list[str] | None = None) -> str:
    lines = [
        f"item: {index}",
        f"type: {_item_type(item)}",
        f"page: {_item_page(item)}",
        f"element_id: {item.get('element_id') or item.get('id')}",
    ]
    if diff_notes:
        lines.append("diff:")
        lines.extend(f"- {note}" for note in diff_notes[:6])
    rows = item.get("rows")
    if isinstance(rows, list):
        columns = max((len(row) for row in rows if isinstance(row, list)), default=0)
        lines.append(f"table: {len(rows)} rows x {columns} columns")
        lines.append("请在表格差异预览中查看完整单元格对照。")
    else:
        lines.append(f"text: {str(_item_text(item)).replace(chr(10), ' ')}")
    return "\n".join(lines)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
