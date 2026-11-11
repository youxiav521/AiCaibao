from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from regression_mvp.utils.json_io import load_json


def build_subject_match_request(
    *,
    file_id: str,
    report_list: list[Any],
    org_tables: list[Any],
) -> dict[str, Any]:
    return {
        "fileId": file_id,
        "reportList": report_list,
        "tables": [_table_payload(table, table_index) for table_index, table in enumerate(org_tables)],
    }


def load_local_report_list(
    root_dir: Path,
    api_config: dict[str, Any],
    *,
    template_config_key: str = "subject_match_template_file",
    presentation_by_report_type: dict[str, str] | None = None,
) -> list[Any]:
    report_list = load_raw_local_report_list(
        root_dir,
        api_config,
        template_config_key=template_config_key,
    )

    default_calibers = _configured_report_calibers(api_config)
    result: list[dict[str, Any]] = []
    for report_index, report in enumerate(report_list):
        if not isinstance(report, dict):
            raise ValueError(f"subject match reportList[{report_index}] must be an object")
        selected_presentation = None
        if presentation_by_report_type is not None:
            report_type = _required_text(
                report.get("reportType"),
                f"reportList[{report_index}].reportType",
            ).upper()
            selected_presentation = presentation_by_report_type.get(report_type)
            if selected_presentation is None:
                continue
        report_calibers = (
            [_report_caliber(report["reportCaliber"], f"reportList[{report_index}].reportCaliber")]
            if report.get("reportCaliber") is not None
            else default_calibers
        )
        for caliber in report_calibers:
            payload = _report_payload(
                report,
                report_index,
                caliber,
                selected_presentation=selected_presentation,
            )
            if len(report_calibers) > 1:
                payload["reportId"] = f'{payload["reportId"]}:caliber:{caliber}'
            result.append(payload)
    return result


def load_raw_local_report_list(
    root_dir: Path,
    api_config: dict[str, Any],
    *,
    template_config_key: str = "subject_match_template_file",
) -> list[Any]:
    configured_path = str(api_config.get(template_config_key) or "").strip()
    if not configured_path:
        raise ValueError(f"{template_config_key} must be configured")
    template_file = Path(configured_path)
    if not template_file.is_absolute():
        template_file = root_dir / template_file
    value = load_json(template_file)
    if isinstance(value, dict) and isinstance(value.get("reportList"), list):
        report_list = value["reportList"]
    elif isinstance(value, list):
        report_list = value
    else:
        raise ValueError(f"subject match template file must be a reportList array: {template_file}")
    return report_list


def local_template_config_key(org_tables: list[Any]) -> str:
    return (
        "subject_match_english_template_file"
        if statement_tables_are_english(org_tables)
        else "subject_match_template_file"
    )


def statement_tables_are_english(org_tables: list[Any]) -> bool:
    latin_texts = 0
    chinese_texts = 0
    for text in _iter_statement_texts(org_tables):
        has_latin = bool(re.search(r"[A-Za-z]", text))
        has_chinese = bool(re.search(r"[\u3400-\u9fff]", text))
        if has_latin:
            latin_texts += 1
        if has_chinese:
            chinese_texts += 1

    return latin_texts >= 3 and latin_texts > chinese_texts


def _iter_statement_texts(org_tables: list[Any]):
    for table in org_tables:
        if not isinstance(table, dict):
            continue
        for key in ("reportName", "text", "category"):
            text = _clean(table.get(key))
            if text:
                yield text
        rows = table.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, list):
                continue
            for cell in row:
                text = _clean(cell.get("text")) if isinstance(cell, dict) else _clean(cell)
                if text:
                    yield text


def _configured_report_calibers(api_config: dict[str, Any]) -> list[int]:
    configured = api_config.get("subject_match_template_report_calibers")
    if configured is None:
        configured = [api_config.get("subject_match_template_report_caliber", 2)]
    if not isinstance(configured, list) or not configured:
        raise ValueError("subject_match_template_report_calibers must be a non-empty array")

    calibers: list[int] = []
    for index, value in enumerate(configured):
        caliber = _report_caliber(value, f"subject_match_template_report_calibers[{index}]")
        if caliber not in calibers:
            calibers.append(caliber)
    return calibers


def _report_payload(
    report: Any,
    report_index: int,
    default_caliber: int,
    *,
    selected_presentation: str | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError(f"subject match reportList[{report_index}] must be an object")
    subject_list = report.get("subjectList")
    if not isinstance(subject_list, list):
        raise ValueError(f"subject match reportList[{report_index}].subjectList must be an array")

    if selected_presentation is not None:
        subject_list = [
            subject
            for subject in subject_list
            if isinstance(subject, dict)
            and _clean(subject.get("presentation")) == selected_presentation
        ]
        if not subject_list:
            raise ValueError(
                "subject match reportList[{}] has no subjects for presentation {!r}".format(
                    report_index,
                    selected_presentation,
                )
            )
    included_subject_ids = {
        _clean(subject.get("subjectId"))
        for subject in subject_list
        if isinstance(subject, dict)
    }

    return {
        "reportId": _required_text(report.get("reportId"), f"reportList[{report_index}].reportId"),
        "reportName": _required_text(
            report.get("reportName"),
            f"reportList[{report_index}].reportName",
        ),
        "reportCaliber": _report_caliber(
            report.get("reportCaliber", default_caliber),
            f"reportList[{report_index}].reportCaliber",
        ),
        "subjectList": [
            _subject_payload(
                subject,
                report_index,
                subject_index,
                included_subject_ids=included_subject_ids,
            )
            for subject_index, subject in enumerate(subject_list)
        ],
    }


def _subject_payload(
    subject: Any,
    report_index: int,
    subject_index: int,
    *,
    included_subject_ids: set[str],
) -> dict[str, Any]:
    path = f"reportList[{report_index}].subjectList[{subject_index}]"
    if not isinstance(subject, dict):
        raise ValueError(f"subject match {path} must be an object")
    return {
        "subjectId": _required_text(subject.get("subjectId"), f"{path}.subjectId"),
        "subjectName": _required_text(subject.get("subjectName"), f"{path}.subjectName"),
        "synonymSubject": _string_list(subject.get("synonymSubject"), f"{path}.synonymSubject"),
        "summaryRelation": [
            subject_id
            for subject_id in _string_list(
                subject.get("summaryRelation"),
                f"{path}.summaryRelation",
            )
            if subject_id in included_subject_ids
        ],
    }


def _report_caliber(value: Any, path: str) -> int:
    try:
        caliber = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"subject match {path} must be 1 or 2") from exc
    if caliber not in (1, 2):
        raise ValueError(f"subject match {path} must be 1 or 2")
    return caliber


def _string_list(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"subject match {path} must be an array or null")
    return [text for item in value if (text := _clean(item))]


def _required_text(value: Any, path: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"subject match {path} must not be empty")
    return text


def _table_payload(table: Any, table_index: int) -> dict[str, Any]:
    if not isinstance(table, dict):
        table = {}
    table_id = _clean(
        table.get("primaryElementId")
        or table.get("id")
        or f"table_{table_index + 1}"
    )
    report_name = _scoped_report_name(
        _clean(table.get("reportName") or table.get("text") or table.get("category")),
        table,
    )
    table_page = _safe_int(table.get("pageAt") or table.get("page") or 1)
    return {
        "reportId": _clean(table.get("reportId")),
        "reportName": report_name,
        "text": report_name,
        "pageAt": table_page,
        "bBox": _bbox(table.get("bBox") or table.get("bbox")),
        "rows": _table_rows_payload(table.get("rows"), table_id, table_page),
    }


def _scoped_report_name(report_name: str, table: dict[str, Any]) -> str:
    if not report_name or any(token in report_name for token in ("合并", "单体", "母公司")):
        return report_name

    caliber_label = _clean(table.get("caliberLabel"))
    report_caliber = table.get("reportCaliber")
    if caliber_label == "合并" or str(report_caliber).strip() == "2":
        return f"合并{report_name}"
    if caliber_label in {"单体", "母公司"} or str(report_caliber).strip() == "1":
        return f"单体{report_name}"
    return report_name


def _table_rows_payload(
    value: Any,
    table_id: str,
    table_page: int,
) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list):
        return []
    result: list[list[dict[str, Any]]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            continue
        row_page = _row_page(row, table_page)
        result.append(
            [
                _cell_payload(
                    cell,
                    table_id,
                    row_index,
                    col_index,
                    fallback_page=row_page,
                )
                for col_index, cell in enumerate(row)
            ]
        )
    return result


def _cell_payload(
    cell: Any,
    table_id: str,
    row_index: int,
    col_index: int,
    *,
    fallback_page: int,
) -> dict[str, Any]:
    if not isinstance(cell, dict):
        cell = {"text": "" if cell is None else str(cell)}
    return {
        "cellId": _clean(cell.get("cellId") or cell.get("id") or f"{table_id}_r{row_index}_c{col_index}"),
        "subjectId": _clean(cell.get("subjectId")) or None,
        "text": _clean(cell.get("text")),
        "rowspan": _safe_int(cell.get("rowspan") or cell.get("rowSpan") or 1),
        "colspan": _safe_int(cell.get("colspan") or cell.get("colSpan") or 1),
        "page": _safe_int(
            cell.get("source_page_no")
            or cell.get("sourcePageNo")
            or cell.get("page")
            or cell.get("pageAt")
            or fallback_page
        ),
        "bBox": _bbox(cell.get("bBox") or cell.get("bbox")),
    }


def _row_page(row: list[Any], fallback_page: int) -> int:
    for cell in row:
        if not isinstance(cell, dict):
            continue
        page = (
            cell.get("source_page_no")
            or cell.get("sourcePageNo")
            or cell.get("page")
            or cell.get("pageAt")
        )
        if page is not None:
            return _safe_int(page)
    return fallback_page


def _bbox(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("b", "l", "r", "t", "coordOrigin", "coord_origin"):
        if key in value:
            result[key] = value[key]
    return result


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
