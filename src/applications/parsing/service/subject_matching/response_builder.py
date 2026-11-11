from __future__ import annotations

from typing import Any, List, Optional

from src.applications.parsing.schemas.subject_match_schema import SubjectMatchRequest, SubjectMatchResponse
from src.applications.parsing.utils.financial_recognition_and_extraction_utils import (
    detect_gaap_type,
    infer_column_index_to_caliber_for_table,
    infer_column_index_to_gaap_for_table,
    infer_column_index_to_statement_type_for_table,
    infer_column_index_to_year_for_table,
    map_to_statement_type,
    match_caliber_label,
)


def resolve_cell_column_key(cell: Any, fallback_idx: int) -> int:
    if isinstance(cell, dict):
        v = cell.get("colStart") or cell.get("colstart") or cell.get("col_start")
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
    return fallback_idx


def build_subject_match_rows_response(
    request: SubjectMatchRequest,
    response: SubjectMatchResponse,
) -> List[List[Any]]:
    """Build the router response rows for subject matching.

    Kept separate from the HTTP route so domestic matching behavior remains the
    same while the service and route layers can be tested independently.
    """
    report_id_to_caliber = {
        r.reportId: r.reportCaliber for r in request.reportList if r.reportId
    }

    all_rows: list[list[Any]] = []
    for table in response.tables:
        inferred_caliber: Optional[int] = None
        if table.reportId and table.reportId in report_id_to_caliber:
            inferred_caliber = report_id_to_caliber.get(table.reportId)
        elif table.reportName:
            if "合并" in table.reportName:
                inferred_caliber = 2
            elif "母公司" in table.reportName or "单体" in table.reportName:
                inferred_caliber = 1

        caliber_label = match_caliber_label(inferred_caliber, table.reportName or "")
        table_statement_type = map_to_statement_type(table.reportName or "", table.reportName or "")
        rows_text = "\n".join(
            " | ".join((c.text or "").strip() for c in row[:4])
            for row in (table.rows or [])[:20]
        )
        table_gaap = detect_gaap_type(table.reportName or "", rows_text)
        if caliber_label == "未知" and "合并" not in (table.reportName or "") and "母公司" not in (table.reportName or ""):
            caliber_label = "单体"
        col_year = infer_column_index_to_year_for_table(
            table.rows,
            report_name=table.reportName or "",
        )
        col_caliber = infer_column_index_to_caliber_for_table(
            table.rows,
            table_caliber=caliber_label,
            report_name=table.reportName or "",
        )
        col_statement_type = infer_column_index_to_statement_type_for_table(
            table.rows,
            report_name=table.reportName or "",
            fallback_statement_type=table_statement_type,
        )
        col_gaap = infer_column_index_to_gaap_for_table(
            table.rows,
            report_name=table.reportName or "",
            context_blob=rows_text,
            fallback_gaap=table_gaap,
        )
        data_col_calibers = {v for k, v in col_caliber.items() if k > 0 and v in {"合并", "单体", "混合"}}
        if "合并" in data_col_calibers and "单体" in data_col_calibers:
            table_caliber_effective = "混合"
        elif "合并" in data_col_calibers:
            table_caliber_effective = "合并"
        elif "单体" in data_col_calibers:
            table_caliber_effective = "单体"
        else:
            table_caliber_effective = caliber_label

        for row in table.rows:
            serialized_row = []
            for col_idx, cell in enumerate(row):
                cell_dict = cell.model_dump(
                    by_alias=True,
                    exclude_none=False,
                    exclude={
                        "instanceId",
                        "docId",
                        "docFolderId",
                        "areaId",
                        "financialsId",
                        "beTitle",
                        "rowNum",
                        "colNum",
                    },
                )
                cell_dict.setdefault("reportId", table.reportId)
                cell_dict.setdefault("reportName", table.reportName)
                cell_dict.setdefault("reportCaliber", inferred_caliber)
                cell_dict.setdefault("tablePageAt", table.pageAt)
                col_key = resolve_cell_column_key(cell_dict, col_idx)
                cell_dict["caliberLabel"] = (
                    (col_caliber.get(col_key) if col_key > 0 else table_caliber_effective)
                    or table_caliber_effective
                )
                cell_dict["accountingYear"] = (
                    col_year.get(col_key) if col_key > 0 else None
                )
                cell_dict["statementType"] = (
                    (col_statement_type.get(col_key) if col_key > 0 else table_statement_type)
                    or table_statement_type
                )
                cell_dict["gaapType"] = (
                    (col_gaap.get(col_key) if col_key > 0 else table_gaap)
                    or table_gaap
                )
                if "bBox" in cell_dict and cell_dict["bBox"]:
                    bbox = cell_dict["bBox"]
                    if (
                        bbox.get("b") == 0.0
                        and bbox.get("l") == 0.0
                        and bbox.get("r") == 0.0
                        and bbox.get("t") == 0.0
                    ):
                        cell_dict["bBox"] = {}
                serialized_row.append(cell_dict)
            all_rows.append(serialized_row)

    return all_rows
