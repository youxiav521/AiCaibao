"""SP Setia 等左右对开 BS 合并时，章节标题行（EQUITY AND LIABILITIES）不得被当作列头跳掉。"""

from src.applications.parsing.service.three_statements_frontend_service import (
    _detect_merge_header_skip,
    _merge_balance_sheet_split_candidates,
)


def _row(*texts: str) -> list[dict]:
    return [{"text": t, "rowSpan": 1, "colSpan": 1} for t in texts]


def test_detect_skip_only_real_header_row_for_sp_setia_bs() -> None:
    r1 = [
        _row("", "Group/2024RM'000", "Group/2023RM'000", "Company/2024RM'000", "Company/2023RM'000"),
        _row("ASSETS", "", "", "", ""),
        _row("Property, plant and equipment", "751,792", "808,271", "4,033", "3,938"),
    ]
    r2 = [
        _row("", "Group/2024RM'000", "Group/2023RM'000", "Company/2024RM'000", "Company/2023RM'000"),
        _row("EQUITY AND LIABILITIES", "", "", "", ""),
        _row("Equity", "", "", "", ""),
        _row("Share capital", "9,238,060", "8,722,775", "9,238,060", "8,722,775"),
    ]
    assert _detect_merge_header_skip(r1, r2) == 1


def test_detect_skip_two_rows_when_dual_row_column_header() -> None:
    r1 = [
        _row("", "Group", "Group", "Company", "Company"),
        _row("", "2024", "2023", "2024", "2023"),
        _row("ASSETS", "", "", "", ""),
    ]
    r2 = [
        _row("", "Group", "Group", "Company", "Company"),
        _row("", "2024", "2023", "2024", "2023"),
        _row("EQUITY AND LIABILITIES", "", "", "", ""),
    ]
    assert _detect_merge_header_skip(r1, r2) == 2


def test_merge_preserves_section_title_for_sp_setia_bs() -> None:
    t1 = {
        "rows": [
            _row("", "Group/2024RM'000", "Group/2023RM'000"),
            _row("ASSETS", "", ""),
            _row("Property, plant and equipment", "751,792", "808,271"),
            _row("TOTAL ASSETS", "27,603,810", "29,006,378"),
        ],
        "md": "",
    }
    t2 = {
        "rows": [
            _row("", "Group/2024RM'000", "Group/2023RM'000"),
            _row("EQUITY AND LIABILITIES", "", ""),
            _row("Equity", "", ""),
            _row("Share capital", "9,238,060", "8,722,775"),
            _row("TOTAL EQUITY AND LIABILITIES", "27,603,810", "29,006,378"),
        ],
        "md": "",
    }
    prev_cand = {
        "label": "合并资产负债表",
        "table": t1,
        "expanded_idx": 0,
        "page": 1,
        "continuation_indices": [],
    }
    next_cand = {
        "label": "合并资产负债表",
        "table": t2,
        "expanded_idx": 1,
        "page": 1,
    }
    out = _merge_balance_sheet_split_candidates(prev_cand, next_cand, locale=None)
    rows = out["table"]["rows"]
    texts = [
        (r[0].get("text") if r and isinstance(r[0], dict) else "")
        for r in rows
    ]
    assert "EQUITY AND LIABILITIES" in texts, texts
    assert "ASSETS" in texts
    assert "TOTAL ASSETS" in texts
    assert "TOTAL EQUITY AND LIABILITIES" in texts
    i_total_assets = texts.index("TOTAL ASSETS")
    i_eql = texts.index("EQUITY AND LIABILITIES")
    assert i_total_assets < i_eql, (
        "EQUITY AND LIABILITIES 应在 TOTAL ASSETS 之后；当前顺序：" + str(texts)
    )


if __name__ == "__main__":
    test_detect_skip_only_real_header_row_for_sp_setia_bs()
    test_detect_skip_two_rows_when_dual_row_column_header()
    test_merge_preserves_section_title_for_sp_setia_bs()
    print("ok")
