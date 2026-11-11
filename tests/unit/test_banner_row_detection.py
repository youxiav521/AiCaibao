"""``_is_banner_row`` 专项单元测试：保证章节横幅行不会污染列表头文本。"""

from __future__ import annotations

from src.applications.parsing.utils.financial_recognition_and_extraction_utils import (
    _is_banner_row,
    _infer_column_header_texts,
)


def _cell(text: str, *, col_start: int = 0, col_span: int = 1) -> dict:
    return {"text": text, "colstart": col_start, "colspan": col_span, "rowstart": 0, "rowspan": 1}


def test_single_cell_with_colspan_ge_2_is_banner() -> None:
    row = [_cell("STATEMENT OF COMPREHENSIVE INCOME CONSOLIDATED AND SEPARATE",
                 col_start=0, col_span=5)]
    assert _is_banner_row(row) is True


def test_single_cell_with_colspan_1_is_not_banner() -> None:
    row = [_cell("Cash and equivalents", col_start=0, col_span=1)]
    assert _is_banner_row(row) is False


def test_repeated_text_across_cells_is_banner() -> None:
    row = [
        _cell("EQUITY AND LIABILITIES", col_start=0),
        _cell("EQUITY AND LIABILITIES", col_start=1),
        _cell("EQUITY AND LIABILITIES", col_start=2),
    ]
    assert _is_banner_row(row) is True


def test_distinct_cells_not_banner() -> None:
    row = [
        _cell("", col_start=0),
        _cell("Consolidated 2024", col_start=1),
        _cell("Standalone 2024", col_start=2),
    ]
    assert _is_banner_row(row) is False


def test_empty_row_not_banner() -> None:
    assert _is_banner_row([]) is False
    assert _is_banner_row([_cell("", col_start=0)]) is False


def test_header_extraction_skips_banner_row_with_colspan() -> None:
    """章节 banner 行（单 cell colspan=5）必须被跳过，不能把 banner 文本注入到所有列。"""
    rows = [
        [
            _cell("Statement of comprehensive income - Net of tax", col_start=0),
            _cell("Consolidated01/01/2022-31/12/2022", col_start=1),
            _cell("Standalone01/01/2022-31/12/2022", col_start=2),
            _cell("Consolidated01/01/2021-31/12/2021", col_start=3),
            _cell("Standalone01/01/2021-31/12/2021", col_start=4),
        ],
        [
            _cell("STATEMENT OF COMPREHENSIVE INCOME CONSOLIDATED AND SEPARATE",
                  col_start=0, col_span=5)
        ],
        [
            _cell("Net Profit / (Loss) for the Period", col_start=0),
            _cell("19,173", col_start=1),
            _cell("18,543", col_start=2),
            _cell("12,651", col_start=3),
            _cell("12,457", col_start=4),
        ],
    ]
    headers = _infer_column_header_texts(rows)
    assert "STATEMENT OF COMPREHENSIVE INCOME CONSOLIDATED AND SEPARATE" not in headers.get(2, "")
    assert "STATEMENT OF COMPREHENSIVE INCOME CONSOLIDATED AND SEPARATE" not in headers.get(4, "")
    # Standalone 列原始头不应被破坏（数据行 row 2 也会被纳入前 3 行表头窗口）。
    assert "Standalone" in headers.get(2, "")
    assert "Standalone" in headers.get(4, "")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
