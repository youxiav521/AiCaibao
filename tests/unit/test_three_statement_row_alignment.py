from src.applications.parsing.service.three_statements.cells import (
    align_table_rows,
    coerce_rows,
)
from src.applications.parsing.service.three_statements.header_prepend import (
    prepend_clean_statement_header_row,
)


def _cell(
    text: str,
    col_start: int,
    *,
    col_span: int = 1,
    row_span: int = 1,
    row_start: int = 0,
) -> dict:
    return {
        "text": text,
        "colstart": col_start,
        "colspan": col_span,
        "rowspan": row_span,
        "rowstart": row_start,
    }


def test_align_table_rows_fills_missing_cells_without_breaking_rowspan() -> None:
    rows = coerce_rows(
        [
            [_cell("项目", 0), _cell("2025年", 1), _cell("2024年", 2)],
            [_cell("中间列为空", 0, row_start=1), _cell("100", 2, row_start=1)],
            [{"text": "末尾列为空"}, {"text": "200"}],
            [_cell("一、经营活动产生的现金流量：", 0, col_span=3, row_start=3)],
            [
                _cell("跨两行科目", 0, row_span=2, row_start=4),
                _cell("300", 1, row_start=4),
                _cell("400", 2, row_start=4),
            ],
            [_cell("500", 1, row_start=5), _cell("600", 2, row_start=5)],
        ],
        fallback_page_at=1,
    )

    align_table_rows(rows)

    assert [[cell["text"] for cell in row] for row in rows[:5]] == [
        ["项目", "2025年", "2024年"],
        ["中间列为空", "", "100"],
        ["末尾列为空", "200", ""],
        ["一、经营活动产生的现金流量：", "", ""],
        ["跨两行科目", "300", "400"],
    ]
    assert [cell["text"] for cell in rows[5]] == ["500", "600"]
    assert [cell["colStart"] for cell in rows[5]] == [1, 2]


def test_align_table_rows_clamps_stale_section_heading_colspan() -> None:
    rows = coerce_rows(
        [
            [_cell("项目", 0), _cell("2021年", 1), _cell("2020年", 2)],
            [_cell("税前利润", 0), _cell("15,217", 1), _cell("25,484", 2)],
            [_cell("调节项目", 0, col_span=5, row_start=2)],
        ],
        fallback_page_at=1,
    )

    align_table_rows(rows)

    assert [cell["text"] for cell in rows[2]] == ["调节项目", "", ""]
    assert [cell["colStart"] for cell in rows[2]] == [0, 1, 2]


def test_header_cleanup_preserves_sparse_amount_column_for_final_alignment() -> None:
    header = [
        _cell("项目", 0),
        _cell("附注", 1),
        _cell("2025年12月31日", 2),
        _cell("2024年12月31日", 3),
    ]
    table = {
        "type": "table",
        "hasHeader": True,
        "headerCells": [header],
    }
    rows = [
        header,
        [
            _cell("其他权益工具", 0, row_start=1),
            _cell("14,894,442", 3, row_start=1),
        ],
    ]

    cleaned = prepend_clean_statement_header_row(
        rows,
        table,
        fallback_page_at=1,
    )
    assert cleaned is not None

    coerced = coerce_rows(cleaned, fallback_page_at=1)
    align_table_rows(coerced)

    assert [cell["text"] for cell in coerced[0]] == [
        "项目",
        "2025年12月31日",
        "2024年12月31日",
    ]
    assert [cell["text"] for cell in coerced[1]] == [
        "其他权益工具",
        "",
        "14,894,442",
    ]
