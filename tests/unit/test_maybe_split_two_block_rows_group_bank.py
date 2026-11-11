"""_maybe_split_two_block_rows 不得把 Group+Bank 并列的 6 列误切成 3+3。"""

from src.applications.parsing.service.three_statements_frontend_service import _maybe_split_two_block_rows


def _cell(text: str, *, col_start: int, col_span: int = 1) -> dict:
    return {"text": text, "colStart": col_start, "colSpan": col_span}


def test_skip_two_block_split_for_group_bank_six_column_layout() -> None:
    # 表头含 group + bank；章节行在多列重复同一词，会满足「左右均为科目」启发式。
    rows = [
        [
            _cell("", col_start=0),
            _cell("Note", col_start=1),
            _cell("Group", col_start=2),
            _cell("2023", col_start=3),
            _cell("Bank", col_start=4),
            _cell("2023", col_start=5),
        ],
        [
            _cell("Cash", col_start=0),
            _cell("5", col_start=1),
            _cell("40m", col_start=2),
            _cell("28m", col_start=3),
            _cell("39m", col_start=4),
            _cell("29m", col_start=5),
        ],
        [
            _cell("Assets", col_start=0),
            _cell("Assets", col_start=1),
            _cell("Assets", col_start=2),
            _cell("Assets", col_start=3),
            _cell("Assets", col_start=4),
            _cell("Assets", col_start=5),
        ],
    ]
    out = _maybe_split_two_block_rows(rows)
    assert out is rows
    assert len(out) == 3
    assert len(out[1]) == 6


if __name__ == "__main__":
    test_skip_two_block_split_for_group_bank_six_column_layout()
    print("ok")
