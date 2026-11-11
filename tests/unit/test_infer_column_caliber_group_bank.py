"""Group/Bank 双列头下列口径应拆成合并 + 单体（马来银行常见版式）。"""

import json
from pathlib import Path

from src.applications.parsing.utils.financial_recognition_and_extraction_utils import (
    infer_column_index_to_caliber_for_table,
)


def test_infer_caliber_maybank_group_bank_columns() -> None:
    root = Path(__file__).resolve().parents[2]
    p = root / (
        "storage/mineru_output/converted_output/"
        "a68779b27e4d4c4e9bedd32bc7fc039cuVqI5FwICz_mineru_converted.json"
    )
    assert p.is_file(), "fixture MinerU JSON missing: {}".format(p)
    items = json.loads(p.read_text(encoding="utf-8"))
    el3 = next(x for x in items if isinstance(x, dict) and x.get("element_id") == 3)
    rows = el3["rows"]
    m = infer_column_index_to_caliber_for_table(
        rows, table_caliber="未知", report_name="资产负债表"
    )
    assert m.get(2) == "合并"
    assert m.get(3) == "合并"
    assert m.get(4) == "单体"
    assert m.get(5) == "单体"


if __name__ == "__main__":
    test_infer_caliber_maybank_group_bank_columns()
    print("ok")
