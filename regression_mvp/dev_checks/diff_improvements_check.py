from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.comparator import compare_items
from regression_mvp.utils.metrics import diff_metrics
from regression_mvp.utils.normalizer import normalize_items


def check_normalizer_cleans_unicode_whitespace_and_dynamic_fields() -> None:
    value = {
        "file_id": "dynamic",
        "text": "\uff21  \u200b\u201chello\u201d\nworld",
        "nested": {"created_at": "dynamic", "value": None},
    }

    normalized = normalize_items(value, {"ignore_fields": []})

    assert "file_id" not in normalized
    assert normalized["text"] == 'A  "hello"\nworld'
    assert "created_at" not in normalized["nested"]
    assert normalized["nested"]["value"] == ""


def check_compare_items_matches_inserted_mineru_item_without_cascade_diff() -> None:
    expected = [
        {"type": "text", "page": 1, "text": "A", "bBox": {"l": 0, "t": 0, "r": 10, "b": 10}},
        {"type": "text", "page": 1, "text": "B", "bBox": {"l": 0, "t": 20, "r": 10, "b": 30}},
        {"type": "text", "page": 1, "text": "C", "bBox": {"l": 0, "t": 40, "r": 10, "b": 50}},
    ]
    actual = [
        {"type": "text", "page": 1, "text": "A", "bBox": {"l": 0, "t": 0, "r": 10, "b": 10}},
        {"type": "text", "page": 1, "text": "X", "bBox": {"l": 0, "t": 10, "r": 10, "b": 18}},
        {"type": "text", "page": 1, "text": "B", "bBox": {"l": 0, "t": 20, "r": 10, "b": 30}},
        {"type": "text", "page": 1, "text": "C", "bBox": {"l": 0, "t": 40, "r": 10, "b": 50}},
    ]

    result = compare_items(expected, actual)

    assert not result.passed
    assert [diff["type"] for diff in result.diffs] == ["length_changed", "added_item"]
    assert result.diffs[1]["actual"]["text"] == "X"


def check_compare_items_matches_local_text_split_without_major_diff() -> None:
    expected = [
        {"type": "text", "page": 1, "value": "Total current assets and liabilities"},
    ]
    actual = [
        {"type": "text", "page": 1, "value": "Total current assets"},
        {"type": "text", "page": 1, "value": " and liabilities"},
    ]

    result = compare_items(expected, actual)

    assert [diff["type"] for diff in result.diffs] == ["text_split_merged"]
    assert result.diffs[0]["severity"] == "minor"


def check_compare_items_allows_small_bbox_drift() -> None:
    expected = [{"type": "text", "page": 1, "text": "A", "bBox": {"l": 1.0, "t": 2.0, "r": 10.0, "b": 12.0}}]
    actual = [{"type": "text", "page": 1, "text": "A", "bBox": {"l": 2.5, "t": 3.5, "r": 11.5, "b": 13.5}}]

    result = compare_items(expected, actual)

    assert result.passed


def check_compare_items_marks_larger_bbox_drift_minor() -> None:
    expected = [{"type": "text", "page": 1, "value": "A", "bBox": {"l": 76.0, "t": 2.0, "r": 10.0, "b": 12.0}}]
    actual = [{"type": "text", "page": 1, "value": "A", "bBox": {"l": 82.0, "t": 2.0, "r": 10.0, "b": 12.0}}]

    result = compare_items(expected, actual)

    assert result.diffs == [{"path": "items[0~0].bBox.l", "expected": 76.0, "actual": 82.0, "type": "bbox_changed", "severity": "minor"}]


def check_compare_items_marks_whitespace_only_text_change_minor() -> None:
    expected = [{"type": "text", "page": 1, "value": "2025 年"}]
    actual = [{"type": "text", "page": 1, "value": "2025年"}]

    result = compare_items(expected, actual)

    assert result.diffs[0]["type"] == "text_whitespace_changed"
    assert result.diffs[0]["severity"] == "minor"


def check_compare_items_marks_image_description_once_as_minor() -> None:
    expected = [{"type": "image", "page": 1, "md": "a blue chart", "value": "a blue chart"}]
    actual = [{"type": "image", "page": 1, "md": "a green chart", "value": "a green chart"}]

    result = compare_items(expected, actual)

    assert result.diffs == [{"path": "items[0~0].image_description", "expected": "a blue chart", "actual": "a green chart", "type": "image_description_changed", "severity": "minor"}]


def check_compare_items_uses_value_without_duplicate_md_diff() -> None:
    expected = [{"type": "text", "page": 1, "value": "Revenue", "md": "Revenue", "bBox": {"l": 0, "t": 0, "r": 10, "b": 10}}]
    actual = [{"type": "text", "page": 1, "value": "Income", "md": "Income", "bBox": {"l": 0, "t": 0, "r": 10, "b": 10}}]

    result = compare_items(expected, actual)

    assert len(result.diffs) == 1
    assert result.diffs[0]["path"] == "items[0~0].value"
    assert result.diffs[0]["severity"] == "major"


def check_compare_items_skips_derived_table_fields() -> None:
    expected = [{"type": "table", "page": 1, "rows": [[{"text": "Revenue"}]], "html": "<table>old</table>", "hasHeader": False, "headerCells": []}]
    actual = [{"type": "table", "page": 1, "rows": [[{"text": "Revenue"}]], "html": "<table>new</table>", "hasHeader": True, "headerCells": [[{"text": "Revenue"}]]}]

    result = compare_items(expected, actual)

    assert result.passed


def check_compare_items_marks_secondary_metadata_and_missing_bbox_minor() -> None:
    expected = [{"type": "text", "page": 1, "value": "Revenue", "headingLevel": 2, "bBox": {"l": 0, "t": 0, "r": 10, "b": 10}}]
    actual = [{"type": "text", "page": 1, "value": "Revenue", "headingLevel": 3}]

    result = compare_items(expected, actual)

    assert {diff["type"] for diff in result.diffs} == {"metadata_changed", "bbox_changed"}
    assert {diff["severity"] for diff in result.diffs} == {"minor"}


def check_compare_items_allows_small_score_drift() -> None:
    expected = [{"type": "text", "page": 1, "text": "A", "score": 0.9812}]
    actual = [{"type": "text", "page": 1, "text": "A", "score": 0.9818}]

    result = compare_items(expected, actual)

    assert result.passed


def check_compare_items_marks_large_score_drift_minor() -> None:
    expected = [{"type": "text", "page": 1, "text": "A", "confidence": 0.98}]
    actual = [{"type": "text", "page": 1, "text": "A", "confidence": 0.95}]

    result = compare_items(expected, actual)

    assert not result.passed
    assert result.diffs[0]["type"] == "float_changed"
    assert result.diffs[0]["severity"] == "minor"
    assert "1 minor" in result.summary
    assert diff_metrics(result.diffs)["by_severity"] == {"minor": 1}


def check_compare_items_marks_small_text_change_minor() -> None:
    expected = [{"type": "text", "page": 1, "text": "Total current assets"}]
    actual = [{"type": "text", "page": 1, "text": "Total current asset"}]

    result = compare_items(expected, actual)

    assert not result.passed
    assert result.diffs[0]["type"] == "text_changed"
    assert result.diffs[0]["severity"] == "minor"


def check_compare_items_marks_large_text_change_major() -> None:
    expected = [{"element_id": "same-item", "type": "text", "page": 1, "text": "Total current assets"}]
    actual = [{"element_id": "same-item", "type": "text", "page": 1, "text": "Revenue"}]

    result = compare_items(expected, actual)

    assert not result.passed
    assert result.diffs[0]["type"] == "text_changed"
    assert result.diffs[0]["severity"] == "major"


def check_compare_items_matches_financial_tables_by_identity() -> None:
    expected = [
        {"category": "balance_sheet", "text": "A", "pageAt": 1, "rows": [[{"text": "asset", "rowStart": 0}]]},
        {"category": "income_statement", "text": "B", "pageAt": 2, "rows": [[{"text": "profit", "rowStart": 0}]]},
    ]
    actual = [
        {"category": "income_statement", "text": "B", "pageAt": 2, "rows": [[{"text": "profit", "rowStart": 1}]]},
        {"category": "balance_sheet", "text": "A", "pageAt": 1, "rows": [[{"text": "asset", "rowStart": 1}]]},
    ]

    result = compare_items(expected, actual)

    assert result.passed


def check_compare_items_matches_shifted_table_rows_without_cascade_diff() -> None:
    expected = [
        {
            "category": "balance_sheet",
            "text": "A",
            "pageAt": 1,
            "rows": [
                [{"text": "Assets", "rowStart": 0}, {"text": "100", "rowStart": 0}],
                [{"text": "Liabilities", "rowStart": 1}, {"text": "80", "rowStart": 1}],
                [{"text": "Equity", "rowStart": 2}, {"text": "20", "rowStart": 2}],
            ],
        }
    ]
    actual = [
        {
            "category": "balance_sheet",
            "text": "A",
            "pageAt": 1,
            "rows": [
                [{"text": "Header", "rowStart": 0}, {"text": "", "rowStart": 0}],
                [{"text": "Assets", "rowStart": 1}, {"text": "100", "rowStart": 1}],
                [{"text": "Liabilities", "rowStart": 2}, {"text": "80", "rowStart": 2}],
                [{"text": "Equity", "rowStart": 3}, {"text": "20", "rowStart": 3}],
            ],
        }
    ]

    result = compare_items(expected, actual)

    assert not result.passed
    assert [diff["type"] for diff in result.diffs] == ["added_row"]
    assert result.diffs[0]["actual"][0]["text"] == "Header"


def check_compare_items_marks_cross_page_table_split_as_structural_drift() -> None:
    header = [{"text": "Subject", "source_page_no": 1}, {"text": "Amount", "source_page_no": 1}]
    page_one_row = [{"text": "Revenue", "source_page_no": 1}, {"text": "100", "source_page_no": 1}]
    page_two_row = [{"text": "Cost", "source_page_no": 2}, {"text": "80", "source_page_no": 2}]
    expected = [{"type": "table", "page": 1, "rows": [header, page_one_row, page_two_row]}]
    actual = [
        {"type": "table", "page": 1, "rows": [header, page_one_row]},
        {
            "type": "table",
            "page": 2,
            "rows": [
                [{"text": "Subject", "source_page_no": 2}, {"text": "Amount", "source_page_no": 2}],
                page_two_row,
            ],
        },
    ]

    result = compare_items(expected, actual)

    assert [diff["type"] for diff in result.diffs] == ["table_split_merged"]
    assert result.diffs[0]["severity"] == "minor"


def check_compare_items_downgrades_split_table_layout_changes() -> None:
    header = [{"text": "Subject", "source_page_no": 1}, {"text": "Amount", "source_page_no": 1}]
    page_one_row = [{"text": "Revenue", "source_page_no": 1}, {"text": "100", "source_page_no": 1}]
    page_two_row = [
        {"text": "Cost", "source_page_no": 2, "colspan": 1},
        {"text": "80", "source_page_no": 2, "colspan": 1},
    ]
    expected = [{"type": "table", "page": 1, "rows": [header, page_one_row, page_two_row]}]
    actual = [
        {"type": "table", "page": 1, "rows": [header, page_one_row]},
        {
            "type": "table",
            "page": 2,
            "rows": [
                [{"text": "Subject", "source_page_no": 2}, {"text": "Amount", "source_page_no": 2}],
                [
                    {"text": "Cost", "source_page_no": 2, "colspan": 2},
                    {"text": "80.0", "source_page_no": 2, "colspan": 1},
                ],
            ],
        },
    ]

    result = compare_items(expected, actual)

    layout_diffs = [diff for diff in result.diffs if diff["type"] == "table_layout_changed"]
    assert layout_diffs
    assert {diff["severity"] for diff in layout_diffs} == {"minor"}
    assert not any(diff["severity"] == "major" for diff in result.diffs)


def check_compare_items_excludes_reviewed_item_on_both_sides() -> None:
    expected = [
        {"type": "image", "page": 3, "md": "模糊图片的原始描述", "bBox": {"l": 10, "t": 20, "r": 120, "b": 180}},
        {"type": "text", "page": 3, "text": "需要继续比较的文本", "bBox": {"l": 10, "t": 220, "r": 200, "b": 250}},
    ]
    actual = [
        {"type": "image", "page": 3, "md": "本次模型生成的不同描述", "bBox": {"l": 18, "t": 27, "r": 128, "b": 187}},
        {"type": "text", "page": 3, "text": "需要继续比较的文本", "bBox": {"l": 10, "t": 220, "r": 200, "b": 250}},
    ]
    exclusions = [
        {
            "id": "exclude_blurry_image",
            "scope": "item",
            "reason": "原始图片模糊，人工确认不作为稳定基线",
            "selector": {"page": 3, "type": "image", "bbox": {"l": 10, "t": 20, "r": 120, "b": 180}},
        }
    ]

    result = compare_items(expected, actual, exclusions=exclusions)

    assert result.passed
    assert result.ignored == [
        {
            "id": "exclude_blurry_image",
            "scope": "item",
            "reason": "原始图片模糊，人工确认不作为稳定基线",
            "expected_count": 1,
            "actual_count": 1,
            "selector": exclusions[0]["selector"],
        }
    ]


def main() -> None:
    check_normalizer_cleans_unicode_whitespace_and_dynamic_fields()
    check_compare_items_matches_inserted_mineru_item_without_cascade_diff()
    check_compare_items_matches_local_text_split_without_major_diff()
    check_compare_items_allows_small_bbox_drift()
    check_compare_items_marks_larger_bbox_drift_minor()
    check_compare_items_marks_whitespace_only_text_change_minor()
    check_compare_items_marks_image_description_once_as_minor()
    check_compare_items_uses_value_without_duplicate_md_diff()
    check_compare_items_skips_derived_table_fields()
    check_compare_items_marks_secondary_metadata_and_missing_bbox_minor()
    check_compare_items_allows_small_score_drift()
    check_compare_items_marks_large_score_drift_minor()
    check_compare_items_marks_small_text_change_minor()
    check_compare_items_marks_large_text_change_major()
    check_compare_items_matches_financial_tables_by_identity()
    check_compare_items_matches_shifted_table_rows_without_cascade_diff()
    check_compare_items_marks_cross_page_table_split_as_structural_drift()
    check_compare_items_downgrades_split_table_layout_changes()
    check_compare_items_excludes_reviewed_item_on_both_sides()
    print("diff improvement checks passed")


if __name__ == "__main__":
    main()
