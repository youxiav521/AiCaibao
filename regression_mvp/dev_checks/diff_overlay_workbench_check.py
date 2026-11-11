from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.artifact_renderer import _overlay_html


def _table_item(cell_text: str) -> dict[str, object]:
    return {
        "type": "table",
        "page": 1,
        "bBox": {"l": 20, "t": 30, "r": 300, "b": 220},
        "rows": [[{"text": "项目"}], [{"text": cell_text}]],
    }


def _text_item(page: int, text: str) -> dict[str, object]:
    return {
        "type": "text",
        "page": page,
        "bBox": {"l": 30, "t": 40, "r": 360, "b": 90},
        "text": text,
    }


def main() -> None:
    expected_text = "预期表格单元格完整内容 " * 20
    actual_text = "实际表格单元格完整内容 " * 20
    html_output = _overlay_html(
        [
            {"page": 1, "pdf_width": 600, "pdf_height": 800, "image_width": 1200, "image_height": 1600, "image_path": "pages/page_001.png"},
            {"page": 2, "pdf_width": 600, "pdf_height": 800, "image_width": 1200, "image_height": 1600, "image_path": "pages/page_002.png"},
        ],
        [_table_item(actual_text)],
        title="差异定位工作台检查",
        highlight_indexes={1},
        comparison_items=[_table_item(expected_text), _text_item(2, "预期存在但本次缺失的文本")],
        diff_detail=[
            {
                "path": "items[0~0].rows[1~1][0~0].text",
                "type": "text_changed",
                "severity": "major",
                "expected": expected_text,
                "actual": actual_text,
            },
            {
                "path": "items[1]",
                "type": "removed_item",
                "severity": "major",
                "expected": "预期存在但本次缺失的文本",
                "actual": "<missing>",
            },
        ],
        ignore_rule_url="/cases/mineru_to_json/demo/compare-exclusions/new?run_id=demo_run",
    )

    assert 'id="page-1"' in html_output
    assert 'id="page-2"' in html_output
    assert 'id="diff-detail-actual-1"' in html_output
    assert 'id="diff-detail-expected_missing-2"' in html_output
    assert 'id="diff-item-expected_missing-2"' in html_output
    assert 'severity-group major' in html_output
    assert "受影响表格行对照" in html_output
    assert expected_text in html_output
    assert actual_text in html_output
    assert 'data-diff-index="actual-1"' in html_output
    assert "将此项排除出比较" in html_output
    assert "compare-exclusions/new?run_id=demo_run&amp;page=1" in html_output

    coordinate_html = _overlay_html(
        [{"page": 1, "pdf_width": 600, "pdf_height": 800, "image_width": 1200, "image_height": 1600, "image_path": "pages/page_001.png"}],
        [_table_item(actual_text), _text_item(1, "用于坐标预览右侧详情的完整文本")],
        title="坐标预览检查",
    )
    assert 'class="coordinate-inspector"' in coordinate_html
    assert 'id="coordinate-detail-1"' in coordinate_html
    assert 'data-coordinate-index="1"' in coordinate_html
    assert "解析表格（2 行）" in coordinate_html
    assert "用于坐标预览右侧详情的完整文本" in coordinate_html
    logger.success("差异定位工作台渲染检查通过")


if __name__ == "__main__":
    main()
