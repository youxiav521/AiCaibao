from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.mineru_stability_profile import build_stability_profile


def run() -> None:
    baseline = [
        {"type": "text", "source_page_no": 1, "text": "Revenue 1,234"},
        {"type": "table", "source_page_no": 1, "rows": [[{"text": "Cash 100"}]]},
    ]
    split_variant = [
        {"type": "text", "source_page_no": 1, "text": "Revenue"},
        {"type": "text", "source_page_no": 1, "text": "1,234"},
        {"type": "table", "source_page_no": 1, "rows": [[{"text": "Cash 100"}]]},
    ]
    profile = build_stability_profile(
        [baseline, baseline, split_variant],
        compare_config={"ignore_fields": [], "mineru_assessment": {}},
    )
    assert len(profile["comparisons"]) == 2
    assert "min_text_coverage" in profile["threshold_recommendation"]
    assert profile["observed_range"]["text_coverage"]["min"] >= 0.9


if __name__ == "__main__":
    run()
    logger.info("MinerU stability profile checks passed")
