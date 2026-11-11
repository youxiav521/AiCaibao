from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.report_store import rebuild_store, write_dashboard
from regression_mvp.utils.reporter import rebuild_reports_from_existing, write_html_reports_from_existing


ROOT_DIR = Path(__file__).resolve().parent
REPORT_ROOT = ROOT_DIR / "reports"
CONFIG_PATH = ROOT_DIR / "config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regression SQLite history and dashboard.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild SQLite history from existing reports/*/runs/*/report.json files.",
    )
    parser.add_argument(
        "--recompare",
        action="store_true",
        help="Rebuild historical report diffs and artifacts from saved expected/actual JSON without calling APIs.",
    )
    args = parser.parse_args()

    if args.recompare:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        print(json.dumps(rebuild_reports_from_existing(REPORT_ROOT, config["compare"]), ensure_ascii=False))
    elif args.rebuild:
        dashboard_path = rebuild_store(REPORT_ROOT)
        write_html_reports_from_existing(REPORT_ROOT)
        print(dashboard_path)
    else:
        dashboard_path = write_dashboard(REPORT_ROOT)
        print(dashboard_path)


if __name__ == "__main__":
    main()
