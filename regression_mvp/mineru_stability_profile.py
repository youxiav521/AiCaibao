from __future__ import annotations

"""Collect repeated MinerU runs before changing regression thresholds.

Edit the configuration block and run this file from PyCharm, or pass
--endpoint / --case-name / --runs on the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.case_loader import RegressionCase, load_enabled_cases
from regression_mvp.utils.comparator import compare_items
from regression_mvp.utils.json_io import dump_json, load_json
from regression_mvp.utils.metrics import run_id_timestamp
from regression_mvp.utils.mineru_assessment import MINERU_ENDPOINTS, assess_mineru_regression
from regression_mvp.utils.normalizer import normalize_items
from regression_mvp.utils.runners.mineru_to_json_runner import run_mineru_to_json
from regression_mvp.utils.runners.mineru_upload_async_runner import run_mineru_upload_async
from regression_mvp.utils.runners.ade_jobs_async_runner import run_ade_jobs_async


# PyCharm entry configuration. Set CASE_NAME before clicking Run.
ENDPOINT = "mineru_to_json"
CASE_NAME = ""
RUNS = 3

ROOT_DIR = Path(__file__).resolve().parent


def main() -> int:
    args = _parse_args()
    endpoint = args.endpoint or ENDPOINT
    case_name = args.case_name or CASE_NAME
    run_count = args.runs or RUNS
    if endpoint not in MINERU_ENDPOINTS:
        raise ValueError(f"stability sampling supports only: {sorted(MINERU_ENDPOINTS)}")
    if not case_name:
        raise ValueError("set CASE_NAME at the top of mineru_stability_profile.py before running")
    if run_count < 2:
        raise ValueError("runs must be at least 2")

    config = load_json(ROOT_DIR / "config.json")
    case = _find_case(config, endpoint, case_name)
    output_dir = ROOT_DIR / config["paths"]["report_dir"] / "stability_profiles" / endpoint / case.case_name / run_id_timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = _runner(endpoint)
    outputs: list[list[Any]] = []
    for index in range(1, run_count + 1):
        result = runner(config["api"], case.input_file)
        actual_items = result.actual_items
        outputs.append(actual_items)
        dump_json(output_dir / f"run_{index:02d}.items.json", actual_items)

    profile = build_stability_profile(
        outputs,
        compare_config=config["compare"],
        input_file=case.input_file,
    )
    profile.update(
        {
            "endpoint": endpoint,
            "case_name": case.case_name,
            "input_file": str(case.input_file),
            "run_count": run_count,
            "baseline": "run_01.items.json",
        }
    )
    dump_json(output_dir / "stability_profile.json", profile)
    logger.info("MinerU stability profile written | path={}", output_dir / "stability_profile.json")
    return 0


def build_stability_profile(
    outputs: list[list[Any]],
    *,
    compare_config: dict[str, Any],
    input_file: Path | None = None,
) -> dict[str, Any]:
    """Compare repeated outputs to the first run and suggest, never apply, limits."""
    if len(outputs) < 2:
        raise ValueError("at least two outputs are required")

    baseline = outputs[0]
    normalized_baseline = normalize_items(baseline, compare_config)
    comparisons: list[dict[str, Any]] = []
    for index, actual in enumerate(outputs[1:], start=2):
        diff = compare_items(normalized_baseline, normalize_items(actual, compare_config))
        assessment = assess_mineru_regression(
            baseline,
            actual,
            diff.diffs,
            input_file=input_file,
            profile=compare_config.get("mineru_assessment"),
        )
        content = assessment["content"]
        structure = assessment["structure"]
        comparisons.append(
            {
                "run": index,
                "status": assessment["status"],
                "raw_diff_count": len(diff.diffs),
                "text_coverage": content["text_coverage"],
                "table_text_coverage": content["table_text_coverage"],
                "number_coverage": content["number_coverage"],
                "added_text_ratio": content["added_text_ratio"],
                "item_delta_ratio": structure["item_delta_ratio"],
                "text_split_merge_count": structure["text_split_merge_count"],
                "bbox_change_count": structure["bbox_change_count"],
            }
        )
    return {
        "version": "mineru_stability_profile_v1",
        "comparisons": comparisons,
        "observed_range": _observed_range(comparisons),
        "threshold_recommendation": _threshold_recommendation(comparisons),
        "note": "Recommendations are observations from repeated runs, not automatic config changes.",
    }


def _observed_range(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = [
        "text_coverage",
        "table_text_coverage",
        "number_coverage",
        "added_text_ratio",
        "item_delta_ratio",
    ]
    return {
        field: {
            "min": round(min(float(item[field]) for item in comparisons), 4),
            "max": round(max(float(item[field]) for item in comparisons), 4),
        }
        for field in fields
    }


def _threshold_recommendation(comparisons: list[dict[str, Any]]) -> dict[str, float]:
    observed = _observed_range(comparisons)
    return {
        "min_text_coverage": _lower_bound(observed["text_coverage"]["min"], 0.005),
        "min_table_text_coverage": _lower_bound(observed["table_text_coverage"]["min"], 0.005),
        "min_number_coverage": _lower_bound(observed["number_coverage"]["min"], 0.001),
        "max_added_text_ratio": _upper_bound(observed["added_text_ratio"]["max"], 0.005),
        "max_item_delta_ratio_observed": observed["item_delta_ratio"]["max"],
    }


def _lower_bound(value: float, margin: float) -> float:
    return round(max(0.0, float(value) - margin), 4)


def _upper_bound(value: float, margin: float) -> float:
    return round(min(1.0, float(value) + margin), 4)


def _runner(endpoint: str) -> Callable[[dict[str, Any], Path], Any]:
    if endpoint == "mineru_to_json":
        return run_mineru_to_json
    if endpoint == "ade_jobs_async":
        return run_ade_jobs_async
    return run_mineru_upload_async


def _find_case(config: dict[str, Any], endpoint: str, case_name: str) -> RegressionCase:
    for case in load_enabled_cases(ROOT_DIR, config, endpoint_type=endpoint):
        if case.case_name == case_name:
            return case
    raise ValueError(f"enabled {endpoint} case not found: {case_name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile repeated MinerU output stability for one case")
    parser.add_argument("--endpoint", choices=sorted(MINERU_ENDPOINTS))
    parser.add_argument("--case-name")
    parser.add_argument("--runs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
