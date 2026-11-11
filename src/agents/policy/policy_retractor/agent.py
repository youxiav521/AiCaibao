import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from .extractor import ExtractorConfig, PolicyExtractor
    from .preprocess import PolicyDocumentLoader
    from .schemas import Framework
except ImportError:  # pragma: no cover
    from extractor import ExtractorConfig, PolicyExtractor
    from preprocess import PolicyDocumentLoader
    from schemas import Framework

from src.utils.model_factory import ModelFactory, Config


class PolicyRetractorAgent:
    """High-level entry point for policy framework extraction."""

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        requirement_callback: Optional[Callable[[Any], Any]] = None,
    ):
        """Create loader + extractor with optional extractor config."""
        self.loader = PolicyDocumentLoader()
        self.extractor = PolicyExtractor(
            config=config or ExtractorConfig(),
            requirement_callback=requirement_callback,
        )

    def run(self, input_path: str | Path) -> Framework:
        """Run extraction for one input path and return framework model."""
        document = self.loader.load(input_path)
        return self.extractor.extract(document)

    def run_payload(self, payload: dict) -> Framework:
        """Run extraction for one in-memory Java payload."""
        document = self.loader.load_payload(payload)
        return self.extractor.extract(document)

    def run_to_file(self, input_path: str | Path, output_path: str | Path) -> Framework:
        """Run extraction and persist framework json to output_path."""
        framework = self.run(input_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(framework.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return framework

    def run_payload_to_file(self, payload: dict, output_path: str | Path) -> Framework:
        """Run extraction for one in-memory payload and persist framework json."""
        framework = self.run_payload(payload)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(framework.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return framework


def default_output_path(input_path: str | Path) -> Path:
    """Build default output path under current module directory."""
    base_dir = Path(__file__).resolve().parent
    source_name = Path(input_path).stem if Path(input_path).suffix else Path(input_path).name
    safe_name = source_name.replace(".", "_").replace("-", "_")
    return base_dir / f"{safe_name}_framework_output.json"


def default_metrics_log_path() -> Path:
    """Return default JSONL metrics log path."""
    return Path("storage/policy_retractor/policy_retractor_runs.jsonl")


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO8601 seconds precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_model_info(model_id: str) -> dict:
    """Resolve model metadata for run metrics logging."""
    custom_models = Config.get_custom_models()
    config = dict(custom_models.get(model_id, {}))
    base_url = config.get("base_url") or os.getenv("LLM_API_BASE", "")
    return {
        "model_id": model_id,
        "model_name": config.get("model", ""),
        "model_provider": config.get("model_provider", ""),
        "base_url": base_url,
    }


def append_metrics_log(log_path: Path, record: dict) -> None:
    """Append one metrics record to JSONL file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_controls(framework: Framework) -> int:
    """Count controls across all extracted requirements."""
    return sum(len(requirement.controls) for requirement in framework.requirements)


def main() -> None:
    """CLI entry point for policy extraction with run metrics."""
    parser = argparse.ArgumentParser(description="Extract framework/requirements/controls from policy OCR output.")
    parser.add_argument("--input", required=True, help="Path to a MinerU json file or directory.")
    parser.add_argument("--output", default=None, help="Optional output json path.")
    parser.add_argument("--enable-llm", action="store_true", help="Use LLM enhancement if model config is available.")
    parser.add_argument("--model-id", default="model_qwen35_35b", help="Structured-output model id.")
    parser.add_argument(
        "--metrics-log",
        default=str(default_metrics_log_path()),
        help="JSONL file used to append one metrics record per run.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full framework JSON to stdout after extraction.",
    )
    parser.add_argument(
        "--max-section-concurrency",
        type=int,
        default=16,
        help="Maximum number of section extraction requests sent concurrently.",
    )
    parser.add_argument(
        "--auto-section-concurrency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto scale section concurrency based on section count.",
    )
    parser.add_argument(
        "--stable-max-section-concurrency",
        type=int,
        default=16,
        help="Stable max concurrency used in auto mode: effective=min(section_count, stable_max, hard_cap).",
    )
    parser.add_argument(
        "--max-chars-per-section",
        type=int,
        default=12000,
        help="Soft limit of chars per section window before splitting.",
    )
    parser.add_argument(
        "--section-overlap-chunks",
        type=int,
        default=1,
        help="Number of overlapped chunks between neighboring windows.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else default_output_path(args.input)
    metrics_log_path = Path(args.metrics_log)
    started_at = utc_now_iso()
    started_perf = time.perf_counter()
    model_info = resolve_model_info(args.model_id)

    record = {
        "started_at": started_at,
        "input_path": str(Path(args.input).resolve()),
        "output_path": str(output_path.resolve()),
        "enable_llm": bool(args.enable_llm),
        "max_section_concurrency": max(1, args.max_section_concurrency),
        "auto_section_concurrency": bool(args.auto_section_concurrency),
        "stable_max_section_concurrency": max(1, args.stable_max_section_concurrency),
        "max_chars_per_section": max(500, args.max_chars_per_section),
        "section_overlap_chunks": max(0, args.section_overlap_chunks),
        "metrics_log_path": str(metrics_log_path.resolve()),
        **model_info,
    }

    framework: Optional[Framework] = None
    try:
        agent = PolicyRetractorAgent(
            config=ExtractorConfig(
                enable_llm=args.enable_llm,
                model_id=args.model_id,
                max_section_concurrency=max(1, args.max_section_concurrency),
                auto_section_concurrency=bool(args.auto_section_concurrency),
                stable_max_section_concurrency=max(1, args.stable_max_section_concurrency),
                max_chars_per_section=max(500, args.max_chars_per_section),
                section_overlap_chunks=max(0, args.section_overlap_chunks),
            )
        )
        framework = agent.run_to_file(args.input, output_path)
        elapsed_seconds = round(time.perf_counter() - started_perf, 2)
        record.update(
            {
                "finished_at": utc_now_iso(),
                "elapsed_seconds": elapsed_seconds,
                "result": "success",
                "framework_status": framework.status,
                "framework_id": framework.framework_id,
                "file_id": framework.file_id,
                "requirements_count": len(framework.requirements),
                "controls_count": count_controls(framework),
                "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
            }
        )
        append_metrics_log(metrics_log_path, record)
        print(f"Saved output to: {output_path}")
        print(f"Metrics logged to: {metrics_log_path}")
        print(
            "Run summary: "
            f"model_id={args.model_id}, model_name={model_info['model_name']}, "
            f"elapsed_seconds={elapsed_seconds}, requirements={len(framework.requirements)}, "
            f"controls={count_controls(framework)}, status={framework.status}"
        )
        if args.print_json:
            print(json.dumps(framework.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2))
    except Exception as exc:
        elapsed_seconds = round(time.perf_counter() - started_perf, 2)
        record.update(
            {
                "finished_at": utc_now_iso(),
                "elapsed_seconds": elapsed_seconds,
                "result": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        append_metrics_log(metrics_log_path, record)
        raise


if __name__ == "__main__":
    main()
