from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
KG_EVALUATION_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.json_utils import dump, load  # noqa: E402


DEFAULT_INPUT = KG_EVALUATION_DIR / "test_folder.json"
DEFAULT_OUTPUT = KG_EVALUATION_DIR / "test_folder_structured.json"

TOP_LEVEL_KEYS = ("id", "projectId", "projectName", "files", "entities", "relations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize kg_evaluation_test/test_folder.json to the kg_test/*.json schema.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source JSON path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Structured JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--keep-embeddings",
        action="store_true",
        help="Keep entities[].embed vectors instead of setting them to null.",
    )
    parser.add_argument(
        "--escape-non-ascii",
        action="store_true",
        help="Escape non-ASCII characters to avoid editor ambiguous Unicode warnings.",
    )
    return parser.parse_args()


def require_mapping(payload: Any, input_path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object: {input_path}")
    return payload


def require_top_level_keys(payload: dict[str, Any]) -> None:
    missing = [key for key in ("files", "entities", "relations") if key not in payload]
    if missing:
        raise ValueError(f"Missing required top-level keys: {', '.join(missing)}")


def normalize_entities(
    entities: Any,
    *,
    keep_embeddings: bool,
) -> tuple[Any, int]:
    if not isinstance(entities, list):
        raise ValueError("Expected 'entities' to be a list")

    cleared_embeddings = 0
    normalized_entities = []
    for entity in entities:
        if not isinstance(entity, dict):
            normalized_entities.append(entity)
            continue

        normalized_entity = dict(entity)
        if not keep_embeddings and normalized_entity.get("embed") is not None:
            normalized_entity["embed"] = None
            cleared_embeddings += 1
        normalized_entities.append(normalized_entity)

    return normalized_entities, cleared_embeddings


def normalize_payload(
    payload: dict[str, Any],
    *,
    keep_embeddings: bool,
) -> tuple[dict[str, Any], int]:
    require_top_level_keys(payload)

    project_id = payload.get("projectId") or payload.get("folderId")
    project_name = payload.get("projectName") or payload.get("folderName")
    if not project_id:
        raise ValueError("Missing projectId/folderId")
    if not project_name:
        raise ValueError("Missing projectName/folderName")

    entities, cleared_embeddings = normalize_entities(
        payload["entities"],
        keep_embeddings=keep_embeddings,
    )

    normalized = {
        "id": payload.get("id"),
        "projectId": project_id,
        "projectName": project_name,
        "files": payload["files"],
        "entities": entities,
        "relations": payload["relations"],
    }

    for key, value in payload.items():
        if key not in TOP_LEVEL_KEYS and key not in ("folderId", "folderName"):
            normalized[key] = value

    return normalized, cleared_embeddings


def write_json(payload: dict[str, Any], output_path: Path, *, escape_non_ascii: bool) -> None:
    if escape_non_ascii:
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return

    dump(payload, output_path, indent=2)


def main() -> None:
    args = parse_args()

    payload = require_mapping(load(args.input), args.input)
    normalized, cleared_embeddings = normalize_payload(
        payload,
        keep_embeddings=args.keep_embeddings,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(normalized, args.output, escape_non_ascii=args.escape_non_ascii)

    print(
        "Structured JSON written: "
        f"{args.output} | files={len(normalized['files'])}, "
        f"entities={len(normalized['entities'])}, "
        f"relations={len(normalized['relations'])}, "
        f"cleared_embeddings={cleared_embeddings}, "
        f"escape_non_ascii={args.escape_non_ascii}"
    )


if __name__ == "__main__":
    main()
