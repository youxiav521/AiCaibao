from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def strip_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_none(v) for v in obj]
    return obj


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def diff_summary(a: Any, b: Any, max_items: int = 3) -> str:
    # Provide a brief, human-friendly difference summary
    if type(a) is not type(b):
        return f"Type mismatch: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, list):
        if len(a) != len(b):  # type: ignore[arg-type]
            print(f"List length: {len(a)} vs {len(b)}") ##
        for i, (ai, bi) in enumerate(zip(a, b)):
            if ai != bi:
                return f"First differing index {i}:\n ai:{ai}\n!=\nbi:{bi}"
        return "Lists appear equal but not identical (unexpected)"
    if isinstance(a, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        if keys_a != keys_b:
            only_a = sorted(keys_a - keys_b)[:max_items]
            only_b = sorted(keys_b - keys_a)[:max_items]
            return f"Key sets differ. only_in_a={only_a} only_in_b={only_b}"
        for k in sorted(keys_a):
            if a[k] != b[k]:
                return f"Key '{k}' differs: {truncate(a[k])} != {truncate(b[k])}"
        return "Dicts appear equal but not identical (unexpected)"
    return f"Values differ: {truncate(a)} != {truncate(b)}"


def truncate(v: Any, limit: int = 120) -> str:
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify two JSON files are semantically equivalent")
    parser.add_argument("original", nargs="?", default="20250919.json", help="Original JSON file path")
    parser.add_argument("generated", nargs="?", default="test.json", help="Generated JSON file path")
    parser.add_argument("--no-strip-none", action="store_true", help="Do not remove None/null fields before comparison")
    args = parser.parse_args()

    a = load_json(Path(args.original))
    b = load_json(Path(args.generated))

    if not args.no_strip_none:
        a = strip_none(a)
        b = strip_none(b)

    if a == b:
        print("Equivalent ✅")
        return 0

    print("Not equivalent ❌")
    print(diff_summary(a, b))
    return 1


if __name__ == "__main__":
    sys.exit(main())
