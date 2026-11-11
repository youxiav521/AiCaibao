"""
Smoke test script for the new lightweight cited QA API.

Endpoint:
  POST /api/content_generate_light/answer_with_citations

Usage (PowerShell):
  python scripts/test_light_qa_api.py --base-url http://127.0.0.1:8000

Notes:
  - This script only tests the new interface and does not affect existing APIs.
  - `questionType="text-json"` will (by design) return citations=[] in current reporter implementation.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Literal

import httpx


def build_payload(
    *,
    question_description: str,
    question_type: Literal["text", "text-json"],
    question_prompt: str,
    enable_sliding_windows: bool,
) -> Dict[str, Any]:
    # Minimal example: only the fields required by BatchQueryItemHttp are used.
    rag_items: List[Dict[str, Any]] = [
        {
            "questionId": "q1",
            "questionText": None,
            "elementIds": [101, 102],
            "fileId": "file_abc",
            "fileName": "财报.pdf",
            "score": 987,
            "text": "……2025年实现营业收入123亿元……",
            "imageStoragePath": None,
            "tableCaptionContextElementIds": [],
        }
    ]

    return {
        "questionDescription": question_description,
        "questionType": question_type,
        "questionPrompt": question_prompt,
        "ragItems": rag_items,
        "answerLang": "zh_CN",
        "modelId": "model_bailian_80b",
        "enableSlidingWindows": enable_sliding_windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:6300", help="FastAPI server base url")
    parser.add_argument("--question", default="公司2025年收入是多少？", help="questionDescription")
    parser.add_argument("--type", default="text", choices=["text", "text-json"], help="questionType")
    parser.add_argument("--prompt", default="", help="questionPrompt")
    parser.add_argument(
        "--enable-sliding-windows",
        action="store_true",
        default=True,
        help="Enable reporter sliding windows when context too long (default: enabled)",
    )
    parser.add_argument(
        "--disable-sliding-windows",
        action="store_true",
        default=False,
        help="Disable sliding windows (reject if context too long)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    enable_sliding_windows = bool(args.enable_sliding_windows) and (not bool(args.disable_sliding_windows))

    payload = build_payload(
        question_description=args.question,
        question_type=args.type,
        question_prompt=args.prompt,
        enable_sliding_windows=enable_sliding_windows,
    )

    url = args.base_url.rstrip("/") + "/api/content_generate_light/answer_with_citations"
    print("POST", url)
    print("Request payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    with httpx.Client(timeout=args.timeout) as client:
        resp = client.post(url, json=payload)
        print("\nStatus:", resp.status_code)
        resp.raise_for_status()

        body = resp.json()
        print("\nResponse JSON:")
        print(json.dumps(body, ensure_ascii=False, indent=2))

        answer_text = body.get("answerText") or body.get("answer_text")
        citations = body.get("citations") or []
        print("\nanswerText:")
        print(answer_text)
        print("\ncitations_count:", len(citations))


if __name__ == "__main__":
    main()

