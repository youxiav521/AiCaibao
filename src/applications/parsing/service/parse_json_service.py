from __future__ import annotations

import logging
import re
from json.decoder import scanstring
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from src.applications.parsing.utils.ragflow_parsers import RAGFlowJsonParser, get_text
from src.utils import json_utils

logger = logging.getLogger(__name__)


_JSON_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")


class _JsonLineSpanParser:
    """构建 JSON path -> (start_line, end_line) 映射（1-based, inclusive）。

    - 对象成员的 start_line 以 key 所在行记；
    - 数组元素的 start_line 以元素值所在行记。
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._line = 1
        self._spans: Dict[Tuple[Any, ...], Tuple[int, int]] = {}

    def parse(self) -> Dict[Tuple[Any, ...], Tuple[int, int]]:
        self._skip_ws()
        self._parse_value(())
        return self._spans

    def _peek(self) -> str:
        if self._pos >= len(self._text):
            raise ValueError("unexpected end of JSON input")
        return self._text[self._pos]

    def _skip_ws(self) -> None:
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch in (" ", "\t", "\r"):
                self._pos += 1
                continue
            if ch == "\n":
                self._pos += 1
                self._line += 1
                continue
            break

    def _parse_string_value(self) -> str:
        if self._peek() != '"':
            raise ValueError("expected string")
        # Python 3.13+: `strict` is positional-only.
        value, end = scanstring(self._text, self._pos + 1, True)
        self._pos = end
        return value

    def _parse_number(self) -> None:
        match = _JSON_NUMBER_RE.match(self._text, self._pos)
        if not match:
            raise ValueError("invalid number")
        self._pos = match.end()

    def _parse_value(self, path: Tuple[Any, ...], *, start_line_override: int | None = None) -> int:
        self._skip_ws()
        start_line = int(start_line_override) if start_line_override is not None else self._line

        ch = self._peek()
        if ch == "{":
            end_line = self._parse_object(path)
        elif ch == "[":
            end_line = self._parse_array(path)
        elif ch == '"':
            self._parse_string_value()
            end_line = start_line
        elif ch in "-0123456789":
            self._parse_number()
            end_line = start_line
        elif self._text.startswith("true", self._pos):
            self._pos += 4
            end_line = start_line
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            end_line = start_line
        elif self._text.startswith("null", self._pos):
            self._pos += 4
            end_line = start_line
        else:
            raise ValueError(f"unexpected token at pos {self._pos}")

        self._spans[path] = (start_line, end_line)
        return end_line

    def _parse_object(self, path: Tuple[Any, ...]) -> int:
        if self._peek() != "{":
            raise ValueError("expected '{'")
        self._pos += 1
        self._skip_ws()
        if self._pos < len(self._text) and self._text[self._pos] == "}":
            self._pos += 1
            return self._line

        while True:
            self._skip_ws()
            key_start_line = self._line
            key = self._parse_string_value()
            self._skip_ws()
            if self._peek() != ":":
                raise ValueError("expected ':'")
            self._pos += 1
            self._parse_value(path + (key,), start_line_override=key_start_line)
            self._skip_ws()

            ch = self._peek()
            if ch == ",":
                self._pos += 1
                continue
            if ch == "}":
                self._pos += 1
                return self._line
            raise ValueError("expected ',' or '}'")

    def _parse_array(self, path: Tuple[Any, ...]) -> int:
        if self._peek() != "[":
            raise ValueError("expected '['")
        self._pos += 1
        self._skip_ws()
        if self._pos < len(self._text) and self._text[self._pos] == "]":
            self._pos += 1
            return self._line

        idx = 0
        while True:
            self._skip_ws()
            element_start_line = self._line
            self._parse_value(path + (idx,), start_line_override=element_start_line)
            self._skip_ws()

            ch = self._peek()
            if ch == ",":
                self._pos += 1
                idx += 1
                continue
            if ch == "]":
                self._pos += 1
                return self._line
            raise ValueError("expected ',' or ']'")


def _maybe_int_index(key: Any) -> Any:
    if isinstance(key, str) and key.isdigit():
        try:
            return int(key)
        except Exception:
            return key
    return key


def _collect_leaf_paths(
    node: Any,
    path: Tuple[Any, ...],
    original_node: Any,
    out: List[Tuple[Any, ...]],
) -> None:
    if isinstance(node, dict):
        if not node:
            out.append(path)
            return
        if isinstance(original_node, list):
            for key, child in node.items():
                idx = _maybe_int_index(key)
                orig_child = None
                if isinstance(idx, int) and 0 <= idx < len(original_node):
                    orig_child = original_node[idx]
                _collect_leaf_paths(child, path + (idx,), orig_child, out)
            return

        for key, child in node.items():
            orig_child = original_node.get(key) if isinstance(original_node, dict) else None
            _collect_leaf_paths(child, path + (key,), orig_child, out)
        return

    if isinstance(node, list):
        if not node:
            out.append(path)
            return
        for idx, child in enumerate(node):
            orig_child = None
            if isinstance(original_node, list) and 0 <= idx < len(original_node):
                orig_child = original_node[idx]
            _collect_leaf_paths(child, path + (idx,), orig_child, out)
        return

    out.append(path)


def _compute_chunk_line_span(
    chunk_obj: Any,
    *,
    original_wrapped: Any,
    spans: Mapping[Tuple[Any, ...], Tuple[int, int]],
) -> Tuple[int, int]:
    leaf_paths: List[Tuple[Any, ...]] = []
    _collect_leaf_paths(chunk_obj, (), original_wrapped, leaf_paths)
    matched = [spans[p] for p in leaf_paths if p in spans]
    if not matched:
        return spans.get((), (0, 0))
    start_row = min(s for s, _ in matched)
    end_row = max(e for _, e in matched)
    return start_row, end_row


class JsonParsingService:
    """JSON 文件解析服务。"""

    def __init__(self, chunk_token_num: int = 128) -> None:
        self.chunk_token_num = int(chunk_token_num)
        self.parser = RAGFlowJsonParser()

    def parse_to_items(self, file_path: str) -> List[Dict[str, Any]]:
        """将 JSON/JSONL 文件解析为 element 列表。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        content = get_text(str(path))
        is_jsonl = self.parser.is_jsonl_format(content)

        chunks_with_span: List[Tuple[str, Tuple[int, int]]] = []

        if is_jsonl:
            # JSONL：每条记录就是一个物理行，startRow/endRow 可精确映射到该行。
            for line_no, line in enumerate(content.splitlines(), start=1):
                raw = str(line).strip()
                if not raw:
                    continue
                try:
                    data = json_utils.loads(raw)
                except Exception:
                    continue

                wrapped = self.parser._wrap_root(data)
                parts = self.parser._splitter.split_text(
                    json_data=wrapped,
                    convert_lists=self.parser.convert_lists,
                    ensure_ascii=self.parser.ensure_ascii,
                )
                for part in parts:
                    part_text = str(part or "").strip()
                    if not part_text:
                        continue
                    chunks_with_span.append((part_text, (line_no, line_no)))
        else:
            try:
                data = json_utils.loads(content)
            except Exception:
                return []

            wrap_root = not isinstance(data, dict)
            wrapped = self.parser._wrap_root(data)
            original_wrapped: Any = wrapped

            try:
                spans = _JsonLineSpanParser(content).parse()
            except Exception:
                spans = {}

            if wrap_root and spans:
                # list/scalar root 会被包装为 {"root": ...}，这里把 path 同样加上 root 前缀。
                prefixed: Dict[Tuple[Any, ...], Tuple[int, int]] = {}
                for p, span in spans.items():
                    if p == ():
                        prefixed[()] = span
                        prefixed[("root",)] = span
                        continue
                    prefixed[("root",) + p] = span
                spans = prefixed

            parts = self.parser._splitter.split_text(
                json_data=wrapped,
                convert_lists=self.parser.convert_lists,
                ensure_ascii=self.parser.ensure_ascii,
            )
            for part in parts:
                part_text = str(part or "").strip()
                if not part_text:
                    continue
                try:
                    chunk_obj = json_utils.loads(part_text)
                except Exception:
                    chunks_with_span.append((part_text, (0, 0)))
                    continue
                chunk_span = _compute_chunk_line_span(
                    chunk_obj,
                    original_wrapped=original_wrapped,
                    spans=spans,
                )
                chunks_with_span.append((part_text, chunk_span))

        items: List[Dict[str, Any]] = []
        for chunk_text, (start_row, end_row) in chunks_with_span:
            items.append(
                {
                    "type": "text",
                    "value": chunk_text,
                    "md": chunk_text,
                    "startRow": int(start_row),
                    "endRow": int(end_row),
                }
            )

        return items
