from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - optional dependency fallback
    RecursiveCharacterTextSplitter = None

try:
    from langchain_text_splitters.json import RecursiveJsonSplitter
except Exception:  # pragma: no cover - optional dependency fallback
    RecursiveJsonSplitter = None

from .common import approx_token_len
from ..schemas import (
    ChildChunk,
    ChunkFields,
    GrandchildChunk,
    ParentChunk,
)
from ..utils.chunking_helpers import hash_id as _hash_id
from src.utils import json_utils

_TEXT_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ";", ",", " "]
_JSON_SEPARATORS = ["\n\n", "\n", "},", "],", "}", "]", ",", " "]


class _FallbackSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        text = text or ""
        if self.chunk_size <= 0:
            return [text] if text else []
        if len(text) <= self.chunk_size:
            return [text]
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            chunk = text[i : i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
            if i + self.chunk_size >= n:
                break
            i += step
        return chunks


def _build_splitter(chunk_size: int, chunk_overlap: int, separators: List[str]):
    if RecursiveCharacterTextSplitter is None:
        return _FallbackSplitter(chunk_size, chunk_overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=separators,
        length_function=approx_token_len,
    )


def _flatten_json_text(text: str) -> str:
    try:
        data = json_utils.loads(text)
    except Exception:
        return text

    lines: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_str = str(key)
                if key_str.isdigit():
                    next_path = f"{path}[{key_str}]" if path else f"[{key_str}]"
                else:
                    next_path = f"{path}.{key_str}" if path else key_str
                walk(value, next_path)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                next_path = f"{path}[{idx}]"
                walk(value, next_path)
        else:
            if isinstance(node, str):
                value = node.replace("\n", " ").strip()
            else:
                value = json_utils.dumps(node)
            lines.append(f"{path}: {value}" if path else str(value))

    walk(data, "")
    return "\n".join(lines) if lines else text


def _wrap_root_json(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    return {"root": data}


def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (src or {}).items():
        if key in dst and isinstance(dst.get(key), dict) and isinstance(value, dict):
            _deep_merge_dict(dst[key], value)
        else:
            dst[key] = value
    return dst


def _extract_elements(items: List[Dict[str, Any]], *, flatten_json: bool) -> List[Tuple[Optional[int], str]]:
    elements: List[Tuple[Optional[int], str]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        text = it.get("md") or it.get("value") or it.get("text") or ""
        text = str(text)
        if not text.strip():
            continue
        if flatten_json:
            text = _flatten_json_text(text)
        if not text.strip():
            continue
        eid = it.get("element_id")
        elements.append((eid, text))
    return elements


def _group_element_chunks(
    elements: List[Tuple[Optional[int], str]],
    max_tokens: int,
    orphan_min: int,
) -> List[List[Tuple[Optional[int], str]]]:
    if not elements:
        return []
    if max_tokens <= 0:
        return [elements]

    groups: List[List[Tuple[Optional[int], str]]] = []
    buf: List[Tuple[Optional[int], str]] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            groups.append(buf)
        buf = []
        buf_tokens = 0

    for eid, text in elements:
        tks = approx_token_len(text)
        if buf and (buf_tokens + tks > max_tokens):
            flush()
        buf.append((eid, text))
        buf_tokens += tks

    flush()

    if orphan_min > 0 and len(groups) > 1:
        last_group = groups[-1]
        last_text = "\n\n".join(text for _, text in last_group).strip()
        if approx_token_len(last_text) < orphan_min:
            groups[-2].extend(last_group)
            groups.pop()

    return groups


def _group_to_text_and_ids(group: List[Tuple[Optional[int], str]]) -> Tuple[str, List[int]]:
    text = "\n\n".join(t for _, t in group).strip()
    ids: List[int] = []
    for eid, _ in group:
        if eid is None:
            continue
        try:
            ids.append(int(eid))
        except Exception:
            continue
    return text, list(dict.fromkeys(ids))


def _make_chunks_from_items(
    items: List[Dict[str, Any]],
    file_id: str,
    *,
    strategy: str,
    parent_chunk_size: int,
    child_chunk_size: int,
    orphan_min: int,
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
    flatten_json: bool = False,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    elements = _extract_elements(items, flatten_json=flatten_json)
    if not elements:
        return [], []

    parent_groups = _group_element_chunks(elements, parent_chunk_size, orphan_min)
    child_orphan_min = max(1, min(int(child_chunk_size * 0.5), child_chunk_size))

    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    for idx, pgroup in enumerate(parent_groups):
        ptext, pids = _group_to_text_and_ids(pgroup)
        if not ptext:
            continue
        parent_id = f"{file_id}:p:{_hash_id(f'{idx}:{ptext}')}"
        pmeta = {
            "file_id": file_id,
            "folder_id": folder_id,
            "file_name": file_name or "",
            "source_element_ids": pids,
        }
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                text=ptext,
                **pmeta,
            )
        )

        child_groups = _group_element_chunks(pgroup, child_chunk_size, child_orphan_min)
        for jdx, cgroup in enumerate(child_groups):
            ctext, cids = _group_to_text_and_ids(cgroup)
            if not ctext:
                continue
            child_id = f"{parent_id}:{jdx}"
            cmeta = {
                "file_id": file_id,
                "folder_id": folder_id,
                "file_name": file_name or "",
                "source_element_ids": cids,
            }
            children.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    text=ctext,
                    **cmeta,
                )
            )

    return parents, children


def make_chunks_for_txt_items(
    items: List[Dict[str, Any]],
    file_id: str,
    *,
    parent_chunk_size: int,
    child_chunk_size: int,
    orphan_min: int,
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    return _make_chunks_from_items(
        items,
        file_id,
        strategy="txt",
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        orphan_min=orphan_min,
        folder_id=folder_id,
        file_name=file_name,
        flatten_json=False,
    )


def make_chunks_for_json_items(
    items: List[Dict[str, Any]],
    file_id: str,
    *,
    parent_chunk_size: int,
    child_chunk_size: int,
    orphan_min: int,
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    # 优先使用 RecursiveJsonSplitter 保留 JSON 父子层级关系：
    # - parent：按元素分组后 deep-merge 为合法 JSON，再作为父切片
    # - child：对父切片 JSON 再次递归切片，生成子切片
    # - grandchild：在 build_grandchildren_from_children_text 中对 child 做 leaf flatten
    #
    # 若依赖缺失或 JSON 解析失败，则回退到旧逻辑（flatten -> grouped）。
    if RecursiveJsonSplitter is None:
        return _make_chunks_from_items(
            items,
            file_id,
            strategy="json",
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            orphan_min=orphan_min,
            folder_id=folder_id,
            file_name=file_name,
            flatten_json=True,
        )

    parsed_by_eid: Dict[str, Dict[str, Any]] = {}
    grouping_elements: List[Tuple[Optional[int], str]] = []

    for it in items or []:
        if not isinstance(it, dict):
            continue
        text = it.get("md") or it.get("value") or it.get("text") or ""
        text = str(text)
        if not text.strip():
            continue
        eid = it.get("element_id")
        grouping_elements.append((eid, text))
        if eid is None:
            continue
        try:
            parsed = json_utils.loads(text)
            parsed_by_eid[str(eid)] = _wrap_root_json(parsed)
        except Exception:
            # 回退：任一元素无法 JSON 解析时，避免混合策略导致不可控行为
            return _make_chunks_from_items(
                items,
                file_id,
                strategy="json",
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                orphan_min=orphan_min,
                folder_id=folder_id,
                file_name=file_name,
                flatten_json=True,
            )

    if not grouping_elements:
        return [], []

    parent_groups = _group_element_chunks(grouping_elements, parent_chunk_size, orphan_min)
    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    for idx, pgroup in enumerate(parent_groups):
        merged: Dict[str, Any] = {}
        pids: List[int] = []
        for eid, text in pgroup:
            if eid is not None:
                try:
                    pids.append(int(eid))
                except Exception:
                    pass
            data = None
            if eid is not None:
                data = parsed_by_eid.get(str(eid))
            if data is None:
                try:
                    data = _wrap_root_json(json_utils.loads(text))
                except Exception:
                    continue
            _deep_merge_dict(merged, data)

        parent_text = json_utils.dumps(merged)
        if not parent_text.strip():
            continue

        parent_id = f"{file_id}:p:{_hash_id(f'{idx}:{parent_text}')}"
        pmeta = {
            "file_id": file_id,
            "folder_id": folder_id,
            "file_name": file_name or "",
            "source_element_ids": list(dict.fromkeys(pids)),
        }
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                text=parent_text,
                **pmeta,
            )
        )

        approx_tokens = max(approx_token_len(parent_text), 1)
        ratio = max(len(parent_text) / approx_tokens, 1.0)
        child_max_chars = max(int(ratio * int(child_chunk_size)), 200)
        try:
            splitter = RecursiveJsonSplitter(max_chunk_size=child_max_chars)
            child_chunks = splitter.split_text(
                json_data=merged,
                convert_lists=True,
                ensure_ascii=False,
            )
        except Exception:
            child_chunks = [parent_text]

        for jdx, ctext in enumerate(child_chunks):
            ctext_norm = str(ctext or "").strip()
            if not ctext_norm:
                continue
            child_id = f"{parent_id}:{jdx}"
            cmeta = {
                "file_id": file_id,
                "folder_id": folder_id,
                "file_name": file_name or "",
                "source_element_ids": list(dict.fromkeys(pids)),
            }
            children.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    text=ctext_norm,
                    **cmeta,
                )
            )

    return parents, children


def build_grandchildren_from_children_text(
    children: Sequence[ChildChunk],
    file_id: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
) -> List[GrandchildChunk]:
    # JSON 策略：先将 child 的 JSON 文本 flatten 成 leaf 路径行，再按行切分，
    # 避免在 key path 上按 "." 等分隔符拆碎语义。
    if strategy == "json":
        separators = ["\n\n", "\n", " "]
    else:
        separators = _TEXT_SEPARATORS if strategy == "txt" else _JSON_SEPARATORS
    splitter = _build_splitter(chunk_size, chunk_overlap, separators)

    grandchildren: List[GrandchildChunk] = []
    for ch in children or []:
        child_id = ch.child_id
        parent_id = ch.parent_id
        text = ch.text
        if not child_id or not parent_id or not str(text).strip():
            continue

        split_text = str(text)
        if strategy == "json":
            split_text = _flatten_json_text(split_text)
        pieces = splitter.split_text(split_text) if splitter else [split_text]
        for idx, piece in enumerate(pieces):
            piece_text = str(piece).strip()
            if not piece_text:
                continue
            gid = f"{file_id}:g:{_hash_id(f'{child_id}:{idx}:{piece_text[:80]}')}"
            gmeta = {
                "file_id": file_id,
                "folder_id": folder_id or "",
                "file_name": file_name or "",
                "type": "text",
                "source_element_ids": list(ch.source_element_ids),
            }
            grandchildren.append(
                GrandchildChunk(
                    grandchild_id=gid,
                    parent_id=parent_id,
                    child_id=child_id,
                    text=piece_text,
                    **gmeta,
                )
            )

    return grandchildren
