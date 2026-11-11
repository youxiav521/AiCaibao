from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional
from loguru import logger

from ..schemas import ChildChunk, ChunkFields, GrandchildChunk
from ..utils.text_normalize import add_spacing_for_tokenizer as _add_spacing_for_tokenizer
from ..utils.chunking_helpers import hash_id as _hash_id, cell_text as _cell_text

_EXPLICIT_CHUNK_FIELDS = {"grandchild_id", "parent_id", "child_id", "text"}


def _safe_int(v: Any, default: int = 1) -> int:
    """安全地将值转为正整数，失败返回默认值。"""
    try:
        iv = int(v) if v is not None else default
        return iv if iv > 0 else default
    except Exception:
        return default


def _build_col_header_paths(header_cells: List[List[Dict[str, Any]]]) -> Dict[int, List[str]]:
    """从 headerCells 构建列索引到层级标题路径的映射。"""
    col2headers: Dict[int, List[str]] = {}
    if not isinstance(header_cells, list):
        return col2headers
    for row in header_cells:
        if not isinstance(row, list):
            continue
        for cell in row:
            try:
                text = str(cell.get("text", "")).strip()
                if not text:
                    continue
                cstart = cell.get("colstart")
                if cstart is None:
                    cstart = cell.get("colStart")
                if cstart is None:
                    cstart = cell.get("col_start")
                cstart_i = _safe_int(cstart, 0)
                cs = _safe_int(cell.get("colspan") if cell.get("colspan") is not None else cell.get("col_span", 1), 1)
                for ci in range(cstart_i, cstart_i + cs):
                    path = col2headers.get(ci) or []
                    path.append(text)
                    col2headers[ci] = path
            except Exception:
                continue
    for k, arr in list(col2headers.items()):
        seen = set()
        dedup: List[str] = []
        for t in arr:
            if t and t not in seen:
                seen.add(t)
                dedup.append(t)
        col2headers[k] = dedup
    
    # 兼容：部分解析结果可能使用 1-based 列索引（导致第 0 列缺失、整体右移）
    # 若最小列号为 1 且不存在 0 列，则整体左移 1，修正到 0-based。
    try:
        if col2headers and (0 not in col2headers):
            min_k = min(col2headers.keys())
            if min_k == 1:
                col2headers = {k - 1: v for k, v in col2headers.items() if isinstance(k, int) and k >= 1}
    except Exception:
        pass
    return col2headers


def _guess_header_from_first_row(rows: List[List[Any]]) -> Dict[int, List[str]]:
    """若缺少 headerCells，则从首行推测列标题。"""
    if not rows:
        return {}
    first = rows[0]
    col2headers: Dict[int, List[str]] = {}
    for j, c in enumerate(first):
        txt = _cell_text(c)
        if txt:
            col2headers[j] = [txt]
    return col2headers


def _make_table_label(entry: Dict[str, Any]) -> str:
    """生成表格标签名称。"""
    tname = str(entry.get("table_type") or "").strip()
    if tname:
        return tname
    html_preview = str(entry.get("html") or "").strip()
    if html_preview:
        return "表"
    return "表"

# ── 表格行级孙切片 fallback 原因常量 ──
_FALLBACK_REASON_SPLIT_STRATEGY_CHAR = "split_strategy_char"
_FALLBACK_REASON_NO_ROW_RANGE_NON_OWNER = "table_no_row_range_non_owner"
_FALLBACK_REASON_MISSING_STRUCTURE = "table_row_missing_structure"
_FALLBACK_REASON_TABLE_ALREADY_COVERED = "table_already_covered_by_row_children"


def _has_header_compat(entry: Dict[str, Any]) -> bool:
    """兼容 hasHeader / has_header 两种命名。"""
    v = entry.get("hasHeader")
    if v is None:
        v = entry.get("has_header")
    return bool(v)


def _header_cells_compat(entry: Dict[str, Any]) -> Optional[List[List[Dict[str, Any]]]]:
    """兼容 headerCells / header_cells 两种命名。"""
    v = entry.get("headerCells")
    if v is None:
        v = entry.get("header_cells")
    return v if isinstance(v, list) else None


def _make_row_text(
    row: List[Any],
    row_idx: int,
    *,
    has_header: bool,
    col2headers: Dict[int, List[str]],
    table_label: str,
    heading_path: str,
    include_heading_prefix: bool,
) -> str:
    """将单行表格数据转为带语义的文本。"""
    if not has_header:
        values = [_cell_text(c) for c in row]
        text = (
            (f"{table_label} | Row {row_idx}: " if table_label else f"Row {row_idx}: ")
            + " | ".join([v for v in values if v])
        )
    else:
        row_values = [_cell_text(c) for c in row]
        row_parts: List[str] = []
        for j, v in enumerate(row_values):
            hp = col2headers.get(j) or []
            header_str = " ".join(hp) if hp else ""
            row_parts.append(f"{header_str}：{v}")
        text = "| " + " | ".join(row_parts) + " |"
        if table_label:
            text = f"{table_label}\n{text}"
    if include_heading_prefix and heading_path and heading_path != "__preface__":
        text = f"【{heading_path}】\n{text}"
    text = _add_spacing_for_tokenizer(text)
    return text


def _emit_table_fallback(
    *,
    child_text: str,
    child_id: str,
    parent_id: Optional[str],
    order: Any,
    element_id: Any,
    file_id: str,
    folder_id: Optional[str],
    file_name: Optional[str],
    fallback_reason: str,
    out: List[GrandchildChunk],
) -> None:
    """为表格 child 生成单条 fallback 孙切片。"""
    text = _add_spacing_for_tokenizer(child_text or "")
    if not text.strip():
        return
    file_id_local = str(file_id)
    eid_s = "" if element_id is None else str(element_id)
    base_gid = f"{file_id_local}:g:{_hash_id(eid_s + ':table_fallback:' + text[:100])}"
    gid = f"{base_gid}:{child_id}"
    meta = ChunkFields(
        file_id=file_id_local,
        folder_id=folder_id or "",
        file_name=file_name or "",
        element_id=element_id,
        type="table_row",
        parent_id=parent_id or "",
        child_id=child_id,
        source_element_ids=[element_id] if element_id is not None else [],
    )
    out.append(
        GrandchildChunk(
            grandchild_id=gid,
            parent_id=parent_id or "",
            child_id=child_id,
            text=text,
            **meta.model_dump(exclude=_EXPLICIT_CHUNK_FIELDS),
        )
    )


def _emit_non_table_piece_grandchild(
    *,
    child_text: str,
    child_id: str,
    parent_id: Optional[str],
    order: Any,
    element_id: Any,
    file_id: str,
    folder_id: Optional[str],
    file_name: Optional[str],
    element_type: str,
    split_strategy: str,
    out: List[GrandchildChunk],
) -> None:
    text = _add_spacing_for_tokenizer(child_text or "")
    if not text.strip():
        return
    file_id_local = str(file_id)
    eid_s = "" if element_id is None else str(element_id)
    gid = f"{file_id_local}:g:{_hash_id(eid_s + ':' + split_strategy + ':' + text[:100])}:{child_id}"
    meta = ChunkFields(
        file_id=file_id_local,
        folder_id=folder_id or "",
        file_name=file_name or "",
        element_id=element_id,
        type=element_type or "text",
        parent_id=parent_id or "",
        child_id=child_id,
        split_strategy=split_strategy,
        source_element_ids=[element_id] if element_id is not None else [],
    )
    out.append(
        GrandchildChunk(
            grandchild_id=gid,
            parent_id=parent_id or "",
            child_id=child_id,
            text=text,
            **meta.model_dump(exclude=_EXPLICIT_CHUNK_FIELDS),
        )
    )


def create_grandchildren(
    children: Sequence[ChildChunk],
    items: List[Dict[str, Any]],
    file_id: str,
    *,
    include_titles: bool = True,
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
    include_heading_prefix: bool = False,
) -> List[GrandchildChunk]:
    """基于子切片对象生成微粒度孙切片。"""

    eid_to_item: Dict[str, Dict[str, Any]] = {}
    items_with_eid = 0
    items_without_eid = 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        eid = it.get("element_id")
        if eid is None:
            items_without_eid += 1
            continue
        items_with_eid += 1
        eid_to_item[str(eid)] = it

    total_items = len(items or [])
    logger.debug(
        "Items 统计: 总数={}, 有element_id={}, 无element_id={}",
        total_items,
        items_with_eid,
        items_without_eid,
    )

    out: List[GrandchildChunk] = []
    all_src_ids: set[str] = set()
    children_with_src_ids = 0
    children_without_src_ids = 0
    skipped_stats: Dict[str, int] = {
        "empty_src_ids": 0,
        "entry_not_found": 0,
        "non_table_empty_text": 0,
        "table_fallback": 0,
    }
    skipped_elements: List[Dict[str, Any]] = []
    no_range_owner_map: Dict[str, str] = {}
    table_rows_covered: set[str] = set()

    for ch in children or []:
        try:
            ch_meta = ch.model_copy(deep=True)
            child_id = ch.child_id
            if not child_id:
                continue
            parent_id = ch.parent_id
            order = None
            src_ids = list(ch_meta.source_element_ids)
            primary_src_ids = list(src_ids)
            heading_ids = list(ch_meta.heading_node_ids)
            heading_path = str(ch_meta.heading_path or "").strip()
            split_strategy = ch_meta.split_strategy

            if split_strategy in {"char", "paragraph"} and len(primary_src_ids) == 1:
                element_id = primary_src_ids[0]
                entry = eid_to_item.get(str(element_id)) if element_id is not None else None
                etype = str(entry.get("type") or "").lower() if isinstance(entry, dict) else ""
                if etype not in {"table", "excel_table"}:
                    _emit_non_table_piece_grandchild(
                        child_text=ch.text,
                        child_id=child_id,
                        parent_id=parent_id,
                        order=order,
                        element_id=element_id,
                        file_id=file_id,
                        folder_id=folder_id,
                        file_name=file_name,
                        element_type=etype or "text",
                        split_strategy=str(split_strategy),
                        out=out,
                    )
                    continue

            if split_strategy == "table_row":
                element_id = src_ids[0] if src_ids else None
                rs_raw = ch_meta.table_row_start
                re_raw = ch_meta.table_row_end
                try:
                    row_start = int(rs_raw) if rs_raw is not None else None
                except (ValueError, TypeError):
                    row_start = None
                try:
                    row_end = int(re_raw) if re_raw is not None else None
                except (ValueError, TypeError):
                    row_end = None

                entry = eid_to_item.get(str(element_id)) if element_id is not None else None
                etype = str(entry.get("type") or "").lower() if isinstance(entry, dict) else ""
                if (
                    element_id is None
                    or not isinstance(entry, dict)
                    or etype not in {"table", "excel_table"}
                ):
                    _emit_table_fallback(
                        child_text=ch.text,
                        child_id=child_id,
                        parent_id=parent_id,
                        order=order,
                        element_id=element_id,
                        file_id=file_id,
                        folder_id=folder_id,
                        file_name=file_name,
                        fallback_reason=_FALLBACK_REASON_MISSING_STRUCTURE,
                        out=out,
                    )
                    continue

                if row_start is None or row_end is None or row_start < 0 or row_end < row_start:
                    raise ValueError(
                        f"table_row grandchild has invalid row range: "
                        f"file_id={file_id}, child_id={child_id}, parent_id={parent_id}, "
                        f"row_start={rs_raw}, row_end={re_raw}"
                    )

                table_eid_key = str(element_id) if element_id is not None else ""
                if table_eid_key:
                    table_rows_covered.add(table_eid_key)

                rows = entry.get("rows") or []
                has_header = _has_header_compat(entry)
                header_cells = _header_cells_compat(entry)
                col2headers: Dict[int, List[str]] = {}
                if has_header:
                    col2headers = (
                        _build_col_header_paths(header_cells)
                        if header_cells
                        else _guess_header_from_first_row(rows)
                    )
                table_label = _make_table_label(entry) if include_titles else ""
                start_i = max(0, min(row_start, len(rows)))
                end_i = max(0, min(row_end, len(rows) - 1))
                if end_i < start_i:
                    start_i, end_i = end_i, start_i

                file_id_local = str(file_id)
                eid_s = "" if element_id is None else str(element_id)
                for i in range(start_i, end_i + 1):
                    r = rows[i] if i < len(rows) else None
                    if not isinstance(r, list):
                        continue
                    text_local = _make_row_text(
                        r,
                        i,
                        has_header=has_header,
                        col2headers=col2headers,
                        table_label=table_label,
                        heading_path=heading_path,
                        include_heading_prefix=include_heading_prefix,
                    )
                    if not text_local.strip():
                        continue
                    base_gid = f"{file_id_local}:g:{_hash_id(eid_s + ':r' + str(i) + ':' + text_local[:100])}"
                    gid_local = f"{base_gid}:{child_id}"
                    meta_local = ChunkFields(
                        file_id=file_id_local,
                        folder_id=folder_id or "",
                        file_name=file_name or "",
                        element_id=element_id,
                        type=etype,
                        parent_id=parent_id,
                        child_id=child_id,
                        source_element_ids=[element_id] if element_id is not None else [],
                        table_row_start=row_start,
                        table_row_end=row_end,
                    )
                    out.append(
                        GrandchildChunk(
                            grandchild_id=gid_local,
                            parent_id=parent_id,
                            child_id=child_id,
                            text=text_local,
                            **meta_local.model_dump(exclude=_EXPLICIT_CHUNK_FIELDS),
                        )
                    )
                continue

            if heading_ids:
                src_ids = list(dict.fromkeys(src_ids + heading_ids))
            if not src_ids:
                children_without_src_ids += 1
                skipped_stats["empty_src_ids"] += 1
                skipped_elements.append(
                    {
                        "reason": "empty_src_ids",
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "order": order,
                    }
                )
                continue

            children_with_src_ids += 1
            for eid in src_ids:
                if eid is not None:
                    all_src_ids.add(str(eid))

            inherited_row_start = None
            inherited_row_end = None
            has_inherited_range = False

            seen_eids: set[str] = set()
            for eid in src_ids:
                if eid is None:
                    continue
                key = str(eid)
                if key in seen_eids:
                    continue
                seen_eids.add(key)
                entry = eid_to_item.get(key)
                if not entry:
                    skipped_stats["entry_not_found"] += 1
                    skipped_elements.append(
                        {
                            "reason": "entry_not_found",
                            "element_id": key,
                            "child_id": child_id,
                            "parent_id": parent_id,
                        }
                    )
                    continue

                etype = str(entry.get("type") or "").lower()
                file_id_local = str(file_id)
                element_id = entry.get("element_id")

                if etype not in {"table", "excel_table"}:
                    text = str(entry.get("md") or entry.get("value") or entry.get("text") or "").strip()
                    if not text:
                        skipped_stats["non_table_empty_text"] += 1
                        skipped_elements.append(
                            {
                                "reason": "non_table_empty_text",
                                "element_id": element_id,
                                "type": etype,
                                "child_id": child_id,
                                "entry_keys": list(entry.keys()) if isinstance(entry, dict) else [],
                            }
                        )
                        continue
                    if include_heading_prefix and heading_path and heading_path != "__preface__":
                        text = f"【{heading_path}】\n{text}"
                    text = _add_spacing_for_tokenizer(text)
                    if not text.strip():
                        skipped_stats["non_table_empty_text"] += 1
                        continue

                    eid_s = "" if element_id is None else str(element_id)
                    base_gid = f"{file_id_local}:g:{_hash_id(eid_s + ':' + text)}"
                    gid = f"{base_gid}:{child_id}"
                    merged_meta = ChunkFields(
                        file_id=file_id_local,
                        folder_id=folder_id or "",
                        file_name=file_name or "",
                        element_id=element_id,
                        type=etype or "text",
                        parent_id=parent_id,
                        child_id=child_id,
                        source_element_ids=[element_id] if element_id is not None else [],
                    )
                    out.append(
                        GrandchildChunk(
                            grandchild_id=gid,
                            parent_id=parent_id,
                            child_id=child_id,
                            text=text,
                            **merged_meta.model_dump(exclude=_EXPLICIT_CHUNK_FIELDS),
                        )
                    )
                    continue

                table_eid_key = str(element_id) if element_id is not None else ""
                if split_strategy == "char":
                    _emit_table_fallback(
                        child_text=ch.text,
                        child_id=child_id,
                        parent_id=parent_id,
                        order=order,
                        element_id=element_id,
                        file_id=file_id,
                        folder_id=folder_id,
                        file_name=file_name,
                        fallback_reason=_FALLBACK_REASON_SPLIT_STRATEGY_CHAR,
                        out=out,
                    )
                    skipped_stats["table_fallback"] += 1
                    continue

                rows = entry.get("rows") or []
                has_header = _has_header_compat(entry)
                header_cells = _header_cells_compat(entry)
                col2headers: Dict[int, List[str]] = {}
                if has_header:
                    col2headers = (
                        _build_col_header_paths(header_cells)
                        if header_cells
                        else _guess_header_from_first_row(rows)
                    )
                table_label = _make_table_label(entry) if include_titles else ""
                eid_s = "" if element_id is None else str(element_id)

                if has_inherited_range:
                    if table_eid_key:
                        table_rows_covered.add(table_eid_key)
                    start_i = max(0, min(inherited_row_start, len(rows)))
                    end_i = max(0, min(inherited_row_end, len(rows) - 1))
                    if end_i < start_i:
                        start_i, end_i = end_i, start_i
                    for i in range(start_i, end_i + 1):
                        r = rows[i] if i < len(rows) else None
                        if not isinstance(r, list):
                            continue
                        text_local = _make_row_text(
                            r,
                            i,
                            has_header=has_header,
                            col2headers=col2headers,
                            table_label=table_label,
                            heading_path=heading_path,
                            include_heading_prefix=include_heading_prefix,
                        )
                        if not text_local.strip():
                            continue
                        base_gid = f"{file_id_local}:g:{_hash_id(eid_s + ':r' + str(i) + ':' + text_local[:100])}"
                        gid_local = f"{base_gid}:{child_id}"
                        meta_local = ChunkFields(
                            file_id=file_id_local,
                            folder_id=folder_id or "",
                            file_name=file_name or "",
                            element_id=element_id,
                            type=etype,
                            parent_id=parent_id,
                            child_id=child_id,
                            source_element_ids=[element_id] if element_id is not None else [],
                        )
                        out.append(
                            GrandchildChunk(
                                grandchild_id=gid_local,
                                parent_id=parent_id,
                                child_id=child_id,
                                text=text_local,
                                **meta_local.model_dump(exclude=_EXPLICIT_CHUNK_FIELDS),
                            )
                        )
                    continue

                if table_eid_key and table_eid_key in table_rows_covered:
                    _emit_table_fallback(
                        child_text=ch.text,
                        child_id=child_id,
                        parent_id=parent_id,
                        order=order,
                        element_id=element_id,
                        file_id=file_id,
                        folder_id=folder_id,
                        file_name=file_name,
                        fallback_reason=_FALLBACK_REASON_TABLE_ALREADY_COVERED,
                        out=out,
                    )
                    skipped_stats["table_fallback"] += 1
                    continue

                if table_eid_key and table_eid_key not in no_range_owner_map:
                    no_range_owner_map[table_eid_key] = child_id
                    table_rows_covered.add(table_eid_key)
                elif table_eid_key:
                    _emit_table_fallback(
                        child_text=ch.text,
                        child_id=child_id,
                        parent_id=parent_id,
                        order=order,
                        element_id=element_id,
                        file_id=file_id,
                        folder_id=folder_id,
                        file_name=file_name,
                        fallback_reason=_FALLBACK_REASON_NO_ROW_RANGE_NON_OWNER,
                        out=out,
                    )
                    skipped_stats["table_fallback"] += 1
                    continue

                data_start = 0
                if has_header:
                    data_start = len(header_cells) if header_cells else 1

                for i in range(data_start, len(rows)):
                    r = rows[i]
                    if not isinstance(r, list):
                        continue
                    text_local = _make_row_text(
                        r,
                        i,
                        has_header=has_header,
                        col2headers=col2headers,
                        table_label=table_label,
                        heading_path=heading_path,
                        include_heading_prefix=include_heading_prefix,
                    )
                    if not text_local.strip():
                        continue
                    base_gid = f"{file_id_local}:g:{_hash_id(eid_s + ':r' + str(i) + ':' + text_local[:100])}"
                    gid_local = f"{base_gid}:{child_id}"
                    meta_local = ChunkFields(
                        file_id=file_id_local,
                        folder_id=folder_id or "",
                        file_name=file_name or "",
                        element_id=element_id,
                        type=etype,
                        parent_id=parent_id,
                        child_id=child_id,
                        source_element_ids=[element_id] if element_id is not None else [],
                    )
                    out.append(
                        GrandchildChunk(
                            grandchild_id=gid_local,
                            parent_id=parent_id,
                            child_id=child_id,
                            text=text_local,
                            **meta_local.model_dump(exclude=_EXPLICIT_CHUNK_FIELDS),
                        )
                    )
        except Exception:
            logger.exception("Failed to generate grandchild chunks for child: {}", ch)
            raise

    total_skipped = sum(skipped_stats.values())
    items_eid_set = set(eid_to_item.keys())
    missing_eids = items_eid_set - all_src_ids
    extra_eids = all_src_ids - items_eid_set

    logger.debug(
        "子切片统计: 总数={}, 有src_ids={}, 无src_ids={}",
        len(children or []),
        children_with_src_ids,
        children_without_src_ids,
    )
    logger.debug(
        "Source Element IDs 统计: 去重后总数={}, 来自items={}, 缺失={}, 多余={}",
        len(all_src_ids),
        len(items_eid_set),
        len(missing_eids),
        len(extra_eids),
    )
    if missing_eids:
        logger.debug("缺失的 element_id 示例（前20个，共{}个）: {}", len(missing_eids), list(missing_eids)[:20])
    if extra_eids:
        logger.debug("多余的 element_id 示例（前20个，共{}个）: {}", len(extra_eids), list(extra_eids)[:20])
    logger.debug("孙切片生成统计: 总共跳过 {} 个元素", total_skipped)
    logger.debug("跳过原因统计: {}", skipped_stats)
    if skipped_elements:
        sample_size = min(50, len(skipped_elements))
        logger.debug("被跳过的元素示例（前 {} 个，共 {} 个）:", sample_size, len(skipped_elements))
        for i, elem in enumerate(skipped_elements[:sample_size]):
            logger.debug("  跳过元素 #{}: {}", i + 1, elem)
        if len(skipped_elements) > sample_size:
            logger.debug("  ... 还有 {} 个被跳过的元素未显示", len(skipped_elements) - sample_size)
    logger.debug("最终生成孙切片数量: {}", len(out))
    return out
