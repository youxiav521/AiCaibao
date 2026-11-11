from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Set
import os
import json
from copy import deepcopy
from time import perf_counter

from .cross_page_table_merge_llm import merge_agent, do_merge
from loguru import logger



def _safe_get_page_no(obj: Dict[str, Any]) -> Optional[int]:
    try:
        prov_list = obj.get("prov") or []
        if isinstance(prov_list, list) and prov_list:
            first = prov_list[0]
            if isinstance(first, dict):
                page_no = first.get("page_no")
                return int(page_no) if page_no is not None else None
    except Exception:
        return None
    return None


def _safe_get_bbox_lr(obj: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    try:
        prov_list = obj.get("prov") or []
        if isinstance(prov_list, list) and prov_list:
            first = prov_list[0]
            if isinstance(first, dict):
                bbox = first.get("bbox") or {}
                l = bbox.get("l")
                r = bbox.get("r")
                return (float(l) if l is not None else None, float(r) if r is not None else None)
    except Exception:
        return (None, None)
    return (None, None)


def _horizontal_overlap_ratio(box_a: Tuple[Optional[float], Optional[float]], box_b: Tuple[Optional[float], Optional[float]]) -> float:
    l_a, r_a = box_a
    l_b, r_b = box_b
    if l_a is None or r_a is None or l_b is None or r_b is None:
        return 0.0
    inter = max(0.0, min(r_a, r_b) - max(l_a, l_b))
    union = max(r_a, r_b) - min(l_a, l_b)
    if union <= 0:
        return 0.0
    return inter / union


def _width_and_center(box_lr: Tuple[Optional[float], Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    l, r = box_lr
    if l is None or r is None:
        return None, None
    w = float(r) - float(l)
    c = float(l) + w / 2.0
    return (w, c)


def _build_ref_index(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """构建 self_ref -> 对象 的索引，便于通过 $ref 查对象。"""
    ref_index: Dict[str, Dict[str, Any]] = {}
    for key in ("texts", "tables", "groups", "pictures"):
        for obj in (doc.get(key) or []):
            ref = obj.get("self_ref")
            if isinstance(ref, str):
                ref_index[ref] = obj
    # body 自身也加上（不是必须）
    body = doc.get("body") or {}
    if isinstance(body, dict) and isinstance(body.get("self_ref"), str):
        ref_index[body["self_ref"]] = body
    return ref_index


def _resolve_ref(ref_index: Dict[str, Dict[str, Any]], ref: str) -> Optional[Dict[str, Any]]:
    if not isinstance(ref, str):
        return None
    return ref_index.get(ref)


def _get_body_children_refs(doc: Dict[str, Any]) -> List[str]:
    body = doc.get("body") or {}
    children = body.get("children") or []
    refs: List[str] = []
    for ch in children:
        if isinstance(ch, dict) and "$ref" in ch and isinstance(ch["$ref"], str):
            refs.append(ch["$ref"])
    return refs


def _cleaned_body_refs_without_headers(doc: Dict[str, Any], ref_index: Dict[str, Dict[str, Any]]) -> List[str]:
    """返回移除页眉/页脚/页码后的 body 引用序列（不修改原 doc）。"""
    skip_labels = {"page_header", "page_footer", "pageNumber", "page_number"}
    refs = _get_body_children_refs(doc)
    cleaned: List[str] = []
    for r in refs:
        target = _resolve_ref(ref_index, r)
        if not isinstance(target, dict):
            cleaned.append(r)
            continue
        label = target.get("label")
        if label in skip_labels:
            logger.debug(f"merge_debug: skip body ref due to label '{label}', ref={r}")
            continue
        cleaned.append(r)
    return cleaned


def _extract_body_table_adjacency_pairs(doc: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """基于 body 顺序（移除页眉/页脚后），提取相邻表格对 (a_idx, b_idx)。"""
    ref_index = _build_ref_index(doc)
    cleaned_refs = _cleaned_body_refs_without_headers(doc, ref_index)
    pairs: Set[Tuple[int, int]] = set()

    logger.debug(f"merge_debug: cleaned_refs={cleaned_refs}")

    def _parse_table_idx(ref: str) -> Optional[int]:
        if not isinstance(ref, str):
            return None
        prefix = "#/tables/"
        if ref.startswith(prefix):
            try:
                return int(ref[len(prefix):])
            except Exception:
                return None
        return None

    n = len(cleaned_refs)
    i = 0
    while i < n:
        a_ref = cleaned_refs[i]
        a_idx = _parse_table_idx(a_ref)
        if a_idx is None:
            i += 1
            continue
        # 仅当下一个元素就是表格时，才认为二者相邻
        j = i + 1
        if j < n:
            b_ref = cleaned_refs[j]
            b_idx = _parse_table_idx(b_ref)
            if b_idx is not None:
                pairs.add((a_idx, b_idx))
        i += 1

    if pairs:
        logger.debug(f"merge_debug: body adjacency pairs (a_idx->b_idx): {sorted(list(pairs))}")
    else:
        logger.debug("merge_debug: no body adjacency pairs detected")
    return pairs


def _compute_num_cols(table_obj: Dict[str, Any]) -> int:
    try:
        data = table_obj.get("data") or {}
        cells = data.get("table_cells") or []
        max_end = 0
        for c in cells:
            end_idx = c.get("end_col_offset_idx")
            if isinstance(end_idx, int) and end_idx > max_end:
                max_end = end_idx
        return int(max_end)
    except Exception:
        return 0


def _get_num_rows(table_obj: Dict[str, Any]) -> int:
    try:
        data = table_obj.get("data") or {}
        num_rows = data.get("num_rows")
        return int(num_rows) if num_rows is not None else 0
    except Exception:
        return 0


def _set_num_rows(table_obj: Dict[str, Any], value: int) -> None:
    data = table_obj.setdefault("data", {})
    data["num_rows"] = int(value)


def _remove_header_rows_from_table_cells(table_obj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, Set[int]]:
    data = table_obj.get("data") or {}
    cells: List[Dict[str, Any]] = list(data.get("table_cells") or [])
    header_rows: Set[int] = set()
    for c in cells:
        if c.get("column_header") is True:
            try:
                header_rows.add(int(c.get("start_row_offset_idx") or 0))
            except Exception:
                continue
    if not header_rows:
        return cells, 0, set()
    filtered = [c for c in cells if int(c.get("start_row_offset_idx") or 0) not in header_rows]
    return filtered, len(header_rows), header_rows


def _merge_B_into_A_vertical(A: Dict[str, Any], B: Dict[str, Any]) -> None:
    # 1) 删除 B 的表头行（若存在）
    filtered_cells_B, removed_header_cnt, header_rows = _remove_header_rows_from_table_cells(B)

    # 2) 计算 A 目前的行数，用于 B 的行偏移
    a_rows = _get_num_rows(A)

    # 3) 复制并平移 B 的单元格行索引后追加到 A
    for c in filtered_cells_B:
        try:
            c["start_row_offset_idx"] = int(c.get("start_row_offset_idx") or 0) + a_rows
            c["end_row_offset_idx"] = int(c.get("end_row_offset_idx") or 0) + a_rows
        except Exception:
            pass

    data_A = A.setdefault("data", {})
    cells_A: List[Dict[str, Any]] = data_A.setdefault("table_cells", [])
    cells_A.extend(filtered_cells_B)

    # 4) 更新 A 的行数
    b_rows_total = _get_num_rows(B)
    added_rows = max(0, b_rows_total - removed_header_cnt)
    _set_num_rows(A, a_rows + added_rows)

    # 5) 记录合并元数据
    data_A.setdefault("merge_meta", {})
    mm = data_A["merge_meta"]
    merged_from: List[str] = mm.get("merged_from") or []
    merged_from.append(B.get("self_ref") or "")
    mm["merged_from"] = merged_from

    pages: Set[int] = set(mm.get("pages") or [])
    page_a = _safe_get_page_no(A)
    page_b = _safe_get_page_no(B)
    if page_a is not None:
        pages.add(int(page_a))
    if page_b is not None:
        pages.add(int(page_b))
    mm["pages"] = sorted(list(pages))
    mm["type"] = "vertical"

    # 6) 标记 B 为已合并并清空其数据，避免被再次消费
    B["label"] = "table_fragment_merged"
    data_B = B.setdefault("data", {})
    data_B["table_cells"] = []
    data_B["merged_into"] = A.get("self_ref") or ""
    _set_num_rows(B, 0)


def merge_vertical_chains(
    doc: Dict[str, Any],
    horizontal_overlap_threshold: float = 0.7,
    remove_secondary_header: bool = True,
) -> Dict[str, Any]:
    """
    对 Docling 导出的字典结构执行简化版 "垂直跨页表格合并"。

    规则：
    - 连续页 page_no(p) -> page_no(p+1)
    - 水平重叠占比 >= 阈值（基于 prov[0].bbox 的 l/r）
    - 将次页的表头行（column_header=True 的行）剔除后拼接到上一页表格中

    返回：修改后的同一 doc 引用（原地修改）。
    """
    tables: List[Dict[str, Any]] = list(doc.get("tables") or [])
    if not tables:
        return doc

    # 为快速访问构建索引和缓存
    info_list: List[Dict[str, Any]] = []
    for idx, t in enumerate(tables):
        page_no = _safe_get_page_no(t)
        l, r = _safe_get_bbox_lr(t)
        num_cols = _compute_num_cols(t)
        num_rows = _get_num_rows(t)
        info_list.append({
            "idx": idx,
            "page": page_no,
            "lr": (l, r),
            "num_cols": num_cols,
            "num_rows": num_rows,
            "label": t.get("label"),
        })

    # 将表按页分组
    by_page: Dict[int, List[int]] = {}
    for it in info_list:
        p = it["page"]
        if p is None:
            continue
        by_page.setdefault(int(p), []).append(it["idx"])

    # 基于 body 顺序（移除页眉/页脚后）的相邻表格对
    body_adjacent_pairs = _extract_body_table_adjacency_pairs(doc)

    # 记录哪些表已经作为 B 被合并过
    consumed: Set[int] = set()

    # 按页序遍历，尝试从每个表出发向后（页页相连）扩展链
    for p in sorted(by_page.keys()):
        current_indices = by_page.get(p, [])
        for a_idx in current_indices:
            if a_idx in consumed:
                continue
            A = tables[a_idx]
            if (A.get("data") or {}).get("table_cells") in (None, []):
                continue

            a_info = info_list[a_idx]
            a_cols = a_info["num_cols"]
            if a_cols <= 0:
                continue

            # 迭代向后合并：p+1, p+2, ...；每个后续页最多合并一个候选，避免误将同页其他独立表并入
            next_page = int(p) + 1
            # 锚点固定为起始表的 lr，避免随链漂移导致误合
            anchor_lr: Tuple[Optional[float], Optional[float]] = a_info["lr"]
            anchor_w, anchor_c = _width_and_center(anchor_lr)
            # 使用 prev_idx 跟随链路，确保严格按 body 相邻顺序推进
            prev_idx: int = a_idx
            while next_page in by_page:
                candidates = [b_idx for b_idx in by_page[next_page] if b_idx not in consumed]
                if not candidates:
                    break

                # 在候选里找与锚点水平重叠最大，且宽度/中心差异受限的一个
                best_b: Optional[int] = None
                best_overlap: float = 0.0
                for b_idx in candidates:
                    b_info = info_list[b_idx]
                    # 仅允许与上一个片段 prev_idx 在 body 中相邻的候选
                    if (prev_idx, b_idx) not in body_adjacent_pairs:
                        logger.debug(
                            f"merge_debug: reject candidate b_idx={b_idx} (prev_idx={prev_idx}) reason=not_adjacent_in_body"
                        )
                        continue
                    # 页号连续约束：相对 prev_idx
                    if (info_list[prev_idx]["page"] is None) or (b_info["page"] is None):
                        logger.debug(
                            f"merge_debug: reject candidate b_idx={b_idx} reason=page_missing prev_page={info_list[prev_idx]['page']} b_page={b_info['page']}"
                        )
                        continue
                    if int(b_info["page"]) != int(info_list[prev_idx]["page"]) + 1:
                        logger.debug(
                            f"merge_debug: reject candidate b_idx={b_idx} reason=page_not_consecutive prev_page={info_list[prev_idx]['page']} b_page={b_info['page']}"
                        )
                        continue
                    ov = _horizontal_overlap_ratio(anchor_lr, b_info["lr"])
                    if ov < horizontal_overlap_threshold:
                        logger.debug(
                            f"merge_debug: reject candidate b_idx={b_idx} reason=low_overlap overlap={ov:.3f} threshold={horizontal_overlap_threshold:.3f} anchor_lr={anchor_lr} b_lr={b_info['lr']}"
                        )
                        continue
                    # 额外约束：宽度差异与中心偏移
                    bw, bc = _width_and_center(b_info["lr"])
                    if anchor_w is None or anchor_c is None or bw is None or bc is None:
                        logger.debug(
                            f"merge_debug: reject candidate b_idx={b_idx} reason=insufficient_bbox anchor_w={anchor_w} anchor_c={anchor_c} bw={bw} bc={bc}"
                        )
                        continue
                    # 宽度相似度：相对差异不超过 20%
                    width_diff_ratio = abs(anchor_w - bw) / max(anchor_w, bw) if max(anchor_w, bw) > 0 else 1.0
                    if width_diff_ratio > 0.2:
                        logger.debug(
                            f"merge_debug: reject candidate b_idx={b_idx} reason=width_diff width_diff_ratio={width_diff_ratio:.3f} limit=0.200 anchor_w={anchor_w} bw={bw}"
                        )
                        continue
                    # 中心偏移约束：不超过 max(anchor_w, bw) 的 25%
                    center_delta_ratio = abs(anchor_c - bc) / max(anchor_w, bw) if max(anchor_w, bw) > 0 else 1.0
                    if center_delta_ratio > 0.25:
                        logger.debug(
                            f"merge_debug: reject candidate b_idx={b_idx} reason=center_delta center_delta_ratio={center_delta_ratio:.3f} limit=0.250 anchor_c={anchor_c} bc={bc} anchor_w={anchor_w} bw={bw}"
                        )
                        continue

                    if ov > best_overlap:
                        best_overlap = ov
                        best_b = b_idx

                if best_b is None:
                    break

                # 合并 best_b into A（每页最多合并一个）
                B = tables[best_b]
                _merge_B_into_A_vertical(A, B)
                consumed.add(best_b)

                # 更新 prev_idx 为刚并入的片段，进入下一页
                prev_idx = best_b
                # 进入下一页
                next_page += 1

    return doc


__all__ = [
    "merge_vertical_chains",
    "complete_cross_page_merge_with_llm",
]


def complete_cross_page_merge_with_llm(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    遍历 Docling JSON 中被标记为 "table_fragment_merged" 的表格，真正完成跨页表格的内容合并。

    规则：
    - 若表格对象的 label 为 "table_fragment_merged"，并且其 data.merged_into 字段存在，则：
      - 找到目标表格（被合并的目标）作为 table_json_str1
      - 当前表格作为 table_json_str2
      - 通过 LLM 的 merge_agent 判定调用 calling_merge 或 calling_merge_cell 完成最终合并
      - 用 LLM 返回的结果更新目标表格与当前表格
    - 若找不到目标表格或字段不全，则跳过

    返回：原地修改并返回同一 doc 引用。
    """
    if not isinstance(doc, dict):
        return doc

    tables: List[Dict[str, Any]] = list(doc.get("tables") or [])

    # 计算预测 CSV 基础名：优先使用原始文件名（去扩展名），否则使用 doc 的 name 字段
    origin = doc.get("origin") or {}
    origin_filename = origin.get("filename")
    raw_basename: Optional[str] = None
    if isinstance(origin_filename, str) and origin_filename:
        raw_basename = os.path.splitext(os.path.basename(origin_filename))[0]
    elif isinstance(doc.get("name"), str) and doc.get("name"):
        raw_basename = str(doc.get("name"))
    # 构建 self_ref -> 索引
    ref_to_idx: Dict[str, int] = {}
    for i, t in enumerate(tables):
        ref = t.get("self_ref")
        if isinstance(ref, str):
            ref_to_idx[ref] = i

    # 遍历所有已被标记 merged 的表格
    for idx, t in enumerate(tables):
        if t.get("label") != "table_fragment_merged":
            continue
        data = t.get("data") or {}
        merged_into_ref = data.get("merged_into")
        if not isinstance(merged_into_ref, str) or not merged_into_ref:
            continue
        if merged_into_ref not in ref_to_idx:
            continue

        target_idx = ref_to_idx[merged_into_ref]
        target_table = tables[target_idx]

        # 准备 JSON 字符串
        table_json_str1 = json.dumps(target_table, ensure_ascii=False)
        table_json_str2 = json.dumps(t, ensure_ascii=False)

        # 调用 LLM 判定并执行合并并记录耗时
        start_llm = perf_counter()

        resp = merge_agent(table_json_str1, table_json_str2, pred_csv_basename=raw_basename)
        t_llm_ms = (perf_counter() - start_llm) * 1000.0

        start_merge = perf_counter()
        result = do_merge(resp, table_json_str1, table_json_str2)
        t_merge_ms = (perf_counter() - start_merge) * 1000.0

        logger.bind(performance=True).info(
            f"complete_cross_page_merge_with_llm 耗时：merge_agent={t_llm_ms:.1f}ms, do_merge={t_merge_ms:.1f}ms, A={target_table.get('self_ref', '')}, B={t.get('self_ref', '')}"
        )
        if not result:
            continue

        # 用返回内容更新 A、B
        merged_A = result.get("merged_A")
        merged_B = result.get("merged_B")
        if isinstance(merged_A, dict):
            tables[target_idx] = merged_A
        if isinstance(merged_B, dict):
            tables[idx] = merged_B

    doc["tables"] = tables
    return doc


