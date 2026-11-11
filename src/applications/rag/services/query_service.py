from __future__ import annotations

from typing import Any, List, Optional

from src.base.langfuse_observe import observe
from loguru import logger

from ..config.rag_settings import RAG_SETTINGS
from ..schemas import BatchQueryItem, BatchQueryRequest, SearchType
from .vectorstore import VectorStoresService

_GRANDCHILD_BATCH_QUERY_OUTPUT_FIELDS: tuple[str, ...] = (
    "text",
    "file_id",
    "file_name",
    "element_id",
    "source_element_ids",
    "image_storage_path",
    "table_caption_context_element_ids",
)


def assign_rank_scores(docs: List[Any], score_key: str = "score") -> List[Any]:
    """基于排名重分配分数，保证排序稳定且分数可比较。

    使用公式：`score = (total_count - rank) / total_count`
    - 第 1 名的分数为 `1.0`
    - 最后一名的分数为 `1 / total_count`
    """
    if not docs:
        return docs

    total = len(docs)
    for rank, doc in enumerate(docs):
        # 保留原始分数，便于调试召回与重排结果
        raw = doc.metadata.get(score_key)
        if raw is not None:
            try:
                doc.metadata["raw_score"] = float(raw)
            except (ValueError, TypeError):
                pass

        # 改用基于排名的归一化分数，避免不同检索器分值体系不一致
        doc.metadata[score_key] = (total - rank) / total
        doc.metadata["rank"] = rank + 1

    logger.debug("基于排名分配分数完成: 共 {} 个文档", total)
    return docs


def _coerce_int_id(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _pick_nearest_heading_id(source_element_ids: List[Any], heading_node_ids: List[Any]) -> Optional[int]:
    """补 1 个最近标题的 element_id，用于前端更稳定地定位高亮区域。

    规则：
    1. 优先选择落在正文跨度内、且与正文元素相邻的标题 ID。
    2. 否则选择位于正文前方、距离最近的标题 ID。
    """
    src_ids = [sid for sid in (_coerce_int_id(v) for v in (source_element_ids or [])) if sid is not None]
    hdr_ids = [hid for hid in (_coerce_int_id(v) for v in (heading_node_ids or [])) if hid is not None]
    if not src_ids or not hdr_ids:
        return None

    src_set = set(src_ids)
    hdr_set = set(hdr_ids)
    # 如果 source_element_ids 已经包含标题节点，则无需额外补充
    if src_set & hdr_set:
        return None

    min_src = min(src_set)
    max_src = max(src_set)

    in_span = [hid for hid in hdr_set if min_src <= hid <= max_src and hid not in src_set]
    if in_span:
        adjacent = [hid for hid in in_span if (hid - 1 in src_set) or (hid + 1 in src_set)]
        candidates = adjacent or in_span
        return min(candidates, key=lambda hid: (abs(hid - min_src), hid))

    before = [hid for hid in hdr_set if hid < min_src and hid not in src_set]
    return max(before) if before else None


def _resolve_doc_kinds(summary_mode: str) -> Optional[List[str]]:
    """将 summary_mode 转成底层向量库过滤所需的 doc_kind 列表。"""
    mode = str(summary_mode or "off").strip().lower()
    if mode == "only":
        return ["summary"]
    if mode == "on":
        return ["chunk", "summary", ""]
    return ["chunk", ""]


def _normalize_source_element_ids(values: Any) -> tuple[int, ...]:
    """将任意格式的 source element id 规范化为升序且去重的整数元组。"""
    if isinstance(values, list):
        normalized = {_coerce_int_id(value) for value in values}
        return tuple(sorted(value for value in normalized if value is not None))
    value = _coerce_int_id(values)
    return (value,) if value is not None else ()


def _normalize_heading_path_for_dedupe(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item or "").strip())
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(part for part in text.split("/") if part)


def _get_doc_score(doc: Any) -> float:
    """优先读取 rerank_score，其次读取 score，用于去重时保留高分结果。"""
    metadata = getattr(doc, "metadata", {}) or {}
    for key in ("rerank_score", "score"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _dedupe_child_docs(child_docs: List[Any]) -> List[Any]:
    """对子切片做精确去重。

    使用 `(file_id, source_element_ids)` 作为键，仅保留得分更高的那条结果。
    """
    passthrough: List[Any] = []
    ordered_keys: List[Any] = []
    best_by_key: dict[Any, Any] = {}

    for doc in child_docs or []:
        metadata = getattr(doc, "metadata", {}) or {}
        child_id = str(metadata.get("child_id") or "").strip()
        if child_id:
            dedupe_key: Any = ("child_id", child_id)
        else:
            file_id = str(metadata.get("file_id") or "")
            source_ids = _normalize_source_element_ids(metadata.get("source_element_ids", []))
            if not file_id or not source_ids:
                passthrough.append(doc)
                continue
            dedupe_key = (
                "metadata",
                file_id,
                source_ids,
                str(metadata.get("split_strategy") or ""),
                metadata.get("table_row_start"),
                metadata.get("table_row_end"),
                _normalize_heading_path_for_dedupe(metadata.get("heading_path")),
                str(metadata.get("type") or ""),
            )
        existing = best_by_key.get(dedupe_key)
        if existing is None:
            ordered_keys.append(dedupe_key)
            best_by_key[dedupe_key] = doc
            continue
        if _get_doc_score(doc) > _get_doc_score(existing):
            best_by_key[dedupe_key] = doc

    deduped = list(passthrough)
    for key in ordered_keys:
        deduped.append(best_by_key[key])
    return deduped


@observe(name="rag_retrieve_grandchildren", as_type="retriever", capture_input=False, capture_output=False)
async def _retrieve_grandchildren(
    question_text: str,
    top_k: int,
    file_ids: Optional[List[str]] = None,
    folder_id: Optional[str] = None,
    search_type: Optional[str] = None,
    doc_kinds: Optional[List[str]] = None,
    use_hybrid: bool = True,
    output_fields: Optional[list[str] | tuple[str, ...]] = None,
):
    """检索孙切片；混合检索由底层向量库适配器负责。"""
    try:
        retriever = VectorStoresService.get_grandchildren_retriever(
            top_k=top_k,
            file_ids=file_ids,
            folder_id=folder_id,
            search_type=search_type,
            doc_kinds=doc_kinds,
            use_hybrid=use_hybrid,
            output_fields=output_fields,
        )
        return await retriever.aget_relevant_documents(question_text)
    except Exception as error:
        logger.error("孙切片检索失败: {}", error)
        return []


@observe(name="rag_retrieve_children_via_grandchildren", as_type="retriever", capture_input=False, capture_output=False)
async def _retrieve_children_via_grandchildren(
    question_text: str,
    top_k: int,
    file_ids: Optional[List[str]] = None,
    folder_id: Optional[str] = None,
    search_type: Optional[str] = None,
    doc_kinds: Optional[List[str]] = None,
    use_hybrid: bool = True,
):
    """PCR（Grandchild -> Child）：检索孙切片，但返回子切片。"""
    try:
        retriever = VectorStoresService.get_child_retriever_via_grandchildren(
            top_k=top_k,
            file_ids=file_ids,
            folder_id=folder_id,
            search_type=search_type,
            doc_kinds=doc_kinds,
            use_hybrid=use_hybrid,
        )
        return await retriever.ainvoke(question_text)
    except Exception as error:
        logger.error("子切片(PCR)检索失败: {}", error)
        return []


async def run_batch_query(body: BatchQueryRequest) -> List[BatchQueryItem]:
    """批量执行 RAG 查询。

    工作流：
    1. 按问题检索孙切片候选。
    2. 可选按 `file_ids`、`folder_id`、`search_type`、`summary_mode` 过滤。
    3. 若 `return_child_chunks=True`，则经 PCR 返回子切片；否则直接返回孙切片。
       `search_type=image` 时始终返回图片孙切片，忽略 `return_child_chunks`
       （PCR 会替换成可能缺少 image_storage_path 的子切片）。
    """

    results: List[BatchQueryItem] = []
    allowed_files = set(body.file_ids or [])
    folder_id_filter = body.folder_id
    rerank_default = RAG_SETTINGS.rerank_enabled
    doc_kinds = _resolve_doc_kinds(body.summary_mode)
    hybrid_flag = (
        body.hybrid_enabled
        if body.hybrid_enabled is not None
        else RAG_SETTINGS.hybrid_enabled
    )
    use_hybrid = bool(hybrid_flag)

    logger.info(
        "收到 RAG 查询请求: questions={}, top_k={}, return_child_chunks={}, rerank_enabled={}, "
        "hybrid_enabled={}, use_hybrid_pipeline={}, search_type={}, summary_mode={}, doc_kinds={}",
        len(body.questions),
        body.top_k,
        body.return_child_chunks,
        body.rerank_enabled,
        hybrid_flag,
        use_hybrid,
        body.search_type,
        body.summary_mode,
        doc_kinds,
    )

    for idx, q in enumerate(body.questions):
        logger.info("处理问题 {}/{}: {}", idx + 1, len(body.questions), q.question_text[:80])

        # 扩大候选集，给重排器和精确去重留出空间
        candidate_limit = max(body.top_k * 5, 60)
        # 允许通过配置继续放大候选窗口
        if RAG_SETTINGS.rerank_top_k is not None:
            candidate_limit = max(candidate_limit, RAG_SETTINGS.rerank_top_k)

        file_ids_list = list(allowed_files) if allowed_files else None

        # 情况 A：PCR，检索孙切片但返回子切片
        # image 的 path/type 在孙切片上，PCR 替换后常丢，故 search_type=image 不走 PCR
        if body.return_child_chunks and body.search_type != SearchType.IMAGE:
            try:
                child_docs = await _retrieve_children_via_grandchildren(
                    question_text=q.question_text,
                    top_k=candidate_limit,
                    file_ids=file_ids_list,
                    folder_id=folder_id_filter,
                    search_type=body.search_type,
                    doc_kinds=doc_kinds,
                    use_hybrid=use_hybrid,
                )
                logger.info("问题 {} PCR 召回完成: {} 条子切片候选", idx + 1, len(child_docs))
            except Exception as e:
                logger.exception("问题 {} 的 PCR 检索异常: {}", idx + 1, e)
                continue

            # 先重排，再做精确去重与分数归一
            try:
                reranker = VectorStoresService.get_reranker()
                enabled = body.rerank_enabled if body.rerank_enabled is not None else rerank_default
                if reranker and enabled and child_docs:
                    threshold = RAG_SETTINGS.rerank_score_threshold
                    docs_text = [d.page_content or "" for d in child_docs]
                    order = await reranker.arerank(q.question_text, docs_text, top_k=len(child_docs))
                    reranked = []
                    for item in order:
                        try:
                            doc_index = int(item.get("index"))
                        except Exception:
                            continue
                        if not (0 <= doc_index < len(child_docs)):
                            continue
                        doc = child_docs[doc_index]
                        score = item.get("score")
                        if score is not None:
                            try:
                                score_f = float(score)
                            except (TypeError, ValueError):
                                score_f = None
                            if score_f is not None:
                                doc.metadata["rerank_score"] = score_f
                                # 相关性兜底过滤：低于阈值的切片视为与查询文本不相关，直接丢弃
                                if score_f < threshold:
                                    continue
                        reranked.append(doc)
                    child_docs = reranked
            except Exception as e:
                logger.warning("问题 {} 重排失败: {}", idx + 1, e)

            # 同一个 file/source_element_ids 只保留得分更高的结果
            deduped_child_docs = _dedupe_child_docs(child_docs)
            if len(deduped_child_docs) != len(child_docs):
                logger.debug(
                    "问题 {} 子切片精确去重: {} -> {}",
                    idx + 1,
                    len(child_docs),
                    len(deduped_child_docs),
                )
            child_docs = deduped_child_docs

            # 对最终顺序重新分配统一分值
            assign_rank_scores(child_docs, score_key="score")
            final_children = child_docs[: body.top_k]

            child_items: List[BatchQueryItem] = []
            for child_doc in final_children:
                cid = ""
                try:
                    metadata = dict(child_doc.metadata or {})
                    fid = str(metadata.get("file_id") or "")
                    cid = str(metadata.get("child_id") or "")

                    score_val = metadata.get("score")
                    try:
                        score_f = float(score_val) if score_val is not None else 0.0
                    except Exception:
                        score_f = 0.0

                    # 合并正文元素与最近标题，提升前端高亮命中率
                    source_elem_ids = metadata.get("source_element_ids", [])
                    if not isinstance(source_elem_ids, list):
                        source_elem_ids = [source_elem_ids] if source_elem_ids else []

                    heading_node_ids = metadata.get("heading_node_ids", [])
                    if not isinstance(heading_node_ids, list):
                        heading_node_ids = [heading_node_ids] if heading_node_ids else []

                    closest_heading_id = _pick_nearest_heading_id(source_elem_ids, heading_node_ids)
                    elem_ids_set = {eid for eid in (_coerce_int_id(v) for v in source_elem_ids) if eid is not None}
                    if closest_heading_id is not None:
                        elem_ids_set.add(closest_heading_id)

                    child_items.append(
                        BatchQueryItem(
                            question_id=q.question_id,
                            question_text=q.question_text,
                            element_ids=sorted(elem_ids_set),
                            file_id=fid,
                            file_name=metadata.get("file_name") or "",
                            score=int(round(score_f * 1000)),
                            text=child_doc.page_content,
                            image_storage_path=metadata.get("image_storage_path"),
                            metadata=metadata,
                        )
                    )
                except Exception as e:
                    logger.warning("构建子切片结果异常: {}, child_id={}", e, cid)
                    continue

            results.extend(child_items)
            continue

        # 情况 B：直接返回孙切片
        try:
            grand_docs = await _retrieve_grandchildren(
                question_text=q.question_text,
                top_k=candidate_limit,
                file_ids=file_ids_list,
                folder_id=folder_id_filter,
                search_type=body.search_type,
                doc_kinds=doc_kinds,
                use_hybrid=use_hybrid,
                output_fields=_GRANDCHILD_BATCH_QUERY_OUTPUT_FIELDS,
            )
            logger.info("问题 {} 召回完成: {} 条孙切片候选", idx + 1, len(grand_docs))
        except Exception as e:
            logger.exception("问题 {} 的检索异常: {}", idx + 1, e)
            continue

        question_items: List[BatchQueryItem] = []
        for grand_doc in grand_docs[: body.top_k]:
            metadata = dict(grand_doc.metadata or {})
            fid = str(metadata.get("file_id") or "")
            score_val = metadata.get("score")
            try:
                score_f = float(score_val) if score_val is not None else 0.0
            except Exception:
                score_f = 0.0

            elem_id = _coerce_int_id(metadata.get("element_id"))
            if elem_id is not None:
                chosen_elem_ids = [elem_id]
            else:
                chosen_elem_ids = list(_normalize_source_element_ids(metadata.get("source_element_ids", [])))

            question_items.append(
                BatchQueryItem(
                    question_id=q.question_id,
                    question_text=q.question_text,
                    element_ids=chosen_elem_ids,
                    file_id=fid,
                    file_name=metadata.get("file_name") or "",
                    score=int(round(score_f * 1000)),
                    text=grand_doc.page_content,
                    image_storage_path=metadata.get("image_storage_path"),
                    metadata=metadata,
                )
            )
        results.extend(question_items)

    logger.info("RAG 查询完成: 返回 {} 条结果", len(results))
    return results
