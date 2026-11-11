"""文档结构召回服务。

从 MinIO 加载文档的 converted.json，构建标题层级树，通过大模型选择相关标题后，
提取对应章节的全文内容并返回。

流程：
    1. 根据 file_id 从 MinIO 加载 converted.json
    2. 通过 tree_builder.build_tree 构建标题树
    3. 展开标题节点的层级编号（如 1.、1.1.、2.1.1.）
    4. 调用大模型根据问题选择相关标题
    5. 按 max_tokens 和 max_section_num 截取选中章节的全文
    6. 按 questionId 分组组装响应
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.base.storage_service import storage_service
from src.applications.parsing.utils.mineru_storage import MINERU_OUTPUT_BUCKET
from src.utils import json_utils
from src.utils.tokenizer import Tokenizer
from src.agents.content_generate_agent.utils import (
    apply_prompt_template,
    ainvoke_llm_with_structured_output,
)
from src.agents.content_generate_agent.retrieval_workflow.nodes import (
    ChapterSelectionResponse,
    BatchChapterSelectionResponse,
)
from src.agents.content_generate_agent.hierarchical_summary.tree_builder import (
    Node,
    build_tree,
    MAX_HEADING_LEVEL,
)
from src.agents.content_generate_agent.input_models import (
    DocumentItems,
    TextItem,
    TableItem,
    ImageItem,
    ListItem,
)
from src.applications.rag.schemas import BatchQueryItemHttp
from src.applications.rag.config.rag_settings import RAG_SETTINGS

STRUCTURE_RECALL_MAX_HEADING_LEVEL = int(os.getenv("STRUCTURE_RECALL_MAX_HEADING_LEVEL", 3))
"""构造标题树的最大标题等级，通过环境变量 ``STRUCTURE_RECALL_MAX_HEADING_LEVEL`` 控制，默认沿用 tree_builder 的 ``MAX_HEADING_LEVEL``。"""

DEFAULT_SCORE_THRESHOLD = 5

_tokenizer = Tokenizer()


class _FileLabel:
    """文件名称签，用于在标题树中标记文件节点。"""
    __slots__ = ("value", "file_id")

    def __init__(self, value: str, file_id: str = ""):
        self.value = value
        self.file_id = file_id


class _SheetLabel:
    """Excel sheet 名称签，用于在标题树中标记 sheet 节点。"""
    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value


def _get_node_title(node: Node) -> tuple[str, int, Optional[int]]:
    """从节点中提取标题文本、层级和页码。"""
    for item in node.content:
        if isinstance(item, _FileLabel):
            return (item.value, 0, None)
        if isinstance(item, _SheetLabel):
            return (item.value, 1, None)
        if isinstance(item, TextItem) and getattr(item, "headingLevel", None) is not None:
            return (item.value or "", item.headingLevel or 1, item.page)
    return ("", 1, None)


def _get_node_element_ids(node: Node) -> List[int]:
    """收集章节节点的 element_id。"""
    ids: List[int] = []

    for item in _get_section_items(node):
        if isinstance(item, (_FileLabel, _SheetLabel)):
            continue
        eid = getattr(item, "element_id", None)
        if eid is not None:
            ids.append(eid)
    return ids


def _flatten_headings(root: Node) -> List[dict]:
    """前序遍历标题树，展开为带层级编号的扁平列表。

    每项包含 index（全局序号）、number（如 1.、1.1.）、title、level、
    page 和 node，用于后续 LLM 选择。

    Args:
        root: 标题树的根节点。

    Returns:
        扁平标题列表，每项为包含标题信息的字典。
    """
    result: List[dict] = []

    def _walk(node: Node, prefix: str = ""):
        for i, child in enumerate(node.childs, 1):
            num = f"{prefix}{i}." if prefix else f"{i}."
            title, level, page = _get_node_title(child)
            result.append({
                "index": len(result) + 1,
                "number": num,
                "title": title,
                "level": level,
                "page": page,
                "node": child,
            })
            _walk(child, num)

    _walk(root)
    return result


def _is_xlsx_file(file_name: str) -> bool:
    return Path(file_name).suffix.lower() == ".xlsx"


def _build_xlsx_sheet_nodes(items: List[Any]) -> List[Node]:
    sheet_nodes: List[Node] = []
    sheet_map: Dict[tuple[int, str], Node] = {}

    for item in items:
        if getattr(item, "type", None) != "excel_table":
            continue
        sheet_name = (getattr(item, "sheetName", "") or "").strip()
        sheet_index = int(getattr(item, "sheetIndex", 0) or 0)
        title = sheet_name or f"Sheet{sheet_index + 1}"
        key = (sheet_index, title)

        if key not in sheet_map:
            node = Node(content=[_SheetLabel(title)])
            sheet_map[key] = node
            sheet_nodes.append(node)
        sheet_map[key].content.append(item)

    return sheet_nodes


async def _get_file_name(file_id: str) -> str:
    """从 MinIO 的 ``{file_id}/original/`` 目录获取原始文件名。"""
    try:
        objects = await storage_service.list_objects(
            MINERU_OUTPUT_BUCKET,
            prefix=f"{file_id}/original/",
            recursive=False,
        )
        if objects:
            original_path = objects[0]
            return original_path.rsplit("/", 1)[-1] or file_id
    except Exception:
        logger.exception("获取文件名失败 | file_id={}", file_id)
    return file_id


def _build_structured_input(flat_titles: List[dict]) -> tuple[str, Dict[int, dict]]:
    """将扁平标题列表构建为层级 JSON 结构，同时建立 chapter 编号到条目的反向映射。

    JSON 结构形如：
        {"documents": [{"chapter": 1, "file_name": "...", "outline": [...]}]}

    每个文件/标题节点都有唯一的 ``chapter`` 编号，对应 ``flat_titles`` 中的 ``index``。
    目录节点用 ``outline`` / ``children`` 嵌套表达层级关系。

    Args:
        flat_titles: ``_flatten_headings`` 返回的扁平标题列表。

    Returns:
        (JSON 字符串, chapter 编号到条目的映射)。
    """
    chapter_to_entry: Dict[int, dict] = {}
    documents: List[dict] = []
    current_doc: Optional[dict] = None
    outline_stack: List[tuple] = []  # (parent_children_list, level)

    for t in flat_titles:
        chapter_to_entry[t["index"]] = t
        if t["level"] == 0:
            current_doc = {
                "chapter": t["index"],
                "file_name": t["title"].replace("文件: ", ""),
                "outline": [],
            }
            documents.append(current_doc)
            outline_stack = [(current_doc["outline"], 0)]
        else:
            entry = {
                "chapter": t["index"],
                "level": t["level"],
                "title": t["title"],
                "children": [],
            }
            while outline_stack and outline_stack[-1][1] >= t["level"]:
                outline_stack.pop()
            parent_children = outline_stack[-1][0]
            parent_children.append(entry)
            outline_stack.append((entry["children"], t["level"]))

    def _strip_empty_children(node: dict) -> dict:
        if "outline" in node:
            node["outline"] = [_strip_empty_children(c) for c in node["outline"]]
            if not node["outline"]:
                del node["outline"]
        if "children" in node:
            node["children"] = [_strip_empty_children(c) for c in node["children"]]
            if not node["children"]:
                del node["children"]
        return node

    documents = [_strip_empty_children(d) for d in documents]
    if not documents:
        return "{}", chapter_to_entry

    return json_utils.dumps({"documents": documents}, ensure_ascii=False), chapter_to_entry


async def _select_structure_via_llm_batch(
    questions: List[Dict[str, str]],
    titles_text: str,
    answer_lang: str = "zh_CN",
) -> BatchChapterSelectionResponse:
    """批量调用大模型，一次请求处理多个问题以复用文档结构的 KV cache。

    将文档结构作为所有问题共享的前缀内容，仅在末尾拼接不同的问题，大模型
    可以复用 titles_text 部分的 KV cache，避免 N 次独立调用时的重复计算。

    Args:
        questions: 问题列表，每项为 ``{"question_id": ..., "question_text": ...}``。
        titles_text: ``_build_structured_input`` 产出的 JSON 文档结构字符串。
        answer_lang: prompt 语言（zh_CN 或 en_US）。

    Returns:
        LLM 返回的批量章节选择结果。
    """
    model_id = RAG_SETTINGS.structure_recall_model_id
    max_tokens = RAG_SETTINGS.structure_recall_prompt_max_tokens
    questions_text = json_utils.dumps(
        [
            {"question_index": i, "question": q["question_text"]}
            for i, q in enumerate(questions)
        ],
        ensure_ascii=False,
    )

    base_prompt = apply_prompt_template(
        "structure_selection_batch",
        {"titles_text": "", "questions_text": ""},
        locale=answer_lang,
    )
    base_tokens = _tokenizer(base_prompt)
    titles_tokens = _tokenizer(titles_text)
    questions_tokens = _tokenizer(questions_text)

    logger.info(
        "prompt截断前 | base_tokens={} titles_tokens={} questions_tokens={} "
        "total={} max_tokens={} question_count={}",
        base_tokens, titles_tokens, questions_tokens,
        base_tokens + titles_tokens + questions_tokens, max_tokens, len(questions),
    )

    # 分配策略：优先保证问题完整（问题不可见则完全无法选择），剩余预算给文档结构
    available_tokens = max_tokens - base_tokens
    if available_tokens <= 0:
        logger.error(
            "structure_selection_batch 基础prompt token数({})已超过限制({})",
            base_tokens, max_tokens,
        )
        safe_titles_text = ""
        safe_questions_text = ""
    else:
        if questions_tokens <= available_tokens:
            safe_questions_text = questions_text
            titles_budget = available_tokens - questions_tokens
        else:
            logger.warning(
                "问题列表超过token预算，将被截断 | questions_tokens={} available={} question_count={}",
                questions_tokens, available_tokens, len(questions),
            )
            safe_questions_text = _tokenizer.ensure(questions_text, max_tokens=available_tokens)
            titles_budget = 0
        safe_titles_text = _tokenizer.ensure(titles_text, max_tokens=titles_budget) if titles_budget > 0 else ""

    prompt = apply_prompt_template(
        "structure_selection_batch",
        {"titles_text": safe_titles_text, "questions_text": safe_questions_text},
        locale=answer_lang,
    )

    response = await ainvoke_llm_with_structured_output(
        prompt=prompt,
        model_id=model_id,
        response_type=BatchChapterSelectionResponse,
    )
    return response


def _resolve_selections(
    response: ChapterSelectionResponse,
    chapter_to_entry: Dict[int, dict],
    score_threshold: int = DEFAULT_SCORE_THRESHOLD,
) -> List[dict]:
    """解析 LLM 返回的章节选择结果，通过 chapter 编号映射定位节点。

    每个 ``chapter_index`` 对应 JSON 结构中的 ``chapter`` 字段，直接查映射表。

    Args:
        response: LLM 返回的章节选择结果。
        chapter_to_entry: ``_build_structured_input`` 产出的 chapter 编号到条目的映射。
        score_threshold: 相关性评分阈值，低于此值的标题被过滤。

    Returns:
        去重后的最终选中标题列表，每项包含标题信息及 relevance_score、selection_reason。
    """
    seen: set = set()
    result: List[dict] = []
    for s in response.selected_chapters:
        if s.relevance_score < score_threshold:
            continue
        if s.chapter_index in seen:
            continue
        entry = chapter_to_entry.get(s.chapter_index)
        if entry is None:
            continue
        seen.add(s.chapter_index)
        result.append({
            "index": s.chapter_index,
            "number": entry["number"],
            "title": entry["title"],
            "level": entry["level"],
            "page": entry["page"],
            "node": entry["node"],
            "relevance_score": s.relevance_score,
            "selection_reason": s.selection_reason,
        })
    return result


def _item_to_text(item: Any) -> str:
    """将单个文档元素转换为文本。

    按元素类型提取文本字段：
        - TextItem / ListItem：value 或 md
        - TableItem：md 或 html
        - ImageItem：拼接 image_caption 和 image_footnote
        - 其他：按属性回退

    Args:
        item: 文档元素对象。

    Returns:
        元素对应的文本，无内容时返回空字符串。
    """
    if isinstance(item, (_FileLabel, _SheetLabel)):
        return ""
    if isinstance(item, TextItem):
        return item.value or item.md or ""
    elif isinstance(item, TableItem):
        return item.md or item.html or ""
    elif isinstance(item, ImageItem):
        caption = item.image_caption or ""
        footnote = item.image_footnote or ""
        if caption or footnote:
            return f"[图片: {caption} {footnote}]"
        return ""
    elif isinstance(item, ListItem):
        return item.value or item.md or ""
    else:
        return getattr(item, "value", None) or getattr(item, "md", None) or ""


def _get_section_items(node: Node) -> List[Any]:
    if node.childs and all(isinstance(item, _FileLabel) for item in node.content):
        return [
            item
            for child in node.childs
            for item in child.content
        ]
    return node.content


def _collect_section_text(node: Node) -> str:
    """收集标题节点的文本内容。

    Args:
        node: 标题节点。

    Returns:
        拼接后的完整章节文本。
    """
    parts: List[str] = []

    for item in _get_section_items(node):
        text = _item_to_text(item)
        if text:
            parts.append(text)

    return "\n".join(parts)


def _get_node_file_info(node: Node, file_names: Dict[str, str]) -> tuple[str, str]:
    """从节点内容中提取 file_id 和 file_name。

    优先从 ``_FileLabel`` 的 ``file_id`` 属性获取，否则从 content 中
    第一个带 ``file_id`` 的元素提取。

    Args:
        node: 标题节点。
        file_names: file_id 到 file_name 的映射表。

    Returns:
        (file_id, file_name) 元组。
    """
    for item in node.content:
        if isinstance(item, _FileLabel) and item.file_id:
            fid = item.file_id
            return (fid, file_names.get(fid, fid))
    for item in node.content:
        fid = getattr(item, "file_id", None)
        if fid:
            fid_s = str(fid)
            return (fid_s, file_names.get(fid_s, fid_s))
    return ("", "")


def _extract_sections(
    selected_titles: List[dict],
    file_names: Dict[str, str],
    question_id: str,
    question_text: str,
    max_tokens: int = 30000,
    max_structure_num: int = 5,
) -> List[BatchQueryItemHttp]:
    """从选中标题节点中提取章节全文，构造响应对象。

    按标题序号排序，可限制 structure 数量和 token 数，<= 0 时不限制。
    file_id 和 file_name 从节点内部的 content 元素中自动提取。

    Args:
        selected_titles: ``_resolve_selections`` 返回的最终选中标题列表。
        file_names: file_id 到 file_name 的映射表。
        question_id: 问题 ID。
        question_text: 问题文本。
        max_tokens: 当前问题的 section 文本 token 上限，<= 0 不限制。
        max_structure_num: 最多返回的 structure 数，<= 0 不限制。

    Returns:
        ``BatchQueryItemHttp`` 列表，兼容 RAG 召回接口返回结构。
    """
    sections: List[BatchQueryItemHttp] = []
    total_tokens = 0

    selected_titles.sort(key=lambda x: x["index"])
    if max_structure_num > 0:
        selected_titles = selected_titles[:max_structure_num]

    for title in selected_titles:
        node = title["node"]
        section_text = _collect_section_text(node)
        section_tokens = _tokenizer(section_text)

        if max_tokens > 0:
            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break
            if section_tokens > remaining:
                section_text = _tokenizer.ensure(
                    section_text, max_tokens=remaining
                )

        total_tokens += _tokenizer(section_text)

        fid, fname = _get_node_file_info(node, file_names)

        sections.append(BatchQueryItemHttp(
            questionId=question_id,
            questionText=question_text,
            elementIds=_get_node_element_ids(node),
            fileId=fid,
            fileName=fname,
            score=title["relevance_score"],
            text=section_text,
        ))

    return sections


async def _build_merged_title_tree(
    file_ids: List[str],
) -> tuple[List[Node], Dict[str, str], str, Dict[int, dict]]:
    """加载文件、构建标题树、合并为跨文件树并生成结构化 JSON。

    Args:
        file_ids: 文件 ID 列表。

    Returns:
        (file_roots, file_names, titles_text, chapter_to_entry)
        若加载失败返回空列表则前三项为空，调用方应直接返回 []。
    """
    file_roots: List[Node] = []
    file_names: Dict[str, str] = {}

    for file_id in file_ids:
        object_name = f"{file_id}/converted.json"
        try:
            raw = await storage_service.download(MINERU_OUTPUT_BUCKET, object_name)
            elements = json_utils.loads(raw.decode("utf-8"))
        except Exception:
            logger.exception("加载文件失败 | file_id={}", file_id)
            continue

        if not elements:
            logger.warning("文件无内容 | file_id={}", file_id)
            continue

        fname = await _get_file_name(file_id)
        file_names[file_id] = fname

        try:
            doc = DocumentItems.model_validate(elements)
            items = doc.root
        except Exception:
            logger.exception("解析文档元素失败 | file_id={}", file_id)
            continue

        file_root = build_tree(items, max_heading_level=STRUCTURE_RECALL_MAX_HEADING_LEVEL)
        sheet_nodes = _build_xlsx_sheet_nodes(items) if _is_xlsx_file(fname) else []
        if sheet_nodes:
            file_node = Node(content=[_FileLabel(f"文件: {fname}", file_id=file_id)], childs=sheet_nodes)
            logger.info(
                "XLSX文件使用sheet作为标题 | file_id={} file_name={} sheet_count={}",
                file_id, fname, len(sheet_nodes),
            )
        elif not file_root.childs:
            file_node = Node(content=[_FileLabel(f"文件: {fname}", file_id=file_id), *file_root.content])
            logger.info("文件无标题，使用文件名作为标题 | file_id={} file_name={}", file_id, fname)
        else:
            file_node = Node(content=[_FileLabel(f"文件: {fname}", file_id=file_id)], childs=list(file_root.childs))
        file_roots.append(file_node)
        logger.info(
            "标题树构建完成 | file_id={} file_name={} title_count={}",
            file_id, fname, len(file_node.childs),
        )

    if not file_roots:
        return [], {}, "", {}

    merged_root = Node(childs=list(file_roots))
    flat_titles = _flatten_headings(merged_root)
    titles_text, chapter_to_entry = _build_structured_input(flat_titles)

    logger.info(
        "跨文件标题合并完成 | file_count={} total_title_count={}",
        len(file_roots), len(flat_titles),
    )
    return file_roots, file_names, titles_text, chapter_to_entry


async def recall_structure_batch(
    questions: List[Dict[str, str]],
    file_ids: List[str],
    max_tokens: int ,
    max_structure_num: int ,
    answer_lang: str = "zh_CN",
) -> List[BatchQueryItemHttp]:
    """文档结构批量召回主入口。

    将多个问题对同一批文件的结构召回合并为一次大模型调用，
    使文档结构的 KV cache 在多个问题间共享，减少重复计算。

    流程：
        1. 加载所有文件并构建标题树（所有问题共享）
        2. 合并为跨文件树并构建结构化 JSON
        3. 一次 LLM 调用返回所有问题的章节选择
        4. 按问题分别提取选中章节的全文

    Args:
        questions: 问题列表，每项为 ``{"question_id": ..., "question_text": ...}``。
        file_ids: 文件 ID 列表，只查询这些文件。
        max_tokens: 每个问题返回正文的 token 上限，-1=不限制。
        max_structure_num: 每个问题最多返回的 structure 数，-1=不限制。
        answer_lang: prompt 语言（zh_CN 或 en_US）。

    Returns:
        ``BatchQueryItemHttp`` 列表，按问题顺序排列。
    """
    if not questions:
        return []

    if not file_ids:
        logger.warning("file_ids 为空，无法进行批量召回")
        return []

    file_roots, file_names, titles_text, chapter_to_entry = await _build_merged_title_tree(file_ids)
    if not file_roots:
        return []

    logger.info(
        "文档结构批量召回 | question_count={} file_count={}",
        len(questions), len(file_roots),
    )

    # 第三阶段：一次 LLM 调用，处理所有问题（共享 titles_text 的 KV cache）
    try:
        batch_response = await _select_structure_via_llm_batch(
            questions=questions,
            titles_text=titles_text,
            answer_lang=answer_lang,
        )
        logger.info("批量LLM结构选择响应 | question_count={}", len(batch_response.results))
    except Exception:
        logger.exception("批量LLM结构选择失败")
        return []

    # 第四阶段：按问题分别解析选择和提取章节
    all_sections: List[BatchQueryItemHttp] = []

    for question_result in batch_response.results:
        if question_result.question_index >= len(questions):
            logger.warning(
                "LLM返回了无效的问题序号 | question_index={} total_questions={}",
                question_result.question_index, len(questions),
            )
            continue

        question_info = questions[question_result.question_index]
        qid = question_info["question_id"]
        qtext = question_info["question_text"]

        # 构建该问题的 ChapterSelectionResponse 以便复用 _resolve_selections
        single_response = ChapterSelectionResponse(
            selected_chapters=list(question_result.selected_chapters)
        )
        selected = _resolve_selections(single_response, chapter_to_entry)

        logger.info(
            "问题结构选择完成 | question_id={} selected_count={} total_selected_by_llm={}",
            qid, len(selected), len(question_result.selected_chapters),
        )

        if not selected:
            continue

        sections = _extract_sections(
            selected_titles=selected,
            file_names=file_names,
            question_id=qid,
            question_text=qtext,
            max_tokens=max_tokens,
            max_structure_num=max_structure_num,
        )
        all_sections.extend(sections)

    logger.info(
        "批量章节提取完成 | file_count={} question_count={} total_section_count={}",
        len(file_roots), len(questions), len(all_sections),
    )
    return all_sections
