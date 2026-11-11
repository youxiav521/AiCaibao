from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple
import re
import asyncio
import time
from pydantic import BaseModel, Field
from .config import create_default_configs
from src.base.langfuse_observe import trace_span
from .workflow import ContentGenService
from .input_models import DocumentItems, Item
from .retrieval_workflow.workflow import RetrivalService
from .retrieval_workflow.state import SelectedContent, Block
from .retrieval_workflow.context_reduction_utils import build_item_key_to_item, resolve_piece_text, filter_long_context_if_needed
from .utils import (
    count_tokens,
    apply_prompt_template,
    build_prompt_state,
    ensure_within_token_limit,
    ainvoke_llm_with_prompt,
    ainvoke_llm_with_structured_output,
    abatch_llm_with_prompt,
    ensure_retrieval_queries_for_tasks,
    contains_cjk,
    get_localized_message,
    normalize_locale,
    judge_simple_question,
    log_error_type_and_message,
)
from .retrieval_workflow.block_manager import BlockManager
from .thinking_steps_builder import attach_thinking_steps_to_answer
from .models import (
    FinalAnswer,
    ContentGenerateResponseEvent,
    QuestionTask,
    CodingAgentTool,
    ChunkIndexRetrievalTool,
    OtherDataSource,
)
from .retrieval_workflow.context_reduction_utils import rename_selected_content_for_single_question, sort_and_rename_retrieval_result
from .nodes import reporter_core
from src.applications.parsing.utils.mineru_storage import load_mineru_items
from src.applications.rag.services.summary_tree_service import load_summary_tree_if_available
from src.applications.rag.services.summary_service_utils import (
    coerce_int,
    get_field_value,
    normalize_nonempty_string,
)
import json
from loguru import logger

_PAGE_LABEL_RE = re.compile(r"^第\s*(\d+)\s*页$")


class ThinkingStepTranslationBatch(BaseModel):
    """Batch translation payload for localized thinking-step display."""

    translations: List[str] = Field(
        default_factory=list,
        description="English translations in the same order as the input texts.",
    )

class ContentGenerateAgent:
    def __init__(self, model_id: Optional[str] = "model_for_content_generation", answer_lang: str = "zh_CN"):
        self.model_id = model_id
        logger.info(
            f"[i18n] ContentGenerateAgent初始化收到语言参数: "
            f"model_id={model_id}, answer_lang={answer_lang}"
        )
        self.answer_lang = normalize_locale(answer_lang)
        logger.info(
            f"[i18n] ContentGenerateAgent初始化语言归一化完成: "
            f"model_id={model_id}, 归一化后answer_lang={self.answer_lang}"
        )
        
        # 根据配置常量选择配置
        self.test_cfg = create_default_configs()
        logger.info(
            f"ContentGenerateAgent 初始化: model_id={self.model_id}, "
            f"answer_lang={self.answer_lang}, context_limit={self.test_cfg.context_token_limit}"
        )
        self._summary_tree_cache: Dict[Tuple[Tuple[str, int], ...], Any] = {}
        self._summary_tree_cache_lock = asyncio.Lock()
        
        # self.block_manager = BlockManager()
        # self.retriever = LLMRetriever(test_config=self.test_cfg, model_id=self.model_id)
        # self.generator = ContentGenerator(test_config=self.test_cfg)
    
    def _get_retrieval_result_for_question(self, question_id: str, retrieval_data: Dict[str, Any]) -> Optional[SelectedContent]:
        """
        从检索数据的数据结构中获取指定问题question_id的检索结果。
        """

        # 从selected_content中查找对应的问题
        selected_content_list = retrieval_data.get("selected_content", [])
        for content in selected_content_list:
            if getattr(content, "question_id", None) == question_id:
                logger.info(f'找到问题 {question_id} {getattr(content, "question", "")} 的检索结果，包含 {len(content.selected_pieces)} 个片段')
                return content
        
        logger.error(f"警告：未找到问题 {question_id} 的检索结果")
        return None
    
    def _build_question_meta_by_id(self, question_tasks: List[QuestionTask]) -> Dict[str, Dict[str, Any]]:
        """
        从问题任务列表构建问题元数据查找表。
        
        Args:
            question_tasks: QuestionTask 对象列表
            
        Returns:
            问题元数据字典，key为question_id，value包含question_text、question_type、extra_instruction
        """
        question_meta_by_id: Dict[str, Dict[str, Any]] = {}
        for task in question_tasks:
            logger.info(f"processing question task: {task.model_dump_json()}")
            try:
                qid = task.id
                question_meta_by_id[qid] = {
                    "question_text": task.description,
                    "question_type": task.type,
                    "extra_instruction": task.prompt or "",
                }
            except Exception as e:
                log_error_type_and_message(
                    logger=logger,
                    context="准备问题元数据查找表时出错",
                    e=e,
                    exc_info=False,
                    max_message_chars=200,
                )
                continue
        return question_meta_by_id
    
    def _get_extra_instruction_for_question(self, question_id: str, question_meta_by_id: Dict[str, Dict[str, Any]]) -> Optional[str]:
        meta = question_meta_by_id.get(question_id, {})
        extra_instruction = meta.get("extra_instruction")
        return extra_instruction

    async def _ensure_retrieval_queries_for_tasks(self, question_tasks: List[QuestionTask]) -> None:
        """在检索开始前，为所有 QuestionTask 统一生成 retrieval_query（若尚未设置）。"""
        await ensure_retrieval_queries_for_tasks(
            question_tasks,
            self.model_id,
            locale=self.answer_lang,
        )

    def _summary_tree_cache_key(self, origin_content: List[Item]) -> Tuple[Tuple[str, int], ...]:
        key_parts: List[Tuple[str, int]] = []
        for item in origin_content or []:
            file_id = normalize_nonempty_string(get_field_value(item, "file_id"))
            element_id = coerce_int(get_field_value(item, "element_id"))
            if file_id is None or element_id is None:
                continue
            key_parts.append((file_id, element_id))
        return tuple(key_parts)

    async def _get_summary_tree_for_request(self, origin_content: List[Item]) -> Any:
        cache_key = self._summary_tree_cache_key(origin_content)
        if not cache_key:
            return None

        async with self._summary_tree_cache_lock:
            if cache_key in self._summary_tree_cache:
                logger.info("命中请求内摘要树缓存 | key_size={}", len(cache_key))
                return self._summary_tree_cache[cache_key]

        root = await load_summary_tree_if_available(origin_content)
        async with self._summary_tree_cache_lock:
            self._summary_tree_cache[cache_key] = root
        logger.info(
            "写入请求内摘要树缓存 | key_size={} available={}",
            len(cache_key),
            root is not None,
        )
        return root

    def _build_retrieval_state(
        self,
        question_tasks: List[QuestionTask],
        typed_items: List[Item],
    ) -> Dict[str, Any]:
        """
        构建检索工作流状态字典。

        Args:
            question_tasks: 问题任务列表
            typed_items: 已转换的Item列表

        Returns:
            检索状态字典
        """
        retrieval_state = {
            "question_tasks": question_tasks,
            "origin_content": typed_items,
            "model_id": self.model_id,  # 用于batch操作的模型
            "model_id_for_judge": self.model_id,  # 用于judge_if_continue节点的模型
            "context_token_limit": self.test_cfg.context_token_limit,  # 传递token限制
            "use_batch_for_llm": True,
            "answer_lang": self.answer_lang,
            "summary_tree_provider": self._get_summary_tree_for_request,
        }
        logger.info(
            f"构建 retrieval_state: question_count={len(question_tasks)}, "
            f"answer_lang={self.answer_lang}, typed_items_len={len(typed_items)}"
        )

        return retrieval_state

    async def _build_origin_content_for_task(
        self,
        base_typed_items: List[Item],
        task: QuestionTask,
    ) -> List[Item]:
        """
        为单个问题构建专属的 origin_content：
        - 基础内容为 base_typed_items
        - 统一收集该问题需要额外加载的 file_id（来源：task.knowledge 与
          task.data_sources 中 ChunkIndexRetrievalTool 显式配置的 file_ids），
          排除已在 base_typed_items 中的 file_id，调用外部接口拉取 docParseContent，
          转为 Item 后追加到该问题的 origin_content 中。

        严格模式：任一 file_id 加载失败（MinIO 对象不存在、下载失败、
        JSON 解析失败、内容格式不符）都将抛出 ValueError，
        由上层捕获并使该题失败，避免"空 <doc> 注入"问题。
        """
        per_task_typed_items: List[Item] = list(base_typed_items)
        question_id = task.id

        # 1) 统一收集需要额外加载的 file_id（保持稳定顺序、去重）
        base_file_ids = {str(item.file_id) for item in base_typed_items if getattr(item, "file_id", None)}
        extra_file_ids: List[str] = []

        # 来源 a: task.knowledge
        extra_file_ids.extend(
            fid.strip() for fid in (task.knowledge or []) if fid and fid.strip()
        )

        # 来源 b: data_sources 中 ChunkIndexRetrievalTool 显式配置的 file_ids
        # 注：file_ids=None 表示"用 origin_content 中的所有 file_id 作为检索范围"，
        # 此时无需额外加载；只有显式配置时才需要处理
        for ds in (task.get_resolved_data_sources() or []):
            if isinstance(ds, ChunkIndexRetrievalTool) and ds.file_ids:
                extra_file_ids.extend(
                    str(fid).strip() for fid in ds.file_ids if str(fid).strip()
                )

        # 去重（保持顺序），并排除已在 base_typed_items 中的 file_id
        missing_file_ids = [
            fid for fid in dict.fromkeys(extra_file_ids)
            if fid and fid not in base_file_ids
        ]

        if not missing_file_ids:
            logger.info(
                f"_build_origin_content_for_task: 问题 {question_id} 无需额外加载文件内容"
                f"（knowledge/data_sources 中的 file_id 已全部在 base_typed_items 中或为空）"
            )
            return per_task_typed_items

        logger.info(
            f"_build_origin_content_for_task: 问题 {question_id} 检测到需要额外加载的 file_id 列表，"
            f"数量={len(missing_file_ids)}, file_ids={missing_file_ids}"
        )

        # 2) 直接从 MinIO 读取解析产物 converted.json（严格模式：任一失败立即抛出）
        merged_contents: List[Dict[str, Any]] = []
        for fid in missing_file_ids:
            logger.info(f"为问题 {question_id} 从 MinIO 加载文件内容: file_id={fid}")
            try:
                content_list = await load_mineru_items(fid)
            except Exception as e:
                raise ValueError(
                    f"问题 {question_id} 加载文件内容失败: file_id={fid}, error={e}"
                ) from e

            logger.info(
                f"_build_origin_content_for_task: 为问题 {question_id} 成功加载文件内容: "
                f"file_id={fid}, docParseContent 条目数={len(content_list)}"
            )
            merged_contents.extend(content_list)

        # 4) 解析为 Item，严格模式：解析失败立即抛出
        try:
            # 现在是将所有文件的 Items 合并故可以这么写，后续考虑按文件分组处理时可能需要修改
            extra_items = DocumentItems.model_validate(merged_contents).root
        except Exception as e:
            raise ValueError(
                f"问题 {question_id} 解析 docParseContent 失败，file_ids={missing_file_ids}，错误: {e}"
            ) from e

        # DocumentItems 会跳过单条解析失败的 item（仅告警），
        # 若全部失败则等于注入了空内容，严格模式下同样视为失败
        if not extra_items:
            raise ValueError(
                f"问题 {question_id} 解析 docParseContent 失败：file_ids={missing_file_ids} "
                f"共 {len(merged_contents)} 条记录无一可转换为 Item"
            )

        per_task_typed_items.extend(extra_items)
        logger.info(
            f"_build_origin_content_for_task: 为问题 {question_id} 追加额外条目，"
            f"转换为 {len(extra_items)} 个 Item，对该题可检索内容总数={len(per_task_typed_items)}"
        )
        return per_task_typed_items

    async def _run_single_task_default_retrieval(
        self,
        task: QuestionTask,
        per_task_typed_items: List[Item],
        retrieval_service: RetrivalService,
        all_timings: Dict[str, float],
    ) -> Tuple[Optional[SelectedContent], Optional[Dict[str, Any]]]:
        """
        对单个问题执行默认检索（run_streaming），用于 data_sources 为空但需走 custom 桶的场景
        （如仅配置了 knowledge/tools）。返回该题的检索结果与 thinking_steps 条目。
        """
        question_id = task.id
        retrieval_state = self._build_retrieval_state(
            [task],
            per_task_typed_items,
        )
        selected_content_for_q: Optional[SelectedContent] = None
        thinking_steps_entry: Optional[Dict[str, Any]] = None
        try:
            async for producer_result in retrieval_service.run_streaming(retrieval_state):
                if producer_result.get("type") == "final_state":
                    # 从 final_state 中收集节点耗时信息
                    final_state = producer_result.get("retrieval_data", {})
                    if isinstance(final_state, dict) and "node_timings" in final_state:
                        all_timings.update(final_state["node_timings"])
                    continue
                if producer_result.get("type") == "error":
                    raise ValueError(producer_result.get("error") or "default retrieval error")
                if producer_result.get("type") != "finished_questions":
                    continue
                node_state = producer_result.get("retrieval_data", {}) or {}
                for sc in node_state.get("selected_content", []) or []:
                    if getattr(sc, "question_id", None) == question_id:
                        selected_content_for_q = sc
                        break
                tmap = node_state.get("thinking_steps_map", {}) or {}
                if isinstance(tmap, dict) and question_id in tmap:
                    thinking_steps_entry = tmap.get(question_id)
        except Exception as e:
            raise
        if not selected_content_for_q:
            return None, thinking_steps_entry
        retrieval_result_for_question = self._process_retrieval_result_for_question(
            question_id, {"selected_content": [selected_content_for_q]}
        )
        return retrieval_result_for_question, thinking_steps_entry
    def _use_default_workflow(self, task: QuestionTask) -> bool:
        """是否对该问题使用默认完整流程（planner->coder->reporter）。无 tools 或存在已启用的 CodingAgentTool 时为 True，否则仅走 reporter 流程（返回 False）。"""
        if not task.tools:
            return True
        for tool in task.tools:
            if isinstance(tool, CodingAgentTool):
                return getattr(tool, "enable", True)
        return False
    
    def _process_retrieval_result_for_question(
        self,
        question_id: str,
        retrieval_data: Dict[str, Any],
    ) -> Optional[SelectedContent]:
        """
        处理单个问题的检索结果：查找、排序、重命名。
        
        Args:
            question_id: 问题ID
            retrieval_data: 检索数据字典
            
        Returns:
            处理后的检索结果，如果未找到则返回None
        """
        # 获取该问题的检索结果
        retrieval_result = self._get_retrieval_result_for_question(question_id, retrieval_data)
        if not retrieval_result:
            logger.error(f"无法找到问题 {question_id} 的检索结果")
            return None
        
        # 统一处理排序和重命名
        retrieval_result = sort_and_rename_retrieval_result(retrieval_result)
        
        return retrieval_result
    

    def _localize_page_label_for_output(self, page: str) -> str:
        """Convert hard-coded Chinese page labels to English for output display."""
        match = _PAGE_LABEL_RE.fullmatch((page or "").strip())
        if not match:
            return page
        return f"Page {match.group(1)}"

    async def _translate_thinking_texts_to_english(self, texts: List[str]) -> List[str]:
        """Translate thinking-step display text to English while preserving markers and code-like tokens."""
        if not texts:
            return texts
        logger.info(
            f"开始英文化 thinking steps 文本: answer_lang={self.answer_lang}, text_count={len(texts)}"
        )

        prompt = (
            "You are a careful translator for an AI product UI.\n"
            "Translate each input text into natural English.\n"
            "Requirements:\n"
            "- Keep the output list length and order exactly the same as the input list.\n"
            "- Preserve any ids, file names, doc ids, query strings, code, formulas, variables, and bracketed markers exactly as they appear.\n"
            "- Preserve content inside square brackets when it acts as a citation or identifier.\n"
            "- If a text is already English or should not be translated, return it unchanged.\n"
            "- Return JSON only.\n\n"
            "Input texts JSON:\n"
            f"{json.dumps(texts, ensure_ascii=False)}"
        )

        try:
            response = await ainvoke_llm_with_structured_output(
                prompt=prompt,
                model_id=self.model_id,
                response_type=ThinkingStepTranslationBatch,
            )
        except Exception as e:
            logger.warning(
                f"思考过程英文化失败，将保留原文: answer_lang={self.answer_lang}, "
                f"text_count={len(texts)}, error_type={type(e).__name__}, error={e}"
            )
            return texts

        translations = response.translations or []
        if len(translations) != len(texts):
            logger.warning(
                f"思考过程英文化返回数量不匹配，保留原文。 answer_lang={self.answer_lang} "
                f"expected={len(texts)} actual={len(translations)}"
            )
            return texts

        normalized_translations: List[str] = []
        for original, translated in zip(texts, translations):
            normalized_translations.append(translated if isinstance(translated, str) and translated.strip() else original)
        return normalized_translations

    async def _localize_thinking_steps_for_output(self, answer: FinalAnswer) -> FinalAnswer:
        """Temporarily localize thinking-step display text for English output without changing core generation flow."""
        if self.answer_lang != "en_US" or answer.thinking_steps is None:
            return answer

        localized_answer = answer.model_copy(deep=True)
        thinking_steps = localized_answer.thinking_steps
        text_refs: List[Tuple[Any, str]] = []

        def _collect_model_text(container: Any, field_name: str) -> None:
            value = getattr(container, field_name, None)
            if isinstance(value, str) and contains_cjk(value):
                text_refs.append((container, field_name))

        def _collect_nested_texts(node: Any) -> None:
            if isinstance(node, list):
                for item in node:
                    _collect_nested_texts(item)
                return
            if not isinstance(node, dict):
                return

            for key, value in node.items():
                if key == "page" and isinstance(value, str):
                    node[key] = self._localize_page_label_for_output(value)
                    continue
                if key in {"text", "description"} and isinstance(value, str) and contains_cjk(value):
                    text_refs.append((node, key))
                    continue
                if key in {"code", "formula", "query", "file_id", "doc_id", "id", "element_ids"}:
                    continue
                if isinstance(value, (dict, list)):
                    _collect_nested_texts(value)

        for stage_name in ("chunk_retrieval", "summary_retrieval", "document_retrieval"):
            stage = getattr(thinking_steps, stage_name, None)
            if stage is None:
                continue
            for item in getattr(stage, "items", []) or []:
                item_type = getattr(item, "type", None)
                if item_type == "analysis":
                    _collect_model_text(item, "text")
                elif item_type == "query":
                    _collect_model_text(item, "text")
                elif item_type == "resources":
                    for resource in getattr(item, "items", []) or []:
                        page = getattr(resource, "page", None)
                        if isinstance(page, str):
                            resource.page = self._localize_page_label_for_output(page)

        for step in thinking_steps.answer_agent_steps or []:
            _collect_model_text(step, "text")
            if step.items:
                _collect_nested_texts(step.items)

        if not text_refs:
            return localized_answer

        originals: List[str] = []
        for container, key in text_refs:
            if isinstance(container, dict):
                originals.append(container.get(key, ""))
            else:
                originals.append(getattr(container, key, ""))

        translated_texts = await self._translate_thinking_texts_to_english(originals)
        for (container, key), translated in zip(text_refs, translated_texts):
            if isinstance(container, dict):
                container[key] = translated
            else:
                setattr(container, key, translated)

        return localized_answer

    async def _build_output_payload(self, question_id: str, answer: FinalAnswer, question_meta_by_id: Dict[str, Dict[str, Any]], retrieval_result: Optional[SelectedContent] = None) -> ContentGenerateResponseEvent:
        """
        构建输出载荷。
        """
        meta = question_meta_by_id.get(question_id, {})
        localized_answer = await self._localize_thinking_steps_for_output(answer)
        
        return ContentGenerateResponseEvent(
            question_id=question_id,
            question_text=meta.get("question_text"),
            question_type=meta.get("question_type"),
            answer=localized_answer,
        )

    def _build_unanswerable_answer(self, *, message_key: str) -> FinalAnswer:
        return FinalAnswer(
            is_answerable=False,
            answer_text=get_localized_message(message_key, self.answer_lang),
            citations=[],
        )
    
    async def _generate_answer_for_question(
        self,
        question_id: str,
        retrieval_result: SelectedContent,
        typed_items: List[Item],
        question_meta_by_id: Dict[str, Dict[str, Any]],
        task: QuestionTask,
        history_messages: Optional[List[tuple[str, str]]] = None,
        dependency_context: str = "",
    ) -> FinalAnswer:
        """
        为单个问题生成答案（统一处理有/无 CodingAgentTool 的情况）。
        
        Args:
            question_id: 问题ID
            retrieval_result: 检索结果
            typed_items: 原始内容列表
            question_meta_by_id: 问题元数据字典
            task: 问题任务（用于检查是否有 CodingAgentTool）
            history_messages: 可选的对话历史
            
        Returns:
            最终答案对象
        """
        # 获取额外指令
        extra_instruction = self._get_extra_instruction_for_question(question_id, question_meta_by_id)
        if task.type == "text-json-citation":
            reporter_template = "text_json_citation"
        elif task.type == "text-json":
            reporter_template = "text_json"
        else:
            reporter_template = "default"

        reporter_model_id = None
        if task.type == "text" and self.test_cfg.reporter_text_model_id:
            reporter_model_id = self.test_cfg.reporter_text_model_id
            logger.info(f"问题 {question_id} type=text，reporter 使用微调模型: {reporter_model_id}")
        elif task.type == "text-json-citation" and self.test_cfg.reporter_text_json_citation_model_id:
            reporter_model_id = self.test_cfg.reporter_text_json_citation_model_id
            logger.info(f"问题 {question_id} type=text-json-citation，reporter 使用微调模型: {reporter_model_id}")

        use_full_workflow = self._use_default_workflow(task)
        logger.info(f"开始生成问题 {question_id} 的答案，use_default_workflow={use_full_workflow}")

        if use_full_workflow:
            # 使用完整 workflow（planner->coder->reporter）
            logger.info(f"问题 {question_id} 使用完整流程（planner->coder->reporter）")
            final_answer = await self.answer_single_question(
                retrieval_result,
                typed_items,
                extra_instruction=extra_instruction,
                history_messages=history_messages,
                reporter_template=reporter_template,
                reporter_model_id=reporter_model_id,
                dependency_context=dependency_context,
            )
        else:
            # 仅 reporter_core
            logger.info(f"问题 {question_id} 使用仅 reporter 流程（直接调用 reporter_core）")

            # 使用统一的方法准备 content 和 docid_to_source
            content, docid_to_source = self._prepare_content_from_retrieval_results(
                retrieval_result,
                typed_items,
            )

            # 直接调用 reporter_core（coder_output 为空列表）
            final_answer = await reporter_core(
                question=retrieval_result.question,
                content=content,
                coder_output=[],  # 不使用 CodingAgent，coder_output 为空
                docid_to_source=docid_to_source,
                model_id=reporter_model_id or self.model_id,
                reporter_instruction=extra_instruction or "",
                reporter_template=reporter_template,
                history_messages=history_messages,
                answer_lang=self.answer_lang,
                dependency_context=dependency_context,
            )

            # 仅 reporter 模式下，手动追加 reporter 步骤到 thinking_steps
            try:
                from .models import AnswerAgentStepItem, ThinkingSteps

                if final_answer.thinking_steps is None:
                    final_answer.thinking_steps = ThinkingSteps()
                if final_answer.thinking_steps.answer_agent_steps is None:
                    final_answer.thinking_steps.answer_agent_steps = []

                final_answer.thinking_steps.answer_agent_steps.append(
                    AnswerAgentStepItem(
                        type="reporter",
                        text=final_answer.answer_text,
                    )
                )
            except Exception as e:
                log_error_type_and_message(
                    logger=logger,
                    context="手动添加 reporter 步骤到 answer_agent_steps 失败",
                    e=e,
                    exc_info=False,
                    max_message_chars=200,
                )
                if final_answer.thinking_steps is None:
                    from .models import ThinkingSteps
                    final_answer.thinking_steps = ThinkingSteps()
        
        return final_answer

    @trace_span(name="ContentGenerateAgent._run_workflow")
    async def _run_workflow(
        self,
        question,
        content,
        reporter_instruction,
        reporter_template: str = "default",
        reporter_model_id: Optional[str] = None,
        docid_to_source: Optional[Dict[str, Any]] = None,
        history_messages: Optional[List[tuple[str, str]]] = None,
        dependency_context: str = "",
    ):
        """
        Arg:
            question: 问题
            content: 相关内容文本列表
            reporter_instruction: 报告指令
            reporter_model_id: reporter 使用的模型ID，若为 None 则使用默认 model_id
            docid_to_source: 文本片段id转换为element_ids/file_id的映射
        """

        # 创建状态
        state = {
            "question": question,
            "content": content,
            "model_id": self.model_id,
            "content_cnt": len(content),
            "reporter_instruction": reporter_instruction,
            "reporter_template": reporter_template,
            "reporter_model_id": reporter_model_id,
            "docid_to_source": docid_to_source or {},
            "history_messages": history_messages,
            "answer_lang": self.answer_lang,
            "dependency_context": dependency_context,
        }

        logger.info(
            f"[i18n] ContentGenerateAgent._run_workflow已写入语言状态: "
            f"question={question[:100]}, answer_lang={self.answer_lang}"
        )

        logger.info("开始执行工作流...")
        return await ContentGenService.run_question(state)

    def _prepare_content_from_retrieval_results(
        self,
        retrieval_results: SelectedContent,
        origin_content: List[Item],
        max_context_tokens: Optional[int] = None,
    ) -> tuple[List[dict], Dict[str, Dict[str, Any]]]:
        """从检索结果中准备 content 和 docid_to_source。
        
        此方法统一处理从 SelectedContent 中提取内容、进行 token 预算管理、
        构建 content 列表和 docid_to_source 映射的逻辑。
        
        doc_id 归一化：将原始 doc_id（如 q_xxx_RAGChunk_0）替换为 doc1, doc2, doc3 格式，
        减少 token 消耗并降低模型引用错误率。原始 doc_id 保留在 docid_to_source 的
        _original_doc_id 字段中用于溯源。
        
        Args:
            retrieval_results: 检索结果对象
            origin_content: 原始内容列表（Item 对象列表），用于回查文本
            max_context_tokens: 最大上下文 token 数，如果为 None 则使用配置的默认值
            
        Returns:
            tuple: (content列表, docid_to_source字典)
                - content: 包含 doc_id, text, line_str 的字典列表
                - docid_to_source: doc_id 到 source 信息的映射
        """
        results_raw = retrieval_results.selected_pieces or []
        content: List[dict] = []
        docid_to_source: Dict[str, Dict[str, Any]] = {}
        
        # 构建 item_map 用于回查文本
        item_map = build_item_key_to_item(origin_content)
        
        # 使用传入的 token 限制或默认配置
        if max_context_tokens is None:
            max_context_tokens = self.test_cfg.context_token_limit
        
        # 添加 token 预算管理，避免上下文过长
        total_tokens = 0
        
        for i, r in enumerate(results_raw, 1):
            original_doc_id = r.doc_id
            normalized_doc_id = f"doc{i}"

            # 通过回查机制获取文本
            text = resolve_piece_text(r, item_map)
            # Safety net：跳过空文本片段，避免"空 <doc> 注入"污染 LLM 上下文。
            # 上游已通过 _build_origin_content_for_task 主动加载 file_id 内容治本，
            # 此处只兜底处理 summary 检索路径或 RAG 与 docParseContent 版本不同步等边缘场景。
            if not text:
                logger.warning(
                    f"跳过空文本片段: doc_id={original_doc_id}, "
                    f"file_id={getattr(r, 'file_id', '')}, element_ids={getattr(r, 'element_ids', [])}, "
                    f"source_stage={getattr(r, 'source_stage', '')}"
                )
                continue
            piece_tokens = count_tokens(text)  # 粗略估算 token 数

            # 如果添加这个片段会超过预算，则停止添加
            if total_tokens + piece_tokens > max_context_tokens:
                logger.warning(f"上下文 token 预算不足，已截断到 {i-1} 个片段")
                break

            line_str = f"<doc id={normalized_doc_id}>\n{text}\n</doc>"
            content.append({
                "doc_id": normalized_doc_id,
                "text": text,
                "line_str": line_str,
            })
            total_tokens += piece_tokens

            # 收集来源字段，用于 citation 还原。raw_text_only 的片段无真实文档位置，只写占位并标记
            if getattr(r, "raw_text_only", False):
                docid_to_source[normalized_doc_id] = {"element_ids": [], "file_id": "", "raw_text_only": True, "_original_doc_id": original_doc_id}
            else:
                source_info: Dict[str, Any] = {}
                if getattr(r, "element_ids", None):
                    source_info["element_ids"] = list(r.element_ids)
                if getattr(r, "file_id", None):
                    source_info["file_id"] = r.file_id
                source_info["_original_doc_id"] = original_doc_id
                if source_info:
                    docid_to_source[normalized_doc_id] = source_info
        
        logger.info(f"为问题准备了 {len(content)} 个相关片段，预估 token 数: {total_tokens}")
        return content, docid_to_source

    @trace_span(name="ContentGenerateAgent.answer_single_question")
    async def answer_single_question(
        self,
        retrieval_results: SelectedContent,
        origin_content: List[Item],
        extra_instruction: Optional[str] = None,
        history_messages: Optional[List[tuple[str, str]]] = None,
        reporter_template: str = "default",
        reporter_model_id: Optional[str] = None,
        dependency_context: str = "",
    ):
        question: str = retrieval_results.question
        
        # 使用统一的方法准备 content 和 docid_to_source
        lines, docid_to_source = self._prepare_content_from_retrieval_results(retrieval_results, origin_content)
        
        # 用question和context_text调用langgraph写的流程，判断是否需要计算，并生成报告
        result = await self._run_workflow(
            question,
            lines,
            extra_instruction,
            reporter_template=reporter_template,
            reporter_model_id=reporter_model_id,
            docid_to_source=docid_to_source,
            history_messages=history_messages,
            dependency_context=dependency_context,
        )
        node_timings = result.get("node_timings", {})
        logger.info(f"node_timings: {node_timings}")
        final_answer = result["answer"]
        
        # answer_agent_steps已经在节点中直接添加，无需再次处理
        return final_answer

    async def text_answer_generator(
        self,
        typed_items: List[Item],
        question_tasks: List[QuestionTask] = None,
        dependency_context_by_qid: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        """
        生成器方法，逐个yield每个问题的答案
        
        使用流式处理模式：生产者-消费者架构，finished_question立即生成答案

        Args:
            typed_items: 已转换的Item列表
            question_tasks: QuestionTask 对象列表

        Yields:
            ContentGenerateResponseEvent: 响应事件对象
        """
        logger.info(f"answer_generator开始，共 {len(typed_items)} 个Item对象")

        if not question_tasks:
            logger.warning("question_tasks 为空，无法生成答案")
            return

        # 在检索开始前，为所有问题统一生成检索用 query
        await self._ensure_retrieval_queries_for_tasks(question_tasks)

        # 准备问题元数据查找表（按 question_id 匹配）
        question_meta_by_id = self._build_question_meta_by_id(question_tasks)

        logger.info("启动流式处理模式，启动生产者-消费者架构")
        async for result in self._text_answer_generator_streaming(
            typed_items,
            question_tasks,
            question_meta_by_id,
            dependency_context_by_qid=dependency_context_by_qid,
        ):
            yield result
    
    async def _text_answer_generator_streaming(
        self,
        typed_items: List[Item],
        question_tasks: List[QuestionTask],
        question_meta_by_id: Dict[str, Dict[str, Any]],
        dependency_context_by_qid: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        """
        流式处理模式：生产者-消费者架构 + 并发处理优化
        
        生产者：检索工作流持续产出finished_question_ids
        消费者：并发处理finished_question并生成答案（优化：使用asyncio.gather并发处理多个问题）
        """
        logger.info("启动流式处理模式（并发优化版）")
        
        # 创建检索服务实例（生产者）
        retrieval_service = RetrivalService()
        
        # 构建检索工作流状态，直接使用 question_tasks
        retrieval_state = self._build_retrieval_state(
            question_tasks, typed_items
        )
        
        logger.info(f"启动检索流程，问题数量: {len(question_tasks)}")

        # 用于跟踪已处理的问题
        # processed_questions 是局部变量，只在当前 generator 内部使用，不会有竞态问题
        processed_questions = set()
        task_by_id: Dict[str, QuestionTask] = {t.id: t for t in question_tasks}
        producer_result_count = 0

        # 用于收集所有耗时统计
        all_timings: Dict[str, float] = {}
        # 用于收集每个问题的答案生成耗时统计
        answer_timings_by_qid: Dict[str, Dict[str, float]] = {}
        
        try:
            # 启动生产者（检索工作流）
            async for producer_result in retrieval_service.run_streaming(retrieval_state):
                producer_result_count += 1
                logger.info(f"收到生产者结果 No.{producer_result_count}, type: {producer_result.get('type', 'unknown')}")
                
                if producer_result.get("type") == "final_state":
                    # 从 final_state 中收集节点耗时信息
                    final_state = producer_result.get("retrieval_data", {})
                    if isinstance(final_state, dict) and "node_timings" in final_state:
                        all_timings.update(final_state["node_timings"])
                    continue
                
                if producer_result["type"] == "finished_questions":
                    logger.info(f'收到检索流程完成产出，本次产出问题id: {producer_result["finished_question_ids"]}')
                    finished_ids = producer_result["finished_question_ids"]
                    retrieval_data = producer_result["retrieval_data"]
                    
                    # 收集检索节点耗时信息
                    if isinstance(retrieval_data, dict) and "node_timings" in retrieval_data:
                        all_timings.update(retrieval_data["node_timings"])
                    logger.info(f"开始处理finished_question_ids: {finished_ids}")
                    
                    # 在进入答案生成前，按配置决定是否筛选超过 context_token_limit 的上下文
                    selected_content_list = retrieval_data.get("selected_content", [])
                    if selected_content_list and self.test_cfg.enable_filter_long_context:
                        logger.info(f"进入答案生成前，开始筛选上下文")
                        filter_start_time = time.perf_counter()
                        selected_content_list = await filter_long_context_if_needed(
                            selected_content=selected_content_list,
                            question_ids=finished_ids,
                            origin_content=typed_items,
                            model_id=self.model_id,
                            token_threshold=self.test_cfg.context_token_limit,
                        )
                        filter_elapsed = time.perf_counter() - filter_start_time
                        # 记录筛选上下文耗时
                        node_timings = retrieval_data.get("node_timings", {})
                        node_timings["filter_context_before_answer"] = filter_elapsed
                        retrieval_data["node_timings"] = node_timings
                        logger.info(f"上下文筛选完成，耗时: {filter_elapsed:.3f}s")
                        # 更新 retrieval_data 中的 selected_content
                        retrieval_data["selected_content"] = selected_content_list
                    elif selected_content_list and not self.test_cfg.enable_filter_long_context:
                        logger.info(f"enable_filter_long_context=False，跳过上下文筛选")
                    
                    # ========== 优化点：并发处理多个finished_question ==========
                    # 过滤掉已处理的问题
                    new_finished_ids = [qid for qid in finished_ids if qid not in processed_questions]
                    
                    if not new_finished_ids:
                        logger.warning("所有finished_question都已处理过，跳过")
                        continue
                    
                    logger.info(f"开始并发处理 {len(new_finished_ids)} 个问题")
                    
                    async def process_single_question(
                        question_id: str,
                        retrieval_data: Dict[str, Any],
                        question_meta_by_id: Dict[str, Dict[str, Any]],
                        typed_items: List[Item],
                    ) -> Optional[ContentGenerateResponseEvent]:
                        try:
                            logger.info(f"开始处理问题: {question_id}")
                            
                            # 处理检索结果（查找、排序、重命名）
                            retrieval_result = self._process_retrieval_result_for_question(
                                question_id, retrieval_data
                            )
                            if not retrieval_result:
                                # 必回调：找不到检索结果时也返回不可回答的 FinalAnswer
                                final_answer = self._build_unanswerable_answer(
                                    message_key="retrieval_result_missing"
                                )
                                output_payload = await self._build_output_payload(
                                    question_id, final_answer, question_meta_by_id, None
                                )
                                logger.error(
                                    f"问题 {question_id} 未找到检索结果，将返回不可回答答案并继续"
                                )
                                return output_payload
                            
                            # 生成答案（统一入口，支持 dependency_context 注入）
                            task = task_by_id.get(question_id)
                            if task is None:
                                raise ValueError(f"无法找到问题 {question_id} 对应的 QuestionTask")
                            final_answer = await self._generate_answer_for_question(
                                question_id,
                                retrieval_result,
                                typed_items,  # 传入 origin_content 用于回查文本
                                question_meta_by_id,
                                task=task,
                                dependency_context=(dependency_context_by_qid or {}).get(question_id, "") if dependency_context_by_qid else "",
                            )
                            
                            thinking_steps_entry = retrieval_data.get("thinking_steps_map", {}).get(question_id)
                            logger.info(f"问题 {question_id} 的thinking_steps_entry: {thinking_steps_entry}")
                            final_answer = attach_thinking_steps_to_answer(final_answer, thinking_steps_entry)
                            
                            # 构建输出载荷
                            output_payload = await self._build_output_payload(
                                question_id, final_answer, question_meta_by_id, retrieval_result
                            )
                            
                            logger.info(f"问题 {question_id} 处理完成")
                            return output_payload
                            
                        except Exception as e:
                            log_error_type_and_message(
                                logger=logger,
                                context=f"处理问题 {question_id} 时出错",
                                e=e,
                                exc_info=True,
                                max_message_chars=200,
                            )
                            # 必回调：单题处理出错时也返回不可回答的 FinalAnswer，避免调用方无感知
                            final_answer = FinalAnswer(
                                is_answerable=False,
                                answer_text=get_localized_message(
                                    "generic_question_processing_error",
                                    self.answer_lang,
                                    error=str(e),
                                ),
                                citations=[],
                            )
                            output_payload = await self._build_output_payload(
                                question_id, final_answer, question_meta_by_id, None
                            )
                            return output_payload
                    
                    # 创建并发任务（带question_id标记，用于错误日志）
                    # tasks 是一个字典：{task对象: question_id}，用于在后续处理时通过task找到对应的question_id
                    tasks = {
                        asyncio.create_task(
                            process_single_question(
                                question_id=qid,
                                retrieval_data=retrieval_data,
                                question_meta_by_id=question_meta_by_id,
                                typed_items=typed_items,
                            )
                        ): qid
                        for qid in new_finished_ids
                    }
                    
                    # 按完成顺序处理结果（谁先完成谁先yield）
                    # 使用 asyncio.wait() 的 FIRST_COMPLETED 模式，按任务完成顺序处理
                    # 这是一个更通用的方法，不依赖 as_completed() 函数
                    pending_tasks = set(tasks.keys())
                    
                    while pending_tasks:
                        # 等待至少一个任务完成
                        done, pending_tasks = await asyncio.wait(
                            pending_tasks,
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # 处理所有已完成的任务
                        for task in done:
                            # 通过task对象找到对应的question_id（用于日志和错误处理）
                            question_id = tasks[task]
                            try:
                                # 获取已完成任务的结果
                                result = await task
                                
                                # 检查结果是否有效（process_single_question在失败时返回None）
                                if result is not None:
                                    # 标记为已处理，避免重复处理
                                    processed_questions.add(question_id)
                                    
                                    logger.info(f"问题 {question_id} 处理完成，立即yield给前端")
                                    # 立即yield结果给前端，不需要等待其他问题完成
                                    yield result
                            except Exception as e:
                                # 捕获任务执行过程中的异常（虽然process_single_question内部已处理异常，
                                # 但这里提供双重保护，确保异常不会中断整个流程）
                                log_error_type_and_message(
                                    logger=logger,
                                    context=f"处理问题 {question_id} 时出现异常",
                                    e=e,
                                    exc_info=True,
                                    max_message_chars=200,
                                )
                
                elif producer_result["type"] == "error":
                    logger.error(f"检索流程报告错误, error message: {producer_result['error']}")
                    raise ValueError(producer_result.get("error") or "检索流程报告错误")
                else:
                    logger.error(f'检索流程报告错误, type = {producer_result["type"]}')
                    
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="流式处理出现异常",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            raise ValueError(f"流式处理出现异常: {e}")

        # 流式结束兜底：补齐“从未被 finished_questions 产出”的问题，避免静默丢题
        try:
            all_question_ids = set(task_by_id.keys())
            missing_question_ids = sorted(all_question_ids - processed_questions)
            if missing_question_ids:
                logger.error(
                    "检索流式处理结束但存在未处理问题，将补发不可回答事件。"
                    f"missing_count={len(missing_question_ids)}, missing_ids={missing_question_ids}"
                )
                for qid in missing_question_ids:
                    final_answer = self._build_unanswerable_answer(
                        message_key="retrieval_stream_incomplete"
                    )
                    yield await self._build_output_payload(
                        qid, final_answer, question_meta_by_id, None
                    )
                    processed_questions.add(qid)
        except Exception as e:
            # 兜底逻辑自身不应影响主流程：只记录日志
            log_error_type_and_message(
                logger=logger,
                context="流式结束补齐缺失问题失败",
                e=e,
                exc_info=True,
                max_message_chars=200,
            )
        logger.info(f"流式处理完成，共处理 {len(processed_questions)} 个问题")

        # 打印完整的耗时统计报告
        if all_timings:
            # total_time = sum(all_timings.values())
            timing_lines = [
                f"{'=' * 60}",
                "耗时统计报告",
                f"{'=' * 60}",
            ]
            for task_name, elapsed in sorted(all_timings.items()):
                # percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                timing_lines.append(f"{task_name:40s} {elapsed:8.3f}s")
            timing_lines.append(f"{'-' * 60}")
            # timing_lines.append(f"{'TOTAL':40s} {total_time:8.3f}s (100.0%)")
            timing_lines.append(f"{'=' * 60}")
            logger.info("\n" + "\n".join(timing_lines))
        else:
            logger.info("没有收集到耗时统计数据")
       
    
    async def custom_answer_generator(
        self,
        typed_items: List[Item],
        question_tasks: List[QuestionTask] = None,
        dependency_context_by_qid: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        """
        自定义答案生成器，使用custom_retrieval_workflow进行检索。
        
        此方法用于处理带有tools字段的问题任务，使用可配置的检索流程。
        
        Args:
            typed_items: 已转换的Item列表
            question_tasks: QuestionTask 对象列表（应包含tools字段）
            
        Yields:
            ContentGenerateResponseEvent: 响应事件对象
        """
        logger.info(f"custom_answer_generator开始，共 {len(typed_items)} 个Item对象")
        
        if not question_tasks:
            logger.warning("question_tasks 为空，无法生成答案")
            return

        # 在检索开始前，为所有问题统一生成检索用 query
        await self._ensure_retrieval_queries_for_tasks(question_tasks)
        
        # 准备问题元数据查找表（按 question_id 匹配）
        question_meta_by_id = self._build_question_meta_by_id(question_tasks)
        
        # 创建检索服务实例（可复用）
        retrieval_service = RetrivalService()

        # 用于收集所有耗时统计
        all_timings: Dict[str, float] = {}

        # 逐题处理：每个问题可以拥有自己的 knowledge 附加文件（仅对该题生效）
        for task in question_tasks:
            question_id = task.id
            logger.info(f"custom_answer_generator: 开始处理问题 {question_id}")

            try:
                # 1) 为当前问题构建专属的 origin_content：
                #    基础 typed_items + 该题的 knowledge / data_sources.file_ids 加载的内容
                #    严格模式：任一 file_id 加载失败将抛出 ValueError，由下方 except 捕获并重新抛出
                start_time = time.perf_counter()
                per_task_typed_items: List[Item] = await self._build_origin_content_for_task(
                    typed_items, task
                )
                elapsed = time.perf_counter() - start_time
                all_timings[f"{question_id}_build_origin_content"] = elapsed

                # 2) 按 data_sources 是否为空选择检索方式
                resolved_data_sources = task.get_resolved_data_sources() or []
                # 当所有 data_sources 都是 other_datasource 占位类型时（无真实 fileId），
                # 视为无有效 data_sources，回退到默认检索，避免嵌套问题拿不到文档上下文
                has_effective_sources = any(
                    not isinstance(ds, OtherDataSource) for ds in resolved_data_sources
                ) if resolved_data_sources else False
                has_data_sources = len(resolved_data_sources) > 0 and has_effective_sources

                if has_data_sources:
                    # data_sources 非空：走 run_custom（可配置检索）
                    retrieval_state = self._build_retrieval_state(
                        [task],
                        per_task_typed_items,
                    )
                    logger.info(
                        f"custom_answer_generator: 问题 {question_id} data_sources 非空，启动可配置检索(run_custom)，"
                        f"origin_content_len={len(per_task_typed_items)}"
                    )
                    start_time = time.perf_counter()
                    retrieval_result = await retrieval_service.run_custom(retrieval_state)
                    elapsed = time.perf_counter() - start_time
                    all_timings[f"{question_id}_custom_retrieval"] = elapsed

                    # 收集检索节点耗时
                    if "node_timings" in retrieval_result:
                        for k, v in retrieval_result["node_timings"].items():
                            all_timings[f"{question_id}_{k}"] = v

                    selected_content_list = retrieval_result.get("selected_content", []) or []
                    retrieval_result_for_question = self._process_retrieval_result_for_question(
                        question_id, {"selected_content": selected_content_list}
                    )
                    thinking_steps_entry = retrieval_result.get("thinking_steps_map", {}).get(question_id, None)
                else:
                    # data_sources 为空：走默认检索 run_streaming
                    logger.info(
                        f"custom_answer_generator: 问题 {question_id} data_sources 为空，使用默认检索(run_streaming)，"
                        f"origin_content_len={len(per_task_typed_items)}"
                    )
                    start_time = time.perf_counter()
                    retrieval_result_for_question, thinking_steps_entry = await self._run_single_task_default_retrieval(
                        task, per_task_typed_items, retrieval_service, all_timings
                    )
                    elapsed = time.perf_counter() - start_time
                    all_timings[f"{question_id}_default_retrieval"] = elapsed

                if not retrieval_result_for_question:
                    logger.error(f"custom_answer_generator: 无法找到问题 {question_id} 的任何检索结果，将返回不可回答答案并继续")
                    final_answer = self._build_unanswerable_answer(
                        message_key="retrieval_result_missing"
                    )
                    output_payload = await self._build_output_payload(
                        question_id, final_answer, question_meta_by_id, None
                    )
                    yield output_payload
                    continue

                # 3) 生成答案（统一处理有/无 CodingAgentTool 的情况），注意 origin_content 使用 per_task_typed_items
                start_time = time.perf_counter()
                final_answer = await self._generate_answer_for_question(
                    question_id,
                    retrieval_result_for_question,
                    per_task_typed_items,
                    question_meta_by_id,
                    task=task,
                    dependency_context=(dependency_context_by_qid or {}).get(question_id, "") if dependency_context_by_qid else "",
                )
                elapsed = time.perf_counter() - start_time
                all_timings[f"{question_id}_answer_generation"] = elapsed
                logger.info(f"问题 {question_id} 的thinking_steps_entry: {thinking_steps_entry}")
                final_answer = attach_thinking_steps_to_answer(final_answer, thinking_steps_entry)

                # 构建输出载荷
                output_payload = await self._build_output_payload(
                    question_id, final_answer, question_meta_by_id, retrieval_result_for_question
                )

                logger.info(f"问题 {question_id} 处理完成，立即yield给前端")
                yield output_payload

            except Exception as e:
                log_error_type_and_message(
                    logger=logger,
                    context=f"custom_answer_generator: 处理问题 {question_id} 时出现异常",
                    e=e,
                    exc_info=True,
                    max_message_chars=200,
                )
                raise

        logger.info(f"custom_answer_generator完成，共处理 {len(question_tasks)} 个问题")

        # 打印完整的耗时统计报告
        if all_timings:
            # total_time = sum(all_timings.values())
            timing_lines = [
                f"{'=' * 60}",
                "耗时统计报告",
                f"{'=' * 60}",
            ]
            for task_name, elapsed in sorted(all_timings.items()):
                # percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                timing_lines.append(f"{task_name:40s} {elapsed:8.3f}s")
            timing_lines.append(f"{'-' * 60}")
            # timing_lines.append(f"{'TOTAL':40s} {total_time:8.3f}s (100.0%)")
            timing_lines.append(f"{'=' * 60}")
            logger.info("\n" + "\n".join(timing_lines))
        else:
            logger.info("没有收集到耗时统计数据")
