from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

from src.agents.content_generate_agent.utils import get_localized_message
from src.applications.content_generate.services.image_answer_service import ImageAnswerService
from src.applications.content_generate.services.log_redaction import redact_data_uris_for_log

from .models import ContentGenerateResponseEvent, FinalAnswer, QuestionTask

if TYPE_CHECKING:
    from .content_generate_agent import ContentGenerateAgent


from .utils import log_error_type_and_message
from loguru import logger


def _get_normalized_base_on_context(task: QuestionTask) -> List[str]:
    """
    统一读取依赖字段，保证后续逻辑始终拿到 list。

    约定：
    - None / [] 都视为“无依赖”，返回 []
    - 非空时返回一个浅拷贝，避免原地修改 task 对象
    """
    deps = getattr(task, "base_on_context", None)
    return list(deps) if deps else []


def _topo_sort_dependent_tasks(
    all_tasks: List[QuestionTask],
) -> Tuple[List[QuestionTask], Dict[str, str]]:
    """
    对“有依赖”的任务做确定性拓扑排序（Kahn + 输入顺序 tie-break）。

    返回：
    - ordered_dependent_tasks: 拓扑顺序的 dependent tasks（仅包含 base_on_context 非空者）
    - pre_error_by_qid: 需要直接报错的任务，对应错误信息前缀为：
      - missing_dependency
      - cycle_dependency_or_downstream
    """
    # 1) 预处理：构建索引，便于按 qid 查任务/校验依赖是否存在
    id_to_task: Dict[str, QuestionTask] = {t.id: t for t in all_tasks}
    all_ids: Set[str] = set(id_to_task.keys())
    # 记录输入顺序，用于“同层节点”的稳定排序（保证结果可复现）
    input_index: Dict[str, int] = {t.id: i for i, t in enumerate(all_tasks)}

    # 仅对“有依赖”的任务做拓扑；无依赖任务在上游已走并行流程
    dependent_ids: List[str] = [t.id for t in all_tasks if _get_normalized_base_on_context(t)]
    dependent_set: Set[str] = set(dependent_ids)

    # 预错误：缺失依赖 / 成环或环下游
    pre_error_by_qid: Dict[str, str] = {}

    # Kahn 算法所需结构：入度 + 邻接表
    indeg: Dict[str, int] = {qid: 0 for qid in dependent_ids}
    out_edges: Dict[str, List[str]] = {qid: [] for qid in dependent_ids}

    # 2) 第一遍：先识别 missing_dependency 节点，避免后续把它们当成“必须释放入度”的前置节点
    missing_error_ids: Set[str] = set()
    for qid in dependent_ids:
        task = id_to_task[qid]
        deps = _get_normalized_base_on_context(task)  # deps: 当前问题的依赖列表
        missing = [d for d in deps if d not in all_ids]
        if missing:
            pre_error_by_qid[qid] = f"missing_dependency: missing={missing}"
            missing_error_ids.add(qid)

    # 3) 第二遍：只对“非 missing 节点”建边；
    #    且当依赖节点本身是 missing 时，忽略该依赖边（避免下游被误判为 cycle_dependency_or_downstream）
    for qid in dependent_ids:
        if qid in missing_error_ids:
            continue
        task = id_to_task[qid]
        deps = _get_normalized_base_on_context(task)
        # dep_id -> qid（表示 qid 依赖 dep_id）
        for dep_id in deps:
            if dep_id in dependent_set and dep_id not in missing_error_ids:
                out_edges[dep_id].append(qid)
                indeg[qid] += 1

    # 4) 初始化队列：入度为 0 且非预错误节点，按输入顺序稳定排序
    ready: List[str] = sorted(
        [qid for qid in dependent_ids if indeg.get(qid, 0) == 0 and qid not in pre_error_by_qid],
        key=lambda x: input_index.get(x, 10**9),
    )

    # 5) Kahn 主循环：不断弹出“可执行”节点并释放后继入度
    ordered: List[str] = []
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for nxt in out_edges.get(cur, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0 and nxt not in pre_error_by_qid:
                ready.append(nxt)
                ready.sort(key=lambda x: input_index.get(x, 10**9))

    # 6) 主循环后仍未出队且非 missing 的节点，统一视为
    #    “环及环上节点的下游节点”（实现上不区分纯环节点与环下游）
    remaining = [qid for qid in dependent_ids if qid not in ordered and qid not in pre_error_by_qid]
    if remaining:
        cycle_nodes = sorted(remaining, key=lambda x: input_index.get(x, 10**9))
        for qid in cycle_nodes:
            pre_error_by_qid[qid] = (
                "cycle_dependency_or_downstream: "
                f"nodes={cycle_nodes}, msg=检测到环及环上节点的下游节点"
            )

    return [id_to_task[qid] for qid in ordered], pre_error_by_qid


def _build_dependency_context_text(
    *,
    task: QuestionTask,
    answered_events_by_id: Dict[str, ContentGenerateResponseEvent],
) -> str:
    """
    将当前任务依赖的问题答案拼成一段“补充上下文”。

    说明：
    - 只拼“已回答且可取到事件”的依赖项；
    - 对 ERROR/不可回答 的依赖题答案直接忽略（不注入 dependency_context）；
    - 缺失项直接跳过（缺失依赖属于上游硬错误，这里只做容错拼接）；
    - 返回空字符串表示无需注入 dependency_context。
    """
    deps = _get_normalized_base_on_context(task)
    if not deps:
        return ""

    parts: List[str] = ["以下是当前问题依赖的已回答问题及其答案（供参考）："]
    for dep_id in deps:
        # 依赖题未出现在已回答缓存中时，忽略该项
        ev = answered_events_by_id.get(dep_id)
        if ev is None:
            logger.warning(f"依赖问题 {dep_id} 未出现在已回答缓存中，跳过")
            continue
        # 依赖题为 ERROR 时忽略
        if _is_error_event(ev):
            logger.info(f"依赖问题 {dep_id} 为 ERROR，跳过其答案注入")
            continue
        # 依赖题不可回答时忽略
        if getattr(ev.answer, "is_answerable", True) is False:
            logger.info(f"依赖问题 {dep_id} 不可回答(is_answerable=False)，跳过其答案注入")
            continue
        q = (ev.question_text or "").strip()
        a = (getattr(ev.answer, "answer_text", "") or "").strip()
        if not a:
            logger.warning(f"依赖问题 {dep_id} 答案为空，跳过其答案注入")
            continue
        parts.append(f"\n[依赖问题 {dep_id}] {q}\n[依赖答案 {dep_id}]\n{a}\n")
    return "\n".join(parts).strip() + "\n"


def _is_error_event(ev: ContentGenerateResponseEvent) -> bool:
    """判断事件是否为统一错误格式（answer_text 以 [ERROR] 开头）。"""
    return (getattr(ev.answer, "answer_text", "") or "").startswith("[ERROR]")



def _has_custom_config(task: QuestionTask) -> bool:
    """
    判断问题是否需要走自定义（custom）流程的配置：
    - 只要 tools / knowledge / data_sources 三者中任意一个为非空列表，即认为该问题带有自定义配置。
    """
    has_tools = task.tools is not None and len(task.tools) > 0
    has_knowledge = task.knowledge is not None and len(task.knowledge) > 0
    has_data_sources = task.data_sources is not None and len(task.data_sources) > 0
    return has_tools or has_knowledge or has_data_sources


def _create_error_event(
    task: QuestionTask,
    error_message: str,
    locale: str = "zh_CN",
) -> ContentGenerateResponseEvent:
    error_answer = FinalAnswer(
        is_answerable=False,
        answer_text=get_localized_message("generic_question_processing_error", locale, error=error_message),
        citations=[],
    )
    return ContentGenerateResponseEvent(
        question_id=task.id,
        question_text=task.description,
        question_type=task.type,
        answer=error_answer,
    )


@dataclass(frozen=True)
class IndependentBuckets:
    no_tools_text: List[QuestionTask]
    with_tools_text: List[QuestionTask]
    no_tools_table: List[QuestionTask]
    with_tools_table: List[QuestionTask]
    image: List[QuestionTask]


def bucket_independent_questions_by_tools_and_type(
    independent_questions: List[QuestionTask],
) -> IndependentBuckets:
    """
    按 “tools 是否非空” 再按 “type(text/table)” 分桶。
    - `text-json` 与 `text` 同路径：除最后一步 reporter 模板不同外，其余处理逻辑一致，故也归入 text 桶。
    TODO: 注意：当前 table 流程不支持 tools 影响检索/回答策略；with_tools_table 会按 table 方式处理并记录 warning。
    """
    no_tools_text: List[QuestionTask] = []
    with_tools_text: List[QuestionTask] = []
    no_tools_table: List[QuestionTask] = []
    with_tools_table: List[QuestionTask] = []
    image: List[QuestionTask] = []

    for t in independent_questions:
        if t.type == "image":
            image.append(t)
            continue
        has_tools = _has_custom_config(t)
        if t.type == "table":
            (with_tools_table if has_tools else no_tools_table).append(t)
        else:
            # 默认将非 table 视为 text（text-json 也走该分支）
            (with_tools_text if has_tools else no_tools_text).append(t)

    return IndependentBuckets(
        no_tools_text=no_tools_text,
        with_tools_text=with_tools_text,
        no_tools_table=no_tools_table,
        with_tools_table=with_tools_table,
        image=image,
    )


async def _forward_generator(
    gen: AsyncGenerator[ContentGenerateResponseEvent, None],
    out_queue: "asyncio.Queue[Optional[ContentGenerateResponseEvent]]",
    on_error: Optional[Callable[[Exception], Iterable[ContentGenerateResponseEvent]]] = None,
) -> None:
    logger.info(f"子生成器开始执行，类型: {type(gen)}")
    try:
        async for ev in gen:
            await out_queue.put(ev)
            logger.info(
                "子生成器输出事件: {}",
                redact_data_uris_for_log(ev.model_dump(by_alias=True)),
            )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="子生成器执行失败，将触发兜底错误事件",
            e=e,
            exc_info=True,
            max_message_chars=200,
        )
        if on_error is not None:
            try:
                for err_ev in on_error(e):
                    await out_queue.put(err_ev)
            except Exception:
                logger.exception("兜底错误事件生成失败", exc_info=True)
    finally:
        # sentinel: 表示该子生成器结束
        await out_queue.put(None)


async def merge_event_streams(
    generators: List[AsyncGenerator[ContentGenerateResponseEvent, None]],
    *,
    error_handlers: Optional[List[Optional[Callable[[Exception], Iterable[ContentGenerateResponseEvent]]]]] = None,
) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
    """
    合并多个 async generator 为一个 event stream。
    - 不保证跨 stream 的顺序（以“先完成先输出”为准）
    - 单个 stream 内的顺序由其自身保证
    """
    if not generators:
        return

    if error_handlers is None:
        logger.warning(f"error_handlers 为 None，设置为 [None] * {len(generators)}")
        error_handlers = [None] * len(generators)
    if len(error_handlers) != len(generators):
        raise ValueError("error_handlers 与 generators 长度不一致")

    q: "asyncio.Queue[Optional[ContentGenerateResponseEvent]]" = asyncio.Queue()
    tasks: List[asyncio.Task] = []

    for gen, eh in zip(generators, error_handlers):
        tasks.append(asyncio.create_task(_forward_generator(gen, q, on_error=eh)))

    finished_streams = 0
    try:
        while finished_streams < len(tasks):
            item = await q.get()
            if item is None:
                finished_streams += 1
                continue
            yield item
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class ContentGenerateQuestionScheduler:
    """
    统一调度器：按 tools → type 分类，并尽量并行执行不同 bucket。
    - 不同独立问题 buckets 尽量并行
    """

    def __init__(
        self,
        *,
        table_disable_asegment: bool = True,
        table_disable_keyword_extract: bool = False,
    ) -> None:
        self.table_disable_asegment = table_disable_asegment
        self.table_disable_keyword_extract = table_disable_keyword_extract

    async def _run_independent_text_no_tools(
        self,
        *,
        typed_items: List[Any],
        content_agent: "ContentGenerateAgent",
        tasks: List[QuestionTask],
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        if not tasks:
            logger.warning("问题任务列表为空，无法处理")
            return

        async for ev in content_agent.text_answer_generator(typed_items=typed_items, question_tasks=tasks):
            yield ev

    async def _run_independent_text_with_tools(
        self,
        *,
        typed_items: List[Any],
        content_agent: "ContentGenerateAgent",
        tasks: List[QuestionTask],
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        if not tasks:
            logger.warning("问题任务列表为空，无法处理")
            return

        async for ev in content_agent.custom_answer_generator(typed_items=typed_items, question_tasks=tasks):
            yield ev

    async def _run_independent_tables(
        self,
        *,
        typed_items: List[Any],
        model_id: str,
        tasks: List[QuestionTask],
        answer_lang: str,
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        """
        独立表格问题（无 tools）的处理流程。
        Args:
            typed_items: 原始内容列表
            model_id: 模型ID
            tasks: 问题任务列表
        Returns:
            AsyncGenerator[ContentGenerateResponseEvent, None]: 内容生成事件生成器
        """
        if not tasks:
            logger.warning("问题任务列表为空，无法处理")
            return

        # 延迟导入：避免在纯“分类/分桶”场景下引入 table 依赖链
        from .table_answer_components.answer_graph.answer_table_agent import AnswerTableAgentService
        from .config import create_default_configs
        cfg=create_default_configs()
        coding_model_id=cfg.coding_model_id
        max_iterations=cfg.coding_max_retry_limit
        if coding_model_id is None or coding_model_id == "":
            coding_model_id = model_id
            logger.warning(f"coding_model_id set to none fallback to default model_id:{coding_model_id}")
        if max_iterations is None or max_iterations == 0:
            max_iterations = 5
            logger.warning(f"coding_max_retry_limit set to none, using default:{max_iterations}")
        table_agent_service = AnswerTableAgentService()
        async for ev in table_agent_service.table_answering(
            typed_items=typed_items,
            question_tasks=tasks,
            disable_keyword_extract=self.table_disable_keyword_extract,
            model_id=coding_model_id,
            max_iterations=max_iterations,
            answer_lang=answer_lang,
        ):
            yield ev

    async def _run_image_question(
        self,
        *,
        typed_items: List[Any],
        task: QuestionTask,
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        """处理一题图片答案，保持既有事件与回调结构不变。"""
        answer = await ImageAnswerService().answer(task=task, typed_items=typed_items)
        yield ContentGenerateResponseEvent(
            question_id=task.id,
            question_text=task.description,
            question_type="image",
            answer=answer,
        )

    async def run(
        self,
        *,
        typed_items: List[Any],
        question_tasks: List[QuestionTask],
        content_agent: "ContentGenerateAgent",
        model_id_for_table: Optional[str] = None,
    ) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
        if not question_tasks:
            return

        model_id_for_table = model_id_for_table or content_agent.model_id

        # Step 1) 依赖分组：无依赖（可并行） vs 有依赖（最后串行）
        independent_tasks: List[QuestionTask] = [t for t in question_tasks if not _get_normalized_base_on_context(t)]
        dependent_tasks: List[QuestionTask] = [t for t in question_tasks if _get_normalized_base_on_context(t)]

        # Step 2) 先跑无依赖任务（沿用原有分桶并行逻辑）
        buckets = bucket_independent_questions_by_tools_and_type(independent_tasks)

        # TODO: 待添加 tools 支持 table 流程
        # 注意：table 流程目前不支持 tools 的特殊行为，但仍按 tools→type 分类路径走到 table handler
        if buckets.with_tools_table:
            logger.error(
                f"检测到 {len(buckets.with_tools_table)} 个 Table 问题带 tools 配置，当前 table 流程不会使用 tools；tools字段将被忽略。"
                f"ids={[t.id for t in buckets.with_tools_table]}"
            )

        gens: List[AsyncGenerator[ContentGenerateResponseEvent, None]] = []
        ehs: List[Optional[Callable[[Exception], Iterable[ContentGenerateResponseEvent]]]] = []

        def _eh_for_tasks(ts: List[QuestionTask]) -> Callable[[Exception], Iterable[ContentGenerateResponseEvent]]:
            def _handler(e: Exception) -> Iterable[ContentGenerateResponseEvent]:
                return [_create_error_event(t, str(e), locale=content_agent.answer_lang) for t in ts]

            return _handler

        # independent tasks：每个问题独立一个 stream，全部并行
        for task in buckets.no_tools_text:
            gens.append(
                self._run_independent_text_no_tools(
                    typed_items=typed_items,
                    content_agent=content_agent,
                    tasks=[task],
                )
            )
            ehs.append(_eh_for_tasks([task]))

        for task in buckets.with_tools_text:
            gens.append(
                self._run_independent_text_with_tools(
                    typed_items=typed_items,
                    content_agent=content_agent,
                    tasks=[task],
                )
            )
            ehs.append(_eh_for_tasks([task]))

        for task in buckets.no_tools_table + buckets.with_tools_table:
            gens.append(
                self._run_independent_tables(
                    typed_items=typed_items,
                    model_id=model_id_for_table,
                    tasks=[task],
                    answer_lang=content_agent.answer_lang,
                )
            )
            ehs.append(_eh_for_tasks([task]))

        for task in buckets.image:
            gens.append(
                self._run_image_question(
                    typed_items=typed_items,
                    task=task,
                )
            )
            ehs.append(_eh_for_tasks([task]))

        answered_events_by_id: Dict[str, ContentGenerateResponseEvent] = {}

        async for ev in merge_event_streams(gens, error_handlers=ehs):
            answered_events_by_id[ev.question_id] = ev
            yield ev

        # Step 3) 最后串行回答所有有依赖任务（拓扑顺序）
        if not dependent_tasks:
            return

        ordered_dependent_tasks, pre_error_by_qid = _topo_sort_dependent_tasks(question_tasks)
        # 算法校验：当前实现下 ordered 与 pre_error 应互斥，故可直接做一行长度校验
        if len(ordered_dependent_tasks) + len(pre_error_by_qid) != len(dependent_tasks):
            logger.error(
                "依赖任务拓扑排序结果覆盖异常："
                f"dependent_count={len(dependent_tasks)}, "
                f"ordered_count={len(ordered_dependent_tasks)}, "
                f"pre_error_count={len(pre_error_by_qid)}"
            )
            raise ValueError(
                "依赖任务拓扑排序结果覆盖异常：ordered_dependent_tasks 与 pre_error_by_qid 未完整覆盖 dependent_tasks"
            )

        # 先输出所有 pre_error 任务（保持 dependent_tasks 输入顺序）：
        # - missing_dependency
        # - cycle_dependency_or_downstream
        yielded_dependent_ids: Set[str] = set()
        for t in dependent_tasks:
            if t.id in pre_error_by_qid:
                err_ev = _create_error_event(t, pre_error_by_qid[t.id])
                answered_events_by_id[err_ev.question_id] = err_ev
                yield err_ev
                yielded_dependent_ids.add(err_ev.question_id)

        # 再串行执行可排序任务：按拓扑顺序逐题回答（保证依赖已回答完毕）
        for task in ordered_dependent_tasks:
            if task.id in pre_error_by_qid:
                raise ValueError(
                    "拓扑排序结果异常：ordered_dependent_tasks 与 pre_error_by_qid 存在交集，"
                    f"重复任务id={task.id}, error={pre_error_by_qid[task.id]}"
                )

            dependency_context = _build_dependency_context_text(
                task=task,
                answered_events_by_id=answered_events_by_id,
            )

            # 依赖上下文将通过下层模板变量注入；这里以 per-question 方式透传
            dependency_context_by_qid = {task.id: dependency_context}

            if task.type == "image":
                async for ev in self._run_image_question(
                    typed_items=typed_items,
                    task=task,
                ):
                    answered_events_by_id[ev.question_id] = ev
                    yield ev
                    yielded_dependent_ids.add(ev.question_id)
                continue

            has_custom = _has_custom_config(task)

            if task.type == "table":
                # TODO: 表格这块待二次审核
                # 表格单题：将 dependency_context 作为历史消息注入（不污染 task.prompt）
                from .table_answer_components.answer_graph.answer_table_agent import AnswerTableAgentService
                from .config import create_default_configs
                cfg=create_default_configs()
                coding_model_id=cfg.coding_model_id
                max_iterations=cfg.coding_max_retry_limit
                if coding_model_id is None or coding_model_id == "":
                    coding_model_id = model_id_for_table
                    logger.warning(f"coding_model_id set to none fallback to default model_id:{coding_model_id}")
                if max_iterations is None or max_iterations == 0:
                    max_iterations = 5
                    logger.warning(f"coding_max_retry_limit set to none, using default:{max_iterations}")
                table_agent_service = AnswerTableAgentService()
                history_messages = [("human", dependency_context)] if dependency_context else []
                async for ev in table_agent_service.answer_single_table_with_msg(
                    typed_items=typed_items,
                    question_task=task,
                    history_messages=history_messages,
                    model_id=coding_model_id,
                    max_iterations=max_iterations,
                    disable_keyword_extract=self.table_disable_keyword_extract,
                ):
                    answered_events_by_id[ev.question_id] = ev
                    yield ev
                    yielded_dependent_ids.add(ev.question_id)
                continue

            gen = (
                content_agent.custom_answer_generator(
                    typed_items=typed_items,
                    question_tasks=[task],
                    dependency_context_by_qid=dependency_context_by_qid,
                )
                if has_custom
                else content_agent.text_answer_generator(
                    typed_items=typed_items,
                    question_tasks=[task],
                    dependency_context_by_qid=dependency_context_by_qid,
                )
            )
            async for ev in gen:
                answered_events_by_id[ev.question_id] = ev
                yield ev
                yielded_dependent_ids.add(ev.question_id)

        # 最终一致性校验：dependent tasks 都应已输出
        for t in dependent_tasks:
            if t.id in yielded_dependent_ids:
                continue
            raise ValueError(
                "依赖任务输出一致性异常：存在未输出的 dependent task，"
                f"task_id={t.id}, "
                f"in_pre_error={t.id in pre_error_by_qid}, "
                f"in_answered_cache={t.id in answered_events_by_id}"
            )

