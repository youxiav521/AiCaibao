"""
ContentGenerateQuestionScheduler 单元测试

验证目标
--------
1. bucket_independent_questions_by_tools_and_type 分桶正确性
2. _topo_sort_dependent_tasks 拓扑排序正确性（线性/菱形/缺依赖/成环）
3. _has_custom_config 判"自定义流程"是否正确
4. _build_dependency_context_text 构建依赖上下文正确性
5. _create_error_event 生成错误事件正确性
6. merge_event_streams 多流合并与并行执行正确性
7. ContentGenerateQuestionScheduler.run() 每道独立问题一个 stream、并行执行
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.agents.content_generate_agent.question_scheduler as question_scheduler_module
from src.agents.content_generate_agent.models import (
    ContentGenerateResponseEvent,
    FinalAnswer,
    QuestionTask,
)
from src.agents.content_generate_agent.question_scheduler import (
    ContentGenerateQuestionScheduler,
    _build_dependency_context_text,
    _create_error_event,
    _get_normalized_base_on_context,
    _has_custom_config,
    _topo_sort_dependent_tasks,
    bucket_independent_questions_by_tools_and_type,
    merge_event_streams,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_task(
    qid: str,
    description: str = "",
    qtype: str = "text",
    base_on_context: list[str] | None = None,
    tools: list[dict] | None = None,
    knowledge: list[str] | None = None,
    data_sources: list[dict] | None = None,
) -> QuestionTask:
    return QuestionTask(
        id=qid,
        description=description or f"question {qid}",
        type=qtype,
        base_on_context=base_on_context,
        tools=tools,
        knowledge=knowledge,
        data_sources=data_sources,
    )


def _make_event(
    question_id: str,
    question_text: str = "",
    question_type: str = "text",
    answer_text: str = "answer",
    is_answerable: bool = True,
) -> ContentGenerateResponseEvent:
    return ContentGenerateResponseEvent(
        question_id=question_id,
        question_text=question_text or f"question {question_id}",
        question_type=question_type,
        answer=FinalAnswer(
            is_answerable=is_answerable,
            answer_text=answer_text,
            citations=[],
        ),
    )


async def _async_gen_from_list(
    items: list[ContentGenerateResponseEvent],
) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
    for item in items:
        yield item


async def _delayed_async_gen(
    items: list[ContentGenerateResponseEvent],
    delay: float,
) -> AsyncGenerator[ContentGenerateResponseEvent, None]:
    await asyncio.sleep(delay)
    for item in items:
        yield item


# ──────────────────────────────────────────────────────────────────────────────
# get_normalized_base_on_context
# ──────────────────────────────────────────────────────────────────────────────


class TestGetNormalizedBaseOnContext:
    def test_none(self):
        task = _make_task("q1", base_on_context=None)
        assert _get_normalized_base_on_context(task) == []

    def test_empty_list(self):
        task = _make_task("q1", base_on_context=[])
        assert _get_normalized_base_on_context(task) == []

    def test_has_deps(self):
        task = _make_task("q1", base_on_context=["q_a", "q_b"])
        assert _get_normalized_base_on_context(task) == ["q_a", "q_b"]

    def test_returns_copy_not_same_list(self):
        deps = ["q_a"]
        task = _make_task("q1", base_on_context=deps)
        result = _get_normalized_base_on_context(task)
        assert result == deps
        assert result is not deps


# ──────────────────────────────────────────────────────────────────────────────
# bucket_independent_questions_by_tools_and_type
# ──────────────────────────────────────────────────────────────────────────────


class TestBucketIndependentQuestionsByToolsAndType:
    def test_empty_list(self):
        result = bucket_independent_questions_by_tools_and_type([])
        assert result.no_tools_text == []
        assert result.with_tools_text == []
        assert result.no_tools_table == []
        assert result.with_tools_table == []
        assert result.image == []

    def test_no_tools_text(self):
        tasks = [_make_task("q1", qtype="text")]
        result = bucket_independent_questions_by_tools_and_type(tasks)
        assert len(result.no_tools_text) == 1
        assert result.with_tools_text == []

    def test_text_json_goes_to_no_tools_text(self):
        tasks = [_make_task("q1", qtype="text-json")]
        result = bucket_independent_questions_by_tools_and_type(tasks)
        assert len(result.no_tools_text) == 1

    def test_with_tools_text(self):
        tasks = [_make_task("q1", qtype="text", tools=[{"type": "coding_agent", "enable": True}])]
        result = bucket_independent_questions_by_tools_and_type(tasks)
        assert len(result.with_tools_text) == 1
        assert result.no_tools_text == []

    def test_no_tools_table(self):
        tasks = [_make_task("q1", qtype="table")]
        result = bucket_independent_questions_by_tools_and_type(tasks)
        assert len(result.no_tools_table) == 1

    def test_with_tools_table(self):
        tasks = [_make_task("q1", qtype="table", tools=[{"type": "coding_agent", "enable": True}])]
        result = bucket_independent_questions_by_tools_and_type(tasks)
        assert len(result.with_tools_table) == 1

    def test_mixed_all_buckets(self):
        tasks = [
            _make_task("q1", qtype="text"),
            _make_task("q2", qtype="text", tools=[{"type": "coding_agent", "enable": True}]),
            _make_task("q3", qtype="table"),
            _make_task("q4", qtype="table", tools=[{"type": "coding_agent", "enable": True}]),
        ]
        result = bucket_independent_questions_by_tools_and_type(tasks)
        assert [t.id for t in result.no_tools_text] == ["q1"]
        assert [t.id for t in result.with_tools_text] == ["q2"]
        assert [t.id for t in result.no_tools_table] == ["q3"]
        assert [t.id for t in result.with_tools_table] == ["q4"]

    def test_image_questions_use_the_image_bucket(self):
        result = bucket_independent_questions_by_tools_and_type([
            _make_task("q-image", qtype="image"),
        ])
        assert [task.id for task in result.image] == ["q-image"]
        assert result.no_tools_text == []


# ──────────────────────────────────────────────────────────────────────────────
# _has_custom_config
# ──────────────────────────────────────────────────────────────────────────────


class TestHasCustomConfig:
    def test_all_none(self):
        task = _make_task("q1")
        assert _has_custom_config(task) is False

    def test_with_tools(self):
        task = _make_task("q1", tools=[{"type": "coding_agent", "enable": True}])
        assert _has_custom_config(task) is True

    def test_with_tools_empty_list(self):
        task = _make_task("q1", tools=[])
        assert _has_custom_config(task) is False

    def test_with_knowledge(self):
        task = _make_task("q1", knowledge=["file_1"])
        assert _has_custom_config(task) is True

    def test_with_knowledge_empty_list(self):
        task = _make_task("q1", knowledge=[])
        assert _has_custom_config(task) is False

    def test_with_data_sources(self):
        task = _make_task("q1", data_sources=[{"type": "other_datasource", "data": "some data"}])
        assert _has_custom_config(task) is True

    def test_with_data_sources_empty_list(self):
        task = _make_task("q1", data_sources=[])
        assert _has_custom_config(task) is False


# ──────────────────────────────────────────────────────────────────────────────
# _create_error_event
# ──────────────────────────────────────────────────────────────────────────────


class TestCreateErrorEvent:
    def test_creates_unanswerable_event(self):
        task = _make_task("q1", description="test question")
        event = _create_error_event(task, "something went wrong")
        assert event.question_id == "q1"
        assert event.question_text == "test question"
        assert event.question_type == "text"
        assert event.answer.is_answerable is False
        assert "something went wrong" in event.answer.answer_text

    def test_locale_en(self):
        task = _make_task("q1", description="test question")
        event = _create_error_event(task, "error msg", locale="en_US")
        assert event.question_id == "q1"
        assert event.answer.is_answerable is False
        assert "error msg" in event.answer.answer_text


# ──────────────────────────────────────────────────────────────────────────────
# _topo_sort_dependent_tasks
# ──────────────────────────────────────────────────────────────────────────────


class TestTopoSortDependentTasks:
    def test_no_dependencies_returns_empty(self):
        tasks = [_make_task("q1"), _make_task("q2")]
        ordered, errors = _topo_sort_dependent_tasks(tasks)
        assert ordered == []
        assert errors == {}

    def test_linear_chain(self):
        tasks = [
            _make_task("q1"),
            _make_task("q2", base_on_context=["q1"]),
            _make_task("q3", base_on_context=["q2"]),
        ]
        ordered, errors = _topo_sort_dependent_tasks(tasks)
        assert errors == {}
        ids = [t.id for t in ordered]
        assert ids == ["q2", "q3"]

    def test_missing_dependency(self):
        tasks = [
            _make_task("q1", base_on_context=["q_unknown"]),
        ]
        ordered, errors = _topo_sort_dependent_tasks(tasks)
        assert ordered == []
        assert "q1" in errors
        assert "missing_dependency" in errors["q1"]
        assert "q_unknown" in errors["q1"]

    def test_cycle_detection(self):
        tasks = [
            _make_task("q1", base_on_context=["q2"]),
            _make_task("q2", base_on_context=["q1"]),
        ]
        ordered, errors = _topo_sort_dependent_tasks(tasks)
        assert "q1" in errors
        assert "q2" in errors

    def test_diamond_deps(self):
        tasks = [
            _make_task("q1"),
            _make_task("q2", base_on_context=["q1"]),
            _make_task("q3", base_on_context=["q1"]),
            _make_task("q4", base_on_context=["q2", "q3"]),
        ]
        ordered, errors = _topo_sort_dependent_tasks(tasks)
        assert errors == {}
        ids = [t.id for t in ordered]
        assert len(ids) == 3
        assert ids.index("q2") < ids.index("q4")
        assert ids.index("q3") < ids.index("q4")

    def test_independent_tasks_are_excluded(self):
        tasks = [
            _make_task("q1"),
            _make_task("q2", base_on_context=["q1"]),
        ]
        ordered, errors = _topo_sort_dependent_tasks(tasks)
        ids = [t.id for t in ordered]
        assert "q1" not in ids
        assert "q2" in ids

    def test_missing_dep_excluded_from_ordered(self):
        tasks = [
            _make_task("q1", base_on_context=["q_unknown"]),
            _make_task("q2", base_on_context=["q1"]),
        ]
        ordered, errors = _topo_sort_dependent_tasks(tasks)
        assert "q1" in errors
        assert len(ordered) <= 1


# ──────────────────────────────────────────────────────────────────────────────
# _build_dependency_context_text
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildDependencyContextText:
    def test_no_deps(self):
        task = _make_task("q1")
        result = _build_dependency_context_text(task=task, answered_events_by_id={})
        assert result == ""

    def test_no_answered_events(self):
        task = _make_task("q1", base_on_context=["q0"])
        result = _build_dependency_context_text(task=task, answered_events_by_id={})
        assert "q0" not in result

    def test_with_single_answered_dep(self):
        dep_ev = _make_event("q0", answer_text="dep answer text")
        task = _make_task("q1", base_on_context=["q0"])
        result = _build_dependency_context_text(task=task, answered_events_by_id={"q0": dep_ev})
        assert "q0" in result
        assert "dep answer text" in result

    def test_error_event_skipped(self):
        dep_ev = _make_event("q0", answer_text="[ERROR] something failed")
        task = _make_task("q1", base_on_context=["q0"])
        result = _build_dependency_context_text(task=task, answered_events_by_id={"q0": dep_ev})
        assert "something failed" not in result

    def test_unanswerable_skipped(self):
        dep_ev = _make_event("q0", answer_text="some text", is_answerable=False)
        task = _make_task("q1", base_on_context=["q0"])
        result = _build_dependency_context_text(task=task, answered_events_by_id={"q0": dep_ev})
        assert "some text" not in result

    def test_empty_answer_skipped(self):
        dep_ev = _make_event("q0", answer_text="")
        task = _make_task("q1", base_on_context=["q0"])
        result = _build_dependency_context_text(task=task, answered_events_by_id={"q0": dep_ev})
        assert "q0" not in result

    def test_multiple_deps_mixed(self):
        dep_ev_a = _make_event("qA", answer_text="answer A")
        dep_ev_b = _make_event("qB", answer_text="[ERROR] fail")
        dep_ev_c = _make_event("qC", answer_text="answer C", is_answerable=False)
        task = _make_task("q1", base_on_context=["qA", "qB", "qC"])
        result = _build_dependency_context_text(
            task=task, answered_events_by_id={"qA": dep_ev_a, "qB": dep_ev_b, "qC": dep_ev_c}
        )
        assert "answer A" in result
        assert "answer C" not in result
        assert "fail" not in result


# ──────────────────────────────────────────────────────────────────────────────
# merge_event_streams
# ──────────────────────────────────────────────────────────────────────────────


class TestMergeEventStreams:
    @pytest.mark.asyncio
    async def test_empty_generators(self):
        events = [e async for e in merge_event_streams([])]
        assert events == []

    @pytest.mark.asyncio
    async def test_single_generator(self):
        ev = _make_event("q1")
        gen = _async_gen_from_list([ev])
        events = [e async for e in merge_event_streams([gen])]
        assert len(events) == 1
        assert events[0].question_id == "q1"

    @pytest.mark.asyncio
    async def test_multiple_generators_merge_all_events(self):
        gen_a = _async_gen_from_list([_make_event("a1"), _make_event("a2")])
        gen_b = _async_gen_from_list([_make_event("b1")])
        gen_c = _async_gen_from_list([_make_event("c1"), _make_event("c2"), _make_event("c3")])

        events = [e async for e in merge_event_streams([gen_a, gen_b, gen_c])]
        qids = {e.question_id for e in events}
        assert len(events) == 6
        assert qids == {"a1", "a2", "b1", "c1", "c2", "c3"}

    @pytest.mark.asyncio
    async def test_parallel_execution_time(self):
        """验证 merge_event_streams 并行执行：总耗时接近最长任务而非所有任务之和。"""
        delays = [0.1, 0.2, 0.15]
        gens = [
            _delayed_async_gen([_make_event(f"q{i}")], d)
            for i, d in enumerate(delays)
        ]

        t0 = time.perf_counter()
        events = [e async for e in merge_event_streams(gens)]
        elapsed = time.perf_counter() - t0

        assert len(events) == 3
        assert elapsed < sum(delays) * 1.5  # 并行总时间应远小于串行之和
        assert elapsed >= max(delays) * 0.8  # 但至少接近最长任务

    @pytest.mark.asyncio
    async def test_error_handler_produces_fallback_events(self):
        async def bad_gen() -> AsyncGenerator[ContentGenerateResponseEvent, None]:
            raise ValueError("gen failed")
            yield  # pragma: no cover

        def error_handler(exc: Exception) -> Iterable[ContentGenerateResponseEvent]:
            return [
                _create_error_event(_make_task("fallback"), str(exc))
            ]

        events = [e async for e in merge_event_streams([bad_gen()], error_handlers=[error_handler])]
        assert len(events) == 1
        assert events[0].question_id == "fallback"
        assert "gen failed" in events[0].answer.answer_text

    @pytest.mark.asyncio
    async def test_no_error_handler_swallows_and_continues(self):
        async def bad_gen() -> AsyncGenerator[ContentGenerateResponseEvent, None]:
            raise ValueError("gen failed")
            yield  # pragma: no cover

        ev = _make_event("q_good")
        events = [e async for e in merge_event_streams([bad_gen(), _async_gen_from_list([ev])])]
        assert len(events) == 1
        assert events[0].question_id == "q_good"


# ──────────────────────────────────────────────────────────────────────────────
# ContentGenerateQuestionScheduler
# ──────────────────────────────────────────────────────────────────────────────


class TestContentGenerateQuestionScheduler:
    def test_constructor_defaults(self):
        s = ContentGenerateQuestionScheduler()
        assert s.table_disable_asegment is True
        assert s.table_disable_keyword_extract is False

    def test_constructor_custom(self):
        s = ContentGenerateQuestionScheduler(
            table_disable_asegment=False,
            table_disable_keyword_extract=True,
        )
        assert s.table_disable_asegment is False
        assert s.table_disable_keyword_extract is True

    @pytest.mark.asyncio
    async def test_run_empty_question_tasks(self):
        scheduler = ContentGenerateQuestionScheduler()
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=[],
                content_agent=content_agent,
            )
        ]
        assert events == []

    @pytest.mark.asyncio
    async def test_run_single_no_tools_text(self):
        """单个独立 text 问题：走 text_answer_generator。"""
        scheduler = ContentGenerateQuestionScheduler()
        task = _make_task("q1", qtype="text")
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        ev = _make_event("q1", answer_text="hello")
        content_agent.text_answer_generator = MagicMock()
        content_agent.text_answer_generator.return_value = _async_gen_from_list([ev])

        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=[task],
                content_agent=content_agent,
            )
        ]
        assert len(events) == 1
        assert events[0].question_id == "q1"
        assert events[0].answer.answer_text == "hello"

        content_agent.text_answer_generator.assert_called_once()
        call_kwargs = content_agent.text_answer_generator.call_args.kwargs
        assert len(call_kwargs["question_tasks"]) == 1
        assert call_kwargs["question_tasks"][0] is task

    @pytest.mark.asyncio
    async def test_run_single_with_tools_text(self):
        """单个带 tools 的 text 问题：走 custom_answer_generator。"""
        scheduler = ContentGenerateQuestionScheduler()
        task = _make_task("q1", qtype="text", tools=[{"type": "coding_agent", "enable": True}])
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        ev = _make_event("q1", answer_text="custom answer")
        content_agent.custom_answer_generator = MagicMock()
        content_agent.custom_answer_generator.return_value = _async_gen_from_list([ev])

        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=[task],
                content_agent=content_agent,
            )
        ]
        assert len(events) == 1
        assert events[0].question_id == "q1"
        assert events[0].answer.answer_text == "custom answer"

        content_agent.custom_answer_generator.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_single_image_uses_image_answer_service(self, monkeypatch):
        scheduler = ContentGenerateQuestionScheduler()
        task = _make_task("q-image", qtype="image")
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        class FakeImageAnswerService:
            async def answer(self, task, typed_items):
                return FinalAnswer(
                    is_answerable=True,
                    answer_text="data:image/png;base64,eA==",
                    citations=[],
                )

        monkeypatch.setattr(
            question_scheduler_module,
            "ImageAnswerService",
            FakeImageAnswerService,
            raising=False,
        )
        events = [
            event async for event in scheduler.run(
                typed_items=[],
                question_tasks=[task],
                content_agent=content_agent,
            )
        ]

        assert len(events) == 1
        assert events[0].question_type == "image"
        assert events[0].answer.answer_text == "data:image/png;base64,eA=="

    @pytest.mark.asyncio
    async def test_dependent_image_question_uses_image_answer_service(self, monkeypatch):
        scheduler = ContentGenerateQuestionScheduler()
        independent_task = _make_task("q-text", qtype="text")
        image_task = _make_task("q-image", qtype="image", base_on_context=["q-text"])
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        async def text_generator(typed_items, question_tasks, **kwargs):
            yield _make_event(question_tasks[0].id, answer_text="文本答案")

        class FakeImageAnswerService:
            async def answer(self, task, typed_items):
                return FinalAnswer(
                    is_answerable=True,
                    answer_text="data:image/svg+xml;base64,c3Zn",
                    citations=[],
                )

        content_agent.text_answer_generator = text_generator
        monkeypatch.setattr(
            question_scheduler_module,
            "ImageAnswerService",
            FakeImageAnswerService,
            raising=False,
        )
        events = [
            event async for event in scheduler.run(
                typed_items=[],
                question_tasks=[independent_task, image_task],
                content_agent=content_agent,
            )
        ]

        assert [event.question_id for event in events] == ["q-text", "q-image"]
        assert events[-1].answer.answer_text == "data:image/svg+xml;base64,c3Zn"

    @pytest.mark.asyncio
    async def test_each_independent_question_gets_its_own_stream(self):
        """重构核心验证：每个独立问题各自一个 stream，不再分批。"""
        scheduler = ContentGenerateQuestionScheduler()

        tasks = [_make_task(f"q{i}", qtype="text") for i in range(5)]
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        call_count = 0

        async def mock_text_gen(typed_items, question_tasks, **kwargs):
            nonlocal call_count
            call_count += 1
            assert len(question_tasks) == 1, (
                f"期望单个 task，实际收到 {len(question_tasks)} 个"
            )
            yield _make_event(question_tasks[0].id, answer_text=f"answer_{question_tasks[0].id}")

        content_agent.text_answer_generator = mock_text_gen

        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=tasks,
                content_agent=content_agent,
            )
        ]
        assert len(events) == 5
        assert call_count == 5
        qids = {e.question_id for e in events}
        assert qids == {f"q{i}" for i in range(5)}

    @pytest.mark.asyncio
    async def test_parallel_execution_independent_questions(self):
        """验证独立问题通过 merge_event_streams 并行执行。"""
        scheduler = ContentGenerateQuestionScheduler()
        tasks = [_make_task(f"q{i}", qtype="text") for i in range(3)]
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        async def delayed_mock(typed_items, question_tasks, **kwargs):
            await asyncio.sleep(0.15)
            t = question_tasks[0]
            yield _make_event(t.id, answer_text=f"answer_{t.id}")

        content_agent.text_answer_generator = delayed_mock

        t0 = time.perf_counter()
        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=tasks,
                content_agent=content_agent,
            )
        ]
        elapsed = time.perf_counter() - t0

        assert len(events) == 3
        # 并行：总时间接近 0.15s，远小于串行的 0.45s
        assert elapsed < 0.45 * 0.8, f"并行执行应远快于串行，但耗时 {elapsed:.3f}s"
        assert elapsed >= max(0.15, 0.15, 0.15) * 0.7

    @pytest.mark.asyncio
    async def test_mixed_bucket_independent_parallel(self):
        """混合桶（no_tools_text + with_tools_text）的独立问题应全部并行。"""
        scheduler = ContentGenerateQuestionScheduler()
        tasks = [
            _make_task("q1", qtype="text"),
            _make_task("q2", qtype="text", tools=[{"type": "coding_agent", "enable": True}]),
            _make_task("q3", qtype="text"),
        ]
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        async def delayed_text(typed_items, question_tasks, **kwargs):
            await asyncio.sleep(0.15)
            t = question_tasks[0]
            yield _make_event(t.id, answer_text=f"text_{t.id}")

        async def delayed_custom(typed_items, question_tasks, **kwargs):
            await asyncio.sleep(0.15)
            t = question_tasks[0]
            yield _make_event(t.id, answer_text=f"custom_{t.id}")

        content_agent.text_answer_generator = delayed_text
        content_agent.custom_answer_generator = delayed_custom

        t0 = time.perf_counter()
        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=tasks,
                content_agent=content_agent,
            )
        ]
        elapsed = time.perf_counter() - t0

        assert len(events) == 3
        # 3 个问题不管桶不同，都应并行
        assert elapsed < 0.45 * 0.8

    @pytest.mark.asyncio
    async def test_dependent_tasks_serial_after_independent(self):
        """有依赖的问题在独立问题全部完成后才串行执行。"""
        scheduler = ContentGenerateQuestionScheduler()

        independent_task = _make_task("q_ind", qtype="text")
        dependent_task = _make_task("q_dep", qtype="text", base_on_context=["q_ind"])

        tasks = [independent_task, dependent_task]
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        execution_order = []

        async def mock_text_gen(typed_items, question_tasks, dependency_context_by_qid=None, **kwargs):
            t = question_tasks[0]
            execution_order.append(t.id)
            yield _make_event(t.id, answer_text=f"answer_{t.id}")

        content_agent.text_answer_generator = mock_text_gen

        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=tasks,
                content_agent=content_agent,
            )
        ]
        assert len(events) == 2
        assert execution_order == ["q_ind", "q_dep"]

    @pytest.mark.asyncio
    async def test_dependent_task_receives_context(self):
        """有依赖的问题能收到已回答问题的上下文。"""
        scheduler = ContentGenerateQuestionScheduler()

        independent_task = _make_task("q_ind", qtype="text")
        dependent_task = _make_task("q_dep", qtype="text", base_on_context=["q_ind"])

        tasks = [independent_task, dependent_task]
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        captured_context = {}

        async def mock_independent(typed_items, question_tasks, **kwargs):
            yield _make_event("q_ind", answer_text="independent answer")

        async def mock_dependent(typed_items, question_tasks, dependency_context_by_qid=None, **kwargs):
            t = question_tasks[0]
            captured_context["ctx"] = dependency_context_by_qid
            yield _make_event(t.id, answer_text="dependent answer")

        content_agent.text_answer_generator = mock_dependent
        ind_gen = mock_independent([], [independent_task])

        with patch.object(
            scheduler,
            "_run_independent_text_no_tools",
            return_value=ind_gen,
        ):
            events = [
                e async for e in scheduler.run(
                    typed_items=[],
                    question_tasks=tasks,
                    content_agent=content_agent,
                )
            ]
        assert len(events) == 2
        assert captured_context["ctx"] is not None
        assert "q_dep" in captured_context["ctx"]

    @pytest.mark.asyncio
    async def test_missing_dependency_error(self):
        """依赖不存在的任务时产生错误事件。"""
        scheduler = ContentGenerateQuestionScheduler()

        broken_task = _make_task("q_broken", qtype="text", base_on_context=["q_missing"])
        tasks = [broken_task]
        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=tasks,
                content_agent=content_agent,
            )
        ]
        assert len(events) == 1
        assert events[0].answer.is_answerable is False
        assert "missing_dependency" in events[0].answer.answer_text

    @pytest.mark.asyncio
    async def test_content_generate_tasks_integration_style(self):
        """模拟 process_content_generate 中的调用方式：用 ContentGenerateAgent mock 跑完整调度。"""
        scheduler = ContentGenerateQuestionScheduler()

        plain_text = {
            "id": "q_plain",
            "description": "some text question",
            "type": "text",
        }
        custom_text = {
            "id": "q_custom",
            "description": "custom question",
            "type": "text",
            "tools": [{"type": "coding_agent", "enable": True}],
            "data_sources": [
                {"type": "chunk_index_retrieval", "query": "test", "top_k": 10}
            ],
        }
        dep_text = {
            "id": "q_dep",
            "description": "depends on q_plain",
            "type": "text",
            "base_on_context": ["q_plain"],
        }

        raw_tasks = [plain_text, custom_text, dep_text]
        questions = [QuestionTask.model_validate(qt) for qt in raw_tasks]

        content_agent = MagicMock()
        content_agent.model_id = "test_model"
        content_agent.answer_lang = "zh_CN"

        async def mock_text(typed_items, question_tasks, dependency_context_by_qid=None, **kwargs):
            for t in question_tasks:
                yield _make_event(t.id, answer_text=f"answer_{t.id}")

        async def mock_custom(typed_items, question_tasks, dependency_context_by_qid=None, **kwargs):
            for t in question_tasks:
                yield _make_event(t.id, answer_text=f"custom_answer_{t.id}")

        content_agent.text_answer_generator = mock_text
        content_agent.custom_answer_generator = mock_custom

        events = [
            e async for e in scheduler.run(
                typed_items=[],
                question_tasks=questions,
                content_agent=content_agent,
            )
        ]
        assert len(events) == 3
        qids = {e.question_id for e in events}
        assert qids == {"q_plain", "q_custom", "q_dep"}
