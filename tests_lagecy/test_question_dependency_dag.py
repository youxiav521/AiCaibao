import pytest


from src.agents.content_generate_agent.models import QuestionTask
from src.agents.content_generate_agent.question_scheduler import _topo_sort_dependent_tasks


def _qt(qid: str, deps=None) -> QuestionTask:
    return QuestionTask.model_validate(
        {
            "id": qid,
            "description": f"question {qid}",
            "type": "text",
            "prompt": "",
            "baseOnContext": deps,
        }
    )


def test_base_on_context_normalization_strip_dedupe_and_empty_to_none():
    t = _qt("q2", deps=[" q1 ", "", "   ", "q1"])
    assert t.base_on_context == ["q1"]

    t2 = _qt("q3", deps=[])
    assert t2.base_on_context is None

    t3 = _qt("q4", deps=None)
    assert t3.base_on_context is None


def test_topo_sort_dependent_tasks_orders_by_dependencies():
    q1 = _qt("q1", deps=None)
    q2 = _qt("q2", deps=["q1"])
    q3 = _qt("q3", deps=["q2"])

    ordered, pre_err = _topo_sort_dependent_tasks([q1, q2, q3])
    assert [t.id for t in ordered] == ["q2", "q3"]
    assert pre_err == {}


def test_topo_sort_missing_dependency_marks_error():
    q1 = _qt("q1", deps=None)
    q2 = _qt("q2", deps=["qx"])

    ordered, pre_err = _topo_sort_dependent_tasks([q1, q2])
    assert [t.id for t in ordered] == []
    assert "q2" in pre_err
    assert pre_err["q2"].startswith("missing_dependency:")


def test_topo_sort_cycle_dependency_marks_error():
    q2 = _qt("q2", deps=["q3"])
    q3 = _qt("q3", deps=["q2"])

    ordered, pre_err = _topo_sort_dependent_tasks([q2, q3])
    assert [t.id for t in ordered] == []
    assert pre_err["q2"].startswith("cycle_dependency_or_downstream:")
    assert pre_err["q3"].startswith("cycle_dependency_or_downstream:")


def test_topo_sort_cycle_and_downstream_all_marked_as_cycle_or_downstream():
    # q2 <-> q3 成环，q4 依赖 q2（环下游）
    q2 = _qt("q2", deps=["q3"])
    q3 = _qt("q3", deps=["q2"])
    q4 = _qt("q4", deps=["q2"])

    ordered, pre_err = _topo_sort_dependent_tasks([q2, q3, q4])
    assert [t.id for t in ordered] == []
    assert set(pre_err.keys()) == {"q2", "q3", "q4"}
    assert pre_err["q2"].startswith("cycle_dependency_or_downstream:")
    assert pre_err["q3"].startswith("cycle_dependency_or_downstream:")
    assert pre_err["q4"].startswith("cycle_dependency_or_downstream:")


def test_topo_sort_complex_valid_dependencies_with_branching():
    # q1 独立；q2/q3 依赖 q1；q4 依赖 q2+q3；q5 依赖 q3
    q1 = _qt("q1", deps=None)
    q2 = _qt("q2", deps=["q1"])
    q3 = _qt("q3", deps=["q1"])
    q4 = _qt("q4", deps=["q2", "q3"])
    q5 = _qt("q5", deps=["q3"])

    ordered, pre_err = _topo_sort_dependent_tasks([q1, q2, q3, q4, q5])
    ordered_ids = [t.id for t in ordered]
    assert pre_err == {}
    assert set(ordered_ids) == {"q2", "q3", "q4", "q5"}

    pos = {qid: i for i, qid in enumerate(ordered_ids)}
    # 依赖约束：q4 在 q2/q3 之后，q5 在 q3 之后
    assert pos["q2"] < pos["q4"]
    assert pos["q3"] < pos["q4"]
    assert pos["q3"] < pos["q5"]


def test_topo_sort_missing_node_not_causing_false_cycle_for_dependent():
    # q2 缺失依赖，q3 依赖 q2；当前策略应忽略 q3->q2 这条边，不把 q3 误判为环
    q2 = _qt("q2", deps=["qx"])   # missing
    q3 = _qt("q3", deps=["q2"])   # 依赖 missing 节点

    ordered, pre_err = _topo_sort_dependent_tasks([q2, q3])
    assert [t.id for t in ordered] == ["q3"]
    assert set(pre_err.keys()) == {"q2"}
    assert pre_err["q2"].startswith("missing_dependency:")

