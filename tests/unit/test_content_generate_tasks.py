from src.agents.content_generate_agent.models import QuestionTask
from src.applications.content_generate.tasks.content_generate_tasks import (
    _allows_empty_content,
)


def _question(question_type: str) -> QuestionTask:
    return QuestionTask(
        id=f"q-{question_type}",
        description="test question",
        type=question_type,
    )


def test_allows_empty_content_only_for_image_tasks():
    """纯图片任务依赖 fileIds 检索，不应被空 contentList 拦截。"""
    assert _allows_empty_content([_question("image")]) is True
    assert _allows_empty_content([_question("image"), _question("image")]) is True
    assert _allows_empty_content([_question("text")]) is False
    assert _allows_empty_content([_question("image"), _question("table")]) is False
    assert _allows_empty_content([]) is False
