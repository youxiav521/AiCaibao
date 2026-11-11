from __future__ import annotations

from src.applications.content_generate.services.log_redaction import redact_data_uris_for_log


def test_redacts_nested_image_data_uri_without_changing_normal_text() -> None:
    data_uri = "data:image/svg+xml;base64," + "a" * 64
    value = {
        "questionId": "q-image-1",
        "answer": {"answerText": data_uri},
        "message": "normal text",
    }

    redacted = redact_data_uris_for_log(value)

    assert redacted["answer"]["answerText"]["redacted"] is True
    assert redacted["answer"]["answerText"]["mime"] == "image/svg+xml"
    assert redacted["message"] == "normal text"
    assert data_uri not in str(redacted)
