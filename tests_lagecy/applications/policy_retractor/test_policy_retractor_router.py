from __future__ import annotations

import json

from fastapi import BackgroundTasks

from src.agents.policy.policy_retractor.schemas import (
    Control,
    ControlType,
    Framework,
    Requirement,
    RiskLevel,
    SourceLocation,
)
from src.applications.policy_retractor.routers.policy_retractor_router import (
    DEFAULT_CALLBACK_BASE_URL,
    DEFAULT_CALLBACK_PATH,
    RuleExtractSubmitRequest,
    _build_inline_payload,
    _build_java_event_response,
    _resolve_callback_url,
    _serialize_requirement_payload,
    _serialize_framework_for_java,
    submit_rule_extract_job,
)


def test_build_inline_payload_from_java_request() -> None:
    request = RuleExtractSubmitRequest(
        fileId="FILE-001",
        docContent=[
            {
                "page": 1,
                "type": "text",
                "value": "测试正文",
                "md": "测试正文",
                "element_id": 1,
                "file_id": "FILE-001",
            }
        ],
    )

    payload = _build_inline_payload(request)

    assert payload is not None
    assert payload["fileId"] == "FILE-001"
    assert len(payload["docContent"]) == 1


def test_submit_rule_extract_job_accepts_java_payload() -> None:
    request = RuleExtractSubmitRequest(
        fileId="FILE-002",
        docContent=[
            {
                "page": 1,
                "type": "text",
                "value": "测试正文",
                "md": "测试正文",
                "element_id": 1,
                "file_id": "FILE-002",
            }
        ],
    )

    result = submit_rule_extract_job(request, BackgroundTasks())
    payload = json.loads(result["data"])

    assert result["jobId"]
    assert result["frameworkId"] == "FW-FILE-002"
    assert result["fileId"] == "FILE-002"
    assert result["event"] == "start"
    assert payload == {}


def test_serialize_framework_for_java_uses_string_element_ids() -> None:
    framework = Framework(
        framework_id="FW-FILE-003",
        file_id="FILE-003",
        name="测试框架",
        description="测试描述",
        requirements=[
            Requirement(
                requirement_id="REQ-1",
                framework_id="FW-FILE-003",
                title="测试要求",
                description="测试要求描述",
                source_text="测试要求原文",
                source_location=SourceLocation(file_id="FILE-003", element_ids=[1, 2], page_no=7),
                controls=[
                    Control(
                        control_id="RULE-1",
                        requirement_id="REQ-1",
                        content="测试控制",
                        control_type=ControlType.DOCUMENT_CHECK,
                        risk=RiskLevel.MEDIUM,
                        source_text="测试控制原文",
                        logic_description="测试控制逻辑",
                        source_location=SourceLocation(file_id="FILE-003", element_ids=[3, 4], page_no=8),
                    )
                ],
            )
        ],
    )

    payload = _serialize_framework_for_java(framework)

    assert payload["requirements"][0]["sourceLocation"]["elementIds"] == "1,2"
    assert payload["requirements"][0]["sourceLocation"]["pageNo"] == 7
    assert payload["requirements"][0]["controls"][0]["sourceLocation"]["elementIds"] == "3,4"
    assert payload["requirements"][0]["controls"][0]["sourceLocation"]["pageNo"] == 8


def test_build_java_event_response_wraps_requirement_tree() -> None:
    result = _build_java_event_response(
        framework_id="FW-FILE-004",
        file_id="FILE-004",
        event="complete",
        data={"requirements": [{"requirementId": "REQ-1"}]},
    )
    payload = json.loads(result["data"])

    assert result["frameworkId"] == "FW-FILE-004"
    assert result["fileId"] == "FILE-004"
    assert result["event"] == "complete"
    assert payload["requirements"][0]["requirementId"] == "REQ-1"


def test_serialize_requirement_payload_returns_single_requirement_tree() -> None:
    requirement = Requirement(
        requirement_id="REQ-2",
        framework_id="FW-FILE-005",
        title="测试要求",
        description="测试要求描述",
        source_text="测试要求原文",
        source_location=SourceLocation(file_id="FILE-005", element_ids=[9]),
        controls=[],
    )

    payload = _serialize_requirement_payload("FW-FILE-005", "FILE-005", requirement)

    assert payload["frameworkId"] == "FW-FILE-005"
    assert payload["fileId"] == "FILE-005"
    assert len(payload["requirements"]) == 1
    assert payload["requirements"][0]["requirementId"] == "REQ-2"


def test_resolve_callback_url_prefers_request_value() -> None:
    request = RuleExtractSubmitRequest(callbackUrl="http://127.0.0.1:8080/api/policy/doc/modelCallback")
    assert _resolve_callback_url(request) == "http://127.0.0.1:8080/api/policy/doc/modelCallback"


def test_resolve_callback_url_falls_back_to_default_base_url(monkeypatch) -> None:
    monkeypatch.delenv("POLICY_CALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("CONTENT_GENERATE_CALLBACK_URL", raising=False)

    request = RuleExtractSubmitRequest()

    assert _resolve_callback_url(request) == f"{DEFAULT_CALLBACK_BASE_URL}{DEFAULT_CALLBACK_PATH}"


def test_default_callback_path_constant() -> None:
    assert DEFAULT_CALLBACK_PATH == "/api/policy/doc/modelcallback"
