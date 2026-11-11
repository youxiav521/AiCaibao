from src.agents.policy.policy_check_agent.agents.tools import (
    ResourceRepositoryContext,
    _finalize_check_payload,
)
from src.agents.policy.policy_check_agent.schemas import ResourceRepository


def _build_single_file_repository() -> ResourceRepository:
    return ResourceRepository.model_validate(
        {
            "id": "repo_single_file",
            "title": "single file repo",
            "description": "repository for policy check tool tests",
            "files": [
                {
                    "fileName": "营业执照.png",
                    "fileId": "file_license_001",
                    "content": [
                        {
                            "page": 1,
                            "type": "text",
                            "bBox": {"l": 0, "t": 0, "r": 100, "b": 40, "coord_origin": "TOPLEFT"},
                            "value": "营业执照信息清晰可见",
                            "md": "营业执照信息清晰可见",
                            "file_id": "file_license_001",
                            "element_id": 1,
                        }
                    ],
                }
            ],
        }
    )


def test_finalize_check_payload_keeps_pass_and_builds_single_file_evidence() -> None:
    repo = _build_single_file_repository()
    with ResourceRepositoryContext(repo):
        payload = _finalize_check_payload(
            payload_type="control_result",
            result="pass",
            reason="已核验《营业执照.png》符合要求。",
            evidence=None,
        )

    assert payload["result"] == "pass"
    assert len(payload["evidence"]) == 1
    assert payload["evidence"][0]["fileId"] == "file_license_001"
    assert "Validation notes:" not in payload["reason"]


def test_finalize_check_payload_infers_evidence_from_reason_without_downgrading_pass() -> None:
    repo = _build_single_file_repository()
    with ResourceRepositoryContext(repo):
        payload = _finalize_check_payload(
            payload_type="requirement_result",
            result="pass",
            reason="已核验营业执照.png符合要求，主体信息一致。",
            evidence=[],
        )

    assert payload["result"] == "pass"
    assert payload["evidence"]
    assert payload["evidence"][0]["fileId"] == "file_license_001"
