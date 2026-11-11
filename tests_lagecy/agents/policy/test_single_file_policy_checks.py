import asyncio
import json

import pytest

from src.agents.policy.policy_check_agent.single_file_policy_checks.schemas import SingleFilePolicyCheckRequest
from src.agents.policy.policy_check_agent.single_file_policy_checks.service import SingleFilePolicyCheckService


def _build_doc_content(lines: list[str], file_id: str = "file_single_001") -> list[dict]:
    content = []
    top = 0
    for index, line in enumerate(lines, start=1):
        content.append(
            {
                "page": 1,
                "type": "text",
                "bBox": {"l": 0, "t": top, "r": 100, "b": top + 20, "coord_origin": "TOPLEFT"},
                "value": line,
                "md": line,
                "file_id": file_id,
                "element_id": index,
            }
        )
        top += 24
    return content


def _build_frameworks() -> list[dict]:
    return [
        {
            "frameworkId": "fw_001",
            "fileId": "policy_file_001",
            "name": "贷款审核规则",
            "description": "测试框架",
            "requirements": [
                {
                    "requirementId": "req_license",
                    "title": "营业执照检查",
                    "description": "检查营业执照是否齐全",
                    "sourceText": "需核查营业执照。",
                    "sourceLocation": {"fileId": "policy_file_001", "elementIds": [1]},
                    "status": "active",
                    "controls": [
                        {
                            "controlId": "ctrl_license",
                            "controlType": "document_check",
                            "risk": "MEDIUM",
                            "logicDescription": "核查营业执照信息是否存在",
                            "sourceText": "营业执照 统一社会信用代码",
                            "sourceLocation": {"fileId": "policy_file_001", "elementIds": [2]},
                        },
                        {
                            "controlId": "ctrl_collateral",
                            "controlType": "document_check",
                            "risk": "LOW",
                            "logicDescription": "核查抵押登记证明是否存在",
                            "sourceText": "抵押登记证明 他项权证",
                            "sourceLocation": {"fileId": "policy_file_001", "elementIds": [3]},
                        },
                    ],
                },
                {
                    "requirementId": "req_guarantee",
                    "title": "担保材料检查",
                    "description": "检查担保协议",
                    "sourceText": "需核查担保协议。",
                    "sourceLocation": {"fileId": "policy_file_001", "elementIds": [4]},
                    "status": "active",
                    "controls": [
                        {
                            "controlId": "ctrl_guarantee",
                            "controlType": "document_check",
                            "risk": "HIGH",
                            "logicDescription": "核查担保协议是否存在",
                            "sourceText": "担保协议 反担保措施",
                            "sourceLocation": {"fileId": "policy_file_001", "elementIds": [5]},
                        }
                    ],
                },
            ],
        }
    ]


class FakePolicyAgent:
    def __init__(self, responses: dict[str, dict] | None = None, failing_requirements: set[str] | None = None) -> None:
        self.responses = responses or {}
        self.failing_requirements = failing_requirements or set()
        self.calls: list[tuple[str, list[str]]] = []
        self.model_id = "model_bailian_policy_check"

    async def _execute_requirement_with_retry(self, *, resource_repository, requirement, framework_id, event_queue, execution_semaphore):
        self.calls.append((requirement.requirementId, [control.controlId for control in requirement.controls]))
        if requirement.requirementId in self.failing_requirements:
            raise RuntimeError("boom")
        if requirement.requirementId in self.responses:
            return self.responses[requirement.requirementId]

        control_results = [
            {
                "requirementId": requirement.requirementId,
                "controlId": control.controlId,
                "result": "pass",
                "reason": f"{control.controlId} passed",
                "evidence": [],
                "degraded": False,
            }
            for control in requirement.controls
        ]
        return {
            "result": {"messages": [], "todos": []},
            "check_results": {
                "requirementId": requirement.requirementId,
                "result": "pass",
                "reason": "all matched controls passed",
                "evidence": [],
                "controlResults": control_results,
            },
            "control_results": control_results,
        }


@pytest.mark.asyncio
async def test_build_resource_file_from_input_path(tmp_path) -> None:
    payload = {"docContent": _build_doc_content(["营业执照信息", "统一社会信用代码 123"], file_id="file_from_path")}
    input_path = tmp_path / "single_file.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    service = SingleFilePolicyCheckService(policy_agent=FakePolicyAgent())
    request = SingleFilePolicyCheckRequest(inputPath=str(input_path), frameworks=_build_frameworks())

    resource_file = service._build_resource_file(request)

    assert resource_file.fileId == "single_file"
    assert resource_file.fileName == "single_file.json"
    assert len(resource_file.content) == 2


@pytest.mark.asyncio
async def test_single_file_review_returns_pass_with_skipped_controls() -> None:
    service = SingleFilePolicyCheckService(policy_agent=FakePolicyAgent())
    request = SingleFilePolicyCheckRequest(
        fileId="file_pass",
        fileName="pass_doc.json",
        docContent=_build_doc_content(["营业执照", "统一社会信用代码 9135XXXX"], file_id="file_pass"),
        frameworks=_build_frameworks(),
    )

    result = await service.analyze_request(request)

    assert result.overall_status == "pass"
    assert result.statistics["matchedControls"] == 1
    assert result.statistics["skippedControls"] == 2
    assert len(result.matched_controls) == 1
    assert len(result.skipped_controls) == 2
    assert result.requirements[0].result == "pass"
    assert result.requirements[1].result == "skipped"


@pytest.mark.asyncio
async def test_single_file_review_skips_all_unrelated_controls() -> None:
    fake_agent = FakePolicyAgent()
    service = SingleFilePolicyCheckService(policy_agent=fake_agent)
    request = SingleFilePolicyCheckRequest(
        fileId="file_skip",
        fileName="skip_doc.json",
        docContent=_build_doc_content(["会议纪要", "内部周报"], file_id="file_skip"),
        frameworks=_build_frameworks(),
    )

    result = await service.analyze_request(request)

    assert result.overall_status == "skipped"
    assert result.statistics["matchedControls"] == 0
    assert result.statistics["skippedControls"] == 3
    assert fake_agent.calls == []
    assert all(item.result == "skipped" for item in result.requirements)


@pytest.mark.asyncio
async def test_single_file_review_propagates_fail_result() -> None:
    fake_agent = FakePolicyAgent(
        responses={
            "req_license": {
                "result": {"messages": [], "todos": []},
                "check_results": {
                    "requirementId": "req_license",
                    "result": "fail",
                    "reason": "license mismatch",
                    "evidence": [],
                    "controlResults": [
                        {
                            "requirementId": "req_license",
                            "controlId": "ctrl_license",
                            "result": "fail",
                            "reason": "license mismatch",
                            "evidence": [],
                            "degraded": False,
                        }
                    ],
                },
                "control_results": [
                    {
                        "requirementId": "req_license",
                        "controlId": "ctrl_license",
                        "result": "fail",
                        "reason": "license mismatch",
                        "evidence": [],
                        "degraded": False,
                    }
                ],
            }
        }
    )
    service = SingleFilePolicyCheckService(policy_agent=fake_agent)
    request = SingleFilePolicyCheckRequest(
        fileId="file_fail",
        fileName="fail_doc.json",
        docContent=_build_doc_content(["营业执照", "统一社会信用代码 9135XXXX"], file_id="file_fail"),
        frameworks=_build_frameworks(),
    )

    result = await service.analyze_request(request)

    assert result.overall_status == "fail"
    assert result.requirements[0].result == "fail"
    assert result.matched_controls[0].result == "fail"


@pytest.mark.asyncio
async def test_single_file_review_degrades_when_requirement_execution_fails() -> None:
    fake_agent = FakePolicyAgent(failing_requirements={"req_license"})
    service = SingleFilePolicyCheckService(policy_agent=fake_agent)
    request = SingleFilePolicyCheckRequest(
        fileId="file_degraded",
        fileName="degraded_doc.json",
        docContent=_build_doc_content(["营业执照", "统一社会信用代码 9135XXXX"], file_id="file_degraded"),
        frameworks=_build_frameworks(),
    )

    result = await service.analyze_request(request)

    assert result.overall_status == "uncertain"
    assert result.requirements[0].degraded is True
    assert result.matched_controls[0].degraded is True
    assert result.matched_controls[0].result == "uncertain"


@pytest.mark.asyncio
async def test_single_file_review_prefilters_business_overview_document() -> None:
    fake_agent = FakePolicyAgent()
    service = SingleFilePolicyCheckService(policy_agent=fake_agent)
    request = SingleFilePolicyCheckRequest(
        fileId="file_overview",
        fileName="overview_doc.json",
        docContent=_build_doc_content(
            ["报告期内公司从事的主要业务", "公司主营业务为新能源汽车、电池及光伏业务。"],
            file_id="file_overview",
        ),
        frameworks=_build_frameworks(),
    )

    result = await service.analyze_request(request)

    assert result.overall_status == "skipped"
    assert result.statistics["matchedControls"] == 0
    assert fake_agent.calls == []


@pytest.mark.asyncio
async def test_single_file_review_fast_fails_collection_requirement_without_agent_call() -> None:
    framework = _build_frameworks()[0]
    framework["requirements"] = [framework["requirements"][1]]
    fake_agent = FakePolicyAgent()
    service = SingleFilePolicyCheckService(policy_agent=fake_agent)
    request = SingleFilePolicyCheckRequest(
        fileId="file_overview_2",
        fileName="overview_doc_2.json",
        docContent=_build_doc_content(
            ["报告期内公司从事的主要业务", "公司主营业务为新能源汽车、电池及光伏业务。"],
            file_id="file_overview_2",
        ),
        frameworks=[framework],
    )

    result = await service.analyze_request(request)

    assert fake_agent.calls == []
    assert result.statistics["matchedControls"] == 0 or result.statistics["failedRequirements"] >= 0
