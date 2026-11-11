import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.policy.policy_check_agent.agents.agent import run_policy_check
from src.agents.policy.policy_check_agent.schemas import Framework, ResourceRepository


def _build_sample_payload() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    resource_repository = {
        "id": "repo_local_test_001",
        "title": "本地流程测试材料库",
        "description": "用于验证 policy_check_agent 主流程是否可以跑通。",
        "files": [
            {
                "fileName": "信息安全管理制度.pdf",
                "fileId": "file_local_001",
                "content": [
                    {
                        "page": 1,
                        "type": "text",
                        "bBox": {"l": 0, "t": 0, "r": 100, "b": 40, "coord_origin": "TOPLEFT"},
                        "value": "公司建立了信息安全管理制度。",
                        "md": "公司建立了信息安全管理制度。",
                        "file_id": "file_local_001",
                        "element_id": 1,
                    },
                    {
                        "page": 1,
                        "type": "text",
                        "bBox": {"l": 0, "t": 45, "r": 100, "b": 85, "coord_origin": "TOPLEFT"},
                        "value": "制度中明确了访问权限申请、审批与定期复核要求。",
                        "md": "制度中明确了访问权限申请、审批与定期复核要求。",
                        "file_id": "file_local_001",
                        "element_id": 2,
                    },
                ],
            }
        ],
    }

    frameworks = [
        {
            "frameworkId": "framework_local_001",
            "fileId": "framework_source_001",
            "name": "本地测试规则框架",
            "description": "最小化 requirement/control 样例。",
            "requirements": [
                {
                    "requirementId": "REQ_LOCAL_001",
                    "title": "应建立信息安全管理制度",
                    "description": "检查材料库中是否存在信息安全制度，并明确访问控制要求。",
                    "sourceText": "机构应建立信息安全管理制度，明确访问控制要求。",
                    "sourceLocation": {
                        "fileId": "framework_source_001",
                        "elementIds": [1],
                    },
                    "status": "active",
                    "controls": [
                        {
                            "controlId": "CTRL_LOCAL_001",
                            "controlType": "document",
                            "risk": "high",
                            "logicDescription": "检查文件中是否存在信息安全管理制度，并包含访问权限申请、审批或复核要求。",
                            "sourceText": "应建立制度并明确访问控制。",
                            "sourceLocation": {
                                "fileId": "framework_source_001",
                                "elementIds": [2],
                            },
                        }
                    ],
                }
            ],
        }
    ]

    return resource_repository, frameworks


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_payload(
    resource_repository_path: Path | None,
    frameworks_path: Path | None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if resource_repository_path is None and frameworks_path is None:
        return _build_sample_payload()

    if resource_repository_path is None or frameworks_path is None:
        raise ValueError("`--resource-repository` 和 `--frameworks` 必须同时提供。")

    return _load_json(resource_repository_path), _load_json(frameworks_path)


async def _run(args: argparse.Namespace) -> None:
    resource_repository_data, frameworks_data = _load_payload(
        args.resource_repository,
        args.frameworks,
    )

    resource_repository = ResourceRepository.model_validate(resource_repository_data)
    frameworks = [Framework.model_validate(item) for item in frameworks_data]

    print(
        json.dumps(
            {
                "mode": "sample" if args.resource_repository is None else "json_files",
                "model_id": args.model_id,
                "max_concurrency": args.max_concurrency,
                "file_count": len(resource_repository.files),
                "framework_count": len(frameworks),
                "requirement_count": sum(len(framework.requirements) for framework in frameworks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    async for result in run_policy_check(
        resource_repository=resource_repository,
        frameworks=frameworks,
        model_id=args.model_id,
        max_concurrency=args.max_concurrency,
    ):
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="本地运行 policy_check_agent 流程测试。默认使用内置最小样例。",
    )
    parser.add_argument(
        "--resource-repository",
        type=Path,
        default=None,
        help="资源库 JSON 文件路径。",
    )
    parser.add_argument(
        "--frameworks",
        type=Path,
        default=None,
        help="frameworks JSON 文件路径。",
    )
    parser.add_argument(
        "--model-id",
        default="model_bailian_policy_check",
        help="模型 ID，默认使用 model_bailian_policy_check。",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="并发数，默认 1，便于本地排查问题。",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
