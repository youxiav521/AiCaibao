import argparse
import json
from pathlib import Path

from src.agents.policy.policy_check_agent.material_layer.service import MaterialUnderstandingService
from src.agents.policy.policy_check_agent.schemas import ResourceRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Build material understanding index from resource repository JSON.")
    parser.add_argument("--resource-repository", type=Path, required=True)
    args = parser.parse_args()

    resource_data = json.loads(args.resource_repository.read_text(encoding="utf-8"))
    if isinstance(resource_data, list):
        resource_data = {
            "id": "resource_repository_local",
            "title": args.resource_repository.stem,
            "description": "Auto wrapped from file list JSON",
            "files": resource_data,
        }

    resource_repository = ResourceRepository.model_validate(resource_data)
    result = MaterialUnderstandingService().build(resource_repository)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
