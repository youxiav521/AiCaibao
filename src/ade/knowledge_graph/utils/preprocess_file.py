"""将 converted.json 预处理为 FileInput 的工具函数"""
from pathlib import Path
from typing import List

from src.ade.knowledge_graph.schema.kg_schema import FileInput, ElementMeta
from src.utils.json_utils import load


def converted_json_to_fileinput(
    json_path: str | Path,
    file_id: str = "",
    file_name: str = "",
) -> FileInput:
    """将一个 converted.json 转为 FileInput（按 page 分组，md 内容拼接）。

    Args:
        json_path: converted.json 的文件路径
        file_id: 文件 ID（默认取文件上级目录名）
        file_name: 文件名称（默认取文件上级目录名）

    Returns:
        FileInput: 可直接喂给 KnowledgeGraphService.generate()
    """
    json_path = Path(json_path)

    if not file_name:
        file_name = json_path.parent.name
    if not file_id:
        file_id = file_name

    items = load(json_path)

    page_groups: dict[int, list[dict]] = {}
    for item in items:
        page = item.get("page", 0)
        page_groups.setdefault(page, []).append(item)

    elements: list[ElementMeta] = []
    for page_no in sorted(page_groups.keys()):
        group = page_groups[page_no]
        for item in group:
            md = item.get("md") or ""
            elem_type = item.get("type", "text")
            # 构建 ElementMeta 的参数字典
            elem_kwargs = {
                "element_id": item["element_id"],
                "type": elem_type,
                "md": md,
                "page": page_no,
            }
            # 只有当 type 是 table 时，才添加 rows 字段
            if elem_type == "table":
                elem_kwargs["rows"] = item.get("rows")
            elements.append(ElementMeta(**elem_kwargs))

    return FileInput(id=file_id, name=file_name, elements=elements)


def build_file_inputs_from_folder(converted_folder: str | Path) -> list[FileInput]:
    """扫描 converted_folder 下的子目录，将每个 converted.json 转为 FileInput。

    文件夹结构约定：
        converted_folder/
        ├── 文档A/
        │   └── converted.json
        ├── 文档B/
        │   └── converted.json
        └── ...

    Args:
        converted_folder: 存放 converted.json 的文件夹路径

    Returns:
        list[FileInput]: 所有文件的 FileInput 列表
    """
    converted_folder = Path(converted_folder)
    files: list[FileInput] = []

    for sub_dir in sorted(converted_folder.iterdir()):
        if not sub_dir.is_dir():
            continue
        json_path = sub_dir / "converted.json"
        if not json_path.exists():
            continue
        file_input = converted_json_to_fileinput(json_path)
        files.append(file_input)

    return files
