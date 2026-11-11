from typing import Any, Dict, List


def format_to_hierarchical(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将带有 headingLevel 的扁平化 JSON 列表转换为层次化结构。
    """
    hierarchical_content: List[Dict[str, Any]] = []
    path_stack: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {'S': 0, 'T': 0, 'TBL': 0}

    for item in items:
        item_type = item.get("type")
        heading_level = item.get("headingLevel")

        # 1) 处理标题
        if item_type == "text" and heading_level is not None:
            new_section: Dict[str, Any] = {
                "section_title": (item.get("value", "") or item.get("md", "")).strip(),
                "page_number": item.get("page"),
                "bounding_box": item.get("bBox"),
                "text_content": [],
                "table_data": [],
                "sub_sections": []
            }

            if not path_stack or heading_level == 1:
                path_stack = [new_section]
                hierarchical_content.append(new_section)
                if isinstance(item.get("section_id"), str) and item.get("section_id").strip():
                    new_section["section_id"] = item.get("section_id")
                else:
                    counters['S'] += 1
                    new_section["section_id"] = f"S{counters['S']}"
            else:
                # 回溯到正确父级
                while path_stack and heading_level <= path_stack[-1].get('_level', 0):
                    path_stack.pop()

                if not path_stack:
                    path_stack = [new_section]
                    hierarchical_content.append(new_section)
                    if isinstance(item.get("section_id"), str) and item.get("section_id").strip():
                        new_section["section_id"] = item.get("section_id")
                    else:
                        counters['S'] += 1
                        new_section["section_id"] = f"S{counters['S']}"
                else:
                    parent = path_stack[-1]
                    parent.setdefault("sub_sections", []).append(new_section)
                    path_stack.append(new_section)
                    if isinstance(item.get("section_id"), str) and item.get("section_id").strip():
                        new_section["section_id"] = item.get("section_id")
                    else:
                        parent_id = parent.get("section_id", "S")
                        child_count = len(parent.get("sub_sections", []))
                        new_section["section_id"] = f"{parent_id}_{child_count}"

            new_section['_level'] = heading_level

        # 2) 非标题内容
        elif path_stack:
            current_section = path_stack[-1]

            if item_type == "text":
                text_node: Dict[str, Any] = {
                    "text_id": item.get("text_id") if isinstance(item.get("text_id"), str) and item.get("text_id").strip() else None,
                    "value": (item.get("value", "") or item.get("md", "")).strip(),
                    "page_number": item.get("page"),
                    "bounding_box": item.get("bBox")
                }
                if text_node["text_id"] is None:
                    counters['T'] += 1
                    text_node["text_id"] = f"T{counters['T']}"
                current_section.setdefault("text_content", []).append(text_node)

            elif item_type == "table":
                table_node: Dict[str, Any] = {
                    "table_id": item.get("table_id") if isinstance(item.get("table_id"), str) and item.get("table_id").strip() else None,
                    "page_number": item.get("page"),
                    "bounding_box": item.get("bBox"),
                    "html": item.get("html"),
                    "md": item.get("md"),
                    "rows": item.get("rows"),
                    "table_type": item.get("table_type")
                }
                if table_node["table_id"] is None:
                    counters['TBL'] += 1
                    table_node["table_id"] = f"TBL{counters['TBL']}"
                current_section.setdefault("table_data", []).append(table_node)

    # 清理临时键
    def _clean(node: Dict[str, Any]) -> None:
        if '_level' in node:
            del node['_level']
        for sub in node.get("sub_sections", []):
            _clean(sub)

    for section in hierarchical_content:
        _clean(section)

    return {"content": hierarchical_content}


