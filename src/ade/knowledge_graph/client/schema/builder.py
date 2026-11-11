"""Schema 动态生成器。

从 definitions.py 的单一数据源派生 TABLE、LIST、MAP 等视图。
"""

from src.ade.knowledge_graph.client.schema.definitions import (
    ENTITY_TYPES,
    RELATION_TYPES,
)

# 分隔符（与 prompt.py 保持一致，此处仅用于表格生成）
_SOURCE_TYPE_SEP = " / "

_PROJECT_NAME_PLACEHOLDER = "{project_name}"


class SchemaBuilder:
    """从类型定义生成 TABLE / LIST / MAP 等视图。"""

    # ==================== 实体类型 ====================

    @staticmethod
    def entity_type_list() -> list[str]:
        return [e.name for e in ENTITY_TYPES]

    @staticmethod
    def entity_type_map() -> dict[str, str]:
        mapping = {e.name: e.name_en for e in ENTITY_TYPES}
        mapping["项目"] = "PROJECT"  # 图谱中人工注入的项目实体，不来自LLM抽取
        mapping["未知"] = "UNKNOWN"  # 关系端点未解析时人工补全的存根实体
        return mapping

    @staticmethod
    def entity_types_table() -> str:
        header = "|序号|实体类型|属性（可选，描述性）|说明|"
        sep = "|--|----|----|--|"
        rows = [
            f"|{e.id}|{e.name}|{'、'.join(e.attributes) if e.attributes else ''}|{e.description}|"
            for e in ENTITY_TYPES
        ]
        return "\n".join([header, sep, *rows])

    # ==================== 关系类型 ====================

    @staticmethod
    def relation_type_list() -> list[str]:
        return [r.name for r in RELATION_TYPES]

    @staticmethod
    def relation_type_map() -> dict[str, str]:
        return {r.name: r.name_en for r in RELATION_TYPES}

    @staticmethod
    def relation_types_table() -> str:
        header = "|序号|关系类型|源实体类型|目标实体类型|属性|关系描述|"
        sep = "|--|----|------------|------------|----|---------------------------|"
        rows = [
            f"|{r.id}|{r.name}|{_SOURCE_TYPE_SEP.join(r.source_types)}|{_SOURCE_TYPE_SEP.join(r.target_types)}|{'、'.join(r.attributes) if r.attributes else ''}|{r.description}|"
            for r in RELATION_TYPES
        ]
        return "\n".join([header, sep, *rows])

    @staticmethod
    def project_relation_note(
        project_name: str = _PROJECT_NAME_PLACEHOLDER,
    ) -> str:
        """生成项目关系的注意事项。"""
        count = sum(1 for r in RELATION_TYPES if "项目" in r.target_types)
        return (
            f"上述第 1-{count} 项的目标实体为本次调研项目本身，"
            f'请使用该项目的真实名称 "{project_name}" 作为目标实体名称，不要使用"项目"二字。'
        )
