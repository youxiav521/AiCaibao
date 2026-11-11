from __future__ import annotations

from typing import Any, Dict, List
from .base import BaseSectionExtractor


class BaseInfoExtractor(BaseSectionExtractor):
    @classmethod
    def section_key(cls) -> str:
        return "base_info"

    @classmethod
    def recall_keywords(cls) -> List[str]:
        return [
            "公司简介",
            "基本情况",
            "公司信息",
            "公司概况",
            "发行人基本情况",
            "年度报告",
            "半年度报告",
            "第一季度报告",
            "第三季度报告",
        ]

    @classmethod
    def recall_indices(cls, items: List[Dict[str, Any]]) -> List[int]:
        from .base import _recall_by_keywords
        front_indices = list(range(0, min(len(items), 150)))
        indices = _recall_by_keywords(items, cls.recall_keywords(), max_hits=8, max_per_hit=120)
        merged = list(dict.fromkeys(indices + front_indices))
        return merged[:800]
    def title(self) -> str:
        return "基础信息（企业/代码/简称/期数）"

    def schema_hint(self) -> Dict[str, Any]:
        return {
            "企业全称": {"value": None, "source_element_id": None},
            "证券代码": {
                "A股": {"value": None, "source_element_id": None},
                "港股": {"value": None, "source_element_id": None},
                "美股": {"value": None, "source_element_id": None},
            },
            "交易简称": {
                "A股": {"value": None, "source_element_id": None},
                "港股": {"value": None, "source_element_id": None},
            },
            "当前期数": {"value": None, "source_element_id": None},
        }

    def extra_rules(self) -> str:
        return (
        "具体指令：\n"
        "- **信息优先级**: 优先从文档中明确标注的表格（如'公司信息'、'基本信息'、'企业概况'）提取信息，若相同字段在不同表格出现冲突，以'公司信息'表格中的对应字段为准。\n"
        "- **企业全称**: 从'公司全称'字段或文档中带有'法定名称'、'注册名称'标签的要素中提取信息，若存在字段值包含'（股份公司）'、'（集团）'等限定词，需完整保留并优先采用。\n"
        "- **当前期数**: 在文档的封面页、目录页或报告标题栏寻找包含'年度报告'、'季度报告'、'半年度报告'等时间性质描述的字段。**必须提取完整的期数描述**（如'2024年度报告'、'2024年第三季度报告'），而不是只提取年份。若文档中只有年份没有报告类型，则提取格式应为'YYYY年度报告'（默认为年度报告）。\n"
        "- **证券代码/简称**: \n"
        "  - **精确定位**: 在'公司信息'、'基本情况'章节中的表格中定位'股票代码'和'股票简称'字段，若字段值包含'（代码）'、'（简称）'等说明性文字，需分别提取并标注。\n"
        "  - **市场判断 (A股/港股)**: 在'上市交易所'字段或文档中包含'深交所'、'上交所'、'港交所'等关键词的要素中识别市场类型。若无明确字段，通过股票代码前缀判断：A股代码以6开头（如600001），港股代码以H开头（如000001.HK）。若代码前缀不匹配任何市场规则，对应值保持 `null`。\n"
        "- **精确溯源**: `source_element_id` 必须关联到文档中与该信息直接对应的表格行标识符或段落编号，若字段值存在于多个元素中，以包含完整数据的最小元素ID为准。"
        )

    def parse(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "企业全称": raw.get("企业全称", {"value": None, "source_element_id": None}),
            "证券代码": {
                "A股": raw.get("证券代码", {}).get("A股", {"value": None, "source_element_id": None}),
                "港股": raw.get("证券代码", {}).get("港股", {"value": None, "source_element_id": None}),
                "美股": raw.get("证券代码", {}).get("美股", {"value": None, "source_element_id": None}),
            },
            "交易简称": {
                "A股": raw.get("交易简称", {}).get("A股", {"value": None, "source_element_id": None}),
                "港股": raw.get("交易简称", {}).get("港股", {"value": None, "source_element_id": None}),
            },
            "当前期数": raw.get("当前期数", {"value": None, "source_element_id": None}),
        }


