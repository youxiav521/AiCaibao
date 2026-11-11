"""内容生成 E2E 测试题目样本（来源于 notebook 与现有题库，均为真实题目）。"""

from __future__ import annotations

from typing import Any

# 来自 storage/question_tasks_poc1226.json（text，不带自定义 data_sources/tools）
QUESTION_TEXT_PLAIN: dict[str, Any] = {
    "id": "q_5",
    "description": "分行初审意见及联系方式",
    "type": "text",
    "html": "",
    "contents": "",
}

# 来自 notebook 手动 question_tasks（text，带 custom data_sources + tools）
QUESTION_TEXT_CUSTOM: dict[str, Any] = {
    "id": "q_2",
    "description": "分行初审意见及身份证号",
    "prompt": "",
    "type": "text",
    "html": "",
    "data_sources": [
        {
            "type": "chunk_index_retrieval",
            "query": "分行初审意见",
            "top_k": 30,
            "return_child_chunk": True,
            "rerank_enabled": False,
            "hybrid_enabled": True,
            "search_table": False,
        },
        {
            "type": "extract",
            "data": [
                {
                "tagName": "身份证",
                "files": [
                    {
                    "fileId": "..",
                    "objectSource": [
                        {
                            "element_id": "1",
                            "value": {
                                "姓名": "张三"
                            }
                        },
                        {
                            "element_id": "6",
                            "value": {
                                "身份证号": "110111100020008765"
                            }
                        }
                    ]
                    }
                ]
                },
            ]
        }
    ],
    "tools":[
        {
            "type": "coding_agent",
            "enable": False
        }
    ]
}

# 来自 storage/question_tasks_poc1226.json（table）
QUESTION_TABLE: dict[str, Any] = {
    "id": "q_28",
    "description": "公司近三年及2025年一季度新签合同额、政国企类项目占比、5亿元及以上项目占比的变化趋势如何？",
    "type": "table",
    "html": "<table><tr><td>项目</td><td>2022年</td><td>2023年</td><td>2024年</td><td>2025.1-3</td></tr></table>",
    "contents": "",
}

# 来自 notebook 手动 question_tasks（text-json）
QUESTION_TEXT_JSON: dict[str, Any] = {
    "id": "q_1_json",
    "description": "前五大下游企业明细表，提取：序号、客户名称、是否存在关联关系、是否国企、采购金额（万元）",
    "prompt": (
        "提取：序号、客户名称、是否存在关联关系、是否国企、采购金额（万元）"
        ' 示例：[{"序号":"1","客户名称":"客户一","是否存在关联关系":"","是否国企":"","采购金额":""}]'
    ),
    "type": "text-json",
    "html": "",
    "human": "True",
}
