"""
基于 LangGraph + Function Calling 的财务指标变动提取器

工作流程：
1. 识别变动表格 (identify_change_tables)
2. 提取表格数据 (extract_table_data)
3. 查找变动原因 (find_change_reasons)
4. 匹配科目与原因 (match_reasons_to_items)
5. 生成结构化结果 (generate_output)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Literal
from loguru import logger
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import dotenv

from src.utils.model_factory import ModelFactory
from .base import BaseSectionExtractor

dotenv.load_dotenv()


# ============ 状态定义 ============
class ChangeItemState(BaseModel):
    """单个变动项的状态"""
    变动科目: Optional[str] = None
    本期数值: Optional[str] = None
    上期数值: Optional[str] = None
    变动幅度: Optional[str] = None
    变动原因: Optional[str] = None
    source_element_id: Optional[int] = None


class ChangesExtractorState(MessagesState):
    """变动提取器的工作流状态"""
    # 输入数据
    elements: List[Dict[str, Any]] = Field(default_factory=list, description="输入的元素列表")
    model_id: str = Field(default="model_for_extractor", description="使用的模型ID")
    
    # 中间状态
    identified_tables: List[Dict[str, Any]] = Field(default_factory=list, description="识别出的表格")
    extracted_items: List[Dict[str, Any]] = Field(default_factory=list, description="提取的变动项")
    reasons_pool: List[Dict[str, str]] = Field(default_factory=list, description="原因池")
    
    # 输出（流式输出时关心的字段）
    final_results: List[Dict[str, Any]] = Field(default_factory=list, description="最终结果")
    
    # 进度信息
    current_step: str = Field(default="start", description="当前步骤")


# ============ 工具函数定义 (Function Calling) ============

class TableIdentificationTool(BaseModel):
    """识别变动表格的工具"""
    table_element_ids: List[int] = Field(
        description="包含财务指标变动信息的表格元素ID列表。表格需包含：科目名称、本期数值、上期数值、变动比例等字段"
    )
    reasoning: str = Field(
        description="为什么选择这些表格的理由"
    )


class TableDataTool(BaseModel):
    """提取表格数据的工具"""
    items: List[Dict[str, Any]] = Field(
        description="从表格中提取的变动项列表，每项包含：变动科目、本期数值、上期数值、变动幅度、source_element_id"
    )


class ReasonItem(BaseModel):
    """单个变动原因项"""
    subject: str = Field(description="科目名称，如：营业收入、财务费用等")
    reason: str = Field(description="变动原因的完整描述")
    source_element_id: int = Field(description="原因来源的元素ID")


class ReasonExtractionTool(BaseModel):
    """提取变动原因的工具"""
    reasons: List[ReasonItem] = Field(
        description="提取的变动原因列表"
    )


class ReasonMatchingTool(BaseModel):
    """匹配科目与原因的工具"""
    matches: List[Dict[str, Any]] = Field(
        description="科目与原因的匹配结果，每项包含完整的变动信息"
    )


# ============ LLM 辅助函数 ============

def _get_model_from_state(state: ChangesExtractorState):
    """从状态中获取 model_id 并创建模型"""
    model_id = state.get("model_id", "model_for_extractor")
    return ModelFactory.get_model(key=model_id, temperature=0.01)


# ============ 节点函数 ============

def identify_tables_node(state: ChangesExtractorState) -> Dict[str, Any]:
    """节点1: 识别包含变动信息的表格"""
    logger.info("[ChangesExtractor] 步骤1: 识别变动表格")
    
    elements = state.get("elements", [])
    model = _get_model_from_state(state)
    
    # 过滤出表格元素
    table_elements = []
    for elem in elements:
        md_content = elem.get("md", "")
        # 支持三种表格格式：
        # 1. type字段标识
        # 2. HTML表格
        # 3. Markdown表格（以 | 开头，包含多个 | 分隔符）
        if (elem.get("type") == "table" or 
            "<table>" in md_content or 
            (md_content.strip().startswith("|") and md_content.count("|") >= 6)):
            table_elements.append(elem)
    
    if not table_elements:
        logger.warning("[ChangesExtractor] 未找到表格元素")
        return {
            "identified_tables": [],
            "current_step": "extract_data"
        }
    
    # 构建提示词
    elements_info = []
    for elem in table_elements[:20]:  # 限制数量
        elements_info.append({
            "element_id": elem.get("element_id"),
            "page": elem.get("page"),
            "md_preview": elem.get("md", "")[:500]  # 预览前500字符
        })
    
    prompt = f"""
请识别包含**财务指标变动信息**的表格。

这类表格的特征：
1. 包含科目名称列（如：营业收入、净利润、财务费用等）
2. 包含本期数值和上期数值（或同比数据）
3. 包含变动比例或变动幅度
4. 通常标题包含"变动"、"同比"、"对比"等关键词

可用的表格元素：
{json.dumps(elements_info, ensure_ascii=False, indent=2)}

**重要**：你必须严格按照以下JSON格式返回结果（字段名必须是英文）：
{{
  "table_element_ids": [33, 176, 182],
  "reasoning": "这些表格符合财务指标变动信息的特征..."
}}

注意：
- table_element_ids: 包含财务指标变动信息的表格元素ID列表（数组）
- reasoning: 为什么选择这些表格的理由（字符串）
"""
    
    try:
        # 使用 with_structured_output 获取结构化响应
        model_with_structure = model.with_structured_output(TableIdentificationTool)
        
        messages = [
            SystemMessage(content="你是资深财报分析专家。"),
            HumanMessage(content=prompt)
        ]
        
        result: TableIdentificationTool = model_with_structure.invoke(messages)
        
        identified_ids = result.table_element_ids
        reasoning = result.reasoning
        
        identified_tables = [
            elem for elem in table_elements
            if elem.get("element_id") in identified_ids
        ]
        
        logger.info(f"[ChangesExtractor] 识别到 {len(identified_tables)} 个变动表格: {reasoning}")
        
        return {
            "identified_tables": identified_tables,
            "current_step": "extract_data"
        }
        
    except Exception as e:
        logger.exception(f"[ChangesExtractor] 识别表格失败: {e}")
        return {
            "identified_tables": [],
            "current_step": "extract_data"
        }


def extract_table_data_node(state: ChangesExtractorState) -> Dict[str, Any]:
    """节点2: 从表格中提取数据"""
    logger.info("[ChangesExtractor] 步骤2: 提取表格数据")
    
    identified_tables = state.get("identified_tables", [])
    model = _get_model_from_state(state)
    
    if not identified_tables:
        logger.warning("[ChangesExtractor] 无表格可提取")
        return {
            "extracted_items": [],
            "current_step": "find_reasons"
        }
    
    all_items = []
    
    for table in identified_tables:
        element_id = table.get("element_id")
        md_content = table.get("md", "")
        
        prompt = f"""
从以下表格中提取财务指标变动数据。

表格内容（Markdown格式）：
{md_content}

提取规则：
1. 识别每一行的科目名称（第一列）
2. 提取本期数值、上期数值（通常是第2、3列）
3. 提取变动比例/变动幅度（通常是最后一列）
4. 只提取变动幅度绝对值 >= 10% 的项目，或标题中包含"重大变动"的项目
5. source_element_id 设置为: {element_id}

**重要**：你必须严格按照以下JSON格式返回结果（字段名必须是英文）：
{{
  "items": [
    {{
      "变动科目": "营业收入",
      "本期数值": "1000000",
      "上期数值": "800000", 
      "变动幅度": "25%",
      "source_element_id": {element_id}
    }}
  ]
}}

注意：
- items: 提取的变动项列表（数组），每项包含上述字段
"""
        
        try:
            model_with_structure = model.with_structured_output(TableDataTool)
            
            messages = [
                SystemMessage(content="你是资深财报分析专家。"),
                HumanMessage(content=prompt)
            ]
            
            result: TableDataTool = model_with_structure.invoke(messages)
            items = result.items
            
            logger.info(f"[ChangesExtractor] 从表格 {element_id} 提取到 {len(items)} 个变动项")
            all_items.extend(items)
            
        except Exception as e:
            logger.exception(f"[ChangesExtractor] 提取表格数据失败: {e}")
            continue
    
    return {
        "extracted_items": all_items,
        "current_step": "find_reasons"
    }


def find_reasons_node(state: ChangesExtractorState) -> Dict[str, Any]:
    """节点3: 查找变动原因"""
    logger.info("[ChangesExtractor] 步骤3: 查找变动原因")
    
    elements = state.get("elements", [])
    identified_tables = state.get("identified_tables", [])
    extracted_items = state.get("extracted_items", [])
    model = _get_model_from_state(state)
    
    # 提取所有科目名称
    subjects = [item.get("变动科目") for item in extracted_items if item.get("变动科目")]
    
    if not subjects:
        logger.warning("[ChangesExtractor] 无科目需要查找原因")
        return {
            "reasons_pool": [],
            "current_step": "match_reasons"
        }
    
    # 收集可能包含原因的元素
    reason_sources = []
    
    # 1. 从表格的 table_footnote 中提取
    for table in identified_tables:
        footnotes = table.get("table_footnote", [])
        if footnotes:
            reason_sources.append({
                "type": "footnote",
                "element_id": table.get("element_id"),
                "content": "\n".join(footnotes)
            })
    
    # 2. 从表格附近的文本元素中提取
    table_ids = {t.get("element_id") for t in identified_tables}
    for elem in elements:
        elem_id = elem.get("element_id")
        elem_type = elem.get("type", "")
        
        # 如果元素在表格附近（前后5个元素内）
        if elem_type == "text" and any(abs(elem_id - tid) <= 5 for tid in table_ids):
            md_text = elem.get("md", "")
            if any(kw in md_text for kw in ["原因", "由于", "主要系", "因为"]):
                reason_sources.append({
                    "type": "text",
                    "element_id": elem_id,
                    "content": md_text
                })
    
    if not reason_sources:
        logger.warning("[ChangesExtractor] 未找到原因来源")
        return {
            "reasons_pool": [],
            "current_step": "match_reasons"
        }
    
    # 使用 LLM 提取原因
    prompt = f"""
从以下文本中提取财务指标的变动原因。

需要查找的科目：
{json.dumps(subjects, ensure_ascii=False)}

可能包含原因的文本：
{json.dumps(reason_sources, ensure_ascii=False, indent=2)}

提取规则：
1. 识别每个科目的变动原因描述
2. 原因描述应完整、准确，直接来源于原文
3. 如果是脚注，需要根据编号（如"(1)"）匹配科目
4. 如果在正文中，需要识别"XXX变动的原因是..."这样的表述
5. 每个原因需要关联到具体的 source_element_id

**重要**：你必须严格按照以下JSON格式返回结果（字段名必须是英文）：
{{
  "reasons": [
    {{
      "subject": "营业收入",
      "reason": "主要系产品销售增加所致",
      "source_element_id": 35
    }}
  ]
}}

注意：
- reasons: 提取的原因列表（数组），每项包含subject（科目名称）、reason（原因描述）、source_element_id（来源ID）
"""
    
    try:
        model_with_structure = model.with_structured_output(ReasonExtractionTool)
        
        messages = [
            SystemMessage(content="你是资深财报分析专家。"),
            HumanMessage(content=prompt)
        ]
        
        result: ReasonExtractionTool = model_with_structure.invoke(messages)
        reasons = result.reasons
        
        logger.info(f"[ChangesExtractor] 提取到 {len(reasons)} 条变动原因")
        
        # 将 ReasonItem 对象列表转换为字典列表，便于后续处理
        reasons_dict = [reason.model_dump() for reason in reasons]
        
        return {
            "reasons_pool": reasons_dict,
            "current_step": "match_reasons"
        }
        
    except Exception as e:
        logger.exception(f"[ChangesExtractor] 提取原因失败: {e}")
        return {
            "reasons_pool": [],
            "current_step": "match_reasons"
        }


def match_reasons_node(state: ChangesExtractorState) -> Dict[str, Any]:
    """节点4: 匹配科目与原因"""
    logger.info("[ChangesExtractor] 步骤4: 匹配科目与原因")
    
    extracted_items = state.get("extracted_items", [])
    reasons_pool = state.get("reasons_pool", [])
    model = _get_model_from_state(state)
    
    if not extracted_items:
        logger.warning("[ChangesExtractor] 无变动项需要匹配")
        return {
            "final_results": [],
            "current_step": "generate_output"
        }
    
    prompt = f"""
请将变动项与对应的变动原因进行匹配。

变动项列表：
{json.dumps(extracted_items, ensure_ascii=False, indent=2)}

原因池：
{json.dumps(reasons_pool, ensure_ascii=False, indent=2)}

匹配规则：
1. 根据科目名称进行精确或模糊匹配
2. 如果找到匹配的原因，将其填入对应的变动项中
3. 如果没有找到匹配的原因，变动原因字段保持为 null
4. 确保每个变动项都包含完整的字段

**重要**：你必须严格按照以下JSON格式返回结果（字段名必须是英文）：
{{
  "matches": [
    {{
      "变动科目": "营业收入",
      "本期数值": "1000000",
      "上期数值": "800000",
      "变动幅度": "25%",
      "变动原因": "主要系产品销售增加所致",
      "source_element_id": 33
    }}
  ]
}}

注意：
- matches: 匹配后的完整变动项列表（数组），每项包含所有字段
"""
    
    try:
        model_with_structure = model.with_structured_output(ReasonMatchingTool)
        
        messages = [
            SystemMessage(content="你是资深财报分析专家。"),
            HumanMessage(content=prompt)
        ]
        
        result: ReasonMatchingTool = model_with_structure.invoke(messages)
        matches = result.matches
        
        logger.info(f"[ChangesExtractor] 完成 {len(matches)} 个变动项的匹配")
        
        return {
            "final_results": matches,
            "current_step": "generate_output"
        }
        
    except Exception as e:
        logger.exception(f"[ChangesExtractor] 匹配原因失败: {e}")
        # 如果匹配失败，返回原始数据
        return {
            "final_results": extracted_items,
            "current_step": "generate_output"
        }


def generate_output_node(state: ChangesExtractorState) -> Dict[str, Any]:
    """节点5: 生成最终输出"""
    logger.info("[ChangesExtractor] 步骤5: 生成输出")
    
    final_results = state.get("final_results", [])
    
    formatted_results = []
    for item in final_results:
        formatted_results.append({
            "变动科目": item.get("变动科目"),
            "本期数值": item.get("本期数值"),
            "上期数值": item.get("上期数值"),
            "变动幅度": item.get("变动幅度"),
            "变动原因": item.get("变动原因"),
            "source_element_id": item.get("source_element_id")
        })
    
    logger.info(f"[ChangesExtractor] 最终输出 {len(formatted_results)} 个变动项")
    
    return {
        "final_results": formatted_results,
        "current_step": "end"
    }


# ============ 构建工作流图 ============

def create_changes_workflow():
    """创建变动提取工作流，并添加 Langfuse tracing"""
    workflow = StateGraph(ChangesExtractorState)
    
    # 添加节点
    workflow.add_node("identify_tables", identify_tables_node)
    workflow.add_node("extract_data", extract_table_data_node)
    workflow.add_node("find_reasons", find_reasons_node)
    workflow.add_node("match_reasons", match_reasons_node)
    workflow.add_node("generate_output", generate_output_node)
    
    # 定义流程
    workflow.add_edge(START, "identify_tables")
    workflow.add_edge("identify_tables", "extract_data")
    workflow.add_edge("extract_data", "find_reasons")
    workflow.add_edge("find_reasons", "match_reasons")
    workflow.add_edge("match_reasons", "generate_output")
    workflow.add_edge("generate_output", END)
    
    # Langfuse 追踪由 OTel LangchainInstrumentor 自动处理，无需手动 CallbackHandler
    return workflow.compile()


# ============ 提取器类 ============

class ChangesExtractorLangGraph(BaseSectionExtractor):
    """基于 LangGraph 的财务指标变动提取器"""
    
    @classmethod
    def section_key(cls) -> str:
        return "changes"
    
    @classmethod
    def recall_keywords(cls):
        return [
            "变动原因", "主要原因", "原因说明", "增减变动原因",
            "主要系", "原因为", "主要由于",
            "较上年同期增", "较上年同期减", "同比增长", "同比下降", "重大变动",
            "经营情况讨论与分析", "管理层讨论与分析",
            "主要会计数据和财务指标", "合并财务报表项目注释","资产及负债状况分析"
        ]
    
    @classmethod
    def recall_params(cls):
        return {"max_hits": 8, "max_per_hit": 200}
    
    def title(self) -> str:
        return "指标变动说明"
    
    def schema_hint(self) -> Dict[str, Any]:
        return {"指标变动说明": [{"变动科目": None, "本期数值": None, "上期数值": None, "变动幅度": None, "变动原因": None}]}
    
    def parse(self, raw: Dict[str, Any]) -> Any:
        """兼容 BaseSectionExtractor 的 parse 接口，但实际上 LangGraph 版本不使用这个方法"""
        return raw
    
    def __init__(self, *, items: List[Dict[str, Any]], indices: List[int], model_id: str = "model_for_extractor"):
        super().__init__(items=items, indices=indices, model_id=model_id)
        self.workflow = create_changes_workflow()
    
    def _make_elements(self, max_chars: int = 3000) -> List[Dict[str, Any]]:
        from .base import _make_elements
        return _make_elements(self.items, self.indices, max_chars=max_chars)
    
    def extract(self) -> Dict[str, Any]:
        elements = self._make_elements()
        logger.info(f"[ChangesExtractorLangGraph] 开始提取，共 {len(elements)} 个元素  model = {self.model_id}")
        

        logger.info("--------------------------")
        logger.info(elements)
        logger.info("--------------------------")


        # 初始化状态，直接在 State 中存储数据
        initial_state = {
            "elements": elements,
            "model_id": self.model_id,
            "identified_tables": [],
            "extracted_items": [],
            "reasons_pool": [],
            "final_results": [],
            "current_step": "start",
            "messages": []
        }
        
        # 运行工作流
        try:
            result = self.workflow.invoke(initial_state)
            final_results = result.get("final_results", [])
            
            return {
                "指标变动说明": final_results
            }
        except Exception as e:
            logger.exception(f"[ChangesExtractorLangGraph] 提取失败: {e}")
            return {
                "指标变动说明": []
            }
    
    def extract_to_html(self) -> str:
        result = self.extract()
        items = result.get("指标变动说明", [])
        
        if not items:
            return "<p>未提取到变动数据</p>"
        
        html_parts = [
            '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">',
            '  <thead>',
            '    <tr style="background-color: #f2f2f2;">',
            '      <th>变动科目</th>',
            '      <th>本期数值</th>',
            '      <th>上期数值</th>',
            '      <th>变动幅度</th>',
            '      <th>变动原因</th>',
            '    </tr>',
            '  </thead>',
            '  <tbody>'
        ]
        
        for item in items:
            html_parts.append('    <tr>')
            html_parts.append(f'      <td>{item.get("变动科目") or "-"}</td>')
            html_parts.append(f'      <td>{item.get("本期数值") or "-"}</td>')
            html_parts.append(f'      <td>{item.get("上期数值") or "-"}</td>')
            html_parts.append(f'      <td>{item.get("变动幅度") or "-"}</td>')
            html_parts.append(f'      <td>{item.get("变动原因") or "-"}</td>')
            html_parts.append('    </tr>')
        
        html_parts.extend([
            '  </tbody>',
            '</table>'
        ])
        
        return '\n'.join(html_parts)
