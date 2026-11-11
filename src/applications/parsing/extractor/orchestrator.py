from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from loguru import logger

from .base import BaseSectionExtractor, _recall_by_keywords, _make_elements, _call_llm_json
from .base_info_extractor import BaseInfoExtractor
from .audit_extractor import AuditExtractor
from .changes_extractor_langgraph import ChangesExtractorLangGraph


def _format_report_period(text: Optional[str]) -> Optional[str]:
    """
    格式化报告期数为标准日期格式
    将"2024年度报告"等文本转换为标准日期：
    - 年度报告/年报 -> YYYY-12-31
    - 第三季度报告/三季报 -> YYYY-09-30
    - 半年度报告/半年报 -> YYYY-06-30
    - 第一季度报告/一季报 -> YYYY-03-31
    - 纯年份（如"2024"）-> YYYY-12-31（默认为年度报告）
    """
    
    logger.info("[financial_meta] 格式化报告期数: {}", text)

    if not isinstance(text, str) or not text.strip():
        return text
    
    # 提取年份
    year_match = re.search(r"(\d{4})", text)
    if not year_match:
        return text
    year = year_match.group(1)
    
    # 按优先级匹配报告类型
    if "第三季度报告" in text or "三季报" in text or "三季度" in text:
        return f"{year}-09-30"
    elif "第一季度报告" in text or "一季报" in text or "一季度" in text:
        return f"{year}-03-31"
    elif "半年度报告" in text or "半年报" in text or "中期报告" in text or "半年度" in text:
        return f"{year}-06-30"
    elif "年度报告" in text or "年报" in text:
        return f"{year}-12-31"
    else:
        # 如果只有纯年份（如"2024"），默认为年度报告
        # 判断是否为纯年份：去除空格后长度为4且全是数字
        stripped = text.strip()
        if re.match(r"^\d{4}年?$", stripped):
            logger.warning(
                "[financial_meta] 期数格式不完整，仅包含年份'{}'，默认为年度报告 -> {}-12-31",
                text, year
            )
            return f"{year}-12-31"
        # 其他无法识别的格式，返回原值
        return text


class FinancialMetadataOrchestrator:
    def __init__(self, mineru_json: List[Dict[str, Any]]) -> None:
        self.items: List[Dict[str, Any]] = [obj for obj in mineru_json if isinstance(obj, dict)]
        if not self.items:
            logger.warning("[financial_meta] 输入为空或非标准结构，后续召回将可能失败")
        self.model_id: str = "model_for_extractor"
        logger.info("[financial_meta] 使用模型: {}", self.model_id)

    def _registry(self) -> List[type[BaseSectionExtractor]]:
        return [
            BaseInfoExtractor,
            AuditExtractor,
            ChangesExtractorLangGraph,
        ]

    def _recall(self) -> Dict[str, List[int]]:
        items = self.items
        recall_map: Dict[str, List[int]] = {}
        for extractor_cls in self._registry():
            key = extractor_cls.section_key() or extractor_cls.__name__
            custom = extractor_cls.recall_indices(items)
            if isinstance(custom, list) and custom:
                recall_map[key] = list(dict.fromkeys(custom))
                continue
            kws = extractor_cls.recall_keywords() or []
            params = extractor_cls.recall_params() or {}
            max_hits = int(params.get("max_hits", 6))
            max_per_hit = int(params.get("max_per_hit", 120))
            if kws:
                recall_map[key] = _recall_by_keywords(items, kws, max_hits=max_hits, max_per_hit=max_per_hit)
            else:
                recall_map[key] = []
        logger.info("[financial_meta] 召回统计: {}", {k: len(v or []) for k, v in recall_map.items()})
        return recall_map

    def run(self) -> Dict[str, Any]:
        items = self.items
        recall_map = self._recall()

        base_key = BaseInfoExtractor.section_key() or BaseInfoExtractor.__name__
        audit_key = AuditExtractor.section_key() or AuditExtractor.__name__
        changes_key = ChangesExtractorLangGraph.section_key() or ChangesExtractorLangGraph.__name__

        base_part = BaseInfoExtractor(items=items, indices=recall_map.get(base_key, []) or [], model_id=self.model_id).extract()
        audit_part = AuditExtractor(items=items, indices=recall_map.get(audit_key, []) or [], model_id=self.model_id).extract()
        changes_part = ChangesExtractorLangGraph(items=items, indices=recall_map.get(changes_key, []) or [], model_id=self.model_id).extract()

        result: Dict[str, Any] = {
            "企业全称": base_part.get("企业全称"),
            "证券代码": base_part.get("证券代码"),
            "交易简称": base_part.get("交易简称"),
            "当前期数": base_part.get("当前期数"),
            "审计": audit_part.get("审计"),
            "指标变动说明": changes_part.get("指标变动说明", []),
        }

        current_period_obj = result.get("当前期数")
        if current_period_obj and isinstance(current_period_obj.get("value"), str):
            original_value = current_period_obj["value"]
            formatted_value = _format_report_period(original_value)
            result["当前期数"]["value"] = formatted_value
            logger.info("[financial_meta] “当前期数” 格式化: '{}' -> '{}'", original_value, formatted_value)

        return result


