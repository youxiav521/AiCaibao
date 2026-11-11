from __future__ import annotations

from typing import Any, Dict, List


def parse_financial_metadata(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """兼容层：调用新版 Orchestrator 提取财报元数据。

    参数
    - items: MinerU 转换后的扁平元素列表（dict 列表）

    返回
    - metadata_v7 风格的字典，键包含：
      "企业全称"、"证券代码"、"交易简称"、"当前期数"、"审计"、"指标变动说明"
    """
    # 延迟导入以避免启动时循环依赖
    from ..extractor.orchestrator import FinancialMetadataOrchestrator

    orchestrator = FinancialMetadataOrchestrator(items)
    return orchestrator.run()


def parse_three_statements(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """提取三大表（资产负债表/利润表/现金流量表）信息。

    返回结构
    - { "三大表": [ { "element": {...}, "单位": {"value": str | None, "source_element_id": int | None } } ] }
    """
    # 延迟导入以避免启动时循环依赖
    from ..extractor.three_statements_extractor import ThreeStatementsExtractor

    safe_items: List[Dict[str, Any]] = [obj for obj in items if isinstance(obj, dict)]
    extractor = ThreeStatementsExtractor(
        items=safe_items,
        indices=ThreeStatementsExtractor.recall_indices(safe_items),
        model_id="model_for_extractor",
    )
    return extractor.extract()


