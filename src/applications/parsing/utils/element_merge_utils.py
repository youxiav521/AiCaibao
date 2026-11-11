"""element 合并工具函数。

用于合并短且语义不完整的 element，提升解析质量。
"""
from typing import Any, Dict, List
from loguru import logger


def merge_short_elements(
    items: List[Dict[str, Any]],
    min_length: int = 8,
    merge_types: set = {"text", "list"},
) -> List[Dict[str, Any]]:
    """合并短且语义不完整的 element 到前一个 element。
    
    合并规则：
    1. 当前 element 不是任何标题（headingLevel 为空或 0）
    2. 当前 element 类型在 merge_types 中（默认只合并 text 和 list）
    3. 当前 element 的文本长度小于 min_length
    4. 前一个 element 存在且类型也在 merge_types 中
    5. 前一个 element 不是1级或2级标题（避免污染重要标题）
    6. 前一个 element 与当前 element 在同一页（page 字段相同）
    7. 合并时更新 bbox 为两个 bbox 的外接矩形
    8. 合并时同时合并 text、md、value 字段
    
    Args:
        items: element 列表
        min_length: 最小长度阈值，小于此值的 element 会被考虑合并
        merge_types: 允许合并的 element 类型集合
        
    Returns:
        合并后的 element 列表
        
    Examples:
        >>> items = [
        ...     {"type": "text", "text": "这是一段正常的文本", "headingLevel": 0},
        ...     {"type": "text", "text": "短文本", "headingLevel": 0},
        ... ]
        >>> merged = merge_short_elements(items, min_length=10)
        >>> len(merged)
        1
    """
    if not items or not isinstance(items, list):
        return items
    
    merged_items: List[Dict[str, Any]] = []
    merged_count = 0
    
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            merged_items.append(item)
            continue
            
        item_type = item.get("type", "")
        heading_level = item.get("headingLevel", 0)
        text = item.get("value", "")
        
        # 判断当前元素是否是标题（任何等级的标题都不能作为"被合并者"）
        is_heading = heading_level and heading_level > 0
        
        # 判断是否是可合并类型
        is_mergeable_type = item_type in merge_types
        
        # 判断文本是否过短
        is_short = len(text) < min_length
        
        # 判断是否应该合并
        should_merge = (
            not is_heading  # 不是任何标题
            and is_mergeable_type  # 是可合并类型
            and is_short  # 文本过短
            and merged_items  # 有前一个元素
        )
        
        if should_merge:
            # 检查前一个元素
            prev_item = merged_items[-1]
            prev_type = prev_item.get("type", "")
            prev_heading_level = prev_item.get("headingLevel", 0)
            prev_page = prev_item.get("page")
            item_page = item.get("page")
            
            # 前一个元素是否是重要标题（1级或2级标题不能被合并到）
            prev_is_important_heading = prev_heading_level and (prev_heading_level == 1 or prev_heading_level == 2)
            
            # 跨页检查：必须在同一页才能合并
            is_same_page = (prev_page is not None and item_page is not None and prev_page == item_page)
            
            # 检查所有条件
            if prev_type in merge_types and is_same_page and not prev_is_important_heading:
                # 修复：统一使用 value 字段
                prev_value = prev_item.get("value", "")
                item_value = item.get("value", "")
                
                # 根据类型决定分隔符
                separator = "\n" if prev_type == "list" or item_type == "list" else " "
                
                # 合并 value 字段
                merged_value = prev_value + separator + item_value
                prev_item["value"] = merged_value
                
                # 合并 md 字段（如果存在）
                if "md" in prev_item or "md" in item:
                    prev_md = prev_item.get("md", "")
                    item_md = item.get("md", "")
                    prev_item["md"] = prev_md + separator + item_md
                
                # 合并 bbox：取两个 bbox 的外接矩形
                if "bBox" in prev_item and "bBox" in item:
                    prev_bbox = prev_item["bBox"]
                    item_bbox = item["bBox"]
                    if isinstance(prev_bbox, dict) and isinstance(item_bbox, dict):
                        # 修复：只在字段存在时才参与计算
                        merged_bbox = {}
                        for key, use_min in [("l", True), ("t", True), ("r", False), ("b", False)]:
                            if key in prev_bbox and key in item_bbox:
                                if use_min:
                                    merged_bbox[key] = min(prev_bbox[key], item_bbox[key])
                                else:
                                    merged_bbox[key] = max(prev_bbox[key], item_bbox[key])
                            elif key in prev_bbox:
                                merged_bbox[key] = prev_bbox[key]
                            elif key in item_bbox:
                                merged_bbox[key] = item_bbox[key]
                        
                        # 保留 coord_origin
                        if "coord_origin" in prev_bbox:
                            merged_bbox["coord_origin"] = prev_bbox["coord_origin"]
                        prev_item["bBox"] = merged_bbox
                
                merged_count += 1
                logger.debug(
                    f"合并短 element: idx={idx}, page={item_page}, type={item_type}, "
                    f"len={len(text)} -> 合并到前一个元素"
                )
                continue
        
        # 不合并，添加到结果列表
        merged_items.append(item)
    
    if merged_count > 0:
        logger.info(
            f"短 element 合并完成: 原始数量={len(items)}, "
            f"合并后数量={len(merged_items)}, 合并了 {merged_count} 个元素"
        )
    
    return merged_items
