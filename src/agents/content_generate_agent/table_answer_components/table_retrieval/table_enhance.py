"""
表格上下文增强模块

该模块提供了增强表格检索结果上下文的功能，通过添加相邻元素来提供更完整的上下文信息。
"""

from typing import List, Set, Tuple

from ...input_models import Item, TableItem
from loguru import logger


def _find_adjacent_elements_range(
    table_item: TableItem,
    item_key_to_item: dict,
    added_item_keys: Set[Tuple[str, int]],
    context_window: int
) -> Tuple[List[Item], List[Item]]:
    """
    为单个 table_item 查找前n个和后n个相邻元素
    Args:
        table_item: 要查找相邻元素的表格项
        item_key_to_item: (file_id, element_id) -> Item 的映射字典
        added_item_keys: 已添加的 item key 集合，用于避免重复
        context_window: 要查找的前后元素数量
    Returns:
        Tuple[List[Item], List[Item]]: (前n个元素列表, 后n个元素列表)，按element_id顺序排列
    """
    file_id = table_item.file_id
    element_id = table_item.element_id
    
    prev_items: List[Item] = []
    next_items: List[Item] = []
    
    # 查找前n个元素（element_id - 1, element_id - 2, ..., element_id - context_window）
    for i in range(1, context_window + 1):
        prev_key = (str(file_id), element_id - i)
        if prev_key in item_key_to_item and prev_key not in added_item_keys:
            prev_item = item_key_to_item[prev_key]
            prev_items.append(prev_item)
            added_item_keys.add(prev_key)
            logger.debug(f"找到前第{i}个元素: file_id={file_id}, element_id={element_id - i}")
    
    # 查找后n个元素（element_id + 1, element_id + 2, ..., element_id + context_window）
    for i in range(1, context_window + 1):
        next_key = (str(file_id), element_id + i)
        if next_key in item_key_to_item and next_key not in added_item_keys:
            next_item = item_key_to_item[next_key]
            next_items.append(next_item)
            added_item_keys.add(next_key)
            logger.debug(f"找到后第{i}个元素: file_id={file_id}, element_id={element_id + i}")
    
    # 反转前n个元素，使其按 element_id 从小到大排列：element_id-n, element_id-n+1, ..., element_id-1
    prev_items.reverse()
    
    return prev_items, next_items


def enhance_table_items_context(
    content_list: List[Item],
    table_items: List[TableItem],
    context_window: int = 1
) -> List[Item]:
    """
    增强表格检索结果的上下文，通过添加相邻元素来提供更完整的信息。
    Args:
        content_list: 所有内容项列表，用于查找相邻元素
        table_items: 检索到的表格项列表
        context_window: 要添加的前后相邻元素数量，默认为1（即前后各1个元素）
    Returns:
        List[Item]: 增强后的表格项列表，每个 table_item 的相邻元素紧跟在它前后
    """
    if not table_items:
        logger.info("table_items 为空，返回空列表")
        return []
    
    if not content_list:
        logger.warning("content_list 为空，无法查找相邻元素，返回原始 table_items")
        return list(table_items)
    
    # 验证 context_window 参数
    if context_window < 0:
        logger.warning(f"context_window 参数为负数 ({context_window})，将使用默认值 1")
        context_window = 1
    
    if context_window == 0:
        logger.info("context_window 为 0，不添加相邻元素，返回原始 table_items")
        return list(table_items)
    
    # 创建 (file_id, element_id) -> Item 的映射，提高查找效率
    item_key_to_item = {
        (str(item.file_id), item.element_id): item 
        for item in content_list
    }
    
    logger.info(f"创建了 {len(item_key_to_item)} 个 item 的映射，用于查找相邻元素，context_window={context_window}")
    
    # 使用 (file_id, element_id) 作为唯一标识符
    added_item_keys: Set[Tuple[str, int]] = set()
    
    # 首先将所有 table_items 的 key 添加到已添加集合中，避免重复添加原始 table_items
    for table_item in table_items:
        table_item_key = (str(table_item.file_id), table_item.element_id)
        added_item_keys.add(table_item_key)
    
    # 存储增强后的结果，按照顺序排列
    enhanced_items: List[Item] = []
    adjacent_count = 0
    
    # 遍历所有 table_items，查找每个 item 的相邻元素并按顺序插入
    for table_item in table_items:
        prev_items, next_items = _find_adjacent_elements_range(
            table_item=table_item,
            item_key_to_item=item_key_to_item,
            added_item_keys=added_item_keys,
            context_window=context_window
        )
        
        # prev_items 已经反转，按 element_id 从小到大排列：element_id-n, element_id-n+1, ..., element_id-1
        # enhanced_items.extend(prev_items)
        ##### 提取前后相邻元素的 md 字段, 合并成字符串
        prev_md = "\n".join([item.md for item in prev_items if getattr(item, "md", None)])
        next_md = "\n".join([item.md for item in next_items if getattr(item, "md", None)])

        parts = []
        if prev_md:
            parts.append(prev_md)
            
        # 如果 table_caption 是非空 List[str]，且合并后字符串非空，则 append 合并结果
        if getattr(table_item, "table_caption", None) and isinstance(table_item.table_caption, list):
            caption_str = "".join(str(s) for s in table_item.table_caption if s is not None)
            if caption_str.strip():
                parts.append("当前表格的名字以及单位信息: " + caption_str)
        parts.append(table_item.md)
        if next_md:
            parts.append(next_md)
        table_item.md = "\n".join(parts)
        ##### 
        adjacent_count += len(prev_items)
        
        enhanced_items.append(table_item)
        
        # next_items 按 element_id 从小到大排列：element_id+1, element_id+2, ..., element_id+n
        # enhanced_items.extend(next_items)
        adjacent_count += len(next_items)
    
    logger.info(
        f"上下文增强完成: 原始 table_items 数量={len(table_items)}, "
        f"找到相邻元素数量={adjacent_count}, "
        f"增强后总数量={len(enhanced_items)}"
    )
    
    return enhanced_items

