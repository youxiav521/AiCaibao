"""
文档提取服务的工具函数
"""
from typing import List, Dict, Any, Union
from loguru import logger

from src.utils.chinese_converter import traditional_to_simplified


# 元素信息类型定义
ElementInfo = Dict[str, Union[str, List[Any]]]


def format_element_text(elem: ElementInfo) -> str:
    """
    将单个元素格式化为紧凑的文本行，用于拼入 LLM prompt。
    格式: "[{element_id}|{type}] {content}"
    内容会自动进行繁体→简体转换，确保 LLM 基于简体中文做提取。
    """
    elem_id = elem.get('id', '')
    content = elem.get('content', '')
    elem_type = elem.get('type', 'unknown')
    if isinstance(content, str):
        content = traditional_to_simplified(content)
    elif isinstance(content, list):
        content = traditional_to_simplified(str(content))
    return f"[{elem_id}|{elem_type}] {content}"


def calc_element_bytes(elem: ElementInfo) -> int:
    """计算单个元素格式化后的 UTF-8 字节数（含末尾换行）"""
    return len((format_element_text(elem) + "\n").encode('utf-8'))


def split_elements_into_chunks(
    element_info: List[ElementInfo],
    max_chunk_bytes: int = 30000
) -> List[List[ElementInfo]]:
    """
    将元素列表按字节大小分 chunk，并使用单元素滑动窗口保持相邻 chunk 的上下文连续。

    规则：
    1. 每个 chunk 总字节数不超过 max_chunk_bytes
    2. 首个 chunk 不带重叠元素
    3. 从第二个 chunk 开始，最多只回带上一个 chunk 的最后一个元素
    4. 若加入该重叠元素后无法再容纳至少一个新的元素，则不保留重叠
    5. 若单个元素本身超过 max_chunk_bytes，则强制单独成 chunk

    Args:
        element_info   : 元素信息列表
        max_chunk_bytes: 每个 chunk 的最大 UTF-8 字节数（默认 30KB）

    Returns:
        分 chunk 后的元素列表的列表
    """
    if not element_info:
        return []

    chunks: List[List[ElementInfo]] = []
    i = 0

    while i < len(element_info):
        current_chunk: List[ElementInfo] = []
        current_bytes = 0
        current_elem = element_info[i]
        current_elem_bytes = calc_element_bytes(current_elem)

        # 单个元素超过窗口大小，强制单独成 chunk
        if current_elem_bytes > max_chunk_bytes:
            logger.warning(
                f"元素 id={current_elem.get('id')} 大小 {current_elem_bytes} 字节"
                f" 超过 max_chunk_bytes={max_chunk_bytes}，单独成 chunk"
            )
            chunks.append([current_elem])
            i += 1
            continue

        # 从第二个 chunk 开始，尝试回带上一个 chunk 的最后一个元素
        if chunks:
            prev_elem = chunks[-1][-1]
            prev_elem_bytes = calc_element_bytes(prev_elem)
            if prev_elem_bytes + current_elem_bytes <= max_chunk_bytes:
                current_chunk.append(prev_elem)
                current_bytes += prev_elem_bytes

        # 当前 chunk 必须至少加入 1 个新的元素
        current_chunk.append(current_elem)
        current_bytes += current_elem_bytes
        i += 1

        # 继续贪心填充后续新元素
        while i < len(element_info):
            elem_bytes = calc_element_bytes(element_info[i])
            if current_bytes + elem_bytes <= max_chunk_bytes:
                current_chunk.append(element_info[i])
                current_bytes += elem_bytes
                i += 1
            else:
                break

        chunks.append(current_chunk)

    logger.info(
        f"split_elements_into_chunks: {len(element_info)} 个元素 → "
        f"{len(chunks)} 个 chunk（single-element overlap）"
    )
    return chunks


def _is_empty_value(value: Any) -> bool:
    """
    判断值是否为空（用于合并时跳过空值）
    
    Args:
        value: 要判断的值
        
    Returns:
        True如果值为空
    """
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list):
        # 空列表或所有元素都为空
        return len(value) == 0 or all(_is_empty_value(v) for v in value)
    if isinstance(value, dict):
        # 空字典或所有字段值都为空
        return len(value) == 0 or all(_is_empty_value(v) for v in value.values())
    if isinstance(value, (int, float)) and value == 0:
        # 数字0可能是有效值，不算空（除非明确为null）
        return False
    return False


def _merge_dict_values(dict1: Dict, dict2: Dict) -> Dict:
    """
    递归合并两个字典，后面的非空值覆盖前面的空值
    
    Args:
        dict1: 第一个字典
        dict2: 第二个字典
        
    Returns:
        合并后的字典
    """
    merged = dict1.copy()
    
    for key, value2 in dict2.items():
        if key not in merged:
            # 新字段，直接添加
            merged[key] = value2
        else:
            value1 = merged[key]
            
            # 如果value1为空，用value2覆盖（无论value2是否为空）
            if _is_empty_value(value1):
                merged[key] = value2
            # 如果value2为空，保留value1
            elif _is_empty_value(value2):
                pass  # 保持value1不变
            # 两个都非空，根据类型合并
            elif isinstance(value1, dict) and isinstance(value2, dict):
                # 递归合并字典
                merged[key] = _merge_dict_values(value1, value2)
            elif isinstance(value1, list) and isinstance(value2, list):
                # 合并列表
                # 检查是否是对象数组（数组中的元素都是字典）
                all_dicts_v1 = all(isinstance(v, dict) for v in value1) if value1 else False
                all_dicts_v2 = all(isinstance(v, dict) for v in value2) if value2 else False
                
                if all_dicts_v1 and all_dicts_v2:
                    # 对象数组：直接追加（每个对象字段相同但值不同，应保留所有对象）
                    merged_list = value1.copy()
                    for obj in value2:
                        if not _is_empty_value(obj):
                            merged_list.append(obj)
                    logger.debug(f"对象数组追加 - 字段 '{key}': {len(value1)} + {len(value2)} = {len(merged_list)} 个对象")
                else:
                    # 非对象数组：简单类型去重后追加
                    merged_list = value1.copy()
                    existing_simple = {str(v) for v in value1 if not isinstance(v, (dict, list))}
                    for v in value2:
                        if isinstance(v, (dict, list)):
                            merged_list.append(v)
                        elif str(v) not in existing_simple:
                            merged_list.append(v)
                
                merged[key] = merged_list
            else:
                # 两个都非空且不同：保留第一个值（chunk按文档顺序处理，先出现的通常更可靠）
                if value1 != value2:
                    logger.debug(f"合并冲突 - 字段 '{key}': 保留第一个值 '{value1}'，忽略 '{value2}'")
                # 保持value1不变
    
    return merged


def merge_chunk_results(
    chunk_results: List[Dict[str, Any]],
    json_schema: str = None
) -> Dict[str, Any]:
    """
    合并多个chunk的提取结果
    智能处理空值：后续chunk的非空值会覆盖前面chunk的空值
    支持 objectValue 为字典或列表类型
    
    Args:
        chunk_results: 多个chunk的提取结果列表
        json_schema: JSON Schema（用于判断合并策略，当前未使用）
        
    Returns:
        合并后的结果，包含objectValue和objectSource
    """
    if not chunk_results:
        return {"objectValue": {}, "objectSource": []}
    
    if len(chunk_results) == 1:
        return chunk_results[0]
    
    # 检测 objectValue 类型（取第一个非空的值来判断）
    first_value = None
    for chunk_result in chunk_results:
        v = chunk_result.get("objectValue")
        if v is not None and (isinstance(v, list) and len(v) > 0 or isinstance(v, dict) and len(v) > 0):
            first_value = v
            break
    
    is_list_type = isinstance(first_value, list)
    
    # 合并objectValue
    merged_object_value = [] if is_list_type else {}
    all_object_sources = []
    
    for idx, chunk_result in enumerate(chunk_results):
        chunk_value = chunk_result.get("objectValue", [] if is_list_type else {})
        chunk_source = chunk_result.get("objectSource", [])
        
        if is_list_type:
            # 列表类型：直接追加所有元素
            if isinstance(chunk_value, list):
                merged_object_value.extend(chunk_value)
            elif chunk_value:  # 非空的非列表值也添加
                merged_object_value.append(chunk_value)
        else:
            # 字典类型：使用智能合并函数
            if isinstance(chunk_value, dict):
                merged_object_value = _merge_dict_values(merged_object_value, chunk_value)
        
        # 收集所有objectSource
        all_object_sources.extend(chunk_source)
    
    return {
        "objectValue": merged_object_value,
        "objectSource": all_object_sources
    }
