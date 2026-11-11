import re
from typing import Any, Dict, List, Optional, DefaultDict, Tuple

# ---------------- 标题分级规则：模式匹配优先 + 状态堆栈 ----------------

# 步骤 1: 预定义标题模式 （能被匹配上的，就是 标题）
# 优先级越低，数字越大
TITLE_PATTERNS = [
    {'name': 'chapter', 'regex': re.compile(r"^第[一二三四五六七八九十百零]{1,3}[节章]"), 'priority': 1},
    {'name': 'chinese_numeral_dot', 'regex': re.compile(r"^[一二三四五六七八九十百零]{1,3}[、．.]"), 'priority': 2},
    {'name': 'chinese_numeral_paren', 'regex': re.compile(r"^[（(][一二三四五六七八九十百零]{1,3}[）)]"), 'priority': 4},
    {'name': 'digit_dot', 'regex': re.compile(r"^\d{1,3}[、．.]"), 'priority': 4},
    {'name': 'digit_paren', 'regex': re.compile(r"^[（(]\d{1,3}[）)]"), 'priority': 5},
    {'name': 'circle_digit', 'regex': re.compile(r"^[\u2460-\u2473]"), 'priority': 6},
]

# 名称到优先级的快速查找表
NAME_TO_PRIORITY = {p['name']: p['priority'] for p in TITLE_PATTERNS}

# 便捷集合：哪些模式属于“带编号”的标题
NUMBERED_PATTERNS = {
    'chinese_numeral_dot', 'chinese_numeral_paren',
    'digit_dot', 'digit_paren', 'circle_digit'
}

# 等优先级下的“父→子”映射
EQUAL_PRIORITY_CHILDREN: Dict[str, set] = {
    'digit_dot': {'chinese_numeral_paren'},
    'chinese_numeral_dot': {'digit_paren'},
    'chinese_numeral_paren': {'digit_dot'},
}

def _chinese_numeral_to_int(token: str) -> Optional[int]:
    if not token:
        return None
    digits = {
        '零': 0,
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9
    }
    if token in digits:
        return digits[token]
    if token == '十':
        return 10
    if len(token) == 2:
        a, b = token[0], token[1]
        if a == '十' and b in digits:
            return 10 + digits[b]
        if a in digits and b == '十':
            return digits[a] * 10
    if len(token) == 3:
        a, b, c = token[0], token[1], token[2]
        if a in digits and b == '十' and c in digits:
            return digits[a] * 10 + digits[c]
    return None

def _extract_number_from_text(text: str, pattern_name: str) -> Optional[int]:
    if not text or pattern_name not in NUMBERED_PATTERNS:
        return None
    cleaned = re.sub(r"\s+", "", text)
    try:
        if pattern_name == 'digit_dot':
            m = re.match(r"^(\d{1,3})[、．.]", cleaned)
            return int(m.group(1)) if m else None
        if pattern_name == 'digit_paren':
            m = re.match(r"^[（(](\d{1,3})[）)]", cleaned)
            return int(m.group(1)) if m else None
        if pattern_name == 'chinese_numeral_dot':
            m = re.match(r"^([一二三四五六七八九十百零]{1,3})[、．.]", cleaned)
            return _chinese_numeral_to_int(m.group(1)) if m else None
        if pattern_name == 'chinese_numeral_paren':
            m = re.match(r"^[（(]([一二三四五六七八九十百零]{1,3})[）)]", cleaned)
            return _chinese_numeral_to_int(m.group(1)) if m else None
        if pattern_name == 'circle_digit':
            ch = cleaned[0]
            code = ord(ch)
            if 0x2460 <= code <= 0x2473:
                return (code - 0x2460) + 1
            return None
    except Exception:
        return None
    return None

def _is_toc_like_block(text: str) -> bool:
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    if len(lines) < 2:
        return False
    heading_line_count = 0
    trailing_page_num_count = 0
    dot_leader_count = 0
    for ln in lines:
        cleaned_ln = re.sub(r"\s+", "", ln)
        if re.match(r"^第[一二三四五六七八九十百零]{1,3}[节章]", cleaned_ln):
            heading_line_count += 1
        if re.search(r"[\.．…·•⋯]{2,}", ln):
            dot_leader_count += 1
        if re.search(r"\d{1,4}\s*$", ln):
            trailing_page_num_count += 1
    if heading_line_count >= 2 and (trailing_page_num_count >= 2 or dot_leader_count >= 1):
        return True
    occurrences = len(re.findall(r"第[一二三四五六七八九十百零]{1,3}[节章]", re.sub(r"\s+", "", text)))
    return occurrences >= 3

def _preprocess_and_extract_features(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    potential_titles: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if it.get("type") != "text":
            continue
        text = (it.get("md") or it.get("value") or "").strip()
        if not text:
            continue
        if _is_toc_like_block(text):
            continue
        matched_pattern: Optional[Dict[str, Any]] = None
        cleaned_text = re.sub(r"\s+", "", text)
        for pattern in TITLE_PATTERNS:
            if pattern['regex'].match(cleaned_text):
                matched_pattern = pattern
                break
        if matched_pattern:
            pattern_name = matched_pattern['name']
            num = _extract_number_from_text(text, pattern_name)
            potential_titles.append({
                'original_index': i,
                'text': text,
                'pattern_name': pattern_name,
                'pattern_priority': matched_pattern['priority'],
                'num': num,
            })
    return potential_titles

def assign_heading_levels(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    为转换后的 items 赋予 headingLevel，将第一个识别到的标题视为一级。
    """
    if not isinstance(items, list) or not items:
        return items

    potential_titles = _preprocess_and_extract_features(items)

    if not potential_titles:
        return items

    title_stack: List[Dict[str, Any]] = []
    levels = [0] * len(items)

    for current_title in potential_titles:
        original_index = current_title['original_index']
        current_pattern = current_title['pattern_name']
        current_priority = current_title['pattern_priority']
        current_num = current_title['num']

        # 规则1：栈为空时，将当前标题作为第一级标题
        if not title_stack:
            final_level = 1
            stack_item = {
                'level': final_level,
                'pattern_name': current_pattern,
                'pattern_priority': current_priority,
                'num': current_num
            }
            title_stack.append(stack_item)
            levels[original_index] = final_level
            continue

        # --- 通用层级判断逻辑开始 ---
        last_in_stack = title_stack[-1]

        # 规则2：同级判断
        is_sequential_sibling = (
            current_pattern in NUMBERED_PATTERNS and
            current_pattern == last_in_stack['pattern_name'] and
            current_num is not None and
            last_in_stack.get('num') is not None and
            current_num == last_in_stack['num'] + 1
        )

        if is_sequential_sibling:
            final_level = last_in_stack['level']
            title_stack.pop()
        else:
            # 规则3：同模式非连续，若当前编号为 1，则视为下沉到子级
            is_same_pattern_non_consecutive = (
                current_pattern == last_in_stack['pattern_name'] and
                current_num is not None and
                last_in_stack.get('num') is not None and
                current_num != last_in_stack['num'] + 1
            )
            if is_same_pattern_non_consecutive and current_num == 1:
                final_level = min(last_in_stack['level'] + 1, 6)
                stack_item = {
                    'level': final_level,
                    'pattern_name': current_pattern,
                    'pattern_priority': current_priority,
                    'num': current_num
                }
                title_stack.append(stack_item)
                levels[original_index] = final_level
                continue
            # 规则4：非同级兄弟，向上回溯，寻找正确的父级
            while title_stack:
                last_in_stack = title_stack[-1]
                if current_priority < last_in_stack['pattern_priority']:
                    title_stack.pop()
                    continue
                if current_pattern == last_in_stack['pattern_name']:
                    last_num = last_in_stack.get('num')
                    if (
                        current_num is not None and last_num is not None and
                        current_num != last_num + 1
                    ):
                        if current_num != 1:
                            title_stack.pop()
                            continue
                    break
                if current_priority == last_in_stack['pattern_priority'] and current_pattern != last_in_stack['pattern_name']:
                    children = EQUAL_PRIORITY_CHILDREN.get(last_in_stack['pattern_name'], set())
                    if current_pattern in children and (current_num is None or current_num == 1):
                        break
                    title_stack.pop()
                    continue
                break

            # 规则5：根据回溯后的父级，确定当前标题的层级
            parent = title_stack[-1] if title_stack else None
            if not parent:
                # 栈被弹空，说明当前标题是最高级标题
                final_level = 1
            else:
                if parent['pattern_name'] == current_pattern:
                    final_level = parent['level']
                    title_stack.pop()
                else:
                    final_level = parent['level'] + 1

        # 规则6：入栈与赋值
        final_level = min(final_level, 6)
        stack_item = {
            'level': final_level,
            'pattern_name': current_pattern,
            'pattern_priority': current_priority,
            'num': current_num
        }
        title_stack.append(stack_item)
        levels[original_index] = final_level

    # 写回 headingLevel 到原始 items
    for i, lv in enumerate(levels):
        if lv > 0:
            items[i]["headingLevel"] = lv

    return items

# ---------------- 在扁平 items 上分配 ID ----------------

def assign_ids_in_flat(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(items, list) or not items:
        return items
    section_counter = 0
    text_counter = 0
    table_counter = 0
    title_stack: List[Dict[str, Any]] = []
    children_count_by_parent: Dict[str, int] = {}

    def get_level(entry: Dict[str, Any]) -> Optional[int]:
        lvl = entry.get("headingLevel")
        if isinstance(lvl, int):
            return lvl
        try:
            if isinstance(lvl, str) and lvl.strip().isdigit():
                return int(lvl.strip())
        except Exception:
            pass
        return None

    for it in items:
        t = it.get("type")
        lvl = get_level(it)
        if t == "text" and lvl is not None:
            if not title_stack or lvl == 1:
                section_counter += 1
                section_id = f"S{section_counter}"
                title_stack = [{"level": lvl, "section_id": section_id}]
                children_count_by_parent.setdefault(section_id, 0)
                it["section_id"] = section_id
                continue
            while title_stack and lvl <= title_stack[-1]["level"]:
                title_stack.pop()
            if not title_stack:
                section_counter += 1
                section_id = f"S{section_counter}"
                title_stack = [{"level": lvl, "section_id": section_id}]
                children_count_by_parent.setdefault(section_id, 0)
                it["section_id"] = section_id
                continue
            parent = title_stack[-1]
            parent_id = parent["section_id"]
            children_count_by_parent[parent_id] = children_count_by_parent.get(parent_id, 0) + 1
            child_idx = children_count_by_parent[parent_id]
            section_id = f"{parent_id}_{child_idx}"
            it["section_id"] = section_id
            title_stack.append({"level": lvl, "section_id": section_id})
        else:
            if title_stack:
                current_section_id = title_stack[-1]["section_id"]
                it["section_ref"] = current_section_id
            if t == "text":
                text_counter += 1
                it.setdefault("text_id", f"T{text_counter}")
            elif t == "table":
                table_counter += 1
                it.setdefault("table_id", f"TBL{table_counter}")
    return items


