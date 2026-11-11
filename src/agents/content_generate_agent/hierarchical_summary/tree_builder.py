from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

from ..input_models import DocumentItems, ImageItem, TableItem, TextItem, Item
from ..input_models import load_items
from ....utils.langfuse_tracing import trace_span
from loguru import logger

MAX_HEADING_LEVEL = 2 # 最大标题层级 大于该层级的视为非标题项 TODO：应该暴露出去而不是写在这里

@dataclass
class Node:
    # 子节点列表
    # 注：使用 = field(default_factory=list)而非[] 确保每个实例都有自己独立的空列表
    childs: List[Node] = field(default_factory=list)
    # 内容列表（元素为 Item 对象），按出现顺序存储；按 element_id 去重
    content: List[Item] = field(default_factory=list)
    # 该节点的内容总结（由 SummaryGenerator 生成，通常仅叶节点会赋值）
    summary: str | None = None


def _iter_items(doc: DocumentItems) -> Iterable[TextItem | TableItem | ImageItem]:
    return doc.root

# 由下至上汇总内容到父节点（并集，按 element_id 去重）
def post_order_accumulate(node: Node) -> List[Item]:
    # 后序遍历：先得到所有子节点内容
    for child in node.childs:
        post_order_accumulate(child)

    # 合并：父节点的内容 = 自身直系内容 + 子节点内容（基于 element_id 去重，保持首次出现顺序）
    seen_ids = set()
    merged: List[Item] = []

    # 先加入自身直系内容
    for it in node.content:
        eid = getattr(it, "element_id", None)
        if eid is None or eid in seen_ids:
            continue
        merged.append(it)
        seen_ids.add(eid)

    # 再合并所有子节点内容
    for child in node.childs:
        for it in child.content:
            eid = getattr(it, "element_id", None)
            if eid is None or eid in seen_ids:
                continue
            merged.append(it)
            seen_ids.add(eid)
            
    # 按 element_id 升序排列
    node.content = sorted(merged, key=lambda it: getattr(it, "element_id", 0))
    return node.content

def _build_tree_for_file(doc: List[Item], max_heading_level: int = MAX_HEADING_LEVEL) -> Node:
    """
    为单个文件构建树（原有逻辑）
    """
    root = Node()

    # 栈元素为 (level, node)
    stack: List[Tuple[int, Node]] = [(0, root)]

    for item in doc:
        if isinstance(item, TextItem) and item.headingLevel is not None and item.headingLevel <= max_heading_level:
            level = int(item.headingLevel)
            # 回退到比当前 level 小的层级，即为当前节点的父节点
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else root
            new_node = Node()
            # 标题本身也作为内容加入当前新节点；去重留在汇总阶段
            new_node.content.append(item) # 引用，不是拷贝
            parent.childs.append(new_node)
            stack.append((level, new_node))
        else:
            # 非标题项或标题缺失，作为最近标题节点内容
            target_node = stack[-1][1] if len(stack) > 1 else root
            # 去重留在汇总阶段
            target_node.content.append(item)  # type: ignore[attr-defined]

    post_order_accumulate(root)
    return root

@trace_span(name="tree_builder.build_tree")
def build_tree(doc: List[Item], max_heading_level: int = MAX_HEADING_LEVEL) -> Node:
    """
    方案1：按文件分组，为每个文件单独构建树，然后合并到全局根节点
    这样可以避免不同文件的标题/章节被错误地组织到同一棵树中
    """
    # 按 file_id 分组，保持稳定顺序
    file_order: List[Optional[str]] = []
    file_to_items: Dict[Optional[str], List[Item]] = {}

    for item in doc:
        fid = getattr(item, "file_id", None)
        if fid not in file_to_items:
            file_to_items[fid] = []
            file_order.append(fid)
        file_to_items[fid].append(item)

    # 创建全局根节点
    global_root = Node()

    # 为每个文件单独构建树，然后合并
    for fid in file_order:
        items = file_to_items.get(fid, [])
        if not items:
            logger.warning(f"警告：文件 {fid} 没有内容，已跳过")
            continue
            
        file_tree = _build_tree_for_file(items, max_heading_level=max_heading_level)
        
        # 将文件树的children直接挂到全局根（避免多余一层空节点）
        for child in file_tree.childs:
            global_root.childs.append(child)
        # 同时合并文件根的直系内容
        for it in file_tree.content:
            global_root.content.append(it)

    post_order_accumulate(global_root)
    return global_root


def build_tree_from_path(path: Path) -> Node:
    doc = load_items(path)
    return build_tree(doc.root)



if __name__ == "__main__":
    path = Path("./storage/20250919.json")
    tree = build_tree_from_path(path)
    print(tree)
