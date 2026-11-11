"""Step2: 过滤排序 boxes，在内存中处理，不落盘"""
from .utils import filter_and_sort_boxes


def execute(detect_results: list[dict]) -> list[dict]:
    """对 detect 结果过滤排序 boxes，返回新的 dict 列表。"""
    out = []
    for d in detect_results:
        boxes = filter_and_sort_boxes(d.get("boxes", []))
        out.append({
            "input_path": d.get("input_path"),
            "page_index": d.get("page_index"),
            "boxes": boxes,
        })
    return out
