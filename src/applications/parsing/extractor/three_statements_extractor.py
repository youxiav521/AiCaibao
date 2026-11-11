from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger

from .base import (
    BaseSectionExtractor,
    _collect_subtree,
    _is_heading,
    _make_elements,
)


_THREE_STATEMENT_LABELS = {
    "合并资产负债表",
    "母公司资产负债表",
    "合并利润表",
    "母公司利润表",
    "合并现金流量表",
    "母公司现金流量表",
}

_THREE_STATEMENT_KEYWORDS = ["资产负债表", "利润表", "现金流量表"]


def _get_table_type(element: Dict[str, Any], items: List[Dict[str, Any]], idx: int) -> Optional[str]:
    t = element.get("table_type")
    if isinstance(t, str) and t.strip():
        return t.strip()
    # 兜底：回看上方最近的标题作为可能的类型线索
    try:
        page = element.get("page")
        j = idx - 1
        while j >= 0:
            it = items[j]
            if page is not None and it.get("page") is not None and it.get("page") != page:
                break
            is_head, _lv = _is_heading(it)
            if is_head:
                txt = (it.get("md") or it.get("value") or "").strip()
                if txt:
                    return txt
                break
            j -= 1
    except Exception:
        pass
    return None


def _is_three_statement_table(element: Dict[str, Any], items: List[Dict[str, Any]], idx: int) -> bool:
    if element.get("type") not in {"table", "excel_table"}:
        return False
    label = _get_table_type(element, items, idx)
    if not isinstance(label, str):
        return False
    label = label.strip()
    if label in _THREE_STATEMENT_LABELS:
        return True
    return any(kw in label for kw in _THREE_STATEMENT_KEYWORDS)


def _find_section_context_indices(items: List[Dict[str, Any]], start_index: int, max_per_hit: int = 200) -> List[int]:
    # 基于最近的上方标题，收集该标题的“同级+子树”作为上下文
    try:
        page = items[start_index].get("page")
    except Exception:
        page = None
    head_idx: Optional[int] = None
    head_level: int = 0
    i = start_index
    while i >= 0:
        it = items[i]
        if page is not None and it.get("page") is not None and it.get("page") != page:
            break
        is_head, lv = _is_heading(it)
        if is_head:
            head_idx = i
            head_level = lv
            break
        i -= 1
    if head_idx is not None:
        return _collect_subtree(items, head_idx, head_level, max_blocks=max_per_hit)
    # 兜底：同页的局部窗口
    start = max(0, start_index - 30)
    end = min(len(items), start_index + 1 + 100)
    picked: List[int] = []
    for j in range(start, end):
        it = items[j]
        if page is not None and it.get("page") is not None and it.get("page") != page:
            continue
        picked.append(j)
        if len(picked) >= max_per_hit:
            break
    return picked


class ThreeStatementsExtractor(BaseSectionExtractor):
    @classmethod
    def section_key(cls) -> str:
        return "three_statements"

    @classmethod
    def recall_indices(cls, items: List[Dict[str, Any]]) -> List[int]:
        indices: List[int] = []
        for idx, it in enumerate(items):
            try:
                if _is_three_statement_table(it, items, idx):
                    indices.append(idx)
            except Exception:
                continue
        # 去重并限制数量
        seen = set()
        deduped: List[int] = []
        for i in indices:
            if i not in seen:
                deduped.append(i)
                seen.add(i)
        return deduped[:24]

    def title(self) -> str:
        return "识别三大表对应的金额单位（逐表）"

    def schema_hint(self) -> Dict[str, Any]:
        return {"单位": {"value": None, "source_element_id": None}}

    def extra_rules(self) -> Optional[str]:
        return (
            "要求：仅基于提供的 elements 文本判断该表格使用的金额单位。\n"
            "- 如有‘单位：人民币元/万元/亿元’等明确标注，直接取之；\n"
            "- 若存在‘除特别说明外，金额单位为…’等全局声明且出现在上下文，也可作为依据；\n"
            "- 不要臆测；找不到就返回 null。\n"
            "- source_element_id 指向含有该单位文字的最直接元素（通常为‘单位：…’那一行）。"
        )

    def parse(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {"单位": raw.get("单位", {"value": None, "source_element_id": None})}

    def _max_chars(self) -> int:
        return 2400

    def extract(self) -> Dict[str, Any]:  # type: ignore[override]
        results: List[Dict[str, Any]] = []
        if not (self.indices and isinstance(self.indices, list)):
            logger.info("[three_statements] 未召回到任何三大表")
            return {"三大表": results}

        for t_idx in self.indices:
            try:
                table_el = self.items[t_idx]
                if table_el.get("type") not in {"table", "excel_table"}:
                    continue
                ctx_indices = _find_section_context_indices(self.items, t_idx, max_per_hit=220)
                elements = _make_elements(self.items, ctx_indices, max_chars=self._max_chars())

                prompt = self._make_prompt(elements)
                raw = self._call_llm(prompt)
                parsed = self.parse(raw)

                element_id = table_el.get("element_id") or (t_idx + 1)
                page = table_el.get("page")
                table_type = _get_table_type(table_el, self.items, t_idx)
                entry_el: Dict[str, Any] = {
                    "element_id": element_id,
                    "table_type": table_type,
                }
                if isinstance(page, int):
                    entry_el["page"] = page
                # 拷贝表格核心字段
                for k in ["rows", "html", "two_d_data"]:
                    if k in table_el:
                        entry_el[k] = table_el.get(k)

                results.append({
                    "element": entry_el,
                    "单位": parsed.get("单位", {"value": None, "source_element_id": None}),
                })
            except Exception as e:
                logger.warning("[three_statements] 处理表格失败: idx={}, err={}", t_idx, str(e)[:200])
                continue

        # 后处理：按类型仅保留每类“最长”的表格（合并/母公司 × 三大表）
        try:
            filtered = self._select_longest_per_type(results)
            return {"三大表": filtered}
        except Exception as e:
            logger.warning("[three_statements] 后处理失败，返回原始结果: {}", str(e)[:200])
            return {"三大表": results}

    def _call_llm(self, prompt_text: str) -> Dict[str, Any]:
        try:
            from .base import _call_llm_json
        except Exception as e:
            logger.exception("[three_statements] 无法导入 LLM 调用: {}", e)
            return {}
        return _call_llm_json(self.model_id, prompt_text)

    # ---------- 后处理辅助函数 ----------
    @staticmethod
    def _normalize_group_key(table_type: Optional[str]) -> Optional[str]:
        if not isinstance(table_type, str):
            return None
        label = table_type.strip()
        # 基类关键词判断
        base: Optional[str] = None
        if "资产负债表" in label:
            base = "资产负债表"
        elif "利润表" in label:
            base = "利润表"
        elif "现金流量表" in label:
            base = "现金流量表"
        if base is None:
            return None
        scope: Optional[str] = None
        if "合并" in label:
            scope = "合并"
        elif "母公司" in label:
            scope = "母公司"
        else:
            # 若无法分辨合并/母公司，则不纳入六大类的竞争
            return None
        return f"{scope}{base}"

    @staticmethod
    def _estimate_table_size(entry_element: Dict[str, Any]) -> int:
        # 优先 two_d_data 的 行×最大列
        try:
            two_d = entry_element.get("two_d_data")
            if isinstance(two_d, list) and two_d:
                rows = len(two_d)
                max_cols = 0
                for r in two_d:
                    if isinstance(r, list):
                        c = len(r)
                    elif isinstance(r, dict):
                        c = len(r.get("cells", []) if isinstance(r.get("cells"), list) else [])
                    else:
                        c = 0
                    if c > max_cols:
                        max_cols = c
                if rows > 0 and max_cols > 0:
                    return rows * max_cols
        except Exception:
            pass
        # 次选 rows 的条目数
        try:
            rows_obj = entry_element.get("rows")
            if isinstance(rows_obj, list) and rows_obj:
                return len(rows_obj)
        except Exception:
            pass
        # 兜底 html 长度
        try:
            html = entry_element.get("html")
            if isinstance(html, str) and html:
                return len(html)
        except Exception:
            pass
        # 兜底固定较小值
        return 0

    def _select_longest_per_type(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in results:
            try:
                el = item.get("element", {}) if isinstance(item, dict) else {}
                key = self._normalize_group_key(el.get("table_type"))
                if key is None:
                    continue
                groups.setdefault(key, []).append(item)
            except Exception:
                continue

        # 每组选择“最长”的一个
        winners: List[Dict[str, Any]] = []
        for key, items in groups.items():
            try:
                best = max(
                    items,
                    key=lambda it: self._estimate_table_size(
                        it.get("element", {}) if isinstance(it, dict) else {}
                    ),
                )
                winners.append(best)
            except Exception:
                continue

        # 可按固定顺序输出，提升稳定性
        order = [
            "合并资产负债表",
            "母公司资产负债表",
            "合并利润表",
            "母公司利润表",
            "合并现金流量表",
            "母公司现金流量表",
        ]
        order_map = {k: i for i, k in enumerate(order)}
        winners.sort(
            key=lambda it: order_map.get(
                self._normalize_group_key(
                    (it.get("element") or {}).get("table_type") if isinstance(it, dict) else None
                ),
                999,
            )
        )
        return winners


