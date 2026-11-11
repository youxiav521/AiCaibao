"""ORDS Oman IncomeStatement PDF：同一页竖排三张独立表，必须全部输出。

源数据有三张 MinerU table 元素：
- id=1：``Income statement`` —— 标准利润表，table_type 已为「利润表」。
- id=2：``Statement of comprehensive income - Net of tax`` —— 独立 OCI 表
        （不含 Revenue 等 P&L 主体项），需要由 OCI 关键词识别为
        「利润表及其他综合收益」。
- id=3：``Analysis of Income and Expense`` —— table_type 是「其他表」，
        但表第一格内嵌标题，需要走「内嵌标题回退」+ 英文利润分析关键词
        识别为「利润表」。

历史 bug：id=2/id=3 都被丢弃，最终只输出 id=1 → 一张表。

修复后：3 张候选都进入输出（id=1 进 best，id=2 标签不冲突也进 best，
id=3 与 id=1 同 label「利润表」+ 同页 + bbox 不重叠 → 进 siblings）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_ords_items() -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    p = root / (
        "storage/mineru_output/converted_output/"
        "67fbcc9ab08c46949c1debf69579cfb5vmJwdXMHWC_mineru_converted.json"
    )
    assert p.is_file(), "fixture missing: {}".format(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _stub_llm(monkeypatch) -> None:
    """只 stub 表行过滤所用的 LLM 调用；列含义推断（caliber/year/...）走规则版本。"""
    from src.applications.parsing.service import three_statements_frontend_service as svc

    monkeypatch.setattr(
        svc, "_call_table_column_filter_llm", lambda prompt_text: {}, raising=False
    )


def test_ords_classification_recognizes_three_distinct_tables() -> None:
    """分类阶段：3 个 table 元素全部被分类（不依赖 LLM 推断列含义）。"""
    from src.applications.parsing.service.three_statements_frontend_service import (
        _normalize_statement_label,
        _normalize_pl_oci_combined_label,
        _normalize_english_statement_label,
        _extract_table_inline_title,
        _table_caption_joined,
        _looks_like_independent_same_page_tables,
    )

    items = _load_ords_items()
    cands: list[dict[str, Any]] = []
    for it in items:
        if it.get("type") != "table":
            continue
        tt = it.get("table_type")
        md = it.get("md", "") or ""
        norm = _normalize_statement_label(tt)
        oci_blob = "\n".join(
            x for x in (_table_caption_joined(it), str(md or "")[:4000]) if x
        )
        oci_label = _normalize_pl_oci_combined_label(oci_blob)
        if oci_label:
            norm = oci_label
        if not norm:
            inline_title = _extract_table_inline_title(it)
            if inline_title:
                norm = _normalize_english_statement_label(inline_title)
                if not norm:
                    inline_oci = _normalize_pl_oci_combined_label(inline_title)
                    if inline_oci:
                        norm = inline_oci
        cands.append({"id": it.get("element_id"), "label": norm, "page": it.get("page"), "table": it})

    by_id = {c["id"]: c for c in cands}
    assert by_id[1]["label"] == "利润表", by_id[1]
    assert by_id[2]["label"] == "利润表及其他综合收益", by_id[2]
    assert by_id[3]["label"] == "利润表", by_id[3]

    # id=1 与 id=3 同 label + 同页，bbox 不重叠 → 应被识别为同页独立兄弟表，进 siblings。
    assert _looks_like_independent_same_page_tables(by_id[1], by_id[3]) is True


def test_ords_extract_dual_caliber_splits_each_source_into_two(monkeypatch) -> None:
    """端到端：3 张源表都应做合并/母公司拆分 → 输出 6 张表。

    历史 bug（本次修复点）：id=2「Statement of comprehensive income - Net of tax」
    第 2 行是 ``STATEMENT OF COMPREHENSIVE INCOME CONSOLIDATED AND SEPARATE``
    的章节横幅（单 cell + ``colSpan=5``）。该 banner 被当作表头并入每一列文本，
    把 "consolidated" 词污染到本应为 standalone 的列，导致按列推断口径出现「混合」，
    dual-caliber 拆分不触发，最终 id=2 只输出 1 张。

    修复后：banner 行被识别并跳过，col 1/3 = 合并、col 2/4 = 单体，dual-caliber
    拆分正常触发，id=2 输出 2 张（合并 + 母公司）。
    """
    _stub_llm(monkeypatch)
    from src.applications.parsing.service.three_statements_frontend_service import (
        extract_three_statements_frontend,
        _strip_org_table_derivative_suffixes,
    )

    items = _load_ords_items()
    out = extract_three_statements_frontend(items, locale=None)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for t in out:
        for mid in t.get("mergeIds") or []:
            base = _strip_org_table_derivative_suffixes(str(mid))
            by_source.setdefault(base, []).append(t)

    assert {"1", "2", "3"}.issubset(set(by_source.keys())), (
        "三张源表都应该至少被一张输出表覆盖：seen={}".format(set(by_source.keys()))
    )

    for src in ("1", "2", "3"):
        outs = by_source[src]
        pairs = [(t.get("category"), t.get("caliberLabel")) for t in outs]
        assert len(outs) == 2, "源 id={} 应拆为合并/单体两张：{}".format(src, pairs)
        # 新契约：顶层 category 不带前缀，口径在 caliberLabel。
        consolidated = [p for p in pairs if p[1] == "合并"]
        standalone = [p for p in pairs if p[1] == "单体"]
        assert len(consolidated) == 1, (src, pairs)
        assert len(standalone) == 1, (src, pairs)
        # 同源拆出来的两张 category 必须相同（同一基础类型）。
        assert consolidated[0][0] == standalone[0][0], (src, pairs)
        # category 必须不带前缀。
        for cat, _ in pairs:
            assert "合并" not in str(cat) and "母公司" not in str(cat), (src, pairs)

    assert len(out) == 6, [(t.get("category"), t.get("caliberLabel")) for t in out]


def test_normalize_statement_label_preserves_pl_oci_scoped_titles() -> None:
    """「合并/母公司利润表及其他综合收益」须整体识别，不得误归并为「合并/母公司利润表」。"""
    from src.applications.parsing.service.three_statements_frontend_service import (
        _normalize_statement_label,
    )

    assert _normalize_statement_label("合并利润表及其他综合收益") == "合并利润表及其他综合收益"
    assert _normalize_statement_label("母公司利润表及其他综合收益") == "母公司利润表及其他综合收益"
    assert _normalize_statement_label("利润表及其他综合收益") == "利润表及其他综合收益"


def test_ords_id2_columns_caliber_after_banner_skip() -> None:
    """``infer_column_index_to_caliber_for_table`` 对 ORDS id=2 的列口径推断必须正确：
    col 1/3 → 合并，col 2/4 → 单体，不再因 banner 行被污染成「混合」。"""
    from src.applications.parsing.utils.financial_recognition_and_extraction_utils import (
        infer_column_index_to_caliber_for_table,
    )

    items = _load_ords_items()
    rows = next(it for it in items if it.get("element_id") == 2).get("rows") or []
    cb = infer_column_index_to_caliber_for_table(
        rows,
        table_caliber="合并",
        report_name="Statement of comprehensive income",
    )
    assert cb.get(1) == "合并", cb
    assert cb.get(2) == "单体", cb
    assert cb.get(3) == "合并", cb
    assert cb.get(4) == "单体", cb


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
