"""SP Setia 同一页左右对开两半 BS：必须独立两半，各自做合并/单体拆分输出。"""

import json
from pathlib import Path

from src.applications.parsing.service.three_statements_frontend_service import (
    extract_three_statements_frontend,
    _looks_like_independent_balance_sheet_halves,
)


def _load_sp_setia_items() -> list:
    root = Path(__file__).resolve().parents[2]
    p = root / (
        "storage/mineru_output/converted_output/"
        "9cd3a8ebe8f14746b8cfb9b26f09628bPvFhqQgEp1_mineru_converted.json"
    )
    assert p.is_file(), "fixture missing: {}".format(p)
    return json.loads(p.read_text(encoding="utf-8"))


def test_sp_setia_bs_halves_kept_independent_and_caliber_split() -> None:
    items = _load_sp_setia_items()
    out = extract_three_statements_frontend(items, locale=None)

    # 新契约：顶层 ``category`` 不再带「合并/母公司」前缀，口径放在独立的
    # ``caliberLabel`` 字段中。期望 7 张表：BS 两半 × 2（合并/母公司）
    # + IS × 2 + 合并 SOCE × 1 = 7。
    assert len(out) >= 7, "至少应有 7 张表，实际 {}：{}".format(
        len(out), [(t.get("category"), t.get("caliberLabel")) for t in out]
    )

    def _is(t, base_cat, caliber):
        return t.get("category") == base_cat and t.get("caliberLabel") == caliber

    # 顶层 ``category`` 已经归并到三大主表枚举：
    #   - SOCE（所有者权益变动表） → "资产负债表"
    #   - 利润表及其他综合收益 → "利润表"
    # 因此「资产负债表-合并」会包含 BS 两半 + 合并 SOCE，共 3 张。
    bs_consolidated = [t for t in out if _is(t, "资产负债表", "合并")]
    bs_company = [t for t in out if _is(t, "资产负债表", "单体")]
    is_consolidated = [t for t in out if _is(t, "利润表", "合并")]
    is_company = [t for t in out if _is(t, "利润表", "单体")]
    # SOCE 在归并后 category="资产负债表"，按 mergeIds=["11"] 精准识别。
    soce_consolidated = [
        t for t in out
        if t.get("mergeIds") == ["11"]
        and t.get("category") == "资产负债表"
        and t.get("caliberLabel") == "合并"
    ]

    # BS 两半各自拆 2 份 => 合并 BS 来自 id=5/id=6 共 2 张，再加 SOCE 合并(id=11) = 3 张。
    assert len(bs_consolidated) == 3, [t.get("mergeIds") for t in bs_consolidated]
    assert len(bs_company) == 2, [t.get("mergeIds") for t in bs_company]
    assert len(is_consolidated) == 1
    assert len(is_company) == 1
    # SOCE 单口径（合并 only），与 BS/IS 是否带前缀互不干扰，不能再被「跨基础类型互斥」误删。
    assert len(soce_consolidated) == 1, [
        (t.get("category"), t.get("caliberLabel"), t.get("mergeIds")) for t in out
    ]
    soce = soce_consolidated[0]
    # 顶层 category 不能再带前缀；caliberLabel 必须是受控枚举之一。
    for t in out:
        assert "合并" not in str(t.get("category")) and "母公司" not in str(t.get("category")), (
            "顶层 category 不应再带口径前缀：{}".format(t.get("category"))
        )
        assert t.get("caliberLabel") in {"合并", "单体", "混合", "未知"}, (
            "顶层 caliberLabel 字段缺失或值非法：{}".format(t.get("caliberLabel"))
        )
    # SOCE 输出字段结构应与 BS/IS 一致：每个 cell 都有 statementType / gaapType / caliberLabel / accountingYear。
    rows = soce.get("rows") or []
    assert rows, "SOCE rows 不应为空"
    for row in rows:
        cells = row if isinstance(row, list) else row.get("cells", [])
        for c in cells:
            if not isinstance(c, dict):
                continue
            assert "statementType" in c
            assert "gaapType" in c
            assert "caliberLabel" in c
            assert "accountingYear" in c

    # 每个 合并BS 必须只来自一个独立的 MinerU 元素（不再合并成一张）。
    bs_cons_merge_ids = {tuple(t.get("mergeIds") or ()) for t in bs_consolidated}
    assert ("5",) in bs_cons_merge_ids
    assert ("6",) in bs_cons_merge_ids

    def _row_texts(t):
        out_texts = []
        for row in t.get("rows") or []:
            cells = row if isinstance(row, list) else row.get("cells", [])
            for c in cells:
                if isinstance(c, dict):
                    text = (c.get("text") or "").strip()
                    if text:
                        out_texts.append(text)
        return out_texts

    bs_assets_half = next(t for t in bs_consolidated if t.get("mergeIds") == ["5"])
    bs_eql_half = next(t for t in bs_consolidated if t.get("mergeIds") == ["6"])
    assert bs_assets_half.get("category") == "资产负债表"
    assert bs_assets_half.get("caliberLabel") == "合并"

    assert "TOTAL ASSETS" in _row_texts(bs_assets_half), "Assets 半张应保留 TOTAL ASSETS"
    assert "EQUITY AND LIABILITIES" in _row_texts(bs_eql_half), (
        "E&L 半张应保留 EQUITY AND LIABILITIES 章节标题"
    )
    assert "TOTAL EQUITY AND LIABILITIES" in _row_texts(bs_eql_half)

    # 章节不能跨半张混入：Assets 半张里不应出现 EQUITY AND LIABILITIES 等。
    assets_text_blob = "\n".join(_row_texts(bs_assets_half))
    assert "EQUITY AND LIABILITIES" not in assets_text_blob


def test_helper_detects_independent_halves_and_skips_for_continuation() -> None:
    items = _load_sp_setia_items()
    el5 = next(x for x in items if x.get("element_id") == 5)
    el6 = next(x for x in items if x.get("element_id") == 6)

    cand5 = {
        "label": "资产负债表",
        "table": el5,
        "expanded_idx": 4,
        "page": el5.get("page"),
    }
    cand6 = {
        "label": "资产负债表",
        "table": el6,
        "expanded_idx": 5,
        "page": el6.get("page"),
    }
    assert _looks_like_independent_balance_sheet_halves(cand5, cand6) is True

    # 跨页（page 1 -> page 2）则不是「同页对开两半」，不应触发独立判定。
    cand6_cross = dict(cand6, page=2)
    assert _looks_like_independent_balance_sheet_halves(cand5, cand6_cross) is False


if __name__ == "__main__":
    test_helper_detects_independent_halves_and_skips_for_continuation()
    test_sp_setia_bs_halves_kept_independent_and_caliber_split()
    print("ok")
