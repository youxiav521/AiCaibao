from __future__ import annotations

import re


_LABELS = {
    "case.yaml": "用例配置",
    "candidate.yaml": "候选配置",
    "input": "输入文件",
    "source_input": "原始文件",
    "expected": "预期答案",
    "actual": "实际结果",
    "actual_items": "实际结果JSON",
    "actual_response": "接口原始响应",
    "metrics": "指标",
    "diff": "差异",
    "diff_detail": "差异明细",
    "html": "HTML报告",
    "md": "Markdown报告",
    "markdown": "Markdown报告",
    "json": "JSON报告",
    "log": "完整日志",
    "preview": "解析预览",
    "overlay": "坐标预览",
    "grid": "表格网格预览",
    "merge": "合并表格预览",
    "summary": "差异摘要",
    "preview_md": "解析预览MD",
    "actual_preview": "实际解析预览",
    "actual_overlay": "实际坐标预览",
    "expected_preview": "预期解析预览",
    "expected_overlay": "预期坐标预览",
    "diff_preview": "差异对比预览",
    "diff_overlay": "差异定位工作台",
    "table_diff_preview": "表格差异预览",
    "statement_grid_preview": "财报表格网格预览",
    "merge_tables_preview": "合并表格预览",
    "subject_match_table": "科目匹配完整模板预览",
    "subject_match_source_table": "科目匹配源文件预览",
    "subject_match_request": "科目匹配请求",
    "打开 dashboard": "打开总览",
    "打开完整日志": "完整日志",
    "打开 html 报告": "HTML报告",
    "打开 markdown 报告": "Markdown报告",
    "打开 json 报告": "JSON报告",
    "新窗口打开原文件": "打开原文件",
    "下载 pdf": "下载PDF",
}

_PREFIXES = {
    "candidate_": "候选·",
    "latest_": "上次测试·",
    "run_": "",
    "source_": "来源·",
}


def artifact_label(label: object) -> str:
    text = "" if label is None else str(label)
    if not text:
        return ""

    lower_text = text.lower()
    if text in _LABELS:
        return _LABELS[text]
    if lower_text in _LABELS:
        return _LABELS[lower_text]

    for prefix, display_prefix in _PREFIXES.items():
        if text.startswith(prefix):
            return display_prefix + artifact_label(text[len(prefix):])

    page_match = re.fullmatch(r"page_(\d+)", text)
    if page_match:
        return f"第{int(page_match.group(1))}页截图"

    return text.replace("_", " ")
