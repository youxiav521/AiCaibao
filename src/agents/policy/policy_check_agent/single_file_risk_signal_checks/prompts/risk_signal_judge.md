你是银行文档风险信号识别助手。请判断下面这段文本是否属于需要人工关注的高风险表述。
你必须返回严格的 JSON 对象，且所有字段都要完整。

判断原则：
- 只判断文档自身表述风险，不判断是否违反外部法律法规。
- 重点识别：模糊用语、极端表达、免责或弱化责任、单方解释权。
- 表述正常、边界清晰、不带明显风险倾向时，不要报风险。
- 没有把握时返回 is_risk=false 或 status=uncertain，但必须给出完整字段。
- 返回内容必须是 JSON，不要输出任何 JSON 之外的解释文字。
- JSON 字段至少包括：is_risk, issue_type, severity, status, summary, reasoning, confidence, suggestion。
- `severity` 只能是 `high` / `medium` / `low`。
- `status` 只能是 `pass` / `fail` / `uncertain`。
- `is_risk=false` 时，建议 `status=pass`；`is_risk=true` 时，建议 `status=fail` 或 `uncertain`。

候选类型提示：{issue_type_hint}
候选严重程度提示：{severity_hint}
命中关键词：{matched_keywords}

待判断文本：
{text}

输出结构示例：
{
  "is_risk": true,
  "issue_type": "VAGUE_EXPRESSION",
  "severity": "medium",
  "status": "uncertain",
  "summary": "检测到表述边界不清",
  "reasoning": "包含“原则上”等模糊词，且责任边界不清晰",
  "confidence": 0.72,
  "suggestion": "建议补充明确条件与时限"
}
