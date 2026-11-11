你需要判断：单个银行/金融文档中，某个必备章节或等价内容是否存在。

仅返回 JSON。

规则：
- 只判断当前文档是否已包含该章节或语义等价内容。
- 标题不同可以接受，但内容必须明确承担相同功能。
- 不要猜测；若片段不足，请返回 `status="uncertain"`。
- 明显缺失时，返回 `exists=false` 且 `status="fail"`。
- 明显存在时，返回 `exists=true` 且 `status="pass"`。
- 只要可能，`evidenceSegmentIds` 必须包含支撑判断的 segment id。

必备章节：
{section_name}

已知别名：
{aliases}

文档片段：
{document_excerpt}

输出结构：
{{
  "exists": true,
  "status": "pass",
  "matchedSectionName": "可选：匹配到的标题或语义标签",
  "reasoning": "简短理由",
  "confidence": 0.0,
  "evidenceSegmentIds": ["seg_1", "seg_2"]
}}
