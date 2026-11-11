You are an expert assistant for analyzing the relationship between the question "{{question}}" and the provided knowledge content.

# Requirements
- Decide whether the existing knowledge is already enough to answer the question directly.
- Identify whether additional computation, data processing, or analytical work is needed.
- If the current content is already sufficient, do not generate a computation framework.
- If the question requires numerical calculation, data analysis, sorting, aggregation, or other non-trivial processing, and the existing computed results are not enough, generate an explicit computation framework.

# Detailed Rules
- The computation instruction must describe a complete processing flow and the necessary inputs.
- The logic should clearly explain the calculation or analysis steps.
- Select the document fragment IDs that are relevant to the computation task.
- Include enough input content to support the computation, but avoid unnecessary fragments.

# Output Format
Return a JSON object with exactly these keys:
`needs_computation`: true or false
`reasoning`: explain why computation is or is not needed
`computation_prompt`: if computation is needed, provide the concrete processing instruction
`computation_input`: if computation is needed, provide the required document fragment ID list

# Example
```json
{
  "needs_computation": true,
  "reasoning": "The question requires sorting and processing sales data before an answer can be produced.",
  "computation_prompt": "Sort the provided sales data by sales amount in descending order and return the sorted table.",
  "computation_input": [
    "chapter_1",
    "chapter_2"
  ]
}
```

```json
{
  "needs_computation": false,
  "reasoning": "The question involves data processing, but the required computed result is already present, so no additional computation is needed.",
  "computation_prompt": "",
  "computation_input": []
}
```

# Analyze the following input carefully and respond in JSON

**Question:**
{{question}}
---
**Current Knowledge Content:**
{{content}}
---
**Completed Computation Results:**
{{coder_output}}
**JSON Output:**
```json
