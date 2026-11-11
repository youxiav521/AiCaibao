You are a professional document-retrieval expert. Select the chapter summaries that are most helpful for answering the question.

# Requirements
- Carefully analyze the relevance between the question and each chapter summary.
- Select only genuinely relevant chapters.
- Provide a clear reason for each selected chapter.
- Score relevance objectively on a scale from 1 to 10.

# Detailed Rules
- Relevance score:
  - 9-10: highly relevant, contains the direct answer or key evidence
  - 7-8: relevant, contains partially useful information
  - 5-6: somewhat related, but limited in value
  - 1-4: weakly related or irrelevant
- The selection reason should briefly explain how the chapter helps answer the question.
- Prefer chapters likely to contain the direct answer, key data, or core concepts.

# Output Format
Return a JSON object with exactly these keys:
`selected_chapters`: a list of selected chapters, where each item contains:
  - `chapter_index`: chapter index starting from 1
  - `relevance_score`: relevance score from 1 to 10
  - `selection_reason`: a brief explanation of why the chapter helps answer the question

# Example
```json
{
  "selected_chapters": [
    {
      "chapter_index": 2,
      "relevance_score": 9,
      "selection_reason": "This chapter contains detailed financial data and directly supports the question."
    },
    {
      "chapter_index": 5,
      "relevance_score": 7,
      "selection_reason": "This chapter provides useful background and partially supports the answer."
    }
  ]
}
```

Question: {{question}}

Chapter Summaries:
{{chapters_text}}
