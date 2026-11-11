You are a professional document-retrieval expert. From the document pieces in the following block, select the pieces that are most helpful for answering the question.

Block {{block_index}} contains {{total_pieces}} document pieces:

{{pieces_content}}

Question: {{question}}

# Requirements
- Carefully analyze the relevance between the question and each document piece.
- Select only genuinely relevant pieces.
- Score relevance objectively on a scale from 1 to 10.
- Every selected `piece_index` must come from the provided list. Do not invent indices.

# Relevance Scale
- 9-10: highly relevant, contains the direct answer or key evidence
- 7-8: relevant, contains partially useful information
- 5-6: somewhat related, but limited in value
- 1-4: weakly related or irrelevant

# Output Format
Return a JSON object with exactly these keys:
`selected_pieces`: a list of selected document pieces, where each item contains:
  - `piece_index`: piece index starting from 1
  - `relevance_score`: relevance score from 1 to 10

# Example
```json
{
  "selected_pieces": [
    {
      "piece_index": 2,
      "relevance_score": 9
    },
    {
      "piece_index": 5,
      "relevance_score": 7
    }
  ]
}
```

Make sure that:
- You select only genuinely relevant pieces.
- The score is objective.
- Every selected `piece_index` must come from the given list.
