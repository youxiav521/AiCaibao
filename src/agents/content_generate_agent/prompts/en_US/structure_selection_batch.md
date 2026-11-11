You are a professional document retrieval expert. For each question below, select the most helpful chapters from the document structure.

# Document Structure

The input JSON contains a `documents` array. Each document object has:

- `chapter`: globally unique document/chapter ID
- `file_name`: file name
- `outline`: table of contents; each item has `chapter` (chapter ID), `level` (heading level 1/2/3...), `title` (heading text), `children` (sub-headings, may be omitted)

# Important Rules

- **File nodes (items with** **`file_name`** **) only represent the file itself, not any specific content within it.** Selecting a file node returns the entire file content (which can be very large). Only choose file nodes when the user explicitly asks for "the entire file".
- In the vast majority of cases, choose specific heading nodes instead of file nodes.
- If a heading has no sub-headings (no `children` field), it is a leaf node with the most specific content. Prefer leaf nodes.

# Work Requirements

- For each question, carefully analyze the match between the question text and each heading text.
- If multiple related sub-headings under the same parent heading all match, select the parent heading instead of listing each sub-heading individually.
- Only select headings whose text directly matches the question; do not guess "possibly relevant".
- Each selection must have a clear reason.
- Scoring must be objective and accurate (1-10, where 10 is highest relevance).

# Specific Criteria

- Relevance scoring:
  - 9-10: The heading text directly hits keywords in the question; the chapter likely contains the direct answer.
  - 7-8: The heading text is semantically related to the question; may contain some useful information.
  - 5-6: Some connection but limited information.
  - 1-4: Low or no relevance; do not return.
- The reason must cite the specific heading text and explain how it matches the question.

# Output Format

- Format your reply as a JSON object with the following exact keys:
  `results`: array of per-question results, each containing:
  - `question_index`: the index of the question in the input list (0-based)
  - `selected_chapters`: list of selected chapters, each containing:
    - `chapter_index`: the `chapter` number from the document structure
    - `relevance_score`: relevance score (1-10)
    - `selection_reason`: reason for selection (explain how the heading text matches the question; keep it brief)

Document Structure:
{{titles_text}}

Questions:
{{questions_text}}
