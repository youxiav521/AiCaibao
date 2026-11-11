from typing import Iterable

try:
    from .schemas import NormalizedChunk, PolicyDocument, SectionExtractionResult
except ImportError:  # pragma: no cover
    from schemas import NormalizedChunk, PolicyDocument, SectionExtractionResult


SECTION_SYSTEM_PROMPT = """你是银行政策规则抽取助手。请仅基于输入文本抽取 requirement 与 control。
要求：
1. 不得编造，信息不足时返回空数组。
2. requirement 是完整政策要求；control 是可执行、可检查的最小控制点。
3. 对并列条款优先拆成多个 control；必要时拆成多个 requirement。
4. 去重：语义重复内容只保留一条。
5. 输出内容使用中文。
6. control 必须与 sourceText 语义一致，不得补充 sourceText 未出现的约束条件（如“合法有效”“符合审批内容”“加盖公章”）。
7. 对目的、原则、定义、适用范围等概括性条款，可仅输出 requirement，controls 允许为空。
字段约束：
- controlType 只能是：
  qualification_check
  document_check
  quantitative_check
  fund_use_check
  collateral_check
  prohibited_item_check
  monitoring_check
- risk 只能是 HIGH / MEDIUM / LOW
- logicDescription 要写成可执行检查说明，尽量包含“检查对象 + 检查动作 + 判定标准 + 触发时点”中的至少三项。
输出必须是严格 JSON，且只能使用下面结构：
{
  "sectionSummary": "string",
  "requirements": [
    {
      "title": "string",
      "description": "string",
      "sourceText": "string",
      "controls": [
        {
          "content": "string",
          "controlType": "qualification_check|document_check|quantitative_check|fund_use_check|collateral_check|prohibited_item_check|monitoring_check",
          "risk": "HIGH|MEDIUM|LOW",
          "sourceText": "string",
          "logicDescription": "string"
        }
      ]
    }
  ]
}
"""


FRAMEWORK_SYSTEM_PROMPT = """你是银行政策框架摘要助手。
基于文档名、章节标题和 requirement 摘要，生成 framework 标题与简介。
要求：简洁、客观、不编造。
只输出 JSON，结构如下：
{
  "title": "string",
  "summary": "string"
}
"""


def build_section_messages(section_index: int, chunks: Iterable[NormalizedChunk]) -> list[tuple[str, str]]:
    """Build system+human messages for one section extraction call."""
    return [
        ("system", SECTION_SYSTEM_PROMPT),
        (
            "human",
            "请抽取下面这个政策章节的 requirement 和 control。\n\n"
            + f"[section_index]\n{section_index}\n\n"
            + "[chunks]\n"
            + _render_chunks(chunks),
        ),
    ]


def build_framework_messages(
    document: PolicyDocument,
    groups: list[list[NormalizedChunk]],
    section_results: list[SectionExtractionResult],
) -> list[tuple[str, str]]:
    """Build framework summarization prompt from extracted section results."""
    section_titles = []
    for index, group in enumerate(groups, start=1):
        heading = group[0].heading_path[-1] if group and group[0].heading_path else f"Section {index}"
        section_titles.append(f"{index}. {heading}")

    requirement_summaries = []
    for section in section_results:
        for requirement in section.requirements:
            requirement_summaries.append(f"- {requirement.title}: {requirement.description}")

    return [
        ("system", FRAMEWORK_SYSTEM_PROMPT),
        (
            "human",
            "请基于以下信息生成 framework 标题与描述。\n\n"
            + f"[document_name]\n{document.file_name}\n\n"
            + "[section_titles]\n"
            + "\n".join(section_titles[:80])
            + "\n\n[requirement_summaries]\n"
            + "\n".join(requirement_summaries[:200]),
        ),
    ]


def _render_chunks(chunks: Iterable[NormalizedChunk]) -> str:
    """Render chunks into compact prompt text for section extraction.

    Keep only minimal metadata required for traceability and extraction quality.
    """
    parts: list[str] = []
    for chunk in chunks:
        heading = " / ".join(chunk.heading_path) if chunk.heading_path else "ROOT"
        parts.append(
            "\n".join(
                [
                    # element_id is used later to map extracted controls back to source.
                    f"[element_id] {chunk.source_child_id}",
                    f"[heading] {heading}",
                    "[text]",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(parts)
