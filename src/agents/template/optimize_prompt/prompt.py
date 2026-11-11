"""
@Time    : 2026/04/07
@Desc    : 提示词优化模板
提供中英文版本的优化提示词模板
"""

OPTIMIZATION_PROMPT_CHINESE = """你是一个专业的提示词工程专家，擅长优化和改进AI提示词。

## 任务
请参照要解决的问题，对以下原始提示词进行优化，详细对齐要解决的问题和提示词中的描述，生成一个优化版本。

## 任务要求
1. 添加更详细和明确的指令或者要求，本版本应该比原始提示词更加：
   - 清晰明确：指令更具体，减少歧义
   - 结构完整：包含必要的上下文和约束条件
   - 效果更好：更容易让AI理解并产生高质量输出
2. 保持原始提示词解决问题的核心意图不变，不要偏离原提示词要解决的问题

## 输出格式
直接返回优化后的提示词文本，不要包含任何额外的说明、JSON格式或其他格式标记。
只返回优化后的提示词内容本身。

## 优化要求
- 优化提示词时，注意参照原提示词要解决的问题，新提示词应该遵循需要解决的问题
- 添加更详细的描述来补充原始提示词中的描述，不要偏离原提示词要解决的问题
- 要求优化后的提示词中要求的描述要符合要解决的问题中所有的要点
- 注意匹配要解决的问题中的要点和原始提示词中的要点，有遗漏的要点描述应该添加，有多余的要点描述应该删除
- **问题涉及金额时，详细提示信息中都应该包含具体金额以及单位，并包含一个回答的示例在详细提示词中，例如："输出金额时需要携带单位：例如: 12345万元"**

例如：
要解决的问题为：申请人的完整企业名称和统一社会信用代码，应为该公司依法注册的全称
原始提示词为：请填写申请人的完整企业名称，该名称必须与工商登记注册的法定全称完全一致
则优化时应该添加与统一社会信用代码相关的描述

要解决的问题为：申请人的完整企业名称，应为该公司依法注册的全称
原始提示词为：请填写申请人的完整企业名称和统一社会信用代码，该名称必须与工商登记注册的法定全称完全一致
则优化时应该删除与统一社会信用代码相关的描述

优化重点：{optimization_focus}，本版本特别关注：{focus}
## 问题和原始提示词如下
{original_prompt}

请使用中文回答所有描述和详细提示信息。
"""

OPTIMIZATION_PROMPT_ENGLISH = """You are a professional prompt engineer expert, specializing in optimizing and improving AI prompts.

## Task
Please optimize the following original prompt by carefully aligning the problem to be solved with the descriptions in the prompt, and generate an optimized version.

## Task Requirements
1. Add more detailed and explicit instructions or requirements. This version should be more:
   - Clear and precise: instructions should be more specific with less ambiguity
   - Structurally complete: include necessary context and constraints
   - More effective: easier for AI to understand and generate high-quality output
2. Maintain the core intent of the original prompt's problem-solving purpose, do not deviate from the original problem

## Output Format
Return the optimized prompt text directly, without any additional explanations, JSON format, or other formatting markers.
Only return the optimized prompt content itself.

## Optimization Requirements
- When optimizing prompts, refer to the problem to be solved. The new prompt should follow the problem that needs to be solved
- Add more detailed descriptions to supplement the original prompt's descriptions without deviating from the problem
- Ensure that the descriptions required in the optimized prompt match all the key points in the problem to be solved
- Pay attention to matching the key points in the problem with the key points in the original prompt. Add descriptions for any missing key points and remove any extra key points
- **When the problem involves amounts, the detailed prompts should include specific amounts with units, and include an example answer in the prompt, e.g., "When outputting amounts, include units: for example: 12345万元 (ten thousand yuan)"**

Examples:
Problem: The complete company name and unified social credit code of the applicant, should be the full legal name registered by the company
Original prompt: Please fill in the complete company name, which must exactly match the legal full name registered with the business administration
In this case, you should add descriptions related to the unified social credit code

Problem: The complete company name of the applicant, should be the full legal name registered by the company
Original prompt: Please fill in the complete company name and unified social credit code of the applicant, which must exactly match the legal full name registered with the business administration
In this case, you should remove descriptions related to the unified social credit code

Optimization focus: {optimization_focus}, this version focuses particularly on: {focus}

## Problem and Original Prompt
{original_prompt}

Please respond in English for all descriptions and detailed prompts in the optimized output.
"""


def get_optimization_prompt(
    optimization_focus: str,
    focus: str,
    original_prompt: str,
    use_english: bool = False
) -> str:
    """
    根据语言设置获取对应的优化提示词模板
    
    Args:
        optimization_focus: 优化重点
        focus: 当前版本的特别关注点
        original_prompt: 原始提示词
        use_english: 是否使用英文版本
    
    Returns:
        格式化后的优化提示词
    """
    if use_english:
        template = OPTIMIZATION_PROMPT_ENGLISH
    else:
        template = OPTIMIZATION_PROMPT_CHINESE
    
    return template.format(
        optimization_focus=optimization_focus,
        focus=focus,
        original_prompt=original_prompt
    )
