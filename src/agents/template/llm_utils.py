import traceback

from typing import List, Literal, Tuple
from pydantic import BaseModel, Field

from src.agents.content_generate_agent.utils import abatch_llm_with_prompt, abatch_llm_with_structured_output
from src.base.prompt_template_factory import PromptManager
from src.agents.template.utils import is_empty
from loguru import logger

class QuestionAnalysis(BaseModel):
    """问题类型判断和描述生成的结构化输出"""
    type: Literal['text', 'table','financial_table'] = Field(..., description="问题类型：text 或 table 或者 financial_table(财报表格类型)。输入文本<q_xx></q_xx>标签中内容有明显多行表格内容才分类为table,q标签中空或无明显表格特征都为text类型")
    description: str = Field(..., description="问题的简短描述，简洁明了地总结该问题的内容和要求")
    prompt: str = Field(..., description="问题详细提示信息，生成问题的提示信息，包含详细的内容要求、规则、要点和约束等")

class TocItem(BaseModel):
    """目录项结构（不使用行号版本）"""
    text: str = Field(..., description="标题文本内容")
    level: int = Field(..., description="目录层级，1表示一级标题，2表示二级标题，以此类推")

class TocExtractionResult(BaseModel):
    """目录提取结果"""
    tocs: List[TocItem] = Field(..., description="提取的目录结构列表，按在文档中出现的顺序排列")

async def abatch_question_analysis(
    texts: List[str], 
    model_id: str = "model_bailian_80b", 
    use_english: bool = False,
) -> List[QuestionAnalysis]:
    """
    异步批量分析问题类型和生成描述
    Args:
        texts: 输入文本列表（包含问题上下文的文本）
        model_id: 模型ID，默认为 "model_bailian_80b"
        use_english: 是否使用英文生成提示词，默认 False
    Returns:
        QuestionAnalysis 对象列表，包含类型判断和描述
    """
    # 构建提示词列表
    english_prompt_params = {
        "use_english_begin_tag": "本次任务要求的回答语言为英文，生成描述和详细提示信息都必须使用英文回答，Please respond in English\n",
        "use_english_end_tag": "- You MUST output strictly in the required JSON format\n- 且本次任务所有的描述信息以及详细提示信息都必须使用英文回答，plase respond in English",
        "use_english_tag": "# 生成语言要求（包括描述和详细提示信息）\n[Please respond in English]\n",
    }

    prompts = [
        PromptManager.getPrompt(
            templateId="template_question_analysis",
            param=(
                {"text": text, **english_prompt_params}
                if use_english
                else {
                    "text": text,
                    "use_english_begin_tag": "",
                    "use_english_end_tag": "",
                    "use_english_tag": "",
                }
            )
        ) for text in texts
    ]
    
    # 使用已有的批量方法
    try:
        return await abatch_llm_with_structured_output(
            prompts=prompts, 
            model_id=model_id, 
            response_type=QuestionAnalysis,
        )
    except Exception as e:
        logger.error(f"批量问题分析失败: {e}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise


async def abatch_toc_extraction(
    text_chunks: List[str],
    model_id: str = "model_bailian_80b",
) -> List[List[TocItem]]:
    """
    异步批量提取文档目录结构（使用原始文本，不依赖行号）
    
    Args:
        text_chunks: 文本块列表（每个块是文档的一部分）
        model_id: 模型ID，默认为 "model_bailian_80b"
    
    Returns:
        每个文本块的目录项列表，格式为 List[List[TocItem]]，每个内层列表对应一个文本块的提取结果
    """
    # 构建提示词列表
    prompts = [
        f"""你是专业的文档结构分析助手，擅长从文档中提取带层级的目录结构。

你将收到一段文档文本。你的任务是：

1. 识别文档中所有明显的标题/目录项（如章节标题，一级标题、二级标题、三级标题等）
2. 判断每个标题的层级（level）：
   - level 1: 一级标题（通常是文档的主要章节，如"第一部分 XXX"、"一、XXX"、"第一章 XXX"等）
   - level 2: 二级标题（章节下的子章节，如"一、XXX"、"1. XXX"、"（一）XXX"等）
   - level 3: 三级标题（更细分的章节，如"1. XXX"、"1.1 XXX"、"（1）XXX"等）
   - level 4......
   - 以此类推，根据标题的缩进、格式、编号等特征判断层级
3. 提取每个标题的完整文本内容

标题的典型特征（必须同时满足以下条件）：
- **长度限制**：标题必须很短，一般不超过30个字符（不包括编号部分）。如果文本超过50个字符，绝对不是标题
- **结构特征**：标题通常是名词短语或简短的主题描述，不是完整的句子
- **编号格式**：有明确的编号格式（如"一、二、三"、"1. 2. 3."、"（一）（二）"等），但编号后的内容必须简短
- **格式标记**：可能有格式标记（如Markdown的#、##、###，或加粗、下划线等）
- **独立性**：独立成行，前后有空行或换行
- **非句子性**：不包含完整的句子结构，不是描述性段落

**严格排除的内容（绝对不能提取为标题）**：
- **长段落文本**：任何超过50个字符的文本都不是标题，即使有编号
- **描述性句子**：完整的描述性句子，即使有编号（如"（1）流动资金贷款、银行承兑汇票..."这种长句）
- **空行**：内容为空或只有空白字符的行
- **表格行**：以"|"开头或包含多个"|"分隔符的行，如Markdown表格
- **普通段落**：完整的句子，没有标题特征
- **数据行**：包含大量数字、日期等数据的行
- **分隔线**：如"---"、"==="等

**判断标准**：
- 如果一段文本有编号但内容超过50个字符，它可能是带编号的段落，不是标题
- 如果一段文本是完整的句子（包含主谓宾结构），即使有编号，也不是标题
- 标题应该是简洁的主题概括，而不是详细的描述或说明
- 标题也可能没有编号，但是本身是简短的主题概括

重要要求：
- 必须严格按照文档中标题或者目录出现的顺序提取，并判断层级的level
- 必须正确判断层级关系，保证 level 的准确性（子标题的 level 必须大于父标题）
- **严格限制标题长度**：只提取明显简短的主题性标题，不要提取任何长段落
- 提取标题的完整文本内容，去除Markdown标记（如#、*等）但保留标题文本

输出格式为 JSON：
{{
  "tocs": [
    {{"text": "一、企业基本情况", "level": 1}},
    {{"text": "1. 企业规模", "level": 2}},
    ...
  ]
}}

注意：只输出目录结构（文本和层级），不要输出其他内容。**严格排除所有长段落、描述性句子和带编号的段落文本**。

文档文本：
{text_chunk}""" for text_chunk in text_chunks
    ]
    
    # 使用批量方法调用大模型
    results = await abatch_llm_with_structured_output(
        prompts=prompts,
        model_id=model_id,
        response_type=TocExtractionResult,
    )
    
    # 返回每个块的结果列表（不合并），保持与 toc_level 类似的逻辑
    chunk_results = [result.tocs for result in results]
    
    return chunk_results

async def abatch_extract_table_html(
    table_items: List[Tuple[object, str]],
    model_id: str = "model_bailian_80b",
) -> None:
    """
    批量提取表格的 HTML 代码
    Args:
        table_items: 元组列表，每个元组包含 (qdetail, q_html)
                    - qdetail: QuestionDetail 对象，需要修改其 html 和 type 属性
                    - q_html: 包含表格内容的 HTML 字符串（应该已经过滤过，确保不为空）
        model_id: 模型ID，默认为 "model_bailian_80b"
    """
    if not table_items:
        return
    
    logger.info(f"开始批量提取 {len(table_items)} 个表格的 HTML 代码...")
    
    # 构建提示词列表
    prompts = [
        f"""提取原文中的表格结构为可展示的html表格的表头代码，不需要要格式代码仅返回table的表头结果，同时不需要具体数据。

**重要要求**：
1. **只提取表格结构**：仅提取表头（保留表头信息），不要包含具体数据内容
2. **数据**：表头等起到引导填表作用的内容应该保留，具体的表格数据单元格中应该去除，不要提取原文中的具体数值、文本等数据
3. **无表格处理**：如果原文中**没有表格格式的数据**，必须返回字符串 "none_table"（不要返回空字符串或其他内容）

例如：
<table class="finance-table">
    <tr><th>项目</th><th></th><th>前三年（万元）</th><th>前两年（万元）</th><th>前一年（万元）</th><th>最近一期（万元）</th></tr>
    <tr class="category-row"><td rowspan="3">偿债能力</td><td>资产负债率(%)</td><td>91.18</td><td>88.47</td><td>75.16</td><td></td></tr>
    <tr><td>流动比率(%)</td><td>109.68</td><td>112.89</td><td>114.42</td><td></td></tr>
    <tr><td>速动比率(%)</td><td>140.62</td><td>155.16</td><td>71.40</td><td></td></tr>

    <tr class="category-row"><td rowspan="3">营运能力</td><td>应收账款周转率(次)</td><td>15.31</td><td>23.45</td><td>27.33</td><td></td></tr>
    <tr><td>存货周转率(次)</td><td>8.2</td><td>13.68</td><td>41.30</td><td></td></tr>
    <tr><td>总资产周转率(次)</td><td>1.75</td><td>3.17</td><td>3.4</td><td></td></tr>

    <tr class="category-row"><td rowspan="3">盈利能力</td><td>销售利润率(%)</td><td>0.23</td><td>0.87</td><td>1.01</td><td></td></tr>
    <tr><td>营业利润率(%)</td><td>0.23</td><td>0.87</td><td>1.0</td><td></td></tr>
    <tr><td>净利润率(%)</td><td>0.2</td><td>0.79</td><td>1.0</td><td></td></tr>
</table>

应该只提取下面部分：
<table> <tr> <th>项目</th> <th></th> <th>前三年（万元）</th> <th>前两年（万元）</th> <th>前一年（万元）</th> <th>最近一期（万元）</th> </tr> </table>

请提取下面原文中的表格结构：
{q_html}""" for _, q_html in table_items
    ]
    
    try:
        # 批量调用 LLM 提取 HTML
        html_results = await abatch_llm_with_prompt(
            prompts=prompts,
            model_id=model_id,
        )
        
        # 处理结果
        for idx, (qdetail, _) in enumerate(table_items):
            # 清理换行符
            extracted_html = html_results[idx].strip().replace("\n", "") if html_results[idx] else ""
            
            # 如果返回 "none_table" 或为空，将 type 改为 'text'
            if extracted_html.lower() == "none_table" or is_empty(extracted_html):
                logger.warning(f"问题 {qdetail.id} 原文中没有表格格式数据（返回: {extracted_html}），将 type 从 {qdetail.type} 改为 'text'")
                qdetail.type = 'text'
                qdetail.html = ""
            else:
                qdetail.html = extracted_html
        
        logger.info(f"批量提取 HTML 完成，共处理 {len(table_items)} 个表格")
        
    except Exception as e:
        logger.error(f"批量提取表格 HTML 失败: {str(e)}", exc_info=True)
        # 如果提取失败，将所有处理中的项目的 type 改为 'text'
        for qdetail, _ in table_items:
            logger.warning(f"问题 {qdetail.id} HTML 提取失败，将 type 改为 'text'")
            qdetail.type = 'text'
            qdetail.html = ""