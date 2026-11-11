"""
SN (问题位置) 识别模块
从标注后的文档文本中提取需要填写的问题位置 SN 编号
"""
from typing import List
from src.agents.content_generate_agent.utils import abatch_llm_with_structured_output
from loguru import logger
from src.agents.template.identify.identify_agent.identify_examples import identify_examples
from src.agents.template.identify.models.sn_items import SNItem, SNExtractionResult


async def abatch_sn_extraction(
    text_chunks: List[str],
    model_id: str = "model_bailian_80b",
) -> List[List[SNItem]]:
    """
    异步批量提取文档中的问题位置 SN 编号
    
    Args:
        text_chunks: 文本块列表（每个块是标注后的文档一部分，带有 #1, #2 等 SN 编号）
        model_id: 模型ID，默认为 "model_bailian_80b"
    
    Returns:
        每个文本块的问题位置 SN 列表，格式为 List[List[SNItem]]
    """
    # 构建提示词列表
    prompts = []
    for text_chunk in text_chunks:
        # 遍历示例列表，添加所有示例到提示词中
        examples_text = "\n\n".join(identify_examples)
        
        prompt = f"""你是专业的文档分析助手，擅长从已标注 SN 编号的文档中提取问题的位置。

你将收到一段报告模板的文档文本，其中已经用 #1, #2, #3 ... 的格式标注了每个段落/表格的位置，你需要理解文档的意图，选择其中问题对应的答案段落/表格。

你的任务是：
识别文档中所有问题对应的需要用户"填写"或"作答"的内容位置（这些位置可能是空的段落、示例的回答、回答的说明等），提取每个回答位置对应的 SN 编号（用开始SN和结束SN表示）;
整段文本中没有需要填写的内容时，返回空的 sn_items 列表。

输入文本说明
- 输入文本为文档文本，其中已经用 #1, #2, #3 ... 的格式标注了每个段落/表格的位置。
- 输入的文本可能是一个表格，需要判断该表格是需要循环添加内容（增加行）的动态信息表格，还是只需要填写的静态信息表格。
- 输入文本中可能以及包含了示例的回答文本，遇到示例回答文本时应该将整段示例回答都做为一个需要填写的内容。
- 输入文本一般是一个报告模板的部分文本，需要你选择需要填写的内容的sn位置，有的模板会有示例的回答。
- 输入文本中问题下面可能有示例的回答或者回答说明，需要选择开始段落和结束段落的sn作为答案位置
- 输入文本中可能有标题，章节标题，需要和需要填写的内容区分开。

需要提取的内容类型：
- 问题后如果没有空的行作为答案位置，则选择问题所在的段落作为答案位置
- 有示例回答的位置一般都是答案的位置，优先标记示例回答的位置作为答案位置
- 示例回答或者回答说明应该也是提取位置：示例回答或者回答说明的位置可能多个段落，需要选择开始段落和结束段落的sn作为答案位置
- 跨位置答案：一个问题的答案可能占据多个 SN 位置（如多段示例回答、多段问题解释说明）
- 循环的动态信息表格：当输入为明显的循环表格，需要循环行或者列不断填入内容时，选择开始和结束的sn作为答案位置，此时answer_type为'dynamic_table'
- 对于静态信息表格，有前单元格作为问题，后空白单元格或者示例答案单元格作为答案，所以选择后单元格的sn作为答案位置，answer_type为'static_table'。

重要规则：
- 使用 start_sn 和 end_sn 表示答案位置的范围，格式如 {{"start_sn": "#1", "end_sn": "#5"}}，start_sn 和 end_sn 相同时表示单个位置
- 如果某个 SN 位置没有需要填写的内容，不要提取，返回空的 sn_items 列表。
- 一些明显的章节标题，以及表格标题，不需要用户填写，不应该选择其对应的答案位置。
- 动态信息表格选择整个表格作为问题，start_sn和end_sn应该为表格的开始和结束的sn。
- 静态信息表格选择有前单元格作为问题，后空白单元格或者示例答案单元格作为答案的单元格作为问题，start_sn和end_sn应该为该单元格的sn。
- 如果输入为明显的信息表格，有前单元格作为问题，后单元格作为答案，所以选择后单元格的sn作为答案位置，answer_type为'static_table'。
- answer_type 必须为 'default' 或 'dynamic_table' 或 'static_table'
- 如果一个问题需要填写的内容跨越多个 SN 位置
    - 当所有sn答案位置的内容为空时，选择第一个sn作为 start_sn 和 end_sn
    - 当所有sn答案位置的内容都不为空，为示例的答案文本时，选择第一个sn作为 start_sn，最后一个sn作为 end_sn

输出格式为 JSON：
{{
  "sn_items": [
    {{"answer_type": "default", "start_sn": "#3", "end_sn": "#3"}},
    {{"answer_type": "dynamic_table", "start_sn": "#5", "end_sn": "#15"}},
    {{"answer_type": "static_table", "start_sn": "#23", "end_sn": "#23"}},
    ...
  ]
}}

提取示例：
{examples_text}

注意：只输出 SN 提取结果，不要输出其他内容。如果没有任何需要填写的内容，返回空的 sn_items 列表。

文档文本：
{text_chunk}"""
        prompts.append(prompt)
    
    # 使用批量方法调用大模型
    results = await abatch_llm_with_structured_output(
        prompts=prompts,
        model_id=model_id,
        response_type=SNExtractionResult,
    )
    
    # 返回每个块的结果列表
    chunk_results = [result.sn_items for result in results]
    
    return chunk_results


async def extract_sn_from_text(
    text: str,
    model_id: str = "model_bailian_80b"
) -> List[SNItem]:
    """
    从单个文本中提取问题位置 SN 编号（便捷函数）
    
    Args:
        text: 标注后的文档文本
        model_id: 模型ID
    
    Returns:
        SNItem 列表
    """
    results = await abatch_sn_extraction(
        text_chunks=[text],
        model_id=model_id,
    )
    return results[0] if results else []
