"""
标题层级优化工具 - 使用LLM辅助优化文档标题层次结构, minerU的逻辑
"""

import asyncio
import re
import json
from pathlib import Path
from loguru import logger
import json_repair

from src.agents.content_generate_agent.utils import (
    ainvoke_llm_with_prompt as content_generate_ainvoke_llm_with_prompt,
)

_DEFAULT_HEADING_AIDED_MODEL_ID = "model_bailian_heading_aided"

def full_to_half(text: str) -> str:
    """将全角字符转换为半角字符"""
    result = []
    for char in text:
        code = ord(char)
        # 全角字母和数字 (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # 转换到ASCII范围
        else:
            result.append(char)
    return ''.join(result)


def detect_language(text: str) -> str:
    """简单的语言检测，判断是否为中日韩语言"""
    if not text:
        return 'en'
    
    # 统计中日韩字符
    cjk_count = 0
    for char in text:
        code = ord(char)
        # 中文范围
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
            cjk_count += 1
        # 日文平假名和片假名
        elif (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
            cjk_count += 1
        # 韩文
        elif 0xAC00 <= code <= 0xD7AF:
            cjk_count += 1
    
    # 如果CJK字符占比超过30%，认为是CJK语言
    if len(text) > 0 and cjk_count / len(text) > 0.3:
        return 'zh'  # 简化处理，统一返回zh
    return 'en'


def is_hyphen_at_line_end(text: str) -> bool:
    """检查文本是否以字母+连字符结尾"""
    return bool(re.search(r'[A-Za-z]+-\s*$', text))


def escape_special_markdown_char(content: str) -> str:
    """转义Markdown特殊字符"""
    special_chars = ["*", "`", "~", "$"]
    for char in special_chars:
        content = content.replace(char, "\\" + char)
    return content


def merge_para_with_text(para_block: dict) -> str:
    """将段落块中的文本内容合并成一个连续的字符串"""
    if 'lines' not in para_block:
        return ''
    
    # 第一遍：提取所有文本用于语言检测
    block_text = ''
    for line in para_block['lines']:
        if 'spans' not in line:
            continue
        for span in line['spans']:
            span_type = span.get('type', 'text')
            if span_type in ['text', 'TEXT']:
                content = span.get('content', '')
                content = full_to_half(content)
                block_text += content
    
    # 检测语言
    block_lang = detect_language(block_text)
    
    # 第二遍：合并文本
    para_text = ''
    for i, line in enumerate(para_block['lines']):
        if 'spans' not in line:
            continue
        
        # 检查是否是列表起始行（如果有这个标记）
        if i >= 1 and line.get('is_list_start_line', False):
            para_text += '  \n'
        
        for j, span in enumerate(line['spans']):
            span_type = span.get('type', 'text')
            content = ''
            
            # 处理不同类型的内容
            if span_type in ['text', 'TEXT']:
                content = span.get('content', '')
                content = full_to_half(content)
                content = escape_special_markdown_char(content)
            
            elif span_type in ['inline_equation', 'INLINE_EQUATION']:
                if span.get('content', ''):
                    content = f"${span['content']}$"
            
            elif span_type in ['interline_equation', 'INTERLINE_EQUATION']:
                if span.get('content', ''):
                    content = f"\n$$\n{span['content']}\n$$\n"
            
            content = content.strip()
            
            if content:
                is_last_span = (j == len(line['spans']) - 1)
                
                # 根据语言决定是否添加空格
                if block_lang in ['zh', 'ja', 'ko']:
                    # 中日韩语言：一般不需要空格，但行内公式后需要
                    if is_last_span and span_type not in ['inline_equation', 'INLINE_EQUATION']:
                        para_text += content
                    else:
                        para_text += f'{content} '
                else:
                    # 西方语言：需要空格分隔
                    if span_type in ['text', 'TEXT', 'inline_equation', 'INLINE_EQUATION']:
                        # 处理行末连字符
                        if is_last_span and span_type in ['text', 'TEXT'] and is_hyphen_at_line_end(content):
                            para_text += content[:-1]  # 删除连字符
                        else:
                            para_text += f'{content} '
                    elif span_type in ['interline_equation', 'INTERLINE_EQUATION']:
                        para_text += content
    
    return para_text.strip()


async def optimize_title_hierarchy(middle_json: list) -> list:
    """使用LLM优化文档标题的层次结构"""
    model_id = _DEFAULT_HEADING_AIDED_MODEL_ID
    
    # 收集所有标题信息
    title_dict = {}
    origin_title_list = []
    title_index = 0
    
    for page_info in middle_json:
        if 'para_blocks' not in page_info:
            continue
        
        blocks = page_info["para_blocks"]
        for block in blocks:
            if block.get("type") != "title":
                continue
            
            # 保存原始标题块引用
            origin_title_list.append(block)
            
            # 提取标题文本
            title_text = merge_para_with_text(block)
            
            # 计算平均行高
            if 'line_avg_height' in block:
                line_avg_height = block['line_avg_height']
            else:
                title_block_line_height_list = []
                if 'lines' in block:
                    for line in block['lines']:
                        if 'bbox' in line:
                            bbox = line['bbox']
                            title_block_line_height_list.append(int(bbox[3] - bbox[1]))
                
                if len(title_block_line_height_list) > 0:
                    line_avg_height = sum(title_block_line_height_list) / len(title_block_line_height_list)
                elif 'bbox' in block:
                    line_avg_height = int(block['bbox'][3] - block['bbox'][1])
                else:
                    line_avg_height = 20  # 默认值
            
            # 获取页码
            page_idx = int(page_info.get('page_idx', 0)) + 1
            
            # 存储标题信息
            title_dict[str(title_index)] = [title_text, line_avg_height, page_idx]
            title_index += 1
    
    # 如果没有标题，直接返回
    if not title_dict:
        logger.info("未找到任何标题块")
        return middle_json
    
    logger.info(f"找到 {len(title_dict)} 个标题，开始优化层次结构...")
    
    # 构建LLM提示词
    title_optimize_prompt = f"""输入的内容是一篇文档中所有标题组成的字典，请根据以下指南优化标题的结果，使结果符合正常文档的层次结构：

1. 字典中每个value均为一个list，包含以下元素：
    - 标题文本
    - 文本行高是标题所在块的平均行高
    - 标题所在的页码

2. 保留原始内容：
    - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
    - 请务必保证输出的字典中元素的数量和输入的数量一致

3. 保持字典内key-value的对应关系不变

4. 优化层次结构：
    - 为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题从前至后的层级必须是连续的，不能跳过层级
    - 标题层级最多为5级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息

5. 合理性检查与微调：
    - 在完成初步分级后，仔细检查分级结果的合理性
    - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
    - 确保最终的分级结果符合文档的实际结构和逻辑
    - 字典中可能包含被误当成标题的正文，你可以通过将其层级标记为 0 来排除它们

IMPORTANT: 
请直接返回优化过的由标题层级组成的字典，格式为{{标题id:标题层级}}，如下：
{{
  "0":1,
  "1":2,
  "2":2,
  "3":3
}}
不需要对字典格式化，不需要返回任何其他信息。

Input title list:
{title_dict}

Corrected title list:
"""
    
    # 重试机制
    retry_count = 0
    max_retries = 3
    dict_completion = None
    
    while retry_count < max_retries:
        try:
            content = str(
                await content_generate_ainvoke_llm_with_prompt(
                    title_optimize_prompt,
                    model_id=model_id,
                )
                or ""
            ).strip()
            
            # 移除可能的思考标签（某些模型会输出）
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[idx:].strip()
            
            # 解析JSON（使用json_repair处理可能不完整的JSON）
            dict_completion = json_repair.loads(content)
            dict_completion = {int(k): int(v) for k, v in dict_completion.items()}
            
            # 验证结果数量是否一致
            if len(dict_completion) == len(title_dict):
                # 更新原始标题块的层级
                for i, origin_title_block in enumerate(origin_title_list):
                    origin_title_block["level"] = int(dict_completion[i])
                
                logger.info(f"标题层次结构优化成功，共处理 {len(origin_title_list)} 个标题")
                break
            else:
                logger.warning(
                    f"LLM返回的标题数量({len(dict_completion)})与输入数量({len(title_dict)})不一致，重试..."
                )
                retry_count += 1
        
        except Exception as e:
            logger.error(f"处理标题时出错: {e}")
            if retry_count < max_retries - 1:
                delay = 2.0 * (2 ** retry_count)  # 指数退避: 2s, 4s, 8s
                logger.warning(f"标题分级失败，{delay:.1f}s 后重试 ({retry_count + 1}/{max_retries})")
                await asyncio.sleep(delay)
            retry_count += 1
    
    # 检查是否成功
    if dict_completion is None:
        logger.error("达到最大重试次数后仍未成功，标题层级未被优化")
    
    return middle_json


async def llm_aided_title(pdf_info: list) -> None:
    """使用LLM辅助优化标题层级"""
    await optimize_title_hierarchy(pdf_info)
