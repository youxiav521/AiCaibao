import json
import os
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import traceback

from ...utils.model_invoke_tool import ainvoke_model_with_structured_output
from ...models.document_model import Document
from ..state import ExtractState
from src.base.prompt_template_factory import PromptManager
from ..state import language_guidance
from loguru import logger


class ExtractResult(BaseModel):
    """单个提取结果模型"""
    description:str = Field(..., description="描述这段数据提取的是什么，如何用来解决用户提出的问题问题")
    document_name: str = Field(..., description="提取的文档的原始标题名称，用来指代这段数据提取自那一段文本数据")
    extract_data: Any = Field(..., description="提取的数据，可以是一段文本，或者结构化的方式将问题相关的数据提取出来")

class ExtractResults(BaseModel):
    """多个提取结果"""
    results: List[ExtractResult] = Field(...,description="多段提取结果的列表")

def merge_extract_results_to_documents(
    results: ExtractResults,
    relative_documents: List[Document],
) -> List[Document]:
    """
    将提取结果合并到文档列表中，返回新的 List[Document]（不修改入参）。
    逻辑：
    1. 第一次出现某个 title 时，尝试在 relative_documents 中匹配并覆盖该 document 的 table/text
    2. 如果第一次没匹配到，创建新 Document
    3. 后续再出现同一个 title 时，将 title 重命名为 title_x 并追加到列表
    """
    if not hasattr(results, "results") or not results.results:
        return list(relative_documents)

    processed_titles = set()
    title_counters = {}
    output: List[Document] = []
    used_indices = set()

    for extract_result in results.results:
        doc_name = extract_result.document_name
        extract_data_json = json.dumps(extract_result.extract_data, ensure_ascii=False)
        matched = False

        if doc_name in processed_titles:
            title_counters[doc_name] = title_counters.get(doc_name, 0) + 1
            new_title = f"{doc_name}_{title_counters[doc_name]}"
            output.append(
                Document(
                    id="",
                    title=new_title,
                    text=extract_result.description,
                    table=extract_data_json,
                    file_id=None,
                    element_ids=[],
                )
            )
            matched = True
        else:
            for i, doc in enumerate(relative_documents):
                if i in used_indices:
                    continue
                if str(doc.title) == str(doc_name):
                    output.append(
                        doc.model_copy(
                            update={"table": extract_data_json, "text": extract_result.description}
                        )
                    )
                    used_indices.add(i)
                    processed_titles.add(doc_name)
                    matched = True
                    break
            if not matched:
                output.append(
                    Document(
                        id="",
                        title=doc_name,
                        text=extract_result.description,
                        table=extract_data_json,
                        file_id=None,
                        element_ids=[],
                    )
                )
                processed_titles.add(doc_name)

    for i, doc in enumerate(relative_documents):
        if i not in used_indices:
            output.append(doc)
    return output

# 遍历 relative_documents_list，将只有 text 有值而 table 为空的 document 单独筛选
def split_documents_node(state: ExtractState):
    """
    将 relative_documents_list 中的 document 按 text/table 是否为空分类。
    只有 text 有值且 table 为空的 document 需要数据提取。
    state 中 relative_documents_list 为 List[Document]，反序列化后可能为 dict，此处统一转为 Document。
    """
    raw_list = state.get("relative_documents_list", [])
    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in raw_list]
    text_documents = []
    table_documents = []
    for doc in docs:
        has_text = bool(doc.text and str(doc.text).strip())
        has_table = bool(doc.table and str(doc.table).strip())
        if has_text and not has_table:
            text_documents.append(doc.to_dict())
        elif has_table:
            table_documents.append(doc.to_dict())
    state["text_documents"] = text_documents
    state["table_documents"] = table_documents
    return state

def judge_need_extract_node(state: ExtractState):
    """
    判断是否需要提取数据：
    只有 text 字段有值且 table 字段没有值的 document 需要数据提取
    """
    text_documents = state.get("text_documents", [])
    if text_documents and len(text_documents) > 0:
        return "extract_node"
    logger.info(f"---judge_need_extract_node---: 不需要提取数据，text_documents({len(text_documents)})")
    return "END"

async def extract_node(state: ExtractState):
    """数据提取节点"""
    logger.info("---extract_node---: 数据提取...")
    question_description = state["question_description"]
    user_prompt = state.get("user_prompt", "")
    raw_list = state.get("relative_documents_list", [])
    # state 中为 List[Document]，反序列化后可能为 dict
    relative_documents = [Document.from_dict(d) if isinstance(d, dict) else d for d in raw_list]
    text_documents = state.get("text_documents", [])
    need_calculate_metrics = state.get("need_calculate_metrics", {})

    context_str = "\n".join(
        [f"相关上下文标题:{d.get('title', '')}\n相关上下文内容:{d.get('text', '')}" for d in text_documents]
    )
    answer_language = state.get("answer_language", "zh_CN")
    language_instruction = language_guidance(answer_language)
    answer_system_prompt = PromptManager.getPrompt(
        templateId="data_analyst_extract",
        param={
            "context_str": context_str,
            "question":question_description,
            "user_prompt":user_prompt,
            "need_calculate_metrics":need_calculate_metrics,
            "answer_language_instruction": language_instruction
        }
    )
    system_message = SystemMessage(content=answer_system_prompt)
    # 从模板中提取用户消息部分（对应 prompt 模板 142-163 行）
    user_message_content = f"""以下是用户真实的问题，也是你此次数据提取的主要任务，请分析用户真实的问题，按照要求提取相关数据，保证相关数据完整，并能正确支持计算和回答用户的真实问题
相关上下文数据如下
{context_str}

用户问题：
{question_description}

用户提示词：
{user_prompt}

相关计算需要：
{need_calculate_metrics}

注意只提取真实上下文中与问题相关的数据，不要进行任何单位转换，不要进行任何数据处理
你不是要直接按照问题要求回答，而是分析原文中与问题相关（如果有相关计算需要也应该满足相关计算需要）的数据，将这些数据抽取出来
抽取时应该保证数据完整且正确，并要求正确提取不同年份的相关数据，**保证数据完整，不遗漏重要信息**。**重要：当组织extract_data时，如果数据包含多年份，必须按照项目/指标分组，将不同年份作为不同的列（例如"2021年度"、"2022年度"、"2023年度"作为列名），每个项目一行，这样可以直接被pandas DataFrame读取为表格格式。如果某个项目在某一年份没有数据，该年份字段应该为空字符串""或null，不要省略该字段**
只提取原文中存在的数据，不要幻觉数据或者自行推理计算数据（例如合计等数据项目，相关原上下文中存在时可以提取，否则认为需要计算，不进行提取）
你不需要按照问题的要求回答问题，而是从相关原上下文中找出与问题相关的数据，并按照数据文本描述做结构化数据抽取
提取的所有数据都严格要求是原上下文中包含的数据，不应该自行进行计算或者单位转换，不存在于原上下文中的数据不要进行提取
用户问题中可能包含不在原上下文中的数据，这类数据不要自行捏造或者是自行计算，应当跳过这种不在上下文中出现的数据，不进行提取
请按照要求提取问题相关的结构化数据，并输出要求的 JSON 格式数据
{language_instruction}
"""
    user_message = HumanMessage(content=user_message_content)
    # 与 coding_agent_nodes 一致：将 history_messages 拼到 system 后
    history_messages = state.get("history_messages", []) or []
    messages = [system_message, user_message] + list(history_messages)

    try:
        results, model_time = await ainvoke_model_with_structured_output(messages=messages, response_type=ExtractResults)
        
        # 更新性能指标
        state["model_call_count"] += 1
        state["model_call_times_ms"].append(model_time)
        
        ai_message = AIMessage(content=results.model_dump_json())
        state["messages"].append(ai_message)
        logger.info(f"extract_results:{results}")
        
        # 使用提取结果更新文档列表，state 中直接存 List[Document]
        updated_docs = merge_extract_results_to_documents(results, relative_documents)
        state["relative_documents_list"] = updated_docs
    except Exception as e:
        logger.error(f"Validation error in extract_node: {e}\n{traceback.format_exc()}")
        # 可以视需求决定如何处理—此处抛出异常或返回错误信息
        raise
    
    return state