# 临时添加以测试
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from typing import List, Dict
import os
import tiktoken
import time
from datetime import datetime
from src.base.langfuse_observe import trace_llm_call, trace_span, trace_agent_workflow
from openai import OpenAI

API_KEY = api_key=os.getenv("LLM_API_KEY")  # 百炼平台 API Key
QUESTION = "请根据领域知识里提供的财报，判断公司是否是上市公司。给出判断理由，在100字左右。"  # 填写你的问题

# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# COP_LLM_API_BASE = "http://103.154.31.78:20001/compatible-mode/v1"
SERVER_13_BASE_URL = "http://10.192.72.13:8088/v1"
COP_API_KEY = "None" #
MODEL = "Qwen3-32B"

USE_TOKEN_COUNT = True  # 设置为 True 使用基于 token 的截断，False 使用基于字符的截断
MAX_INPUT_TOKENS = 30_000  # 最大输入 token 数（仅在 USE_TOKEN_COUNT=True 时生效）
MAX_OUTPUT_TOKENS = 50  # 最大输出 token 数


def read_domain_knowledge_by_tokens(file_path: Path, max_tokens: int = 32_000) -> str:
    """读取领域知识文件内容，并基于 token 数量截断以避免超长上下文。
    
    Args:
        file_path: full.md 文件路径
        max_tokens: 最大输入 token 数
    Returns:
        文本内容（可能被截断）
    """
    if not file_path.exists():
        raise FileNotFoundError(f"未找到领域知识文件: {file_path}")
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenizer.encode(text)
    original_token_count = len(tokens)
    
    print(f"原始文本 token 数: {original_token_count}")
    print(f"最大允许 token 数: {max_tokens}")
    
    if len(tokens) <= max_tokens:
        print(f"无需截断，使用完整文本")
        return text
    
    # 基于 token 数量截断
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    final_token_count = len(truncated_tokens)
    
    print(f"截断后 token 数: {final_token_count}")
    print(f"截断比例: {final_token_count/original_token_count*100:.2f}%")
    
    return truncated_text


def build_messages(domain_text: str, question: str) -> List[Dict[str, str]]:
    """构造聊天消息，将 full.md 作为领域知识注入上下文。"""
    system_prompt = (
        "你是一名领域特化的智能助理。请严格依据提供的领域知识作答，"
        "若知识中未包含所需信息，请明确说明并给出合理的推断边界。"
    )
    user_content = (
        f"问题：{question} \n"
        "以下是领域知识：\n" 
        "```markdown\n" + domain_text + "\n```\n\n" 
        "基于上述知识回答问题:\n" 
        # "/no_think" # for vllm Qwen3, reference: https://huggingface.co/Qwen/Qwen3-32B
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

@trace_llm_call(name = "llm_call")
def llm_call(client, messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=MAX_OUTPUT_TOKENS,  # 控制最大输出 token 数
        # enable_thinking = False,
        # extra_body = {"enable_thinking": False}
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # 对vllm部署的Qwen3-32B的关闭thinking的参数传法
    )
    return response

@trace_agent_workflow(name = "main")
def main() -> None:
    api_key = API_KEY 
    question = QUESTION

    if not api_key:
        raise SystemExit("未提供 API_KEY")
    if not question:
        raise SystemExit("未提供问题，请在代码中填写 QUESTION")

    domain_file = Path("storage/CCB-2023-minerU.md")

    domain_text = read_domain_knowledge_by_tokens(domain_file, MAX_INPUT_TOKENS)

    # client = OpenAI(api_key=api_key, base_url=BASE_URL)
    client = OpenAI(api_key=COP_API_KEY, base_url=SERVER_13_BASE_URL)

    messages = build_messages(domain_text, question)

    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    start_perf = time.perf_counter()

    response = llm_call(client = client, messages = messages)

    end_perf = time.perf_counter()
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    elapsed_seconds = end_perf - start_perf

    content = response.choices[0].message.content if response.choices else ""
    print(content)
    # 输出content的token数到控制台
    try:
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        # AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
        print(f"content的token数: {len(tokenizer.encode(content))} \n")
    except Exception as e:
        print(f"警告：无法加载 tokenizer: {e}")
    
    print("\n—— 时间统计 ——")
    print(f"开始时间: {start_ts}")
    print(f"结束时间: {end_ts}")
    print(f"耗时: {elapsed_seconds:.3f} s")

if __name__ == "__main__":
    main()