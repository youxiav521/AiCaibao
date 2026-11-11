import re

# 使用正则表达式去除<think>...</think>块
def remove_think_blocks(text):
    """
    使用正则表达式去除<think>...</think>块
    """
    # 先去除<think>块，再去除文本头部的空行
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*\n", "", text, flags=re.MULTILINE)
    return text

    
def extract_bash_result_from_tool_message(text):
    """
    从ToolMessage的执行结果中，提取```bash ```代码块中的内容
    """
    # 匹配```bash ...```代码块
    match = re.search(r"```bash\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


def extract_code_error(text):
    """
    从ToolMessage的执行结果中，提取```error ...```代码块中的内容
    """
    # 匹配```error ...```代码块
    match = re.search(r"```error\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""
