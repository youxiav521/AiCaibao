import re
def get_sn_int(text:str,index=0)->int:
    """从字符串中提取数字并返回整数,默认0第一个，-1则最后一个"""
    # match = re.search(r'\d+', text)
    if isinstance(text, int):
        return text
    if not isinstance(text, str):
        text = str(text)
    matches = re.findall(r'#(\d+)', text)
    if matches:
        return int(matches[index])
    return 0

def to_sn(sn)->str:
    """字符串转为'#数字'格式序号"""
    return f'#{get_sn_int(sn)}'

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as file:
            content = file.read()
    return content

def extract_text_between_strings(text, start_str, end_str):
    """
    提取两个字符串之间的文本内容
    """
    pattern = re.escape(start_str) + r'(.*?)' + re.escape(end_str)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None