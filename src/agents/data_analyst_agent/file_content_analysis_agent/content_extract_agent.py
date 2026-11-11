import base64
from langchain_core.messages import HumanMessage,SystemMessage
from src.utils.model_factory import ModelFactory
from src.utils.doc.doc_extract_config_utils import DocExtractConfig
from loguru import logger
# 提取图片中所有内容
def extract_content_from_image_path(image_path:str) -> str:
    logger.info(f"开始-获取图片内容,图片路径:{image_path}")
    with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = f"data:image/jpeg;base64,{encoded_string}"
    
    chat = ModelFactory.get_model(key="model_for_image_extract", vision=True)
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "提取出图片中所有内容"},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    )
    response = chat.invoke([message])
    contentT = response.content
    logger.info(f"完成-获取图片内容,内容如下:{contentT}")
    return contentT

# 将输入的内容结合json_schema提取出结构化数据返回
def extract_content_to_json(json_schema_id:str,file_content:str) -> dict:
    logger.info(f"开始-结构化提取,接收内容:{file_content},json_schema:{json_schema_id}")
   
    model = ModelFactory.get_model(key="model_for_doc_extract")

    doc_config = DocExtractConfig()
    myConfig = doc_config.config[json_schema_id]
    json_schema = myConfig["schema"]
    prompt = myConfig["prompt"]
    logger.info(f"开始-结构化提取,得到json_schema:{json_schema}")
    # 构造消息
    messages = [
        SystemMessage(content=f"{prompt},注意日期格式要符合format规范,结果输出为JSON,json_schema格式如下{json_schema}"),
        HumanMessage(content=file_content)
    ]

    response = model.invoke(
        messages,
        response_format={
            "type": "json_object"
        }
    )
    result = response.content
    logger.info(f"完成-结构化提取,结果如下:{result}")
    return result

# 通过输入的图片路径获取图片信息，结合定义的json_schema提取出结构化数据返回
def extract_image_to_json(json_schema_id:str,image_path:str) -> dict:
    logger.info(f"开始-结构化提取,接收文件路径:{image_path},json_schema:{json_schema_id}")
   
    with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = f"data:image/jpeg;base64,{encoded_string}"
    
    chat = ModelFactory.get_model(key="model_for_image_extract", vision=True);
    
    doc_config = DocExtractConfig()
    myConfig = doc_config.config[json_schema_id]
    json_schema = myConfig["schema"]
    logger.info(f"开始-结构化提取,得到json_schema:{json_schema}")
    promptT = myConfig["prompt"]
    prompt = f"提取出图片中的内容，{promptT},注意日期格式要符合format规范,结果输出为JSON,json_schema格式如下{json_schema}"
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    )
    response = chat.invoke(
        [message], 
        response_format={
            "type": "json_object"
        })
    result = response.content
    logger.info(f"完成-结构化提取,结果如下:{result}")
    return result

# 将输入的内容结合json_schema提取出结构化数据返回
def extract_content_to_json_tool(prompt:str,file_content:str) -> str:
    logger.info(f"开始-结构化提取,接收内容:{file_content},prompt:{prompt}")
    file_content = truncate_string_by_bytes(file_content)
    model = ModelFactory.get_model(key="model_for_doc_extract")
    # 构造消息
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=file_content)
    ]

    response = model.invoke(
        messages,
        response_format={
            "type": "json_object"
        }
    )
    result = response.content
    logger.info(f"完成-结构化提取,结果如下:{result}")
    return result

def truncate_string_by_bytes(text, max_bytes=153400):  # 150KB = 153600 bytes
    """
    截取字符串，确保UTF-8编码后不超过指定字节数 150KB
    """
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    
    # 截取前max_bytes个字节
    truncated_bytes = encoded[:max_bytes]
    
    # 安全解码，避免截断多字节字符导致的错误
    try:
        return truncated_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # 如果截断在多字节字符中间，逐步减少字节直到能正确解码
        for i in range(1, 4):  # UTF-8最多4字节
            try:
                return truncated_bytes[:-i].decode('utf-8')
            except UnicodeDecodeError:
                continue
    return truncated_bytes.decode('utf-8', errors='ignore')