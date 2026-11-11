from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from src.utils.model_factory import ModelFactory
from src.utils.doc.doc_classify_config_utils import DocClassifierConfig
import base64
from langchain_core.messages import HumanMessage,SystemMessage
# 根据传入的文档内容字符串获得出文档所属的类型
def classify_document_type(file_content:str) -> str:
    logger.info(f"开始-解析内容，内容长度: {len(file_content)} 字符")
    docConfig = DocClassifierConfig();
    
    # 根据传入的文档内容字符串获得出文档所属的类型
    prompts = "有以下类型：" + "\n"
    for key, value in docConfig.config.items():
        t = key + "." +value["name"] + ":" + value["prompt"] + " " + value["example"] + "\n"
        prompts = prompts + t;
      
    # 创建模型
    model = ModelFactory.get_model(key="model_for_doc_classify");

    # 定义提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts),
        ("human", "根据输入的内容给出一个类型即可，不需要返回多余内容,也不需要序号,输入内容如下:{file_content}")])

    # 组合链调用大模型
    chain = prompt | model
    response = chain.invoke({"file_content": file_content})
    biz_type = response.content
    logger.info(f"完成-得到所属类型: {biz_type}")
    return biz_type

# 根据传入的图片路径获取图片内容后得出所属的类型
def classify_image(image_path:str) -> str:
    logger.info(f"开始-解析内容，图片路径: {image_path}")
    docConfig = DocClassifierConfig();
    
    prompts = "有以下类型：" + "\n"
    for key, value in docConfig.config.items():
        t = key + "." +value["name"] + ":" + value["prompt"] + " " + value["example"] + "\n"
        prompts = prompts + t;
    prompts = prompts + "请根据图片内容给出一个类型即可，不需要返回多余内容,也不需要序号"
    with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = f"data:image/jpeg;base64,{encoded_string}"
    
    chat = ModelFactory.get_model(key="model_for_image_classify", vision=True);
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompts},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    )
    response = chat.invoke([message])
    bizType = response.content
    logger.info(f"完成-图片所属类型如下:{bizType}")
    return bizType

def classify_document_type_tool(prompt:str,file_content:str) -> str:
    logger.info(f"开始-解析内容，内容长度: {len(file_content)} 字符")
    
    # 输入长度校验：要求输入长度在 [1, 129024] 范围内
    if not file_content or len(file_content.strip()) == 0:
        logger.warning("文件内容为空，返回默认类型")
        return "未知类型"
    
    # 截断过长内容，预留空间给 prompt 和系统消息
    MAX_CONTENT_LENGTH = 100000
    if len(file_content) > MAX_CONTENT_LENGTH:
        logger.warning(f"文件内容过长 ({len(file_content)} 字符)，截断至 {MAX_CONTENT_LENGTH} 字符")
        file_content = file_content[:MAX_CONTENT_LENGTH]
  
    # 创建模型
    model = ModelFactory.get_model(key="model_for_doc_classify");

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=file_content)
    ]

    response = model.invoke(messages)
    biz_type = response.content
    logger.info(f"完成-得到所属类型: {biz_type}")
    return biz_type
