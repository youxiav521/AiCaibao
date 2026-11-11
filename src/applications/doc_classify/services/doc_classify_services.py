"""
文档分类服务
"""
import os
import json
import asyncio
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger
import aiofiles
import aiofiles.os
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from src.utils.model_factory import ModelFactory
from src.utils.document_render import build_image_data_uri, convert_to_png, CLASSIFY_MAX_VISION_TOKENS
from src.applications.doc_classify.schemas.doc_classify_schema import ClassifyItem

TEXT_EXTENSIONS = {'.txt', '.csv', '.md', '.log'}


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """判断 LLM 调用异常是否值得重试。

    - 可重试：网络断连、超时、限流（429）、服务端 5xx
    - 不可重试：4xx 业务错误（如 BadRequestError 400、AuthenticationError 401、
      PermissionDeniedError 403、NotFoundError 404、UnprocessableEntityError 422）
      以及非 LLM SDK 异常（如本地 ValueError）。
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIError):
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code >= 500
        # 没有明确状态码，谨慎不重试
        return False
    return False


class DocClassifyService:
    """文档分类服务类"""
    
    def __init__(self):
        """初始化服务"""
        self.model = ModelFactory.get_model(key="doc_classify")

    async def _classify_with_vlm(
        self,
        image_paths: List[str],
        system_prompt: str,
        category_list: List[str],
        use_english: bool = False,
    ) -> Dict[str, Any]:
        """
        使用VLM进行文档分类（支持多页图片输入）
        
        Args:
            image_paths: 图片路径列表（多页文档）
            system_prompt: 系统提示词
            category_list: 分类列表
            use_english: 是否使用英文提示词
            
        Returns:
            仅包含分类结果的字典
        """
        try:
            # 构建多模态消息内容（文本 + 图片）
            message_content = []
            
            # 添加文本提示
            if use_english:
                if len(image_paths) == 1:
                    message_content.append({
                        "type": "text",
                        "text": "Please classify this document."
                    })
                else:
                    message_content.append({
                        "type": "text",
                        "text": f"Please classify this multi-page document (total {len(image_paths)} pages). Please synthesize content from all pages to make the classification judgment."
                    })
            else:
                if len(image_paths) == 1:
                    message_content.append({
                        "type": "text",
                        "text": "请对这个文档进行分类。"
                    })
                else:
                    message_content.append({
                        "type": "text",
                        "text": f"请对这个多页文档进行分类（共{len(image_paths)}页）。请综合所有页面的内容进行分类判断。"
                    })
            
            # 添加图片（编码为 data-uri，必要时自动缩放压缩）
            for i, img_path in enumerate(image_paths):
                logger.info(f"编码第 {i+1} 页图片: {img_path}")
                image_data_uri = build_image_data_uri(img_path, max_vision_tokens=CLASSIFY_MAX_VISION_TOKENS)
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri
                    }
                })
            
            # 构建消息
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message_content)
            ]
            
            logger.info(f"发送 {len(image_paths)} 张图片给 VLM 进行分类")
            
            # 调用模型（仅对网络/超时/限流/5xx 等瞬时错误重试，4xx 业务错误直接抛出）
            max_retries = 3
            initial_delay = 5.0

            for attempt in range(max_retries + 1):
                try:
                    response = await self.model.ainvoke(messages)
                    if attempt > 0:
                        logger.info("VLM 分类重试成功 | attempt={}", attempt + 1)
                    break
                except Exception as exc:
                    if attempt >= max_retries or not _is_retryable_llm_error(exc):
                        if attempt > 0:
                            logger.error(
                                "VLM 分类调用失败，已重试 {} 次仍失败 | error_type={}",
                                attempt,
                                type(exc).__name__,
                            )
                        raise
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(
                        "VLM 分类调用失败，将重试 | attempt={}, delay_s={}, error_type={}, error={}",
                        attempt + 1,
                        delay,
                        type(exc).__name__,
                        exc,
                    )
                    await asyncio.sleep(delay)
            
            # 解析响应
            result_text = response.content
            logger.info(f"VLM分类响应: {result_text}")
            
            # 尝试解析JSON
            try:
                result_json = json.loads(result_text)
            except json.JSONDecodeError:
                # 如果不是标准JSON，尝试提取JSON部分
                import re
                # 使用更健壮的正则匹配提取包含 category 的 JSON
                json_match = re.search(r'\{[^{}]*"category"[^{}]*\}', result_text, re.DOTALL)
                if not json_match:
                    # 备选：匹配第一个完整的 JSON 对象
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                
                if json_match:
                    try:
                        result_json = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        result_json = {
                            "category": category_list[0] if category_list else "unknown"
                        }
                else:
                    # 无法解析，返回默认结果
                    result_json = {
                        "category": category_list[0] if category_list else "unknown"
                    }
            
            predicted_category = result_json.get("category", category_list[0] if category_list else "unknown")
            if predicted_category not in category_list:
                logger.warning(f"模型返回的分类 '{predicted_category}' 不在允许的列表中，使用第一个分类")
                predicted_category = category_list[0] if category_list else "unknown"

            return {
                "cls": predicted_category
            }
            
        except Exception as e:
            logger.error(f"VLM分类失败: {str(e)}")
            error_message = str(e)
            if "Exceeded limit on max bytes per data-uri item" in error_message:
                raise ValueError("分类图片过大，已超过模型 data-uri 限制，请减少页数或继续降低图片尺寸") from e
            raise  # 直接抛出异常，由上层处理

    def _build_classify_system_prompt(self, classify_items: List[ClassifyItem]) -> str:
        categories_lines = []
        for item in classify_items:
            if item.prompt:
                categories_lines.append(f'- "{item.name}"：{item.prompt}')
            else:
                categories_lines.append(f'- "{item.name}"')
        categories_str = "\n".join(categories_lines)

        default_prompt = f"""你是一位拥有20年经验的金融文档审核专家，擅长识别各类与金融相关的文档。

你的任务是接收用户上传的文档图片（可能包含多页），仔细阅读并分析文档内容，将其精准归类到给定的【可选分类类型】中。

【可选分类类型】：
{categories_str}

【分析策略】：
1. **视觉特征识别**：
   - **标题区**：优先寻找文档顶部的大号字体标题（如"营业执照"、"资产负债表"）。
   - **印章与签字**：检测红章（公章/财务章）和手写签名，这是判断合同/证明类文档的关键。
   - **版式结构**：识别是表格（如财务报表）、文本段落（如说明/简介）还是证件卡片（如身份证）。

2. **多页综合判断**：
   - 如果提供了多页，请综合首尾页信息。
   - 首页通常包含标题，尾页通常包含签署/印章信息。

3. **归类逻辑**：
   - 将提取的关键信息与【可选分类类型】进行语义匹配。
   - 必须选择最匹配的一个类别，**严禁**创造列表之外的新类别。

【输出要求】：
请严格只输出一个JSON对象，不要包含Markdown标记（如```json），格式如下：
{{
    "category": "必须从上述【可选分类类型】的分类名称中完全复制"
}}

注意：
- **category** 字段必须严格等于给定的分类名称，不要修改或缩写。"""
        return default_prompt

    def _build_text_classify_system_prompt(self, classify_items: List[ClassifyItem]) -> str:
        categories_lines = []
        for item in classify_items:
            if item.prompt:
                categories_lines.append(f'- "{item.name}"：{item.prompt}')
            else:
                categories_lines.append(f'- "{item.name}"')
        categories_str = "\n".join(categories_lines)

        default_prompt = f"""你是一位拥有20年经验的金融文档审核专家，擅长识别各类与金融相关的文档。

你的任务是接收用户上传的文本内容，仔细阅读并分析，将其精准归类到给定的【可选分类类型】中。

【可选分类类型】：
{categories_str}

【分析策略】：
1. **内容语义识别**：
   - 寻找文本中的关键词和专业术语（如"营业执照"、"资产负债表"、"合同"等）。
   - 分析文本结构：是否包含表格数据、条款列表、说明性段落等。
   - 识别文本的格式特征（如 CSV 的列标题、Markdown 的标题层级）。

2. **归类逻辑**：
   - 将提取的关键信息与【可选分类类型】进行语义匹配。
   - 必须选择最匹配的一个类别，**严禁**创造列表之外的新类别。

【输出要求】：
请严格只输出一个JSON对象，不要包含Markdown标记（如```json），格式如下：
{{
    "category": "必须从上述【可选分类类型】的分类名称中完全复制"
}}

注意：
- **category** 字段必须严格等于给定的分类名称，不要修改或缩写。"""
        return default_prompt

    def _build_classify_system_prompt_en(self, classify_items: List[ClassifyItem]) -> str:
        categories_lines = []
        for item in classify_items:
            if item.prompt:
                categories_lines.append(f'- "{item.name}": {item.prompt}')
            else:
                categories_lines.append(f'- "{item.name}"')
        categories_str = "\n".join(categories_lines)

        default_prompt = f"""You are a financial document review expert with 20 years of experience, skilled at identifying various types of financial documents.

Your task is to receive the document image(s) uploaded by the user, carefully read and analyze the document content, and accurately classify it into one of the given [Available Categories].

[Available Categories]:
{categories_str}

[Analysis Strategy]:
1. **Visual Feature Recognition**:
   - **Title Area**: Prioritize looking for large-font titles at the top of the document (e.g., "Business License", "Balance Sheet").
   - **Seals and Signatures**: Detect red seals (company seal/financial seal) and handwritten signatures, which are key indicators for contracts/certificates.
   - **Layout Structure**: Identify whether it is a table (e.g., financial statements), text paragraphs (e.g., descriptions/introductions), or identification cards (e.g., ID card).

2. **Multi-Page Comprehensive Judgment**:
   - If multiple pages are provided, synthesize information from the first and last pages.
   - The first page usually contains the title, and the last page usually contains signing/seal information.

3. **Classification Logic**:
   - Semantically match the extracted key information with the [Available Categories].
   - You must select the single best matching category. It is strictly prohibited to create new categories outside the list.

[Output Requirements]:
Output strictly a single JSON object, without Markdown markers (e.g., ```json), in the following format:
{{
    "category": "must be copied exactly from the category name in [Available Categories] above"
}}

Note:
- The **category** field must exactly match one of the given category names without modification or abbreviation."""
        return default_prompt

    def _build_text_classify_system_prompt_en(self, classify_items: List[ClassifyItem]) -> str:
        categories_lines = []
        for item in classify_items:
            if item.prompt:
                categories_lines.append(f'- "{item.name}": {item.prompt}')
            else:
                categories_lines.append(f'- "{item.name}"')
        categories_str = "\n".join(categories_lines)

        default_prompt = f"""You are a financial document review expert with 20 years of experience, skilled at identifying various types of financial documents.

Your task is to receive the text content uploaded by the user, carefully read and analyze it, and accurately classify it into one of the given [Available Categories].

[Available Categories]:
{categories_str}

[Analysis Strategy]:
1. **Content Semantic Recognition**:
   - Look for keywords and specialized terms in the text (e.g., "Business License", "Balance Sheet", "Contract", etc.).
   - Analyze the text structure: whether it contains tabular data, clause lists, descriptive paragraphs, etc.
   - Identify text format characteristics (e.g., CSV column headers, Markdown heading hierarchy).

2. **Classification Logic**:
   - Semantically match the extracted key information with the [Available Categories].
   - You must select the single best matching category. It is strictly prohibited to create new categories outside the list.

[Output Requirements]:
Output strictly a single JSON object, without Markdown markers (e.g., ```json), in the following format:
{{
    "category": "must be copied exactly from the category name in [Available Categories] above"
}}

Note:
- The **category** field must exactly match one of the given category names without modification or abbreviation."""
        return default_prompt
        
    async def classify(
        self,
        file_content: bytes,
        filename: str,
        classify_items: List[ClassifyItem],
        max_pages: int = 3,
        use_english: bool = False,
    ) -> Dict[str, Any]:
        category_list = [item.name for item in classify_items]

        logger.info(f"开始对文件进行分类: {filename}")
        logger.info(f"可选分类类型: {category_list}")
        logger.info(f"使用页数: {max_pages}")
        
        # 获取文件扩展名
        file_ext = Path(filename).suffix.lower()
        
        # 创建临时文件保存上传内容（异步写入）
        tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
        input_path = tmp_file.name
        tmp_file.close()
        async with aiofiles.open(input_path, "wb") as f:
            await f.write(file_content)
        
        logger.info(f"临时文件已创建: {input_path}")

        converted_images = []
        try:
            # 文本类型：直接读取内容发给大模型
            if file_ext in TEXT_EXTENSIONS:
                logger.info(f"检测到文本类型文件: {file_ext}，使用纯文本分类路径")
                try:
                    try:
                        text_content = file_content.decode('utf-8')
                    except UnicodeDecodeError:
                        text_content = file_content.decode('gbk', errors='replace')

                    MAX_TEXT_CHARS = 8000
                    if len(text_content) > MAX_TEXT_CHARS:
                        logger.warning(f"文本内容超过 {MAX_TEXT_CHARS} 字符，截断处理")
                        text_content = text_content[:MAX_TEXT_CHARS]

                    if use_english:
                        system_prompt = self._build_text_classify_system_prompt_en(classify_items)
                        user_prompt = f"Please classify the following text content:\n\n{text_content}"
                    else:
                        system_prompt = self._build_text_classify_system_prompt(classify_items)
                        user_prompt = f"请对以下文本内容进行分类：\n\n{text_content}"
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                    max_retries = 3
                    initial_delay = 5.0
                    for attempt in range(max_retries + 1):
                        try:
                            response = await self.model.ainvoke(messages)
                            if attempt > 0:
                                logger.info(f"LLM 文本分类重试成功（第 {attempt + 1} 次尝试）")
                            break
                        except Exception as e:
                            if attempt < max_retries:
                                delay = initial_delay * (2 ** attempt)
                                logger.warning(f"LLM 文本分类调用失败（第 {attempt + 1} 次），{delay}s 后重试: {e}")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"LLM 文本分类调用失败，已达最大重试次数 {max_retries}: {e}")
                                raise
                    result_text = response.content
                    logger.info(f"LLM文本分类响应: {result_text}")

                    try:
                        result_json = json.loads(result_text)
                    except json.JSONDecodeError:
                        import re
                        json_match = re.search(r'\{[^{}]*"category"[^{}]*\}', result_text, re.DOTALL)
                        if not json_match:
                            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                        if json_match:
                            try:
                                result_json = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                result_json = {"category": category_list[0] if category_list else "unknown"}
                        else:
                            result_json = {"category": category_list[0] if category_list else "unknown"}

                    predicted_category = result_json.get("category", category_list[0] if category_list else "unknown")
                    if predicted_category not in category_list:
                        logger.warning(f"模型返回的分类 '{predicted_category}' 不在允许的列表中，使用第一个分类")
                        predicted_category = category_list[0] if category_list else "unknown"

                    logger.info(f"文本分类完成: {predicted_category}")
                    return {"cls": predicted_category, "pages_bytes": []}
                finally:
                    try:
                        if input_path and await aiofiles.os.path.exists(input_path):
                            await aiofiles.os.remove(input_path)
                            logger.debug(f"已清理临时文件: {input_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {str(e)}")

            # 先转换为图片（如果需要）
            if file_ext not in ['.jpg', '.jpeg', '.png']:
                logger.info("将文档转换为PNG格式...")
                converted_images = await asyncio.to_thread(convert_to_png, input_path, max_pages=max_pages)
                image_paths = converted_images
            else:
                image_paths = [input_path]
            
            if use_english:
                system_prompt = self._build_classify_system_prompt_en(classify_items)
            else:
                system_prompt = self._build_classify_system_prompt(classify_items)
            
            # 使用前N页进行分类（提高准确度）
            if image_paths:
                # 限制使用的页数
                pages_to_use = image_paths[:max_pages]
                logger.info(f"使用 {len(pages_to_use)} 页进行分类（共 {len(image_paths)} 页）")
                
                result = await self._classify_with_vlm(
                    pages_to_use,
                    system_prompt,
                    category_list,
                    use_english=use_english,
                )
            else:
                raise ValueError("无法获取图片路径进行分类")

            # 在清理临时文件之前，读取参与分类的所有页面字节，供调用方生成缩略图复用
            pages_bytes: List[bytes] = []
            for page_path in pages_to_use:
                try:
                    async with aiofiles.open(page_path, "rb") as f:
                        pages_bytes.append(await f.read())
                except Exception as e:
                    logger.warning(f"读取图片失败 {page_path}: {e}")

            result["pages_bytes"] = pages_bytes
            logger.info(f"分类完成: {result['cls']}")
            return result
            
        except Exception as e:
            logger.error(f"文档分类失败: {str(e)}")
            raise
        finally:
            # 清理转换生成的临时图片文件
            if converted_images:
                for img_path in converted_images:
                    try:
                        if await aiofiles.os.path.exists(img_path):
                            await aiofiles.os.remove(img_path)
                            logger.debug(f"已清理转换图片: {img_path}")
                    except Exception as e:
                        logger.warning(f"清理转换图片失败 {img_path}: {str(e)}")
                
                # 清理转换图片所在的目录（如果为空）
                try:
                    if converted_images:
                        img_dir = os.path.dirname(converted_images[0])
                        if await aiofiles.os.path.exists(img_dir) and not os.listdir(img_dir):
                            await aiofiles.os.rmdir(img_dir)
                            logger.debug(f"已清理空目录: {img_dir}")
                except Exception as e:
                    logger.warning(f"清理目录失败: {str(e)}")
            
            # 清理原始上传的临时文件
            try:
                if input_path and await aiofiles.os.path.exists(input_path):
                    await aiofiles.os.remove(input_path)
                    logger.debug(f"已清理临时文件: {input_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")
    