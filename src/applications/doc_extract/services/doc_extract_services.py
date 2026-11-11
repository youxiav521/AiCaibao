"""
文档结构化提取服务
"""
import json
import asyncio
from typing import List, Dict, Any, Union
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from src.utils.model_factory import ModelFactory
from src.applications.doc_extract.utils.doc_extract_utils import (
    ElementInfo,
    split_elements_into_chunks,
    merge_chunk_results,
    format_element_text,
    calc_element_bytes,
)


class DocExtractService:
    """文档结构化提取服务类"""
    
    def __init__(self):
        """初始化服务"""
        self.model = ModelFactory.get_model(key="doc_extract")

    def _try_json_loads(self, v: Any) -> Any:
        """如果 v 是 JSON 字符串则尝试解析；否则原样返回。"""
        if not isinstance(v, str):
            return v
        s = v.strip()
        if not s:
            return v
        # 只在看起来像 JSON 时尝试解析，避免误伤普通文本
        if not (s.startswith("{") or s.startswith("[") or s.startswith('"')):
            return v
        try:
            return json.loads(s)
        except Exception:
            return v

    def _normalize_object_value(self, llm_result: Dict[str, Any]) -> Any:
        """
        规范化 LLM 返回的 objectValue，修复常见的"双层 objectValue"与"objectValue 为 JSON 字符串"问题。

        目标：extract() 最终返回的 objectValue（业务字段）只包含"结构化结果对象"，而不是再包一层 {"objectValue": ...}
        """
        obj = llm_result.get("objectValue", {})
        obj = self._try_json_loads(obj)

        # 常见错误：模型把 objectValue 又包了一层或多层：{"objectValue": {...}}
        while isinstance(obj, dict) and set(obj.keys()) == {"objectValue"}:
            obj = self._try_json_loads(obj.get("objectValue"))

        return obj
    
    async def _extract_with_llm(
        self,
        extract_prompt: str,
        json_schema: str,
        element_info: List[ElementInfo]
    ) -> Dict[str, Any]:
        """
        使用LLM进行结构化提取和溯源匹配

        Args:
            extract_prompt: 提取提示词
            json_schema: JSON Schema字符串，定义输出格式
            element_info: 元素信息列表，用于溯源匹配

        Returns:
            提取的结构化数据，包含提取结果和溯源信息
        """
        logger.info(f"开始结构化提取，extractPrompt: {extract_prompt}")

        # 计算总大小，判断是否需要分chunk
        max_bytes = 30000  # 30KB限制（约对应12K tokens）
        total_bytes = sum(calc_element_bytes(elem) for elem in element_info)

        if total_bytes > max_bytes:
            logger.info(f"总大小 {total_bytes} 字节超过限制，自动切换到异步分chunk处理")
            return await self._extract_with_batch(
                extract_prompt=extract_prompt,
                json_schema=json_schema,
                element_info=element_info,
                max_concurrent=3
            )

        logger.info(f"构建elements_text完成，大小: {total_bytes} 字节，包含 {len(element_info)} 个元素")
        return await self._extract_with_llm_async(
            extract_prompt=extract_prompt,
            json_schema=json_schema,
            element_info=element_info,
            chunk_index=0,
            max_retries=0
        )
    
    async def _extract_with_llm_async(
        self,
        extract_prompt: str,
        json_schema: str,
        element_info: List[ElementInfo],
        chunk_index: int = 0,
        max_retries: int = 3,
        initial_delay: float = 10.0
    ) -> Dict[str, Any]:
        """
        异步版本的LLM提取（用于batch处理），支持重试机制
        
        Args:
            extract_prompt: 提取提示词
            json_schema: JSON Schema
            element_info: 元素信息列表
            chunk_index: chunk索引
            max_retries: 最大重试次数（默认3次）
            initial_delay: 初始重试延迟秒数（默认10秒，使用指数退避：10s, 20s, 40s）
            
        Returns:
            提取的结构化数据
        """
        # 构建元素信息文本（调用方 _extract_with_batch 已保证 chunk 大小 ≤30KB，无需再截断）
        elements_text = "\n".join(
            format_element_text(elem)
            for elem in element_info
        )
        
        # 构建prompt（与同步版本相同的system_prompt）
        system_prompt = """
你是一个只输出 JSON 的结构化信息抽取器。

## 最高优先级：输出外层结构（违反此规则视为错误输出）
你的输出 JSON **必须且只能**包含两个顶层 key：`objectValue` 和 `objectSource`。
- 严禁把业务字段直接平铺在顶层（例如 `{"姓名": "xxx", "年龄": 18}`）
- 严禁省略 `objectValue` 或 `objectSource` 任何一个
- 正确做法：业务字段必须包装在 `objectValue` 里，溯源信息必须放在 `objectSource` 里

## 输出格式
请严格按照以下JSON格式输出结果：
{
  "objectValue": <根据JSON Schema生成的结构化结果对象；即使 schema 字段看起来就是"业务字段"，也必须包在此 key 内>,
  "objectSource": [
    {"element_id": "元素ID", "value": {"字段名": "字段值", ...}, "type": "元素类型"}
    // 注意：当type为"table"且有标题时，可添加字段: "table_caption": "表格标题"
  ]
}

## 输出示例（必须这样！）
假设 JSON Schema 定义字段 `姓名`、`年龄`，正确输出：
{
  "objectValue": {"姓名": "张三", "年龄": 25},
  "objectSource": [
    {"element_id": "1", "value": {"姓名": "张三", "年龄": 25}, "type": "text"}
  ]
}

错误输出（禁止！会被判为无效结果）：
{"姓名": "张三", "年龄": 25}

## 重要：字段抽取规则
**必须以JSON Schema为准！**
1. **必须抽取JSON Schema中定义的所有字段**，即使extractPrompt中没有明确列出
2. **参考字段描述**：JSON Schema中的`description`字段提供了每个字段的详细说明，请仔细阅读以理解字段含义和提取方式
3. extractPrompt只是任务的补充说明，不是字段范围的限制
4. 如果文档中存在JSON Schema定义的字段，就必须提取，不能因为extractPrompt中未提及而跳过
5. **严格保持层级结构**：必须完全按照JSON Schema的嵌套层级输出，禁止省略或合并任何层级（即使层级名称相同）

## 溯源规则

### 核心规则
1. **唯一性**：每个element_id只能出现一次，同一个元素的所有字段必须合并到一个溯源项中
2. **表格溯源**：表格数据必须且只能使用type为table/excel_table的元素ID
3. **ID约束**：只能使用文档元素信息中列出的element_id，禁止使用自定义ID或不在列表中的ID
4. **类型一致**：objectSource中的值必须与objectValue中的数据类型完全一致（数字/字符串/布尔值/数组/对象）
5. **完整溯源**：objectSource中的value必须包含该元素贡献的所有字段，保持相同的嵌套结构

### 正确示例
假设从元素"1"中提取了3个字段，正确做法：
[
  {"element_id": "1", "value": {"字段A": "值A", "字段B": "值B", "字段C": "值C"}, "type": "text"}
]

### 错误示例（禁止这样做）
// 错误：将同一元素的多个字段拆分为多个溯源项
[
  {"element_id": "1", "value": {"字段A": "值A"}, "type": "text"},
  {"element_id": "1", "value": {"字段B": "值B"}, "type": "text"},
  {"element_id": "1", "value": {"字段C": "值C"}, "type": "text"}
]

// 错误：使用不存在的element_id
[
  {"element_id": "不存在的ID", "value": {...}, "type": "text"}
]

// 错误：类型不一致
[
  {"element_id": "1", "value": {"数字字段": "123"}, "type": "text"}  // 应该是数字123，不是字符串"123"
]

## 输出要求
1. **格式**：整个输出必须是有效的JSON格式，不要包含任何其他文字
2. **objectValue**：必须作为顶层 key 输出；它的值是符合 JSON Schema 的结构化对象；该对象内部不要再嵌套一个叫 "objectValue" 的 key（即不要出现 {"objectValue": {"objectValue": {...}}}）
3. **objectSource**：
   - 为对象数组，每个对象包含：element_id、value、type
   - type值为：text、list、table、excel_table
   - 只有当type为"table"且有标题时，才添加字段：table_caption
4. **跳过规则**：如果某个字段无法提取，在objectValue中使用null或空字符串，不要为其创建溯源项
5. **数量匹配**：溯源项数量 = 匹配到的不同元素数量（不是字段值数量）

## 数据类型对照
- number → 数字（如：123, 45.67）
- string → 字符串（如："hello"）
- boolean → 布尔值（如：true/false，小写）
- array → 数组（如：[1, 2, 3]）
- object → 对象（如：{"key": "value"}）

接下来你会收到：
1. **文档元素信息**：格式为 "[element_id|type] 内容"，每行一个元素
   - 方括号内是元素ID和类型，后面是元素内容
   - objectSource 中请使用方括号内的 element_id 作为溯源ID
2. **本次任务说明**（extractPrompt）：任务的补充说明，仅供参考
3. **JSON Schema**：这是你必须遵守的字段定义，每个字段可能会提供`description`属性，提供了该字段的说明，请仔细阅读并作为提取参考

请严格按要求输出。
"""
        
        human_prompt = f"""## 文档元素信息（用于提取和溯源；只能使用下列 element_id）
{elements_text}

## 本次任务说明（extractPrompt，仅供参考，不限制抽取字段）
{extract_prompt}

## JSON Schema（这是你必须遵守的字段定义！必须抽取schema中的所有字段；注意：schema 描述的是 objectValue 内部的结构，顶层 JSON 仍然必须有 "objectValue" 和 "objectSource" 两个 key）
{json_schema}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # 重试机制：最多重试 max_retries 次
        last_exception = None
        for attempt in range(max_retries + 1):  # 包括首次尝试
            try:
                # 异步调用模型
                response = await self.model.ainvoke(
                    messages,
                    response_format={"type": "json_object"}
                )
                
                result_str = response.content
                logger.info("[Chunk {}] LLM 原始返回: {}", chunk_index, result_str)
                
                # 尝试解析JSON
                result = json.loads(result_str)
                # 二次解析：处理被引号包裹的JSON字符串
                if isinstance(result, str):
                    result = json.loads(result)
                if not isinstance(result, dict):
                    raise ValueError(f"LLM返回结果不是JSON对象: {type(result).__name__}")
                
                # 成功，返回结果
                if attempt > 0:
                    logger.info(f"[Chunk {chunk_index}] 重试成功")
                return result
                
            except json.JSONDecodeError as e:
                last_exception = e
                logger.error(f"[Chunk {chunk_index}] 解析JSON失败（尝试 {attempt + 1}/{max_retries + 1}）: {e}")
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt)  # 指数退避
                    logger.info(f"[Chunk {chunk_index}] 将在 {delay} 秒后重试...")
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                last_exception = e
                logger.error(f"[Chunk {chunk_index}] LLM调用失败（尝试 {attempt + 1}/{max_retries + 1}）: {e}")
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt)  # 指数退避
                    logger.info(f"[Chunk {chunk_index}] 将在 {delay} 秒后重试...")
                    await asyncio.sleep(delay)
        
        # 所有重试均失败
        logger.error(f"[Chunk {chunk_index}] 已达到最大重试次数 {max_retries}，提取失败")
        raise ValueError(f"LLM提取失败，已重试 {max_retries} 次: {last_exception}")
    
    async def _extract_with_batch(
        self,
        extract_prompt: str,
        json_schema: str,
        element_info: List[ElementInfo],
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        使用batch模式处理大量元素（分chunk并发调用）
        
        Args:
            extract_prompt: 提取提示词
            json_schema: JSON Schema
            element_info: 元素信息列表
            max_concurrent: 最大并发数
            
        Returns:
            合并后的提取结果
        """
        # 分chunk处理（调用方 _extract_with_llm 已保证数据 >30KB，无需再判断）
        chunks = split_elements_into_chunks(element_info)
        logger.info(f"分为 {len(chunks)} 个chunk，最大并发数: {max_concurrent}")
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_chunk(chunk_elements, index):
            async with semaphore:
                try:
                    return await self._extract_with_llm_async(extract_prompt, json_schema, chunk_elements, index)
                except Exception as e:
                    logger.error(f"[Chunk {index}] 处理失败: {type(e).__name__}: {e}")
                    return {"objectValue": {}, "objectSource": []}
        
        # 并发处理所有chunk
        tasks = [process_chunk(chunk, idx) for idx, chunk in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # 合并结果
        logger.info(f"所有chunk处理完成，开始合并结果")
        merged_result = merge_chunk_results(chunk_results)
        
        return merged_result
    
    def _generate_object_name(self, naming_rule: str, extracted_data: Dict[str, Any], extract_name: str = "") -> str:
        """
        根据命名规则生成对象名称

        逻辑：
        1. 如果naming_rule为空，返回extract_name
        2. 递归遍历 extracted_data（含嵌套字典/数组）的字段名。
        3. 若命名规则字符串中包含某个字段名，则用该字段的值替换命名规则中的该字段名。
           同时支持两种写法：直接字段名或 ${字段名} 占位。
        4. 如果没有任何字段名被替换（命名规则未改变），返回extract_name。
        5. 否则返回替换后的命名规则。

        Args:
            naming_rule: 命名规则模板（任意字符串，例如 "姓名的身份证"）
            extracted_data: 提取的结构化数据
            extract_name: 提取任务名称（作为默认名称）
            
        Returns:
            替换后的对象名称；若无字段匹配或naming_rule为空则返回extract_name
        """
        # 如果命名规则为空，直接返回extract_name
        if not naming_rule or not naming_rule.strip():
            logger.info(f"命名规则为空，使用extractName: {extract_name}")
            return extract_name
        
        # 保存原始命名规则，用于检测是否有替换发生
        original_naming_rule = naming_rule
        
        # 先尝试用提取数据中"与命名规则内容完全匹配的字段名"进行替换
        # 例如命名规则是"姓名的身份证"，且 extracted_data 有键"姓名"，则用该值替换命名规则中的"姓名"
        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield k, v
                    yield from _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    yield from _walk(item)

        for key, value in _walk(extracted_data):
            if not isinstance(key, str):
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            if f"${{{key}}}" in naming_rule:
                naming_rule = naming_rule.replace(f"${{{key}}}", value.strip())
                logger.info(f"用字段值替换命名规则中的占位: {key} -> {value.strip()}")
                return naming_rule
            elif key in naming_rule:
                naming_rule = naming_rule.replace(key, value.strip())
                logger.info(f"用字段值替换命名规则中的字段名: {key} -> {value.strip()}")
                return naming_rule

        # 如果命名规则没有任何改变（没有替换发生），返回extract_name
        if naming_rule == original_naming_rule:
            logger.info(f"命名规则未被替换，使用extractName: {extract_name}")
            return extract_name
        
        return naming_rule

    def _merge_object_source_by_element(self, object_source: List[List[Dict]], element_info: List[ElementInfo] = None) -> List[Dict]:
        """
        按element_id合并objectSource，确保相同element_id的字段值都在同一个value对象中

        Args:
            object_source: 原始的objectSource数组
            element_info: 元素信息列表，用于查找元素类型

        Returns:
            合并后的objectSource数组
        """
        if not object_source:
            return []

        # 使用字典来合并相同element_id的项
        element_type_map = {}
        element_caption_map = {}
        if element_info:
            for elem in element_info:
                elem_id = str(elem.get('id', ''))
                if not elem_id:
                    continue
                element_type_map[elem_id] = elem.get('type', 'unknown')
                if elem.get('type') == 'table' and elem.get('table_caption'):
                    element_caption_map[elem_id] = elem.get('table_caption')
        merged_dict = {}

        def _normalize_object_source_item(item):
            """
            将LLM返回的各种格式的溯源项统一转换为标准格式：
            [{"element_id": "xxx"}, {"value": {...}}, {"type": "xxx"}, {"table_caption": "xxx"}]
            
            LLM可能返回的格式：
            1. 字典格式: {"element_id": "1", "value": {...}, "type": "text"}
            2. 标准列表: [{"element_id": "1"}, {"value": {...}}, {"type": "text"}]
            3. 乱序列表: [{"value": {...}}, {"element_id": "1"}, {"type": "text"}]
            4. 嵌套列表: [[{"element_id": "1"}, {"value": {...}}]]
            """
            # 情况1: 字典格式 {"element_id": "1", "value": {...}, "type": "text"}
            if isinstance(item, dict):
                element_id = item.get("element_id")
                value = item.get("value")
                if element_id is None or value is None:
                    return None
                # 转换为标准列表格式
                normalized = [{"element_id": element_id}, {"value": value}]
                if "type" in item:
                    normalized.append({"type": item.get("type")})
                if item.get("table_caption"):
                    normalized.append({"table_caption": item.get("table_caption")})
                return normalized

            # 情况2-4: 列表格式
            if isinstance(item, list):
                # 情况2: 已经是标准格式 [{"element_id": ...}, {"value": ...}, ...]
                if len(item) >= 2 and isinstance(item[0], dict) and isinstance(item[1], dict):
                    return item
                
                # 情况3: 乱序列表，需要从中找出各个字段
                if all(isinstance(x, dict) for x in item):
                    element_obj = next((x for x in item if "element_id" in x), None)
                    value_obj = next((x for x in item if "value" in x), None)
                    if not element_obj or not value_obj:
                        return None
                    # 重新组装为标准顺序
                    normalized = [element_obj, value_obj]
                    type_obj = next((x for x in item if "type" in x), None)
                    if type_obj:
                        normalized.append(type_obj)
                    caption_obj = next((x for x in item if "table_caption" in x), None)
                    if caption_obj:
                        normalized.append(caption_obj)
                    return normalized
                
                # 情况4: 嵌套列表 [[{...}, {...}]]，先展平再处理
                if len(item) > 0 and isinstance(item[0], list):
                    flat = []
                    for x in item:
                        if isinstance(x, list):
                            flat.extend(x)
                        else:
                            flat.append(x)
                    # 展平后按情况3处理
                    if all(isinstance(x, dict) for x in flat):
                        element_obj = next((x for x in flat if "element_id" in x), None)
                        value_obj = next((x for x in flat if "value" in x), None)
                        if not element_obj or not value_obj:
                            return None
                        normalized = [element_obj, value_obj]
                        type_obj = next((x for x in flat if "type" in x), None)
                        if type_obj:
                            normalized.append(type_obj)
                        caption_obj = next((x for x in flat if "table_caption" in x), None)
                        if caption_obj:
                            normalized.append(caption_obj)
                        return normalized
            # 无法识别的格式，返回None
            return None

        # 遍历所有溯源项，统一格式后进行合并
        for item in object_source:
            # 将各种格式统一转换为标准格式
            normalized_item = _normalize_object_source_item(item)
            if not normalized_item or len(normalized_item) < 2:
                continue

            # 提取element_id和value
            element_obj = normalized_item[0]
            value_obj = normalized_item[1]
            if not isinstance(element_obj, dict) or not isinstance(value_obj, dict):
                continue

            # 提取信息
            element_id = element_obj.get('element_id', '')
            values = value_obj.get('value', {})

            # 验证element_id是否有效，如果无效则跳过这个溯源项
            if element_info and element_id:
                # 检查element_id是否存在于element_info中
                valid_ids = {str(elem.get('id', '')) for elem in element_info}
                if str(element_id) not in valid_ids:
                    logger.warning(f"LLM使用的element_id '{element_id}' 不在提供的element_info中，将跳过此溯源项以确保数据完整性")
                    continue  # 跳过这个无效的溯源项

            valid_element_id = element_id

            # 类型以element_info为准，避免LLM误标为text
            element_type = element_type_map.get(str(valid_element_id), 'unknown')
            if element_type == 'unknown' and len(normalized_item) >= 3:
                type_obj = normalized_item[2]
                element_type = type_obj.get('type', 'unknown')

            # 提取table_caption（优先使用element_info中提供的标题）
            table_caption = element_caption_map.get(str(valid_element_id), '')
            if not table_caption and len(normalized_item) > 3:
                caption_obj = normalized_item[3]
                table_caption = caption_obj.get('table_caption', '')

            if not valid_element_id:
                continue

            if valid_element_id not in merged_dict:
                merged_dict[valid_element_id] = {
                    'element': valid_element_id,
                    'values': {},
                    'type': element_type,
                    'table_caption': table_caption
                }

            # 合并值（字典合并）
            if isinstance(values, dict):
                merged_dict[valid_element_id]['values'].update(values)

        
        result = []
        for element_id, data in merged_dict.items():
            source_item = {
                'element_id': data['element'],
                'value': data['values'],
                'type': data['type']
            }
            # 为表格类型添加table_caption（只有在有内容时才添加）
            if data['type'] in ['table', 'excel_table'] and data['table_caption']:
                source_item['table_caption'] = data['table_caption']
            result.append(source_item)

        return result
    
    async def extract_async(
        self,
        extract_rules: List[Dict[str, str]],
        file_id: str,
        element_info: List[ElementInfo]
    ) -> List[Dict[str, Any]]:
        """
        异步版本的结构化提取，内部使用 ainvoke 调用大模型，不会阻塞事件循环。

        参数和返回值与 extract() 完全一致。
        """
        logger.info(f"[async] 开始处理 {len(extract_rules)} 个提取规则")
        results = []
        for rule in extract_rules:
            extract_id = rule.get("extractId")
            extract_name = rule.get("extractName", "")
            extract_prompt = rule.get("extractPrompt") or ""
            json_schema = rule.get("jsonSchema", "")
            naming_rule = rule.get("namingRule") or ""

            if not extract_id:
                logger.warning("跳过缺少extractId的提取规则")
                continue

            try:
                # 计算总大小，判断是否需要分chunk
                max_bytes = 30000
                total_bytes = sum(calc_element_bytes(elem) for elem in element_info)

                if total_bytes > max_bytes:
                    # 超过30KB，走异步分chunk并发处理
                    logger.info(f"[async] 总大小 {total_bytes} 字节超过限制，走异步分chunk处理")
                    llm_result = await self._extract_with_batch(
                        extract_prompt=extract_prompt,
                        json_schema=json_schema,
                        element_info=element_info,
                        max_concurrent=3
                    )
                else:
                    # ≤30KB，直接异步调用
                    llm_result = await self._extract_with_llm_async(
                        extract_prompt=extract_prompt,
                        json_schema=json_schema,
                        element_info=element_info
                    )

                # 提取objectValue和objectSource
                extracted_data = self._normalize_object_value(llm_result)
                raw_object_source = llm_result.get("objectSource", [])

                # 合并相同element_id的溯源信息
                object_source = self._merge_object_source_by_element(raw_object_source, element_info)

                # 生成对象名称
                object_name = self._generate_object_name(
                    naming_rule,
                    extracted_data,
                    extract_name
                )

                # 将提取的数据转换为JSON字符串
                object_value = json.dumps(extracted_data, ensure_ascii=False)

                result = {
                    "extractId": extract_id,
                    "fileId": file_id,
                    "objectName": object_name,
                    "objectValue": object_value,
                    "objectSource": object_source
                }

                results.append(result)
                logger.info(f"[async] 提取任务 {extract_id} 完成: {object_name}")

            except Exception as e:
                logger.error(f"[async] 提取任务 {extract_id} 失败: {e}")
                results.append({
                    "extractId": extract_id,
                    "fileId": file_id,
                    "objectName": naming_rule,
                    "objectValue": json.dumps({"error": str(e)}, ensure_ascii=False),
                    "objectSource": []
                })

        return results

    async def extract(
        self,
        extract_rules: List[Dict[str, str]],
        file_id: str,
        element_info: List[ElementInfo]
    ) -> List[Dict[str, Any]]:
        """
        执行结构化提取（支持大模型溯源匹配）

        Args:
            extract_rules: 提取规则列表，每个规则包含：
                - extractId: 提取任务唯一标识
                - extractName: 提取任务名称（用作默认对象名称）
                - extractPrompt: 提取提示词
                - jsonSchema: JSON Schema字符串
                - namingRule: 命名规则模板（为空或未替换时使用extractName）
            file_id: 文件唯一标识，用于溯源
            element_info: 元素信息列表，每个元素包含：
                - id: 元素唯一标识（字符串，element_id）
                - content: 元素内容
                - type: 元素类型

        Returns:
            提取结果列表，每个结果包含：
                - extractId: 提取任务唯一标识
                - fileId: 文件唯一标识
                - objectName: 生成的对象名称
                - objectValue: 结构化数据（JSON字符串）
                - objectSource: 溯源信息，字典数组，每个字典包含element_id、value、type，当type为"table"时还包含table_caption
        """
        logger.info(f"开始处理 {len(extract_rules)} 个提取规则")
        results = []
        for rule in extract_rules:
            extract_id = rule.get("extractId")
            extract_name = rule.get("extractName", "")
            extract_prompt = rule.get("extractPrompt") or ""
            json_schema = rule.get("jsonSchema", "")
            naming_rule = rule.get("namingRule") or ""
            
            if not extract_id:
                logger.warning("跳过缺少extractId的提取规则")
                continue
            
            try:
                # 调用LLM提取
                llm_result = await self._extract_with_llm(
                    extract_prompt=extract_prompt,
                    json_schema=json_schema,
                    element_info=element_info
                )

                # 提取objectValue和objectSource
                extracted_data = self._normalize_object_value(llm_result)
                raw_object_source = llm_result.get("objectSource", [])

                # 合并相同element_id的溯源信息
                object_source = self._merge_object_source_by_element(raw_object_source, element_info)

                # 生成对象名称
                object_name = self._generate_object_name(
                    naming_rule,
                    extracted_data,
                    extract_name
                )

                # 将提取的数据转换为JSON字符串
                object_value = json.dumps(extracted_data, ensure_ascii=False)

                # 构建结果
                result = {
                    "extractId": extract_id,
                    "fileId": file_id,
                    "objectName": object_name,
                    "objectValue": object_value,
                    "objectSource": object_source
                }
                
                results.append(result)
                logger.info(f"提取任务 {extract_id} 完成: {object_name}")
                
            except Exception as e:
                logger.error(f"提取任务 {extract_id} 失败: {e}")
                # 即使失败也返回一个结果，但objectValue为空或错误信息
                results.append({
                    "extractId": extract_id,
                    "fileId": file_id,
                    "objectName": naming_rule,
                    "objectValue": json.dumps({"error": str(e)}, ensure_ascii=False),
                    "objectSource": []
                })
        
        return results
