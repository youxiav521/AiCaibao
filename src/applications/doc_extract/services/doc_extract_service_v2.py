"""
文档结构化提取服务V2 - 支持多种模式
"""
import asyncio
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.model_factory import ModelFactory
from src.utils.document_render import convert_to_png, build_image_data_uri
from .doc_extract_services import DocExtractService
from ..utils.doc_extract_utils import ElementInfo


class DocExtractServiceV2:
    """文档结构化提取服务V2类 - 支持LLM和VLM模式"""
    
    def __init__(self):
        """初始化服务"""
        # LLM模式使用的服务
        self.llm_service = DocExtractService()
        # VLM模式使用的模型
        self.vlm_model = ModelFactory.get_model(key="doc_extract_vl", vision=True)
    
    async def extract(
        self,
        extract_rules: List[Dict[str, str]],
        mode: Literal["vlm", "llm"] = "llm",
        element_info: Optional[List[ElementInfo]] = None,
        file_paths: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行结构化提取（支持多种模式）
        
        Args:
            extract_rules: 提取规则列表，每个规则包含：
                - extractId: 提取任务唯一标识
                - extractName: 提取任务名称
                - extractPrompt: 提取提示词
                - jsonSchema: JSON Schema字符串
                - namingRule: 命名规则模板
            mode: 提取模式（vlm/llm）
            element_info: 元素信息列表（LLM模式使用）
            file_paths: 原始文件路径列表（VLM模式使用），按需转PNG+base64
        
        Returns:
            提取结果列表，每个结果包含：
                - extractId: 提取任务唯一标识
                - objectName: 生成的对象名称
                - objectValue: 结构化数据（JSON字符串）
                - objectSource: 溯源信息（仅LLM模式）
        """
        logger.info(f"开始V2提取，模式: {mode}，规则数量: {len(extract_rules)}")
        
        # 根据模式选择处理方式
        if mode == "llm":
            return await self._extract_with_llm_async(extract_rules, element_info)
        elif mode == "vlm":
            return await self._extract_with_vlm_async(extract_rules, file_paths)
        else:
            raise ValueError(f"不支持的提取模式: {mode}")

    async def _extract_with_llm_async(
        self,
        extract_rules: List[Dict[str, str]],
        element_info: Optional[List[ElementInfo]]
    ) -> List[Dict[str, Any]]:
        """
        使用LLM模式进行提取（异步版本，内部走 ainvoke，不阻塞事件循环）
        
        Args:
            extract_rules: 提取规则列表
            element_info: 元素信息列表
            
        Returns:
            提取结果列表
        """
        if element_info is None:
            raise ValueError("LLM模式下element_info不能为空")
        
        logger.info(f"使用LLM模式提取，元素数量: {len(element_info)}")
        
        # 直接调用异步版本，内部使用 ainvoke，无需线程池隔离
        # 底层 extract_async 仍需 file_id 参数（v1路由依赖），v2 不再使用，传空字符串
        return await self.llm_service.extract_async(
            extract_rules=extract_rules,
            file_id="",
            element_info=element_info
        )
    
    async def _extract_with_vlm_async(
        self,
        extract_rules: List[Dict[str, str]],
        file_paths: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """使用VLM模式进行提取（异步版本，按批转PNG+base64）。

        PDF 文件支持分批转图（每批 BATCH_SIZE 页），提前提取完则跳过后续转图。
        非 PDF 文件全量转图后按批 base64。
        """
        if not file_paths:
            raise ValueError("VLM模式下file_paths不能为空")

        file_path = file_paths[0]
        ext = Path(file_path).suffix.lower()

        # PDF 可分批转图；其他类型全量转
        can_batch_convert = ext == ".pdf"
        png_dirs: list[str] = []
        all_png_paths: list[str] = []

        try:
            if can_batch_convert:
                import fitz
                doc = fitz.open(file_path)
                total_pages = len(doc)
                doc.close()
            else:
                all_png_paths = await asyncio.to_thread(convert_to_png, file_path)
                if all_png_paths:
                    png_dirs.append(str(Path(all_png_paths[0]).parent))
                total_pages = len(all_png_paths)

            logger.info("使用VLM模式提取 | file={}, total_pages={}, batch_convert={}", file_path, total_pages, can_batch_convert)

            BATCH_SIZE = 5
            num_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
            results = []
            # 缓存已转换的 PNG 路径，避免多 rule 重复渲染
            batch_png_cache: dict[int, list[str]] = {}

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
                    has_array_leaf = self._has_array_leaf(json_schema)
                    merged_data = {}
                    logger.info("VLM提取任务 {}：共 {} 页，分 {} 批，存在数组叶子字段: {}", extract_id, total_pages, num_batches, has_array_leaf)

                    for batch_idx in range(num_batches):
                        # 获取当前批次的 PNG 路径
                        if can_batch_convert:
                            if batch_idx not in batch_png_cache:
                                batch_png_paths = await asyncio.to_thread(
                                    convert_to_png, file_path, max_pages=BATCH_SIZE, start_page=batch_idx * BATCH_SIZE
                                )
                                if batch_png_paths:
                                    png_dirs.append(str(Path(batch_png_paths[0]).parent))
                                batch_png_cache[batch_idx] = batch_png_paths
                            else:
                                batch_png_paths = batch_png_cache[batch_idx]
                        else:
                            start = batch_idx * BATCH_SIZE
                            batch_png_paths = all_png_paths[start:start + BATCH_SIZE]

                        # 编码为 base64
                        batch_screenshots = self._build_batch_screenshots(batch_png_paths, batch_idx)
                        vlm_result = await self._call_vlm_async(
                            page_screenshots=batch_screenshots,
                            extract_prompt=extract_prompt,
                            json_schema=json_schema
                        )
                        batch_data = vlm_result.get("objectValue", {})
                        if isinstance(batch_data, dict):
                            merged_data = self._merge_extracted(merged_data, batch_data)

                        if not has_array_leaf:
                            missing = self._get_missing_leaf_fields(json_schema, merged_data)
                            if not missing:
                                remaining = total_pages - (batch_idx + 1) * BATCH_SIZE
                                logger.info("VLM提取任务 {}：第 {} 批后所有叶子字段已提取完毕，提前结束（剩余 {} 页未转换）", extract_id, batch_idx + 1, max(remaining, 0))
                                break
                            logger.info("VLM提取任务 {}：第 {} 批完成，剩余未提取叶子字段: {}", extract_id, batch_idx + 1, missing)

                    extracted_data = merged_data

                    # 生成对象名称
                    object_name = self._generate_object_name(naming_rule, extracted_data, extract_name)

                    # 将提取的数据转换为JSON字符串
                    object_value = json.dumps(extracted_data, ensure_ascii=False)

                    # 构建 objectSource（多element_id，兼容V2格式）
                    object_value_data = json.loads(object_value)
                    object_source_list = []

                    for idx, (field_name, field_value) in enumerate(object_value_data.items(), start=1):
                        object_source_list.append({
                            "type": "text",
                            "element_id": str(idx),
                            "value": {field_name: field_value}
                        })

                    # 构建结果
                    result = {
                        "extractId": extract_id,
                        "objectName": object_name,
                        "objectValue": object_value,
                        "objectSource": object_source_list
                    }
                    results.append(result)
                    logger.info("VLM提取任务 {} 完成: {}", extract_id, object_name)

                except Exception as e:
                    logger.error("VLM提取任务 {} 失败: {}", extract_id, e)
                    raise

            return results
        finally:
            for dir_path in png_dirs:
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                except Exception:
                    logger.exception("清理PNG临时目录失败 | path={}", dir_path)

    @staticmethod
    def _build_batch_screenshots(png_paths: List[str], batch_idx: int) -> List[Dict[str, Any]]:
        """将一批 PNG 路径编码为 data-uri，返回 _call_vlm_async 所需的 screenshots 格式。"""
        screenshots = []
        for offset, png_path in enumerate(png_paths):
            page_index = batch_idx * 5 + offset
            uri = build_image_data_uri(png_path)
            screenshots.append({"pageIndex": page_index, "imageData": uri})
        return screenshots

    async def _call_vlm_async(
        self,
        page_screenshots: List[Dict[str, Any]],
        extract_prompt: str,
        json_schema: str
    ) -> Dict[str, Any]:
        """
        调用VLM模型进行提取（异步版本）
        
        Args:
            page_screenshots: 页面截图列表
            extract_prompt: 提取提示词
            json_schema: JSON Schema
            
        Returns:
            提取结果，包含objectValue
        """
        # 构建system prompt
        system_prompt = """
你是一个只输出 JSON 的结构化信息抽取器。

## 输出格式
请严格按照以下JSON格式输出结果：
{
  "objectValue": <根据JSON Schema生成的结构化结果对象>
}
**必须输出JSON对象本体，禁止输出被引号包裹的JSON字符串。**

## 重要：字段抽取规则
**必须以JSON Schema为准！**
1. **必须抽取JSON Schema中定义的所有字段**，即使extractPrompt中没有明确列出
2. **参考字段描述**：JSON Schema中的`description`字段提供了每个字段的详细说明，请仔细阅读以理解字段含义和提取方式
3. extractPrompt只是任务的补充说明，不是字段范围的限制
4. 如果文档中存在JSON Schema定义的字段，就必须提取，不能因为extractPrompt中未提及而跳过
5. **严格保持层级结构**：必须完全按照JSON Schema的嵌套层级输出，禁止省略或合并任何层级（即使层级名称相同）

## 输出要求
1. **格式**：整个输出必须是有效的JSON格式，不要包含任何其他文字
   - **禁止**输出被引号包裹的JSON字符串（如 "{...}" 这种双层字符串）
2. **objectValue**：必须符合JSON Schema定义，直接是结构化结果对象（不要包一层"objectValue"）
3. **跳过规则**：如果某个字段无法提取，在objectValue中使用null或空字符串

## 数据类型对照
- number → 数字（如：123, 45.67）
- string → 字符串（如："hello"）
- boolean → 布尔值（如：true/false，小写）
- array → 数组（如：[1, 2, 3]）
- object → 对象（如：{"key": "value"}）

接下来你会收到：文档截图（多个页面）、本次任务说明（extractPrompt）以及 objectValue 的 JSON Schema。
请严格按要求输出。
"""
        
        # 构建human prompt（包含图片）
        human_content = [
            {
                "type": "text",
                "text": f"""## 本次任务说明（extractPrompt，仅供参考，不限制抽取字段）
{extract_prompt}

## JSON Schema（这是你必须遵守的字段定义！必须抽取schema中的所有字段；objectValue 本身不要再包一层 "objectValue"）
{json_schema}

## 文档截图
以下是文档的所有页面截图，请从中提取信息：
"""
            }
        ]
        
        # 添加所有页面的图片（build_image_data_uri 已做 token 预算压缩）
        for idx, screenshot in enumerate(page_screenshots):
            page_index = screenshot.get("pageIndex", idx)
            image_data = screenshot.get("imageData", "")

            human_content.append({
                "type": "text",
                "text": f"\n第{page_index + 1}页："
            })
            human_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_data
                }
            })
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        
        # 调用VLM模型（异步）
        logger.info(f"调用VLM模型，页面数: {len(page_screenshots)}")
        response = await self.vlm_model.ainvoke(
            messages,
            response_format={"type": "json_object"}
        )
        
        result_str = response.content
        logger.info(f"VLM提取结果: {result_str[:200]}...")
        
        # 解析JSON结果
        try:
            result_dict = json.loads(result_str)
            # 二次解析：处理被引号包裹的JSON字符串
            if isinstance(result_dict, str):
                result_dict = json.loads(result_dict)
            if not isinstance(result_dict, dict):
                raise ValueError(f"VLM返回结果不是JSON对象: {type(result_dict).__name__}")
            
            # 规范化 objectValue
            # 如果 VLM 返回的 JSON 中有 "objectValue" 键，使用它
            # 否则，把整个返回结果当作 objectValue
            if "objectValue" in result_dict:
                obj_value = result_dict.get("objectValue", {})
                if isinstance(obj_value, str):
                    try:
                        obj_value = json.loads(obj_value)
                    except:
                        pass
                
                # 处理双层包装
                while isinstance(obj_value, dict) and set(obj_value.keys()) == {"objectValue"}:
                    obj_value = obj_value.get("objectValue")
                    if isinstance(obj_value, str):
                        try:
                            obj_value = json.loads(obj_value)
                        except:
                            break
            else:
                # VLM 直接返回了提取结果，没有包装 objectValue
                obj_value = result_dict
            
            return {"objectValue": obj_value}
            
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON失败: {e}, 原始结果: {result_str}")
            raise ValueError(f"VLM返回的结果不是有效的JSON格式: {result_str}")
    
    @classmethod
    def _has_array_leaf(cls, json_schema: str) -> bool:
        """递归检查 jsonSchema 的叶子字段是否存在 array 类型。"""
        try:
            schema = json.loads(json_schema)
        except Exception:
            return False
        return cls._schema_has_array_leaf(schema)

    @classmethod
    def _schema_has_array_leaf(cls, schema_node: Any) -> bool:
        if not isinstance(schema_node, dict):
            return False

        node_type = schema_node.get("type")
        if node_type == "array":
            return True

        properties = schema_node.get("properties")
        if isinstance(properties, dict):
            return any(cls._schema_has_array_leaf(child) for child in properties.values())

        return False

    @classmethod
    def _get_missing_leaf_fields(cls, json_schema: str, extracted: dict) -> list:
        """返回 schema 所有叶子字段中缺失或为 null/空 的字段路径。"""
        try:
            schema = json.loads(json_schema)
        except Exception:
            return []
        return cls._collect_missing_leaf_fields(schema, extracted, path="")

    @classmethod
    def _collect_missing_leaf_fields(cls, schema_node: Any, value: Any, path: str) -> list:
        if not isinstance(schema_node, dict):
            return []

        properties = schema_node.get("properties")
        if isinstance(properties, dict) and properties:
            missing: list[str] = []
            value_dict = value if isinstance(value, dict) else {}
            for field_name, child_schema in properties.items():
                child_path = f"{path}.{field_name}" if path else field_name
                child_value = value_dict.get(field_name)
                missing.extend(cls._collect_missing_leaf_fields(child_schema, child_value, child_path))
            return missing

        if cls._is_empty_extracted_value(value):
            return [path]
        return []

    @classmethod
    def _merge_extracted(cls, base: dict, patch: dict) -> dict:
        """递归合并 patch 中非空字段；已有非空叶子值不被覆盖。"""
        result = deepcopy(base)
        for key, patch_value in patch.items():
            base_value = result.get(key)

            if isinstance(base_value, dict) and isinstance(patch_value, dict):
                result[key] = cls._merge_extracted(base_value, patch_value)
                continue

            if cls._is_empty_extracted_value(base_value) and not cls._is_empty_extracted_value(patch_value):
                result[key] = patch_value

        return result

    @staticmethod
    def _is_empty_extracted_value(value: Any) -> bool:
        return value is None or value == "" or value == []

    def _generate_object_name(
        self,
        naming_rule: str,
        extracted_data: Dict[str, Any],
        extract_name: str = ""
    ) -> str:
        """
        根据命名规则生成对象名称
        Args:
            naming_rule: 命名规则模板
            extracted_data: 提取的结构化数据
            extract_name: 提取任务名称（作为默认名称）
            
        Returns:
            对象名称
        """
        return self.llm_service._generate_object_name(naming_rule, extracted_data, extract_name)

