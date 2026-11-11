"""
文档结构化提取路由
"""
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger
import json

from src.applications.doc_extract.services.doc_extract_services import DocExtractService
from src.applications.doc_extract.schema.extract import ExtractRequest, ExtractResult

router = APIRouter(prefix="/api/doc/extract", tags=["文档结构化提取"])


@router.post(
    "",
    summary="对输入的内容进行结构化提取",
    response_model=List[ExtractResult]
)
async def extract_document(request: ExtractRequest) -> List[ExtractResult]:
    """
    对输入的内容进行结构化提取

    接收文件ID、内容、转换结果和提取规则，从内容中提取目标信息并结构化，
    返回与提取规则一一对应的结构化结果。

    Args:
        request: 提取请求，主要需要的字段：
            - docContent: 文档解析内容 
            - extract: 提取规则数组（即ExtractRule的列表，每项包含extractId、extractName、extractPrompt、jsonSchema、namingRule）

    Returns:
        提取结果列表，每个结果包含extractId、fileId、objectName、objectValue、objectSource
        
    注意：objectName生成规则
        - 如果namingRule为空，使用extractName
        - 如果namingRule不为空但无法替换占位符，使用extractName
        - 如果namingRule成功替换占位符，使用替换后的结果
    """
    try:
        doc_content = json.loads(request.docContent)
        file_id = doc_content["file_id"]
        convert_json = doc_content["convert_json"]

        logger.info(f"收到提取请求，file_id: {file_id}")

        # 验证输入
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id不能为空")

        if not request.extract or len(request.extract) == 0:
            raise HTTPException(status_code=400, detail="extract规则数组不能为空")

        if not convert_json or len(convert_json) == 0:
            raise HTTPException(status_code=400, detail="convert_json不能为空")

        # 转换提取规则为字典格式
        extract_rules = []
        for rule in request.extract:
            extract_rules.append({
                "extractId": rule.extractId,
                "extractName": rule.extractName,
                "extractPrompt": rule.extractPrompt or "",
                "jsonSchema": rule.jsonSchema,
                "namingRule": rule.namingRule or ""
            })

        logger.info(f"extract规则: {extract_rules}")

        # 从convert_json中提取element_info
        # 每个元素都有element_id和type字段
        element_info = []

        # 只保留指定的四种类型：table, excel_table, list, text
        allowed_types = {"table", "excel_table", "list", "text"}

        for item in convert_json:
            if isinstance(item, dict) and "element_id" in item and "type" in item and item.get("type") in allowed_types:
                element_type = item.get("type")
                # table/excel_table 使用 rows 原始内容，其他类型使用 md
                if element_type in {"table", "excel_table"}:
                    rows = item.get("rows", [])
                    element_content = rows
                else:
                    element_content = item.get("md", "")

                element_data = {
                    "id": str(item.get("element_id")),
                    "content": element_content,
                    "type": element_type
                }

                # 对于table类型，如果有table_caption则添加该字段
                if element_type == "table":
                    table_caption = item.get("table_caption")
                    if table_caption:  # 只在有值时添加
                        # 如果table_caption是列表，转为字符串（用逗号分隔）
                        if isinstance(table_caption, list):
                            element_data["table_caption"] = ", ".join(str(c) for c in table_caption)
                        else:
                            element_data["table_caption"] = str(table_caption)

                element_info.append(element_data)

        # 创建服务实例并执行提取
        service = DocExtractService()
        results = await service.extract(
            extract_rules=extract_rules,
            file_id=file_id,
            element_info=element_info
        )

        # 转换为响应模型
        response_results = [
            ExtractResult(
                extractId=result["extractId"],
                fileId=result["fileId"],
                objectName=result["objectName"],
                objectValue=result["objectValue"],
                objectSource=result["objectSource"]
            )
            for result in results
        ]

        logger.info(f"提取完成，返回结果数量: {len(response_results)}")
        return response_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提取过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"提取失败: {str(e)}")


