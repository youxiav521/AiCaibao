"""
文档结构化提取V2路由 - 支持 LLM/VLM 两种解析模式
"""
import json
import os
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger

from src.applications.doc_extract.schema.extract_v2 import ExtractRequest, ExtractResult
from src.applications.doc_extract.utils.vlm_file_utils import download_files_from_minio

router = APIRouter(prefix="/api/doc/extract_v2", tags=["文档结构化提取V2"])


@router.post(
    "",
    summary="文档结构化提取V2 - 支持 LLM/VLM 模式",
    response_model=List[ExtractResult]
)
async def extract_document_v2(
    request: ExtractRequest
) -> List[ExtractResult]:
    """
    文档结构化提取V2 - 支持 LLM/VLM 两种解析模式

    模式说明：
    - vlm：从MinIO下载文件转图片后用视觉模型提取，适用于图片、短文档、简单表格场景
    - llm：适用于复杂表格提取或长文档场景（需要docContent）
    """
    try:
        logger.info(f"收到提取V2请求，模式: {request.mode}")

        # 1. 解析docContent
        convert_json = []
        if request.docContent:
            try:
                doc_content_obj = json.loads(request.docContent)
                convert_json = doc_content_obj.get("convert_json", [])
                logger.info(f"docContent已提供，convert_json元素数量: {len(convert_json)}")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"docContent解析失败: {str(e)}")

        # 2. 提取规则校验
        if not request.extract or len(request.extract) == 0:
            raise HTTPException(status_code=400, detail="extract规则数组不能为空")

        validated_extract_rules = []
        for rule in request.extract:
            validated_extract_rules.append({
                "extractId": rule.extractId,
                "extractName": rule.extractName,
                "extractPrompt": rule.extractPrompt or "",
                "jsonSchema": rule.jsonSchema,
                "namingRule": rule.namingRule or ""
            })

        # 3. VLM模式：从MinIO下载文件（转图由service按需完成）
        file_paths = None
        if request.mode == "vlm":
            file_paths = await download_files_from_minio(request.minio, request.files)

        # 4. LLM模式校验
        elif request.mode == "llm":
            if not request.docContent:
                raise HTTPException(status_code=400, detail="llm模式下必须提供docContent（包含convert_json）")
            if not convert_json or len(convert_json) == 0:
                raise HTTPException(status_code=400, detail="docContent中的convert_json不能为空")
            logger.info("使用llm模式")

        # 5. 从convert_json中提取element_info（仅LLM模式需要）
        element_info = None
        if request.mode == "llm":
            element_info = []
            allowed_types = {"table", "excel_table", "list", "text"}

            for item in convert_json:
                if isinstance(item, dict) and "element_id" in item and "type" in item and item.get("type") in allowed_types:
                    element_type = item.get("type")

                    if element_type in {"table", "excel_table"}:
                        element_content = item.get("rows", [])
                    else:
                        element_content = item.get("md", "")

                    element_data = {
                        "id": str(item.get("element_id")),
                        "content": element_content,
                        "type": element_type
                    }

                    if element_type == "table":
                        table_caption = item.get("table_caption")
                        if table_caption:
                            if isinstance(table_caption, list):
                                element_data["table_caption"] = ", ".join(str(c) for c in table_caption)
                            else:
                                element_data["table_caption"] = str(table_caption)

                    element_info.append(element_data)

        # 6. 调用V2服务进行提取
        from src.applications.doc_extract.services.doc_extract_service_v2 import DocExtractServiceV2

        service = DocExtractServiceV2()
        if request.mode == "vlm":
            try:
                results = await service.extract(
                    extract_rules=validated_extract_rules,
                    mode=request.mode,
                    file_paths=file_paths
                )
            finally:
                for f in file_paths or []:
                    try:
                        if os.path.exists(f):
                            os.remove(f)
                    except Exception:
                        logger.exception("清理临时文件失败 | path={}", f)
        else:
            results = await service.extract(
                extract_rules=validated_extract_rules,
                mode=request.mode,
                element_info=element_info,
            )

        # 7. 构建响应
        response_results = []
        for result in results:
            response_results.append(
                ExtractResult(
                    extractId=result["extractId"],
                    objectName=result["objectName"],
                    objectValue=result["objectValue"],
                    objectSource=result.get("objectSource")
                )
            )

        logger.info(f"提取完成，模式: {request.mode}，结果数量: {len(response_results)}")
        return response_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提取过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"提取失败: {str(e)}")