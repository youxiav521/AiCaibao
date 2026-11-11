# """
# 模型问答接口
# """
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# from loguru import logger
#
# from src.utils.model_factory import ModelFactory
#
# router = APIRouter(prefix="/api/ai", tags=["极简问答"])
#
#
# class ChatRequest(BaseModel):
#     file_path: Optional[str] = None
#     question: str
#
#
# class ChatService:
#     def __init__(self):
#         self.model = ModelFactory.get_model(key="model_for_table_classify")
#
#     async def ask(self, question: str):
#         # 直接调用模型，不加工prompt
#         resp = await self.model.ainvoke(question)
#         return resp.content
#
#
# chat_service = ChatService()
#
#
# # 唯一接口
# @router.post("/chat")
# async def chat(req: ChatRequest):
#     try:
#         answer = await chat_service.ask(req.question)
#
#         # ✅ 标准统一出参格式
#         return {
#             "code": 200,
#             "msg": "操作成功",
#             "data": {
#                 "answer": answer
#             }
#         }
#
#     except Exception as e:
#         logger.error(f"模型调用错误：{e}")
#
#         # ✅ 异常也统一格式返回
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "code": 500,
#                 "msg": f"模型调用失败：{str(e)}",
#                 "data": None
#             }
#         )


from fastapi import APIRouter, HTTPException, Body
from loguru import logger
import asyncio

from ..schemas.multimodal_chat_schema import MultimodalChatRequest
from ..services.multimodal_chat_service import MultimodalChatService
from ..utils.minio_util import download_file_from_minio

router = APIRouter(prefix="/api/ai", tags=["统一智能问答(文本+多文件)"])
chat_service = MultimodalChatService()




@router.post("/chat")
async def chat(
    request: MultimodalChatRequest = Body(...)
):
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="question不能为空")

        # 无文件 → 纯文本问答
        if not request.files:
            logger.info("✅ 未检测到文件，使用纯文本大模型问答")
            answer = await chat_service.ask_without_files(
                question=request.question
            )
        else:
            # 有文件 → 多模态问答
            logger.info("✅ 检测到文件，使用多模态模型问答")
            bucket_name = next(iter(request.files.keys()))
            file_list = request.files[bucket_name]
            if len(file_list) > 2:
                raise HTTPException(status_code=400, detail="当前暂时仅支持最多上传2个文件")
            file_contents = []
            for file_info in file_list:
                content = await asyncio.to_thread(
                    download_file_from_minio,
                    request.minio.server_url,
                    request.minio.access_key,
                    request.minio.secret_key,
                    bucket_name,
                    file_info.fileId
                )
                file_contents.append({
                    "content": content,
                    "filename": file_info.fileName
                })

            answer = await chat_service.ask_with_files(
                file_contents=file_contents,
                question=request.question
            )

        return {
            "code": 200,
            "msg": "操作成功",
            "data": {"answer": answer}
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error("接口异常: {!s}", e)
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")