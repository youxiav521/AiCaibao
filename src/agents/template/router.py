import json
import traceback
import aiofiles
from pathlib import Path
from typing import Optional, List, Annotated
from fastapi import APIRouter, File, UploadFile, Body, Query, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
from src.agents.template.utils import read_file, write_file
from src.agents.template.optimize_prompt.optimizer import optimize_prompt
from src.agents.template.identify.template_identify import test_analyze_existing_file
from src.base.task_manager import TaskManager

_TEMPLATE_TASK_TYPE = "template_parse"
_TEMPLATE_TASK_NAME = "process_template_parse"

template_router = APIRouter(prefix="/api/template", tags=["模板解析"])

TEMPLATE_DEFAULT_MODEL_ID = "parse_template"
TEMPLATE_DEFAULT_BATCH_SIZE = 5

class ParseTemplateRequest(BaseModel):
    """模板解析请求体"""
    id: str
    templateId: str
    qIds: Optional[List[str]] = None
    callbackUrl: Optional[str] = None
    modelId: Optional[str] = Field(None, description="模型ID，默认为 " + TEMPLATE_DEFAULT_MODEL_ID)
    batchSize: Optional[int] = Field(TEMPLATE_DEFAULT_BATCH_SIZE, description="批处理大小，默认为" + str(TEMPLATE_DEFAULT_BATCH_SIZE))
    sendCallback: Optional[bool] = Field(True, description="是否发送回调信息，默认为 True")

class OptimizePromptRequest(BaseModel):
    """提示词优化请求体"""
    qId: str = Field(..., description="问题ID")
    question: str = Field(..., description="问题内容")
    prompt: str = Field(..., description="原始提示词")
    uploadFileId: str = Field(..., description="模板ID")

@template_router.post("/upload", summary="模板上传")
async def upload_template(
    files: UploadFile = File(..., description="上传的文件")):
    file_id = await write_file(files)
    return {"id": file_id}

@template_router.post("/parse", summary="模板解析（回调方式）")
async def parse_template(
    request: ParseTemplateRequest,
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """
    使用回调方式解析模板
    
    此接口接收模板ID和问题列表，使用模板解析流程进行问题分析。
    通过HTTP POST回调方式异步返回处理结果。
    
    接口会立即返回，模板解析通过 ARQ 队列在 Worker 进程中异步执行。
    每当处理完成一个问题时，会通过回调接口返回success event和处理结果。
    
    请求头参数:
        - Accept-Language: 可选，用于指定语言环境。当值为 "en_US" 时，use_english=True
    """
    # 解析 Accept-Language 请求头，判断是否使用英文
    use_english = False
    if accept_language and accept_language.strip().lower() == "en_us":
        use_english = True
        logger.info(f"检测到 en_US 语言环境，将使用英文模式处理")
    
    # 提交任务到 ARQ 队列
    job_id = await TaskManager.submit(
        task_name=_TEMPLATE_TASK_NAME,
        task_type=_TEMPLATE_TASK_TYPE,
        kwargs={
            "id": request.id,
            "template_id": request.templateId,
            "q_ids": request.qIds,
            "callback_url": request.callbackUrl,
            "model_id": request.modelId if request.modelId else TEMPLATE_DEFAULT_MODEL_ID,
            "batch_size": request.batchSize if request.batchSize else TEMPLATE_DEFAULT_BATCH_SIZE,
            "use_english": use_english,
            "send_callback": request.sendCallback
        }
    )
    
    logger.info(
        "模板解析任务已提交: job_id={}, template_id={}",
        job_id, request.templateId
    )
    
    # 立即返回响应
    return JSONResponse(
        content={
            "success": True,
            "message": "模板解析任务已提交，正在后台处理",
            "templateId": request.templateId,
            "jobId": job_id
        }
    )

@template_router.post("/optimizePrompt", summary="优化提示词")
async def optimize_prompt_endpoint(
    request: OptimizePromptRequest,
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """
    优化提示词接口
    
    接收问题ID、问题内容、原始提示词和模板ID，将问题和提示词组合后优化，
    返回优化后的提示词。
    """
    # 解析 Accept-Language 请求头，判断是否使用英文
    use_english = False
    if accept_language and accept_language.strip().lower() == "en_us":
        use_english = True
        logger.info(f"检测到 en_US 语言环境，将使用英文模式处理")
        
    try:
        # 如果有 uploadFileId，则尝试读取文件，错误时仅记录日志
        if request.uploadFileId:
            try:
                file_path = await read_file(request.uploadFileId)
                logger.info(f"Successfully read file for uploadFileId {request.uploadFileId}: {file_path}")
            except Exception as file_exc:
                logger.warning(f"读取 uploadFileId={request.uploadFileId} 的文件失败: {file_exc}")
        # 将 question 和 prompt 简单组合
        # 尝试解析 request.prompt，提取 type 为 'md' 的 input 字段内容拼接为 prompt
        prompt_str = request.prompt
        extracted_prompt = ""
        prompt_data = json.loads(prompt_str) if isinstance(prompt_str, str) else prompt_str
        if isinstance(prompt_data, list):
            md_inputs = [
                obj.get("input", "")
                for obj in prompt_data
                if isinstance(obj, dict) and obj.get("type") == "md" and obj.get("input")
            ]
            extracted_prompt = "\n".join(md_inputs)
        # 如果能够成功提取，则用拼接的 input 替换 prompt_str
        if extracted_prompt:
            prompt_str = extracted_prompt
        combined_prompt = f"为解决这个问题: {request.question}\n写了如下提示词: \n{prompt_str}"
        
        # 调用优化函数，生成1个优化版本
        optimized_prompts = await optimize_prompt(
            original_prompt=combined_prompt,
            num_variants=1,
            model_id=TEMPLATE_DEFAULT_MODEL_ID,
            use_english=use_english
        )
        
        # 获取优化后的提示词（取第一个）
        optimized_prompt_text = optimized_prompts[0] if optimized_prompts else request.prompt
        
        # 返回指定格式
        return {
            "qId": request.qId,
            "uploadFileId": request.uploadFileId,
            "question": request.question,
            "prompt": optimized_prompt_text
        }
        
    except Exception as e:
        logger.error(f"提示词优化失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "提示词优化失败",
                "detail": str(e),
                "qId": request.qId,
                "uploadFileId": request.uploadFileId,
                "question": request.question,
                "prompt": request.prompt  # 失败时返回原始提示词
            }
        )

@template_router.post("/test-callback", summary="测试回调接口（临时）")
async def test_callback(
    data: Annotated[str, Field(description="事件数据（JSON字符串）")] = Body(...),
    event: Annotated[str, Field(description="事件类型（start/success/complete/error）")] = Body(...),
    templateId: Annotated[str, Field(description="模板ID")] = Body(...)
):
    """
    临时测试回调接口
    """
    try:
        parsed_data = json.loads(data) if isinstance(data, str) else data

        logger.info(f"收到回调请求 - 事件: {event}, 模板ID: {templateId}")
        logger.info(f"回调数据: {json.dumps(parsed_data, ensure_ascii=False, indent=2)}")

        if event == "start":
            logger.info(f"🚀 任务开始 - 模板ID: {templateId}, 总问题数: {parsed_data.get('total', 'N/A')}")
            # 任务开始时创建新文件（清空旧数据）
            callback_dir = Path(".tmp_llmops/dataset/callback") / templateId
            callback_dir.mkdir(parents=True, exist_ok=True)
            jsonl_file = callback_dir / "data.jsonl"
            # 创建空文件或清空现有文件
            async with aiofiles.open(jsonl_file, "w", encoding="utf-8") as f:
                pass  # 只创建空文件，不写入数据
            logger.debug(f"已创建新文件: {jsonl_file}")
        elif event == "success":
            if parsed_data.get("type") == "header":
                logger.info(f"📋 Header - ID: {parsed_data.get('id')}, 描述: {parsed_data.get('description')}")
            else:
                logger.info(f"✅ 问题处理完成 - ID: {parsed_data.get('id', 'N/A')}")
            # 追加写入成功事件
            callback_dir = Path(".tmp_llmops/dataset/callback") / templateId
            callback_dir.mkdir(parents=True, exist_ok=True)
            jsonl_file = callback_dir / "data.jsonl"
            async with aiofiles.open(jsonl_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(parsed_data, ensure_ascii=False) + "\n")
            logger.debug(f"已追加写入回调数据到: {jsonl_file}")
        elif event == "complete":
            logger.info(f"🎉 任务完成 - 模板ID: {templateId}，问题总数：{parsed_data.get('total', 'N/A')}，处理数量: {parsed_data.get('processed', 'N/A')}，耗时: {parsed_data.get('processing_time', 'N/A')}秒")
            # 完成事件不写入文件
        elif event == "error":
            logger.error(f"❌ 任务错误 - 模板ID: {templateId}, 错误: {parsed_data.get('error', 'N/A')}")
            # 追加写入错误事件
            callback_dir = Path(".tmp_llmops/dataset/callback") / templateId
            callback_dir.mkdir(parents=True, exist_ok=True)
            jsonl_file = callback_dir / "data.jsonl"
            async with aiofiles.open(jsonl_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(parsed_data, ensure_ascii=False) + "\n")
            logger.debug(f"已追加写入错误事件到: {jsonl_file}")

        interrupt = False  # 修改为 True 可以测试中断

        resp_data = {
            'code': '0',
            'total': parsed_data.get('total', 0)
                if isinstance(parsed_data, dict) and 'total' in parsed_data
                else 0,
            'message': None,
            'level': 'info',
            'data': {
                'interrupt': interrupt,
                'templateId': templateId
            }
        }

        if interrupt:
            logger.warning(f"⚠️  返回中断指令 - 模板ID: {templateId}")

        return resp_data

    except Exception as e:
        logger.error(f"测试回调接口处理失败: {e}", exc_info=True)
        return {
            'code': '-1',
            'total': 0,
            'message': str(e),
            'level': 'error',
            'data': {
                'interrupt': False,
                'templateId': templateId if 'templateId' in locals() else 'unknown'
            }
        }

@template_router.post("/test-analyze", summary="测试版面分析")
async def test_analyze(fileId: str = Query(..., description="文件ID")):
    """
    测试接口：根据已上传文件的 file_id 执行版面分析
    先调用 /api/template/upload 上传文件获取 file_id，再调用此接口进行版面分析
    """
    try:
        sn_items = await test_analyze_existing_file(file_id=fileId)
        # 转换为JSON格式返回
        if sn_items:
            # 如果是Pydantic模型列表，转换为字典列表
            json_result = [item.model_dump() if hasattr(item, 'model_dump') else item.dict() for item in sn_items]
        else:
            json_result = []
        return JSONResponse(content={"sn_items": json_result})
    except Exception as e:
        logger.error("版面分析测试失败: {}", e, exc_info=True)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "版面分析测试失败",
                "detail": str(e)
            }
        )

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    app = FastAPI()
    app.include_router(template_router)
    uvicorn.run(app, host="0.0.0.0", port=7007)