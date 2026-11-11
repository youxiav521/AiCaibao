import time
from fastapi import APIRouter, Form, Header, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

from src.utils.model_enhancements import metrics_context_session

from ....agents.data_analyst_agent.coding_agent.graph.agent_full import AgentFullService
from ....agents.data_analyst_agent.coding_agent.code_executor.python_code_executor import _get_executor_instance
from ....agents.data_analyst_agent.coding_agent.graph.metric_query import MetricQuery
from src.agents.content_generate_agent.utils import get_localized_message, resolve_answer_locale
from loguru import logger
from src.agents.data_analyst_agent.coding_agent.models.python_code_model import PythonCodeModel

data_analyst_router = APIRouter(prefix="/api", tags=["数据分析智能体"])  

ANALYST_DEFAULT_MODEL_ID = "data_analyst_agent"
# 模块级Service单例，避免每次请求重新实例化
_full_agent_service = AgentFullService()
_metric_query = MetricQuery()  

class DataAnalystAgentResponse(BaseModel):
    code: int = Field(..., description="返回码")
    error_message: Optional[str] = Field(None, description="错误信息")
    data: Optional[Any] = Field(None, description="数据")

class InstallPackageResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: Optional[str] = Field(None, description="响应消息")
    error: Optional[str] = Field(None, description="错误信息")

@data_analyst_router.post(
    "/test/data_analyst/install_package",
    response_model=InstallPackageResponse,
    summary="安装指定的Python包",
    description="通过执行器安装指定的Python包，包名通过param参数传入"
)
async def install_package(
    param: str = Form(..., description="需要安装的包名"),
    proxy_url: Optional[str] = Form(default=None, description="可选的代理URL，用于安装包时指定网络代理")
):
    """
    安装指定的Python包
    :param param: 需要安装的包名
    """
    try:
        executor = _get_executor_instance()
        result = await executor.install_packages({param}, proxy_url=proxy_url)
        if isinstance(result, dict) and result.get("success", False):
            return InstallPackageResponse(success=True, message=f"包 {param} 安装成功", error=None)
        else:
            return InstallPackageResponse(success=False, message=None, error=f"包 {param} 安装失败: {result.get('error', '')}")
    except Exception as e:
        logger.error(f"安装包时发生异常: {e}")
        return InstallPackageResponse(success=False, message=None, error=str(e))


class FullAgentRequest(BaseModel):
    question: str = Field(..., description="问题描述")
    prompt: Optional[str] = Field(default="", description="提示词")
    documents: Optional[List[dict]] = Field(default=[], description="与问题相关的数据片段列表")
    need_calculate_metrics: Optional[Any] = Field(default={}, description="需要计算的指标以及限制条件")
    disable_planning: Optional[bool] = Field(default=False, description="是否禁用planning")
    disable_extract: Optional[bool]=Field(default=False, description="是否禁用数据提取节点")
    model_id: Optional[str] = Field(default=ANALYST_DEFAULT_MODEL_ID, description="指定使用的模型ID")

def is_invalid_need_metrics(value):
    import json
    # 如果是空字符串或者None，视为无效
    if value is None or value == "" or value == {}:
        return False
    # 如果是字符串，需要可以被正确反序列化为JSON
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, dict) and not obj:
                return False
            return True
        except Exception:
            return True  # 不是好的json字符串，则invalid
    # 如果是dict，如果非空
    if isinstance(value, dict):
        return len(value) > 0
    # 如果是其他类型，例如list之类，认为invalid
    return True

# full agent 接口
@data_analyst_router.post(
    "/data_analyst/full", 
    response_model=DataAnalystAgentResponse,
    summary="调用coding agent进行问题分析，包含数据分析和计算，选择是否开启planning阶段",
    description="调用coding agent，输入问题，返回问题分析结果，包含数据分析和计算，选择是否开启planning阶段",
    responses={
        200: {
            "description": "执行成功",
            "model": DataAnalystAgentResponse
        },
    }
    )
async def call_full_agent(
    request: FullAgentRequest,
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """
    调用full agent，输入问题，返回执行结果或错误信息
    """
    async with metrics_context_session(auto_print=True, compact=True):
        try:
            answer_lang = resolve_answer_locale(accept_language)
            # 保证当 need_calculate_metrics 传入且为非空字符串或不合理json时，disable_planning 为 True
            if is_invalid_need_metrics(request.need_calculate_metrics):
                request.disable_planning = True

            response = await _full_agent_service.run_full_analysis_with_token_limit(
                question=request.question, 
                prompt=request.prompt,
                documents=request.documents,
                need_calculate_metrics=request.need_calculate_metrics,
                disable_planning=request.disable_planning,
                disable_extract=request.disable_extract,
                model_id=request.model_id,
                answer_language=answer_lang
            )
            return DataAnalystAgentResponse(
                code=0,
                data=response,
                error_message=get_localized_message("api_operation_success", answer_lang),
            )
        except Exception as e:
            logger.error(f"调用full agent时发生异常: {e}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            return DataAnalystAgentResponse(
                code=1,
                data=None,
                error_message=get_localized_message("api_operation_failed", answer_lang, error=str(e)),
            )


class ExecuteCodeRequest(BaseModel):
    code: str = Field(..., description="要执行的Python代码")
    description: str = Field(..., description="代码逻辑的描述，描述解决问题的计算步骤")

class ExecuteCodeResponse(BaseModel):
    success: bool = Field(..., description="是否执行成功")
    result: Optional[Any] = Field(None, description="执行结果（格式化的提示信息）")
    error_message: Optional[Any] = Field(None, description="错误信息")

# execute code 接口
@data_analyst_router.post(
    "/test/data_analyst/execute_code",
    response_model=ExecuteCodeResponse,
    summary="执行Python代码",
    description="测试接口，执行指定的Python代码，返回执行结果或错误信息，用于测试代码执行工具是否正常工作",
    responses={
        200: {
            "description": "执行完成",
            "model": ExecuteCodeResponse
        },
    }
)
async def execute_code(
    request: ExecuteCodeRequest,
):
    """
    执行Python代码
    :param code: 要执行的Python代码
    :param description: 代码逻辑的描述
    """
    try:
        table_import="""
# 导入表格工具
from table_utils_for_agent import read_table
from precision_calculator import quantize,add,sub,mul,div

"""
        executor = _get_executor_instance()
        code_model = PythonCodeModel(
            imports=table_import,
            code=request.code,
            description=request.description
        )
        exec_result = await executor.execute(code_model)

        # 根据执行结果拼接代码和结果给大模型
        success=exec_result.get("success")
        error=exec_result.get("error")
        output=exec_result.get("output")
        
        output_str = str(output).strip() if output is not None else ""
        # 检查代码执行成功但output为空的情况
        if success and not output_str:
            logger.warning(f"---execute_code_tool---: 代码执行成功但输出为空，原始output: {repr(output)}, success: {success}, error: {error}")
            success = False
            exec_result["success"] = False
            if not error:
                error = "Error-代码执行成功但没有输出结果，你的代码没有正确使用print()函数打印计算结果，或者打印对象为空（空列表，空字典等），请确认代码中的输出逻辑，修改后重新执行"
        return ExecuteCodeResponse(success=success, result=exec_result, error_message=error)
    except Exception as e:
        logger.error(f"执行代码时发生异常: {e}")
        return ExecuteCodeResponse(success=False, result=None, error_message=str(e))


class MetricQueryResponse(BaseModel):
    code: int = Field(..., description="返回码")
    error_message: Optional[str] = Field(None, description="错误信息")
    data: Optional[Dict[str, Any]] = Field(None, description="数据")

@data_analyst_router.get(
    "/test/data_analyst/metrics",
    response_model=MetricQueryResponse,
    summary="查询 Langfuse 性能指标",
    description="从 Langfuse 查询 Agent 执行的性能指标，支持统计结果和代码错误两种输出模式",
)
async def query_metrics(
    output: Literal["all", "statistics", "code_errors"] = Query(
        default="all",
        description="输出模式: all=全部, statistics=统计结果, code_errors=代码错误"
    ),
    include_text: bool = Query(default=False, description="是否包含格式化文本报告"),
    days: Optional[int] = Query(default=7, description="查询最近 N 天的数据"),
    limit: int = Query(default=100, description="每页数量限制"),
    max_pages: int = Query(default=10, description="最大翻页数"),
):
    """
    查询 Langfuse 性能指标
    
    - **output**: 输出模式选择
        - `all`: 输出统计结果和代码错误
        - `statistics`: 仅输出统计结果
        - `code_errors`: 仅输出代码错误
    - **include_text**: 是否在返回中包含格式化文本报告
    """
    try:
        result: Dict[str, Any] = {}

        include_statistics = output in ("all", "statistics")
        include_code_errors = output in ("all", "code_errors")

        metrics = _metric_query.query_metrics(
            days=days,
            limit=limit,
            max_pages=max_pages,
            include_statistics=include_statistics,
            include_code_errors=include_code_errors
        )

        if include_statistics:
            stats = metrics["statistics"]
            result["statistics"] = stats
            if include_text:
                result["statistics_text"] = _metric_query.format_statistics(stats)

        if include_code_errors:
            errors = metrics["code_errors"]
            result["code_errors"] = errors
            if include_text:
                result["code_errors_text"] = _metric_query.format_code_errors_table(errors)

        return MetricQueryResponse(code=0, data=result, error_message=None)
        
    except Exception:
        logger.exception("查询性能指标失败，请检查Langfuse连接")
        return MetricQueryResponse(code=1, data=None, error_message="查询性能指标失败，请检查Langfuse连接")


