import faulthandler
faulthandler.enable()

from dotenv import load_dotenv
load_dotenv()

# ── 第一步：配置 & 日志（必须最早初始化，后续所有模块 import 时的日志才能正确输出）──
from src.config import config as appConfig
from src.base.logger_factory import initLogger, addContextTextHandler
from loguru import logger

initLogger(appConfig)
addContextTextHandler()

# ── 第二步：OTel / Langfuse（依赖日志已就绪；其 stdlib logging 由 InterceptHandler 接管）──
from src.middleware.otel_langfuse import setup_otel_langfuse, instrument_fastapi
from src.middleware.yusys_tracing import setup_yusys_tracing

setup_otel_langfuse(
    service_name="yusys-ai",
    enabled=bool(getattr(getattr(appConfig, "monitor", None), "enabled", False)),
)

# 独立 yusys_tracing provider：只用于 src/agents.* 的装饰器 span，不切换全局 Provider
setup_yusys_tracing(appConfig.tracing)

# ── 第三步：其余 import ────────────────────────────────────────────────────────
import os
import importlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.base.nacos_factory import NacosManager
from src.base.redis_client import RedisClient
from src.base.tips_factory import TipException, Tips
from src.monitoring import RuntimeMonitoringManager
from src.applications.rag.services.vectorstore import (
    close_default_stores,
    ensure_default_stores_initialized,
)

from .applications.data_analyst.routers.data_analyst_router import data_analyst_router
from .applications.data_analyst.routers.table_agent_router import table_agent_router
from .applications.converter.routers.converter_router import converter_router
from .applications.spiredoc_convert.routers.file_convert_router import spiredoc_router
from .applications.parsing.routers.finreport_router import router as finreport_router
from .applications.parsing.routers.subject_match_router import router as subject_match_router
from .applications.subject_match_i18n.routers.match_router import router as subject_match_i18n_router
from .applications.rag.routers.rag_router import router as rag_router
from .applications.rag.routers.structure_recall_router import router as structure_recall_router
from .applications.rag.mcp.rag_mcp_server import mcp_app as rag_mcp_app
from .applications.doc_extract.routers.doc_extract_router import router as doc_extract_router
from .applications.doc_extract.routers.fin_extract_router import router as fin_extract_router
from .applications.doc_extract.routers.doc_extract_v2_router import router as doc_extract_v2_router
from .applications.doc_classify.routers.doc_classify_router import router as doc_classify_router
from .applications.chat_answer.routers.chat_router import router as chat_router
from .applications.parsing.routers.parse_minerU_router import router as parsing_mineru_router
from .applications.parsing.routers.financial_extract_router import router as financial_extract_router
from .applications.image_classify.routers.image_classify_router import router as image_classify_router
from .applications.tag.routers.tag_router import router as tag_router
from .applications.content_generate.routers.content_generate_router import router as content_generate_router
from .applications.policy_retractor.routers.policy_retractor_router import router as policy_retractor_router
from .applications.content_generate.routers.light_qa_router import router as content_generate_light_qa_router
from src.agents.template.router import template_router
from src.agents.policy.policy_check_agent.router import policy_check_router

from src.ade.extract.router.extract_router import router as ade_extract_router
from src.ade.parse.router.parse_router import router as ade_parse_router
from src.ade.finacial.router import router as ade_finacial_router
from src.ade.knowledge_graph.router.kg_router import router as kg_router
from src.ade.jobs.router import router as ade_jobs_router
from src.ade.health.router import router as ade_health_router

# ── 生命周期钩子 ──────────────────────────────────────────────────────────────
runtime_monitoring_manager = RuntimeMonitoringManager(appConfig)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # fastmcp 3.x 要求把 MCP 子应用的 lifespan 链入父应用
    async with rag_mcp_app.lifespan(app):
        nacosManager = None
        try:
            await runtime_monitoring_manager.start()

            # 初始化Redis（两个客户端：str模式 + bytes模式），验证连通性，失败则阻止启动
            logger.info("开始初始化Redis连接")
            await RedisClient.get_client(decode_responses=True)
            await RedisClient.get_client(decode_responses=False)
            logger.info("Redis 初始化完成")

            logger.info("开始初始化 RAG 向量集合")
            await ensure_default_stores_initialized()
            logger.info("RAG 向量集合初始化完成")

            # 为 ade-jobs bucket 设置生命周期（结果保留 ≥24h 后自动清理）
            # docs/new.md 第 4 节 + ade_jobs_dev_plan.md 第 6.3 节要求
            try:
                from src.base.storage_service import storage_service
                ade_jobs_bucket = os.getenv("ADE_JOBS_BUCKET", "ade-jobs")
                ade_jobs_ttl_days = int(os.getenv("ADE_JOBS_RESULT_TTL_DAYS", "1"))
                await storage_service.set_bucket_lifecycle(
                    ade_jobs_bucket, expiration_days=ade_jobs_ttl_days,
                )
                logger.info(
                    "ade-jobs bucket 生命周期已设置 | bucket={} ttl_days={}",
                    ade_jobs_bucket, ade_jobs_ttl_days,
                )
            except Exception:
                logger.exception("ade-jobs bucket 生命周期设置失败（不影响启动）")

                nacosManager = NacosManager(config=appConfig)
                nacosManager.register()

            yield
        finally:
            if nacosManager is not None:
                try:
                    nacosManager.deregister()
                except Exception:
                    logger.exception("Nacos 注销失败")
            try:
                await RedisClient.close()
            except Exception:
                logger.exception("关闭 Redis 连接失败")
            try:
                await runtime_monitoring_manager.stop()
            except Exception:
                logger.exception("停止 runtime 监控失败")
            try:
                await close_default_stores()
            except Exception:
                logger.exception("关闭 RAG 向量库客户端失败")
            try:
                from src.applications.parsing.cell_traceability.schema_match.content_match import close_shared_client

                await close_shared_client()
            except Exception:
                logger.exception("关闭 content_match 连接池失败")


# ── FastAPI 应用 ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=appConfig.server.name,
    version=appConfig.server.version,
    lifespan=lifespan,
    docs_url="/xxx",
    redoc_url="/xxx",
    default_response_class=ORJSONResponse,
)

instrument_fastapi(app)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if appConfig.prometheus.enabled:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)


# ── 路由注册 ──────────────────────────────────────────────────────────────────
def registerRouters():
    """自动扫描并注册路由"""
    router_dir = os.path.join(os.path.dirname(__file__), 'router')
    if not os.path.exists(router_dir):
        return
    for filename in os.listdir(router_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]
            full_module_name = f'src.router.{module_name}'
            try:
                module = importlib.import_module(full_module_name)
                if hasattr(module, 'router'):
                    app.include_router(module.router)
                    logger.info("成功注册路由 | module={}", full_module_name)
                else:
                    logger.warning("模块没有 router 对象 | module={}", full_module_name)
            except Exception:
                logger.exception("导入路由模块失败 | module={}", full_module_name)


registerRouters()

# ── 初始化 ────────────────────────────────────────────────────────────────────
tips = Tips()


logger.info("应用启动完成")


# ── 异常拦截 ──────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def exceptionHandler(request: Request, exception: Exception):
    if isinstance(exception, TipException):
        return ORJSONResponse(
            status_code=500,
            content={
                "message": tips.getTips(exception.code, language=request.headers.get("Accept-Language", "zh-CN")),
                "code": exception.code
            }
        )

    logger.opt(exception=True).error("未处理异常 | url={}", request.url.path)
    return ORJSONResponse(
        status_code=500,
        content={
            "message": str(exception),
            "code": 500
        }
    )


# ── Swagger 本地化 ────────────────────────────────────────────────────────────
@app.get("/swagger", include_in_schema=False)
async def custom_swagger_ui():
    if appConfig.swagger.enabled == False:
        raise RuntimeError("swagger已禁用")
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css"
    )


# ── 健康检查 ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"state": "ok", "version": appConfig.server.version, "name": appConfig.server.name}


# ── 路由挂载 ──────────────────────────────────────────────────────────────────
app.include_router(converter_router)
app.include_router(spiredoc_router)
app.include_router(data_analyst_router)
app.include_router(table_agent_router)
app.include_router(parsing_mineru_router)
app.include_router(financial_extract_router)
app.include_router(image_classify_router)
app.include_router(finreport_router)
app.include_router(subject_match_router)
app.include_router(subject_match_i18n_router)
app.include_router(rag_router, prefix="/api/rag", tags=["rag"])
app.include_router(structure_recall_router, prefix="/api/rag", tags=["rag"])
# RAG MCP 服务（fastmcp 子应用），暴露 /api/rag/mcp
app.mount("/api/rag", rag_mcp_app)
app.include_router(tag_router)
app.include_router(doc_extract_router)

# ADE 结构化提取
app.include_router(ade_extract_router)

# ADE 文档解析
app.include_router(ade_parse_router)

# ADE 财报抽取（基于 file_id 的元数据 + 三大表）
app.include_router(ade_finacial_router)

# 注册财报结构化提取路由
app.include_router(fin_extract_router)
app.include_router(doc_extract_v2_router)
app.include_router(doc_classify_router)
app.include_router(chat_router)
app.include_router(template_router)
app.include_router(content_generate_router)
app.include_router(content_generate_light_qa_router)
app.include_router(policy_retractor_router)
app.include_router(policy_check_router)

# 知识图谱
app.include_router(kg_router)

# ADE Jobs（docs/new.md 规范的对外解析接口）
app.include_router(ade_jobs_router)

# ADE 健康检查
app.include_router(ade_health_router)


if __name__ == "__main__":
    import uvicorn

    # 本地调试入口：IDEA 中直接 Debug 运行 src/main.py 即可启动 FastAPI。
    # 依赖 app 根目录在 PYTHONPATH 中（IDEA 里将 app 目录标记为 Sources Root）。
    # reload 保持 False：reload 会以子进程方式热重载，容易导致断点不命中。
    uvicorn.run(app, host="127.0.0.1", port=6300, reload=False)
