from loguru import logger
from arq import Retry

from src.ade.knowledge_graph.schema.kg_schema import FileInput
from src.ade.knowledge_graph.service.knowledge_graph_service import KnowledgeGraphService
from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService
from src.ade.knowledge_graph.utils.preprocess_file import build_file_inputs_from_folder
from src.base.distributed_lock import DistributedLock
from src.base.task_manager import TaskManager, TaskStatus
from src.utils import StopWatch


async def create_knowledge_graph(
    ctx: dict,
    folder_id: str,
    folder_name: str,
    files_data: list[dict],
    graph_id: str,
) -> dict | None:
    """知识图谱生成异步任务

    Args:
        ctx: ARQ 任务上下文，包含 job_id 等信息
        folder_id: 知识库ID
        folder_name: 知识库名称
        files_data: 文件数据列表（序列化后的FileInput）
        graph_id: 图谱ID

    Returns:
        dict: 包含图谱ID和统计信息的结果
    """
    job_id = ctx.get("job_id")
    job_try = ctx.get("job_try", 1)

    assert job_id is not None, "没有job_id"

    logger.info(
        "KG任务开始 | job_id={} folder_id={} folder_name={} files_count={} job_try={}",
        job_id, folder_id, folder_name, len(files_data), job_try,
    )

    # 🔍 调试日志：记录 Worker 接收到的文件数据详情
    logger.info("KG任务文件数据详情 | job_id={}", job_id)
    for f_data in files_data:
        f_id = f_data.get("id", "")
        f_name = f_data.get("name", "")
        elements = f_data.get("elements", [])
        logger.info("  - 文件: id={} name={} elements_count={}", f_id, f_name, len(elements))
        if elements:
            total_md_len = sum(len(e.get("md", "")) for e in elements)
            logger.info("    - 总md长度: {} chars", total_md_len)

    sw = StopWatch(f"kg_task_{job_id}")
    try:
        with sw.task("知识图谱生成任务"):
            await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)

            files = [FileInput(**f) for f in files_data]

            graph = await KnowledgeGraphService.generate(
                files=files,
                folder_id=folder_id,
                folder_name=folder_name,
                graph_id=graph_id,
            )

            result = {
                "graph_id": graph.id,
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
            }

            await TaskManager.update(
                job_id, TaskStatus.SUCCESS, result=result, attempts=job_try
            )

        sw.print()
        logger.info(
            "KG任务完成 | job_id={} graph_id={} entities个数={} relations个数={}",
            job_id, result["graph_id"], result["entity_count"], result["relation_count"],
        )

        return result

    except Exception as error:
        logger.exception("知识图谱Task任务失败 | job_id={} 尝试次数={}", job_id, job_try)

        if job_try >= 3:
            await TaskManager.update(
                job_id, TaskStatus.FAILED, error=str(error), attempts=job_try
            )
            logger.error("知识图谱Task任务失败{}次，最终标记为FAILED状态 | job_id={}", job_try, job_id)
        else:
            retry_delay = 10 if job_try == 1 else 30
            raise Retry(defer=retry_delay)


async def create_knowledge_graph_from_folder(
    ctx: dict,
    converted_folder: str,
    folder_id: str,
    folder_name: str,
    graph_id: str,
) -> dict | None:
    """从本地文件夹读取 converted.json 生成知识图谱

    文件夹结构约定：
        converted_folder/
        ├── 文档A/
        │   └── converted.json
        ├── 文档B/
        │   └── converted.json
        └── ...

    与 create_knowledge_graph 的区别：
        - 数据源是本地文件夹而非 HTTP 请求体
        - 内部调用 build_file_inputs_from_folder() 扫描文件夹

    Args:
        ctx: ARQ 任务上下文
        converted_folder: 存放 converted.json 的文件夹路径
        folder_id: 知识库ID
        folder_name: 知识库名称
        graph_id: 图谱ID

    Returns:
        dict: 包含图谱ID和统计信息的结果
    """
    job_id = ctx.get("job_id")
    job_try = ctx.get("job_try", 1)

    assert job_id is not None, "没有job_id"

    logger.info(
        "KG文件夹任务开始 | job_id={} folder_id={} folder={} job_try={}",
        job_id, folder_id, converted_folder, job_try,
    )

    sw = StopWatch(f"kg_folder_task_{job_id}")
    try:
        with sw.task("扫描文件夹并构建 FileInput"):
            await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)
            files = build_file_inputs_from_folder(converted_folder)

        logger.info("文件夹扫描完成 | files={}", len(files))

        with sw.task("知识图谱生成"):
            graph = await KnowledgeGraphService.generate(
                files=files,
                folder_id=folder_id,
                folder_name=folder_name,
                graph_id=graph_id,
            )

        result = {
            "graph_id": graph.id,
            "entity_count": len(graph.entities),
            "relation_count": len(graph.relations),
        }

        await TaskManager.update(
            job_id, TaskStatus.SUCCESS, result=result, attempts=job_try
        )

        sw.print()
        logger.info(
            "KG文件夹任务完成 | job_id={} graph_id={} entities={} relations={}",
            job_id, result["graph_id"], result["entity_count"], result["relation_count"],
        )

        return result

    except Exception as error:
        logger.exception("KG文件夹任务失败 | job_id={} folder={}", job_id, converted_folder)

        if job_try >= 3:
            await TaskManager.update(
                job_id, TaskStatus.FAILED, error=str(error), attempts=job_try
            )
        else:
            retry_delay = 10 if job_try == 1 else 30
            raise Retry(defer=retry_delay)


async def update_knowledge_graph(
    ctx: dict,
    folder_id: str,
    folder_name: str,
    files_data: list[dict],
    graph_id: str,
) -> dict | None:
    """知识图谱增量更新异步任务

    加载已存 graph，与 new_files 做 diff，增量提取并合并。
    若 graph 不存在则降级为全量 generate。

    Args:
        ctx: ARQ 任务上下文
        folder_id: 知识库ID
        folder_name: 知识库名称
        files_data: 全量最新文件列表（序列化后的 FileInput）
        graph_id: 图谱ID
    """
    job_id = ctx.get("job_id")
    job_try = ctx.get("job_try", 1)
    assert job_id is not None, "没有job_id"

    logger.info(
        "KG增量任务开始 | job_id={} folder_id={} files_count={} job_try={}",
        job_id, folder_id, len(files_data), job_try,
    )

    sw = StopWatch(f"kg_update_task_{job_id}")
    try:
        with sw.task("知识图谱增量更新"):
            await TaskManager.update(job_id, TaskStatus.RUNNING, attempts=job_try)
            files = [FileInput(**f) for f in files_data]

            existing_graph = await GraphStorageService.get(folder_id)

            if existing_graph is None:
                # 降级：图谱不存在，走全量 generate
                logger.info("KG增量任务降级为全量 | job_id={} folder_id={}", job_id, folder_id)
                graph = await KnowledgeGraphService.generate(
                    files=files,
                    folder_id=folder_id,
                    folder_name=folder_name,
                    graph_id=graph_id,
                )
            else:
                # 增量：folder 级分布式锁串行更新
                existing_graph.id = graph_id
                existing_graph.folderName = folder_name
                async with DistributedLock.acquire(f"kg:{folder_id}"):
                    graph = await KnowledgeGraphService.update(
                        graph=existing_graph,
                        new_files=files,
                    )

            result = {
                "graph_id": graph.id,
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
            }
            await TaskManager.update(
                job_id, TaskStatus.SUCCESS, result=result, attempts=job_try
            )

        sw.print()
        logger.info(
            "KG增量任务完成 | job_id={} graph_id={} entities={} relations={}",
            job_id, result["graph_id"], result["entity_count"], result["relation_count"],
        )
        return result

    except Exception as error:
        logger.exception("KG增量任务失败 | job_id={} 尝试次数={}", job_id, job_try)
        if job_try >= 3:
            await TaskManager.update(
                job_id, TaskStatus.FAILED, error=str(error), attempts=job_try
            )
            logger.error("KG增量任务失败{}次，最终标记为FAILED | job_id={}", job_try, job_id)
        else:
            retry_delay = 10 if job_try == 1 else 30
            raise Retry(defer=retry_delay)
