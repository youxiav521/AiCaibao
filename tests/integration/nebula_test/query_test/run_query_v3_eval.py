from __future__ import annotations

import asyncio
import html
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from loguru import logger

from src.ade.knowledge_graph.evaluate.report import render_html  
from src.ade.knowledge_graph.nebula.client import NebulaClient  
from src.ade.knowledge_graph.nebula.repository import (  
    NebulaGraphRepository,
)
from src.ade.knowledge_graph.nebula.schema import (  
    BIDIRECTIONAL_RELATION_TYPES,
)
from src.ade.knowledge_graph.query.prompt import (  
    GRAPH_FAIL_RESPONSE,
    QUERY_FAIL_RESPONSE,
)
from src.ade.knowledge_graph.query.query_service_v3 import (  
    _ANCHOR_TOP_K,
    QueryServiceV3,
)
from src.ade.knowledge_graph.schema.kg_schema import (  
    Entity,
    KnowledgeGraph,
    Relation,
    entity_id,
    relation_id,
)
from src.ade.knowledge_graph.service.embed_service import EmbedService  
from src.ade.knowledge_graph.service.graph_storage_service import (  
    GraphStorageService,
)
from src.utils.json_utils import loads as json_loads  
from src.utils.model_factory import ModelFactory  


SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = SCRIPT_DIR.parent / "kg" / "37.json"
QUESTION_SET_PATH = SCRIPT_DIR / "test_set" / "礼光query_set.txt"
RESULT_DIR = SCRIPT_DIR / "result"
EVAL_FOLDER_ID = "kg-query-v3-eval"
EVAL_GRAPH_ID = "kg-query-v3-eval"
NO_MATCH_ANSWERS = {GRAPH_FAIL_RESPONSE, QUERY_FAIL_RESPONSE}


def _load_questions() -> list[str]:
    questions = [
        line.strip()
        for line in QUESTION_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise ValueError(f"问题集为空: {QUESTION_SET_PATH}")
    return questions


def _load_isolated_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph.model_validate(json_loads(SNAPSHOT_PATH.read_bytes()))
    graph.id = EVAL_GRAPH_ID
    graph.folderId = EVAL_FOLDER_ID
    graph.snapshot_version = ""

    id_map: dict[str, str] = {}
    new_ids: set[str] = set()
    for entity in graph.entities:
        if not entity.id or entity.id in id_map:
            raise ValueError(f"原始实体 ID 缺失或重复: {entity.id!r}")
        new_id = entity_id(EVAL_FOLDER_ID, entity.name, entity.type)
        if new_id in new_ids:
            raise ValueError(f"隔离实体 ID 冲突: {new_id!r}")
        id_map[entity.id] = new_id
        new_ids.add(new_id)
        entity.id = new_id
        entity.embed = None

    for relation in graph.relations:
        try:
            relation.from_entity_id = id_map[relation.from_entity_id]
            relation.to_entity_id = id_map[relation.to_entity_id]
        except KeyError as error:
            raise ValueError(f"关系端点不存在: {relation.id!r}") from error
        id_source, id_target = relation.from_entity_id, relation.to_entity_id
        if relation.type in BIDIRECTIONAL_RELATION_TYPES:
            id_source, id_target = sorted((id_source, id_target))
        relation.id = relation_id(id_source, id_target, relation.type)
        relation.embed = None

    return graph


def _preflight_nebula() -> None:
    if not NebulaClient.is_configured():
        raise RuntimeError("NEBULA_HOST 未配置")
    for session_factory in (NebulaClient.session, NebulaClient.query_session):
        with session_factory() as session:
            NebulaClient.execute(session, "YIELD 1 AS ok")


def _preformatted(value: str) -> str:
    return f"<pre>{html.escape(value, quote=False)}</pre>\n\n"


def _details(title: str, value: str) -> str:
    return (
        "<details>\n"
        f"<summary>{title}</summary>\n\n"
        f"{_preformatted(value)}"
        "</details>\n\n"
    )


def _write_html_report(markdown_path: Path) -> Path:
    html_path = markdown_path.with_suffix(".html")
    html_path.write_text(
        render_html(
            markdown_path.read_text(encoding="utf-8"),
            title="KG Query V3 手工评测报告",
        ),
        encoding="utf-8",
    )
    return html_path


def _entity_diagnostics(entities: list[Entity]) -> list[dict]:
    return [
        {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "attributes": entity.attributes,
            "reference_count": len(entity.references),
        }
        for entity in entities
    ]


def _relation_diagnostics(relations: list[Relation]) -> list[dict]:
    return [
        {
            "id": relation.id,
            "type": relation.type,
            "from_entity": relation.from_entity,
            "from_entity_id": relation.from_entity_id,
            "to_entity": relation.to_entity,
            "to_entity_id": relation.to_entity_id,
            "reference_count": len(relation.references),
        }
        for relation in relations
    ]


def _write_question_result(
    report: TextIO,
    *,
    index: int,
    question: str,
    status: str,
    elapsed_seconds: float,
    answer: str = "",
    context: str = "",
    entities: list[Entity] | None = None,
    relations: list[Relation] | None = None,
    stopwatch_report: str = "",
    error_traceback: str = "",
) -> None:
    report.write(f"## {index}. {question}\n\n")
    report.write(f"- 状态：{status}\n")
    report.write(f"- 耗时：{elapsed_seconds:.2f} 秒\n\n")
    if stopwatch_report:
        report.write(_details("阶段耗时（含占比）", stopwatch_report.strip()))
    if answer:
        report.write("### 回答\n\n")
        report.write(_preformatted(answer))
    if context:
        report.write(_details("查询上下文（含 nGQL）", context))
    if entities is not None:
        report.write(
            _details(
                f"命中实体（{len(entities)}）",
                json.dumps(
                    _entity_diagnostics(entities),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            )
        )
    if relations is not None:
        report.write(
            _details(
                f"命中关系（{len(relations)}）",
                json.dumps(
                    _relation_diagnostics(relations),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            )
        )
    if error_traceback:
        report.write(_details("异常堆栈", error_traceback))
    report.flush()


async def _run() -> int:
    questions = _load_questions()
    graph = _load_isolated_graph()
    started_at = datetime.now().astimezone()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULT_DIR / (
        f"query_v3_{started_at.strftime('%Y%m%d_%H%M%S')}.md"
    )

    with report_path.open("x", encoding="utf-8") as report:
        report.write("# KG Query V3 手工评测报告\n\n")
        report.write(f"- 开始时间：{started_at.isoformat(timespec='seconds')}\n")
        report.write(f"- 原始快照：{SNAPSHOT_PATH.relative_to(PROJECT_ROOT)}\n")
        report.write(f"- 问题集：{QUESTION_SET_PATH.relative_to(PROJECT_ROOT)}\n")
        report.write(f"- 隔离 folder：`{EVAL_FOLDER_ID}`\n")
        report.write(f"- 问题数量：{len(questions)}\n")
        report.flush()

        try:
            await asyncio.to_thread(_preflight_nebula)
            await NebulaGraphRepository.initialize_schema()
            await GraphStorageService.save(graph)
        except Exception:
            logger.exception(
                "KG Query V3 手工评测初始化失败 | report_path={}",
                report_path,
            )
            report.write("\n## 初始化失败\n\n")
            report.write(_preformatted(traceback.format_exc()))
            report.flush()
            _write_html_report(report_path)
            return 1

        report.write(f"- 图谱 ID：`{graph.id}`\n")
        report.write(f"- 快照版本：`{graph.snapshot_version}`\n")
        report.write(f"- 实体/关系：{len(graph.entities)}/{len(graph.relations)}\n")
        report.write(
            f"- Query/Embedding 模型："
            f"`{ModelFactory.resolved_model_name_for_key(None)}` / "
            f"`{EmbedService.MODEL or ''}`\n"
        )
        report.write(
            "- V3 参数："
            f"`max_tokens={QueryServiceV3.DEFAULT_MAX_TOKENS}`, "
            f"`anchor_top_k={_ANCHOR_TOP_K}`, "
            f"`anchor_min_score={os.environ.get('KG_ANCHOR_MIN_SCORE', '0.5')}`, "
            f"`max_rows={os.environ.get('KG_NGQL_MAX_ROWS', '200')}`\n\n"
        )
        report.flush()

        success_count = 0
        no_match_count = 0
        error_count = 0
        for index, question in enumerate(questions, 1):
            logger.info(
                "KG Query V3 手工评测开始 | index={} total={} question={}",
                index,
                len(questions),
                question,
            )
            query_started = time.perf_counter()
            stopwatch_reports: list[str] = []
            sink_id = logger.add(
                lambda message: stopwatch_reports.append(message.record["message"]),
                filter=lambda record: record["message"].startswith(
                    f"\nStopWatch 'QueryServiceV3-{EVAL_FOLDER_ID}'"
                ),
            )
            try:
                result, context = await QueryServiceV3.query(
                    EVAL_FOLDER_ID,
                    question,
                    return_context=True,
                )
            except Exception:
                elapsed = time.perf_counter() - query_started
                error_count += 1
                logger.exception(
                    "KG Query V3 手工评测异常 | index={} question={}",
                    index,
                    question,
                )
                _write_question_result(
                    report,
                    index=index,
                    question=question,
                    status="异常",
                    elapsed_seconds=elapsed,
                    stopwatch_report="\n\n".join(stopwatch_reports),
                    error_traceback=traceback.format_exc(),
                )
                continue
            finally:
                logger.remove(sink_id)

            elapsed = time.perf_counter() - query_started
            status = "无匹配" if result.answer in NO_MATCH_ANSWERS else "成功"
            if status == "无匹配":
                no_match_count += 1
            else:
                success_count += 1
            _write_question_result(
                report,
                index=index,
                question=question,
                status=status,
                elapsed_seconds=elapsed,
                answer=result.answer,
                context=context,
                entities=result.entities,
                relations=result.relations,
                stopwatch_report="\n\n".join(stopwatch_reports),
            )
            logger.info(
                "KG Query V3 手工评测完成 | index={} status={} elapsed={:.2f}",
                index,
                status,
                elapsed,
            )

        finished_at = datetime.now().astimezone()
        report.write("## 汇总\n\n")
        report.write(f"- 成功：{success_count}\n")
        report.write(f"- 无匹配：{no_match_count}\n")
        report.write(f"- 异常：{error_count}\n")
        report.write(f"- 结束时间：{finished_at.isoformat(timespec='seconds')}\n")
        report.flush()

    html_report_path = _write_html_report(report_path)
    logger.info(
        "KG Query V3 手工评测报告已生成 | html_path={} markdown_path={} errors={}",
        html_report_path,
        report_path,
        error_count,
    )
    return 1 if error_count else 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception:
        logger.exception("KG Query V3 手工评测脚本失败")
        return 1
    finally:
        NebulaClient.close()


if __name__ == "__main__":
    raise SystemExit(main())
