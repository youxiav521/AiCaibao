import asyncio
import os
import threading
from typing import Any, Optional

import httpx
from loguru import logger

from src.utils import json_utils


class CallbackService:
    _queues: dict[str, "asyncio.Queue[Any]"] = {}
    _consumer_tasks: dict[str, asyncio.Task] = {}
    _interrupt_events: dict[str, asyncio.Event] = {}
    _active_batches: dict[str, int] = {}  # task_id -> 并发的 process_content_generate 调用数
    _generation: dict[str, int] = {}  # task_id -> 当前批次代际（每次 0→1 时递增）
    _sentinel: object = object()
    _lock = threading.Lock()

    @staticmethod
    def _get_callback_url() -> str:
        base_url = os.getenv("CONTENT_GENERATE_CALLBACK_URL")
        if base_url is None or base_url.strip() == "":
            raise RuntimeError(
                "环境变量 CONTENT_GENERATE_CALLBACK_URL 未配置。请在.env文件中设置该变量"
            )
        base_url = base_url.rstrip("/")
        return f"{base_url}/api/aitask/modelCallBack"

    @classmethod
    def send(
        cls,
        event: str,
        data: dict[str, Any],
        task_id: str,
        callback_url: Optional[str] = None,
    ) -> None:
        with cls._lock:
            if task_id not in cls._queues:
                cls._queues[task_id] = asyncio.Queue()
                cls._interrupt_events[task_id] = asyncio.Event()
            queue = cls._queues[task_id]
            need_start = (
                task_id not in cls._consumer_tasks
                or cls._consumer_tasks[task_id].done()
            )

        queue.put_nowait((event, data, task_id, callback_url))

        if need_start:
            loop = asyncio.get_running_loop()
            with cls._lock:
                if (
                    task_id not in cls._consumer_tasks
                    or cls._consumer_tasks[task_id].done()
                ):
                    cls._consumer_tasks[task_id] = loop.create_task(
                        cls._consumer(task_id)
                    )

    @classmethod
    async def _consumer(cls, task_id: str) -> None:
        with cls._lock:
            queue = cls._queues.get(task_id)
            interrupt_event = cls._interrupt_events.get(task_id)
            consumer_gen = cls._generation.get(task_id, 0)
        if queue is None:
            return

        try:
            while True:
                item = await queue.get()
                if item is cls._sentinel:
                    queue.task_done()
                    with cls._lock:
                        current_gen = cls._generation.get(task_id, 0)
                    if current_gen != consumer_gen:
                        logger.info(
                            "CallbackService consumer收到过期sentinel，忽略 "
                            "task_id={} consumer_gen={} current_gen={}",
                            task_id, consumer_gen, current_gen,
                        )
                        continue
                    break
                event, data, tid, callback_url = item
                try:
                    interrupt = await cls._send_one(event, data, tid, callback_url)
                    if interrupt:
                        if interrupt_event is not None:
                            interrupt_event.set()
                        logger.warning(
                            "CallbackService收到中断指令 task_id={}", tid
                        )
                except Exception:
                    logger.exception(
                        "CallbackService消费消息失败 task_id={} event={}", tid, event
                    )
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            logger.info("CallbackService consumer cancelled for task_id={}", task_id)
        except Exception:
            logger.exception(
                "CallbackService consumer异常退出 task_id={}", task_id
            )

    @classmethod
    async def _send_one(
        cls,
        event: str,
        data: dict[str, Any],
        task_id: str,
        callback_url: Optional[str] = None,
    ) -> bool:
        url: Optional[str] = None
        callback_payload: Optional[dict[str, Any]] = None

        try:
            url = callback_url if callback_url else cls._get_callback_url()
            data_json_str = json_utils.dumps(data, ensure_ascii=False)
            callback_payload = {"data": data_json_str, "event": event, "taskId": task_id}

            async with httpx.AsyncClient(timeout=30.0) as client:
                max_attempts = 3
                base_backoff_s = 4.0
                response: Optional[httpx.Response] = None
                request_content, request_headers = json_utils.request_content(
                    callback_payload
                )

                for attempt in range(1, max_attempts + 1):
                    try:
                        response = await client.post(
                            url, content=request_content, headers=request_headers
                        )
                        response.raise_for_status()
                        break
                    except httpx.HTTPError as e:
                        if attempt >= max_attempts:
                            raise
                        sleep_s = base_backoff_s * (2 ** (attempt - 1))
                        logger.warning(
                            "回调请求失败将重试: attempt={}/{} sleep_s={:.2f} "
                            "taskId={} event={} error={}",
                            attempt, max_attempts, sleep_s,
                            task_id, event, str(e),
                        )
                        await asyncio.sleep(sleep_s)

                interrupt_payload: Optional[dict[str, Any]] = None
                try:
                    interrupt_payload = json_utils.response_json(response)
                except Exception:
                    interrupt_payload = None
                    logger.error("回调返回JSON解析失败 task_id={}: {}", task_id, getattr(response, "text", ""))

                if isinstance(interrupt_payload, dict):
                    data_field = interrupt_payload.get("data")
                    if isinstance(data_field, dict):
                        resp_task_id = data_field.get("taskId")
                        interrupt = data_field.get("interrupt")
                        if resp_task_id is None or resp_task_id != task_id:
                            logger.error(
                                "回调返回taskId不匹配: 返回={} 请求={}",
                                resp_task_id, task_id,
                            )
                        elif interrupt is True:
                            logger.warning(
                                "收到中断指令 taskId={}", task_id,
                            )
                            return True
                    else:
                        logger.error(
                            "回调返回格式异常 data字段非字典 task_id={}", task_id,
                        )
                return False
        except httpx.HTTPError as e:
            logger.exception(
                "回调请求最终失败 taskId={} event={} error={}",
                task_id, event, str(e),
            )
        except Exception as e:
            logger.exception(
                "回调过程中发生未知错误 taskId={} event={} error={}",
                task_id, event, str(e),
            )
        return False

    @classmethod
    def is_interrupted(cls, task_id: str) -> bool:
        with cls._lock:
            event = cls._interrupt_events.get(task_id)
        if event is None:
            return False
        return event.is_set()

    @classmethod
    def register_batch(cls, task_id: str) -> None:
        """注册一个并发的 process_content_generate 批次，用于引用计数。
        当 count 从 0→1 时递增代际，防止过期的 drain 误清理新建的资源。"""
        with cls._lock:
            prev = cls._active_batches.get(task_id, 0)
            cls._active_batches[task_id] = prev + 1
            if prev == 0:
                cls._generation[task_id] = cls._generation.get(task_id, 0) + 1
            logger.debug(
                "CallbackService注册批次 task_id={} active_batches={} generation={}",
                task_id, cls._active_batches[task_id], cls._generation.get(task_id, 0),
            )

    @classmethod
    async def signal_task_end(cls, task_id: str) -> None:
        """通知一个批次结束；只有所有批次都结束后才真正关闭回调通道。"""
        with cls._lock:
            count = cls._active_batches.get(task_id, 0)
            if count <= 0:
                logger.warning(
                    "CallbackService.signal_task_end 但 active_batches 已为0 task_id={}",
                    task_id,
                )
                return
            count -= 1
            cls._active_batches[task_id] = count
            if count > 0:
                logger.info(
                    "CallbackService批次结束但仍活跃 task_id={} remaining_batches={}",
                    task_id, count,
                )
                return
            # 所有批次结束，开始关闭
            if task_id not in cls._queues:
                return
            queue = cls._queues[task_id]
        remaining = queue.qsize()
        logger.info(
            "CallbackService收到任务结束信号 task_id={} 剩余待推送event数={}",
            task_id, remaining,
        )
        queue.put_nowait(cls._sentinel)
        await cls._drain_and_cleanup(task_id)

    @classmethod
    async def _drain_and_cleanup(cls, task_id: str) -> None:
        with cls._lock:
            drain_gen = cls._generation.get(task_id, 0)
        try:
            with cls._lock:
                queue = cls._queues.get(task_id)
            if queue is not None:
                try:
                    await asyncio.wait_for(queue.join(), timeout=300.0)
                    logger.info(
                        "CallbackService全部event推送完成 task_id={}", task_id
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "CallbackService后台清理等待队列排空超时 task_id={}", task_id
                    )
            with cls._lock:
                current_gen = cls._generation.get(task_id, 0)
                current_count = cls._active_batches.get(task_id, 0)
            if current_gen != drain_gen or current_count > 0:
                logger.info(
                    "CallbackService跳过过期清理 task_id={} drain_gen={} current_gen={} "
                    "current_count={}",
                    task_id, drain_gen, current_gen, current_count,
                )
                return
            with cls._lock:
                task = cls._consumer_tasks.pop(task_id, None)
                cls._queues.pop(task_id, None)
                cls._interrupt_events.pop(task_id, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except Exception:
            logger.exception(
                "CallbackService _drain_and_cleanup 异常 task_id={}", task_id
            )
