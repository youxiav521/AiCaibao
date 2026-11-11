from __future__ import annotations

import os
import sys
import traceback


def cpython_default_max_workers() -> int:
    """返回 CPython 默认线程池的 `max_workers` 计算结果。

    Python 在未显式指定默认线程池大小时，会使用一套固定公式：
    `min(32, os.cpu_count() + 4)`。这里单独封装出来有两个目的：

    1. 让监控层在“替换 loop 默认 executor”时保持与 CPython 语义一致。
    2. 让默认值逻辑有独立入口，便于测试和后续扩展。

    Returns:
        int: 与 CPython 默认行为一致的线程池最大工作线程数。
    """
    return min(32, (os.cpu_count() or 1) + 4)


def capture_thread_stack(thread_id: int, *, limit: int) -> list[str]:
    """抓取指定线程的 Python 栈。

    该函数用于 event loop 阻塞监控。watchdog 线程在检测到主线程长时间
    未推进时，会直接抓取主线程当前 Python 栈，作为阻塞现场信息写入
    runtime span，便于后续定位“卡在什么调用栈”。

    Args:
        thread_id: 目标线程的 `ident`。
        limit: 最多保留的栈帧数。

    Returns:
        list[str]: 已格式化的栈帧文本列表；如果线程不存在则返回空列表。
    """
    frame = sys._current_frames().get(thread_id)
    if frame is None:
        return []
    return traceback.format_stack(frame, limit=max(1, limit))
