import time
import traceback
from contextlib import contextmanager
from typing import List, Dict, Optional, TypeVar, Awaitable
from loguru import logger

T = TypeVar('T')


class TaskInfo:
    """单个任务的计时信息"""
    def __init__(self, task_name: str, time_nanos: int):
        self.task_name = task_name
        self._time_nanos = time_nanos
        self.details: List[str] = []
        self.subtasks: List[TaskInfo] = []

    @property
    def time_nanos(self) -> int:
        return self._time_nanos

    @time_nanos.setter
    def time_nanos(self, value: int) -> None:
        self._time_nanos = value

    @property
    def time_millis(self) -> float:
        return self.time_nanos / 1_000_000

    @property
    def time_seconds(self) -> float:
        return self.time_nanos / 1_000_000_000

    def add_detail(self, detail: str) -> None:
        """添加任务详情信息"""
        self.details.append(detail)

    def add_subtask(self, subtask: 'TaskInfo') -> None:
        """添加子任务"""
        self.subtasks.append(subtask)


class TaskContext:
    """任务上下文，支持在执行过程中更新详情"""
    def __init__(self, stopwatch: 'StopWatch', task_name: str, parent_task_info: Optional[TaskInfo] = None):
        self._stopwatch = stopwatch
        self._task_name = task_name
        self._task_info: Optional[TaskInfo] = None
        self._parent_task_info = parent_task_info

    def detail(self, detail: str) -> None:
        """添加任务详情"""
        if self._task_info:
            self._task_info.add_detail(detail)

    @contextmanager
    def task(self, task_name: str):
        """
        上下文管理器：在当前任务中创建子任务
        :param task_name: 子任务名称
        用法：
        with stopwatch.task("parent") as parent:
            with parent.task("child"):
                # 子任务代码
        """
        self._stopwatch.start(task_name)
        context = TaskContext(self._stopwatch, task_name, self._task_info)
        try:
            task_info = TaskInfo(task_name, 0)  # 提前创建TaskInfo对象
            context._task_info = task_info
            yield context
        except BaseException as e:
            if task_info is not None:
                tb = e.__traceback__
                if tb is not None:
                    tb_frames = traceback.extract_tb(tb)
                    if tb_frames:
                        last_frame = tb_frames[-1]
                        file_location = f"{last_frame.filename}:{last_frame.lineno}"
                        task_info.add_detail(f"Exception at {file_location}: {e}")
            raise
        finally:
            duration_nanos = self._stopwatch.stop(task_name, return_duration_only=True)
            task_info.time_nanos = duration_nanos
            if self._task_info:
                self._task_info.add_subtask(task_info)
            else:
                # 如果没有父任务，直接添加到stopwatch
                self._stopwatch.task_list.append(task_info)


class StopWatch:
    """
    参考Spring StopWatch的计时工具类，用于统计多个任务的执行时间
    支持并行任务计时：总耗时为StopWatch创建到最后一个任务结束的时间差
    """

    def __init__(self, id: Optional[str] = None):
        self.id = id or ""
        self._created_at_nanos = time.perf_counter_ns()
        self.earliest_start_nanos: Optional[int] = None
        self.latest_end_nanos: Optional[int] = None
        self.task_list: List[TaskInfo] = []
        # 支持并行任务，存储当前运行中的任务（任务名称 -> 开始时间）
        self._running_tasks: Dict[str, int] = {}

    @property
    def running(self) -> bool:
        """检查StopWatch是否有正在运行的任务"""
        return len(self._running_tasks) > 0

    @property
    def current_task_name(self) -> Optional[str]:
        """获取当前运行中任务的名称（如果只有一个）"""
        if len(self._running_tasks) == 1:
            return next(iter(self._running_tasks.keys()))
        return None

    def start(self, task_name: str = "") -> None:
        """
        开始计时一个新任务
        :param task_name: 任务名称，用于标识不同的计时阶段
        """
        if task_name in self._running_tasks:
            raise ValueError(f"StopWatch中同名任务已经在运行中: {task_name}")
        
        start_time = time.perf_counter_ns()
        self._running_tasks[task_name] = start_time
        
        # 更新最早开始时间
        if self.earliest_start_nanos is None or start_time < self.earliest_start_nanos:
            self.earliest_start_nanos = start_time

    def stop(self, task_name: Optional[str] = None, return_duration_only: bool = False) -> None:
        """
        停止任务的计时
        :param task_name: 要停止的任务名称，如果为None且只有一个运行中的任务，则停止该任务
        :param return_duration_only: 是否只返回持续时间而不创建TaskInfo对象（用于上下文管理器）
        """
        if not self._running_tasks:
            raise ValueError("StopWatch当前没有在运行的任务")
        
        # 兼容旧API：如果未指定task_name且只有一个运行中的任务，停止该任务
        if task_name is None:
            if len(self._running_tasks) == 1:
                task_name = next(iter(self._running_tasks.keys()))
            else:
                raise ValueError("存在多个运行中的任务，请指定要停止的task_name")
        
        if task_name not in self._running_tasks:
            raise ValueError(f"StopWatch中没有正在运行的任务: {task_name}")
        
        end_time = time.perf_counter_ns()
        start_time = self._running_tasks.pop(task_name)
        duration_nanos = end_time - start_time
        
        # 更新最晚结束时间
        if self.latest_end_nanos is None or end_time > self.latest_end_nanos:
            self.latest_end_nanos = end_time
        
        if return_duration_only:
            return duration_nanos
        
        # 创建并添加任务信息
        task_info = TaskInfo(task_name, duration_nanos)
        self.task_list.append(task_info)
        
        return task_info

    def is_running(self) -> bool:
        """检查StopWatch是否有正在运行的任务"""
        return len(self._running_tasks) > 0
    
    def is_task_running(self, task_name: str) -> bool:
        """检查指定任务是否正在运行"""
        return task_name in self._running_tasks

    def get_total_time_nanos(self) -> int:
        """获取总耗时，单位纳秒：从StopWatch创建到最后一个任务结束的时间差"""
        if self.latest_end_nanos is None:
            if self._running_tasks:
                now = time.perf_counter_ns()
                return now - self._created_at_nanos
            return 0
        return self.latest_end_nanos - self._created_at_nanos

    def get_total_time_millis(self) -> float:
        """获取总耗时，单位毫秒"""
        return self.get_total_time_nanos() / 1_000_000

    def get_total_time_seconds(self) -> float:
        """获取总耗时，单位秒"""
        return self.get_total_time_nanos() / 1_000_000_000

    def get_task_info(self) -> List[TaskInfo]:
        """获取所有任务的计时信息"""
        return self.task_list.copy()

    def get_task_count(self) -> int:
        """获取已记录的任务数量"""
        return len(self.task_list)

    def pretty_print(self) -> str:
        """生成格式化的计时报告，适合打印输出"""
        if not self.task_list:
            return "StopWatch: 没有记录的任务"
        
        total_seconds = self.get_total_time_seconds()
        lines = []
        
        # 总耗时自动适应显示格式
        if total_seconds < 1:
            total_millis = self.get_total_time_millis()
            lines.append(f"StopWatch '{self.id}': 总耗时 = {total_millis:.1f} 毫秒")
        else:
            lines.append(f"StopWatch '{self.id}': 总耗时 = {total_seconds:.1f} 秒")
        
        lines.append("-" * 80)
        lines.append(f"{'耗时':>10} | {'占比':>7} | {'任务名称':<30} | {'详情':<20}")
        lines.append("-" * 80)
        
        total_nanos = self.get_total_time_nanos()
        
        def print_task(task: TaskInfo, indent: int = 0, parent_nanos: Optional[int] = None):
            task_seconds = task.time_seconds
            # 计算相对于父任务的百分比
            if parent_nanos:
                percentage = (task.time_nanos / parent_nanos) * 100 if parent_nanos > 0 else 0
            else:
                percentage = (task.time_nanos / total_nanos) * 100 if total_nanos > 0 else 0
            
            # 缩进和前缀
            prefix = "  " * indent
            if indent > 0:
                prefix = "  " * (indent - 1) + "├─ "
            
            # 每个任务自动适应显示格式
            if task_seconds < 1:
                time_str = f"{task.time_millis:>10.1f} 毫秒"
            else:
                time_str = f"{task_seconds:>10.1f} 秒"
            
            # 详情部分
            details_str = " | ".join(task.details) if task.details else ""
            
            # 构建行
            lines.append(f"{time_str} | {percentage:>7.1f}% | {prefix}{task.task_name:<{30 - indent*2}} | {details_str:<20}")
            
            # 递归打印子任务
            for subtask in task.subtasks:
                print_task(subtask, indent + 1, task.time_nanos)
        
        # 打印所有根任务
        for task in self.task_list:
            print_task(task)
        
        return "\n".join(lines)

    def print(self) -> None:
        """使用logger.info打印格式化的计时报告"""
        logger.info("\n" + self.pretty_print())

    def to_dict(self) -> Dict:
        """将计时结果转换为字典格式，便于序列化"""
        total_nanos = self.get_total_time_nanos()
        
        def task_to_dict(task: TaskInfo, parent_nanos: Optional[int] = None) -> Dict:
            # 计算相对于父任务的百分比
            if parent_nanos:
                percentage = (task.time_nanos / parent_nanos) * 100 if parent_nanos > 0 else 0
            else:
                percentage = (task.time_nanos / total_nanos) * 100 if total_nanos > 0 else 0
            
            return {
                "name": task.task_name,
                "time_seconds": task.time_seconds,
                "time_millis": task.time_millis,
                "percentage": percentage,
                "details": task.details,
                "subtasks": [task_to_dict(subtask, task.time_nanos) for subtask in task.subtasks]
            }
        
        return {
            "id": self.id,
            "total_time_seconds": self.get_total_time_seconds(),
            "total_time_millis": self.get_total_time_millis(),
            "task_count": self.get_task_count(),
            "tasks": [task_to_dict(task) for task in self.task_list]
        }

    @contextmanager
    def task(self, task_name: str):
        """
        上下文管理器：在with语句块中自动计时
        :param task_name: 任务名称
        用法：
        with stopwatch.task("task_name") as task:
            # 要计时的代码
            task.detail("处理了100条数据")
        """
        self.start(task_name)
        context = TaskContext(self, task_name)
        try:
            task_info = TaskInfo(task_name, 0)  # 提前创建TaskInfo对象
            context._task_info = task_info
            yield context
        except BaseException as e:
            if task_info is not None:
                tb = e.__traceback__
                if tb is not None:
                    tb_frames = traceback.extract_tb(tb)
                    if tb_frames:
                        last_frame = tb_frames[-1]
                        file_location = f"{last_frame.filename}:{last_frame.lineno}"
                        task_info.add_detail(f"Exception at {file_location}: {e}")
            raise
        finally:
            duration_nanos = self.stop(task_name, return_duration_only=True)
            if task_info is not None:
                task_info.time_nanos = duration_nanos
                self.task_list.append(task_info)

    async def time_task(self, task_name: str, coro: Awaitable[T]) -> T:
        """
        便捷方法：异步执行协程并自动计时
        :param task_name: 任务名称
        :param coro: 要执行的协程
        :return: 协程的执行结果
        """
        with self.task(task_name):
            return await coro
