import asyncio
import time
import pytest
from src.utils import StopWatch, TaskInfo


class TestStopWatch:
    """StopWatch工具类单元测试"""

    def test_initialization(self):
        """测试初始化功能"""
        sw = StopWatch()
        assert sw.id == ""
        assert sw.running is False
        assert sw.get_task_count() == 0
        assert sw.get_total_time_nanos() == 0

        sw_with_id = StopWatch("test_id")
        assert sw_with_id.id == "test_id"

    def test_basic_timing(self):
        """测试基础计时功能"""
        sw = StopWatch()
        sw.start("task1")
        assert sw.running is True
        assert sw.current_task_name == "task1"
        
        time.sleep(0.01)  # 休眠10ms
        
        sw.stop()
        assert sw.running is False
        assert sw.get_task_count() == 1
        assert sw.get_total_time_millis() >= 10  # 至少10ms
        assert sw.get_total_time_seconds() >= 0.01

        tasks = sw.get_task_info()
        assert len(tasks) == 1
        assert tasks[0].task_name == "task1"
        assert tasks[0].time_millis >= 10

    def test_multiple_tasks(self):
        """测试多任务计时"""
        sw = StopWatch()
        
        sw.start("task1")
        time.sleep(0.01)
        sw.stop()
        
        sw.start("task2")
        time.sleep(0.02)
        sw.stop()
        
        sw.start("task3")
        time.sleep(0.03)
        sw.stop()
        
        assert sw.get_task_count() == 3
        total_time = sw.get_total_time_millis()
        assert total_time >= 60  # 10+20+30 = 60ms
        
        tasks = sw.get_task_info()
        assert tasks[0].task_name == "task1"
        assert tasks[1].task_name == "task2"
        assert tasks[2].task_name == "task3"
        
        # 验证百分比计算
        assert abs(tasks[0].time_millis / total_time - 1/6) < 0.1  # ~16.6%
        assert abs(tasks[1].time_millis / total_time - 2/6) < 0.1  # ~33.3%
        assert abs(tasks[2].time_millis / total_time - 3/6) < 0.1  # ~50%

    def test_start_when_running_raises_error(self):
        """测试同名任务已经运行时再次调用start抛出异常"""
        sw = StopWatch()
        sw.start("task1")
        
        with pytest.raises(ValueError, match="StopWatch中同名任务已经在运行中: task1"):
            sw.start("task1")
        
        sw.stop()

    def test_stop_when_not_running_raises_error(self):
        """测试未运行时调用stop抛出异常"""
        sw = StopWatch()
        
        with pytest.raises(ValueError, match="StopWatch当前没有在运行"):
            sw.stop()

    def test_pretty_print_format(self):
        """测试格式化输出功能"""
        sw = StopWatch("test_print")
        
        sw.start("task1")
        time.sleep(0.01)
        sw.stop()
        
        sw.start("task2")
        time.sleep(0.02)
        sw.stop()
        
        output = sw.pretty_print()
        assert "StopWatch 'test_print': 总耗时 =" in output
        assert "毫秒" in output
        assert "占比" in output
        assert "任务名称" in output
        assert "task1" in output
        assert "task2" in output
        assert "%" in output

    def test_print_method(self):
        """测试print方法功能正确性（间接验证，通过pretty_print）"""
        sw = StopWatch("test_log")
        sw.start("task1")
        time.sleep(0.01)
        sw.stop()
        
        # print方法内部调用pretty_print并输出到日志，我们直接验证pretty_print的结果
        output = sw.pretty_print()
        assert "StopWatch 'test_log'" in output
        assert "task1" in output
        assert "总耗时 =" in output

    def test_to_dict_method(self):
        """测试转换为字典功能"""
        sw = StopWatch("test_dict")
        
        sw.start("task1")
        time.sleep(0.01)
        sw.stop()
        
        sw.start("task2")
        time.sleep(0.02)
        sw.stop()
        
        result = sw.to_dict()
        assert result["id"] == "test_dict"
        assert result["total_time_seconds"] >= 0.03
        assert result["total_time_millis"] >= 30
        assert result["task_count"] == 2
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["name"] == "task1"
        assert result["tasks"][1]["name"] == "task2"
        assert "percentage" in result["tasks"][0]
        assert "time_seconds" in result["tasks"][0]
        assert "time_millis" in result["tasks"][0]

    @pytest.mark.asyncio
    async def test_time_task_method(self):
        """测试异步time_task便捷方法"""
        sw = StopWatch()
        
        async def async_task(delay):
            await asyncio.sleep(delay)
            return "task_result"
        
        # 测试正常执行
        result = await sw.time_task("async_task1", async_task(0.01))
        assert result == "task_result"
        assert sw.get_task_count() == 1
        assert sw.get_total_time_millis() >= 10
        
        # 测试多个任务
        result2 = await sw.time_task("async_task2", async_task(0.02))
        assert result2 == "task_result"
        assert sw.get_task_count() == 2
        assert sw.get_total_time_millis() >= 30
        
        tasks = sw.get_task_info()
        assert tasks[0].task_name == "async_task1"
        assert tasks[1].task_name == "async_task2"

    @pytest.mark.asyncio
    async def test_time_task_with_exception(self):
        """测试time_task方法在协程抛出异常时仍能正确停止计时"""
        sw = StopWatch()
        
        async def failing_task():
            await asyncio.sleep(0.01)
            raise RuntimeError("Task failed")
        
        with pytest.raises(RuntimeError, match="Task failed"):
            await sw.time_task("failing_task", failing_task())
        
        # 确保计时已经停止且任务被记录
        assert sw.running is False
        assert sw.get_task_count() == 1
        assert sw.get_total_time_millis() >= 10
        assert sw.get_task_info()[0].task_name == "failing_task"

    def test_task_info_properties(self):
        """测试TaskInfo类的属性转换"""
        task = TaskInfo("test", 1_000_000_000)  # 1秒
        assert task.time_nanos == 1_000_000_000
        assert task.time_millis == 1000.0
        assert task.time_seconds == 1.0

        task2 = TaskInfo("test2", 123_456_789)  # 0.123456789秒
        assert abs(task2.time_millis - 123.456789) < 0.0001
        assert abs(task2.time_seconds - 0.123456789) < 0.0000001

    def test_empty_stopwatch_pretty_print(self):
        """测试空StopWatch的格式化输出"""
        sw = StopWatch()
        output = sw.pretty_print()
        assert "没有记录的任务" in output

    def test_zero_time_percentage(self):
        """测试总耗时为0时的百分比计算（避免除零错误）"""
        sw = StopWatch()
        
        sw.start("fast_task")
        sw.stop()
        
        result = sw.to_dict()
        # 确保不会抛出除零异常，百分比应在 0~100 范围内
        assert 0 <= result["tasks"][0]["percentage"] <= 100

    def test_context_manager(self):
        """测试上下文管理器功能"""
        sw = StopWatch()
        
        with sw.task("context_task"):
            time.sleep(0.01)
        
        assert sw.get_task_count() == 1
        assert sw.get_total_time_millis() >= 10
        assert sw.get_task_info()[0].task_name == "context_task"

    def test_context_manager_with_exception(self):
        """测试上下文管理器在异常情况下仍能正确停止计时"""
        sw = StopWatch()
        
        with pytest.raises(RuntimeError, match="Test exception"):
            with sw.task("failing_context_task"):
                time.sleep(0.01)
                raise RuntimeError("Test exception")
        
        # 确保计时已经停止且任务被记录
        assert sw.running is False
        assert sw.get_task_count() == 1
        assert sw.get_total_time_millis() >= 10
        assert sw.get_task_info()[0].task_name == "failing_context_task"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """测试上下文管理器在异步代码中的使用"""
        sw = StopWatch()
        
        with sw.task("async_context_task"):
            await asyncio.sleep(0.01)
        
        assert sw.get_task_count() == 1
        assert sw.get_total_time_millis() >= 10
        assert sw.get_task_info()[0].task_name == "async_context_task"

    def test_subtasks(self):
        """测试子任务功能"""
        sw = StopWatch()
        
        with sw.task("parent_task") as parent:
            time.sleep(0.01)  # 父任务耗时
            
            with parent.task("child_task_1") as child1:
                time.sleep(0.01)  # 子任务1耗时
                child1.detail("子任务1详情1")
                child1.detail("子任务1详情2")
            
            with parent.task("child_task_2") as child2:
                time.sleep(0.02)  # 子任务2耗时
                child2.detail("子任务2详情")
        
        # 验证父任务存在
        assert sw.get_task_count() == 1
        parent_task = sw.get_task_info()[0]
        assert parent_task.task_name == "parent_task"
        assert parent_task.time_millis >= 40  # 10+10+20
        
        # 验证子任务存在
        assert len(parent_task.subtasks) == 2
        child1 = parent_task.subtasks[0]
        child2 = parent_task.subtasks[1]
        
        assert child1.task_name == "child_task_1"
        assert child1.time_millis >= 10
        assert len(child1.details) == 2
        
        assert child2.task_name == "child_task_2"
        assert child2.time_millis >= 20
        assert len(child2.details) == 1
        
        # 验证字典输出包含子任务
        result = sw.to_dict()
        assert len(result["tasks"]) == 1
        assert len(result["tasks"][0]["subtasks"]) == 2
        assert result["tasks"][0]["subtasks"][0]["name"] == "child_task_1"
        assert result["tasks"][0]["subtasks"][1]["name"] == "child_task_2"
        
        # 验证格式化输出包含子任务信息
        output = sw.pretty_print()
        assert "parent_task" in output
        assert "child_task_1" in output
        assert "child_task_2" in output
        assert "子任务1详情1" in output
        assert "|" in output  # 验证使用|作为分隔符
