"""简单的Hello单元测试示例"""
import pytest


def test_hello_world():
    """测试hello world基本功能"""
    assert "hello world" == "hello world"


def test_uppercase():
    """测试字符串大写转换"""
    assert "hello".upper() == "HELLO"


def test_addition():
    """测试加法运算"""
    assert 2 + 2 == 4
    assert 0 + 0 == 0
    assert -1 + 1 == 0


@pytest.mark.parametrize("input_str, expected", [
    ("", True),
    ("   ", True),
    ("hello", False),
    ("  hello  ", False),
])
def test_is_empty(input_str, expected):
    """参数化测试：检查字符串是否为空"""
    result = input_str.strip() == ""
    assert result == expected


class TestHelloOperations:
    """测试Hello相关操作的测试类"""
    
    def test_greeting(self):
        """测试问候语生成"""
        name = "Test"
        greeting = f"Hello, {name}!"
        assert greeting == "Hello, Test!"
    
    def test_multiple_words(self):
        """测试多个单词的hello组合"""
        words = ["Hello", "World", "from", "pytest"]
        result = " ".join(words)
        assert result == "Hello World from pytest"
