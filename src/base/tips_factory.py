import json
from typing import Dict, Any, Optional
from loguru import logger
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
# 单例模式
class Tips(metaclass=SingletonMeta):
    def __init__(self):
        self.tips = self._loadConfig()

    def _loadConfig(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open("config/tips.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        except json.JSONDecodeError:
            raise ValueError("配置文件格式错误")
        
    def getTips(self, code :str,language :Optional[str] = None) -> str:
        if not language:
           language = "zh-CN" 
        """获取提示信息"""
        try:
            return self.tips.get(code).get(language)
        except Exception:
            logger.error(f"tips.json中提示信息未定义: {code},{language}")
            return "提示信息未定义" 

class TipException(Exception):
    """提示异常类"""
    def __init__(self, code:str):
        self.code = code
        super().__init__(self.code)
    
    def __str__(self):
        return f"异常码值: {self.code})"