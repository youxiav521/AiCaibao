# 模型ID的全局配置文件
import json
import os
from threading import Lock

class ModelInvokeConfigSingleton:
    """
    单例方式管理大模型ID的配置，并允许动态更改和读取
    """
    _instance = None
    _lock = Lock()
    # 设置默认模型ID的常量
    default_mode_id = "data_analyst_agent"

    def __new__(cls, *args, **kwargs):
        # 保证单例
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._model_id = cls.default_mode_id
        return cls._instance

    def get_model_id(self) -> str:
        """
        获取当前的模型ID
        """
        with self._lock:
            return self._model_id

    def set_model_id(self, model_id: str):
        """
        设置新的模型ID，并确保其存在于配置文件中
        model_id为空或空字符串时，使用默认值
        """
        # 如果model_id为None或空字符串，使用默认model_id
        if not model_id or str(model_id).strip() == "":
            model_id = self.default_mode_id
        else:
            with self._lock:
                self._model_id = model_id

# 方便直接引用
model_invoke_config = ModelInvokeConfigSingleton()
