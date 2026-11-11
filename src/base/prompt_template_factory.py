from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
# 单例模式
class PromptManager(metaclass=SingletonMeta):
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader("config/prompt"),
            autoescape=False,
            keep_trailing_newline=True,
        )
    @staticmethod
    def getPrompt(templateId: str,param: Dict[str, Any]=None) -> str:
        """
        获取提示词模板，并填充参数
        """
        instance = PromptManager()
        template = instance.env.get_template(f"{templateId}.md")
        if not param: 
            param = {}
        promptStr: str = template.render(**param)
        return promptStr
