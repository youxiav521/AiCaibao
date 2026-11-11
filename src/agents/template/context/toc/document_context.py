from pydantic import BaseModel
class Toc(BaseModel):
    text: str
    level: int
class TocInfo(Toc):
    has_children: bool = False
    id:int = 0
class TocContent(TocInfo):
    content:str

class DocumentContext(BaseModel):
    """文档上下文"""
    name:str = ""
    md5:str = ""
    global_declaration:list[str] = []
    toc:list[TocInfo] = []
    toc_content:list[TocContent] = []
    text_markdown:str = ""
    text_html:str =""