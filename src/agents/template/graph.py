import re
from typing import List, Optional
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from src.agents.template.schema import Context, Question
from src.agents.template.utils import file_name, to_html
from src.agents.template.context.document import DocumentContext

class State(TypedDict):
  template: str
  context: Context
  questions: List[Question]
  document_context: DocumentContext
  model_id: Optional[str]
  # model_id 是可选的，通过 state.get("model_id", "model_bailian_80b") 获取

def load_template(state: State):
    doc = Context(file_path=state["template"])
    from markitdown import MarkItDown
    doc.markdown = re.sub(r"q\\_", "q_", MarkItDown(enable_plugins=False).convert(doc.file_path).text_content)
    doc.html = to_html(doc.file_path)
    doc.name = file_name(doc.file_path)
    return {"context": doc}

async def generate_questions(state: State):
  model_id = state.get("model_id", "model_bailian_80b")
  document_context = DocumentContext(state["context"], model_id=model_id)
  await document_context.init()
  return {"questions": document_context.questions(), "document_context": document_context}

template_graph = (
  StateGraph(State)
  .add_node(load_template)
  .add_node(generate_questions)
  .add_edge(START, "load_template")
  .add_edge("load_template", "generate_questions")
  .add_edge("generate_questions", END)
  .compile()
)


    