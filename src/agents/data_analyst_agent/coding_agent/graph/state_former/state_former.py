from ..state import CodingState, AnsweringState, ExtractState, PlanningState
from ...models.document_model import Document


def _ensure_document_list(raw: list) -> list:
    """将 state 中的 relative_documents_list 统一转为 List[Document]（可能为 dict 或 Document）。"""
    if not raw:
        return []
    return [Document.from_dict(x) if isinstance(x, dict) else x for x in raw]


def form_planning_state(state: dict) -> PlanningState:
    """
    根据输入的 state 字典，提取相关字段，生成 PlanningState 对象
    """
    # 过滤掉 messages 中的 None 值，保留其他有效消息
    messages = [msg for msg in state.get("messages", []) if msg is not None]
    relative_documents_list = _ensure_document_list(state.get("relative_documents_list", []))

    return PlanningState(
        question_description=state.get("question_description", ""),
        relative_documents_list=relative_documents_list,
        user_prompt = state.get("user_prompt", ""),
        need_calculate_metrics = state.get("need_calculate_metrics", {}),
        trace_steps = state.get("trace_steps", []),
        history_messages = state.get("history_messages", []),
        planning_messages = state.get("planning_messages", []),
        plan_text = state.get("plan_text", ""),
        answer_keypoints = state.get("answer_keypoints", ""),
        max_iterations = state.get("max_iterations", 5),
        answer_language = state.get("answer_language", "zh_CN"),
        messages = messages,
        # 性能指标
        model_call_count = state.get("model_call_count", 0),
        model_call_times_ms = state.get("model_call_times_ms", [])
    )

def form_coding_state(state: dict) -> CodingState:
    """
    根据输入的 state 字典，提取相关字段，生成 CodingState 对象
    """
    # 过滤掉 messages 中的 None 值，保留其他有效消息
    messages = [msg for msg in state.get("messages", []) if msg is not None]
    relative_documents_list = _ensure_document_list(state.get("relative_documents_list", []))

    return CodingState(
        question_description=state.get("question_description", ""),
        relative_documents_list=relative_documents_list,
        user_prompt = state.get("user_prompt", ""),
        need_calculate_metrics = state.get("need_calculate_metrics", {}),
        trace_steps = state.get("trace_steps", []),
        history_messages = state.get("history_messages", []),
        max_iterations = state.get("max_iterations", 5),
        iteration = state.get("iteration", 0),
        answer_language = state.get("answer_language", "zh_CN"),
        messages = messages,
        # 性能指标
        model_call_count = state.get("model_call_count", 0),
        model_call_times_ms = state.get("model_call_times_ms", []),
        code_error_count = state.get("code_error_count", 0),
        code_errors = state.get("code_errors", [])
    )

def form_answering_state(state: dict) -> AnsweringState:
    """
    根据输入的 state 字典，提取相关字段，生成 AnsweringState 对象
    """
    # 过滤掉 messages 中的 None 值，保留其他有效消息
    messages = [msg for msg in state.get("messages", []) if msg is not None]
    relative_documents_list = _ensure_document_list(state.get("relative_documents_list", []))

    return AnsweringState(
        question_description=state.get("question_description", ""),
        relative_documents_list=relative_documents_list,
        user_prompt = state.get("user_prompt", ""),
        need_calculate_metrics = state.get("need_calculate_metrics", {}),
        trace_steps = state.get("trace_steps", []),
        context_str = state.get("context_str", ""),
        code_execution_results = state.get("code_execution_results", []),
        history_messages = state.get("history_messages", []),
        saved_info = state.get("saved_info", []),
        answer_language = state.get("answer_language", "zh_CN"),
        messages = messages,
        # 性能指标
        model_call_count = state.get("model_call_count", 0),
        model_call_times_ms = state.get("model_call_times_ms", []),
        code_error_count = state.get("code_error_count", 0),
        code_errors = state.get("code_errors", [])
    )

def form_extract_state(state: dict) -> ExtractState:
    """
    根据输入的 state 字典，提取相关字段，生成 ExtractState 对象
    """
    # 过滤掉 messages 中的 None 值，保留其他有效消息
    messages = [msg for msg in state.get("messages", []) if msg is not None]
    relative_documents_list = _ensure_document_list(state.get("relative_documents_list", []))

    return ExtractState(
        question_description=state.get("question_description", ""),
        relative_documents_list=relative_documents_list,
        user_prompt = state.get("user_prompt", ""),
        need_calculate_metrics = state.get("need_calculate_metrics", {}),
        trace_steps = state.get("trace_steps", []),
        history_messages = state.get("history_messages", []),
        max_iterations = state.get("max_iterations", 5),
        answer_language = state.get("answer_language", "zh_CN"),
        messages = messages,
        # 性能指标
        model_call_count = state.get("model_call_count", 0),
        model_call_times_ms = state.get("model_call_times_ms", [])
    )