
import openai
import pytest
import os
import dotenv

dotenv.load_dotenv()
from src.base.langfuse_observe import trace_span

# 避免多次创建client实例
client = None

@trace_span(name="chatFunc")
def chatFunc(contant: str):
    global client
    local_client = client or openai.OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_API_BASE"),
    )
    completion = local_client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen3-4b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant.Determine whether the sentiment of the user's input is positive or negative.Note that only positive or negative values are output.{response_json}"},
            {"role": "user", "content": contant},
        ],
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )

    json_str = completion.choices[0].message.content
    return json_str


def test_chatFunc_with_fake_client():

    class _Msg:
        def __init__(self, content: str):
            self.message = type("_M", (), {"content": content})()

    class _Completion:
        def __init__(self, content: str):
            self.choices = [_Msg(content)]

    class _Completions:
        def create(self, **kwargs):
            return _Completion('{"sentiment":"positive"}')

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = _Chat()

    global client
    client = _FakeClient()

    result = chatFunc("你好")
    assert result == '{"sentiment":"positive"}'