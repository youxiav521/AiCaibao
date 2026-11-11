from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


def load_full_content_list(json_path: Path) -> list:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_request_body(content_list: list) -> dict:
    # CopilotKit 后端在工作流中从 messages[0].content 提取输入参数
    # 这里将实际参数序列化为字符串放入 TextMessage.content
    input_payload = {
        "content_list": content_list,
        "questions": [
            "分析公司成立时间",
            "公司对当前外部经营环境与政策趋势的判断",
            "公司近三年供应渠道综述",
        ],
        # "top_k": 6,
        "extra_instructions": [
            "",
            "输出数据单位：万元\n仅返回分析结果，不要有多余回答，严格按照要求回答；\n请用自然语言总结，输出自然语言段落，不要带标题，不要分析过程，答案中去掉'根据提供的文档片段'，你的答案：",
            "仅返回分析内容，不要有多余回答，严格按照要求回答\n请用自然语言总结，输出自然语言段落，不要带标题。你的答案：",
        ],
        # "model_id": "model_for_content_generation_backup",
    }
    return {
        "state": {},
        "messages": [
            {
                "id": "msg_001",
                "type": "TextMessage",
                "role": "user",
                "content": json.dumps(input_payload, ensure_ascii=False),
            }
        ],
        "config": {},
    }


def iso_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def stream_post_sse(url: str, payload: dict, log_path: Path | None = None) -> None:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    # 使用 stream=True + iter_lines，同步兼容两种流格式：
    # 1) 标准 SSE（以空行分隔，一个事件多行包含 event:/data: 前缀）
    # 2) 每行一个 JSON 事件（无空行，仅以换行分隔）
    with requests.post(url, headers=headers, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), stream=True) as resp:
        resp.raise_for_status()
        event_type = "message"
        buffer_lines: list[str] = []
        log_f = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # 以写入模式打开，启动即清空旧日志
            log_f = log_path.open("w", encoding="utf-8")

        def out(line: str = "") -> None:
            print(line)
            sys.stdout.flush()
            if log_f is not None:
                log_f.write(line + "\n")
                log_f.flush()
        def flush_event():
            if not buffer_lines:
                return
            nonlocal event_type
            # 优先按照标准 SSE 解析（聚合 data: 行）
            sse_data_lines: list[str] = []
            parsed_from_sse = False
            for l in buffer_lines:
                if l.startswith(":"):
                    continue
                if l.startswith("event:"):
                    event_type = l[len("event:") :].strip()
                    parsed_from_sse = True
                elif l.startswith("data:"):
                    sse_data_lines.append(l[len("data:") :].lstrip())
                    parsed_from_sse = True
            if parsed_from_sse:
                data_str = "\n".join(sse_data_lines).strip()
                out(f"[{iso_ts()}] event={event_type}")
                if data_str:
                    try:
                        obj = json.loads(data_str)
                        out(json.dumps(obj, ensure_ascii=False, indent=2))
                    except Exception:
                        out(data_str)
                out()
                buffer_lines.clear()
                event_type = "message"
                return
            # 否则尝试按“每行一个 JSON 事件”处理
            for l in buffer_lines:
                line = l.strip()
                if not line:
                    continue
                if line.startswith("{") or line.startswith("["):
                    out(f"[{iso_ts()}] event=message")
                    try:
                        obj = json.loads(line)
                        out(json.dumps(obj, ensure_ascii=False, indent=2))
                    except Exception:
                        out(line)
                    out()
            buffer_lines.clear()

        for line in resp.iter_lines(decode_unicode=True):
            # None 表示 keep-alive 心跳
            if line is None:
                continue
            # 标准 SSE：空行代表一个事件结束
            if line == "":
                flush_event()
                continue
            # 非空行：收集到缓冲
            buffer_lines.append(line)
        # 退出前清空剩余缓冲，避免最后一个事件丢失
        flush_event()
        if log_f is not None:
            log_f.close()


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    content_path = root / "storage" / "20250919.json"
    if not content_path.exists():
        print(f"找不到内容文件: {content_path}")
        sys.exit(1)
    content_list = load_full_content_list(content_path)
    payload = build_request_body(content_list)
    log_path = root / "storage" / "sse_test_full_content.log"

    url = "http://10.192.72.11:6300/agui/copilotkit/agent/content_generate"
    print(f"[{iso_ts()}] 开始请求: {url}")
    start = time.time()
    try:
        stream_post_sse(url, payload, log_path=log_path)
    finally:
        dur = (time.time() - start) * 1000
        print(f"[{iso_ts()}] 流式结束，耗时 {dur:.0f} ms")


if __name__ == "__main__":
    main()


