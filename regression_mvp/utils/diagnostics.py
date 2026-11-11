from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Diagnostic:
    category: str
    severity: str
    title: str
    reason: str
    next_steps: list[str]
    evidence: list[str]
    confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_failure(
    *,
    error_message: str = "",
    diff_total: int = 0,
    failure_category: str = "",
    log_text: str = "",
) -> Diagnostic | None:
    text = "\n".join(part for part in [error_message, failure_category, log_text] if part).strip()
    lower = text.lower()
    evidence = _evidence(text)

    if not text and not diff_total:
        return None

    if "mineru_contract_failed" in lower:
        return Diagnostic(
            category="mineru_contract_failed",
            severity="high",
            title="MinerU 解析结果基础契约校验失败",
            reason="返回结果结构不完整，或有有效页面、关键对象在本次解析中消失。",
            next_steps=[
                "先查看 MinerU 解析质量评估中的基础契约问题。",
                "打开受影响页面的预览，确认服务输出后再决定是否更新标准答案。",
            ],
            evidence=evidence,
            confidence="high",
        )

    if "mineru_content_suspected" in lower:
        return Diagnostic(
            category="mineru_content_suspected",
            severity="high",
            title="MinerU 内容疑似回归",
            reason="归一化后的内容覆盖率或数字覆盖率低于配置的容忍范围。",
            next_steps=[
                "先查看 MinerU 解析质量评估中的重点复核页，再阅读原始字段差异。",
                "财务文件请对照原始 PDF 复核被标记的数字变化。",
                "如果确认是模型正常抖动，先采集多次重复运行数据，再调整阈值。",
            ],
            evidence=evidence,
            confidence="high",
        )

    if diff_total:
        return Diagnostic(
            category="regression_diff",
            severity="high",
            title="结果与基线不一致",
            reason=f"本次实际输出与 expected 基线存在 {diff_total} 处差异。",
            next_steps=[
                "优先打开 HTML 报告中的 diff samples、actual 和预览产物。",
                "如果差异符合预期，重新生成候选用例并人工确认后 promote。",
                "如果差异不符合预期，转给对应接口负责人排查解析或抽取逻辑。",
            ],
            evidence=evidence,
            confidence="high",
        )

    if _has_any(lower, ["error code: 502", "502 bad gateway", "status=502", "status code 502"]):
        return Diagnostic(
            category="upstream_service_error",
            severity="high",
            title="上游解析服务返回 502",
            reason="被测接口已收到请求，但内部调用 MinerU/解析服务时上游返回 502。",
            next_steps=[
                "先单独重跑该 case，确认是否为偶发上游抖动。",
                "确认公司 VPN 和内网服务可用，必要时关闭本机梯子后重试。",
                "如果图片型 PDF 反复触发该错误，将 case 标记为 upstream flaky，并转解析服务维护人排查。",
            ],
            evidence=evidence,
            confidence="high",
        )

    if _has_any(lower, ["status=500", "status code 500", "internal server error"]):
        return Diagnostic(
            category="api_server_error",
            severity="high",
            title="被测 API 返回 500",
            reason="请求已经到达被测 API，但服务端处理失败。",
            next_steps=[
                "查看响应 body 中的 detail，确认失败阶段是解析、索引还是抽取。",
                "查看业务 API 同一时间点日志，按 request/file/case 关键字定位堆栈。",
                "如果 detail 指向上游服务，再按上游服务错误处理。",
            ],
            evidence=evidence,
            confidence="medium",
        )

    if _has_any(lower, ["connection refused", "failed to establish a new connection", "max retries exceeded"]):
        return Diagnostic(
            category="api_unreachable",
            severity="high",
            title="被测 API 或内网服务不可达",
            reason="日志中出现连接失败或重试耗尽，通常是服务未启动、VPN 未连接或网络不可达。",
            next_steps=[
                "确认本地 API 已启动在配置的 base_url 上。",
                "确认公司 VPN 已连接，相关 10.x 内网地址可访问。",
                "如果开着系统代理/梯子，关闭后重启 API 再重试。",
            ],
            evidence=evidence,
            confidence="high",
        )

    if _has_any(lower, ["connecttimeout", "readtimeout", "timed out", "timeout"]):
        return Diagnostic(
            category="timeout",
            severity="medium",
            title="接口或上游服务超时",
            reason="请求在等待被测接口或上游服务响应时超时。",
            next_steps=[
                "重跑该 case，确认是否为临时网络或上游服务抖动。",
                "如果同一 PDF 稳定超时，检查文件页数、图片型页面和解析耗时。",
                "必要时将该 case 拆成更小样本，或延长测试超时时间后复核。",
            ],
            evidence=evidence,
            confidence="high",
        )

    if _has_any(lower, ["proxy", "server disconnected without sending a response"]):
        return Diagnostic(
            category="proxy_or_network",
            severity="medium",
            title="疑似代理或网络链路问题",
            reason="日志中出现代理相关字样或服务端未返回响应即断开连接。",
            next_steps=[
                "关闭本机梯子或确认 NO_PROXY 覆盖 localhost 和内网地址。",
                "如果业务 API 已在开代理时启动，关闭代理后重启 API。",
                "再次运行同一 case 观察是否恢复。",
            ],
            evidence=evidence,
            confidence="medium",
        )

    if _has_any(lower, ["file not found", "no such file", "input file is required"]):
        return Diagnostic(
            category="invalid_input",
            severity="low",
            title="输入文件或表单参数不正确",
            reason="测试或候选生成没有拿到有效输入文件。",
            next_steps=[
                "确认上传文件已选择，或备用路径指向 Web 服务所在机器上的真实文件。",
                "确认 case name 非空且没有使用特殊路径字符。",
            ],
            evidence=evidence,
            confidence="high",
        )

    if _has_any(lower, ["case already exists", "use --force", "already exists"]):
        return Diagnostic(
            category="case_already_exists",
            severity="low",
            title="正式用例已存在",
            reason="promote 时目标 case 目录已经存在，框架默认阻止覆盖。",
            next_steps=[
                "如果要保留旧 case，请换一个 case name。",
                "如果确认要替换旧基线，在候选页勾选“覆盖”后再 promote。",
            ],
            evidence=evidence,
            confidence="high",
        )

    if _has_any(lower, ["traceback", "assertionerror", "pytest"]):
        return Diagnostic(
            category="test_framework_error",
            severity="medium",
            title="测试框架或断言执行失败",
            reason="日志中出现 pytest/traceback 信息，需要区分是测试代码问题还是接口返回导致的断言失败。",
            next_steps=[
                "先查看 traceback 最底部异常信息。",
                "如果异常指向 regression_mvp 代码，优先排查测试框架逻辑。",
                "如果异常是接口 RuntimeError 或 HTTP detail，按接口错误分类继续排查。",
            ],
            evidence=evidence,
            confidence="medium",
        )

    return Diagnostic(
        category="unknown_failure",
        severity="medium",
        title="未分类失败",
        reason="当前规则无法明确判断失败原因。",
        next_steps=[
            "查看完整日志和 HTML 报告中的 Error 区域。",
            "补充一条新的 diagnostics 规则，将类似错误沉淀为可识别分类。",
        ],
        evidence=evidence,
        confidence="low",
    )


def diagnostic_counts(diagnostics: list[Diagnostic | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        if diagnostic is None:
            continue
        counts[diagnostic.category] = counts.get(diagnostic.category, 0) + 1
    return dict(sorted(counts.items()))


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _evidence(text: str) -> list[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    selected: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(token in lower for token in ["error", "failed", "status=", "502", "500", "timeout", "connection", "traceback"]):
            selected.append(_clip(line))
        if len(selected) >= 4:
            break
    if selected:
        return selected
    compact = re.sub(r"\s+", " ", text).strip()
    return [_clip(compact)] if compact else []


def _clip(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
