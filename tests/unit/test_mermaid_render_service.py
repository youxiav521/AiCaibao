from __future__ import annotations

import asyncio

from src.applications.equity_diagram.services import mermaid_render_service


def test_render_mermaid_to_svg_uses_no_external_cli(monkeypatch) -> None:
    async def unexpected_cli(*_args, **_kwargs):
        raise AssertionError("SVG 渲染不应调用 Node 或浏览器进程")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_cli)

    svg = asyncio.run(
        mermaid_render_service.render_mermaid_to_svg(
            'flowchart TB\nn0["甲公司"] -->|"持股 100%"| n1["乙公司"]'
        )
    )

    text = svg.decode("utf-8")
    assert "<svg" in text
    assert "甲公司" in text
    assert "乙公司" in text
    assert "持股 100%" in text
