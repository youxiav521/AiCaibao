from __future__ import annotations

import os

import pytest


PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def configure_network_proxy() -> None:
    """Keep the pytest process from using local proxy tools.

    This only affects the pytest process. If the API server was already started
    with proxy environment variables, restart it or configure intranet DIRECT
    rules in the proxy tool.
    """
    for env_name in PROXY_ENV_NAMES:
        os.environ.pop(env_name, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


configure_network_proxy()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-expected",
        action="store_true",
        default=False,
        help="Write expected_items.candidate.json after a successful API call. Review it before replacing expected_items.json.",
    )
