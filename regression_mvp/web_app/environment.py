from __future__ import annotations

from threading import Lock
from time import monotonic
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from regression_mvp.utils.diagnostics import diagnose_failure
from regression_mvp.utils.json_io import load_json


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"
DEFAULT_API_BASE_URL = "http://localhost:6300"
CHECK_TIMEOUT_SECONDS = 0.5
CACHE_SECONDS = 10
_CACHE_LOCK = Lock()
_cached_result: dict[str, Any] | None = None
_cached_at = 0.0


def check_environment() -> dict[str, Any]:
    global _cached_result, _cached_at
    with _CACHE_LOCK:
        if _cached_result is not None and monotonic() - _cached_at < CACHE_SECONDS:
            return dict(_cached_result)

    api_base_url = _api_base_url()
    swagger_url = _swagger_url(api_base_url)
    result: dict[str, Any] = {
        "api_base_url": api_base_url,
        "swagger_url": swagger_url,
        "api_ok": False,
        "message": "",
        "diagnostic": None,
    }

    try:
        with urlopen(swagger_url, timeout=CHECK_TIMEOUT_SECONDS) as response:
            result["api_ok"] = 200 <= response.status < 500
            result["message"] = "Swagger 可访问" if result["api_ok"] else f"Swagger 返回状态码 {response.status}"
    except URLError as exc:
        result["message"] = str(exc)
        diagnostic = diagnose_failure(error_message=str(exc))
        result["diagnostic"] = diagnostic.to_dict() if diagnostic else None
    except Exception as exc:
        result["message"] = str(exc)
        diagnostic = diagnose_failure(error_message=str(exc))
        result["diagnostic"] = diagnostic.to_dict() if diagnostic else None
    with _CACHE_LOCK:
        _cached_result = result
        _cached_at = monotonic()
    return dict(result)


def _api_base_url() -> str:
    try:
        config = load_json(CONFIG_PATH)
    except Exception:
        return DEFAULT_API_BASE_URL

    api_config = config.get("api") if isinstance(config, dict) else {}
    if not isinstance(api_config, dict):
        return DEFAULT_API_BASE_URL

    base_url = str(api_config.get("base_url") or "").strip().rstrip("/")
    return base_url or DEFAULT_API_BASE_URL


def _swagger_url(api_base_url: str) -> str:
    return f"{api_base_url.rstrip('/')}/swagger#/"
