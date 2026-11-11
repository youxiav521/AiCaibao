import importlib
import sys


MODULE_NAME = "src.applications.parsing.service.parse_minerU_service"


def _reload_mineru_service(monkeypatch, trust_env: str | None = None):
    monkeypatch.setenv("MINERU_API_BASE", "http://10.192.72.12:7300")
    if trust_env is None:
        monkeypatch.delenv("MINERU_HTTP_TRUST_ENV", raising=False)
    else:
        monkeypatch.setenv("MINERU_HTTP_TRUST_ENV", trust_env)

    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_mineru_http_client_ignores_env_proxy_by_default(monkeypatch):
    module = _reload_mineru_service(monkeypatch)

    service = module.MinerUParsingService()

    assert service.http_trust_env is False


def test_mineru_http_client_can_opt_in_to_env_proxy(monkeypatch):
    module = _reload_mineru_service(monkeypatch, trust_env="true")

    service = module.MinerUParsingService()

    assert service.http_trust_env is True


def test_mineru_bearer_token_initializes_headers(monkeypatch):
    module = _reload_mineru_service(monkeypatch)

    service = module.MinerUParsingService(bearer_token="token")

    assert service.headers["Authorization"] == "Bearer token"
