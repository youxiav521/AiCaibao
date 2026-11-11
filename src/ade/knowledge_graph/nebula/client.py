"""Minimal NebulaGraph connection-pool wrapper."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from loguru import logger
from nebula3.Config import Config
from nebula3.common.ttypes import NullType, Value
from nebula3.gclient.net import ConnectionPool
from nebula3.gclient.net.Session import Session

from src.ade.knowledge_graph.nebula.schema import validate_identifier


class NebulaGraphError(RuntimeError):
    """Raised when NebulaGraph cannot execute an operation."""


class NebulaClient:
    _pool: ConnectionPool | None = None
    _pool_address: tuple[str, int, int] | None = None
    _query_pool: ConnectionPool | None = None
    _query_pool_address: tuple[str, int, int] | None = None
    _guard = threading.Lock()

    @classmethod
    def is_configured(cls) -> bool:
        return bool(os.environ.get("NEBULA_HOST", "").strip())

    @classmethod
    def space(cls) -> str:
        return validate_identifier(os.environ.get("NEBULA_SPACE", "knowledge_graph"))

    @classmethod
    def _timeout_ms(cls) -> int:
        try:
            timeout_ms = int(os.environ.get("KG_NGQL_TIMEOUT_MS", "5000"))
        except ValueError as error:
            raise NebulaGraphError("KG_NGQL_TIMEOUT_MS must be an integer") from error
        if timeout_ms <= 0:
            raise NebulaGraphError("KG_NGQL_TIMEOUT_MS must be positive")
        return timeout_ms

    @classmethod
    def _settings(cls, *, query: bool = False) -> tuple[str, int, str, str]:
        host = os.environ.get("NEBULA_HOST", "").strip()
        if not host:
            raise NebulaGraphError("NEBULA_HOST is not configured")
        try:
            port = int(os.environ.get("NEBULA_PORT", "9669"))
        except ValueError as error:
            raise NebulaGraphError("NEBULA_PORT must be an integer") from error
        if not query:
            return (
                host,
                port,
                os.environ.get("NEBULA_USER", "root"),
                os.environ.get("NEBULA_PASSWORD", "nebula"),
            )
        user = os.environ.get("NEBULA_QUERY_USER", "").strip()
        password = os.environ.get("NEBULA_QUERY_PASSWORD", "")
        if not user or not password:
            raise NebulaGraphError("NebulaGraph query credentials are not configured")
        return host, port, user, password

    @classmethod
    def _get_pool(cls, *, query: bool = False) -> ConnectionPool:
        host, port, _, _ = cls._settings(query=query)
        timeout_ms = cls._timeout_ms() if query else 0
        address = (host, port, timeout_ms)
        pool_attr = "_query_pool" if query else "_pool"
        address_attr = "_query_pool_address" if query else "_pool_address"
        with cls._guard:
            current_pool = getattr(cls, pool_attr)
            if current_pool is not None and getattr(cls, address_attr) == address:
                return current_pool
            if current_pool is not None:
                current_pool.close()

            pool = ConnectionPool()
            config = Config()
            if query:
                config.timeout = timeout_ms
            if not pool.init([(host, port)], config):
                raise NebulaGraphError(
                    f"Unable to initialize NebulaGraph connection pool: {host}:{port}"
                )
            setattr(cls, pool_attr, pool)
            setattr(cls, address_attr, address)
            logger.info(
                "NebulaClient | 连接池已初始化 | host={} port={} query={} timeout_ms={}",
                host,
                port,
                query,
                timeout_ms,
            )
            return pool

    @classmethod
    @contextmanager
    def session(cls) -> Iterator[Session]:
        _, _, user, password = cls._settings()
        with cls._get_pool().session_context(user, password) as session:
            cls.execute(session, f"USE `{cls.space()}`")
            yield session

    @classmethod
    @contextmanager
    def query_session(cls) -> Iterator[Session]:
        _, _, user, password = cls._settings(query=True)
        with cls._get_pool(query=True).session_context(user, password) as session:
            cls.execute(session, f"USE `{cls.space()}`")
            yield session

    @staticmethod
    def _parameter(value: Any) -> Value:
        if isinstance(value, Value):
            return value
        if value is None:
            return Value(nVal=NullType.__NULL__)
        if isinstance(value, bool):
            return Value(bVal=value)
        if isinstance(value, int):
            return Value(iVal=value)
        if isinstance(value, float):
            return Value(fVal=value)
        if isinstance(value, str):
            return Value(sVal=value.encode())
        raise TypeError(f"Unsupported NebulaGraph parameter type: {type(value)!r}")

    @staticmethod
    def execute(
        session: Session,
        statement: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        result = (
            session.execute_parameter(
                statement,
                {
                    name: NebulaClient._parameter(value)
                    for name, value in parameters.items()
                },
            )
            if parameters is not None
            else session.execute(statement)
        )
        if not result.is_succeeded():
            message = result.error_msg()
            if isinstance(message, bytes):
                message = message.decode("utf-8", errors="replace")
            raise NebulaGraphError(f"NebulaGraph query failed: {message}")
        return result.as_primitive()

    @classmethod
    def close(cls) -> None:
        with cls._guard:
            if cls._pool is not None:
                cls._pool.close()
            if cls._query_pool is not None:
                cls._query_pool.close()
            cls._pool = None
            cls._pool_address = None
            cls._query_pool = None
            cls._query_pool_address = None
