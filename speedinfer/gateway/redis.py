"""Redis client lifecycle and connection manager.

Provides resilient connection pools for atomic metering Lua scripts and
dual token-bucket rate limiting. Falls back gracefully to in-memory FakeRedis
during testing or when external Redis is offline.
"""

import os
from typing import Any

import fakeredis
import fakeredis.aioredis

from speedinfer.config import get_settings

_async_redis_client: Any = None
_sync_redis_client: Any = None
_fake_server: fakeredis.FakeServer | None = None


def get_fake_server() -> fakeredis.FakeServer:
    """Return shared FakeServer for tests and mock environments."""
    global _fake_server
    if _fake_server is None:
        _fake_server = fakeredis.FakeServer()
    return _fake_server


async def get_async_redis() -> Any:
    """FastAPI dependency for obtaining an asynchronous Redis client.

    Returns:
        Any: aioredis or fakeredis client instance supporting Lua scripts.
    """
    settings = get_settings()
    env = os.environ.get("ENVIRONMENT", settings.environment).lower()

    if env == "test":
        server = get_fake_server()
        return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)

    global _async_redis_client
    if _async_redis_client is not None:
        return _async_redis_client

    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        await client.ping()
        _async_redis_client = client
        return _async_redis_client
    except Exception:
        server = get_fake_server()
        return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)


def get_sync_redis() -> Any:
    """Return a synchronous Redis client.

    Returns:
        Any: redis or fakeredis client instance.
    """
    global _sync_redis_client
    if _sync_redis_client is not None:
        return _sync_redis_client

    settings = get_settings()
    env = os.environ.get("ENVIRONMENT", settings.environment).lower()

    if env == "test":
        server = get_fake_server()
        _sync_redis_client = fakeredis.FakeRedis(server=server, decode_responses=True)
        return _sync_redis_client

    try:
        import redis

        client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        client.ping()
        _sync_redis_client = client
        return _sync_redis_client
    except Exception:
        server = get_fake_server()
        _sync_redis_client = fakeredis.FakeRedis(server=server, decode_responses=True)
        return _sync_redis_client


async def close_redis() -> None:
    """Close active Redis connections during gateway shutdown."""
    global _async_redis_client, _sync_redis_client
    if _async_redis_client is not None:
        try:
            await _async_redis_client.aclose()
        except Exception:
            pass
        _async_redis_client = None

    if _sync_redis_client is not None:
        try:
            _sync_redis_client.close()
        except Exception:
            pass
        _sync_redis_client = None
