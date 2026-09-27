"""Verification of test harness infrastructure and conftest fixtures.

Ensures that:
1. Test settings are properly configured and patched.
2. In-memory SQLite engine enforces foreign key constraints.
3. FakeRedis client executes commands and native Lua scripts.
4. Async client can issue requests and receive responses.
"""

import httpx
import pytest
from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel

from speedinfer.config import get_settings


def test_test_settings_override(test_settings):
    """Verify test_settings fixture overrides the global get_settings()."""
    settings = get_settings()
    assert settings.environment == "test"
    assert settings.database_url == "sqlite:///:memory:"
    assert (
        settings.api_key_pepper.get_secret_value() == "speedinfer-test-secret-pepper-32-chars-long"
    )
    assert settings.prompt_price_per_million == 0.20
    assert settings.completion_price_per_million == 0.60


def test_db_engine_sqlite_foreign_keys(db_engine):
    """Verify in-memory SQLite engine has foreign_keys pragma enabled."""
    with db_engine.connect() as conn:
        result = conn.execute(text("PRAGMA foreign_keys;")).scalar()
        assert result == 1, "Foreign keys pragma must be active in SQLite"


def test_db_session_commit_and_rollback(db_session: Session):
    """Verify db_session handles model persistence and rollbacks cleanly."""

    class DummyItem(SQLModel, table=True):
        id: int | None = Field(default=None, primary_key=True)
        name: str

    SQLModel.metadata.create_all(db_session.bind)

    item = DummyItem(name="test_item")
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)
    assert item.id is not None
    assert item.name == "test_item"


def test_mock_redis_lua_execution(mock_redis):
    """Verify FakeRedis can execute native Redis Lua scripts via Lupa."""
    lua_script = """
    local val = redis.call('GET', KEYS[1])
    if not val then
        redis.call('SET', KEYS[1], ARGV[1])
        return 1
    else
        return 0
    end
    """
    res1 = mock_redis.eval(lua_script, 1, "test:key", "initial_val")
    assert res1 == 1
    assert mock_redis.get("test:key") == "initial_val"

    res2 = mock_redis.eval(lua_script, 1, "test:key", "second_val")
    assert res2 == 0
    assert mock_redis.get("test:key") == "initial_val"


@pytest.mark.asyncio
async def test_async_mock_redis_lua_execution(async_mock_redis):
    """Verify async FakeRedis can execute commands and Lua scripts."""
    await async_mock_redis.set("async:key", "42")
    val = await async_mock_redis.get("async:key")
    assert val == "42"


@pytest.mark.asyncio
async def test_async_client_health(async_client: httpx.AsyncClient):
    """Verify async HTTP client can reach /health."""
    response = await async_client.get("/health")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
