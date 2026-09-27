"""SpeedInfer Pytest Master Configuration & Fixture Harness.

Provides opaque-box test fixtures:
- Isolated in-memory SQLite database engine & session (PRAGMA foreign_keys=ON)
- High-fidelity in-memory Redis instance with native Lua scripting support via fakeredis
- Overridden test settings with cryptographically secure test pepper
- Asynchronous HTTP test client with ASGI transport
- Reusable test entity factories
"""

import os
from collections.abc import AsyncGenerator, Generator

import fakeredis
import fakeredis.aioredis
import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from speedinfer.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Test Constants
# ---------------------------------------------------------------------------
TEST_PEPPER: str = "speedinfer-test-secret-pepper-32-chars-long"
TEST_RAW_KEY: str = "sk-speedinfer-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
TEST_KEY_PREFIX: str = "sk-speedinfer-12345678"
TEST_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
TEST_PROMPT_PRICE_PER_M: float = 0.20
TEST_COMPLETION_PRICE_PER_M: float = 0.60


# ---------------------------------------------------------------------------
# Settings Fixture & Overrides
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def test_environment_vars() -> Generator[None, None, None]:
    """Ensure environment variables are configured for testing."""
    old_env = dict(os.environ)
    os.environ["ENVIRONMENT"] = "test"
    os.environ["API_KEY_PEPPER"] = TEST_PEPPER
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    get_settings.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture(scope="function")
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Provide a validated Settings object configured for in-memory testing."""
    settings = Settings(
        app_name="SpeedInfer-Test",
        environment="test",
        log_level="DEBUG",
        database_url="sqlite:///:memory:",
        redis_url="redis://localhost:6379/0",
        api_key_pepper=SecretStr(TEST_PEPPER),
        default_model=TEST_MODEL_NAME,
        vllm_base_url="http://mock-vllm:8000",
        vllm_timeout_seconds=30.0,
        prompt_price_per_million=TEST_PROMPT_PRICE_PER_M,
        completion_price_per_million=TEST_COMPLETION_PRICE_PER_M,
        max_request_tokens=32768,
        public_base_url="http://localhost:8000",
    )
    # Clear the lru_cache on get_settings and override return value
    get_settings.cache_clear()
    monkeypatch.setattr("speedinfer.config.get_settings", lambda: settings)
    return settings


# ---------------------------------------------------------------------------
# Database Fixtures (In-Memory SQLite with Foreign Keys)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def db_engine():
    """Create a pristine in-memory SQLite engine with foreign key enforcement."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    # Automatically create tables if SQLModel models have been defined
    try:
        from speedinfer.database import models  # noqa: F401

        SQLModel.metadata.create_all(engine)
    except (ImportError, AttributeError):
        pass

    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Yield a database session bound to the in-memory engine, rolling back on exit."""
    # Ensure tables exist for newly imported models in current test
    SQLModel.metadata.create_all(db_engine)
    with Session(db_engine) as session:
        yield session
        session.rollback()


# ---------------------------------------------------------------------------
# Redis Fixtures (FakeRedis with Lua Support)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def mock_redis() -> Generator[fakeredis.FakeRedis, None, None]:
    """Provide a synchronous FakeRedis client supporting native Lua scripts."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    yield client
    client.flushall()


@pytest.fixture(scope="function")
async def async_mock_redis() -> AsyncGenerator[fakeredis.aioredis.FakeRedis, None]:
    """Provide an asynchronous FakeRedis client supporting native Lua scripts."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


# ---------------------------------------------------------------------------
# Async Client Fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
async def async_client(tmp_path, async_mock_redis) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Isolated per-request sessions: never use the import-time development database."""
    from speedinfer.database.session import get_session
    from speedinfer.engine.registry import BackendWorker
    from speedinfer.gateway.app import _seed_test_keys_if_needed, app, get_registry
    from speedinfer.gateway.redis import get_async_redis

    saved = dict(app.dependency_overrides)
    isolated_engine = create_engine(
        f"sqlite:///{tmp_path / 'gateway.db'}",
        connect_args={"check_same_thread": False, "timeout": 20},
    )
    SQLModel.metadata.create_all(isolated_engine)
    with Session(isolated_engine) as session:
        _seed_test_keys_if_needed(session, TEST_PEPPER)

    def isolated_session():
        with Session(isolated_engine) as session:
            yield session

    app.dependency_overrides.setdefault(get_session, isolated_session)
    app.dependency_overrides.setdefault(get_async_redis, lambda: async_mock_redis)
    registry = get_registry()
    registry.register_model(
        name=TEST_MODEL_NAME,
        base_model_path=TEST_MODEL_NAME,
        backends=[BackendWorker(url="http://mock-vllm")],
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)
        isolated_engine.dispose()
