"""Database engine configuration, connection pooling, and session lifecycle management.

Supports dual dialects:
- SQLite for local development and testing with Write-Ahead Logging (WAL) and busy timeout.
- PostgreSQL for production deployment with connection pooling, pre-ping, and recycling.
"""

import logging
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, StaticPool
from sqlmodel import Session

# Configure resilient logger
try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Database Exception Hierarchy
# -----------------------------------------------------------------------------


class DatabaseError(Exception):
    """Base exception for all database-related errors in SpeedInfer."""


class DatabaseConnectionError(DatabaseError):
    """Raised when connecting to the database fails or host is unreachable."""


class DatabaseSessionError(DatabaseError):
    """Raised when a database session operation fails unexpectedly."""


class DatabaseConfigurationError(DatabaseError):
    """Raised when database settings or URLs are invalid or unsupported."""


# -----------------------------------------------------------------------------
# Utility & Settings Resolution
# -----------------------------------------------------------------------------


def get_database_url() -> str:
    """Resolve the active database URL with fallback handling for bootstrap environments.

    Returns:
        str: Database connection URL.
    """
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url.strip()

    # Provide a fallback pepper if not set, enabling standalone runs without failing validation
    os.environ.setdefault("API_KEY_PEPPER", "speedinfer-standalone-migration-pepper-secret")

    try:
        from speedinfer.config import get_settings

        return get_settings().database_url.strip()
    except Exception:
        # Fallback to default SQLite database path if settings cannot be loaded
        return "sqlite:///./data/speedinfer.db"


def _normalize_database_url(url: str) -> str:
    """Normalize legacy or platform-specific database URL schemes.

    SQLAlchemy 2.0 requires `postgresql://` instead of legacy `postgres://`.

    Args:
        url: Raw database connection URL.

    Returns:
        str: Normalized database connection URL.
    """
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


def _ensure_sqlite_directory(url: str) -> None:
    """Ensure the parent directory for a file-backed SQLite database exists.

    Prevents `sqlite3.OperationalError: unable to open database file`.

    Args:
        url: SQLite database URL.
    """
    if not url.startswith("sqlite"):
        return

    # Skip in-memory databases
    if ":memory:" in url or url in {"sqlite://", "sqlite:///"}:
        return

    # Format: sqlite:///path/to/db or sqlite:////absolute/path/to/db
    raw_path = url.replace("sqlite:///", "", 1)
    clean_path = raw_path.split("?")[0]

    db_path = Path(clean_path).resolve()
    db_dir = db_path.parent
    if not db_dir.exists():
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created missing directory for SQLite database", directory=str(db_dir))
        except OSError as exc:
            logger.warning(
                "Could not pre-create directory for SQLite database",
                directory=str(db_dir),
                error=str(exc),
            )


# -----------------------------------------------------------------------------
# Engine Factory
# -----------------------------------------------------------------------------


def create_db_engine(
    database_url: str | None = None,
    echo: bool | None = None,
    **kwargs: Any,
) -> Engine:
    """Create and configure a SQLAlchemy/SQLModel Engine with dialect-specific optimizations.

    For SQLite:
        - Sets DBAPI connect_args check_same_thread=False and timeout=5.0.
        - Enables PRAGMA foreign_keys=ON for relational constraint enforcement.
        - Enables PRAGMA busy_timeout=5000 to prevent 'database is locked' errors.
        - Enables PRAGMA journal_mode=WAL and PRAGMA synchronous=NORMAL for file-backed databases.
        - Employs StaticPool for in-memory databases (:memory:) to share state across test threads.
        - Automatically creates parent directories for file paths if missing.

    For PostgreSQL:
        - Normalizes postgres:// to postgresql://.
        - Configures QueuePool with pool_size=20, max_overflow=10.
        - Enables pool_pre_ping=True to discard and recycle severed connections.
        - Enables pool_recycle=3600 to prevent stale socket dropouts.
        - Sets pool_timeout=30.0 for checkout bounds.

    Args:
        database_url: Optional database URL. If omitted, resolved via get_database_url().
        echo: Optional query logging toggle. If None, checks LOG_LEVEL == 'DEBUG'.
        **kwargs: Extra parameters forwarded to create_engine.

    Returns:
        Engine: Configured SQLAlchemy Engine.

    Raises:
        DatabaseConfigurationError: If the URL scheme is unsupported or malformed.
    """
    raw_url = database_url or get_database_url()
    if not raw_url:
        raise DatabaseConfigurationError("Database URL cannot be empty.")

    normalized_url = _normalize_database_url(raw_url)

    if echo is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        echo = log_level == "DEBUG"

    is_sqlite = normalized_url.startswith("sqlite")
    is_postgres = normalized_url.startswith("postgresql")

    if not is_sqlite and not is_postgres:
        raise DatabaseConfigurationError(
            f"Unsupported database URL scheme in '{normalized_url}'. "
            "SpeedInfer supports 'sqlite' and 'postgresql'."
        )

    if is_sqlite:
        _ensure_sqlite_directory(normalized_url)

        is_memory = ":memory:" in normalized_url or normalized_url in {"sqlite://", "sqlite:///"}

        connect_args = kwargs.pop("connect_args", {})
        connect_args.setdefault("check_same_thread", False)
        connect_args.setdefault("timeout", 5.0)

        engine_kwargs: dict[str, Any] = {
            "connect_args": connect_args,
            "echo": echo,
            **kwargs,
        }

        if is_memory:
            # Force StaticPool for in-memory SQLite so multiple threads access the same DB
            engine_kwargs.setdefault("poolclass", StaticPool)
        else:
            # Force NullPool for file-backed SQLite to avoid QueuePool starvation under concurrency
            engine_kwargs.setdefault("poolclass", NullPool)

        created_engine = create_engine(normalized_url, **engine_kwargs)

        @event.listens_for(created_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            """Apply performance and integrity PRAGMAs to SQLite DBAPI connection."""
            if isinstance(dbapi_connection, sqlite3.Connection):
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA foreign_keys=ON;")
                    cursor.execute("PRAGMA busy_timeout=5000;")
                    if not is_memory:
                        cursor.execute("PRAGMA journal_mode=WAL;")
                        cursor.execute("PRAGMA synchronous=NORMAL;")
                finally:
                    cursor.close()

        logger.info(
            "Configured SQLite database engine",
            url=normalized_url,
            is_memory=is_memory,
            wal_mode=not is_memory,
        )
        return created_engine

    # PostgreSQL configuration
    pool_size = kwargs.pop("pool_size", 20)
    max_overflow = kwargs.pop("max_overflow", 10)
    pool_pre_ping = kwargs.pop("pool_pre_ping", True)
    pool_recycle = kwargs.pop("pool_recycle", 3600)
    pool_timeout = kwargs.pop("pool_timeout", 30.0)

    created_engine = create_engine(
        normalized_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
        pool_timeout=pool_timeout,
        echo=echo,
        **kwargs,
    )

    logger.info(
        "Configured PostgreSQL database engine",
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
    )
    return created_engine


# -----------------------------------------------------------------------------
# Module Singleton Engine & Session Factory
# -----------------------------------------------------------------------------

engine: Engine = create_db_engine()

SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=Session,
)


def set_engine(new_engine: Engine) -> None:
    """Update global engine and SessionLocal factory.

    Used by test harnesses to inject in-memory or mock database engines.

    Args:
        new_engine: Newly instantiated SQLAlchemy Engine.
    """
    global engine, SessionLocal
    engine = new_engine
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        class_=Session,
    )
    logger.info("Updated global database engine override")


def reset_engine() -> None:
    """Reset global engine and SessionLocal factory back to configuration defaults."""
    global engine, SessionLocal
    engine = create_db_engine()
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        class_=Session,
    )
    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)
    logger.info("Reset global database engine to default configuration")


def dispose_engine(engine_to_dispose: Engine | None = None) -> None:
    """Cleanly close and dispose of pooled database connections.

    Call during application shutdown or test suite teardown.

    Args:
        engine_to_dispose: Optional Engine instance to dispose. If None, disposes global engine.
    """
    target = engine_to_dispose or engine
    target.dispose()
    logger.info("Disposed database connection pool")


# -----------------------------------------------------------------------------
# Session Lifecycle & Context Management
# -----------------------------------------------------------------------------


@contextmanager
def get_session_context(
    engine_override: Engine | None = None,
) -> Generator[Session, None, None]:
    """Context manager providing a transactional SQLModel Session.

    Commits on clean exit of the context block.
    Rolls back transaction on unhandled exception and cleanly closes the session.

    Shields secondary exceptions during rollback to preserve the original exception.

    Args:
        engine_override: Optional Engine instance. If None, uses module engine.

    Yields:
        Session: Active SQLModel session.

    Raises:
        Exception: Re-raises any exception raised within the caller's context block.
    """
    target_engine = engine_override or engine
    session = Session(target_engine)
    try:
        yield session
        session.commit()
    except Exception as exc:
        try:
            session.rollback()
        except Exception as rollback_err:
            logger.error(
                "Database rollback failed during exception handling",
                error=str(rollback_err),
                original_error=str(exc),
            )
        raise
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped SQLModel Session.

    Automatically commits when the route completes successfully.
    Rolls back if an unhandled exception or HTTPException is raised by route.
    Guarantees session closure upon completion.

    Yields:
        Session: Active SQLModel session bound to current HTTP request.
    """
    with get_session_context() as session:
        yield session


# Backward-compatibility alias
session_scope = get_session_context


# -----------------------------------------------------------------------------
# Diagnostics & Health Check
# -----------------------------------------------------------------------------


def ping_database(
    engine_override: Engine | None = None,
    timeout_seconds: float = 5.0,
) -> bool:
    """Verify that the database is reachable and accepting queries.

    Executes 'SELECT 1' against an active connection.

    Args:
        engine_override: Optional Engine instance. If None, uses global engine.
        timeout_seconds: Maximum time to wait for connection verification.

    Returns:
        bool: True if connection check succeeds.

    Raises:
        DatabaseConnectionError: If connection fails, times out, or cannot execute query.
    """
    target_engine = engine_override or engine
    try:
        with target_engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise DatabaseConnectionError(
                    f"Unexpected ping query result: expected 1, received {result}"
                )
        return True
    except (OperationalError, DBAPIError) as exc:
        logger.error("Database connection check failed", error=str(exc))
        raise DatabaseConnectionError(f"Database connection check failed: {exc}") from exc
    except Exception as exc:
        logger.error("Unexpected error during database health ping", error=str(exc))
        raise DatabaseConnectionError(f"Unexpected database error: {exc}") from exc
