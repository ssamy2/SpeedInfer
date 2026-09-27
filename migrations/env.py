"""Alembic environment configuration for SpeedInfer.

Loads database configuration dynamically from speedinfer.config.get_settings()
and binds SQLModel metadata for auto-generation and schema versioning.
"""

import os
from logging.config import fileConfig
from pathlib import Path
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

# 1. Ensure API_KEY_PEPPER has at least a dummy value if running standalone
os.environ.setdefault("API_KEY_PEPPER", "speedinfer-standalone-migration-pepper-secret")

# 2. Import application settings
# 3. Import database models so SQLModel.metadata discovers all tables
import speedinfer.core.inference_billing  # noqa: F401, E402
import speedinfer.database.models  # noqa: F401, E402
from speedinfer.config import get_settings  # noqa: E402

# Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for Alembic 'autogenerate'
target_metadata = SQLModel.metadata


def get_database_url() -> str:
    """Retrieve database URL from application settings or environment override."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        raw_url = env_url.strip()
    else:
        try:
            raw_url = get_settings().database_url.strip()
        except Exception:
            raw_url = "sqlite:///./data/speedinfer.db"

    if raw_url.startswith("postgres://"):
        return "postgresql://" + raw_url[len("postgres://") :]
    return raw_url


def is_sqlite_url(url: str) -> bool:
    """Check if the provided database URL targets SQLite."""
    return url.startswith("sqlite")


def ensure_sqlite_parent_dir(url: str) -> None:
    """Create parent directories for SQLite file databases if they do not exist."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        path_str = url.replace("sqlite:///", "", 1)
        if "?" in path_str:
            path_str = path_str.split("?", 1)[0]
        db_path = Path(path_str).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine, though an Engine
    is acceptable here as well. By skipping the Engine creation we don't even
    need a DBAPI to be available. Calls to context.execute() emit the given
    string to the script output.
    """
    url = get_database_url()
    ensure_sqlite_parent_dir(url)
    sqlite_mode = is_sqlite_url(url)

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=sqlite_mode,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection with
    the context.
    """
    url = get_database_url()
    ensure_sqlite_parent_dir(url)
    sqlite_mode = is_sqlite_url(url)

    # Allow an externally passed connection (e.g., from pytest fixtures)
    connectable = context.config.attributes.get("connection", None)

    if connectable is None:
        configuration: dict[str, Any] = config.get_section(config.config_ini_section) or {}
        configuration["sqlalchemy.url"] = url

        # SQLite connection arguments
        connect_args: dict[str, Any] = {}
        if sqlite_mode:
            connect_args["check_same_thread"] = False

        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            connect_args=connect_args,
        )

        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=sqlite_mode,
                compare_type=True,
            )

            with context.begin_transaction():
                context.run_migrations()
    else:
        context.configure(
            connection=connectable,
            target_metadata=target_metadata,
            render_as_batch=sqlite_mode,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
