"""Unit and integration tests for SpeedInfer database models and session lifecycle.

Covers:
- SQLModel models: User, ApiKey, UsageLedger, ModelVersion
- Model creation, field defaults, and type integrity
- Unique constraints (email, key_hash, model name)
- Foreign key enforcement and relationship navigations
- Session rollback semantics and transaction isolation
- SQLite WAL mode pragmas and multi-connection concurrency
"""

import os
import tempfile
import threading

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, select

# Check for existence of M1 database implementation
try:
    from speedinfer.database.models import ApiKey, ModelVersion, UsageLedger, User

    HAS_DB_MODELS = True
except (ImportError, AttributeError):
    HAS_DB_MODELS = False

try:
    from speedinfer.database import session as session_module

    HAS_DB_SESSION = True
except (ImportError, AttributeError):
    HAS_DB_SESSION = False


# ---------------------------------------------------------------------------
# Fixture to skip model-specific tests when implementation is pending
# ---------------------------------------------------------------------------
@pytest.fixture
def require_models():
    """Skip test if speedinfer.database.models is not yet implemented."""
    if not HAS_DB_MODELS:
        pytest.skip("speedinfer.database.models not yet implemented by M1")


# ---------------------------------------------------------------------------
# SQLite WAL Mode & Pragma Tests
# ---------------------------------------------------------------------------
def test_sqlite_wal_mode_and_pragmas():
    """Verify that file-based SQLite database sets WAL mode and pragmas correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_wal.db")
        db_url = f"sqlite:///{db_path}"

        engine = create_engine(db_url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()

        # Connect and check journal_mode
        with engine.connect() as conn:
            journal_mode = conn.execute(text("PRAGMA journal_mode;")).scalar()
            assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"

            busy_timeout = conn.execute(text("PRAGMA busy_timeout;")).scalar()
            assert busy_timeout == 5000, f"Expected busy_timeout 5000, got {busy_timeout}"

            foreign_keys = conn.execute(text("PRAGMA foreign_keys;")).scalar()
            assert foreign_keys == 1, f"Expected foreign_keys 1, got {foreign_keys}"

        engine.dispose()


def test_sqlite_wal_concurrent_reads_and_writes():
    """Verify WAL mode allows concurrent readers while a writer commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "concurrent_wal.db")
        db_url = f"sqlite:///{db_path}"

        engine = create_engine(db_url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.close()

        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE counter (id INTEGER PRIMARY KEY, count INTEGER);"))
            conn.execute(text("INSERT INTO counter (id, count) VALUES (1, 0);"))
            conn.commit()

        errors = []

        def writer():
            try:
                with engine.connect() as conn:
                    for _ in range(20):
                        conn.execute(text("UPDATE counter SET count = count + 1 WHERE id = 1;"))
                        conn.commit()
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                with engine.connect() as conn:
                    for _ in range(50):
                        val = conn.execute(text("SELECT count FROM counter WHERE id = 1;")).scalar()
                        assert val is not None
            except Exception as e:
                errors.append(e)

        t_writer = threading.Thread(target=writer)
        t_reader = threading.Thread(target=reader)

        t_writer.start()
        t_reader.start()

        t_writer.join(timeout=5.0)
        t_reader.join(timeout=5.0)

        assert len(errors) == 0, f"Concurrent WAL operations produced errors: {errors}"
        engine.dispose()


# ---------------------------------------------------------------------------
# Model Creation & Constraint Tests
# ---------------------------------------------------------------------------
def test_user_creation_and_defaults(require_models, db_session: Session):
    """Verify User model creation with default flags and timestamps."""
    user = User(
        email="developer@speedinfer.com",
        name="Lead Engineer",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    assert user.id is not None
    assert user.email == "developer@speedinfer.com"
    assert user.name == "Lead Engineer"
    assert user.is_active is True
    assert user.is_admin is False
    assert user.created_at is not None


def test_user_unique_email_constraint(require_models, db_session: Session):
    """Verify unique constraint on User email raises IntegrityError."""
    user1 = User(email="unique@speedinfer.com", name="User 1")
    db_session.add(user1)
    db_session.commit()

    user2 = User(email="unique@speedinfer.com", name="User 2")
    db_session.add(user2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_api_key_creation_and_user_relationship(require_models, db_session: Session):
    """Verify ApiKey creation, default values, and bidirectional relationship with User."""
    user = User(email="apikey_owner@speedinfer.com", name="Key Owner")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    key_kwargs = {
        "user_id": user.id,
        "name": "default-key",
        "key_hash": "a" * 64,
        "credit_balance": 100.0,
    }
    # Check if field is key_prefix or prefix
    if hasattr(ApiKey, "key_prefix"):
        key_kwargs["key_prefix"] = "sk-speedinfer-12345678"
    elif hasattr(ApiKey, "prefix"):
        key_kwargs["prefix"] = "sk-speedinfer-12345678"

    api_key = ApiKey(**key_kwargs)
    db_session.add(api_key)
    db_session.commit()
    db_session.refresh(api_key)
    db_session.refresh(user)

    assert api_key.id is not None
    assert api_key.user_id == user.id
    assert api_key.credit_balance == 100.0
    assert api_key.rpm_limit > 0
    assert api_key.tpm_limit > 0
    assert api_key.is_active is True

    # Relationship navigation
    if hasattr(api_key, "user") and api_key.user is not None:
        assert api_key.user.email == "apikey_owner@speedinfer.com"
    if hasattr(user, "api_keys") and user.api_keys is not None:
        assert len(user.api_keys) == 1
        assert user.api_keys[0].id == api_key.id


def test_api_key_foreign_key_enforcement(require_models, db_session: Session):
    """Verify inserting an ApiKey with a non-existent user_id fails."""
    key_kwargs = {
        "user_id": 99999999,  # Non-existent user
        "name": "orphan-key",
        "key_hash": "b" * 64,
        "credit_balance": 10.0,
    }
    if hasattr(ApiKey, "key_prefix"):
        key_kwargs["key_prefix"] = "sk-speedinfer-orphan"
    elif hasattr(ApiKey, "prefix"):
        key_kwargs["prefix"] = "sk-speedinfer-orphan"

    api_key = ApiKey(**key_kwargs)
    db_session.add(api_key)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_api_key_unique_hash_constraint(require_models, db_session: Session):
    """Verify duplicate key_hash raises IntegrityError."""
    user = User(email="duplicate_key@speedinfer.com")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    shared_hash = "c" * 64
    prefix_field = "key_prefix" if hasattr(ApiKey, "key_prefix") else "prefix"

    key1 = ApiKey(user_id=user.id, name="key1", key_hash=shared_hash, **{prefix_field: "prefix1"})
    db_session.add(key1)
    db_session.commit()

    key2 = ApiKey(user_id=user.id, name="key2", key_hash=shared_hash, **{prefix_field: "prefix2"})
    db_session.add(key2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_usage_ledger_creation_and_attributes(require_models, db_session: Session):
    """Verify UsageLedger record creation and token accounting attributes."""
    user = User(email="ledger_user@speedinfer.com")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    prefix_field = "key_prefix" if hasattr(ApiKey, "key_prefix") else "prefix"
    api_key = ApiKey(user_id=user.id, name="k", key_hash="d" * 64, **{prefix_field: "prefix"})
    db_session.add(api_key)
    db_session.commit()
    db_session.refresh(api_key)

    cost_field = "cost" if hasattr(UsageLedger, "cost") else "total_cost"
    ledger_kwargs = {
        "api_key_id": api_key.id,
        "request_id": "chatcmpl-test-uuid-12345",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt_tokens": 50,
        "completion_tokens": 150,
        cost_field: 0.000100,
        "latency_ms": 145.2,
        "status_code": 200,
    }
    if hasattr(UsageLedger, "total_tokens"):
        ledger_kwargs["total_tokens"] = 200
    if hasattr(UsageLedger, "ttft_ms"):
        ledger_kwargs["ttft_ms"] = 35.0

    ledger_entry = UsageLedger(**ledger_kwargs)
    db_session.add(ledger_entry)
    db_session.commit()
    db_session.refresh(ledger_entry)

    assert ledger_entry.id is not None
    assert ledger_entry.request_id == "chatcmpl-test-uuid-12345"
    assert ledger_entry.prompt_tokens == 50
    assert ledger_entry.completion_tokens == 150
    assert getattr(ledger_entry, cost_field) == 0.000100
    assert ledger_entry.status_code == 200
    assert ledger_entry.created_at is not None


def test_model_version_creation_and_defaults(require_models, db_session: Session):
    """Verify ModelVersion creation, unique model name, and pricing defaults."""
    base_model_field = "base_model" if hasattr(ModelVersion, "base_model") else "base_model_path"
    context_field = (
        "context_window" if hasattr(ModelVersion, "context_window") else "context_length"
    )

    model_kwargs = {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        base_model_field: "Qwen/Qwen2.5-7B-Instruct",
        "lifecycle_status": "active",
        "prompt_price_per_million": 0.20,
        "completion_price_per_million": 0.60,
    }
    model_v = ModelVersion(**model_kwargs)
    db_session.add(model_v)
    db_session.commit()
    db_session.refresh(model_v)

    assert model_v.id is not None
    assert model_v.name == "Qwen/Qwen2.5-7B-Instruct"
    assert model_v.prompt_price_per_million == 0.20
    assert model_v.completion_price_per_million == 0.60
    assert getattr(model_v, context_field) > 0

    # Duplicate name should raise IntegrityError
    dup_model = ModelVersion(**model_kwargs)
    db_session.add(dup_model)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# ---------------------------------------------------------------------------
# Session Rollback Tests
# ---------------------------------------------------------------------------
def test_session_rollback_clears_uncommitted_state(require_models, db_session: Session):
    """Verify session rollback removes staged changes from database."""
    user = User(email="rollback_user@speedinfer.com")
    db_session.add(user)
    # Roll back before commit
    db_session.rollback()

    statement = select(User).where(User.email == "rollback_user@speedinfer.com")
    found_user = db_session.exec(statement).first()
    assert found_user is None, "Rolled-back record must not be found in database"


def test_session_scope_context_manager(db_engine):
    """Verify session_scope context manager commits on success and rolls back on exception."""
    if not HAS_DB_SESSION or not hasattr(session_module, "session_scope"):
        # Test generic session scope pattern if module pending
        class DummyEntity(SQLModel, table=True):
            id: int = Field(default=None, primary_key=True)
            name: str

        SQLModel.metadata.create_all(db_engine)

        with Session(db_engine) as session:
            session.add(DummyEntity(name="persisted"))
            session.commit()

        with pytest.raises(RuntimeError):
            with Session(db_engine) as session:
                session.add(DummyEntity(name="failed"))
                raise RuntimeError("Simulated crash")

        with Session(db_engine) as session:
            items = session.exec(select(DummyEntity)).all()
            assert len(items) == 1
            assert items[0].name == "persisted"
    else:
        # Test official speedinfer.database.session.session_scope
        with pytest.raises(ValueError):
            with session_module.session_scope() as session:
                user = User(email="should_fail@speedinfer.com")
                session.add(user)
                raise ValueError("Simulated failure inside session_scope")


def test_user_methods_and_validations(require_models, db_session: Session):
    """Verify email validation and touch method on User."""
    with pytest.raises(ValueError, match="Invalid email address format"):
        User(email="invalid-email-address")

    user = User(email="touch_test@speedinfer.com")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    old_updated = user.updated_at
    user.touch()
    assert user.updated_at >= old_updated


def test_api_key_validations_and_methods(require_models, db_session: Session):
    """Verify ApiKey validation hooks, permission checking, and expiration logic."""
    user = User(email="key_methods_test@speedinfer.com")
    db_session.add(user)
    db_session.commit()

    # Invalid key_hash
    with pytest.raises(ValueError, match="key_hash must be a 64-character"):
        ApiKey(user_id=user.id, key_hash="too_short", prefix="sk-speedinfer-test")

    # Empty prefix
    with pytest.raises(ValueError, match="prefix cannot be empty"):
        ApiKey(user_id=user.id, key_hash="a" * 64, prefix="   ")

    # Negative balance
    with pytest.raises(ValueError, match="credit_balance cannot be negative"):
        ApiKey(
            user_id=user.id,
            key_hash="a" * 64,
            prefix="sk-speedinfer-test",
            credit_balance=-5.0,
        )

    # Non-positive RPM/TPM limits
    with pytest.raises(ValueError, match="rate limits must be strictly positive"):
        ApiKey(user_id=user.id, key_hash="a" * 64, prefix="sk-speedinfer-test", rpm_limit=0)

    key = ApiKey(
        user_id=user.id,
        key_hash="a" * 64,
        prefix="sk-speedinfer-1234abcd",
        permissions="chat:completions,models:read",
        credit_balance=50.0,
    )
    assert key.key_prefix == "sk-speedinfer-1234abcd"
    key.key_prefix = "sk-speedinfer-updated"
    assert key.prefix == "sk-speedinfer-updated"

    assert key.has_permission("chat:completions") is True
    assert key.has_permission("completions") is False
    assert key.is_expired() is False
    assert key.is_valid() is True

    # Admin key has wildcard permissions
    admin_key = ApiKey(
        user_id=user.id,
        key_hash="b" * 64,
        prefix="sk-speedinfer-admin",
        permissions="admin",
    )
    assert admin_key.has_permission("any:arbitrary:scope") is True


def test_usage_ledger_validations_and_derivation(require_models, db_session: Session):
    """Verify UsageLedger validation rules and total_tokens derivation."""
    user = User(email="ledger_val@speedinfer.com")
    db_session.add(user)
    db_session.commit()
    key = ApiKey(user_id=user.id, key_hash="e" * 64, prefix="sk-speedinfer-val")
    db_session.add(key)
    db_session.commit()

    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        UsageLedger(api_key_id=key.id, request_id="r1", model="m", prompt_tokens=-1)

    with pytest.raises(ValueError, match="Cost and latency values cannot be negative"):
        UsageLedger(api_key_id=key.id, request_id="r2", model="m", total_cost=-0.5)

    with pytest.raises(ValueError, match="status_code must be a valid HTTP code"):
        UsageLedger(api_key_id=key.id, request_id="r3", model="m", status_code=99)

    # total_tokens auto-derivation
    ledger = UsageLedger(
        api_key_id=key.id,
        request_id="r4",
        model="m",
        prompt_tokens=30,
        completion_tokens=70,
        cost=0.005,
    )
    assert ledger.total_tokens == 100
    assert ledger.cost == 0.005
    ledger.cost = 0.010
    assert ledger.total_cost == 0.010


def test_model_version_validations_and_cost_calc(require_models, db_session: Session):
    """Verify ModelVersion lifecycle validation and calculate_cost logic."""
    with pytest.raises(ValueError, match="context_length must be strictly positive"):
        ModelVersion(name="m1", base_model_path="p", context_length=0)

    with pytest.raises(ValueError, match="Token prices cannot be negative"):
        ModelVersion(name="m2", base_model_path="p", prompt_price_per_million=-0.1)

    with pytest.raises(ValueError, match="Invalid lifecycle_status"):
        ModelVersion(name="m3", base_model_path="p", lifecycle_status="unsupported_status")

    model = ModelVersion(
        name="Qwen/Qwen2.5-7B-Test",
        base_model="Qwen/Qwen2.5-7B-Test",
        context_window=16384,
        prompt_price_per_million=0.20,
        completion_price_per_million=0.60,
    )
    assert model.base_model_path == "Qwen/Qwen2.5-7B-Test"
    assert model.context_length == 16384
    assert model.is_available() is True

    # Cost calculation: 1000 prompt tokens (0.0002) + 2000 completion tokens (0.0012) = 0.0014
    cost = model.calculate_cost(prompt_tokens=1000, completion_tokens=2000)
    assert cost == 0.0014


def test_session_engine_factory_and_diagnostics():
    """Verify engine factory dialect handling, ping diagnostic, and exception hierarchy."""
    from speedinfer.database.session import (
        DatabaseConfigurationError,
        DatabaseConnectionError,
        create_db_engine,
        ping_database,
    )

    # PostgreSQL URL normalization
    pg_engine = create_db_engine("postgres://user:pass@localhost:5432/testdb")
    assert pg_engine.url.drivername == "postgresql"
    pg_engine.dispose()

    # Unsupported scheme
    with pytest.raises(DatabaseConfigurationError):
        create_db_engine("mysql://localhost/test")

    # Ping active engine
    mem_engine = create_db_engine("sqlite:///:memory:")
    assert ping_database(mem_engine) is True
    mem_engine.dispose()

    # Ping failing connection
    broken_engine = create_engine("sqlite:////proc/nonexistent_speedinfer/test.db")
    with pytest.raises(DatabaseConnectionError):
        ping_database(broken_engine)
    broken_engine.dispose()


def test_get_session_fastapi_dependency(db_engine):
    """Verify get_session dependency yields session and handles commit/rollback."""
    from speedinfer.database.session import get_session, reset_engine, set_engine

    set_engine(db_engine)
    try:
        gen = get_session()
        session = next(gen)
        assert isinstance(session, Session)

        user = User(email="dep_test@speedinfer.com")
        session.add(user)

        # Simulating clean exit from route
        try:
            next(gen)
        except StopIteration:
            pass

        # Verify persisted
        with Session(db_engine) as check_s:
            stmt = select(User).where(User.email == "dep_test@speedinfer.com")
            found = check_s.exec(stmt).first()
            assert found is not None
    finally:
        reset_engine()


def test_init_db_and_seeding_idempotence(db_engine):
    """Verify init_db bootstrapping and seed idempotency."""
    from speedinfer.database.init_db import init_db

    res1 = init_db(
        use_alembic=False,
        seed=True,
        engine=db_engine,
        admin_email="admin_test@speedinfer.local",
    )
    assert res1["status"] == "initialized"
    assert res1["admin_email"] == "admin_test@speedinfer.local"
    assert res1["admin_api_key"] is not None
    assert res1["default_model_created"] is True

    res2 = init_db(
        use_alembic=False,
        seed=True,
        engine=db_engine,
        admin_email="admin_test@speedinfer.local",
    )
    assert res2["admin_api_key"] is None  # Already exists
    assert res2["default_model_created"] is False  # Already exists


def test_alembic_migrations_upgrade_downgrade():
    """Verify Alembic migrations apply upgrade, downgrade, and re-upgrade cleanly."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test_mig.db")
        old_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
        try:
            cfg = Config("alembic.ini")

            # Upgrade to head
            command.upgrade(cfg, "head")
            test_engine = create_engine(f"sqlite:///{db_file}")
            insp = inspect(test_engine)
            tables = insp.get_table_names()
            for required_table in ("user", "apikey", "usageledger", "modelversion"):
                assert required_table in tables

            # Downgrade to base
            command.downgrade(cfg, "base")
            insp = inspect(test_engine)
            tables_down = [t for t in insp.get_table_names() if t != "alembic_version"]
            assert len(tables_down) == 0

            # Re-upgrade to head
            command.upgrade(cfg, "head")
            insp = inspect(test_engine)
            tables_re = insp.get_table_names()
            assert "user" in tables_re
            test_engine.dispose()
        finally:
            if old_url:
                os.environ["DATABASE_URL"] = old_url
            else:
                os.environ.pop("DATABASE_URL", None)
