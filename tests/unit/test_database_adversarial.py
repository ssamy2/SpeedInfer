"""Adversarial stress-testing suite for SpeedInfer Milestone 1 (Database & Persistence).

Empirically challenges:
1. Rapid successive Alembic upgrade/downgrade cycles and populated data teardown.
2. Forced transaction rollback & session isolation under exceptions and concurrency.
3. Database bootstrap idempotency stress under sequential and concurrent invocation.
4. Comprehensive compatibility alias behavior across models and session lifecycles.
"""

import os
import tempfile
import threading
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from speedinfer.database.init_db import create_tables_direct, init_db, run_alembic_upgrade
from speedinfer.database.models import ApiKey, ModelVersion, UsageLedger, User
from speedinfer.database.session import (
    create_db_engine,
    get_session,
    get_session_context,
    reset_engine,
    session_scope,
    set_engine,
)


# ---------------------------------------------------------------------------
# 1. Rapid Successive Alembic Upgrade / Downgrade Cycles
# ---------------------------------------------------------------------------
def test_rapid_successive_alembic_upgrade_downgrade_cycles():
    """Verify rapid successive Alembic migrations handle schema teardown and recreation.

    Executes:
    - 5 rapid empty upgrade/downgrade cycles.
    - An upgrade followed by populating active relational data.
    - Teardown of the populated database via 'downgrade base' to stress reverse
      foreign-key dependency drop ordering.
    - Re-upgrade to head verifying clean recreation of all tables and indexes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "alembic_cycles.db")
        old_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"

        try:
            cfg = Config("alembic.ini")

            # 1. 5 rapid cycles on empty schema
            for _ in range(5):
                command.upgrade(cfg, "head")
                command.downgrade(cfg, "base")

            # Verify schema is completely empty after downgrade
            engine = create_engine(f"sqlite:///{db_file}")
            insp = inspect(engine)
            tables_down = [t for t in insp.get_table_names() if t != "alembic_version"]
            assert len(tables_down) == 0, f"Expected 0 tables, got {tables_down}"

            # 2. Upgrade to head and populate relational data graph
            command.upgrade(cfg, "head")
            with Session(engine) as session:
                user = User(email="mig_stress@speedinfer.local", name="Mig Tester")
                session.add(user)
                session.commit()
                session.refresh(user)

                key = ApiKey(
                    user_id=user.id,
                    name="mig-key",
                    key_hash="a" * 64,
                    prefix="sk-speedinfer-migstress",
                    credit_balance=250.0,
                )
                session.add(key)
                session.commit()
                session.refresh(key)

                ledger = UsageLedger(
                    api_key_id=key.id,
                    request_id="req-mig-1",
                    model="Qwen/Qwen2.5-7B-Instruct",
                    prompt_tokens=40,
                    completion_tokens=80,
                    total_tokens=120,
                    total_cost=0.0004,
                    latency_ms=120.5,
                    status_code=200,
                )
                session.add(ledger)

                mv = ModelVersion(
                    name="Qwen/Qwen2.5-7B-Instruct",
                    base_model_path="Qwen/Qwen2.5-7B-Instruct",
                )
                session.add(mv)
                session.commit()

            # 3. Stress-test downgrade with populated data under foreign key constraints
            command.downgrade(cfg, "base")
            insp = inspect(engine)
            remaining = [t for t in insp.get_table_names() if t != "alembic_version"]
            assert len(remaining) == 0, f"Tables remained after populated downgrade: {remaining}"

            # 4. Re-upgrade to head verifying clean recreation
            command.upgrade(cfg, "head")
            insp = inspect(engine)
            recreated_tables = set(insp.get_table_names())
            for expected in {"user", "apikey", "usageledger", "modelversion"}:
                assert expected in recreated_tables, f"Missing table {expected} on re-upgrade"

            # 5. Verify programmatic run_alembic_upgrade utility is idempotent at head
            run_alembic_upgrade("alembic.ini")
            run_alembic_upgrade("alembic.ini")

            engine.dispose()
        finally:
            if old_url:
                os.environ["DATABASE_URL"] = old_url
            else:
                os.environ.pop("DATABASE_URL", None)


# ---------------------------------------------------------------------------
# 2. Forced Transaction Rollback & Session Isolation Under Exceptions
# ---------------------------------------------------------------------------
def test_forced_transaction_rollback_and_session_isolation():
    """Verify transaction rollback atomicity, session isolation, and generator cleanup.

    Checks:
    - Multi-entity atomic rollback on unhandled application exception.
    - Atomic rollback on database-level IntegrityError.
    - Read isolation: uncommitted writes in Session A are not visible to Session B.
    - FastAPI dependency generator handles throw() and rolls back transaction.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "rollback_isolation.db")
        engine = create_db_engine(f"sqlite:///{db_file}")
        set_engine(engine)
        create_tables_direct(engine)
        try:
            # A. Multi-entity rollback
            with pytest.raises(RuntimeError, match="Simulated application failure"):
                with get_session_context(engine) as session:
                    user = User(email="atomic_fail@speedinfer.local", name="Atomic Fail")
                    session.add(user)
                    session.flush()

                    key = ApiKey(
                        user_id=user.id,
                        key_hash="b" * 64,
                        prefix="sk-speedinfer-atomic",
                    )
                    session.add(key)
                    session.flush()

                    raise RuntimeError("Simulated application failure")

            # Verify nothing persisted
            with Session(engine) as check_s:
                q_user = select(User).where(User.email == "atomic_fail@speedinfer.local")
                assert check_s.exec(q_user).first() is None
                q_key = select(ApiKey).where(ApiKey.prefix == "sk-speedinfer-atomic")
                assert check_s.exec(q_key).first() is None

            # B. IntegrityError rollback
            with get_session_context(engine) as session:
                session.add(User(email="unique_user@speedinfer.local"))

            with pytest.raises(IntegrityError):
                with get_session_context(engine) as session:
                    session.add(User(email="unique_user@speedinfer.local"))

            # Verify next transaction on fresh session works without dirty state
            with get_session_context(engine) as session:
                session.add(User(email="healthy_user@speedinfer.local"))

            with Session(engine) as check_s:
                q_healthy = select(User).where(User.email == "healthy_user@speedinfer.local")
                assert check_s.exec(q_healthy).first() is not None

            # C. Session isolation between concurrent sessions
            with Session(engine) as session_a:
                session_a.add(User(email="uncommitted@speedinfer.local"))
                session_a.flush()  # Flushed to DB transaction, but not committed

                with Session(engine) as session_b:
                    q_uncommitted = select(User).where(User.email == "uncommitted@speedinfer.local")
                    leaked = session_b.exec(q_uncommitted).first()
                    assert leaked is None, "Uncommitted data in Session A leaked to Session B!"

                session_a.rollback()

            # D. FastAPI get_session generator throw rollback
            gen = get_session()
            request_session = next(gen)
            request_session.add(User(email="generator_route_fail@speedinfer.local"))
            request_session.flush()

            with pytest.raises(ValueError, match="Route HTTP exception"):
                gen.throw(ValueError("Route HTTP exception"))

            with Session(engine) as check_s:
                q_throw = select(User).where(User.email == "generator_route_fail@speedinfer.local")
                persisted = check_s.exec(q_throw).first()
                assert persisted is None, (
                    "FastAPI route exception must trigger transaction rollback"
                )
        finally:
            reset_engine()
            engine.dispose()


def test_concurrent_write_bursts_with_forced_rollbacks():
    """Verify SQLite WAL mode concurrency and rollback resilience under heavy contention.

    Spawns 8 concurrent threads executing 25 transactions each (200 total),
    deliberately failing 33% of transactions with forced exceptions.
    Asserts zero unhandled lock contention errors and exact count match between
    successful commits and database state.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "concurrent_wal_stress.db")
        engine = create_db_engine(f"sqlite:///{db_file}")
        create_tables_direct(engine)

        num_threads = 8
        tx_per_thread = 25
        successful_commits = 0
        deliberate_aborts = 0
        unexpected_errors: list[str] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            nonlocal successful_commits, deliberate_aborts
            for i in range(tx_per_thread):
                email = f"stress_t{tid}_i{i}@speedinfer.local"
                should_fail = i % 3 == 0
                try:
                    with get_session_context(engine) as s:
                        user = User(email=email, name=f"Thread {tid}")
                        s.add(user)
                        s.flush()
                        if should_fail:
                            raise RuntimeError(f"Forced abort thread {tid} iter {i}")
                    with lock:
                        successful_commits += 1
                except RuntimeError:
                    with lock:
                        deliberate_aborts += 1
                except Exception as exc:
                    with lock:
                        unexpected_errors.append(f"Thread {tid} error: {exc}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        err_summary = ", ".join(unexpected_errors)
        assert len(unexpected_errors) == 0, f"Unexpected errors during writes: {err_summary}"
        assert successful_commits > 0
        assert deliberate_aborts > 0

        with Session(engine) as s:
            persisted_users = s.exec(select(User)).all()
            assert len(persisted_users) == successful_commits, (
                f"Persisted count {len(persisted_users)} != commits {successful_commits}"
            )

        engine.dispose()


# ---------------------------------------------------------------------------
# 3. Database Bootstrap Idempotency Stress Test
# ---------------------------------------------------------------------------
def test_database_bootstrap_idempotency_sequential():
    """Verify init_db bootstrapping is fully idempotent over 15 sequential executions.

    Asserts:
    - Exactly 1 admin user, 1 admin API key, and 1 default model version exist.
    - Plaintext API key is returned ONLY on initial run, None thereafter.
    - default_model_created is True on initial run, False thereafter.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "bootstrap_seq.db")
        engine = create_db_engine(f"sqlite:///{db_file}")

        results: list[dict[str, Any]] = []
        for _ in range(15):
            res = init_db(
                use_alembic=False,
                seed=True,
                admin_email="admin_idempotency@speedinfer.local",
                engine=engine,
            )
            results.append(res)

        # First run yields raw API key
        assert results[0]["admin_api_key"] is not None
        assert results[0]["default_model_created"] is True

        # Subsequent runs preserve credentials and return None
        for i, res in enumerate(results[1:], start=2):
            assert res["admin_api_key"] is None, f"Run {i} unexpectedly regenerated API key"
            assert res["default_model_created"] is False, f"Run {i} unexpectedly recreated model"

        # Verify DB entities count
        with Session(engine) as session:
            users = session.exec(select(User)).all()
            keys = session.exec(select(ApiKey)).all()
            models = session.exec(select(ModelVersion)).all()

            assert len(users) == 1
            assert len(keys) == 1
            assert len(models) == 1
            assert users[0].email == "admin_idempotency@speedinfer.local"

        engine.dispose()


def test_database_bootstrap_idempotency_concurrent():
    """Verify init_db seeding idempotency under concurrent initialization bursts.

    Launches 8 threads attempting to run init_db(seed=True) concurrently on a fresh database.
    Asserts zero constraint corruption and deterministic creation of exactly 1 admin user and key.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "bootstrap_conc.db")
        engine = create_db_engine(f"sqlite:///{db_file}")
        create_tables_direct(engine)

        barrier = threading.Barrier(8)
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            barrier.wait()
            try:
                r = init_db(
                    use_alembic=False,
                    seed=True,
                    admin_email="admin_conc@speedinfer.local",
                    engine=engine,
                )
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(f"Thread {tid} error: {e}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent bootstrap raised unexpected errors: {errors}"
        assert len(results) == 8

        # Only one thread should have generated the initial admin key
        keys_generated = [r["admin_api_key"] for r in results if r["admin_api_key"] is not None]
        assert len(keys_generated) == 1, f"Expected 1 key, got {len(keys_generated)}"

        with Session(engine) as session:
            users = session.exec(select(User)).all()
            keys = session.exec(select(ApiKey)).all()
            models = session.exec(select(ModelVersion)).all()

            assert len(users) == 1
            assert len(keys) == 1
            assert len(models) == 1

        engine.dispose()


# ---------------------------------------------------------------------------
# 4. Compatibility Aliases Verification
# ---------------------------------------------------------------------------
def test_compatibility_aliases_api_key():
    """Verify ApiKey aliases: prefix <-> key_prefix and query behavior."""
    # 1. Constructor alias via key_prefix
    k1 = ApiKey(
        user_id=1,
        key_hash="1" * 64,
        key_prefix="sk-speedinfer-alias1",
    )
    assert k1.prefix == "sk-speedinfer-alias1"
    assert k1.key_prefix == "sk-speedinfer-alias1"

    # 2. Constructor canonical via prefix
    k2 = ApiKey(
        user_id=1,
        key_hash="2" * 64,
        prefix="sk-speedinfer-canonical",
    )
    assert k2.prefix == "sk-speedinfer-canonical"
    assert k2.key_prefix == "sk-speedinfer-canonical"

    # 3. Passing both: canonical prefix takes precedence
    k3 = ApiKey(
        user_id=1,
        key_hash="3" * 64,
        prefix="sk-speedinfer-canonical-wins",
        key_prefix="sk-speedinfer-alias-ignored",
    )
    assert k3.prefix == "sk-speedinfer-canonical-wins"
    assert k3.key_prefix == "sk-speedinfer-canonical-wins"

    # 4. Property setter
    k1.key_prefix = "sk-speedinfer-mutated"
    assert k1.prefix == "sk-speedinfer-mutated"
    assert k1.key_prefix == "sk-speedinfer-mutated"

    # 5. Pydantic model_validate limitation (requires canonical field 'prefix')
    with pytest.raises(ValidationError):
        ApiKey.model_validate(
            {"user_id": 1, "key_hash": "4" * 64, "key_prefix": "sk-speedinfer-validate"}
        )

    # 6. SQL Query behavior: ApiKey.prefix is the canonical column
    stmt_canonical = select(ApiKey).where(ApiKey.prefix == "sk-speedinfer-canonical")
    assert "apikey.prefix = :prefix_1" in str(stmt_canonical)


def test_compatibility_aliases_usage_ledger():
    """Verify UsageLedger aliases: total_cost <-> cost and total_tokens derivation."""
    # 1. Constructor alias via cost
    l1 = UsageLedger(
        api_key_id=1,
        request_id="req-alias-1",
        model="Qwen",
        prompt_tokens=25,
        completion_tokens=75,
        cost=0.0025,
    )
    assert l1.total_cost == 0.0025
    assert l1.cost == 0.0025
    assert l1.total_tokens == 100  # Auto-derived

    # 2. Constructor canonical via total_cost and explicit total_tokens
    l2 = UsageLedger(
        api_key_id=1,
        request_id="req-alias-2",
        model="Qwen",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        total_cost=0.0010,
    )
    assert l2.total_cost == 0.0010
    assert l2.cost == 0.0010
    assert l2.total_tokens == 30

    # 3. Passing both cost and total_cost: canonical total_cost wins
    l3 = UsageLedger(
        api_key_id=1,
        request_id="req-alias-3",
        model="Qwen",
        prompt_tokens=5,
        completion_tokens=5,
        total_cost=0.05,
        cost=0.01,
    )
    assert l3.total_cost == 0.05
    assert l3.cost == 0.05

    # 4. Property setter
    l1.cost = 0.0080
    assert l1.total_cost == 0.0080
    assert l1.cost == 0.0080


def test_compatibility_aliases_model_version():
    """Verify ModelVersion aliases: base_model and context_window."""
    # 1. Constructor aliases
    m1 = ModelVersion(
        name="model-alias-1",
        base_model="Qwen/Qwen2.5-7B",
        context_window=16384,
    )
    assert m1.base_model_path == "Qwen/Qwen2.5-7B"
    assert m1.base_model == "Qwen/Qwen2.5-7B"
    assert m1.context_length == 16384
    assert m1.context_window == 16384

    # 2. Canonical constructor fields
    m2 = ModelVersion(
        name="model-alias-2",
        base_model_path="Qwen/Qwen2.5-7B",
        context_length=32768,
    )
    assert m2.base_model_path == "Qwen/Qwen2.5-7B"
    assert m2.base_model == "Qwen/Qwen2.5-7B"
    assert m2.context_length == 32768
    assert m2.context_window == 32768

    # 3. Passing both: canonical fields take precedence
    m3 = ModelVersion(
        name="model-alias-3",
        base_model_path="Canonical/Path",
        base_model="Alias/Path",
        context_length=65536,
        context_window=8192,
    )
    assert m3.base_model_path == "Canonical/Path"
    assert m3.base_model == "Canonical/Path"
    assert m3.context_length == 65536
    assert m3.context_window == 65536

    # 4. Property setters
    m1.base_model = "Updated/Model"
    m1.context_window = 131072
    assert m1.base_model_path == "Updated/Model"
    assert m1.context_length == 131072


def test_compatibility_session_scope_alias():
    """Verify session_scope is an alias to get_session_context."""
    assert session_scope is get_session_context


def test_sqlite_engine_check_constraints_enforced():
    """Verify that SQLite engine strictly enforces table CheckConstraints on raw inserts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "engine_checks.db")
        engine = create_db_engine(f"sqlite:///{db_file}")
        create_tables_direct(engine)

        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO user (id, email, is_active, is_admin, created_at, updated_at) "
                    "VALUES (1, 'ck@test.com', 1, 0, '2026-01-01', '2026-01-01')"
                )
            )
            conn.commit()

            # 1. Negative credit_balance rejected by DB constraint
            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO apikey (user_id, name, permissions, key_hash, prefix, "
                        "credit_balance, rpm_limit, tpm_limit, is_active, created_at) "
                        "VALUES (1, 'k', 'chat', 'hash1', 'sk-ck1', -1.0, 60, 60000, "
                        "1, '2026-01-01')"
                    )
                )
                conn.commit()

            # 2. Zero rpm_limit rejected by DB constraint
            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO apikey (user_id, name, permissions, key_hash, prefix, "
                        "credit_balance, rpm_limit, tpm_limit, is_active, created_at) "
                        "VALUES (1, 'k', 'chat', 'hash2', 'sk-ck2', 10.0, 0, 60000, "
                        "1, '2026-01-01')"
                    )
                )
                conn.commit()

            # 3. Invalid status_code (650) rejected by DB constraint
            conn.execute(
                text(
                    "INSERT INTO apikey (id, user_id, name, permissions, key_hash, prefix, "
                    "credit_balance, rpm_limit, tpm_limit, is_active, created_at) "
                    "VALUES (1, 1, 'k', 'chat', 'hash3', 'sk-ck3', 10.0, 60, 60000, "
                    "1, '2026-01-01')"
                )
            )
            conn.commit()

            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO usageledger (api_key_id, request_id, model, status_code, "
                        "prompt_tokens, completion_tokens, total_tokens, total_cost, latency_ms, "
                        "created_at) VALUES (1, 'r1', 'm', 650, 0, 0, 0, 0.0, 0.0, '2026-01-01')"
                    )
                )
                conn.commit()

        engine.dispose()
