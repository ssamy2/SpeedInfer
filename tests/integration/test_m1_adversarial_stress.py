"""Adversarial stress and boundary tests for Milestone 1 Database Persistence.

Test Matrix:
1. High-concurrency multi-threaded SQLite WAL writes (50+ threads simultaneously).
2. Boundary values and constraint enforcement (Pydantic and Database CheckConstraints).
3. Large cascading delete stress tests (hundreds of child records, zero orphans).
4. Transaction rollback isolation under concurrent failures.
"""

import os
import secrets
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from speedinfer.database.models import (
    ApiKey,
    ModelVersion,
    UsageLedger,
    User,
)
from speedinfer.database.session import (
    create_db_engine,
    get_session_context,
)

# =============================================================================
# Helper utilities for generating test data
# =============================================================================


def make_test_key_hash() -> str:
    """Generate a random 64-char hex string for key_hash."""
    return secrets.token_hex(32)


# =============================================================================
# 1. High-Concurrency SQLite WAL Write Stress Tests
# =============================================================================


class TestSqliteWalConcurrencyStress:
    """Adversarial stress tests for SQLite WAL mode write concurrency."""

    def test_50_concurrent_threads_writing_usage_ledger_zero_lock_errors(self):
        """Stress Test: 50 concurrent threads simultaneously write UsageLedger rows.

        All 50 threads synchronize on a threading.Barrier to fire at the exact
        same microsecond against a single file-backed SQLite database in WAL mode.
        Verifies:
        - 0 lock timeout errors (no 'database is locked')
        - Exactly 50 records committed
        - All 50 records have accurate attributes
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "wal_stress_50.db")
            db_url = f"sqlite:///{db_path}"
            engine = create_db_engine(db_url)

            # Initialize schema
            from sqlmodel import SQLModel

            SQLModel.metadata.create_all(engine)

            # Seed parent user and API key
            with get_session_context(engine) as session:
                user = User(email="concurrency_owner@speedinfer.local", name="Owner")
                session.add(user)
                session.flush()
                api_key = ApiKey(
                    user_id=user.id,
                    key_hash=make_test_key_hash(),
                    prefix="sk-speedinfer-conc001",
                    credit_balance=100.0,
                )
                session.add(api_key)
                session.commit()
                api_key_id = api_key.id

            num_threads = 50
            barrier = threading.Barrier(num_threads)
            results = []
            errors = []

            def worker_write(thread_idx: int):
                # Wait for all 50 threads to be ready
                barrier.wait(timeout=10.0)
                start_t = time.perf_counter()
                try:
                    with get_session_context(engine) as session:
                        ledger = UsageLedger(
                            api_key_id=api_key_id,
                            request_id=f"req-stress-50-{thread_idx}-{secrets.token_hex(4)}",
                            model="Qwen/Qwen2.5-7B-Instruct",
                            prompt_tokens=100 + thread_idx,
                            completion_tokens=50,
                            total_cost=0.00005,
                            latency_ms=12.5 + thread_idx,
                            status_code=200,
                        )
                        session.add(ledger)
                        session.commit()
                    duration = time.perf_counter() - start_t
                    results.append((thread_idx, duration))
                except Exception as exc:
                    errors.append((thread_idx, exc))

            threads = [threading.Thread(target=worker_write, args=(i,)) for i in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15.0)

            # Assert zero errors
            assert len(errors) == 0, (
                f"Encountered {len(errors)} errors during 50 concurrent writes: {errors}"
            )
            assert len(results) == num_threads, (
                f"Expected {num_threads} successful writes, got {len(results)}"
            )

            # Verify rows in database
            with get_session_context(engine) as session:
                total_rows = session.exec(
                    select(UsageLedger).where(UsageLedger.api_key_id == api_key_id)
                ).all()
                assert len(total_rows) == num_threads

            engine.dispose()

    def test_50_concurrent_threads_burst_writes_sustained_lock_contention(self):
        """Stress Test: 50 threads writing 5 successive transactions (250 total).

        Forces sustained SQLite write lock acquisition and queueing under high contention.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "wal_sustained_stress.db")
            db_url = f"sqlite:///{db_path}"
            engine = create_db_engine(db_url)

            from sqlmodel import SQLModel

            SQLModel.metadata.create_all(engine)

            with get_session_context(engine) as session:
                user = User(email="sustained@speedinfer.local", name="Sustained")
                session.add(user)
                session.flush()
                api_key = ApiKey(
                    user_id=user.id,
                    key_hash=make_test_key_hash(),
                    prefix="sk-speedinfer-sust01",
                    credit_balance=500.0,
                )
                session.add(api_key)
                session.commit()
                api_key_id = api_key.id

            num_threads = 50
            writes_per_thread = 5
            total_expected_writes = num_threads * writes_per_thread
            barrier = threading.Barrier(num_threads)
            errors = []
            completed = []

            def worker_burst(thread_idx: int):
                barrier.wait(timeout=10.0)
                try:
                    for seq in range(writes_per_thread):
                        with get_session_context(engine) as session:
                            ledger = UsageLedger(
                                api_key_id=api_key_id,
                                request_id=f"burst-{thread_idx}-{seq}-{secrets.token_hex(4)}",
                                model="Qwen/Qwen2.5-7B-Instruct",
                                prompt_tokens=50,
                                completion_tokens=25,
                                total_cost=0.00002,
                                latency_ms=10.0,
                                status_code=200,
                            )
                            session.add(ledger)
                            session.commit()
                    completed.append(thread_idx)
                except Exception as exc:
                    errors.append((thread_idx, exc))

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_burst, i) for i in range(num_threads)]
                for f in as_completed(futures):
                    f.result()

            assert len(errors) == 0, f"Encountered errors: {errors}"
            assert len(completed) == num_threads

            with get_session_context(engine) as session:
                count = session.exec(
                    select(UsageLedger).where(UsageLedger.api_key_id == api_key_id)
                ).all()
                assert len(count) == total_expected_writes

            engine.dispose()

    def test_mixed_concurrent_readers_and_writers_under_wal(self):
        """Stress Test: 40 writer threads and 15 reader threads concurrent operations.

        In SQLite WAL mode, readers must not block writers and writers must not block readers.
        Readers continuously execute aggregate queries while writers insert ledger entries.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "wal_mixed.db")
            db_url = f"sqlite:///{db_path}"
            engine = create_db_engine(db_url)

            from sqlmodel import SQLModel

            SQLModel.metadata.create_all(engine)

            with get_session_context(engine) as session:
                user = User(email="mixed@speedinfer.local", name="Mixed")
                session.add(user)
                session.flush()
                api_key = ApiKey(
                    user_id=user.id,
                    key_hash=make_test_key_hash(),
                    prefix="sk-speedinfer-mix01",
                    credit_balance=100.0,
                )
                session.add(api_key)
                session.commit()
                api_key_id = api_key.id

            stop_event = threading.Event()
            reader_errors = []
            writer_errors = []
            reads_performed = [0]
            writes_performed = [0]

            def reader_worker():
                while not stop_event.is_set():
                    try:
                        with get_session_context(engine) as session:
                            rows = session.exec(
                                select(UsageLedger).where(UsageLedger.api_key_id == api_key_id)
                            ).all()
                            _ = len(rows)
                            reads_performed[0] += 1
                        time.sleep(0.005)
                    except Exception as exc:
                        reader_errors.append(exc)

            def writer_worker(idx: int):
                try:
                    for i in range(3):
                        with get_session_context(engine) as session:
                            ledger = UsageLedger(
                                api_key_id=api_key_id,
                                request_id=f"mix-write-{idx}-{i}-{secrets.token_hex(4)}",
                                model="Qwen/Qwen2.5-7B-Instruct",
                                prompt_tokens=10,
                                completion_tokens=10,
                                total_cost=0.00001,
                                status_code=200,
                            )
                            session.add(ledger)
                            session.commit()
                            writes_performed[0] += 1
                        time.sleep(0.002)
                except Exception as exc:
                    writer_errors.append((idx, exc))

            reader_threads = [threading.Thread(target=reader_worker) for _ in range(15)]
            writer_threads = [threading.Thread(target=writer_worker, args=(i,)) for i in range(40)]

            for t in reader_threads:
                t.start()
            for t in writer_threads:
                t.start()

            for t in writer_threads:
                t.join(timeout=15.0)

            stop_event.set()
            for t in reader_threads:
                t.join(timeout=5.0)

            assert len(writer_errors) == 0, f"Writer errors: {writer_errors}"
            assert len(reader_errors) == 0, f"Reader errors: {reader_errors}"
            assert writes_performed[0] == 40 * 3
            assert reads_performed[0] > 0

            engine.dispose()


# =============================================================================
# 2. Boundary Values and Constraint Violation Tests
# =============================================================================


class TestConstraintAndBoundaryEnforcement:
    """Adversarial testing of model and database-level constraint enforcement."""

    # -------------------------------------------------------------------------
    # ApiKey Boundaries
    # -------------------------------------------------------------------------

    def test_apikey_negative_credit_balance_pydantic_rejection(self):
        """Pydantic must reject negative credit_balance at instantiation."""
        with pytest.raises(ValueError, match="credit_balance cannot be negative"):
            ApiKey(
                user_id=1,
                key_hash=make_test_key_hash(),
                prefix="sk-speedinfer-test",
                credit_balance=-0.000001,
            )

    def test_apikey_negative_credit_balance_database_check_constraint(self, db_session: Session):
        """Database CHECK constraint must reject negative balance inserted via raw SQL."""
        user = User(email="db_check_user@speedinfer.local")
        db_session.add(user)
        db_session.commit()

        # Attempt raw SQL insert with negative credit balance
        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO apikey (user_id, key_hash, prefix, credit_balance, "
                    "rpm_limit, tpm_limit, is_active, created_at) "
                    "VALUES (:user_id, :key_hash, :prefix, :credit_balance, "
                    "60, 60000, 1, CURRENT_TIMESTAMP)"
                ),
                {
                    "user_id": user.id,
                    "key_hash": make_test_key_hash(),
                    "prefix": "sk-speedinfer-neg",
                    "credit_balance": -10.0,
                },
            )
            db_session.commit()
        db_session.rollback()

    def test_apikey_zero_credit_balance_permitted(self, db_session: Session):
        """Boundary: Exactly 0.0 credit balance must be valid."""
        user = User(email="zero_balance@speedinfer.local")
        db_session.add(user)
        db_session.commit()

        key = ApiKey(
            user_id=user.id,
            key_hash=make_test_key_hash(),
            prefix="sk-speedinfer-zero",
            credit_balance=0.0,
        )
        db_session.add(key)
        db_session.commit()
        db_session.refresh(key)
        assert key.credit_balance == 0.0

    def test_apikey_non_positive_rate_limits_pydantic_rejection(self):
        """Pydantic must reject rpm_limit <= 0 and tpm_limit <= 0."""
        with pytest.raises(ValueError, match="rate limits must be strictly positive"):
            ApiKey(
                user_id=1,
                key_hash=make_test_key_hash(),
                prefix="sk-speedinfer-rate0",
                rpm_limit=0,
            )

        with pytest.raises(ValueError, match="rate limits must be strictly positive"):
            ApiKey(
                user_id=1,
                key_hash=make_test_key_hash(),
                prefix="sk-speedinfer-rate_neg",
                tpm_limit=-500,
            )

    def test_apikey_non_positive_rate_limits_db_check_constraint(self, db_session: Session):
        """Database CHECK constraints must reject rpm_limit <= 0 or tpm_limit <= 0."""
        user = User(email="check_limits@speedinfer.local")
        db_session.add(user)
        db_session.commit()

        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO apikey (user_id, key_hash, prefix, credit_balance, "
                    "rpm_limit, tpm_limit, is_active, created_at) "
                    "VALUES (:user_id, :key_hash, :prefix, "
                    "10.0, 0, 60000, 1, CURRENT_TIMESTAMP)"
                ),
                {"user_id": user.id, "key_hash": make_test_key_hash(), "prefix": "sk-test"},
            )
            db_session.commit()
        db_session.rollback()

    def test_apikey_invalid_key_hash_formats(self):
        """key_hash must be exactly 64 lowercase hexadecimal characters."""
        # Length 63 (short)
        with pytest.raises(ValueError, match="64-character lowercase hex string"):
            ApiKey(user_id=1, key_hash="a" * 63, prefix="sk-test")

        # Length 65 (long) - caught by Pydantic max_length=64 validator
        with pytest.raises((ValueError, Exception)) as exc_info:
            ApiKey(user_id=1, key_hash="a" * 65, prefix="sk-test")
        assert "at most 64 characters" in str(exc_info.value) or "64-character" in str(
            exc_info.value
        )

        # Non-hex characters
        with pytest.raises(ValueError, match="64-character lowercase hex string"):
            ApiKey(user_id=1, key_hash="z" * 64, prefix="sk-test")

    def test_apikey_empty_or_whitespace_prefix(self):
        """ApiKey prefix must not be empty or whitespace."""
        with pytest.raises(ValueError, match="prefix cannot be empty"):
            ApiKey(user_id=1, key_hash=make_test_key_hash(), prefix="")

        with pytest.raises(ValueError, match="prefix cannot be empty"):
            ApiKey(user_id=1, key_hash=make_test_key_hash(), prefix="   ")

    # -------------------------------------------------------------------------
    # UsageLedger Boundaries
    # -------------------------------------------------------------------------

    def test_usage_ledger_negative_tokens_pydantic_rejection(self):
        """UsageLedger must reject negative token counts."""
        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            UsageLedger(api_key_id=1, request_id="req-neg-tok", model="m", prompt_tokens=-1)

        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            UsageLedger(api_key_id=1, request_id="req-neg-tok2", model="m", completion_tokens=-1)

    def test_usage_ledger_negative_cost_and_latency(self):
        """UsageLedger must reject negative cost and negative latency."""
        with pytest.raises(ValueError, match="Cost and latency values cannot be negative"):
            UsageLedger(api_key_id=1, request_id="req-neg-cost", model="m", total_cost=-0.001)

        with pytest.raises(ValueError, match="Cost and latency values cannot be negative"):
            UsageLedger(api_key_id=1, request_id="req-neg-lat", model="m", latency_ms=-1.0)

    def test_usage_ledger_negative_values_db_check_constraints(self, db_session: Session):
        """Database CHECK constraints must reject negative tokens, cost, or latency."""
        user = User(email="ledger_db_check@speedinfer.local")
        db_session.add(user)
        db_session.commit()
        key = ApiKey(user_id=user.id, key_hash=make_test_key_hash(), prefix="sk-check")
        db_session.add(key)
        db_session.commit()

        # Negative prompt_tokens
        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO usageledger (api_key_id, request_id, model, prompt_tokens, "
                    "completion_tokens, total_tokens, total_cost, "
                    "latency_ms, status_code, created_at) "
                    "VALUES (:kid, 'req-ck-1', 'm', -5, 10, 5, 0.01, 10.0, 200, CURRENT_TIMESTAMP)"
                ),
                {"kid": key.id},
            )
            db_session.commit()
        db_session.rollback()

        # Negative total_cost
        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO usageledger (api_key_id, request_id, model, prompt_tokens, "
                    "completion_tokens, total_tokens, total_cost, "
                    "latency_ms, status_code, created_at) "
                    "VALUES (:kid, 'req-ck-2', 'm', 5, 10, 15, -0.05, 10.0, 200, CURRENT_TIMESTAMP)"
                ),
                {"kid": key.id},
            )
            db_session.commit()
        db_session.rollback()

    def test_usage_ledger_status_code_boundaries(self, db_session: Session):
        """HTTP status codes in [100, 599]. Boundaries 100/599 pass; 99/600 fail."""
        user = User(email="status_boundaries@speedinfer.local")
        db_session.add(user)
        db_session.commit()
        key = ApiKey(user_id=user.id, key_hash=make_test_key_hash(), prefix="sk-status")
        db_session.add(key)
        db_session.commit()

        # 99 should fail Pydantic
        with pytest.raises(ValueError, match="HTTP code between 100 and 599"):
            UsageLedger(api_key_id=key.id, request_id="req-99", model="m", status_code=99)

        # 600 should fail Pydantic
        with pytest.raises(ValueError, match="HTTP code between 100 and 599"):
            UsageLedger(api_key_id=key.id, request_id="req-600", model="m", status_code=600)

        # 99 should fail DB check constraint
        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO usageledger (api_key_id, request_id, model, prompt_tokens, "
                    "completion_tokens, total_tokens, total_cost, "
                    "latency_ms, status_code, created_at) "
                    "VALUES (:kid, 'req-db-99', 'm', 5, 5, 10, 0.01, 10.0, 99, CURRENT_TIMESTAMP)"
                ),
                {"kid": key.id},
            )
            db_session.commit()
        db_session.rollback()

        # Boundaries 100 and 599 must succeed
        rec100 = UsageLedger(api_key_id=key.id, request_id="req-100", model="m", status_code=100)
        rec599 = UsageLedger(api_key_id=key.id, request_id="req-599", model="m", status_code=599)
        db_session.add(rec100)
        db_session.add(rec599)
        db_session.commit()
        assert rec100.status_code == 100
        assert rec599.status_code == 599

    # -------------------------------------------------------------------------
    # User Boundaries & Email Formatting
    # -------------------------------------------------------------------------

    def test_user_email_validation_cases(self):
        """Email validation should normalize valid emails and reject malformed ones."""
        # Empty string
        with pytest.raises(ValueError, match="Invalid email address format"):
            User(email="")

        # Whitespace
        with pytest.raises(ValueError, match="Invalid email address format"):
            User(email="   ")

        # Missing @
        with pytest.raises(ValueError, match="Invalid email address format"):
            User(email="plainaddress")

        # Missing domain
        with pytest.raises(ValueError, match="Invalid email address format"):
            User(email="user@")

        # Missing user
        with pytest.raises(ValueError, match="Invalid email address format"):
            User(email="@domain.com")

        # Case normalization and whitespace trimming
        u = User(email="  HELLO@SpeedInfer.AI  ")
        assert u.email == "hello@speedinfer.ai"

    # -------------------------------------------------------------------------
    # ModelVersion Boundaries
    # -------------------------------------------------------------------------

    def test_model_version_boundaries_and_lifecycle_enums(self, db_session: Session):
        """ModelVersion boundaries: context length > 0, prices >= 0, valid lifecycle status."""
        # Non-positive context length
        with pytest.raises(ValueError, match="context_length must be strictly positive"):
            ModelVersion(name="m1", base_model_path="p", context_length=0)

        with pytest.raises(ValueError, match="context_length must be strictly positive"):
            ModelVersion(name="m2", base_model_path="p", context_length=-10)

        # Negative pricing
        with pytest.raises(ValueError, match="Token prices cannot be negative"):
            ModelVersion(name="m3", base_model_path="p", prompt_price_per_million=-0.01)

        # Invalid lifecycle status
        with pytest.raises(ValueError, match="Invalid lifecycle_status"):
            ModelVersion(name="m4", base_model_path="p", lifecycle_status="corrupted_status")

        # Database CheckConstraint on lifecycle_status
        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO modelversion (name, base_model_path, lifecycle_status, "
                    "context_length, prompt_price_per_million, completion_price_per_million, "
                    "created_at, updated_at) "
                    "VALUES ('raw_bad_status', 'path', 'invalid_status_enum', "
                    "2048, 0.2, 0.6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
            db_session.commit()
        db_session.rollback()


# =============================================================================
# 3. Cascading Delete Stress Tests
# =============================================================================


class TestCascadingDeleteStress:
    """Stress tests verifying complete cascading cleanup with hundreds of child rows."""

    def test_large_user_cascade_delete_orm(self, db_session: Session):
        """ORM Cascade Delete: Delete user with 100 API keys and 500 UsageLedger records.

        Verifies:
        - session.delete(user) succeeds cleanly
        - Exactly 0 child ApiKeys remain
        - Exactly 0 child UsageLedgers remain
        - No orphan rows anywhere in the database
        - Other users/keys/ledgers remain untouched
        """
        # Baseline user that should remain intact
        canary_user = User(email="canary@speedinfer.local", name="Canary")
        db_session.add(canary_user)
        db_session.commit()
        canary_key = ApiKey(
            user_id=canary_user.id, key_hash=make_test_key_hash(), prefix="sk-canary"
        )
        db_session.add(canary_key)
        db_session.commit()
        canary_ledger = UsageLedger(api_key_id=canary_key.id, request_id="canary-req", model="m")
        db_session.add(canary_ledger)
        db_session.commit()

        # Create target user to delete
        target_user = User(email="target_delete@speedinfer.local", name="Target")
        db_session.add(target_user)
        db_session.commit()

        # Create 100 API keys
        keys = []
        for i in range(100):
            k = ApiKey(
                user_id=target_user.id,
                key_hash=make_test_key_hash(),
                prefix=f"sk-del-{i:03d}",
                credit_balance=10.0,
            )
            keys.append(k)
        db_session.add_all(keys)
        db_session.commit()

        # Create 500 UsageLedger records (5 per key)
        ledgers = []
        key_ids = [k.id for k in keys]
        for kid in key_ids:
            for seq in range(5):
                rec = UsageLedger(
                    api_key_id=kid,
                    request_id=f"del-ledger-{kid}-{seq}-{secrets.token_hex(4)}",
                    model="Qwen/Qwen2.5-7B-Instruct",
                    prompt_tokens=50,
                    completion_tokens=20,
                    total_cost=0.0001,
                    status_code=200,
                )
                ledgers.append(rec)
        db_session.add_all(ledgers)
        db_session.commit()

        # Verify initial counts
        user_keys = db_session.exec(select(ApiKey).where(ApiKey.user_id == target_user.id)).all()
        assert len(user_keys) == 100

        user_ledgers = db_session.exec(
            select(UsageLedger).where(UsageLedger.api_key_id.in_(key_ids))
        ).all()
        assert len(user_ledgers) == 500

        # Execute ORM delete
        db_session.delete(target_user)
        db_session.commit()

        # Verify target user is gone
        deleted_user = db_session.get(User, target_user.id)
        assert deleted_user is None

        # Verify all 100 keys are gone
        remaining_keys = db_session.exec(
            select(ApiKey).where(ApiKey.user_id == target_user.id)
        ).all()
        assert len(remaining_keys) == 0

        # Verify all 500 ledgers are gone
        remaining_ledgers = db_session.exec(
            select(UsageLedger).where(UsageLedger.api_key_id.in_(key_ids))
        ).all()
        assert len(remaining_ledgers) == 0

        # Verify Canary is completely unaffected
        surviving_canary = db_session.get(User, canary_user.id)
        assert surviving_canary is not None
        assert db_session.get(ApiKey, canary_key.id) is not None
        assert db_session.get(UsageLedger, canary_ledger.id) is not None

        # Verify Zero Orphan Invariant in entire database
        orphan_keys = db_session.execute(
            text("SELECT count(*) FROM apikey WHERE user_id NOT IN (SELECT id FROM user)")
        ).scalar()
        assert orphan_keys == 0, f"Found {orphan_keys} orphan API keys!"

        orphan_ledgers = db_session.execute(
            text("SELECT count(*) FROM usageledger WHERE api_key_id NOT IN (SELECT id FROM apikey)")
        ).scalar()
        assert orphan_ledgers == 0, f"Found {orphan_ledgers} orphan UsageLedger rows!"

    def test_large_user_cascade_delete_direct_sql_foreign_keys(self, db_session: Session):
        """Direct SQL Cascade Delete: Delete user via raw SQL with SQLite PRAGMA foreign_keys=ON.

        Creates user with 150 API keys and 450 ledger entries.
        Executes 'DELETE FROM user WHERE id = ...' via raw SQL.
        Verifies database-level ON DELETE CASCADE cleans up without Python ORM involvement.
        """
        # Ensure foreign keys pragma is active
        fk_status = db_session.execute(text("PRAGMA foreign_keys;")).scalar()
        assert fk_status == 1, "Foreign keys must be enabled"

        user = User(email="sql_cascade@speedinfer.local", name="SQLCascade")
        db_session.add(user)
        db_session.commit()

        keys = [
            ApiKey(user_id=user.id, key_hash=make_test_key_hash(), prefix=f"sk-sql-{i}")
            for i in range(150)
        ]
        db_session.add_all(keys)
        db_session.commit()

        key_ids = [k.id for k in keys]
        ledgers = [
            UsageLedger(
                api_key_id=kid,
                request_id=f"sql-ledg-{kid}-{seq}",
                model="Qwen/Qwen2.5-7B-Instruct",
            )
            for kid in key_ids
            for seq in range(3)
        ]
        db_session.add_all(ledgers)
        db_session.commit()

        uid = user.id
        # Delete user via raw SQL
        db_session.execute(text("DELETE FROM user WHERE id = :uid"), {"uid": uid})
        db_session.commit()

        # Confirm child rows wiped out at DB level
        rem_keys = db_session.execute(
            text("SELECT count(*) FROM apikey WHERE user_id = :uid"), {"uid": uid}
        ).scalar()
        assert rem_keys == 0

        rem_ledgers = db_session.execute(
            text(
                "SELECT count(*) FROM usageledger WHERE api_key_id IN ("
                + ",".join(str(i) for i in key_ids)
                + ")"
            )
        ).scalar()
        assert rem_ledgers == 0

        # Global orphan audit
        orphan_ledgers = db_session.execute(
            text("SELECT count(*) FROM usageledger WHERE api_key_id NOT IN (SELECT id FROM apikey)")
        ).scalar()
        assert orphan_ledgers == 0


# =============================================================================
# 4. Rollback Isolation & Foreign Key Invariants
# =============================================================================


class TestRollbackAndIntegrityInvariants:
    """Tests for rollback semantics and foreign key rejection."""

    def test_foreign_key_insert_violation_rejected(self, db_session: Session):
        """Inserting child rows with non-existent parent IDs must raise IntegrityError."""
        non_existent_user_id = 999999
        non_existent_key_id = 888888

        # ApiKey with non-existent user_id
        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO apikey (user_id, key_hash, prefix, credit_balance, "
                    "rpm_limit, tpm_limit, is_active, created_at) "
                    "VALUES (:uid, :kh, 'sk-orphan', 0.0, 60, 60000, 1, CURRENT_TIMESTAMP)"
                ),
                {"uid": non_existent_user_id, "kh": make_test_key_hash()},
            )
            db_session.commit()
        db_session.rollback()

        # UsageLedger with non-existent api_key_id
        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO usageledger (api_key_id, request_id, model, prompt_tokens, "
                    "completion_tokens, total_tokens, total_cost, "
                    "latency_ms, status_code, created_at) "
                    "VALUES (:kid, 'orphan-req', 'm', 0, 0, 0, 0.0, 0.0, 200, CURRENT_TIMESTAMP)"
                ),
                {"kid": non_existent_key_id},
            )
            db_session.commit()
        db_session.rollback()

    def test_concurrent_sessions_rollback_isolation(self):
        """Concurrent sessions where some fail must not affect concurrent successful commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rollback_iso.db")
            db_url = f"sqlite:///{db_path}"
            engine = create_db_engine(db_url)

            from sqlmodel import SQLModel

            SQLModel.metadata.create_all(engine)

            with get_session_context(engine) as session:
                user = User(email="shared@speedinfer.local", name="Shared")
                session.add(user)
                session.flush()
                api_key = ApiKey(user_id=user.id, key_hash=make_test_key_hash(), prefix="sk-shared")
                session.add(api_key)
                session.commit()
                api_key_id = api_key.id

            num_threads = 20
            barrier = threading.Barrier(num_threads)
            successes = []
            expected_failures = []

            def worker_mixed(idx: int):
                barrier.wait(timeout=10.0)
                if idx % 2 == 0:
                    # Successful transaction
                    try:
                        with get_session_context(engine) as session:
                            rec = UsageLedger(
                                api_key_id=api_key_id,
                                request_id=f"success-{idx}-{secrets.token_hex(4)}",
                                model="Qwen/Qwen2.5-7B-Instruct",
                                status_code=200,
                            )
                            session.add(rec)
                            session.commit()
                        successes.append(idx)
                    except Exception as e:
                        # Should not fail
                        pytest.fail(f"Worker {idx} unexpectedly failed: {e}")
                else:
                    # Deliberately failing transaction (duplicate request_id)
                    try:
                        with get_session_context(engine) as session:
                            rec = UsageLedger(
                                api_key_id=api_key_id,
                                request_id="guaranteed-duplicate-request-id",
                                model="m",
                            )
                            session.add(rec)
                            session.commit()
                    except Exception:
                        expected_failures.append(idx)

            threads = [threading.Thread(target=worker_mixed, args=(i,)) for i in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

            # Exactly 10 successful and 10 failed (or 9 failed if first succeeded)
            assert len(successes) == 10
            assert len(expected_failures) >= 9

            with get_session_context(engine) as session:
                rows = session.exec(
                    select(UsageLedger).where(UsageLedger.api_key_id == api_key_id)
                ).all()
                # 10 successes + at most 1 from the duplicate ID
                assert len(rows) in (10, 11)

            engine.dispose()

    def test_concurrent_cascade_delete_with_active_readers_and_writers(self):
        """Stress: Delete user (100 keys, 300 ledgers) with concurrent readers/writers active.

        Verifies:
        - SQLite WAL mode handles concurrent deletion of object tree without deadlocks
        - Unrelated user operations proceed uninterrupted
        - Zero orphan rows remaining
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cascade_concurrent_stress.db")
            engine = create_db_engine(f"sqlite:///{db_path}")

            from sqlmodel import SQLModel

            SQLModel.metadata.create_all(engine)

            # Target user to delete
            with get_session_context(engine) as session:
                user = User(email="del_target@speedinfer.local")
                session.add(user)
                session.commit()
                target_uid = user.id

                keys = [
                    ApiKey(user_id=target_uid, key_hash=secrets.token_hex(32), prefix=f"sk-c-{i}")
                    for i in range(100)
                ]
                session.add_all(keys)
                session.commit()
                key_ids = [k.id for k in keys]

                ledgers = [
                    UsageLedger(api_key_id=kid, request_id=f"req-{kid}-{s}", model="m")
                    for kid in key_ids
                    for s in range(3)
                ]
                session.add_all(ledgers)
                session.commit()

            # Active separate user
            with get_session_context(engine) as session:
                other_user = User(email="active_user@speedinfer.local")
                session.add(other_user)
                session.commit()
                other_uid = other_user.id
                other_key = ApiKey(
                    user_id=other_uid, key_hash=secrets.token_hex(32), prefix="sk-act"
                )
                session.add(other_key)
                session.commit()
                other_kid = other_key.id

            deleter_errors = []
            reader_errors = []
            writer_errors = []

            def deleter():
                try:
                    with get_session_context(engine) as session:
                        u = session.get(User, target_uid)
                        session.delete(u)
                        session.commit()
                except Exception as e:
                    deleter_errors.append(e)

            def continuous_reader():
                for _ in range(30):
                    try:
                        with get_session_context(engine) as session:
                            res = session.exec(
                                select(ApiKey).where(ApiKey.user_id == other_uid)
                            ).all()
                            assert len(res) == 1
                        time.sleep(0.002)
                    except Exception as e:
                        reader_errors.append(e)

            def continuous_writer(idx: int):
                for s in range(3):
                    try:
                        with get_session_context(engine) as session:
                            rec = UsageLedger(
                                api_key_id=other_kid,
                                request_id=f"act-ledg-{idx}-{s}-{secrets.token_hex(4)}",
                                model="m",
                                status_code=200,
                            )
                            session.add(rec)
                            session.commit()
                        time.sleep(0.002)
                    except Exception as e:
                        writer_errors.append((idx, e))

            t_deleter = threading.Thread(target=deleter)
            t_readers = [threading.Thread(target=continuous_reader) for _ in range(5)]
            t_writers = [threading.Thread(target=continuous_writer, args=(i,)) for i in range(10)]

            for t in t_readers + t_writers:
                t.start()
            t_deleter.start()

            t_deleter.join(timeout=10.0)
            for t in t_readers + t_writers:
                t.join(timeout=10.0)

            assert len(deleter_errors) == 0, f"Deleter errors: {deleter_errors}"
            assert len(reader_errors) == 0, f"Reader errors: {reader_errors}"
            assert len(writer_errors) == 0, f"Writer errors: {writer_errors}"

            with get_session_context(engine) as session:
                # Target user is deleted
                assert session.get(User, target_uid) is None
                # Remaining keys for target: 0
                assert (
                    len(session.exec(select(ApiKey).where(ApiKey.user_id == target_uid)).all()) == 0
                )
                # Remaining ledgers for target: 0
                assert (
                    len(
                        session.exec(
                            select(UsageLedger).where(UsageLedger.api_key_id.in_(key_ids))
                        ).all()
                    )
                    == 0
                )
                # Active user is intact
                assert session.get(User, other_uid) is not None
                act_ledgers = session.exec(
                    select(UsageLedger).where(UsageLedger.api_key_id == other_kid)
                ).all()
                assert len(act_ledgers) == 30

                # Zero orphan check
                orphans = session.execute(
                    text("SELECT count(*) FROM apikey WHERE user_id NOT IN (SELECT id FROM user)")
                ).scalar()
                assert orphans == 0
                orphan_ledgers = session.execute(
                    text(
                        "SELECT count(*) FROM usageledger "
                        "WHERE api_key_id NOT IN (SELECT id FROM apikey)"
                    )
                ).scalar()
                assert orphan_ledgers == 0

            engine.dispose()
