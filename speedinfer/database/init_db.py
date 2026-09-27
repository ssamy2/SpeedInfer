"""Database initialization and seed bootstrap utility.

This module provides programmatic and CLI utilities to:
1. Ensure database parent directory exists (for SQLite).
2. Create schema tables directly via SQLModel metadata OR Alembic migrations.
3. Seed initial admin user and default ModelVersion idempotently.
4. Generate a default administrative API key with HMAC-SHA256 + pepper hashing.
"""

import argparse
import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

# Ensure API_KEY_PEPPER is set before importing config
os.environ.setdefault("API_KEY_PEPPER", "speedinfer-standalone-migration-pepper-secret")

from speedinfer.config import get_settings
from speedinfer.database.models import ApiKey, ModelVersion, User
from speedinfer.database.session import engine as default_engine


def ensure_db_directory(database_url: str) -> None:
    """Ensure parent directories exist for local SQLite file storage."""
    if database_url.startswith("sqlite:///") and not database_url.startswith("sqlite:///:memory:"):
        path_str = database_url.replace("sqlite:///", "", 1)
        if "?" in path_str:
            path_str = path_str.split("?", 1)[0]
        db_path = Path(path_str).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)


def create_tables_direct(engine: Engine | None = None) -> None:
    """Create all SQLModel tables directly using metadata create_all."""
    target_engine = engine or default_engine
    ensure_db_directory(str(target_engine.url))
    SQLModel.metadata.create_all(bind=target_engine)


def run_alembic_upgrade(alembic_ini_path: str = "alembic.ini") -> None:
    """Apply Alembic migrations up to head programmatically."""
    from alembic import command
    from alembic.config import Config

    ini_path = Path(alembic_ini_path).resolve()
    if not ini_path.exists():
        # Search relative to current file or project root
        project_root = Path(__file__).resolve().parent.parent.parent
        ini_path = project_root / "alembic.ini"

    if not ini_path.exists():
        raise FileNotFoundError(f"Could not locate alembic.ini at {ini_path}")

    alembic_cfg = Config(str(ini_path))
    command.upgrade(alembic_cfg, "head")


def seed_admin_user(
    session: Session,
    email: str = "admin@speedinfer.local",
    name: str = "SpeedInfer Admin",
    initial_balance: float = 1000.0,
) -> tuple[User, str | None]:
    """Idempotently seed the initial administrator and return (User, raw_key_or_None).

    If the user already exists, returns (existing_user, None).
    If a new admin user is created, a plaintext API key is generated once and returned.
    Handles concurrent initialization race conditions by catching IntegrityError.
    """
    settings = get_settings()
    existing_user = session.exec(select(User).where(User.email == email)).first()

    if existing_user is not None:
        return existing_user, None

    # 1. Create User
    user = User(
        email=email,
        name=name,
        is_active=True,
        is_admin=True,
    )
    session.add(user)
    try:
        session.commit()
        session.refresh(user)
    except IntegrityError:
        session.rollback()
        existing = session.exec(select(User).where(User.email == email)).first()
        if existing is not None:
            return existing, None
        raise

    # 2. Generate secure API key
    raw_entropy = secrets.token_hex(32)  # 64 hex characters
    raw_key = f"sk-speedinfer-{raw_entropy}"
    prefix = raw_key[:22]  # e.g., 'sk-speedinfer-1234abcd'

    pepper = settings.api_key_pepper.get_secret_value()
    key_hash = hmac.new(
        pepper.encode("utf-8"),
        raw_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    admin_key = ApiKey(
        user_id=user.id,
        name="initial-admin-key",
        key_hash=key_hash,
        prefix=prefix,
        permissions="chat:completions,completions,models:read,usage:read,admin",
        credit_balance=initial_balance,
        rpm_limit=1000,
        tpm_limit=1000000,
        is_active=True,
    )
    session.add(admin_key)
    try:
        session.commit()
        session.refresh(admin_key)
        return user, raw_key
    except IntegrityError:
        session.rollback()
        existing = session.exec(select(User).where(User.email == email)).first()
        return existing or user, None


def seed_default_model(session: Session) -> tuple[ModelVersion, bool]:
    """Idempotently seed the default ModelVersion from application settings.

    Returns (ModelVersion, was_created_boolean).
    Handles concurrent initialization race conditions by catching IntegrityError.
    """
    settings = get_settings()
    model_name = settings.default_model

    existing_model = session.exec(
        select(ModelVersion).where(ModelVersion.name == model_name)
    ).first()

    if existing_model is not None:
        return existing_model, False

    model = ModelVersion(
        name=model_name,
        base_model_path=model_name,
        adapter_path=None,
        lifecycle_status="active",
        context_length=settings.max_request_tokens,
        prompt_price_per_million=settings.prompt_price_per_million,
        completion_price_per_million=settings.completion_price_per_million,
    )
    session.add(model)
    try:
        session.commit()
        session.refresh(model)
        return model, True
    except IntegrityError:
        session.rollback()
        existing = session.exec(select(ModelVersion).where(ModelVersion.name == model_name)).first()
        if existing is not None:
            return existing, False
        raise


def init_db(
    use_alembic: bool = False,
    seed: bool = True,
    admin_email: str = "admin@speedinfer.local",
    initial_balance: float = 1000.0,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Execute complete database initialization and bootstrap sequence.

    Args:
        use_alembic: If True, uses Alembic migrations; otherwise uses SQLModel metadata.
        seed: If True, seeds initial admin user and default ModelVersion.
        admin_email: Email address for default administrator.
        initial_balance: Starting credit balance for seeded admin key.
        engine: Optional custom SQLAlchemy/SQLModel engine.

    Returns:
        Dict summarizing initialization actions, seeded models, and admin key if generated.
    """
    settings = get_settings()
    ensure_db_directory(settings.database_url)

    target_engine = engine or default_engine

    if use_alembic:
        run_alembic_upgrade()
    else:
        create_tables_direct(engine=target_engine)

    results: dict[str, Any] = {
        "status": "initialized",
        "method": "alembic" if use_alembic else "direct_metadata",
        "database_url": str(target_engine.url),
    }

    if seed:
        with Session(target_engine) as session:
            admin_user, raw_key = seed_admin_user(
                session, email=admin_email, initial_balance=initial_balance
            )
            default_model, model_created = seed_default_model(session)

            results["admin_user_id"] = admin_user.id
            results["admin_email"] = admin_user.email
            results["admin_api_key"] = raw_key
            results["default_model"] = default_model.name
            results["default_model_created"] = model_created

    return results


def main() -> None:
    """CLI entrypoint for database initialization."""
    parser = argparse.ArgumentParser(
        description="SpeedInfer Database Bootstrap & Seeding Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alembic",
        action="store_true",
        help="Use Alembic migrations instead of direct SQLModel metadata table creation.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip seeding the initial admin user and default ModelVersion.",
    )
    parser.add_argument(
        "--admin-email",
        type=str,
        default="admin@speedinfer.local",
        help="Email address for the initial administrator.",
    )
    parser.add_argument(
        "--admin-balance",
        type=float,
        default=1000.0,
        help="Initial credit balance assigned to the administrator API key.",
    )

    args = parser.parse_args()

    print("[*] Initializing SpeedInfer database...")
    result = init_db(
        use_alembic=args.alembic,
        seed=not args.no_seed,
        admin_email=args.admin_email,
        initial_balance=args.admin_balance,
    )

    print(f"[+] Database tables created successfully via {result['method']}.")
    if not args.no_seed:
        print(f"[+] Admin User: {result['admin_email']} (ID: {result['admin_user_id']})")
        if result["admin_api_key"]:
            print(f"[+] Generated Admin API Key: {result['admin_api_key']}")
            print("    NOTE: Save this key immediately; it will not be displayed again!")
        else:
            print("    Admin user already exists; existing API keys preserved.")
        model_name = result["default_model"]
        created = result["default_model_created"]
        print(f"[+] Default Model: {model_name} (created={created})")
    print("[+] Database initialization complete.")


if __name__ == "__main__":
    main()
