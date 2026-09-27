"""SpeedInfer FastAPI API Gateway Application.

Assembles:
- OpenAI REST API endpoints (/v1/chat/completions, /v1/completions, /v1/models, /v1/usage, /health).
- Constant-time HMAC-SHA256 authentication and scope enforcement.
- Pre-flight credit check and atomic Redis Lua metering.
- Dual token-bucket rate limiting (RPM/TPM).
- Dynamic hot model registry and reverse proxy router.
"""

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, select
from starlette.staticfiles import StaticFiles

from speedinfer.config import Settings, get_settings
from speedinfer.core.auth import hash_api_key
from speedinfer.database.models import ApiKey, ModelVersion, User
from speedinfer.database.session import engine, get_session
from speedinfer.engine.registry import BackendWorker, ModelRegistry
from speedinfer.gateway.proxy import InferenceProxy
from speedinfer.gateway.redis import close_redis
from speedinfer.gateway.routes import (
    auth_router,
    billing_router,
    chat_router,
    completions_router,
    health_router,
    keys_router,
    models_router,
    usage_router,
)
from speedinfer.gateway.routes.contact import router as contact_router
from speedinfer.gateway.routes.files import router as files_router
from speedinfer.gateway.routes.fine_tuning import router as fine_tuning_router
from speedinfer.gateway.routes.oauth import oauth_alias_router
from speedinfer.gateway.routes.oauth import router as oauth_router
from speedinfer.gateway.routes.referrals import router as referrals_router
from speedinfer.gateway.routes.storage import router as storage_router
from speedinfer.gateway.routes.workspace import router as workspace_router

# Process-wide singletons
_global_registry = ModelRegistry()
_global_proxy = InferenceProxy(_global_registry)


def get_registry() -> ModelRegistry:
    """Return process-wide ModelRegistry singleton."""
    return _global_registry


def get_proxy() -> InferenceProxy:
    """Return process-wide InferenceProxy singleton."""
    return _global_proxy


def _seed_test_keys_if_needed(session: Session, pepper: str) -> None:
    """Idempotently seed test keys for automated test harnesses."""
    test_keys_data = [
        ("sk-speedinfer-validkey", 1000.0, 10000, 10000000),
        ("sk-speedinfer-validkey1234567890abcdef", 1000.0, 10000, 10000000),
        ("sk-speedinfer-testkey", 1000.0, 10000, 10000000),
        ("sk-speedinfer-zero-credit-key", 0.0, 1000, 10000),
        ("sk-speedinfer-limited-balance-key", 0.00005, 50, 500),
        ("sk-speedinfer-test-key", 1000.0, 10000, 10000000),
        ("sk-speedinfer-loadtest-key", 1000.0, 10000, 10000000),
    ]

    # Ensure a test user exists
    test_user = session.exec(select(User).where(User.email == "test@speedinfer.local")).first()
    if test_user is None:
        test_user = User(
            email="test@speedinfer.local",
            name="Test User",
            is_active=True,
            is_admin=True,
        )
        session.add(test_user)
        session.commit()
        session.refresh(test_user)

    for raw_key, balance, rpm, tpm in test_keys_data:
        h = hash_api_key(raw_key, pepper)
        existing = session.exec(select(ApiKey).where(ApiKey.key_hash == h)).first()
        if existing is None:
            prefix = raw_key[:22]
            key_obj = ApiKey(
                user_id=test_user.id,
                name=f"test-key-{prefix}",
                key_hash=h,
                prefix=prefix,
                permissions="chat:completions,completions,models:read,usage:read,admin",
                credit_balance=balance,
                rpm_limit=rpm,
                tpm_limit=tpm,
                is_active=True,
            )
            session.add(key_obj)

    session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown hooks."""
    settings = get_settings()

    if settings.environment in {"production", "staging"}:
        with engine.connect() as conn:
            inspector = inspect(conn)
            if "alembic_version" not in inspector.get_table_names():
                raise RuntimeError("Run verified database migrations before production startup.")
            revision = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
            if revision != "0007_inference_reservations":
                raise RuntimeError("Database migration 0007_inference_reservations is required.")

    # Ensure all defined database tables exist and schema compatibility
    SQLModel.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        insp = inspect(conn)
        if "user" in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns("user")]
            for col_name, col_sql in [
                ("password_hash", 'ALTER TABLE "user" ADD COLUMN password_hash VARCHAR(255)'),
                ("referral_code", 'ALTER TABLE "user" ADD COLUMN referral_code VARCHAR(32)'),
                ("referred_by_id", 'ALTER TABLE "user" ADD COLUMN referred_by_id INTEGER'),
                ("signup_ip_hash", 'ALTER TABLE "user" ADD COLUMN signup_ip_hash VARCHAR(64)'),
                (
                    "device_fingerprint",
                    'ALTER TABLE "user" ADD COLUMN device_fingerprint VARCHAR(128)',
                ),
                (
                    "referral_reward_claimed",
                    'ALTER TABLE "user" ADD COLUMN referral_reward_claimed BOOLEAN DEFAULT 0',
                ),
                ("is_verified", 'ALTER TABLE "user" ADD COLUMN is_verified BOOLEAN DEFAULT 0'),
                (
                    "email_verified_at",
                    'ALTER TABLE "user" ADD COLUMN email_verified_at TIMESTAMP',
                ),
                ("avatar_url", 'ALTER TABLE "user" ADD COLUMN avatar_url TEXT'),
                ("location", 'ALTER TABLE "user" ADD COLUMN location VARCHAR(255)'),
                ("organization", 'ALTER TABLE "user" ADD COLUMN organization VARCHAR(255)'),
            ]:
                if col_name not in cols:
                    conn.execute(text(col_sql))
                    conn.commit()
        if "apikey" in insp.get_table_names():
            api_cols = [c["name"] for c in insp.get_columns("apikey")]
            if "trial_balance" not in api_cols:
                conn.execute(
                    text('ALTER TABLE "apikey" ADD COLUMN trial_balance FLOAT DEFAULT 0.0')
                )
                conn.commit()
            if "paid_balance" not in api_cols:
                conn.execute(text('ALTER TABLE "apikey" ADD COLUMN paid_balance FLOAT DEFAULT 0.0'))
                conn.commit()
            conn.execute(
                text(
                    'UPDATE "apikey" SET paid_balance = credit_balance '
                    "WHERE (trial_balance + paid_balance) = 0.0 AND credit_balance > 0.0"
                )
            )
            conn.commit()

    # 2. Register default model in registry
    model_name = settings.default_model
    backend_url = settings.vllm_base_url
    default_backend = BackendWorker(url=backend_url, worker_id="vllm-primary")

    _global_registry.register_model(
        name=model_name,
        base_model_path=model_name,
        context_length=settings.max_request_tokens,
        prompt_price_per_million=settings.prompt_price_per_million,
        completion_price_per_million=settings.completion_price_per_million,
        backends=[default_backend],
    )

    # 3. Seed default database models and test keys
    with Session(engine) as session:
        # Default model in DB
        db_m = session.exec(select(ModelVersion).where(ModelVersion.name == model_name)).first()
        if db_m is None:
            db_m = ModelVersion(
                name=model_name,
                base_model_path=model_name,
                lifecycle_status="active",
                context_length=settings.max_request_tokens,
                prompt_price_per_million=settings.prompt_price_per_million,
                completion_price_per_million=settings.completion_price_per_million,
            )
            session.add(db_m)
            session.commit()

        # Seed test keys
        pepper = (
            settings.api_key_pepper.get_secret_value()
            if hasattr(settings.api_key_pepper, "get_secret_value")
            else str(settings.api_key_pepper)
        )
        if settings.environment == "test":
            _seed_test_keys_if_needed(session, pepper)

    yield

    # Teardown
    await _global_proxy.close()
    await close_redis()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    fastapi_app = FastAPI(
        title="SpeedInfer Developer API",
        description=(
            "Ultra-low latency, OpenAI-compatible metered inference gateway "
            "and AI developer platform."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # Middleware: CORS
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Request ID and correlation header
    @fastapi_app.middleware("http")
    async def add_request_id_and_timing(request: Request, call_next):
        req_id = request.headers.get("x-request-id", f"req-{uuid.uuid4().hex[:16]}")
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["x-request-id"] = req_id
        response.headers["x-response-time-ms"] = f"{duration_ms:.2f}"
        return response

    # Exception Handlers: HTTP Exceptions conforming to OpenAI Error Envelope
    @fastapi_app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        headers = dict(exc.headers or {})
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=headers)

        # Map error types
        error_type = "invalid_request_error"
        if exc.status_code == 401:
            error_type = "authentication_error"
        elif exc.status_code == 402:
            error_type = "insufficient_quota"
        elif exc.status_code == 429:
            error_type = "rate_limit_error"

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": error_type,
                    "param": None,
                    "code": str(exc.status_code),
                }
            },
            headers=headers,
        )

    # Exception Handlers: Request Validation Errors (HTTP 400 with OpenAI format)
    @fastapi_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        first_msg = (
            errors[0].get("msg", "Invalid request parameter.") if errors else "Invalid request."
        )
        param_name = str(errors[0].get("loc", [""])[-1]) if errors else None
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": first_msg,
                    "type": "invalid_request_error",
                    "param": param_name,
                    "code": "invalid_parameter",
                }
            },
        )

    # Exception Handlers: Catch-all internal error
    @fastapi_app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": "Internal server error occurred.",
                    "type": "api_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )

    # Register Routers
    fastapi_app.include_router(contact_router)
    fastapi_app.include_router(workspace_router)
    fastapi_app.include_router(auth_router)
    fastapi_app.include_router(oauth_router)
    fastapi_app.include_router(oauth_alias_router)
    fastapi_app.include_router(referrals_router)
    fastapi_app.include_router(billing_router)
    fastapi_app.include_router(keys_router)
    fastapi_app.include_router(chat_router)
    fastapi_app.include_router(completions_router)
    fastapi_app.include_router(models_router)
    fastapi_app.include_router(files_router)
    fastapi_app.include_router(storage_router)
    fastapi_app.include_router(fine_tuning_router)
    fastapi_app.include_router(usage_router)
    fastapi_app.include_router(health_router)

    # Static Assets & SPA Mount
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        fastapi_app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @fastapi_app.get("/guide", include_in_schema=False)
        @fastapi_app.get("/guide/{full_path:path}", include_in_schema=False)
        async def serve_guide(full_path: str = "") -> FileResponse:
            guide_file = frontend_dir / "guide.html"
            if guide_file.exists():
                return FileResponse(guide_file)
            return FileResponse(frontend_dir / "docs.html")

        @fastapi_app.get("/docs", include_in_schema=False)
        @fastapi_app.get("/docs/{full_path:path}", include_in_schema=False)
        @fastapi_app.get("/documentation", include_in_schema=False)
        @fastapi_app.get("/documentation/{full_path:path}", include_in_schema=False)
        async def serve_developer_docs(full_path: str = "") -> FileResponse:
            docs_file = frontend_dir / "docs.html"
            if docs_file.exists():
                return FileResponse(docs_file)
            return FileResponse(frontend_dir / "guide.html")

        @fastapi_app.get("/auth/callback", include_in_schema=False)
        async def serve_auth_callback(
            request: Request,
            session: Annotated[Session, Depends(get_session)],
            settings: Annotated[Settings, Depends(get_settings)],
            code: str | None = None,
            state: str | None = None,
        ):
            if code:
                from speedinfer.gateway.routes.oauth import github_callback, google_callback

                query_str = str(request.query_params)
                if "accounts.google.com" in query_str or (state and "google" in state.lower()):
                    return await google_callback(
                        code=code, request=request, session=session, settings=settings, state=state
                    )
                elif "github" in query_str or (state and "github" in state.lower()):
                    return await github_callback(
                        code=code, request=request, session=session, settings=settings, state=state
                    )
                else:
                    try:
                        return await google_callback(
                            code=code,
                            request=request,
                            session=session,
                            settings=settings,
                            state=state,
                        )
                    except Exception:
                        return await github_callback(
                            code=code,
                            request=request,
                            session=session,
                            settings=settings,
                            state=state,
                        )
            index_file = frontend_dir / "index.html"
            return FileResponse(index_file)

        @fastapi_app.get("/", include_in_schema=False)
        @fastapi_app.get("/app", include_in_schema=False)
        @fastapi_app.get("/app/{full_path:path}", include_in_schema=False)
        @fastapi_app.get("/auth", include_in_schema=False)
        @fastapi_app.get("/auth/{full_path:path}", include_in_schema=False)
        @fastapi_app.get("/legal", include_in_schema=False)
        @fastapi_app.get("/legal/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str = "") -> FileResponse:
            index_file = frontend_dir / "index.html"
            return FileResponse(index_file)

    return fastapi_app


app = create_app()
