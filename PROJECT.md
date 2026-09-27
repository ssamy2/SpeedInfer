# Project: SpeedInfer

## Architecture
SpeedInfer is an end-to-end, ultra-low latency, OpenAI-compatible AI platform hosted on high-performance NVIDIA GPU servers (OVHcloud), supporting the full lifecycle of open-source LLMs (fine-tuning, deployment, serving via a secure metered API, and continuous evaluation).

```
                           [ Clients / OpenAI SDK / Curl ]
                                         │
                                         ▼ HTTPS / HTTP/3
                             [ Caddy Reverse Proxy ]
                            (Auto TLS, SSE Flush -1)
                                         │
                                         ▼ HTTP :8000
                ┌──────────────────────────────────────────────────┐
                │          SpeedInfer FastAPI Gateway (M4)         │
                │  - Auth Middleware (Bearer sk-speedinfer-...)    │
                │  - Request ID Correlation (structlog JSON)       │
                │  - Token Bucket Rate Limiter (RPM & TPM via Redis│
                │  - Atomic Balance Metering (Redis Lua + Postgres)│
                │  - Dynamic Model Registry & Health Filter        │
                │  - Load Balancer & Proxy (RoundRobin/LeastConn)  │
                └───────────┬──────────────────────────┬───────────┘
                            │                          │
              SSE / HTTP    │                          │ SSE / HTTP
              Forwarding    │                          │ Forwarding
                            ▼                          ▼
                  ┌──────────────────┐       ┌──────────────────┐
                  │ vLLM Worker #1   │       │ vLLM Worker #2   │
                  │ (Primary Port    │       │ (Replica /       │
                  │  8001, L40S GPU) │       │  Fallback)       │
                  └─────────▲────────┘       └─────────▲────────┘
                            │                          │
                 Managed Subprocess/Service (M3)       │
                 [ speedinfer/engine/vllm_runner.py ]  │
                 - CLI Arg Builder                     │
                 - Process Supervisor & Restart Backoff│
                 - Boot Warm-up Probe (CUDA graph init)│
                 - Health Monitoring Loop              │
```

- **Persistence Layer (M1)**: SQLModel with SQLite for dev/test (WAL mode) and PostgreSQL for production. Alembic migrations provide schema evolution.
- **Authentication & Metering (M2)**: Constant-time HMAC-SHA256 Bearer auth with pepper. Token-bucket rate limiting (RPM/TPM) and atomic balance deduction via Redis Lua script, asynchronously persisted to PostgreSQL `UsageLedger`.
- **Inference Engine (M3)**: Out-of-process vLLM supervisor with warm-up probing, graceful crash recovery with exponential backoff, and dynamic model hot-registry.
- **API Gateway (M4)**: High-throughput FastAPI application routing OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/usage`, `/health`) with unbuffered SSE streaming and least-connections load balancing.
- **Model Lifecycle (M5)**: Complete fine-tuning pipeline (`train.py`), data utilities with chat templates (`data_utils.py`), MLflow tracking, LoRA weight merge & safetensors export (`export.py`), and evaluation harness (`evaluate.py`).
- **Deployment Automation (M6)**: 7-container Docker Compose topology, Caddy auto-TLS reverse proxy, bare-metal systemd units, and idempotent `deploy.sh`.
- **E2E Testing Track**: Comprehensive test suite (Tiers 1-4) derived from user requirements, validating functional correctness, concurrency race conditions, and real-world workloads.
- **Final Milestone**: 100% E2E test pass followed by Tier 5 adversarial coverage hardening.

---

## Feature Inventory
| # | Category | Feature | Description | Milestone | Source |
|---|----------|---------|-------------|-----------|--------|
| 1 | Training | LoRA/QLoRA Fine-Tuning | Fine-tunes open-source LLMs using transformers, peft, trl with YAML config | M5 | ORIGINAL_REQUEST:14 |
| 2 | Training | YAML Configuration Parser | Validates model paths, LoRA hyperparams, quantization, and MLflow logging | M5 | ORIGINAL_REQUEST:14 |
| 3 | Training | Dataset Loading & Formatting | Formats raw conversations to ChatML and applies tokenizer chat template | M5 | ORIGINAL_REQUEST:15 |
| 4 | Training | Train/Val Split & Masking | Deterministic split and assistant-completion-only loss masking | M5 | ORIGINAL_REQUEST:15 |
| 5 | Training | MLflow Experiment Tracking | Logs hyperparams, loss, LR schedules, throughput, and GPU stats | M5 | ORIGINAL_REQUEST:16 |
| 6 | Training | MLflow Artifacts & Registry | Logs adapter checkpoints and registers model in MLflow Registry | M5 | ORIGINAL_REQUEST:16 |
| 7 | Training | LoRA Weight Merge | Merges LoRA adapter into base model in float16/bfloat16 | M5 | ORIGINAL_REQUEST:17 |
| 8 | Training | Safetensors Export in vLLM Format | Saves merged weights in sharded safetensors with configs and tokenizer | M5 | ORIGINAL_REQUEST:17 |
| 9 | Training | Post-Training Quantization | Optional FP8, AWQ, or GPTQ post-training quantization | M5 | ORIGINAL_REQUEST:17 |
| 10 | Training | Perplexity Benchmark | Sliding-window perplexity calculation on validation corpus | M5 | ORIGINAL_REQUEST:18 |
| 11 | Training | Benchmark & Promotion Gate | Evaluates against quality threshold; updates model lifecycle status in DB | M5 | ORIGINAL_REQUEST:18 |
| 12 | Database | SQLModel User Model | Persistent entity for user accounts | M1 | ORIGINAL_REQUEST:22 |
| 13 | Database | SQLModel ApiKey Model | Persistent entity storing hashed secret keys, prefix, balance, RPM/TPM | M1 | ORIGINAL_REQUEST:22 |
| 14 | Database | SQLModel UsageLedger Model | Immutable audit ledger logging requests, token counts, cost, latencies | M1 | ORIGINAL_REQUEST:22 |
| 15 | Database | SQLModel ModelVersion Model | Registry table for model versions, weights, adapters, pricing, lifecycle | M1 | ORIGINAL_REQUEST:22 |
| 16 | Database | Dual-Dialect Session Lifecycle | Engine factory supporting SQLite (dev/test WAL mode) and PostgreSQL (prod) | M1 | ORIGINAL_REQUEST:23 |
| 17 | Database | Database Initialization Routine | Automated table creation and default seed bootstrap | M1 | ORIGINAL_REQUEST:23 |
| 18 | Database | Alembic Migration Setup | Alembic environment configured to SQLModel metadata and database_url | M1 | ORIGINAL_REQUEST:24 |
| 19 | Database | Initial Migration Script | Versioned migration (0001_initial_schema) creating all tables and indexes | M1 | ORIGINAL_REQUEST:24 |
| 20 | Core Auth | Bearer Key Generator | Generates `sk-speedinfer-...` cryptographically secure keys | M2 | ORIGINAL_REQUEST:28 |
| 21 | Core Auth | Constant-Time Key Validation | Hashes key with SHA-256 + pepper, constant-time compare against DB | M2 | ORIGINAL_REQUEST:28 |
| 22 | Core Auth | Key Prefix Logging & Redaction | Logs only safe prefix (`sk-speedinfer-xxxx...`), never full key | M2 | ORIGINAL_REQUEST:28 |
| 23 | Core Auth | Scope-Based Authorization | Validates required permissions before dispatching request | M2 | ORIGINAL_REQUEST:28 |
| 24 | Core Metering | Non-Streaming Token Accounting | Extracts token counts, computes cost from model pricing | M2 | ORIGINAL_REQUEST:29 |
| 25 | Core Metering | SSE Streaming Token Accounting | Parses SSE chunks in real-time, accounts for partial tokens on disconnect | M2 | ORIGINAL_REQUEST:29 |
| 26 | Core Metering | Pre-Flight Credit Check | Rejects request before dispatch if balance < estimated max cost | M2 | ORIGINAL_REQUEST:30 |
| 27 | Core Metering | Atomic Balance Deduction | Redis Lua script ensures zero negative balance under race conditions | M2 | ORIGINAL_REQUEST:31 |
| 28 | Core Metering | Async DB Ledger Sync | Persists usage records and flushes updated balance to PostgreSQL | M2 | ORIGINAL_REQUEST:31 |
| 29 | Core Rate Limit | Redis Token Bucket Rate Limiter | Dual token-bucket enforcing RPM and TPM limits per key | M2 | ORIGINAL_REQUEST:32 |
| 30 | Core Rate Limit | Rate Limit Response Headers | Injects standard rate-limiting headers (limit, remaining, reset) | M2 | ORIGINAL_REQUEST:32 |
| 31 | Engine | vLLM Subprocess Runner | Configures and launches vLLM with TP, memory util, max_len, quantization | M3 | ORIGINAL_REQUEST:36 |
| 32 | Engine | Automated Boot Warm-up | Probes worker with dummy completion to compile CUDA graphs before ready | M3 | ORIGINAL_REQUEST:37 |
| 33 | Engine | Supervisor & Crash Recovery | Monitors process exit, applies backoff, recovers crashed workers | M3 | ORIGINAL_REQUEST:37 |
| 34 | Engine | Dynamic Hot Model Registry | Thread-safe registry for hot-registering models without gateway restart | M3 | ORIGINAL_REQUEST:38 |
| 35 | Engine | Fallback Backend Routing | Routes requests to secondary replica or fallback model if primary degraded | M3 | ORIGINAL_REQUEST:37 |
| 36 | Gateway | POST /v1/chat/completions (Non-Stream)| OpenAI-compatible chat completion endpoint | M4 | ORIGINAL_REQUEST:42 |
| 37 | Gateway | POST /v1/chat/completions (SSE Stream) | OpenAI-compatible SSE streaming completion with unbuffered chunks | M4 | ORIGINAL_REQUEST:42 |
| 38 | Gateway | POST /v1/completions (Legacy) | Legacy text completions endpoint | M4 | ORIGINAL_REQUEST:43 |
| 39 | Gateway | GET /v1/models | Returns active models from dynamic registry | M4 | ORIGINAL_REQUEST:44 |
| 40 | Gateway | GET /v1/usage & GET /health | Returns user credit balance, usage stats, and system health status | M4 | ORIGINAL_REQUEST:45 |
| 41 | Gateway | Reverse Proxy & Load Balancer | Round-robin / least-connections proxying across vLLM workers | M4 | ORIGINAL_REQUEST:46 |
| 42 | Gateway | Structured JSON Logging | Structured JSON logging with request ID correlation | M4 | ORIGINAL_REQUEST:47 |
| 43 | Gateway | Strict Error Response Mapping | Precise mapping of exceptions to OpenAI error JSON format and HTTP codes | M4 | ORIGINAL_REQUEST:67 |
| 44 | Deployment | Multi-Container Docker Compose | 7-container topology (gateway, vllm, postgres, redis, mlflow, prom, grafana)| M6 | ORIGINAL_REQUEST:51 |
| 45 | Deployment | Caddy Auto-TLS & Streaming Proxy | Caddyfile with HTTP/2, HTTP/3, and unbuffered SSE flush_interval -1 | M6 | ORIGINAL_REQUEST:52 |
| 46 | Deployment | Bare-Metal Systemd Unit Services | Systemd service units for gateway and vLLM runner | M6 | ORIGINAL_REQUEST:53 |
| 47 | Deployment | Idempotent deploy.sh Script | Automated repo update, migrations, phased restarts, health rollback | M6 | ORIGINAL_REQUEST:54 |
| 48 | Testing | Pytest Unit & Integration Suite | Tests covering auth, metering race conditions, streaming edge cases | E2E Track | ORIGINAL_REQUEST:58 |
| 49 | Testing | scripts/test_inference.py CLI | Automated verification of streaming/non-streaming and balance deduction | E2E Track | ORIGINAL_REQUEST:59 |
| 50 | Testing | scripts/load_test.py Locust Bench | Load testing validating rate limit enforcement and latency under concurrency| E2E Track | ORIGINAL_REQUEST:60 |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Database & State Persistence | `speedinfer/database/` (models.py, session.py, alembic migrations) | none | PLANNED |
| M2 | Authentication & Metering Engine | `speedinfer/core/` (auth.py, metering.py, rate_limiter.py, redis lua) | M1 | PLANNED |
| M3 | Inference Engine & Process Manager | `speedinfer/engine/` (vllm_runner.py, registry.py, supervisor, warm-up) | M1 | PLANNED |
| M4 | OpenAI-Compatible API Gateway | `speedinfer/gateway/` (app.py, routes, proxy, streaming SSE, schemas) | M2, M3 | PLANNED |
| M5 | Model Lifecycle Management | `speedinfer/training/` (train.py, data_utils.py, export.py, evaluate.py) | M1 | PLANNED |
| M6 | Deployment & Infrastructure Automation | `deploy/` (docker-compose.yml, Caddyfile, systemd, deploy.sh, README, .env)| M1-M5 | PLANNED |
| E2E | E2E Testing Track | `tests/`, `scripts/` (Tiers 1-4 test suites, test_inference.py, load_test.py) | Independent (Dual Track) | PLANNED |
| Final | Final Milestone & Hardening | Pass 100% of E2E test suite + Tier 5 Adversarial Coverage Hardening | M1-M6, E2E | PLANNED |

---

## Interface Contracts

### M1 (Database) ↔ M2 (Core Auth & Metering)
- `ApiKey` model provides:
  * `id: int`, `user_id: int`, `key_hash: str`, `prefix: str`, `name: str`, `permissions: str`, `credit_balance: float`, `rpm_limit: int`, `tpm_limit: int`, `is_active: bool`.
- `UsageLedger` model provides:
  * `id: int`, `api_key_id: int`, `request_id: str`, `model: str`, `prompt_tokens: int`, `completion_tokens: int`, `total_cost: float`, `latency_ms: float`, `ttft_ms: Optional[float]`, `status_code: int`, `created_at: datetime`.
- `get_session()` dependency provides an active SQLAlchemy/SQLModel Session.

### M1 (Database) ↔ M3 (Engine & Registry)
- `ModelVersion` model provides:
  * `id: int`, `name: str`, `base_model_path: str`, `adapter_path: Optional[str]`, `context_length: int`, `prompt_price_per_million: float`, `completion_price_per_million: float`, `lifecycle_status: str` (e.g. "active", "evaluating", "deprecated"), `created_at: datetime`.

### M2 (Core) & M3 (Engine) ↔ M4 (Gateway)
- `authenticate_api_key(authorization_header: str, session: Session) -> ApiKey`:
  * Validates Bearer token in constant time, returns `ApiKey` or raises HTTP 401.
- `check_preflight_credit(api_key: ApiKey, estimated_tokens: int, pricing: ModelPricing) -> None`:
  * Verifies `api_key.credit_balance >= estimated_cost`, raises HTTP 402 if deficient.
- `check_rate_limit(redis_client, api_key: ApiKey, requested_tokens: int) -> RateLimitResult`:
  * Returns `allowed: bool`, `remaining_requests: int`, `remaining_tokens: int`, `reset_seconds: int`. Raises HTTP 429 if not allowed.
- `deduct_balance_atomic(redis_client, api_key_id: int, cost: float) -> float`:
  * Lua script execution returning new balance or raising insufficient balance error.
- `ModelRegistry` provides:
  * `get_model(model_name: str) -> Optional[ModelEntry]`
  * `list_models() -> List[ModelEntry]`
  * `get_healthy_backend(model_name: str) -> Optional[BackendWorker]`

### M4 (Gateway) ↔ Clients (OpenAI v1 REST API)
- `POST /v1/chat/completions`: Accepts `ChatCompletionRequest`, returns `ChatCompletionResponse` or SSE stream of `ChatCompletionChunk`.
- `POST /v1/completions`: Accepts `CompletionRequest`, returns `CompletionResponse`.
- `GET /v1/models`: Returns `ModelListResponse`.
- `GET /v1/usage`: Returns usage summary and remaining balance.
- `GET /health`: Returns `{status: "ok", backends: [...]}`.

---

## Code Layout
```
SpeedInfer/
├── .agents/                        # Agent metadata, plans, progress, handoffs (NO code here)
├── .env.example                    # Documented configuration variables
├── README.md                       # Architecture & operational instructions
├── pyproject.toml                  # Python package configuration & dependencies
├── alembic.ini                     # Alembic migration configuration
├── migrations/                     # Alembic migration scripts
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial_schema.py
├── speedinfer/
│   ├── __init__.py
│   ├── config.py                   # Validated Pydantic Settings
│   ├── logging.py                  # Structlog JSON configuration
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py               # SQLModel entities (User, ApiKey, UsageLedger, ModelVersion)
│   │   ├── session.py              # Dual-dialect session lifecycle (SQLite WAL / PostgreSQL)
│   │   └── init_db.py              # Schema creation and seed bootstrap
│   ├── core/
│   │   ├── __init__.py
│   │   ├── auth.py                 # Constant-time HMAC-SHA256 Bearer validation, prefix logging
│   │   ├── metering.py             # Token accounting (SSE/non-streaming), Redis Lua atomic deduction
│   │   ├── rate_limiter.py         # Dual token-bucket rate limiter (RPM/TPM)
│   │   └── lua/
│   │       ├── balance_deduct.lua  # Atomic balance deduction script
│   │       └── token_bucket.lua    # Atomic token-bucket script
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── vllm_runner.py          # Subprocess manager, warm-up probe, crash supervisor
│   │   └── registry.py             # Hot dynamic model registry and fallback router
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI application entrypoint
│   │   ├── schemas.py              # Pydantic OpenAI v1 request/response schemas
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py             # /v1/chat/completions (streaming & non-streaming)
│   │   │   ├── completions.py      # /v1/completions legacy
│   │   │   ├── models.py           # /v1/models
│   │   │   └── health.py           # /health and /v1/usage
│   │   ├── proxy.py                # Reverse proxy, SSE chunk forwarder, and load balancer
│   │   └── middleware.py           # Request ID, logging, rate limit headers
│   └── training/
│       ├── __init__.py
│       ├── train.py                # LoRA/QLoRA fine-tuning runner
│       ├── config.py               # YAML training config parser
│       ├── data_utils.py           # Dataset loading, tokenization, chat template formatting
│       ├── export.py               # LoRA merge & safetensors export in vLLM format
│       └── evaluate.py             # Perplexity and benchmark evaluation harness
├── deploy/
│   ├── docker-compose.yml          # 7-container topology with NVIDIA GPU runtime
│   ├── Caddyfile                   # Reverse proxy with auto TLS & unbuffered SSE
│   ├── systemd/
│   │   ├── speedinfer-gateway.service
│   │   └── speedinfer-vllm.service
│   └── deploy.sh                   # Idempotent automated deployment script
├── scripts/
│   ├── test_inference.py           # E2E CLI testing streaming/non-streaming & balance diffs
│   └── load_test.py                # Locust load-testing script
└── tests/
    ├── conftest.py                 # Async client, mock redis, sqlite memory engine
    ├── unit/
    │   ├── test_database.py
    │   ├── test_auth.py
    │   ├── test_metering.py
    │   ├── test_engine.py
    │   └── test_training.py
    └── integration/
        ├── test_gateway_chat.py
        ├── test_gateway_streaming.py
        ├── test_concurrency_race.py
        └── test_openai_sdk.py
```
