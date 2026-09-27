# SpeedInfer Test Infrastructure & E2E Testing Strategy

## 1. Test Philosophy

SpeedInfer's testing architecture is built upon four foundational pillars: **Opaque-Box Verification**, **Requirement-Driven Assertions**, **Zero Implementation Coupling**, and **Progressive Milestone Testability**.

```
                ┌──────────────────────────────────────────────┐
                │             Authoritative Sources            │
                │  - ORIGINAL_REQUEST.md                       │
                │  - PROJECT.md Architecture & Contracts       │
                │  - OpenAI v1 API Specification               │
                └──────────────────────┬───────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────┐
                │          Opaque-Box Test Harness             │
                │  - Public REST & SSE Interfaces              │
                │  - Database Invariants & State Mutations     │
                │  - Redis Token-Bucket & Lua State Changes    │
                └──────────────────────┬───────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ Tier 1: Features │          │ Tier 2: Boundary │          │ Tier 3: Combos   │
│ - Happy Path     │          │ - Zero / Max     │          │ - Auth+Rate+Cost │
│ - Contract I/O   │          │ - Concurrency    │          │ - Redis+DB Sync  │
└──────────────────┘          └──────────────────┘          └──────────────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ Tier 4: Scenarios│
                              │ - Real-World SDK │
                              │ - Burst / Drops  │
                              └──────────────────┘
```

### 1.1 Opaque-Box & Requirement-Driven Design
- **Black-Box Boundary**: Tests interact with SpeedInfer strictly through observable boundaries: public API endpoints (FastAPI REST and SSE byte streams), database state commits (`User`, `ApiKey`, `UsageLedger`, `ModelVersion`), and Redis key mutations.
- **Zero Facade Testing**: Tests never mock internal business logic to force passes. Every assertion validates real computation: actual cryptographic HMAC-SHA256 calculations, actual token bucket refills, and actual floating-point billing equations.
- **Specification Invariants**: Every expected value is derived from authoritative specifications in `ORIGINAL_REQUEST.md` and `PROJECT.md`, never by reverse-engineering a specific implementation's temporary quirks.

### 1.2 Progressive Milestone Testability
- Each test suite is structured to be runnable as soon as its corresponding milestone dependencies are satisfied:
  - **M1 (Database)**: Can be verified in isolation using in-memory SQLite and file-based SQLite WAL.
  - **M2 (Auth & Metering)**: Verified using in-memory Redis (`fakeredis` with Lua support) and SQLite database sessions.
  - **M3 (Engine & Registry)**: Verified using process supervision mocks and CPU-based mock HTTP worker backends.
  - **M4 (Gateway)**: Full end-to-end integration verified via `httpx.AsyncClient` with mock vLLM backends.
  - **M5 (Model Lifecycle)**: Verified via offline dataset loaders, dummy model weights, and mock MLflow tracking.
  - **M6 (Deployment)**: Verified via syntax linters, docker-compose configuration validation, and systemd unit verification.

### 1.3 Authoritative Expected Output Derivation
All test calculations use explicit mathematical and logical definitions:
1. **Pricing Equation**:
   $$\text{Total Cost} = \frac{\text{Prompt Tokens} \times \text{Prompt Price} + \text{Completion Tokens} \times \text{Completion Price}}{1,000,000}$$
2. **Preflight Credit Check**:
   $$\text{Estimated Tokens} = \text{Prompt Tokens} + \min(\text{max\_tokens}, \text{context\_window} - \text{prompt\_tokens})$$
   $$\text{Estimated Cost} = \frac{\text{Estimated Tokens} \times \text{Prompt Price}}{1,000,000}$$
   $$\text{Permitted} \iff \text{Credit Balance} \ge \text{Estimated Cost}$$
3. **Dual Token-Bucket Rate Limiter**:
   $$\text{Tokens}_{\text{new}} = \min(\text{Capacity}, \text{Tokens}_{\text{old}} + \Delta t \times \text{Refill Rate})$$
4. **Bearer Token Cryptography**:
   $$\text{Hash} = \text{HMAC-SHA256}(\text{Secret Pepper}, \text{sk-speedinfer-...})$$
   Validated strictly via constant-time `hmac.compare_digest`.
5. **SSE Stream Protocol**:
   Chunks formatted as `data: {"id": "...", "choices": [{"delta": {"content": "..."}}]}\n\n` ending with `data: [DONE]\n\n`.

---

## 2. Feature Inventory & 4-Tier Coverage Goals

| # | Feature / Contract | Tier 1: Feature Coverage | Tier 2: Boundary & Corner Cases | Tier 3: Cross-Feature Combinations | Tier 4: Real-World Scenarios |
|---|--------------------|--------------------------|---------------------------------|------------------------------------|------------------------------|
| **1** | **User Persistence** (`models.py`) | Create user with email, name, default active/admin flags. | Duplicate email triggers `IntegrityError`; invalid email formats; null name handling. | Cascading delete of User removes associated `ApiKey` records cleanly. | Multi-tenant organization creating 50 distinct user accounts concurrently. |
| **2** | **ApiKey Persistence** (`models.py`) | Create key linked to user with hash, prefix, initial balance, RPM/TPM. | Key hash collision handling; zero/negative initial balance check; max length constraints. | Updating `credit_balance` reflected in active session queries and ledger foreign keys. | Key revocation while active inference requests are in flight. |
| **3** | **UsageLedger** (`models.py`) | Log request ID, model, prompt/completion tokens, cost, latency. | Zero tokens; null TTFT; extreme token counts ($>100k$); high precision floating cost. | Linking ledger rows to `ApiKey` and rolling up daily spend aggregations. | Audit reconstruction: verify sum of ledger `cost` equals total balance deducted. |
| **4** | **ModelVersion** (`models.py`) | Store base model path, adapter path, pricing, lifecycle status. | Duplicate model name; zero pricing (free tier); negative pricing rejected. | Transition status from `training` $\to$ `evaluating` $\to$ `active` in DB. | Multi-model catalog with distinct pricing tiers queried under high read concurrency. |
| **5** | **Session Lifecycle & WAL** (`session.py`) | Engine creation; `get_session()` dependency; `session_scope()` context manager. | Rollback on unhandled exception; connection pool saturation; busy timeout under load. | SQLite WAL mode concurrency: simultaneous read and write threads without locking error. | High-frequency ledger write bursts while gateway reads key balances. |
| **6** | **Bearer Token Generation** (`auth.py`) | Key generation with prefix `sk-speedinfer-` and 32 bytes entropy. | Malformed prefixes; non-hex characters; truncated keys ($<20$ chars). | Key generation stores hash in DB and yields plaintext key exactly once. | Key rotation script creating new keys and deprecating old keys with zero downtime. |
| **7** | **Constant-Time Auth** (`auth.py`) | HMAC-SHA256 with pepper verified against stored hash using `compare_digest`. | Empty Authorization header; missing `Bearer ` prefix; invalid hash length; wrong pepper. | Auth middleware populates request context with `ApiKey` and safe prefix. | Simulated timing attack verifying $O(1)$ constant-time execution across 1,000 probes. |
| **8** | **Prefix Logging** (`auth.py`) | Extract 22-character safe prefix (`sk-speedinfer-a1b2c3d4...`) for logs. | Raw secret key never present in `repr()`, logs, or exception traces. | Structlog context binding logs prefix on every correlated request log. | Log forensics inspection verifying zero raw API key leaks across 10,000 log events. |
| **9** | **Scope Authorization** (`auth.py`) | Allow access when required scope (`chat:completions`) is granted. | Empty scope; malformed scope string; missing scope returns HTTP 403 Forbidden. | `admin` scope overrides all endpoint requirements. | Role-based keys: read-only key querying `/v1/models` fails when calling `/v1/chat/completions`. |
| **10** | **Pricing Math & Tokens** (`metering.py`) | Compute exact cost for prompt and completion token counts. | 0 prompt tokens; 0 completion tokens; fractional cents rounded accurately. | Preflight estimate matches actual deduction when completion matches `max_tokens`. | Million-token batch calculation validating zero cumulative floating point drift. |
| **11** | **Preflight Credit Check** (`metering.py`) | Allow dispatch when balance $\ge$ estimated cost; reject when deficient (402). | Exact balance == estimated cost passes; balance == estimated cost - $10^{-6}$ fails. | Preflight check precedes rate-limit consumption (no token wasted on 402). | Exhausted balance key denied dispatch before any worker GPU resources are allocated. |
| **12** | **Atomic Balance Deduction** (`metering.py`) | Redis Lua script deducts cost atomically; returns new balance. | Insufficient balance returns reject code; Redis key miss triggers DB reload. | 50 concurrent requests against single key balance: zero negative balance. | Sustained concurrency: balance drains to exactly $0.00$ with remainder requests receiving 402. |
| **13** | **Dual Token Bucket** (`rate_limiter.py`) | Enforce RPM and TPM limits via Redis Lua token bucket. | Burst request exceeding bucket capacity immediately rejected (429); clock skew backward. | Request consuming both 1 RPM and $N$ TPM; remaining headers match bucket state. | Locust benchmark simulating 20 RPS on 10 RPM key: exactly 10 succeed, 10 rejected with Retry-After. |
| **14** | **Rate Limit Headers** (`rate_limiter.py`) | Inject `x-ratelimit-limit-*`, `x-ratelimit-remaining-*`, `x-ratelimit-reset-*`. | Headers present on both HTTP 200 and HTTP 429 responses. | Reset seconds accurately decreases toward zero as refill interval elapses. | Client SDK consuming rate limit headers to implement proactive client-side throttling. |
| **15** | **vLLM CLI Builder** (`vllm_runner.py`) | Generates deterministic CLI command with TP, memory util, quantization. | Missing optional flags (quantization, chat_template); port bounds validation. | Config object serialized to CLI command executed by subprocess supervisor. | Multi-GPU node command builder generating distinct `--cuda-visible-devices` per worker. |
| **16** | **Automated Boot Warm-up** (`vllm_runner.py`) | Issues ping completion on worker startup to pre-allocate CUDA graphs. | Worker timeout ($>300s$) raises `VLLMStartupTimeoutError`; HTTP 500 on warm-up fails boot. | Worker transitioned from `WARMING` to `READY` in registry only after warm-up succeeds. | Worker restarts after simulated crash; warm-up probe executes before receiving traffic. |
| **17** | **Process Supervisor** (`vllm_runner.py`) | Monitors process exit; applies exponential backoff (2s, 4s, 8s); auto-restarts. | Unrecoverable crash (exit code 127) exhausts retries $\to$ marks `FAILED`. | Process exit removes backend from healthy pool; traffic routes to replica. | Simulated SIGKILL to worker process: supervisor recovers worker and re-registers within 10s. |
| **18** | **Dynamic Model Registry** (`registry.py`) | Hot-register model without gateway restart; query healthy backends. | Registering model with empty backends; unregistering unknown model. | Hot-swapping adapter path while active requests stream responses. | Zero-downtime model upgrade: traffic smoothly shifts from model v1 to model v2 backends. |
| **19** | **Fallback Backend Routing** (`registry.py`) | Routes traffic to secondary worker if primary is degraded or down. | All backends down returns clean HTTP 503 Service Unavailable. | Worker circuit breaker marks degraded on 3 consecutive 5xx errors; auto-recovers. | Primary worker killed mid-benchmark; subsequent requests immediately route to fallback. |
| **20** | **Non-Streaming Chat** (`chat.py`) | `POST /v1/chat/completions` with `stream=false` returns valid OpenAI JSON. | Empty messages array (400); invalid temperature ($>2.0$); unsupported model (404). | Auth $\to$ Rate Limit $\to$ Preflight $\to$ Proxy $\to$ Atomic Deduct $\to$ UsageLedger. | Standard OpenAI Python SDK `.chat.completions.create()` end-to-end conversation. |
| **21** | **Streaming Chat (SSE)** (`chat.py`) | `POST /v1/chat/completions` with `stream=true` returns SSE chunks + `[DONE]`. | Premature client disconnect (`asyncio.CancelledError`); malformed upstream chunk. | Unbuffered chunk delivery; partial token accounting and deduction upon client disconnect. | Streaming response consumed by web client; connection dropped after 5 tokens; exactly 5 metered. |
| **22** | **Legacy Text Completions** (`completions.py`)| `POST /v1/completions` returns valid OpenAI completion response. | Empty prompt; max_tokens exceeding context length; stop sequence handling. | Legacy format shares unified metering, auth, and rate limiting infrastructure. | Legacy LangChain / LlamaIndex integration querying text completion endpoint. |
| **23** | **Models & Health API** (`models.py`, `health.py`) | `GET /v1/models` lists warm models; `GET /health` reports DB, Redis, workers. | Database down returns 503; Redis down returns 503; degraded worker reports status. | Health probe reflects real-time status changes from supervisor background loop. | Kubernetes / Docker health check probe pinging `/health` every 5 seconds under load. |
| **24** | **Reverse Proxy & Balancer** (`proxy.py`, `balancer.py`) | Least-connections / Round-robin proxying across multiple worker backends. | Worker connection timeout; connection reset; retry on secondary before bytes sent. | Active request counter incremented on dispatch and decremented on completion. | 100 concurrent requests balanced evenly across 2 identical vLLM worker replicas. |
| **25** | **Structured JSON Logging** (`middleware.py`)| Logs JSON event with `request_id`, `duration_ms`, `key_prefix`, `tokens`. | Custom `X-Request-ID` header preserved; missing header auto-generates UUID4. | Request ID injected into HTTP response headers and propagated to worker headers. | Distributed trace inspection linking client request ID with gateway and backend worker logs. |
| **26** | **LoRA Training Pipeline** (`train.py`) | Configures `SFTTrainer` with YAML config, QLoRA 4-bit, target modules. | Missing YAML keys; invalid learning rate; CUDA OOM catches and logs guidance. | Training output directory contains valid `adapter_model.safetensors` and config. | Offline fine-tuning dry run on dummy conversation dataset completing 2 steps. |
| **27** | **Data Utils & ChatML** (`data_utils.py`) | Loads JSONL/Parquet; formats to ChatML; applies completion-only masking (-100). | Tokenizer missing chat template triggers fallback; sequences exceeding max_seq_len. | Deterministic train/validation split preserving conversation integrity. | Dataset pipeline preparing 1,000 multi-turn conversations for training loss calculation. |
| **28** | **Model Merge & Export** (`export.py`) | Merges LoRA adapter into base weights; saves sharded safetensors. | Adapter architecture mismatch; output directory permission error. | Exported directory contains `model.safetensors`, `config.json`, tokenizer assets. | Export validation: safetensors model loaded by mock vLLM runner and verified valid. |
| **29** | **Model Evaluation Harness** (`evaluate.py`) | Calculates perplexity on validation corpus; evaluates promotion criteria. | Empty eval dataset; high perplexity fails promotion gate. | Successful evaluation updates `ModelVersion.lifecycle_status = "active"` in DB. | Automated CI/CD promotion gate verifying model quality before production traffic routing. |
| **30** | **Deployment Automation** (`deploy/`) | Syntactically validates Docker Compose, Caddyfile, systemd service units. | Missing environment variables; invalid Caddy syntax; broken systemd dependencies. | Idempotent `deploy.sh` script handles migrations and phased health checks. | Automated deployment simulation validating zero downtime during configuration updates. |

---

## 3. Test Architecture & Runner Infrastructure

### 3.1 Directory Layout
```
tests/
├── conftest.py                     # Master test harness: fixtures, mocks, test settings
├── unit/
│   ├── test_harness.py             # Self-verification of test harness & fixtures
│   ├── test_database.py            # Models, validations, relationships, session rollback, SQLite WAL
│   ├── test_auth.py                # Bearer crypto, HMAC-SHA256, constant-time, prefix logging, scopes
│   ├── test_metering.py            # Pricing math, preflight credit check, Redis Lua atomic deduction
│   ├── test_rate_limiter.py        # Token-bucket algorithm, RPM/TPM limits, headers
│   ├── test_vllm_runner.py         # Subprocess config, CLI builder, supervisor state transitions
│   └── test_registry.py            # Dynamic hot-registry, backend health, fallback routing
└── integration/
    ├── test_gateway_chat.py        # Non-streaming chat completions, balance deduction, DB ledger
    ├── test_gateway_streaming.py   # SSE streaming chunks, client disconnect, partial token accounting
    ├── test_completions_legacy.py  # Legacy text completions endpoint
    ├── test_concurrency_race.py    # 50 concurrent requests against single key balance limit
    ├── test_openai_sdk.py          # Official OpenAI Python SDK compatibility
    └── test_deployment_configs.py  # Syntax and validation of Docker Compose, Caddyfile, systemd
```

### 3.2 Runner Commands
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run all integration tests
pytest tests/integration/ -v

# Run full test suite with coverage
pytest tests/ -v --cov=speedinfer --cov-report=term-missing

# Run concurrency race condition tests specifically
pytest tests/integration/test_concurrency_race.py -v -s

# Run fast fail mode
pytest tests/ -x --tb=short
```

### 3.3 Mocking Strategies (CPU Host Environment)

#### 1. GPU / vLLM Worker Mocking
To execute realistic end-to-end tests on CPU development workstations without NVIDIA GPU hardware:
- **`MockVLLMServer`**: An in-process ASGI mock service mounted via `httpx.AsyncClient(transport=ASGITransport(mock_vllm_app))` or simulated via `respx`.
- **Streaming SSE Generator**:
  - Emits properly delimited chunks:
    ```
    data: {"id":"chatcmpl-mock-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}\n\n
    data: {"id":"chatcmpl-mock-1","object":"chat.completion.chunk","choices":[{"delta":{"content":" world!"}}]}\n\n
    data: [DONE]\n\n
    ```
  - Configurable simulated TTFT latency (e.g. 50ms) and inter-token generation delays.
  - Ability to inject upstream HTTP 500/502 errors to test fallback balancer behavior.
  - Ability to pause midway through generation to simulate abrupt client disconnects (`asyncio.CancelledError`).

#### 2. Redis & Lua Script Mocking
- **`fakeredis[lua]` with `lupa`**:
  - Executes the **exact production Lua scripts** (`balance_deduct.lua`, `token_bucket.lua`) inside an embedded Lua 5.1 runtime.
  - Guarantees 100% fidelity for Redis atomic operations, multi-key scripts, and expiration logic without requiring an external Redis server daemon.
  - Auto-flushed (`redis.flushall()`) between tests to guarantee zero test cross-contamination.

#### 3. Database Isolation Strategy
- **SQLite In-Memory with `StaticPool`**:
  - Uses `sqlite:///:memory:` with SQLAlchemy `StaticPool` and `connect_args={"check_same_thread": False}`.
  - Enables sub-millisecond database setup and teardown for unit tests.
  - Enforces foreign keys via `PRAGMA foreign_keys = ON;`.
- **File-based SQLite WAL Testing**:
  - Dedicated tests in `test_database.py` create a temporary SQLite database file on disk, verify `PRAGMA journal_mode = WAL;`, and execute concurrent multi-threaded writes to validate zero lock contention.

---

## 4. Verification & Execution Status

This test infrastructure is designed for continuous execution during every milestone. As each milestone's implementation code lands, the corresponding test suite activates and provides immediate regression safety.
