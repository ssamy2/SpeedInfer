> **Current release: preview, not production-certified.** See [production readiness audit](docs/PRODUCTION_READINESS_AR.md) and the served `/guide` for supported APIs, billing and GPU launch gates. No measured GPU performance or blanket zero-retention guarantee is claimed.

<p align="center">
  <img src="speedinfer/frontend/assets/speedinfer-icon.svg" width="140" height="140" alt="SpeedInfer AI Logo">
</p>

<h1 align="center">SpeedInfer AI ⚡</h1>

<p align="center">
  <b>Managed Model API & Developer Platform</b><br>
  <i>OpenAI-compatible text serving through a connected model worker; training and deployment workspace workflows are currently simulated.</i>
</p>

<p align="center">
  <a href="https://speedinfer.com"><img src="https://img.shields.io/badge/Live_Site-speedinfer.com-38ef7d?style=for-the-badge&logo=fastapi&logoColor=white" alt="Live Site"></a>
  <a href="https://speedinfer.com/docs"><img src="https://img.shields.io/badge/Docs-API_Reference-3b82f6?style=for-the-badge&logo=gitbook&logoColor=white" alt="Documentation"></a>
  <img src="https://img.shields.io/badge/OpenAI-Drop--in_API-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI Compatible">
</p>

<p align="center">
  <img src="docs/assets/speedinfer-banner.png" width="850" alt="SpeedInfer Overview Banner">
</p>

---

## 🚀 Key Features

- **OpenAI-Compatible API Gateway:** Supported text-only OpenAI-compatible API subset (`/v1/chat/completions` with unbuffered SSE streaming, `/v1/completions`, `/v1/models`, `/v1/usage`, `/health`).
- **High-Performance Inference Engine:** Built on **vLLM** with tensor parallelism, PagedAttention, KV cache management, continuous batching, and circuit-breaker fault tolerance.
- **Enterprise Authentication & Metering:** Cryptographically secure API keys (`sk-speedinfer-...`) with HMAC-SHA256 hashing and pepper, durable database reservations and settlement, pre-flight credit checks, and token-bucket RPM/TPM rate limiting.
- **Model Lifecycle & Fine-Tuning:** Hugging Face `transformers` + `peft` (LoRA/QLoRA) + `trl`, YAML-driven configs, MLflow experiment tracking and model registry, automated adapter merging, and perplexity evaluation harness.
- **Dual Persistence Layer:** SQLModel / SQLAlchemy unified schema supporting SQLite for rapid local testing and PostgreSQL for high-concurrency production deployments, managed via Alembic migrations.
- **Deployment templates (require environment validation):** Multi-container Docker Compose 7-service topology (Gateway, vLLM, PostgreSQL, Redis, MLflow, Prometheus, Grafana), Caddy reverse proxy with automatic TLS, and bare-metal systemd unit definitions.

---

## 🏗️ Repository Architecture

```text
SpeedInfer/
├── speedinfer/
│   ├── config.py                 # Validated Pydantic BaseSettings
│   ├── logging.py                # Structured JSON logging with request ID correlation
│   ├── database/
│   │   ├── models.py             # SQLModel models: User, ApiKey, UsageLedger, ModelVersion
│   │   └── session.py            # Async engine session management (SQLite / PostgreSQL)
│   ├── core/
│   │   ├── auth.py               # Bearer HMAC-SHA256 auth, pepper, prefix masking
│   │   ├── metering.py           # Token accounting, Redis Lua atomic balance deduction
│   │   └── rate_limiter.py       # Dual token-bucket rate limiter (RPM & TPM)
│   ├── engine/
│   │   ├── registry.py           # Hot dynamic model registry with circuit breaking
│   │   └── vllm_runner.py        # vLLM subprocess supervisor & warm-up orchestrator
│   ├── gateway/
│   │   ├── app.py                # FastAPI ASGI application & lifecycle lifespan
│   │   ├── proxy.py              # Streaming HTTP client reverse proxy to vLLM
│   │   ├── redis.py              # Redis connection pool & Lua script manager
│   │   ├── schemas.py            # Pydantic v2 OpenAI-compatible request/response schemas
│   │   └── routes/               # Modular route handlers (chat, completions, models, usage, health)
│   └── training/
│       ├── config.py             # YAML-serializable FineTuningConfig & LoRA schemas
│       ├── data_utils.py         # Dataset loading, tokenization, chat templating, split
│       ├── train.py              # LoRA/QLoRA SFT training loop with MLflow tracking
│       ├── merge_and_export.py   # Adapter weight merger to safetensors
│       ├── evaluate.py           # Perplexity and benchmark evaluation harness
│       └── configs/              # Ready-to-use YAML configs (default_lora.yaml)
├── deploy/
│   ├── docker-compose.yml        # 7-container production topology
│   ├── Caddyfile                 # Auto-TLS reverse proxy with flush_interval -1
│   ├── prometheus.yml            # Prometheus scrape targets
│   ├── deploy.sh                 # Automated idempotent deployment script
│   └── systemd/                  # Bare-metal systemd unit files (gateway & vLLM)
├── migrations/                   # Alembic schema migration environment
├── scripts/
│   ├── test_inference.py         # End-to-end inference verification CLI
│   └── load_test.py              # Locust concurrent load and rate-limit benchmark
├── tests/
│   ├── unit/                     # Fast unit tests for auth, metering, registry, runner
│   └── integration/              # Gateway streaming, OpenAI SDK, and race tests
├── Dockerfile                    # Multi-stage production container build
├── pyproject.toml                # Hatchling build specification and dependencies
└── .env.example                  # Environment variable reference
```

---

## 🛠️ Quickstart: Local Development

### 1. Environment Setup

SpeedInfer requires Python 3.12+ (or 3.14 with compatible virtualenv).

```bash
# Clone repository and enter directory
cd SpeedInfer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core and development dependencies
pip install -e '.[dev,training]'

# Copy environment configuration
cp .env.example .env
```

Ensure `API_KEY_PEPPER` in `.env` is set to a secure secret (minimum 16 characters):
```bash
# Generate random 32-byte hex string
openssl rand -hex 32
```

### 2. Database Migrations

Apply Alembic migrations to initialize the database:
```bash
alembic upgrade head
```

### 3. Run Gateway

```bash
uvicorn speedinfer.gateway.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at `http://localhost:8000/docs`.

---

## 📡 API Usage & OpenAI SDK Integration

SpeedInfer is 100% wire-compatible with the official OpenAI client libraries:

### Python OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-speedinfer-your-api-key",
)

# Non-streaming Chat Completion
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain KV caching in LLMs."},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)

# Streaming Chat Completion (SSE)
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Write a poem on parallel computing."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### cURL

```bash
# List available models
curl -H "Authorization: Bearer sk-speedinfer-test-key" \
     http://localhost:8000/v1/models

# Streaming chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Authorization: Bearer sk-speedinfer-test-key" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-7B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}],
       "stream": true
     }'
```

---

## 🎯 Model Lifecycle & Fine-Tuning Pipeline

SpeedInfer includes an end-to-end model training, export, and evaluation harness located in `speedinfer/training/`.

### 1. Fine-Tuning (LoRA / QLoRA)
```bash
python3 -m speedinfer.training.train \
    --config speedinfer/training/configs/default_lora.yaml
```
- Supports 4-bit NormalFloat4 (NF4) and double quantization.
- Automatically tracks loss, gradient norms, and learning rate schedules in **MLflow**.
- Saves adapter weights to `./output/qwen-lora-adapter`.

### 2. Adapter Merging & Safetensors Export
```bash
python3 -m speedinfer.training.merge_and_export \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --adapter-path "./output/qwen-lora-adapter" \
    --output-dir "./models/qwen2.5-7b-finetuned" \
    --device cuda
```
Produces a standalone, merged safetensors model ready for direct vLLM serving.

### 3. Model Evaluation Harness
```bash
python3 -m speedinfer.training.evaluate \
    --model-path "./models/qwen2.5-7b-finetuned" \
    --dataset-path "./data/eval_set.jsonl" \
    --batch-size 4
```
Calculates cross-entropy loss and perplexity to validate model performance prior to deployment.

---

## 🐳 Production Deployment

### Docker Compose (Full Stack)

The 7-service production topology includes:
- **speedinfer-gateway**: FastAPI reverse proxy and metering layer.
- **vllm**: Accelerated LLM server with NVIDIA GPU support.
- **postgres**: Relational database for API keys and ledger.
- **redis**: Low-latency cache, token-bucket rate limiter, atomic Lua deduction.
- **mlflow**: Experiment tracking and model artifact repository.
- **prometheus**: Real-time metrics collector.
- **grafana**: Operational dashboards.

Deploy using the automated deployment script:
```bash
./deploy/deploy.sh docker
```

Or run directly:
```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Bare-Metal (systemd)

For bare-metal deployments (e.g., OVHcloud NVIDIA HGX/H100 instances):
```bash
./deploy/deploy.sh baremetal
```
Installs and enables `speedinfer-vllm.service` and `speedinfer-gateway.service`.

---

## 🧪 Testing & Benchmarking

### Automated Unit and Integration Tests

Run the unit and integration suites (see the readiness audit for current results):
```bash
pytest -v
```

### End-to-End Inference Verification

Run the automated verification script:
```bash
python3 scripts/test_inference.py \
    --base-url http://localhost:8000 \
    --api-key sk-speedinfer-test-key \
    --model Qwen/Qwen2.5-7B-Instruct
```
Verifies gateway `/health`, initial balance, non-streaming completion, SSE streaming delivery, and credit balance deduction.

### Locust Concurrent Load Testing

Run Locust load tests against the gateway:
```bash
locust -f scripts/load_test.py \
    --headless -u 20 -r 5 --run-time 30s \
    --host http://localhost:8000
```
Tests concurrent TTFT latency, streaming stability, and dual token-bucket rate limits under heavy traffic.

---

## 📄 License

Apache 2.0. See [LICENSE](LICENSE) for details.

