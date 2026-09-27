# =============================================================================
# SpeedInfer Production Gateway Multi-Stage Dockerfile
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python packaging tools
RUN pip install --no-cache-dir --upgrade pip hatchling

# Copy project specification
COPY pyproject.toml .
COPY speedinfer/ ./speedinfer/

# Install dependencies into virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir .

# -----------------------------------------------------------------------------
# Production Runtime Stage
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runner

WORKDIR /app

# Install minimal runtime shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root system user
RUN useradd -m -u 1000 -s /bin/bash speedinfer && \
    mkdir -p /app/data && \
    chown -R speedinfer:speedinfer /app

# Copy application source code
COPY --chown=speedinfer:speedinfer speedinfer/ /app/speedinfer/
COPY --chown=speedinfer:speedinfer migrations/ /app/migrations/
COPY --chown=speedinfer:speedinfer alembic.ini /app/alembic.ini
COPY --chown=speedinfer:speedinfer pyproject.toml /app/pyproject.toml

# Install local package in editable/wheel form inside virtualenv
RUN pip install --no-cache-dir --no-deps -e .

USER speedinfer

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "speedinfer.gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
