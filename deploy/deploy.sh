#!/usr/bin/env bash
# =============================================================================
# SpeedInfer Idempotent Production Deployment Script
# =============================================================================
# Usage:
#   ./deploy/deploy.sh [--mode baremetal|docker]
# =============================================================================

set -euo pipefail

DEPLOY_MODE="${1:-docker}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [INFO] $*"
}

error() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [ERROR] $*" >&2
    exit 1
}

cd "${PROJECT_ROOT}"

log "Starting SpeedInfer deployment in '${DEPLOY_MODE}' mode..."

# 1. Verify Environment File
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        log "Creating initial .env from .env.example..."
        cp .env.example .env
    else
        error ".env file missing and .env.example not found."
    fi
fi

if [[ "${DEPLOY_MODE}" == "docker" ]]; then
    # 2. Check Docker and Docker Compose
    command -v docker >/dev/null 2>&1 || error "Docker is not installed."

    log "Checking docker compose configuration..."
    docker compose -f deploy/docker-compose.yml config >/dev/null

    log "Pulling latest container images..."
    docker compose -f deploy/docker-compose.yml pull postgres redis mlflow prometheus grafana || true

    log "Building SpeedInfer gateway image..."
    docker compose -f deploy/docker-compose.yml build gateway

    log "Starting stateful dependencies (PostgreSQL, Redis)..."
    docker compose -f deploy/docker-compose.yml up -d postgres redis
    sleep 3

    log "Applying database schema migrations (Alembic)..."
    docker compose -f deploy/docker-compose.yml run --rm gateway alembic upgrade head

    log "Starting all services (Gateway, vLLM, Observability)..."
    docker compose -f deploy/docker-compose.yml up -d

    log "Deployment completed successfully in Docker mode."

elif [[ "${DEPLOY_MODE}" == "baremetal" ]]; then
    # Bare-metal systemd deployment
    command -v systemctl >/dev/null 2>&1 || error "systemd not available."

    log "Verifying Python virtual environment..."
    if [[ ! -d "venv" ]]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    log "Installing / updating dependencies..."
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -e ".[dev,training]"

    log "Running database migrations..."
    ./venv/bin/alembic upgrade head

    log "Installing systemd unit files..."
    sudo cp deploy/systemd/speedinfer-*.service /etc/systemd/system/
    sudo systemctl daemon-reload

    log "Restarting SpeedInfer services in order..."
    sudo systemctl restart speedinfer-vllm.service
    sudo systemctl restart speedinfer-gateway.service

    log "Deployment completed successfully in bare-metal mode."
else
    error "Unknown deployment mode '${DEPLOY_MODE}'. Supported: docker, baremetal"
fi
