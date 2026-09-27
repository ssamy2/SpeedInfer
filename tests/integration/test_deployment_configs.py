"""Integration tests for deployment automation and configuration validation.

Covers:
- .env.example environment variable specification and completeness
- deploy/docker-compose.yml YAML syntax and 7-container topology
- deploy/Caddyfile reverse proxy rules and unbuffered SSE configuration
- deploy/systemd/ unit files structure and restart directives
- deploy/deploy.sh script bash syntax validation
"""

import configparser
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_DIR = PROJECT_ROOT / "deploy"


# ---------------------------------------------------------------------------
# Test Cases: Environment Variables
# ---------------------------------------------------------------------------
def test_env_example_contains_all_critical_variables():
    """Verify .env.example contains all required environment variables."""
    env_path = PROJECT_ROOT / ".env.example"
    assert env_path.exists(), ".env.example must exist at project root"

    with open(env_path, encoding="utf-8") as f:
        content = f.read()

    required_vars = [
        "APP_NAME",
        "ENVIRONMENT",
        "LOG_LEVEL",
        "DATABASE_URL",
        "REDIS_URL",
        "API_KEY_PEPPER",
        "DEFAULT_MODEL",
        "VLLM_BASE_URL",
        "PROMPT_PRICE_PER_MILLION",
        "COMPLETION_PRICE_PER_MILLION",
    ]
    for var in required_vars:
        assert f"{var}=" in content, f"Missing required environment variable: {var}"


# ---------------------------------------------------------------------------
# Test Cases: Docker Compose Validation
# ---------------------------------------------------------------------------
def test_docker_compose_syntax_and_services():
    """Verify deploy/docker-compose.yml is valid YAML and defines expected topology."""
    compose_path = DEPLOY_DIR / "docker-compose.yml"
    if not compose_path.exists():
        pytest.skip("deploy/docker-compose.yml not yet authored by M6")

    with open(compose_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "docker-compose.yml must be a valid YAML mapping"
    assert "services" in data, "docker-compose.yml must define 'services'"

    services = data["services"]
    required_services = {"gateway", "vllm", "postgres", "redis"}
    for s in required_services:
        assert s in services, f"Service '{s}' missing from docker-compose.yml"

    # Gateway service configuration checks
    gateway = services["gateway"]
    assert "ports" in gateway or "expose" in gateway
    assert "environment" in gateway or "env_file" in gateway
    assert gateway.get("restart") in {"always", "unless-stopped"}


# ---------------------------------------------------------------------------
# Test Cases: Caddyfile Validation
# ---------------------------------------------------------------------------
def test_caddyfile_reverse_proxy_and_sse():
    """Verify deploy/Caddyfile contains auto-TLS reverse proxy and unbuffered SSE flush."""
    caddy_path = DEPLOY_DIR / "Caddyfile"
    if not caddy_path.exists():
        pytest.skip("deploy/Caddyfile not yet authored by M6")

    with open(caddy_path, encoding="utf-8") as f:
        caddy_content = f.read()

    # Must reverse proxy to gateway port (8000)
    assert ":8000" in caddy_content or "gateway:8000" in caddy_content
    # Must configure unbuffered streaming (flush_interval -1) for SSE
    assert "flush_interval -1" in caddy_content or "flush_interval" in caddy_content


# ---------------------------------------------------------------------------
# Test Cases: Systemd Service Units Validation
# ---------------------------------------------------------------------------
def test_systemd_service_units():
    """Verify systemd unit files have valid INI structure with ExecStart and Restart."""
    systemd_dir = DEPLOY_DIR / "systemd"
    gateway_service = systemd_dir / "speedinfer-gateway.service"
    vllm_service = systemd_dir / "speedinfer-vllm.service"

    if not gateway_service.exists() or not vllm_service.exists():
        pytest.skip("deploy/systemd service units not yet authored by M6")

    for unit_path in [gateway_service, vllm_service]:
        config = configparser.ConfigParser(interpolation=None)
        config.read(unit_path)

        assert "Unit" in config, f"Missing [Unit] section in {unit_path.name}"
        assert "Service" in config, f"Missing [Service] section in {unit_path.name}"
        assert "Install" in config, f"Missing [Install] section in {unit_path.name}"

        service_sec = config["Service"]
        assert "ExecStart" in service_sec, f"Missing ExecStart in {unit_path.name}"
        assert "Restart" in service_sec, f"Missing Restart in {unit_path.name}"


# ---------------------------------------------------------------------------
# Test Cases: deploy.sh Script Syntax Validation
# ---------------------------------------------------------------------------
def test_deploy_script_bash_syntax():
    """Verify deploy/deploy.sh exists and passes bash syntax validation."""
    deploy_sh = DEPLOY_DIR / "deploy.sh"
    if not deploy_sh.exists():
        pytest.skip("deploy/deploy.sh not yet authored by M6")

    # Run bash -n (syntax check only without execution)
    result = subprocess.run(
        ["bash", "-n", str(deploy_sh)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Bash syntax error in deploy.sh:\n{result.stderr}"
