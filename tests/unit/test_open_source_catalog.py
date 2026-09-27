"""Unit and integration tests for the 30+ open source models catalog and OpenRouter integration."""

import pytest
from sqlmodel import Session, select

from speedinfer.database.models import ModelVersion, User
from speedinfer.engine.models_catalog import (
    OPEN_SOURCE_MODELS,
    register_catalog_models,
    seed_catalog_models_db,
    update_catalog_from_openrouter,
)
from speedinfer.engine.registry import ModelRegistry
from speedinfer.gateway.routes.models import get_model, list_models


def test_catalog_has_at_least_30_open_source_models():
    """Verify catalog contains at least 30 top open source models with 5% markup."""
    assert len(OPEN_SOURCE_MODELS) >= 30

    required_keywords = ["llama", "qwen", "deepseek", "mistral", "gemma", "phi"]
    for kw in required_keywords:
        assert any(kw in m["id"].lower() for m in OPEN_SOURCE_MODELS), f"Missing models for {kw}"

    for m in OPEN_SOURCE_MODELS:
        assert "id" in m
        assert "context_length" in m and m["context_length"] > 0
        assert "prompt_price_per_million" in m and m["prompt_price_per_million"] >= 0
        assert "completion_price_per_million" in m and m["completion_price_per_million"] >= 0
        # Verify 5% markup calculation
        expected_p = round(m["openrouter_prompt_per_m"] * 1.05, 6)
        expected_c = round(m["openrouter_completion_per_m"] * 1.05, 6)
        assert abs(m["prompt_price_per_million"] - expected_p) < 1e-5
        assert abs(m["completion_price_per_million"] - expected_c) < 1e-5


def test_register_catalog_models():
    """Verify registry correctly registers catalog models and their aliases."""
    registry = ModelRegistry()
    register_catalog_models(registry, "https://openrouter.ai/api/v1")

    # Check ultra-cheap model
    entry = registry.get_model("meta-llama/llama-3.2-1b-instruct")
    assert entry is not None
    assert entry.context_length == 60000
    assert entry.prompt_price_per_million == 0.02835
    assert entry.completion_price_per_million == 0.21105
    assert len(entry.backends) == 1
    assert entry.backends[0].url == "https://openrouter.ai/api/v1"

    # Check DeepSeek V3 and R1
    v3 = registry.get_model("deepseek/deepseek-chat")
    assert v3 is not None
    assert v3.prompt_price_per_million == 0.27027

    r1 = registry.get_model("deepseek/deepseek-r1")
    assert r1 is not None
    assert r1.prompt_price_per_million == 0.7350

    # Check Qwen2.5 7B alias
    alias_entry = registry.get_model("Qwen/Qwen2.5-7B-Instruct")
    assert alias_entry is not None
    assert alias_entry.base_model_path == "qwen/qwen-2.5-7b-instruct"
    assert alias_entry.prompt_price_per_million == 0.1050


@pytest.mark.asyncio
async def test_list_models_includes_catalog(db_session: Session):
    """Verify /v1/models returns all open source models with marked-up prices."""
    registry = ModelRegistry()
    register_catalog_models(registry, "https://openrouter.ai/api/v1")
    seed_catalog_models_db(db_session)

    caller = User(email="dev@example.test", is_active=True)
    resp = await list_models(caller, registry, db_session)

    model_ids = {m.id for m in resp.data}
    assert "meta-llama/llama-3.2-1b-instruct" in model_ids
    assert "deepseek/deepseek-chat" in model_ids
    assert "deepseek/deepseek-r1" in model_ids
    assert "qwen/qwen-2.5-72b-instruct" in model_ids
    assert "google/gemma-3-27b-it" in model_ids
    assert "microsoft/phi-4" in model_ids

    # Check detail endpoint
    llama_detail = await get_model("meta-llama/llama-3.2-1b-instruct", caller, registry, db_session)
    assert llama_detail.id == "meta-llama/llama-3.2-1b-instruct"
    assert llama_detail.prompt_price_per_million == 0.02835


def test_update_catalog_from_openrouter(db_session: Session):
    """Verify dynamic update from OpenRouter API applies exact 5% markup."""
    registry = ModelRegistry()
    register_catalog_models(registry, "https://openrouter.ai/api/v1")
    seed_catalog_models_db(db_session)

    mock_openrouter_data = {
        "meta-llama/llama-3.2-1b-instruct": {
            "id": "meta-llama/llama-3.2-1b-instruct",
            "context_length": 60000,
            "pricing": {
                "prompt": "0.000000030",  # $0.030 / M
                "completion": "0.000000200",  # $0.200 / M
            },
        }
    }

    update_catalog_from_openrouter(registry, db_session, mock_openrouter_data)

    entry = registry.get_model("meta-llama/llama-3.2-1b-instruct")
    assert entry is not None
    assert entry.prompt_price_per_million == round(0.030 * 1.05, 6)
    assert entry.completion_price_per_million == round(0.200 * 1.05, 6)

    db_m = db_session.exec(
        select(ModelVersion).where(ModelVersion.name == "meta-llama/llama-3.2-1b-instruct")
    ).first()
    assert db_m is not None
    assert db_m.prompt_price_per_million == round(0.030 * 1.05, 6)

    # Verify alias also updated in registry and DB
    alias_entry = registry.get_model("llama-3.2-1b")
    assert alias_entry is not None
    assert alias_entry.prompt_price_per_million == round(0.030 * 1.05, 6)

    db_alias = db_session.exec(
        select(ModelVersion).where(ModelVersion.name == "llama-3.2-1b")
    ).first()
    assert db_alias is not None
    assert db_alias.prompt_price_per_million == round(0.030 * 1.05, 6)


def test_benchmark_models_registered_and_resolve():
    """Verify all models featured in website comparison tables resolve properly."""
    registry = ModelRegistry()
    register_catalog_models(registry, "https://openrouter.ai/api/v1")

    benchmark_ids = [
        "deepseek",
        "deepseek-r1",
        "qwen3-235",
        "qwen3-30",
        "qwen",
        "llama4-maverick",
        "llama4-scout",
        "llama",
        "gemma",
        "mistral-small",
        "nemotron-ultra",
        "llama-nemotron",
        "phi4",
        "mistral",
        "nemotron",
    ]

    for bid in benchmark_ids:
        model = registry.get_model(bid)
        assert model is not None, f"Benchmark model {bid} failed to resolve in registry"
        assert model.base_model_path != "", f"Benchmark model {bid} missing base_model_path"
        assert model.prompt_price_per_million > 0
        assert model.completion_price_per_million > 0

