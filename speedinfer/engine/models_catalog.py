"""Open-source models catalog with OpenRouter integration and +5% marked up pricing.

Contains top open-source models featured in SpeedInfer benchmarks, comparison tables,
and general high-demand open weights catalog, with exact pricing calculated as
OpenRouter upstream price + 5% markup.
"""

from typing import Any

from sqlmodel import Session, select

from speedinfer.database.models import LifecycleStatus, ModelVersion
from speedinfer.engine.registry import BackendWorker, ModelEntry, ModelPricing, ModelRegistry

# Default verified pricing snapshot from OpenRouter + 5% markup:
# prompt_price_per_million = round(openrouter_prompt_per_token * 1_000_000 * 1.05, 6)
# completion_price_per_million = round(openrouter_compl_per_token * 1_000_000 * 1.05, 6)

OPEN_SOURCE_MODELS: list[dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # 1. Ultra-Low-Cost Models (Recommended for cheap testing and verification)
    # -------------------------------------------------------------------------
    {
        "id": "meta-llama/llama-3.2-1b-instruct",
        "name": "Meta: Llama 3.2 1B Instruct",
        "context_length": 60000,
        "openrouter_prompt_per_m": 0.0270,
        "openrouter_completion_per_m": 0.2010,
        "prompt_price_per_million": 0.02835,
        "completion_price_per_million": 0.21105,
        "aliases": ["llama-3.2-1b", "meta-llama/llama-3.2-1b"],
    },
    {
        "id": "meta-llama/llama-3.2-3b-instruct",
        "name": "Meta: Llama 3.2 3B Instruct",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.0500,
        "openrouter_completion_per_m": 0.3300,
        "prompt_price_per_million": 0.0525,
        "completion_price_per_million": 0.3465,
        "aliases": ["llama-3.2-3b"],
    },
    {
        "id": "mistralai/mistral-nemo",
        "name": "Mistral: Mistral NeMo",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.0190,
        "openrouter_completion_per_m": 0.0300,
        "prompt_price_per_million": 0.01995,
        "completion_price_per_million": 0.0315,
        "aliases": ["mistral", "mistral-nemo"],
    },
    {
        "id": "google/gemma-3-4b-it",
        "name": "Google: Gemma 3 4B",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.0500,
        "openrouter_completion_per_m": 0.1000,
        "prompt_price_per_million": 0.0525,
        "completion_price_per_million": 0.1050,
        "aliases": ["gemma-3-4b"],
    },
    {
        "id": "google/gemma-3-12b-it",
        "name": "Google: Gemma 3 12B",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.0500,
        "openrouter_completion_per_m": 0.1500,
        "prompt_price_per_million": 0.0525,
        "completion_price_per_million": 0.1575,
        "aliases": ["gemma-3-12b"],
    },
    {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "name": "Meta: Llama 3.1 8B Instruct",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.0500,
        "openrouter_completion_per_m": 0.0800,
        "prompt_price_per_million": 0.0525,
        "completion_price_per_million": 0.0840,
        "aliases": ["llama-3.1-8b"],
    },
    # -------------------------------------------------------------------------
    # 2. DeepSeek Family (Frontier Open Reasoning & General Intelligence)
    # -------------------------------------------------------------------------
    {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek: DeepSeek V3",
        "context_length": 163840,
        "openrouter_prompt_per_m": 0.2574,
        "openrouter_completion_per_m": 1.0287,
        "prompt_price_per_million": 0.27027,
        "completion_price_per_million": 1.080135,
        "aliases": ["deepseek", "deepseek-ai/DeepSeek-V3", "deepseek-v3", "deepseek-chat"],
    },
    {
        "id": "deepseek/deepseek-r1",
        "name": "DeepSeek: DeepSeek R1",
        "context_length": 64000,
        "openrouter_prompt_per_m": 0.7000,
        "openrouter_completion_per_m": 2.5000,
        "prompt_price_per_million": 0.7350,
        "completion_price_per_million": 2.6250,
        "aliases": ["deepseek-ai/DeepSeek-R1", "deepseek-r1"],
    },
    {
        "id": "deepseek/deepseek-r1-distill-llama-70b",
        "name": "DeepSeek: R1 Distill Llama 70B",
        "context_length": 8192,
        "openrouter_prompt_per_m": 0.8000,
        "openrouter_completion_per_m": 0.8000,
        "prompt_price_per_million": 0.8400,
        "completion_price_per_million": 0.8400,
        "aliases": ["deepseek-r1-distill-llama-70b"],
    },
    {
        "id": "deepseek/deepseek-v3.2",
        "name": "DeepSeek: DeepSeek V3.2",
        "context_length": 163840,
        "openrouter_prompt_per_m": 0.2690,
        "openrouter_completion_per_m": 0.4000,
        "prompt_price_per_million": 0.28245,
        "completion_price_per_million": 0.4200,
        "aliases": ["deepseek-v3.2"],
    },
    # -------------------------------------------------------------------------
    # 3. Qwen Family (Multilingual, Dense & MoE)
    # -------------------------------------------------------------------------
    {
        "id": "qwen/qwen-2.5-7b-instruct",
        "name": "Qwen: Qwen2.5 7B Instruct",
        "context_length": 32768,
        "openrouter_prompt_per_m": 0.1000,
        "openrouter_completion_per_m": 0.2000,
        "prompt_price_per_million": 0.1050,
        "completion_price_per_million": 0.2100,
        "aliases": [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-7B",
            "qwen-2.5-7b-instruct",
            "qwen-2.5-7b",
        ],
    },
    {
        "id": "qwen/qwen-2.5-72b-instruct",
        "name": "Qwen: Qwen2.5 72B Instruct",
        "context_length": 32768,
        "openrouter_prompt_per_m": 0.3600,
        "openrouter_completion_per_m": 0.4000,
        "prompt_price_per_million": 0.3780,
        "completion_price_per_million": 0.4200,
        "aliases": [
            "qwen",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-72B",
            "qwen-2.5-72b-instruct",
            "qwen-2.5-72b",
        ],
    },
    {
        "id": "qwen/qwen-2.5-coder-32b-instruct",
        "name": "Qwen: Qwen2.5 Coder 32B Instruct",
        "context_length": 32768,
        "openrouter_prompt_per_m": 0.6600,
        "openrouter_completion_per_m": 1.0000,
        "prompt_price_per_million": 0.6930,
        "completion_price_per_million": 1.0500,
        "aliases": ["qwen-2.5-coder-32b-instruct", "qwen-2.5-coder-32b"],
    },
    {
        "id": "qwen/qwen3-8b",
        "name": "Qwen: Qwen3 8B",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.1170,
        "openrouter_completion_per_m": 0.4550,
        "prompt_price_per_million": 0.12285,
        "completion_price_per_million": 0.47775,
        "aliases": ["qwen3-8b"],
    },
    {
        "id": "qwen/qwen3-14b",
        "name": "Qwen: Qwen3 14B",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.1200,
        "openrouter_completion_per_m": 0.2400,
        "prompt_price_per_million": 0.1260,
        "completion_price_per_million": 0.2520,
        "aliases": ["qwen3-14b"],
    },
    {
        "id": "qwen/qwen3-32b",
        "name": "Qwen: Qwen3 32B",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.0800,
        "openrouter_completion_per_m": 0.2800,
        "prompt_price_per_million": 0.0840,
        "completion_price_per_million": 0.2940,
        "aliases": ["qwen3-32b"],
    },
    {
        "id": "qwen/qwen3-30b-a3b",
        "name": "Qwen: Qwen3 30B A3B",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.1200,
        "openrouter_completion_per_m": 0.5000,
        "prompt_price_per_million": 0.1260,
        "completion_price_per_million": 0.5250,
        "aliases": ["qwen3-30", "qwen3-30b", "qwen3-30b-a3b"],
    },
    {
        "id": "qwen/qwen3-235b-a22b",
        "name": "Qwen: Qwen3 235B A22B",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.4550,
        "openrouter_completion_per_m": 1.8200,
        "prompt_price_per_million": 0.47775,
        "completion_price_per_million": 1.9110,
        "aliases": ["qwen3-235", "qwen3-235b", "qwen3-235b-a22b"],
    },
    {
        "id": "qwen/qwen3-coder-30b-a3b-instruct",
        "name": "Qwen: Qwen3 Coder 30B A3B Instruct",
        "context_length": 262144,
        "openrouter_prompt_per_m": 0.0700,
        "openrouter_completion_per_m": 0.2800,
        "prompt_price_per_million": 0.0735,
        "completion_price_per_million": 0.2940,
        "aliases": ["qwen3-coder-30b"],
    },
    # -------------------------------------------------------------------------
    # 4. Meta Llama Family
    # -------------------------------------------------------------------------
    {
        "id": "meta-llama/llama-3.1-70b-instruct",
        "name": "Meta: Llama 3.1 70B Instruct",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.4000,
        "openrouter_completion_per_m": 0.4000,
        "prompt_price_per_million": 0.4200,
        "completion_price_per_million": 0.4200,
        "aliases": ["llama-3.1-70b"],
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Meta: Llama 3.3 70B Instruct",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.1000,
        "openrouter_completion_per_m": 0.3200,
        "prompt_price_per_million": 0.1050,
        "completion_price_per_million": 0.3360,
        "aliases": ["llama-3.3-70b"],
    },
    {
        "id": "meta-llama/llama-4-scout",
        "name": "Meta: Llama 4 Scout",
        "context_length": 1310720,
        "openrouter_prompt_per_m": 0.1000,
        "openrouter_completion_per_m": 0.3000,
        "prompt_price_per_million": 0.1050,
        "completion_price_per_million": 0.3150,
        "aliases": ["llama4-scout", "llama-4-scout"],
    },
    {
        "id": "meta-llama/llama-4-maverick",
        "name": "Meta: Llama 4 Maverick",
        "context_length": 1048576,
        "openrouter_prompt_per_m": 0.1875,
        "openrouter_completion_per_m": 0.6525,
        "prompt_price_per_million": 0.196875,
        "completion_price_per_million": 0.685125,
        "aliases": ["llama4-maverick", "llama-4-maverick"],
    },
    # -------------------------------------------------------------------------
    # 5. Mistral AI Family
    # -------------------------------------------------------------------------
    {
        "id": "mistralai/mistral-small-24b-instruct-2501",
        "name": "Mistral: Mistral Small 3 24B",
        "context_length": 32768,
        "openrouter_prompt_per_m": 0.0500,
        "openrouter_completion_per_m": 0.0800,
        "prompt_price_per_million": 0.0525,
        "completion_price_per_million": 0.0840,
        "aliases": ["mistral-small-3", "mistral-small-24b"],
    },
    {
        "id": "mistralai/mistral-small-3.1-24b-instruct",
        "name": "Mistral: Mistral Small 3.1 24B",
        "context_length": 128000,
        "openrouter_prompt_per_m": 0.3510,
        "openrouter_completion_per_m": 0.5550,
        "prompt_price_per_million": 0.36855,
        "completion_price_per_million": 0.58275,
        "aliases": ["mistral-small", "mistral-small-3.1"],
    },
    {
        "id": "mistralai/mistral-large",
        "name": "Mistral: Mistral Large",
        "context_length": 128000,
        "openrouter_prompt_per_m": 2.0000,
        "openrouter_completion_per_m": 6.0000,
        "prompt_price_per_million": 2.1000,
        "completion_price_per_million": 6.3000,
        "aliases": ["mistral-large"],
    },
    {
        "id": "mistralai/codestral-2508",
        "name": "Mistral: Codestral 2508",
        "context_length": 256000,
        "openrouter_prompt_per_m": 0.3000,
        "openrouter_completion_per_m": 0.9000,
        "prompt_price_per_million": 0.3150,
        "completion_price_per_million": 0.9450,
        "aliases": ["codestral", "codestral-2508"],
    },
    # -------------------------------------------------------------------------
    # 6. Google Gemma & Microsoft Phi
    # -------------------------------------------------------------------------
    {
        "id": "google/gemma-2-27b-it",
        "name": "Google: Gemma 2 27B",
        "context_length": 8192,
        "openrouter_prompt_per_m": 0.6500,
        "openrouter_completion_per_m": 0.6500,
        "prompt_price_per_million": 0.6825,
        "completion_price_per_million": 0.6825,
        "aliases": ["gemma-2-27b"],
    },
    {
        "id": "google/gemma-3-27b-it",
        "name": "Google: Gemma 3 27B",
        "context_length": 131072,
        "openrouter_prompt_per_m": 0.0800,
        "openrouter_completion_per_m": 0.4500,
        "prompt_price_per_million": 0.0840,
        "completion_price_per_million": 0.4725,
        "aliases": ["gemma", "gemma-3-27b"],
    },
    {
        "id": "microsoft/phi-4",
        "name": "Microsoft: Phi-4",
        "context_length": 16384,
        "openrouter_prompt_per_m": 0.0700,
        "openrouter_completion_per_m": 0.1400,
        "prompt_price_per_million": 0.0735,
        "completion_price_per_million": 0.1470,
        "aliases": ["phi-4", "phi4"],
    },
    # -------------------------------------------------------------------------
    # 7. NVIDIA Nemotron Family
    # -------------------------------------------------------------------------
    {
        "id": "nvidia/nemotron-3-super-120b-a12b",
        "name": "NVIDIA: Nemotron 3 Super 120B",
        "context_length": 262144,
        "openrouter_prompt_per_m": 0.0800,
        "openrouter_completion_per_m": 0.4500,
        "prompt_price_per_million": 0.0840,
        "completion_price_per_million": 0.4725,
        "aliases": ["nemotron", "nemotron-3-super", "llama-nemotron"],
    },
    {
        "id": "nvidia/nemotron-3-ultra-550b-a55b",
        "name": "NVIDIA: Nemotron 3 Ultra 550B",
        "context_length": 262144,
        "openrouter_prompt_per_m": 0.6000,
        "openrouter_completion_per_m": 2.4000,
        "prompt_price_per_million": 0.6300,
        "completion_price_per_million": 2.5200,
        "aliases": ["nemotron-3-ultra", "nemotron-ultra"],
    },
    # -------------------------------------------------------------------------
    # 8. Frontier 400B+ Open Weight Models (Featured in Benchmarks Cohort)
    # -------------------------------------------------------------------------
    {
        "id": "nousresearch/hermes-3-llama-3.1-405b",
        "name": "Nous: Hermes 3 405B Instruct (Llama 3.1 405B)",
        "context_length": 131072,
        "openrouter_prompt_per_m": 1.0000,
        "openrouter_completion_per_m": 1.0000,
        "prompt_price_per_million": 1.0500,
        "completion_price_per_million": 1.0500,
        "aliases": [
            "llama",
            "llama-3.1-405b",
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-3.1-405b",
            "llama-405b",
        ],
    },
]


def register_catalog_models(registry: ModelRegistry, backend_url: str) -> None:
    """Register all 32 open source catalog models into the memory ModelRegistry."""
    default_worker = BackendWorker(url=backend_url, worker_id="openrouter-upstream")

    for item in OPEN_SOURCE_MODELS:
        model_id = item["id"]
        ctx = item["context_length"]
        p_price = item["prompt_price_per_million"]
        c_price = item["completion_price_per_million"]

        pricing = ModelPricing(
            prompt_price_per_million=p_price,
            completion_price_per_million=c_price,
        )

        # 1. Register canonical model ID
        registry.register_model(
            name=model_id,
            base_model_path=model_id,
            context_length=ctx,
            prompt_price_per_million=p_price,
            completion_price_per_million=c_price,
            pricing=pricing,
            backends=[default_worker],
        )

        # 2. Register any aliases pointing to the canonical model
        for alias in item.get("aliases", []):
            registry.register_model(
                name=alias,
                base_model_path=model_id,
                fallback_model=model_id,
                context_length=ctx,
                prompt_price_per_million=p_price,
                completion_price_per_million=c_price,
                pricing=pricing,
                backends=[default_worker],
            )


def seed_catalog_models_db(session: Session) -> None:
    """Idempotently seed or update the catalog models in the persistent database."""
    for item in OPEN_SOURCE_MODELS:
        model_id = item["id"]
        ctx = item["context_length"]
        p_price = item["prompt_price_per_million"]
        c_price = item["completion_price_per_million"]

        db_model = session.exec(
            select(ModelVersion).where(ModelVersion.name == model_id)
        ).first()

        if db_model is None:
            db_model = ModelVersion(
                name=model_id,
                base_model_path=model_id,
                context_length=ctx,
                prompt_price_per_million=p_price,
                completion_price_per_million=c_price,
                lifecycle_status=LifecycleStatus.ACTIVE.value,
            )
            session.add(db_model)
        else:
            db_model.context_length = ctx
            db_model.prompt_price_per_million = p_price
            db_model.completion_price_per_million = c_price
            db_model.base_model_path = model_id
            db_model.lifecycle_status = LifecycleStatus.ACTIVE.value
            session.add(db_model)

        # Also ensure primary alias exists in DB for backwards compatibility
        for alias in item.get("aliases", []):
            db_alias = session.exec(
                select(ModelVersion).where(ModelVersion.name == alias)
            ).first()
            if db_alias is None:
                db_alias = ModelVersion(
                    name=alias,
                    base_model_path=model_id,
                    context_length=ctx,
                    prompt_price_per_million=p_price,
                    completion_price_per_million=c_price,
                    lifecycle_status=LifecycleStatus.ACTIVE.value,
                )
                session.add(db_alias)
            else:
                db_alias.context_length = ctx
                db_alias.prompt_price_per_million = p_price
                db_alias.completion_price_per_million = c_price
                db_alias.base_model_path = model_id
                session.add(db_alias)

    session.commit()


def update_catalog_from_openrouter(
    registry: ModelRegistry,
    session: Session,
    openrouter_models_map: dict[str, Any],
) -> None:
    """Update catalog pricing dynamically using live OpenRouter API models feed with +5% markup."""
    for item in OPEN_SOURCE_MODELS:
        model_id = item["id"]
        if model_id not in openrouter_models_map:
            continue

        raw = openrouter_models_map[model_id]
        pricing_data = raw.get("pricing", {})
        try:
            raw_prompt = float(pricing_data.get("prompt", 0)) * 1_000_000
            raw_compl = float(pricing_data.get("completion", 0)) * 1_000_000
        except (ValueError, TypeError):
            continue

        # Exact 5% markup
        marked_prompt = round(raw_prompt * 1.05, 6)
        marked_compl = round(raw_compl * 1.05, 6)
        ctx = int(raw.get("context_length", item["context_length"]))

        item["openrouter_prompt_per_m"] = raw_prompt
        item["openrouter_completion_per_m"] = raw_compl
        item["prompt_price_per_million"] = marked_prompt
        item["completion_price_per_million"] = marked_compl
        item["context_length"] = ctx

        # Update in-memory registry
        entry: ModelEntry | None = registry.get_model(model_id)
        if entry:
            entry.context_length = ctx
            entry.prompt_price_per_million = marked_prompt
            entry.completion_price_per_million = marked_compl
            if entry.pricing:
                entry.pricing.prompt_price_per_million = marked_prompt
                entry.pricing.completion_price_per_million = marked_compl

        # Update database entry
        db_m = session.exec(select(ModelVersion).where(ModelVersion.name == model_id)).first()
        if db_m:
            db_m.context_length = ctx
            db_m.prompt_price_per_million = marked_prompt
            db_m.completion_price_per_million = marked_compl
            session.add(db_m)

        # Update all registered aliases with latest pricing
        for alias in item.get("aliases", []):
            alias_entry: ModelEntry | None = registry.get_model(alias)
            if alias_entry:
                alias_entry.context_length = ctx
                alias_entry.prompt_price_per_million = marked_prompt
                alias_entry.completion_price_per_million = marked_compl
                if alias_entry.pricing:
                    alias_entry.pricing.prompt_price_per_million = marked_prompt
                    alias_entry.pricing.completion_price_per_million = marked_compl
            db_alias = session.exec(select(ModelVersion).where(ModelVersion.name == alias)).first()
            if db_alias:
                db_alias.context_length = ctx
                db_alias.prompt_price_per_million = marked_prompt
                db_alias.completion_price_per_million = marked_compl
                session.add(db_alias)

    session.commit()
