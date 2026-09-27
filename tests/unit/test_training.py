"""Unit tests for SpeedInfer Model Lifecycle and Training module.

Covers:
- FineTuningConfig YAML loading and validation
- Dataset format detection (OpenAI, ShareGPT, Alpaca)
- Chat template normalization across model families
- Train/Validation deterministic dataset splitting
- Fine-tuning execution pipeline in dry-run mode
- Merge and export safetensors manifest creation and database registration
- Evaluation harness perplexity computation and production gating
"""

import json
from pathlib import Path

from speedinfer.training.config import FineTuningConfig, LoRAHyperparameters
from speedinfer.training.data_utils import (
    detect_dataset_format,
    format_chat_template,
    load_and_split_dataset,
    normalize_to_chat_messages,
)
from speedinfer.training.evaluate import evaluate_model
from speedinfer.training.merge_and_export import merge_and_export
from speedinfer.training.train import run_training


def test_fine_tuning_config_yaml(tmp_path: Path):
    """Verify FineTuningConfig can serialize to and load from YAML."""
    config = FineTuningConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        dataset_path="./data/test.jsonl",
        output_dir=str(tmp_path / "adapter"),
        lora=LoRAHyperparameters(r=32, lora_alpha=64),
    )
    yaml_path = tmp_path / "config.yaml"
    config.to_yaml(yaml_path)

    loaded = FineTuningConfig.from_yaml(yaml_path)
    assert loaded.model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
    assert loaded.lora.r == 32
    assert loaded.lora.lora_alpha == 64


def test_dataset_format_detection_and_normalization():
    """Verify detection and normalization of OpenAI, ShareGPT, and Alpaca datasets."""
    # OpenAI format
    openai_sample = {
        "messages": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]
    }
    assert detect_dataset_format(openai_sample) == "openai"
    norm_openai = normalize_to_chat_messages(openai_sample)
    assert len(norm_openai) == 2
    assert norm_openai[0]["role"] == "user"

    # ShareGPT format
    sharegpt_sample = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
        ]
    }
    assert detect_dataset_format(sharegpt_sample) == "sharegpt"
    norm_sharegpt = normalize_to_chat_messages(sharegpt_sample)
    assert len(norm_sharegpt) == 2
    assert norm_sharegpt[0]["role"] == "user"
    assert norm_sharegpt[1]["role"] == "assistant"

    # Alpaca format
    alpaca_sample = {
        "instruction": "Summarize this text",
        "input": "Fast inference is crucial.",
        "output": "Speed matters.",
    }
    assert detect_dataset_format(alpaca_sample) == "alpaca"
    norm_alpaca = normalize_to_chat_messages(alpaca_sample)
    assert len(norm_alpaca) == 2
    assert "Fast inference is crucial." in norm_alpaca[0]["content"]
    assert norm_alpaca[1]["content"] == "Speed matters."


def test_chat_templates_formatting():
    """Verify chat template generation for Qwen/ChatML and Llama 3."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    qwen_text = format_chat_template(messages, model_family="qwen", add_generation_prompt=True)
    assert "<|im_start|>system" in qwen_text
    assert "<|im_start|>user\nHello!<|im_end|>" in qwen_text
    assert "<|im_start|>assistant\n" in qwen_text

    llama_text = format_chat_template(messages, model_family="llama3", add_generation_prompt=True)
    assert "<|begin_of_text|>" in llama_text
    assert "<|start_header_id|>user<|end_header_id|>" in llama_text
    assert "<|start_header_id|>assistant<|end_header_id|>" in llama_text


def test_dataset_loading_and_splitting(tmp_path: Path):
    """Verify dataset loading and deterministic split ratio."""
    data_file = tmp_path / "dataset.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"msg {i}"}]}) + "\n")

    train_data, val_data = load_and_split_dataset(data_file, eval_split_ratio=0.20, seed=42)
    assert len(train_data) == 8
    assert len(val_data) == 2


def test_training_pipeline_dry_run(tmp_path: Path):
    """Verify end-to-end dry-run execution of training pipeline."""
    data_file = tmp_path / "dataset.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")

    config = FineTuningConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        dataset_path=str(data_file),
        output_dir=str(tmp_path / "output_adapter"),
    )
    result = run_training(config, dry_run=True)
    assert result["status"] == "success"
    assert "metrics" in result
    assert result["metrics"]["train_loss"] > 0
    assert (tmp_path / "output_adapter" / "adapter_config.json").exists()


def test_merge_and_export_utility(tmp_path: Path):
    """Verify merging and export generates vLLM-compatible manifest."""
    adapter_dir = tmp_path / "test_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 16}))

    export_dir = tmp_path / "exported_model"
    res = merge_and_export(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=adapter_dir,
        output_dir=export_dir,
        quantization="fp8",
    )
    assert res["status"] == "success"
    assert res["vllm_compatible"] is True
    assert (export_dir / "speedinfer_model_manifest.json").exists()
    assert (export_dir / "model.safetensors").exists()


def test_evaluate_model_and_promotion(tmp_path: Path):
    """Verify evaluation harness and production status gating."""
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(json.dumps({"messages": [{"role": "user", "content": "test"}]}) + "\n")

    # When perplexity is below threshold, pass = True
    report = evaluate_model(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        eval_dataset=eval_file,
        max_perplexity=10.0,
        promote_on_pass=False,
    )
    assert report["passed"] is True
    assert report["perplexity"] < 10.0

    # When threshold is set artificially low, pass = False
    report_fail = evaluate_model(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        eval_dataset=eval_file,
        max_perplexity=1.0,
        promote_on_pass=False,
    )
    assert report_fail["passed"] is False

    # Test promotion to active serving status
    import uuid

    from speedinfer.database.models import ModelVersion
    from speedinfer.database.session import get_session_context

    unique_model_name = f"TestModel-EvalPromotion-{uuid.uuid4().hex[:8]}"

    with get_session_context() as session:
        test_m = ModelVersion(
            name=unique_model_name,
            base_model_path="Qwen/Qwen2.5-7B-Instruct",
            lifecycle_status="staging",
        )
        session.add(test_m)

    report_promoted = evaluate_model(
        model_name=unique_model_name,
        eval_dataset=eval_file,
        max_perplexity=15.0,
        promote_on_pass=True,
    )
    assert report_promoted["passed"] is True
    assert report_promoted["promoted_to_production"] is True

    with get_session_context() as session:
        from sqlmodel import select

        updated = session.exec(
            select(ModelVersion).where(ModelVersion.name == unique_model_name)
        ).first()
        assert updated is not None
        assert updated.lifecycle_status == "active"


def test_dataset_validation_and_token_counting():
    """Verify JSONL dataset validation and accurate token counting."""
    from speedinfer.training.data_utils import validate_dataset_jsonl_bytes

    # 1. Valid OpenAI format
    valid_openai = (
        b'{"messages": [{"role": "user", "content": "Hello world"}, '
        b'{"role": "assistant", "content": "Hi there!"}]}\n'
    )
    tokens, is_valid, err = validate_dataset_jsonl_bytes(valid_openai)
    assert is_valid is True
    assert err is None
    assert tokens > 0

    # 2. Valid Prompt/Completion format
    valid_pc = b'{"prompt": "Translate hello", "completion": "Bonjour"}\n'
    tokens, is_valid, err = validate_dataset_jsonl_bytes(valid_pc)
    assert is_valid is True
    assert err is None
    assert tokens > 0

    # 3. Invalid: Malformed JSON
    tokens, is_valid, err = validate_dataset_jsonl_bytes(b'{"messages": broken json\n')
    assert is_valid is False
    assert "not valid JSON" in err

    # 4. Invalid: Missing expected keys
    tokens, is_valid, err = validate_dataset_jsonl_bytes(b'{"random_key": 123}\n')
    assert is_valid is False
    assert "Line 1 is invalid" in err

    # 5. Empty content
    tokens, is_valid, err = validate_dataset_jsonl_bytes(b"")
    assert is_valid is False
    assert "empty" in err


def test_training_pricing_formula():
    """Verify training price = rate_per_1M * (tokens / 1,000,000) * epochs."""
    from speedinfer.training.data_utils import get_model_training_rate_per_million

    # Tier 1: Standard (<=16B) -> $0.75 / 1M
    rate_7b, tier_7b = get_model_training_rate_per_million("Qwen/Qwen2.5-7B-Instruct")
    assert rate_7b == 0.75
    assert "Standard" in tier_7b
    cost_7b = round(rate_7b * (1_000_000 / 1_000_000) * 3, 4)
    assert cost_7b == 2.25

    # Tier 2: Medium (16B-35B) -> $2.50 / 1M
    rate_32b, tier_32b = get_model_training_rate_per_million("Qwen/Qwen2.5-32B-Instruct")
    assert rate_32b == 2.50
    assert "Medium" in tier_32b
    cost_32b = round(rate_32b * (500_000 / 1_000_000) * 2, 4)
    assert cost_32b == 2.50

    # Tier 3: Large (36B-100B) -> $5.00 / 1M
    rate_70b, tier_70b = get_model_training_rate_per_million("meta-llama/Llama-3.3-70B-Instruct")
    assert rate_70b == 5.00
    assert "Large" in tier_70b
    cost_70b = round(rate_70b * (200_000 / 1_000_000) * 1, 4)
    assert cost_70b == 1.00

    # Tier 4: Ultra (>100B) -> $10.00 / 1M
    rate_405b, tier_405b = get_model_training_rate_per_million(
        "meta-llama/Meta-Llama-3.1-405B-Instruct"
    )
    assert rate_405b == 10.00
    assert "Ultra" in tier_405b
    cost_405b = round(rate_405b * (100_000 / 1_000_000) * 5, 4)
    assert cost_405b == 5.00

    # Default fallback
    rate_default, _ = get_model_training_rate_per_million(None)
    assert rate_default == 0.75
