"""Training and fine-tuning configuration schemas with YAML serialization.

Provides:
- LoRAConfig: Low-Rank Adaptation hyperparameter specification.
- QuantizationConfig: 4-bit / 8-bit QLoRA configuration.
- TrainingArgumentsConfig: Optimization, batching, and learning rate schedule.
- MLflowConfig: Experiment tracking and model registry settings.
- FineTuningConfig: Complete top-level YAML-serializable configuration.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LoRAHyperparameters(BaseModel):
    """PEFT LoRA hyperparameters."""

    model_config = ConfigDict(extra="ignore")

    r: int = Field(default=16, ge=1, le=256, description="LoRA rank dimension.")
    lora_alpha: int = Field(default=32, ge=1, le=512, description="LoRA scaling factor.")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="Dropout probability.")
    bias: Literal["none", "all", "lora_only"] = Field(default="none")
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target attention and MLP projection layers for LoRA.",
    )
    task_type: str = Field(default="CAUSAL_LM")


class QLoRAConfig(BaseModel):
    """4-bit / 8-bit quantization configuration for QLoRA."""

    model_config = ConfigDict(extra="ignore")

    load_in_4bit: bool = Field(default=True, description="Enable 4-bit NormalFloat4 quantization.")
    bnb_4bit_quant_type: str = Field(default="nf4", description="Quantization type (nf4 or fp4).")
    bnb_4bit_use_double_quant: bool = Field(
        default=True, description="Enable nested double quantization."
    )
    bnb_4bit_compute_dtype: str = Field(
        default="bfloat16", description="Compute precision (bfloat16 or float16)."
    )


class TrainingHyperparameters(BaseModel):
    """Training run optimization parameters."""

    model_config = ConfigDict(extra="ignore")

    learning_rate: float = Field(default=2e-4, gt=0, description="Peak learning rate.")
    num_train_epochs: float = Field(default=3.0, gt=0, description="Total training epochs.")
    per_device_train_batch_size: int = Field(
        default=4, ge=1, description="Per-GPU micro-batch size."
    )
    per_device_eval_batch_size: int = Field(
        default=4, ge=1, description="Per-GPU evaluation batch size."
    )
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, description="Gradient accumulation steps."
    )
    warmup_ratio: float = Field(
        default=0.05, ge=0.0, le=0.5, description="Warmup fraction of total steps."
    )
    weight_decay: float = Field(default=0.01, ge=0.0, description="Decoupled weight decay.")
    lr_scheduler_type: str = Field(
        default="cosine", description="Learning rate scheduler schedule."
    )
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=100, ge=1)
    max_seq_length: int = Field(default=2048, ge=128, le=32768)
    bf16: bool = Field(default=True, description="Use Brain Floating Point (Ampere/Hopper).")
    fp16: bool = Field(default=False)


class MLflowTrackingConfig(BaseModel):
    """MLflow experiment and artifact tracking configuration."""

    model_config = ConfigDict(extra="ignore")

    experiment_name: str = Field(
        default="speedinfer-finetuning", description="MLflow experiment name."
    )
    tracking_uri: str = Field(
        default="http://localhost:5000", description="Remote MLflow tracking URI."
    )
    registered_model_name: str | None = Field(
        default=None, description="Target name in MLflow model registry."
    )
    tags: dict[str, str] = Field(default_factory=dict)


class FineTuningConfig(BaseModel):
    """Top-level fine-tuning pipeline configuration."""

    model_config = ConfigDict(extra="ignore")

    model_name_or_path: str = Field(description="Hugging Face repo or local path to base model.")
    dataset_path: str = Field(
        description="Path to training dataset file or Hugging Face dataset ID."
    )
    output_dir: str = Field(
        default="./output/adapter", description="Directory to persist checkpoints and adapter."
    )
    eval_split_ratio: float = Field(
        default=0.10, ge=0.0, le=0.5, description="Holdout evaluation fraction."
    )
    lora: LoRAHyperparameters = Field(default_factory=LoRAHyperparameters)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    training: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    mlflow: MLflowTrackingConfig = Field(default_factory=MLflowTrackingConfig)

    @classmethod
    def from_yaml(cls, path_or_str: str | Path) -> "FineTuningConfig":
        """Load configuration from a YAML file or raw YAML string."""
        path = Path(path_or_str)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            content = str(path_or_str)
        data = yaml.safe_load(content)
        return cls.model_validate(data)

    def to_yaml(self, destination_path: str | Path | None = None) -> str:
        """Serialize configuration to a YAML string or write to file."""
        data = self.model_dump()
        dumped = yaml.dump(data, sort_keys=False, indent=2)
        if destination_path is not None:
            Path(destination_path).write_text(dumped, encoding="utf-8")
        return dumped
