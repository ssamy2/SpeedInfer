"""SpeedInfer Model Lifecycle and Training Package.

Provides LoRA/QLoRA fine-tuning, dataset preprocessing, MLflow tracking,
model merge and safetensors export, and automated evaluation gating.
"""

from speedinfer.training.config import FineTuningConfig, LoRAHyperparameters, QLoRAConfig
from speedinfer.training.data_utils import (
    format_chat_template,
    load_and_split_dataset,
    normalize_to_chat_messages,
)
from speedinfer.training.evaluate import evaluate_model
from speedinfer.training.merge_and_export import merge_and_export
from speedinfer.training.train import run_training

__all__ = [
    "FineTuningConfig",
    "LoRAHyperparameters",
    "QLoRAConfig",
    "evaluate_model",
    "format_chat_template",
    "load_and_split_dataset",
    "merge_and_export",
    "normalize_to_chat_messages",
    "run_training",
]
