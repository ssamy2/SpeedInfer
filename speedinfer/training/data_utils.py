"""Dataset preprocessing, chat template formatting, and validation utilities.

Supports:
- Multi-format ingestion: OpenAI messages, ShareGPT conversations, Alpaca prompt/response.
- Chat template application for target LLM families (Qwen, Llama-3, Mistral, DeepSeek).
- Train / Validation splitting with reproducible seeding.
- Tokenization sequence length clamping and padding validation.
"""

import json
from pathlib import Path
from typing import Any


def detect_dataset_format(sample: dict[str, Any]) -> str:
    """Detect format of a sample conversation dictionary.

    Args:
        sample: Dictionary representing a single training record.

    Returns:
        str: 'openai', 'sharegpt', 'alpaca', or 'raw'.
    """
    if "messages" in sample and isinstance(sample["messages"], list):
        return "openai"
    if "conversations" in sample and isinstance(sample["conversations"], list):
        return "sharegpt"
    if "instruction" in sample and "output" in sample:
        return "alpaca"
    return "raw"


def normalize_to_chat_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Convert input sample into standardized OpenAI messages structure.

    Args:
        sample: Input data record.

    Returns:
        list[dict[str, str]]: List of {'role': str, 'content': str} dicts.
    """
    fmt = detect_dataset_format(sample)

    if fmt == "openai":
        messages: list[dict[str, str]] = []
        for m in sample["messages"]:
            messages.append(
                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            )
        return messages

    if fmt == "sharegpt":
        messages = []
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        for c in sample["conversations"]:
            sender = c.get("from", "human")
            role = role_map.get(sender, "user")
            messages.append({"role": role, "content": str(c.get("value", ""))})
        return messages

    if fmt == "alpaca":
        instruction = sample.get("instruction", "")
        extra_input = sample.get("input", "")
        output = sample.get("output", "")

        user_content = (
            f"{instruction}\n\n{extra_input}".strip() if extra_input else instruction.strip()
        )
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.strip()},
        ]

    # Fallback for text/prompt fields
    text_content = sample.get("text", sample.get("prompt", ""))
    return [{"role": "user", "content": str(text_content)}]


def format_chat_template(
    messages: list[dict[str, str]],
    model_family: str = "qwen",
    add_generation_prompt: bool = False,
) -> str:
    """Format messages into a single text prompt using standard chat templates.

    Args:
        messages: List of {'role': ..., 'content': ...} dictionaries.
        model_family: Architecture family: 'qwen', 'llama3', 'mistral', 'deepseek'.
        add_generation_prompt: Whether to append assistant turn header.

    Returns:
        str: Formatted raw text string.
    """
    family = model_family.lower()

    if "qwen" in family or "deepseek" in family:
        # ChatML format (<|im_start|>role\ncontent<|im_end|>\n)
        formatted_turns = []
        for m in messages:
            formatted_turns.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        res = "\n".join(formatted_turns) + "\n"
        if add_generation_prompt:
            res += "<|im_start|>assistant\n"
        return res

    if "llama" in family or "llama3" in family:
        # Llama 3 format (<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>)
        formatted_turns = ["<|begin_of_text|>"]
        for m in messages:
            formatted_turns.append(
                f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
            )
        res = "".join(formatted_turns)
        if add_generation_prompt:
            res += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return res

    # Default generic Markdown format
    formatted_turns = []
    for m in messages:
        formatted_turns.append(f"### {m['role'].capitalize()}:\n{m['content']}")
    res = "\n\n".join(formatted_turns)
    if add_generation_prompt:
        res += "\n\n### Assistant:\n"
    return res


def load_and_split_dataset(
    dataset_path: str | Path,
    eval_split_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load dataset from disk and split into train and validation sets.

    Args:
        dataset_path: Path to JSONL or JSON file.
        eval_split_ratio: Fraction reserved for validation.
        seed: Random seed for deterministic splitting.

    Returns:
        tuple[list[dict], list[dict]]: (train_data, val_data).
    """
    path = Path(dataset_path)
    records: list[dict[str, Any]] = []

    if path.is_file():
        if path.suffix in {".jsonl", ".jsonlines"}:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    records = content
                elif isinstance(content, dict) and "data" in content:
                    records = content["data"]
    else:
        # If path does not exist, return empty lists
        return [], []

    if not records:
        return [], []

    import random

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * eval_split_ratio)) if eval_split_ratio > 0 else 0
    val_data = shuffled[:val_count]
    train_data = shuffled[val_count:]

    return train_data, val_data
