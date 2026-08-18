"""Supervised fine-tuning on LongAlpaca-16k: 5 epochs, 8K-16K context.

Following FreqKV's training protocol for direct comparability.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model

from ..spectral.attention import CompressionConfig, apply_compression_to_model
from .lora_config import create_lora_config

logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME = "Yukang/LongAlpaca-16k-length"
CONTEXT_LENGTH = 16384
NUM_EPOCHS = 5


def load_longalpaca_dataset(
    context_length: int = CONTEXT_LENGTH,
    hf_token: str | None = None,
) -> "datasets.Dataset":
    """Load and tokenize LongAlpaca-16k dataset.

    Args:
        context_length: Maximum context length (8K-16K).
        hf_token: HuggingFace token.

    Returns:
        Tokenized dataset with instruction-response pairs.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"Loaded LongAlpaca-16k: {len(dataset)} samples")

    def format_and_tokenize(example):
        # LongAlpaca has 'instruction' and 'response' fields
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        input_text = example.get("input", "")

        # Format as instruction-response
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        full_text = prompt + response
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=context_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Mask the prompt tokens for loss computation (only learn from response)
        prompt_tokens = tokenizer(
            prompt, truncation=True, max_length=context_length, return_tensors="pt"
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        labels = tokenized["input_ids"].clone()
        labels[:, :prompt_len] = -100  # Mask prompt tokens

        tokenized["labels"] = labels
        return {k: v.squeeze(0) for k, v in tokenized.items()}

    dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    logger.info(f"Tokenized LongAlpaca: {len(dataset)} samples, context={context_length}")
    return dataset


def train_on_longalpaca(
    compression_config: CompressionConfig,
    output_dir: str | Path = "results/checkpoints/longalpaca",
    num_gpus: int = 8,
    batch_size_per_gpu: int = 2,
    num_epochs: int = NUM_EPOCHS,
    context_length: int = CONTEXT_LENGTH,
    learning_rate: float = 1e-5,
    warmup_steps: int = 50,
    deepspeed_config: str | None = None,
    hf_token: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> str:
    """Fine-tune Llama-3-8B-Instruct on LongAlpaca-16k with SFT.

    Args:
        compression_config: KV-cache compression configuration.
        output_dir: Directory to save checkpoints.
        num_gpus: Number of GPUs.
        batch_size_per_gpu: Micro-batch size per GPU (smaller for longer context).
        num_epochs: Number of SFT epochs (default 5).
        context_length: Context window (8K-16K).
        learning_rate: Peak learning rate.
        warmup_steps: Linear warmup steps.
        deepspeed_config: Path to DeepSpeed config JSON.
        hf_token: HuggingFace token.
        resume_from_checkpoint: Path to RedPajama checkpoint to continue from.

    Returns:
        Path to the saved checkpoint directory.
    """
    output_dir = Path(output_dir) / compression_config.variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {compression_config.variant_name} on LongAlpaca-16k")
    logger.info(f"  gamma={compression_config.gamma}, epochs={num_epochs}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )

    # Get model config for compression setup
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # Apply spectral compression
    model = apply_compression_to_model(model, compression_config, num_heads, head_dim)

    # Apply LoRA
    lora_config = create_lora_config(rank=8)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_longalpaca_dataset(context_length=context_length, hf_token=hf_token)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size_per_gpu,
        gradient_accumulation_steps=max(1, 64 // (batch_size_per_gpu * num_gpus)),
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed_config,
        report_to="wandb" if torch.distributed.is_available() else "none",
        run_name=f"longalpaca_{compression_config.variant_name}_gamma{compression_config.gamma}",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    logger.info(f"SFT complete. Checkpoint saved to {output_dir / 'final'}")
    return str(output_dir / "final")
