"""Fine-tuning on RedPajama: 1,000 steps, batch 64, 8K context.

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
DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
CONTEXT_LENGTH = 8192
BATCH_SIZE = 64
NUM_STEPS = 1000


def load_redpajama_dataset(
    context_length: int = CONTEXT_LENGTH,
    num_samples: int | None = None,
) -> "datasets.Dataset":
    """Load and tokenize RedPajama dataset.

    Args:
        context_length: Sequence length for tokenization.
        num_samples: Optional limit on number of samples.

    Returns:
        Tokenized dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(DATASET_NAME, split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    def tokenize_fn(examples):
        texts = examples.get("text", [])
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=context_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    logger.info(f"Loaded RedPajama: {len(dataset)} samples, context={context_length}")
    return dataset


def train_on_redpajama(
    compression_config: CompressionConfig,
    output_dir: str | Path = "results/checkpoints/redpajama",
    num_gpus: int = 8,
    batch_size_per_gpu: int = 8,
    num_steps: int = NUM_STEPS,
    context_length: int = CONTEXT_LENGTH,
    learning_rate: float = 1e-5,
    warmup_steps: int = 100,
    deepspeed_config: str | None = None,
    hf_token: str | None = None,
) -> str:
    """Fine-tune Llama-3-8B-Instruct on RedPajama with spectral compression.

    Args:
        compression_config: KV-cache compression configuration.
        output_dir: Directory to save checkpoints.
        num_gpus: Number of GPUs.
        batch_size_per_gpu: Micro-batch size per GPU.
        num_steps: Number of training steps (default 1000 per FreqKV protocol).
        context_length: Context window size (default 8192).
        learning_rate: Peak learning rate.
        warmup_steps: Linear warmup steps.
        deepspeed_config: Path to DeepSpeed config JSON.
        hf_token: HuggingFace token for gated model access.

    Returns:
        Path to the saved checkpoint directory.
    """
    output_dir = Path(output_dir) / compression_config.variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {compression_config.variant_name} on RedPajama")
    logger.info(f"  gamma={compression_config.gamma}, steps={num_steps}")

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
    dataset = load_redpajama_dataset(context_length=context_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,  # Controlled by max_steps instead
        max_steps=num_steps,
        per_device_train_batch_size=batch_size_per_gpu,
        gradient_accumulation_steps=BATCH_SIZE // (batch_size_per_gpu * num_gpus),
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_steps=num_steps // 4,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed_config,
        report_to="wandb" if torch.distributed.is_available() else "none",
        run_name=f"redpajama_{compression_config.variant_name}_gamma{compression_config.gamma}",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    logger.info(f"Training complete. Checkpoint saved to {output_dir / 'final'}")
    return str(output_dir / "final")
