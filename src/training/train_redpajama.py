"""Phase 1: Continued pre-training on RedPajama with spectral KV compression.

Trains the model with spectral KV-cache compression active during the forward
pass. This teaches the model to operate with the lossy reconstructed K/V
states and (for learnable filters) trains the frequency selection mask.

Following FreqKV protocol:
  - 1000 steps of continued pre-training on RedPajama
  - LoRA rank 8 on attention projections
  - DeepSpeed ZeRO-2 for 8x H200
  - Spectral compression applied per-forward-pass
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from ..spectral import CompressionConfig, apply_spectral_compression
from ..spectral.attention import reset_all_caches
from .lora_config import create_lora_config

logger = logging.getLogger(__name__)


def train_redpajama(
    config,
    hf_token: str | None = None,
    output_dir: str = "checkpoints",
) -> str:
    """Phase 1: Continued pre-training on RedPajama with spectral compression.

    Args:
        config: ExperimentConfig with model_name, compression settings, etc.
        hf_token: HuggingFace token for gated models.
        output_dir: Directory to save checkpoints.

    Returns:
        Path to the saved checkpoint directory.
    """
    model_name = config.model_name
    logger.info(f"Phase 1: RedPajama CPT on {model_name}")
    logger.info(
        f"Compression: {config.transform_type}/{config.filter_type}, "
        f"gamma={config.gamma}"
    )

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model with FlashAttention-2
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # 3. Apply spectral compression BEFORE LoRA
    #    This inserts the spectral cache into each attention layer
    comp_config = CompressionConfig(
        transform_type=config.transform_type,
        filter_type=config.filter_type,
        gamma=config.gamma,
        max_seq_len=config.max_seq_len,
        init_sharpness=config.init_sharpness,
        init_offset=config.init_offset,
    )
    model = apply_spectral_compression(model, comp_config)

    # 4. Apply LoRA (spectral_cache modules saved via modules_to_save)
    lora_config = create_lora_config(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Load and tokenize RedPajama
    logger.info("Loading RedPajama dataset")
    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-1T-Sample",
        split="train",
    )

    def tokenize_fn(examples):
        # RedPajama sample has 'text' field
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_len,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=8,
    )

    # Data collator for causal LM (handles padding)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{config.config_id}/phase1_redpajama",
        num_train_epochs=1,
        max_steps=config.redpajama_steps,
        per_device_train_batch_size=config.batch_size_per_gpu,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        deepspeed=config.deepspeed_config,
        report_to="wandb",
        run_name=f"{config.config_id}_phase1_redpajama",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # 7. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    logger.info("Starting Phase 1 training")
    trainer.train()

    # 8. Save checkpoint
    ckpt_dir = Path(output_dir) / config.config_id / "phase1_redpajama" / "final"
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    logger.info(f"Phase 1 checkpoint saved to {ckpt_dir}")

    return str(ckpt_dir)
