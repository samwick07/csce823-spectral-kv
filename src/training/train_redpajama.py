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

import json
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

    # 2. Load model with eager attention.
    #    We override LlamaAttention.forward entirely with the spectral
    #    compression path (manual Q@K^T + softmax), so FlashAttention-2
    #    is never used for the compressed attention computation. Using
    #    "eager" avoids version-specific SDPA flag confusion.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,  # DeepSpeed launcher owns device placement
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

    # 5. Initialize W&B for this phase (deterministic ID for crash recovery)
    try:
        from ..utils.wandb_utils import init_wandb, finish_wandb
        init_wandb(config, phase="phase1_redpajama")
        use_wandb = True
    except Exception as e:
        logger.warning(f"W&B init failed: {e}. Continuing without W&B.")
        use_wandb = False

    # 6. Load and tokenize RedPajama
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

    # 7. Training arguments
    # Gradient accumulation must match the DeepSpeed config's
    # train_batch_size: global = per_gpu x num_gpus x accumulation.
    # (4-GPU config uses accumulation=2 to preserve the 8-GPU global batch of 64.)
    try:
        _ds = json.loads(Path(config.deepspeed_config).read_text())
        grad_accum = int(_ds.get("gradient_accumulation_steps", 1))
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read DeepSpeed config for accumulation: {e}; assuming 1")
        grad_accum = 1

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{config.config_id}/phase1_redpajama",
        num_train_epochs=1,
        max_steps=config.redpajama_steps,
        per_device_train_batch_size=config.batch_size_per_gpu,
        gradient_accumulation_steps=grad_accum,
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

    # 8. Train (with automatic resume from latest checkpoint)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # Auto-resume: check for existing checkpoints in the output_dir
    resume_ckpt = None
    ckpt_base = Path(training_args.output_dir)
    if ckpt_base.exists():
        checkpoints = sorted(ckpt_base.glob("checkpoint-*"))
        if checkpoints:
            resume_ckpt = str(checkpoints[-1])
            logger.info(f"Resuming Phase 1 from {resume_ckpt}")

    logger.info("Starting Phase 1 training")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 8. Save checkpoint
    ckpt_dir = Path(output_dir) / config.config_id / "phase1_redpajama" / "final"
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    # Write completion marker for orchestrator
    (ckpt_dir / ".training_complete").touch()
    logger.info(f"Phase 1 checkpoint saved to {ckpt_dir}")

    # Finish W&B run for this phase
    if use_wandb:
        try:
            finish_wandb({"final_checkpoint": str(ckpt_dir)})
        except Exception as e:
            logger.warning(f"W&B finish failed: {e}")

    return str(ckpt_dir)
