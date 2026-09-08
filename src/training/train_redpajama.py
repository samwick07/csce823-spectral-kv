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

import os

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

    # 2. Load model with SDPA attention.
    #    For baseline (C00): SDPA uses PyTorch's fused flash backend, which is
    #    O(S*H*D) memory instead of O(S^2*H). At 16K seq len, eager attention
    #    materializes a [B, 32, 16384, 16384] matrix + fp32 softmax = 48+ GiB;
    #    SDPA's flash kernel never materializes it, using ~2 GiB instead.
    #    For compressed configs (C01-C12): the spectral forward overrides
    #    attention entirely, so this setting is irrelevant — _compute_attention
    #    has its own FA2 → SDPA → manual fallback chain.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,  # DeepSpeed launcher owns device placement
    )

    # 3. Apply spectral compression BEFORE LoRA
    #    This inserts the FreqKV chunk-wise compressor into each attention layer
    comp_config = CompressionConfig(
        transform_type=config.transform_type,
        filter_type=config.filter_type,
        gamma=config.gamma,
        max_seq_len=config.max_seq_len,
        init_sharpness=config.init_sharpness,
        init_offset=config.init_offset,
        sink_size=getattr(config, "sink_size", 4),
        recent_size=getattr(config, "recent_size", 8),
        cache_size=getattr(config, "cache_size", 8192),
        use_flash_attn=getattr(config, "use_flash_attn", True),
    )
    # Phase 1: is_iterate=True (matching FreqKV SFT training protocol)
    model = apply_spectral_compression(model, comp_config, is_iterate=True)

    # 4. Apply LoRA (spectral_cache modules saved via modules_to_save)
    lora_config = create_lora_config(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # PEFT + gradient checkpointing: the checkpointing logic freezes input
    # embeddings to save memory, but LoRA only trains adapter weights — so
    # without enable_input_require_grads() the backward chain breaks at the
    # frozen embeddings and the loss tensor loses grad_fn. DeepSpeed then
    # asserts "loss must be a scalar tensor" because grad_fn is None.
    model.enable_input_require_grads()

    # 5. Initialize W&B for this phase (deterministic ID for crash recovery)
    try:
        from ..utils.wandb_utils import init_wandb, finish_wandb
        init_wandb(config, phase="phase1_redpajama")
        use_wandb = True
    except Exception as e:
        logger.warning(f"W&B init failed: {e}. Continuing without W&B.")
        use_wandb = False

    # Safety net: the HF Trainer's WandbCallback (report_to="wandb") opens
    # its own run in the default "huggingface" project unless WANDB_PROJECT
    # is set. Pin it so that even if the explicit init_wandb() above fails,
    # training metrics still land in the experiment project.
    os.environ["WANDB_PROJECT"] = "csce823-spectral-kv"

    # 6. Load and tokenize RedPajama
    logger.info("Loading RedPajama dataset")
    # Original togethercomputer/RedPajama-Data-1T-Sample was removed.
    # liang2kl/RedPajama-Data-1T-Sample-Backup is a parquet mirror of
    # the exact same data (11 parquet shards, no loading script).
    # togethercomputer/RedPajama-Data-1T (full 1T, streaming fallback).
    rp_datasets = [
        ("togethercomputer/RedPajama-Data-1T-Sample", {"split": "train"}),
        ("liang2kl/RedPajama-Data-1T-Sample-Backup", {"split": "train"}),
        ("togethercomputer/RedPajama-Data-1T", {"split": "train", "streaming": True}),
    ]
    dataset = None
    for ds_name, ds_kwargs in rp_datasets:
        try:
            dataset = load_dataset(ds_name, **ds_kwargs)
            if ds_kwargs.get("streaming"):
                from itertools import islice
                dataset = list(islice(dataset, 80000))
                logger.info(f"Loaded {len(dataset)} samples from {ds_name} (streaming)")
            else:
                logger.info(f"Loaded RedPajama from {ds_name}: {len(dataset)} rows")
            break
        except Exception as e:
            logger.warning(f"Could not load {ds_name}: {e}")
    if dataset is None:
        raise RuntimeError(
            "Could not load RedPajama dataset. Tried: "
            + ", ".join(d[0] for d in rp_datasets)
        )

    # Phase 1 RedPajama: use 2048-token sequences for CPT.
    # The config's max_seq_len (16384) is for Phase 2 long-context training.
    # Using 16384 here with batch_size=8 + eager attention would require
    # ~136 GiB for attention weights alone (32 heads × 16384² × 2 bytes),
    # causing CUDA OOM on 140 GiB H200s.
    # At 2048: 32 × 2048² × 2 = ~268 MiB/sample, ~2 GiB for batch 8.
    redpajama_seq_len = 2048

    def tokenize_fn(examples):
        # RedPajama sample has 'text' field
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=redpajama_seq_len,
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
    # Transformers 4.57: the Trainer auto-detects **kwargs in the model's
    # forward signature and injects num_items_in_batch as a loss kwarg.
    # With PEFT-wrapped models + DeepSpeed ZeRO-2 + gradient checkpointing,
    # this causes the loss tensor to lose grad_fn, triggering:
    #   AssertionError: loss must be a scalar tensor
    # Disabling this reverts to the pre-4.57 loss computation path.
    trainer.model_accepts_loss_kwargs = False

    # Auto-resume: check for existing checkpoints in the output_dir.
    # Sort numerically by step: lexicographic sort picks checkpoint-500 over
    # checkpoint-1000 (since '5' > '1'), resuming from an older checkpoint.
    resume_ckpt = None
    ckpt_base = Path(training_args.output_dir)
    if ckpt_base.exists():
        checkpoints = sorted(
            ckpt_base.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )
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
