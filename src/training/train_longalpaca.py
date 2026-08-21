"""Phase 2: Supervised fine-tuning on LongAlpaca-16k with spectral KV compression.

Fine-tunes the Phase 1 checkpoint on long-context instruction data.
The spectral compression remains active, teaching the model to handle
long contexts with the compressed KV cache.

Following FreqKV protocol:
  - 5 epochs of SFT on LongAlpaca-16k
  - Context length up to 16K tokens
  - LoRA adapters from Phase 1 are loaded and continued
  - Spectral compression applied per-forward-pass
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
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


def train_longalpaca(
    config,
    phase1_checkpoint: str,
    hf_token: str | None = None,
    output_dir: str = "checkpoints",
) -> str:
    """Phase 2: SFT on LongAlpaca-16k with spectral compression.

    Args:
        config: ExperimentConfig with model_name, compression settings, etc.
        phase1_checkpoint: Path to Phase 1 (RedPajama) checkpoint directory.
        hf_token: HuggingFace token for gated models.
        output_dir: Directory to save checkpoints.

    Returns:
        Path to the saved Phase 2 checkpoint directory.
    """
    model_name = config.model_name
    logger.info(f"Phase 2: LongAlpaca SFT on {model_name}")
    logger.info(
        f"Compression: {config.transform_type}/{config.filter_type}, "
        f"gamma={config.gamma}"
    )

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load base model with eager attention.
    #    Spectral forward overrides LlamaAttention.forward entirely with
    #    manual attention, so FA2 is never used for compressed computation.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )

    # 3. Apply spectral compression BEFORE loading LoRA adapters
    comp_config = CompressionConfig(
        transform_type=config.transform_type,
        filter_type=config.filter_type,
        gamma=config.gamma,
        max_seq_len=config.max_seq_len,
        init_sharpness=config.init_sharpness,
        init_offset=config.init_offset,
    )
    model = apply_spectral_compression(model, comp_config)

    # 4. Load Phase 1 LoRA adapters
    #    The spectral_cache module is saved by modules_to_save, so it
    #    will be loaded from the Phase 1 checkpoint.
    model = PeftModel.from_pretrained(model, phase1_checkpoint)
    logger.info(f"Loaded Phase 1 LoRA adapters from {phase1_checkpoint}")

    # 5. Initialize W&B for this phase (deterministic ID for crash recovery)
    try:
        from ..utils.wandb_utils import init_wandb, finish_wandb
        init_wandb(config, phase="phase2_longalpaca")
        use_wandb = True
    except Exception as e:
        logger.warning(f"W&B init failed: {e}. Continuing without W&B.")
        use_wandb = False

    # 6. Load and tokenize LongAlpaca-16k
    logger.info("Loading LongAlpaca-16k dataset")
    dataset = load_dataset("Yukang/LongAlpaca-16k", split="train")

    def format_instruction(examples):
        """Format LongAlpaca examples as instruction-response pairs."""
        texts = []
        for question, answer in zip(examples["question"], examples["answer"]):
            text = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{question}\n\n"
                f"### Response:\n{answer}"
            )
            texts.append(text)
        return {"text": texts}

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.longalpaca_context_length,
            padding=False,
        )

    dataset = dataset.map(format_instruction, batched=True, remove_columns=dataset.column_names)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=8)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{config.config_id}/phase2_longalpaca",
        num_train_epochs=config.longalpaca_epochs,
        per_device_train_batch_size=config.batch_size_per_gpu,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=config.longalpaca_epochs,
        deepspeed=config.deepspeed_config,
        report_to="wandb",
        run_name=f"{config.config_id}_phase2_longalpaca",
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
            logger.info(f"Resuming Phase 2 from {resume_ckpt}")

    logger.info("Starting Phase 2 training")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 8. Save checkpoint
    ckpt_dir = Path(output_dir) / config.config_id / "phase2_longalpaca" / "final"
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    # Write completion marker for orchestrator
    (ckpt_dir / ".training_complete").touch()
    logger.info(f"Phase 2 checkpoint saved to {ckpt_dir}")

    # Finish W&B run for this phase
    if use_wandb:
        try:
            finish_wandb({"final_checkpoint": str(ckpt_dir)})
        except Exception as e:
            logger.warning(f"W&B finish failed: {e}")

    return str(ckpt_dir)
