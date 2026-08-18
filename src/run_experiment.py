"""Main experiment runner: orchestrates training and evaluation.

Usage:
    python -m src.run_experiment --config configs/experiment_C01.yaml --mode train
    python -m src.run_experiment --config configs/experiment_C01.yaml --mode eval --seed 0
    python -m src.run_experiment --config configs/experiment_C01.yaml --mode full
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .spectral.attention import CompressionConfig, apply_compression_to_model
from .training.train_redpajama import train_on_redpajama
from .training.train_longalpaca import train_on_longalpaca
from .training.deepspeed_config import save_deepspeed_config
from .eval.pg19 import evaluate_pg19
from .eval.proof_pile import evaluate_proof_pile
from .eval.longbench import evaluate_longbench
from .eval.efficiency import measure_efficiency
from .utils.config import load_config
from .utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run spectral KV-cache experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--mode", choices=["train", "eval", "full"], default="full",
                        help="Run mode: train only, eval only, or full pipeline")
    parser.add_argument("--seed", type=int, default=None,
                        help="Evaluation seed (for eval mode)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated models")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(experiment_name=f"{config.config_id}: {config.variant_name}")

    logger.info(f"Config: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Variant: {config.variant_name}, gamma={config.gamma}")

    # Build compression config
    comp_config = CompressionConfig(
        transform_type=config.transform_type,
        filter_type=config.filter_type,
        gamma=config.gamma,
        max_seq_len=config.max_seq_len,
        init_sharpness=config.init_sharpness,
        init_offset=config.init_offset,
    )

    # Setup DeepSpeed config
    ds_config_path = args.deepspeed
    if ds_config_path is None:
        ds_config_path = str(save_deepspeed_config(
            Path(config.output_dir) / "ds_config.json",
            num_gpus=config.num_gpus,
            batch_size_per_gpu=config.batch_size_per_gpu,
        ))

    # Training phase
    if args.mode in ("train", "full"):
        logger.info("=== Phase 1: RedPajama training ===")
        redpajama_ckpt = train_on_redpajama(
            compression_config=comp_config,
            output_dir=str(Path(config.output_dir) / "checkpoints" / "redpajama"),
            num_gpus=config.num_gpus,
            batch_size_per_gpu=config.batch_size_per_gpu,
            num_steps=config.redpajama_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            deepspeed_config=ds_config_path,
            hf_token=args.hf_token,
        )

        logger.info("=== Phase 2: LongAlpaca SFT ===")
        longalpaca_ckpt = train_on_longalpaca(
            compression_config=comp_config,
            output_dir=str(Path(config.output_dir) / "checkpoints" / "longalpaca"),
            num_gpus=config.num_gpus,
            batch_size_per_gpu=2,  # Smaller for 16K context
            num_epochs=config.longalpaca_epochs,
            context_length=config.longalpaca_context_length,
            learning_rate=config.learning_rate,
            warmup_steps=50,
            deepspeed_config=ds_config_path,
            hf_token=args.hf_token,
            resume_from_checkpoint=redpajama_ckpt,
        )

    # Evaluation phase
    if args.mode in ("eval", "full"):
        seeds = [args.seed] if args.seed is not None else config.seeds

        for seed in seeds:
            logger.info(f"=== Evaluation: seed={seed} ===")
            eval_single_seed(config, comp_config, seed, args.hf_token)


def eval_single_seed(config, comp_config, seed, hf_token):
    """Run all benchmarks for a single seed."""
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )

    # Load LoRA checkpoint
    ckpt_dir = Path(config.output_dir) / "checkpoints" / "longalpaca" / comp_config.variant_name / "final"
    if ckpt_dir.exists():
        model = PeftModel.from_pretrained(model, str(ckpt_dir))

    # Apply compression
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    model = apply_compression_to_model(model, comp_config, num_heads, head_dim)
    model = model.cuda()
    model.eval()

    # Results directory
    results_dir = Path(config.output_dir) / "raw" / config.config_id / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # PG-19
    logger.info("Evaluating PG-19...")
    pg19_results = evaluate_pg19(
        model=model,
        tokenizer=tokenizer,
        num_samples=config.pg19_samples,
        seed=seed,
        hf_token=hf_token,
    )
    _save_result(results_dir / "pg19.json", pg19_results)

    # Proof-pile
    logger.info("Evaluating Proof-pile...")
    proof_results = evaluate_proof_pile(
        model=model,
        tokenizer=tokenizer,
        num_samples=config.proof_pile_samples,
        seed=seed,
        hf_token=hf_token,
    )
    _save_result(results_dir / "proof_pile.json", proof_results)

    # LongBench
    logger.info("Evaluating LongBench V1...")
    lb_results = evaluate_longbench(
        model=model,
        tokenizer=tokenizer,
        tasks=config.longbench_tasks,
        seed=seed,
        hf_token=hf_token,
    )
    _save_result(results_dir / "longbench.json", lb_results)

    # Efficiency
    logger.info("Measuring efficiency...")
    eff_results = measure_efficiency(
        model=model,
        tokenizer=tokenizer,
    )
    _save_result(results_dir / "efficiency.json", {
        "peak_kv_memory_gb": eff_results.peak_kv_memory_gb,
        "decoding_latency_ms_per_token": eff_results.decoding_latency_ms_per_token,
        "compression_overhead_pct": eff_results.compression_overhead_pct,
        "total_decode_time_s": eff_results.total_decode_time_s,
        "num_tokens_generated": eff_results.num_tokens_generated,
    })

    logger.info(f"Seed {seed} evaluation complete. Results in {results_dir}")


def _save_result(path: Path, data: dict):
    """Save evaluation result as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    main()
