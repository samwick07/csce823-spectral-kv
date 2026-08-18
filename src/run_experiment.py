"""Main experiment runner: orchestrates training and evaluation.

Usage:
    python -m src.run_experiment --config configs/experiment_C01.yaml --mode train
    python -m src.run_experiment --config configs/experiment_C01.yaml --mode eval
    python -m src.run_experiment --config configs/experiment_C01.yaml --mode full
    python -m src.run_experiment --config configs/experiment_C01.yaml --mode eval --seed 0

Modes:
    train  - Run Phase 1 (RedPajama) + Phase 2 (LongAlpaca) training
    eval   - Run evaluation (PG-19, Proof-pile, LongBench, efficiency)
    full   - Train then eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from .spectral import CompressionConfig, apply_spectral_compression
from .spectral.attention import reset_all_caches, get_compression_stats
from .training.train_redpajama import train_redpajama
from .training.train_longalpaca import train_longalpaca
from .eval.pg19 import evaluate_pg19
from .eval.proof_pile import evaluate_proof_pile
from .eval.longbench import evaluate_longbench
from .eval.efficiency import measure_efficiency
from .utils.config import load_config
from .utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def run_training(
    config,
    hf_token: str | None = None,
) -> str:
    """Run both training phases for an experiment.

    Args:
        config: ExperimentConfig.
        hf_token: HuggingFace token.

    Returns:
        Path to the final Phase 2 checkpoint.
    """
    # Initialize W&B for training
    try:
        from .utils.wandb_utils import init_wandb, log_spectral_stats, finish_wandb
        wandb_run = init_wandb(config, phase="train")
        use_wandb = True
    except Exception as e:
        logger.warning(f"W&B init failed: {e}. Continuing without W&B.")
        use_wandb = False

    # Phase 1: RedPajama CPT
    logger.info("=" * 60)
    logger.info("PHASE 1: RedPajama Continued Pre-Training")
    logger.info("=" * 60)
    phase1_ckpt = train_redpajama(config, hf_token=hf_token)

    # Log spectral stats after Phase 1
    if use_wandb:
        try:
            # Re-apply compression to log filter masks
            # (The model inside the trainer has been saved; we log from checkpoint)
            log_spectral_stats.__wrapped__ if hasattr(log_spectral_stats, "__wrapped__") else None
        except Exception:
            pass

    # Phase 2: LongAlpaca SFT
    logger.info("=" * 60)
    logger.info("PHASE 2: LongAlpaca Supervised Fine-Tuning")
    logger.info("=" * 60)
    phase2_ckpt = train_longalpaca(config, phase1_ckpt, hf_token=hf_token)

    if use_wandb:
        finish_wandb({"final_checkpoint": phase2_ckpt})

    return phase2_ckpt


def run_evaluation(
    config,
    checkpoint: str | None = None,
    seed: int = 0,
    hf_token: str | None = None,
) -> dict:
    """Run evaluation suite for a trained model.

    Args:
        config: ExperimentConfig.
        checkpoint: Path to trained checkpoint. If None, loads base model
                   with compression (for baseline or untrained comparison).
        seed: Random seed for evaluation stochasticity.
        hf_token: HuggingFace token.

    Returns:
        Dict with all evaluation results.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Initialize W&B for evaluation
    try:
        from .utils.wandb_utils import init_wandb, log_eval_results, finish_wandb
        wandb_run = init_wandb(config, phase=f"eval_seed{seed}")
        use_wandb = True
    except Exception as e:
        logger.warning(f"W&B init failed: {e}. Continuing without W&B.")
        use_wandb = False

    model_name = config.model_name
    logger.info(f"Loading model: {model_name}")

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

    # 3. Apply spectral compression
    comp_config = CompressionConfig(
        transform_type=config.transform_type,
        filter_type=config.filter_type,
        gamma=config.gamma,
        max_seq_len=config.max_seq_len,
        init_sharpness=config.init_sharpness,
        init_offset=config.init_offset,
    )
    model = apply_spectral_compression(model, comp_config)

    # 4. Load trained checkpoint if provided
    if checkpoint:
        logger.info(f"Loading LoRA checkpoint: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)

    model.eval()

    # Log compression stats
    stats = get_compression_stats(model)
    for s in stats[:3]:  # Log first 3 layers
        logger.info(f"  Layer {s['layer']}: ratio={s['compression_ratio']:.4f}")

    all_results = {
        "config_id": config.config_id,
        "seed": seed,
        "model_name": model_name,
        "compression": {
            "transform_type": config.transform_type,
            "filter_type": config.filter_type,
            "gamma": config.gamma,
        },
    }

    # 5. PG-19 evaluation
    logger.info("=" * 40)
    logger.info("Evaluating PG-19")
    logger.info("=" * 40)
    reset_all_caches(model)
    pg19_results = evaluate_pg19(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        num_samples=config.pg19_samples,
        window_size=config.eval_window_size,
        seed=seed,
        hf_token=hf_token,
    )
    all_results["pg19"] = pg19_results
    if use_wandb:
        log_eval_results(pg19_results, config.config_id, seed)

    # 6. Proof-pile evaluation
    logger.info("=" * 40)
    logger.info("Evaluating Proof-pile")
    logger.info("=" * 40)
    reset_all_caches(model)
    proof_results = evaluate_proof_pile(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        num_samples=config.proof_pile_samples,
        window_size=config.eval_window_size,
        seed=seed,
        hf_token=hf_token,
    )
    all_results["proof_pile"] = proof_results
    if use_wandb:
        log_eval_results(proof_results, config.config_id, seed)

    # 7. LongBench evaluation
    logger.info("=" * 40)
    logger.info("Evaluating LongBench V1")
    logger.info("=" * 40)
    reset_all_caches(model)
    longbench_tasks = config.longbench_tasks
    longbench_results = evaluate_longbench(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        tasks=longbench_tasks,
        temperature=config.eval_temperature,
        top_p=config.eval_top_p,
        seed=seed,
        hf_token=hf_token,
    )
    all_results["longbench"] = longbench_results
    if use_wandb:
        log_eval_results(longbench_results, config.config_id, seed)

    # 8. Efficiency measurement
    logger.info("=" * 40)
    logger.info("Measuring Efficiency")
    logger.info("=" * 40)
    reset_all_caches(model)
    prompt = "The quick brown fox jumps over the lazy dog. " * 50
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    efficiency_metrics = measure_efficiency(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        generate_length=128,
    )
    all_results["efficiency"] = {
        "peak_kv_memory_gb": efficiency_metrics.peak_kv_memory_gb,
        "decoding_latency_ms_per_token": efficiency_metrics.decoding_latency_ms_per_token,
        "compression_overhead_pct": efficiency_metrics.compression_overhead_pct,
        "total_decode_time_s": efficiency_metrics.total_decode_time_s,
        "num_tokens_generated": efficiency_metrics.num_tokens_generated,
    }
    if use_wandb:
        log_eval_results(all_results["efficiency"], config.config_id, seed)

    # 9. Save results
    results_dir = Path(config.output_dir) / "raw" / config.config_id / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "all_results.json"
    with open(results_file, "w") as f:
        # Convert numpy types for JSON serialization
        def default_serializer(obj):
            if isinstance(obj, (torch.Tensor)):
                return obj.tolist() if obj.numel() < 1000 else f"Tensor{tuple(obj.shape)}"
            if isinstance(obj, (float, int, str, bool, type(None))):
                return obj
            if isinstance(obj, list):
                return obj
            return str(obj)
        json.dump(all_results, f, indent=2, default=default_serializer)

    # Also save individual benchmark results
    for benchmark_name in ["pg19", "proof_pile", "longbench", "efficiency"]:
        if benchmark_name in all_results:
            bench_file = results_dir / f"{benchmark_name}.json"
            with open(bench_file, "w") as f:
                json.dump(all_results[benchmark_name], f, indent=2, default=default_serializer)

    logger.info(f"Results saved to {results_dir}")

    if use_wandb:
        finish_wandb()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run a spectral KV-cache compression experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "full"],
        default="full",
        help="Run mode: train, eval, or full (train+eval)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (for eval mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for evaluation",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set up logging
    setup_logging(
        log_file=args.log_file,
        experiment_name=f"{config.config_id} ({config.variant_name})",
    )

    logger.info(f"Experiment: {config.config_id}")
    logger.info(f"  Transform: {config.transform_type}")
    logger.info(f"  Filter: {config.filter_type}")
    logger.info(f"  Gamma: {config.gamma}")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Mode: {args.mode}")

    if args.mode in ("train", "full"):
        ckpt = run_training(config, hf_token=args.hf_token)
        if args.mode == "full":
            args.checkpoint = ckpt

    if args.mode in ("eval", "full"):
        results = run_evaluation(
            config,
            checkpoint=args.checkpoint,
            seed=args.seed,
            hf_token=args.hf_token,
        )
        logger.info("Evaluation complete!")
        logger.info(f"  PG-19 PPL: {results.get('pg19', {}).get('mean_perplexity', 'N/A')}")
        logger.info(f"  Proof-pile PPL: {results.get('proof_pile', {}).get('mean_perplexity', 'N/A')}")
        logger.info(f"  LongBench: {results.get('longbench', {}).get('overall_mean', 'N/A')}")


if __name__ == "__main__":
    main()
