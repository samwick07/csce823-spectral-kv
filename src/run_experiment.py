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

# Support both `python -m src.run_experiment` and `python src/run_experiment.py`
# (DeepSpeed launches it as a script, breaking relative imports).
try:
    from .spectral import CompressionConfig, apply_spectral_compression
    from .spectral.attention import (
        reset_all_caches,
        get_compression_stats,
        create_spectral_dynamic_cache,
    )
    from .training.train_redpajama import train_redpajama
    from .training.train_longalpaca import train_longalpaca
    from .eval.pg19 import evaluate_pg19
    from .eval.proof_pile import evaluate_proof_pile
    from .eval.longbench import evaluate_longbench
    from .eval.efficiency import measure_efficiency
    from .utils.config import load_config
    from .utils.logging_utils import setup_logging
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from src.spectral import CompressionConfig, apply_spectral_compression
    from src.spectral.attention import (
        reset_all_caches,
        get_compression_stats,
        create_spectral_dynamic_cache,
    )
    from src.training.train_redpajama import train_redpajama
    from src.training.train_longalpaca import train_longalpaca
    from src.eval.pg19 import evaluate_pg19
    from src.eval.proof_pile import evaluate_proof_pile
    from src.eval.longbench import evaluate_longbench
    from src.eval.efficiency import measure_efficiency
    from src.utils.config import load_config
    from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def run_training(
    config,
    hf_token: str | None = None,
) -> str:
    """Run both training phases for an experiment.

    Each phase initializes its own W&B run with a deterministic ID
    (e.g., "C01_phase1_redpajama") so that crash recovery resumes
    the same run instead of creating a duplicate.

    Args:
        config: ExperimentConfig.
        hf_token: HuggingFace token.

    Returns:
        Path to the final Phase 2 checkpoint.
    """
    # Phase 1: RedPajama CPT
    logger.info("=" * 60)
    logger.info("PHASE 1: RedPajama Continued Pre-Training")
    logger.info("=" * 60)
    phase1_ckpt = train_redpajama(config, hf_token=hf_token)

    # Phase 2: LongAlpaca SFT
    logger.info("=" * 60)
    logger.info("PHASE 2: LongAlpaca Supervised Fine-Tuning")
    logger.info("=" * 60)
    phase2_ckpt = train_longalpaca(config, phase1_ckpt, hf_token=hf_token)

    return phase2_ckpt


def run_evaluation(
    config,
    checkpoint: str | None = None,
    seed: int = 0,
    hf_token: str | None = None,
    benchmarks: list[str] | None = None,
) -> dict:
    """Run evaluation suite for a trained model.

    Args:
        config: ExperimentConfig.
        checkpoint: Path to trained checkpoint. If None, loads base model
                   with compression (for baseline or untrained comparison).
        seed: Random seed for evaluation stochasticity.
        hf_token: HuggingFace token.
        benchmarks: Optional list of benchmark names to run. If None, runs all.
                   When running a subset, existing results are merged (not overwritten).

    Returns:
        Dict with all evaluation results.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Initialize W&B for evaluation
    try:
        try:
            from .utils.wandb_utils import init_wandb, log_eval_results, finish_wandb
        except ImportError:
            from src.utils.wandb_utils import init_wandb, log_eval_results, finish_wandb
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

    # Ensure all submodules (including spectral caches added after
    # from_pretrained) are on the correct device. device_map="auto"
    # places the base model, but newly registered spectral_cache submodules
    # start on CPU and need explicit movement.
    target_device = next(model.parameters()).device
    model = model.to(target_device)

    model.eval()

    # Log compression stats
    stats = get_compression_stats(model)
    for s in stats[:3]:  # Log first 3 layers
        logger.info(f"  Layer {s['layer']}: ratio={s['compression_ratio']:.4f}")

    # Create SpectralDynamicCache for incremental KV caching during generation.
    # This makes HF's generate() pass only the new token at each decode step
    # (K=1 incremental caching), reducing generation from O(N^2) to O(N log N).
    # For baseline configs (no compression), this is None (HF uses its own DynamicCache).
    spectral_cache = create_spectral_dynamic_cache(model) if not config.is_baseline else None
    if spectral_cache is not None:
        logger.info("SpectralDynamicCache enabled for incremental generation (K=1)")

    # Determine which benchmarks to run
    all_benchmarks = ["pg19", "proof_pile", "longbench", "efficiency"]
    run_benchmarks = benchmarks if benchmarks else all_benchmarks

    # Load existing results if doing a partial eval (merge mode)
    _results_dir = Path(config.output_dir) / "raw" / config.config_id / f"seed_{seed}"
    _results_file = _results_dir / "all_results.json"
    if benchmarks and _results_file.exists():
        try:
            with open(_results_file) as _f:
                all_results = json.load(_f)
            logger.info(f"Merging with existing results from {_results_file} "
                        f"(running: {run_benchmarks})")
        except (json.JSONDecodeError, OSError) as _e:
            logger.warning(f"Could not load existing results for merge: {_e}")
            all_results = {}
    else:
        all_results = {}

    all_results["config_id"] = config.config_id
    all_results["seed"] = seed
    all_results["model_name"] = model_name
    all_results["compression"] = {
        "transform_type": config.transform_type,
        "filter_type": config.filter_type,
        "gamma": config.gamma,
    }

    # 5. PG-19 evaluation
    if "pg19" in run_benchmarks:
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
    else:
        logger.info("Skipping PG-19 (not in benchmark list)")
# 6. Proof-pile evaluation
    if "proof_pile" in run_benchmarks:
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
    else:
        logger.info("Skipping Proof-pile (not in benchmark list)")
# 7. LongBench evaluation
    if "longbench" in run_benchmarks:
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
            past_key_values=spectral_cache,
        )
        all_results["longbench"] = longbench_results
        if use_wandb:
            log_eval_results(longbench_results, config.config_id, seed)
    else:
        logger.info("Skipping LongBench V1 (not in benchmark list)")
# 8. Efficiency measurement
    if "efficiency" in run_benchmarks:
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
            past_key_values=spectral_cache,
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
    else:
        logger.info("Skipping Efficiency (not in benchmark list)")
# 9. Save results
    results_dir = Path(config.output_dir) / "raw" / config.config_id / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "all_results.json"

    # Convert numpy types for JSON serialization
    def default_serializer(obj):
        if isinstance(obj, (torch.Tensor)):
            return obj.tolist() if obj.numel() < 1000 else f"Tensor{tuple(obj.shape)}"
        if isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        if isinstance(obj, list):
            return obj
        return str(obj)

    # Atomic write: write to temp file, then rename (prevents corruption
    # if two processes ever write the same file concurrently).
    import tempfile
    def _atomic_write_json(path, data):
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=path.stem
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=default_serializer)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    _atomic_write_json(results_file, all_results)

    # Also save individual benchmark results
    for benchmark_name in ["pg19", "proof_pile", "longbench", "efficiency"]:
        if benchmark_name in all_results:
            bench_file = results_dir / f"{benchmark_name}.json"
            _atomic_write_json(bench_file, all_results[benchmark_name])

    logger.info(f"Results saved to {results_dir}")

    if use_wandb:
        finish_wandb()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run a spectral KV-cache compression experiment"
    )
    # --local_rank is injected by DeepSpeed launcher; accept and ignore it
    # (DeepSpeed sets LOCAL_RANK env var, which transformers reads).
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="Local rank (injected by DeepSpeed, set automatically via env)",
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
        "--deepspeed-config",
        type=str,
        default=None,
        help="Override the DeepSpeed config path from the YAML (auto-selected by orchestrator)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark names (e.g. 'pg19,proof_pile'). "
             "If None, runs all benchmarks. Existing results are merged.",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override DeepSpeed config if provided (orchestrator auto-selects based on GPU count)
    if args.deepspeed_config:
        config.deepspeed_config = args.deepspeed_config
        logger.info(f"Overrode DeepSpeed config: {config.deepspeed_config}")

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

    # Parse --benchmarks
    benchmarks = None
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        logger.info(f"Running benchmarks: {benchmarks}")

    if args.mode in ("eval", "full"):
        results = run_evaluation(
            config,
            checkpoint=args.checkpoint,
            seed=args.seed,
            hf_token=args.hf_token,
            benchmarks=benchmarks,
        )
        logger.info("Evaluation complete!")
        logger.info(f"  PG-19 PPL: {results.get('pg19', {}).get('mean_perplexity', 'N/A')}")
        logger.info(f"  Proof-pile PPL: {results.get('proof_pile', {}).get('mean_perplexity', 'N/A')}")
        logger.info(f"  LongBench: {results.get('longbench', {}).get('overall_mean', 'N/A')}")


if __name__ == "__main__":
    main()
