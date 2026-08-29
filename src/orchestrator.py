"""Resilient experiment orchestrator with automatic crash recovery.

This is the single entry point for running the full spectral KV-cache
ablation study. It:

  1. Tracks per-config training completion via marker files.
  2. Tracks per-(config, seed) eval completion via result JSONs.
  3. Skips work that is already done.
  4. Resumes from the latest DeepSpeed checkpoint after a crash.
  5. Writes a state file (results/orchestrator_state.json) that
     survives power outages and records progress.
  6. Runs the full pipeline: train all configs → eval all seeds →
     aggregate → statistical analysis.

Usage:
    python -m src.orchestrator                    # full sweep
    python -m src.orchestrator --phase train      # training only
    python -m src.orchestrator --phase eval       # eval only
    python -m src.orchestrator --phase analyze    # stats only
    python -m src.orchestrator --pilot            # 3 configs x 5 seeds
    python -m src.orchestrator --config C01       # single config
    python -m src.orchestrator --seed 0           # single seed (eval only)

Design principles:
  - Idempotent: running twice does the same as running once.
  - Crash-safe: no partial state can corrupt completed work.
  - Self-contained: reads HF_TOKEN and WANDB_API_KEY from env.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
STATE_FILE = RESULTS_DIR / "orchestrator_state.json"
LOG_DIR = PROJECT_ROOT / "logs"

ALL_CONFIG_IDS = [
    "C00",  # Baseline
    "C01",  # DCT + fixed + 0.50  (FreqKV reproduction — safest, validates spectral path)
    "C07",  # FFT + fixed + 0.50  (phase preservation vs DCT, same gamma)
    "C10",  # FFT + learnable + 0.50  (full proposed method — headline result)
    "C04",  # DCT + learnable + 0.50  (completes 2x2 at gamma=0.5)
    "C02",  # DCT + fixed + 0.22  (ratio exploration begins)
    "C08",  # FFT + fixed + 0.22  (phase preservation at higher compression)
    "C11",  # FFT + learnable + 0.22  (proposed method at 0.22)
    "C05",  # DCT + learnable + 0.22  (completes 2x2 at 0.22)
    "C03",  # DCT + fixed + 0.01  (aggressive — may not converge)
    "C09",  # FFT + fixed + 0.01  (aggressive — may not converge)
    "C12",  # FFT + learnable + 0.01  (aggressive — may not converge)
    "C06",  # DCT + learnable + 0.01  (aggressive — may not converge)
]
PILOT_CONFIG_IDS = ["C00", "C07", "C10"]
PILOT_SEEDS = list(range(5))


def _default_num_seeds() -> int:
    """Full-run seed count from the config YAML (single source of truth).

    num_seeds defaults to 30 in the YAMLs. Override at runtime with
    --seeds (e.g. --seeds 0 for a single-seed point-estimate run).
    """
    try:
        import yaml

        with open(CONFIGS_DIR / "experiment_C00.yaml") as f:
            data = yaml.safe_load(f) or {}
        return max(1, int(data.get("num_seeds", 30)))
    except (OSError, ValueError, ImportError) as e:
        logger.warning(f"Could not read num_seeds from config (defaulting to 30): {e}")
        return 30


ALL_SEEDS = list(range(_default_num_seeds()))

# Graceful shutdown: set to True by signal handler
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning(f"Signal {signum} received. Will shut down after current "
                   f"subprocess completes. Press Ctrl+C again to force-kill.")
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Second Ctrl+C kills immediately


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    """Load orchestrator state from disk (survives crashes)."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"State file corrupted, starting fresh: {e}")
    return {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "last_update": None,
        "completed_training": [],
        "completed_evals": [],  # list of "C01_seed0" strings
        "completed_analysis": False,
        "completed_exfil": False,
        "current_task": None,
        "crash_count": 0,
    }


def save_state(state: dict[str, Any]) -> None:
    """Save orchestrator state to disk (atomic write)."""
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.rename(STATE_FILE)  # Atomic on POSIX


# ---------------------------------------------------------------------------
# Completion checks
# ---------------------------------------------------------------------------

def is_training_complete(config_id: str) -> bool:
    """Check if both training phases are complete for a config."""
    for phase in ["phase1_redpajama", "phase2_longalpaca"]:
        marker = CHECKPOINT_DIR / config_id / phase / "final" / ".training_complete"
        if not marker.exists():
            return False
    return True


def is_phase1_complete(config_id: str) -> bool:
    """Check if Phase 1 (RedPajama) training is complete."""
    marker = CHECKPOINT_DIR / config_id / "phase1_redpajama" / "final" / ".training_complete"
    return marker.exists()


def is_eval_complete(config_id: str, seed: int) -> bool:
    """Check if evaluation for a given config+seed is complete.

    An eval is complete when the all_results.json file exists, is valid JSON,
    and all four benchmark sections (pg19, proof_pile, longbench, efficiency)
    are present with non-error results. This prevents silently accepting
    partial results where some benchmarks failed (e.g. LongBench OOM).
    """
    result_file = (
        RESULTS_DIR / "raw" / config_id / f"seed_{seed}" / "all_results.json"
    )
    if not result_file.exists():
        return False
    try:
        with open(result_file) as f:
            data = json.load(f)
        if "config_id" not in data or "seed" not in data:
            return False
        # Verify all four benchmark sections are present and not errored
        for bench in ("pg19", "proof_pile", "longbench", "efficiency"):
            if bench not in data:
                return False
            bench_data = data[bench]
            if isinstance(bench_data, dict) and "error" in bench_data:
                return False
            # Check for LongBench all-task-failure (overall_mean=0 with all
            # tasks having errors)
            if bench == "longbench":
                tasks = bench_data.get("tasks", {})
                if tasks and all("error" in t for t in tasks.values()):
                    return False
                if not tasks:
                    return False
        return True
    except (json.JSONDecodeError, OSError):
        return False




def is_eval_phase_complete(
    config_id: str, seed: int, phase: str
) -> bool:
    """Check if a specific eval phase is complete for a config+seed.

    Phase "quick":   pg19 + proof_pile + efficiency present and non-error.
    Phase "longbench": longbench present and non-error (tasks exist, not all errors).
    Phase "all":      all four benchmarks present and non-error (same as is_eval_complete).
    """
    result_file = (
        RESULTS_DIR / "raw" / config_id / f"seed_{seed}" / "all_results.json"
    )
    if not result_file.exists():
        return False
    try:
        with open(result_file) as f:
            data = json.load(f)
        if "config_id" not in data or "seed" not in data:
            return False

        if phase in ("quick", "all"):
            for bench in ("pg19", "proof_pile", "efficiency"):
                if bench not in data:
                    return False
                bench_data = data[bench]
                if isinstance(bench_data, dict) and "error" in bench_data:
                    return False

        if phase in ("longbench", "all"):
            if "longbench" not in data:
                return False
            lb = data["longbench"]
            if isinstance(lb, dict) and "error" in lb:
                return False
            tasks = lb.get("tasks", {}) if isinstance(lb, dict) else {}
            if not tasks:
                return False
            if all("error" in t for t in tasks.values()):
                return False

        return True
    except (json.JSONDecodeError, OSError):
        return False

def get_phase2_checkpoint(config_id: str) -> str | None:
    """Get path to Phase 2 checkpoint if it exists."""
    ckpt = CHECKPOINT_DIR / config_id / "phase2_longalpaca" / "final"
    if ckpt.exists():
        return str(ckpt)
    return None


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_subprocess(
    cmd: list[str],
    timeout: int = 86400,
    env: dict[str, str] | None = None,
    log_prefix: str = "",
) -> bool:
    """Run a subprocess, log output, return True on success.

    Respects _shutdown_requested: if set, waits for the current
    subprocess to finish then returns False.

    Args:
        cmd: Command list to execute.
        timeout: Maximum seconds to wait (default: 86400 = 24h).
                 Training a compressed config takes ~8h with FA2;
                 24h gives ample margin for autonomous execution.
        env: Optional environment overrides (merged with os.environ).
        log_prefix: Prefix for log lines (e.g., "[GPU0] ").
    """
    cmd_str = " ".join(cmd)
    logger.info(f"{log_prefix}Running: {cmd_str}")

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            text=True,
            bufsize=1,
            env=full_env,
        )
    except Exception as e:
        logger.error(f"{log_prefix}Failed to start subprocess: {e}")
        return False

    # Stream output to logger
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if line:
                logger.info(f"{log_prefix}  | {line}")
    except Exception as e:
        logger.warning(f"{log_prefix}Error while streaming output: {e}")

    # Wait for process with timeout — catch TimeoutExpired so it
    # doesn't crash the orchestrator (critical for autonomous runs).
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error(
            f"{log_prefix}Subprocess timed out after {timeout}s. Killing."
        )
        proc.kill()
        proc.wait()
        return False

    if proc.returncode != 0:
        logger.error(f"{log_prefix}Subprocess failed with exit code {proc.returncode}")
        return False

    if _shutdown_requested:
        logger.warning(f"{log_prefix}Shutdown requested. Stopping after this subprocess.")
        return False

    return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_config(config_id: str, hf_token: str | None) -> bool:
    """Train a single config (Phase 1 + Phase 2).

    Returns True if training is complete (either just finished or
    was already done).
    """
    if is_training_complete(config_id):
        logger.info(f"[{config_id}] Training already complete, skipping.")
        return True

    config_file = CONFIGS_DIR / f"experiment_{config_id}.yaml"
    if not config_file.exists():
        logger.error(f"[{config_id}] Config file not found: {config_file}")
        return False

    # Determine how many GPUs to use
    num_gpus = _detect_gpu_count()

    # Auto-select the DeepSpeed config matching the actual GPU count.
    # The YAML defaults to the 4-GPU config; if we detect 8 GPUs we must
    # switch to the 8-GPU config (accum=1, micro_bs=8) to keep
    # train_batch_size consistent (global = micro_bs x num_gpus x accum = 64).
    ds_config = _select_deepspeed_config(config_file, num_gpus)

    # Build the training command.
    # DeepSpeed 0.19+ removed __main__, so use the CLI binary directly.
    # Fall back to -m deepspeed for older versions.
    import shutil
    ds_launcher = shutil.which("deepspeed")
    if ds_launcher:
        cmd = [
            ds_launcher,
            f"--num_gpus={num_gpus}",
            "src/run_experiment.py",
            "--config", str(config_file),
            "--mode", "train",
            "--deepspeed-config", ds_config,
        ]
    else:
        cmd = [
            sys.executable, "-m", "deepspeed",
            f"--num_gpus={num_gpus}",
            "src/run_experiment.py",
            "--config", str(config_file),
            "--mode", "train",
            "--deepspeed-config", ds_config,
        ]
    if hf_token:
        cmd.extend(["--hf-token", hf_token])

    logger.info(f"[{config_id}] Starting training on {num_gpus} GPUs (DS: {ds_config})")

    # Retry loop: transient failures (CUDA OOM, network blips) should
    # not permanently fail a config in autonomous mode.
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        if _shutdown_requested:
            logger.warning(f"[{config_id}] Shutdown requested before training attempt {attempt}.")
            return False

        success = run_subprocess(cmd)

        if success and is_training_complete(config_id):
            logger.info(f"[{config_id}] Training complete.")
            return True
        elif success and is_phase1_complete(config_id):
            logger.info(f"[{config_id}] Phase 1 complete, Phase 2 not done. "
                         f"Will resume Phase 2 on next run.")
            return False
        else:
            if attempt < max_retries:
                logger.warning(
                    f"[{config_id}] Training attempt {attempt}/{max_retries} incomplete. "
                    f"Retrying in 30s (will resume from last DeepSpeed checkpoint)..."
                )
                time.sleep(30)
            else:
                logger.error(
                    f"[{config_id}] Training failed after {max_retries} attempts. "
                    f"Will resume from last checkpoint on next run."
                )
                return False

    return False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_config_seed(
    config_id: str,
    seed: int,
    hf_token: str | None,
    gpu_id: int = 0,
) -> bool:
    """Evaluate a single config+seed on a specific GPU.

    Returns True if eval is complete (just finished or already done).
    """
    if is_eval_complete(config_id, seed):
        logger.info(f"[{config_id} seed={seed}] Already complete, skipping.")
        return True

    config_file = CONFIGS_DIR / f"experiment_{config_id}.yaml"
    if not config_file.exists():
        logger.error(f"[{config_id}] Config file not found: {config_file}")
        return False

    # Get checkpoint (None for baseline C00)
    checkpoint = get_phase2_checkpoint(config_id)

    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    cmd = [
        sys.executable, "src/run_experiment.py",
        "--config", str(config_file),
        "--mode", "eval",
        "--seed", str(seed),
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if hf_token:
        cmd.extend(["--hf-token", hf_token])

    logger.info(f"[{config_id} seed={seed} GPU={gpu_id}] Starting eval")
    success = run_subprocess(cmd, env=env, log_prefix=f"[GPU{gpu_id}]")

    if success and is_eval_complete(config_id, seed):
        logger.info(f"[{config_id} seed={seed}] Eval complete.")
        return True
    else:
        logger.error(f"[{config_id} seed={seed}] Eval incomplete.")
        return False


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(seeds: list[int] | None = None) -> bool:
    """Run the analysis phase.

    With a single seed, produces the N=1 point-estimate table (no
    significance tests). With 2+ seeds, runs the full 7-step statistical
    pipeline (aggregate + analyze).
    """
    if seeds is not None and len(seeds) == 1:
        logger.info("Single-seed run: generating N=1 point-estimate table...")
        cmd = [sys.executable, "-m", "src.stats.point_estimates"]
        if not run_subprocess(cmd):
            logger.error("Point-estimate analysis failed.")
            return False
        logger.info("Analysis complete (point estimates).")
        return True

    logger.info("Running aggregation and statistical analysis...")

    # Aggregate
    cmd = [sys.executable, "-m", "src.stats.aggregate"]
    if not run_subprocess(cmd):
        logger.error("Aggregation failed.")
        return False

    # Analyze
    cmd = [sys.executable, "-m", "src.stats.analyze"]
    if not run_subprocess(cmd):
        logger.error("Statistical analysis failed.")
        return False

    logger.info("Analysis complete.")
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _detect_gpu_count() -> int:
    """Detect number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    # Fallback: check nvidia-smi -L
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split("\n") if l])
    except Exception:
        pass
    logger.warning("Could not detect GPU count, defaulting to 4.")
    return 4


def _select_deepspeed_config(config_file: Path, num_gpus: int) -> str:
    """Auto-select the DeepSpeed config matching the actual GPU count.

    The YAML defaults to the 4-GPU config.  If we detect a different GPU
    count we switch to the matching config so that
    train_batch_size = micro_bs x num_gpus x accum remains 64.

    Phase 1 (RedPajama) always uses the standard config.
    Phase 2 (LongAlpaca) uses the longctx variant (micro_bs=2) because
    LongAlpaca sequences are up to 16K tokens and require smaller
    micro-batches to fit in H200 memory.

    Selection matrix:
        8 GPUs → deepspeed_zero2_8gpu.json         (accum=1, micro=8)
        4 GPUs → deepspeed_zero2_4gpu.json         (accum=2, micro=8)
        other  → YAML default (let it fail with a clear message if wrong)

    Phase 2 always reads the longctx variant from the trainer itself
    (train_longalpaca.py hardcodes the longctx path swap), so this
    function only governs Phase 1 / the orchestrator's --deepspeed-config
    override.
    """
    # Read the YAML to see what it currently points to
    try:
        import yaml as _yaml
        with open(config_file) as f:
            data = _yaml.safe_load(f)
        yaml_ds = data.get("deepspeed_config", "")
    except Exception:
        yaml_ds = ""

    # Determine the standard config for this GPU count
    if num_gpus == 8:
        target = "configs/deepspeed_zero2_8gpu.json"
    elif num_gpus == 4:
        target = "configs/deepspeed_zero2_4gpu.json"
    else:
        # Unknown GPU count — keep YAML default, let DS validate
        return yaml_ds

    # If the YAML already points to the correct config, no change needed
    if yaml_ds.endswith(target):
        return yaml_ds

    # Build absolute path relative to project root
    ds_path = PROJECT_ROOT / target
    if ds_path.exists():
        logger.info(
            f"Auto-selected DS config for {num_gpus} GPUs: {target} "
            f"(YAML default was {yaml_ds})"
        )
        return str(ds_path)

    # Fallback: keep YAML default
    logger.warning(
        f"Expected DS config {target} not found; using YAML default {yaml_ds}"
    )
    return yaml_ds


def run_full_sweep(
    config_ids: list[str],
    seeds: list[int],
    hf_token: str | None,
    phase: str = "full",
    eval_phase: str = "all",
) -> bool:
    """Run the full experiment pipeline with auto-resume.

    Args:
        config_ids: List of config IDs to run (e.g., ["C00", "C01", ...]).
        seeds: List of eval seeds.
        hf_token: HuggingFace token.
        phase: "train", "eval", "analyze", or "full".

    Returns:
        True if all requested work is complete.
    """
    state = load_state()
    if state.get("current_task"):
        logger.info(f"Resuming from previous run. Last task: {state['current_task']}")
        state["crash_count"] += 1
    state["current_task"] = "starting"
    save_state(state)

    num_gpus = _detect_gpu_count()
    logger.info(f"=" * 60)
    logger.info(f"  SPECTRAL KV-CACHE COMPRESSION EXPERIMENT")
    logger.info(f"  Configs: {len(config_ids)} ({config_ids[0]}...{config_ids[-1]})")
    logger.info(f"  Seeds per config: {len(seeds)}")
    logger.info(f"  Total eval runs: {len(config_ids) * len(seeds)}")
    logger.info(f"  GPUs: {num_gpus}")
    logger.info(f"  Phase: {phase}")
    logger.info(f"  Eval phase: {eval_phase}")
    logger.info(f"  Start: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"=" * 60)

    # Record expected eval count in state so monitor.sh shows correct
    # denominator even when --seeds overrides the YAML default.
    state["expected_total_evals"] = len(config_ids) * len(seeds)
    save_state(state)

    all_complete = True

    # --- Phase 1: Training ---
    if phase in ("train", "full"):
        for config_id in config_ids:
            if _shutdown_requested:
                logger.warning("Shutdown requested. Exiting.")
                all_complete = False
                break

            task = f"train_{config_id}"
            state["current_task"] = task
            save_state(state)

            logger.info(f"\n{'='*60}")
            logger.info(f"  TRAINING: {config_id}")
            logger.info(f"{'='*60}")

            if is_training_complete(config_id):
                if config_id not in state["completed_training"]:
                    state["completed_training"].append(config_id)
                    save_state(state)
                logger.info(f"[{config_id}] Already trained, skipping.")
                continue

            success = train_config(config_id, hf_token)

            if success:
                if config_id not in state["completed_training"]:
                    state["completed_training"].append(config_id)
                save_state(state)
            else:
                all_complete = False
                if _shutdown_requested:
                    break
                logger.warning(f"[{config_id}] Training incomplete, "
                               f"continuing to next config.")

    # --- Phase 2: Evaluation ---
    if phase in ("eval", "full") and not _shutdown_requested:
        # Fix 5: Flat work queue -- distribute (config_id, seed) pairs across
        # GPUs regardless of which config they belong to. For N=1 this means
        # ceil(13/4)=4 rounds instead of 13 sequential rounds.
        # Previous behavior: outer loop over configs (sequential), inner loop
        # over seeds (parallel). For N=1 this left 3 of 4 GPUs idle.

        # Determine which benchmarks to run based on eval_phase
        if eval_phase == "quick":
            eval_benchmarks = "pg19,proof_pile,efficiency"
            phase_check_fn = lambda cid, s: is_eval_phase_complete(cid, s, "quick")
        elif eval_phase == "longbench":
            eval_benchmarks = "longbench"
            phase_check_fn = lambda cid, s: is_eval_phase_complete(cid, s, "longbench")
        else:
            eval_benchmarks = None  # run all
            phase_check_fn = lambda cid, s: is_eval_complete(cid, s)

        # Build flat work queue of (config_id, seed) pairs
        eval_queue: list[tuple[str, int]] = []
        for config_id in config_ids:
            # Skip eval if training isn't done (unless it's baseline C00)
            if config_id != "C00" and not is_training_complete(config_id):
                logger.warning(f"[{config_id}] Training not complete, "
                               f"skipping eval.")
                all_complete = False
                continue

            for seed in seeds:
                eval_key = f"{config_id}_seed{seed}"
                # Check phase-specific completion first.
                # A config may be in completed_evals from a previous phase
                # (e.g. quick) but still need work for the current phase
                # (e.g. longbench). Only skip if the current phase is actually
                # complete.
                if phase_check_fn(config_id, seed):
                    if eval_key not in state["completed_evals"]:
                        state["completed_evals"].append(eval_key)
                        save_state(state)
                    continue
                eval_queue.append((config_id, seed))

        # Skip already-completed configs
        if not eval_queue:
            logger.info("All evals already complete, skipping eval phase.")
        else:
            total_evals = len(eval_queue)
            logger.info(f"\n{'='*60}")
            logger.info(f"  EVALUATION: {total_evals} runs across {num_gpus} GPUs")
            logger.info(f"  (flat work queue: {total_evals} jobs, "
                        f"{(total_evals + num_gpus - 1) // num_gpus} rounds)")
            logger.info(f"{'='*60}")

            # Dispatch in batches of num_gpus
            batch_size = min(num_gpus, len(eval_queue))
            for batch_start in range(0, len(eval_queue), batch_size):
                if _shutdown_requested:
                    all_complete = False
                    break

                batch = eval_queue[batch_start:batch_start + batch_size]
                batch_num = batch_start // batch_size + 1
                total_batches = (len(eval_queue) + batch_size - 1) // batch_size
                logger.info(f"\n  Eval batch {batch_num}/{total_batches}: "
                            f"{len(batch)} jobs on GPUs 0-{len(batch)-1}")

                # Launch all evals in this batch as parallel subprocesses
                # Each entry: (config_id, seed, gpu_id, proc)
                procs: list[tuple[str, int, int, subprocess.Popen]] = []

                for gpu_id, (config_id, seed) in enumerate(batch):
                    task = f"eval_{config_id}_seed{seed}"
                    state["current_task"] = task
                    save_state(state)

                    config_file = CONFIGS_DIR / f"experiment_{config_id}.yaml"
                    checkpoint = get_phase2_checkpoint(config_id)

                    cmd = [
                        sys.executable, "src/run_experiment.py",
                        "--config", str(config_file),
                        "--mode", "eval",
                        "--seed", str(seed),
                    ]
                    if eval_benchmarks:
                        cmd.extend(["--benchmarks", eval_benchmarks])
                    if checkpoint:
                        cmd.extend(["--checkpoint", checkpoint])
                    if hf_token:
                        cmd.extend(["--hf-token", hf_token])

                    proc_env = os.environ.copy()
                    proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                    logger.info(f"[{config_id} seed={seed} GPU={gpu_id}] Starting eval")
                    try:
                        p = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            cwd=str(PROJECT_ROOT),
                            text=True,
                            bufsize=1,
                            env=proc_env,
                        )
                        procs.append((config_id, seed, gpu_id, p))
                    except Exception as e:
                        logger.error(f"[GPU{gpu_id}] Failed to start eval subprocess: {e}")
                        all_complete = False

                # Wait for all parallel procs to complete, streaming output
                for config_id, seed, gpu_id, p in procs:
                    try:
                        for line in p.stdout:  # type: ignore[union-attr]
                            line = line.rstrip()
                            if line:
                                logger.info(f"[GPU{gpu_id}]  | {line}")
                    except Exception:
                        pass
                    p.wait()

                    if p.returncode == 0 and phase_check_fn(config_id, seed):
                        logger.info(f"[{config_id} seed={seed}] Eval complete.")
                        eval_key = f"{config_id}_seed{seed}"
                        if eval_key not in state["completed_evals"]:
                            state["completed_evals"].append(eval_key)
                        save_state(state)
                    else:
                        logger.error(f"[{config_id} seed={seed}] Eval incomplete "
                                     f"(exit={p.returncode}).")
                        all_complete = False
                        if _shutdown_requested:
                            break

    # --- Phase 3: Analysis ---
    if phase in ("analyze", "full") and not _shutdown_requested:
        task = "analysis"
        state["current_task"] = task
        save_state(state)

        logger.info(f"\n{'='*60}")
        logger.info(f"  STATISTICAL ANALYSIS")
        logger.info(f"{'='*60}")

        success = run_analysis(seeds)
        if success:
            state["completed_analysis"] = True
            save_state(state)
        else:
            all_complete = False

    # --- Phase 4: Exfiltration ---
    if phase in ("analyze", "full") and not _shutdown_requested and state.get("completed_analysis"):
        task = "exfil"
        state["current_task"] = task
        save_state(state)

        logger.info(f"\n{'='*60}")
        logger.info(f"  EXFILTRATION (HuggingFace Hub)")
        logger.info(f"{'='*60}")

        exfil_cmd = ["bash", "scripts/exfil.sh"]
        exfil_success = run_subprocess(exfil_cmd, log_prefix="[exfil]")

        if exfil_success:
            state["completed_exfil"] = True
            save_state(state)
            logger.info("Exfiltration complete. Results uploaded to HF Hub.")
        else:
            all_complete = False
            logger.warning("Exfiltration failed. Run manually: bash scripts/exfil.sh")

    # --- Phase 4b: GitHub Push ---
    if phase in ("analyze", "full") and not _shutdown_requested and state.get("completed_exfil"):
        task = "github_push"
        state["current_task"] = task
        save_state(state)

        logger.info(f"\n{'='*60}")
        logger.info(f"  GITHUB PUSH (Results)")
        logger.info(f"{'='*60}")

        github_cmd = ["bash", "scripts/push_results_github.sh"]
        github_success = run_subprocess(github_cmd, log_prefix="[github]")

        if github_success:
            state["completed_github_push"] = True
            save_state(state)
            logger.info("GitHub push complete. Results committed to repo.")
        else:
            all_complete = False
            logger.warning("GitHub push failed. Run manually: bash scripts/push_results_github.sh")

    # --- Final state ---
    state["current_task"] = "complete" if all_complete else "interrupted"
    save_state(state)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Training complete: {len(state['completed_training'])}/{len(config_ids)}")
    logger.info(f"  Evals complete: {len(state['completed_evals'])}/{len(config_ids) * len(seeds)}")
    logger.info(f"  Analysis complete: {state['completed_analysis']}")
    logger.info(f"  Exfil complete: {state.get('completed_exfil', False)}")
    logger.info(f"  All work done: {all_complete}")
    logger.info(f"  End: {datetime.now(timezone.utc).isoformat()}")

    return all_complete


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resilient orchestrator for spectral KV-cache experiment"
    )
    parser.add_argument(
        "--phase", type=str,
        choices=["train", "eval", "analyze", "full"],
        default="full",
        help="Which phase to run (default: full)",
    )
    parser.add_argument(
        "--eval-phase", type=str,
        choices=["quick", "longbench", "all"],
        default="all",
        help="Eval sub-phase: 'quick' (pg19+proof_pile+efficiency only), "
             "'longbench' (LongBench only, merges with existing results), "
             "or 'all' (default, runs everything)",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run pilot: 3 configs (C00, C07, C10) x 5 seeds",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Single config ID (e.g., C01). Overrides --pilot.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single seed (eval only). Overrides --pilot.",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed list (e.g. '0' for point-estimate runs, "
             "'0,1,2'). Overrides --pilot and ALL_SEEDS.",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current status and exit",
    )

    args = parser.parse_args()

    # Setup logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Print status and exit
    if args.status:
        state = load_state()
        print(f"\nOrchestrator Status")
        print(f"  Start time: {state.get('start_time', 'unknown')}")
        print(f"  Last update: {state.get('last_update', 'unknown')}")
        print(f"  Crash count: {state.get('crash_count', 0)}")
        print(f"  Current task: {state.get('current_task', 'none')}")
        print(f"  Training complete: {state.get('completed_training', [])}")
        print(f"  Evals complete: {len(state.get('completed_evals', []))}")
        print(f"  Analysis complete: {state.get('completed_analysis', False)}")
        print(f"  Exfil complete: {state.get('completed_exfil', False)}")
        return

    # Determine configs and seeds
    if args.config:
        config_ids = [args.config]
        seeds = [args.seed] if args.seed is not None else ALL_SEEDS
    elif args.pilot:
        config_ids = PILOT_CONFIG_IDS
        seeds = PILOT_SEEDS
    else:
        config_ids = ALL_CONFIG_IDS
        seeds = ALL_SEEDS

    # --seeds overrides the seed list (e.g. '0' for N=1 point estimates)
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        if not seeds:
            parser.error("--seeds parsed to an empty list")

    # If --seed is given, only eval that seed
    if args.seed is not None and args.phase == "eval":
        seeds = [args.seed]

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    logger.info(f"Log file: {log_file}")
    logger.info(f"State file: {STATE_FILE}")

    success = run_full_sweep(config_ids, seeds, hf_token, phase=args.phase, eval_phase=args.eval_phase)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
