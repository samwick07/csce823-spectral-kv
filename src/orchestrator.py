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

ALL_CONFIG_IDS = [f"C{i:02d}" for i in range(13)]
PILOT_CONFIG_IDS = ["C00", "C07", "C10"]
ALL_SEEDS = list(range(30))
PILOT_SEEDS = list(range(5))

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

    An eval is complete when the all_results.json file exists and is
    valid JSON (not a partial write from a crash).
    """
    result_file = (
        RESULTS_DIR / "raw" / config_id / f"seed_{seed}" / "all_results.json"
    )
    if not result_file.exists():
        return False
    try:
        with open(result_file) as f:
            data = json.load(f)
        # Verify it has the expected top-level keys
        return "config_id" in data and "seed" in data
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

def run_subprocess(cmd: list[str], timeout: int = 36000) -> bool:
    """Run a subprocess, log output, return True on success.

    Respects _shutdown_requested: if set, waits for the current
    subprocess to finish then returns False.
    """
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            text=True,
            bufsize=1,
        )
    except Exception as e:
        logger.error(f"Failed to start subprocess: {e}")
        return False

    # Stream output to logger
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        if line:
            logger.info(f"  | {line}")

    proc.wait(timeout=timeout)

    if proc.returncode != 0:
        logger.error(f"Subprocess failed with exit code {proc.returncode}")
        return False

    if _shutdown_requested:
        logger.warning("Shutdown requested. Stopping after this subprocess.")
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

    # Build the training command
    cmd = [
        sys.executable, "-m", "deepspeed",
        f"--num_gpus={num_gpus}",
        "src/run_experiment.py",
        "--config", str(config_file),
        "--mode", "train",
    ]
    if hf_token:
        cmd.extend(["--hf-token", hf_token])

    logger.info(f"[{config_id}] Starting training on {num_gpus} GPUs")
    success = run_subprocess(cmd)

    if success and is_training_complete(config_id):
        logger.info(f"[{config_id}] Training complete.")
        return True
    elif success and is_phase1_complete(config_id):
        logger.info(f"[{config_id}] Phase 1 complete, Phase 2 not done. "
                     f"Will resume Phase 2 on next run.")
        return False
    else:
        logger.error(f"[{config_id}] Training incomplete. Will resume from "
                     f"last checkpoint on next run.")
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

    cmd = [
        sys.executable,
        "-c",
        f"import os; os.environ['CUDA_VISIBLE_DEVICES']='{gpu_id}'; "
        f"exec(open('src/run_experiment.py').read())",
        "--config", str(config_file),
        "--mode", "eval",
        "--seed", str(seed),
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if hf_token:
        cmd.extend(["--hf-token", hf_token])

    # Use direct python call instead of the hacky -c approach
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
    success = run_subprocess(cmd)

    if success and is_eval_complete(config_id, seed):
        logger.info(f"[{config_id} seed={seed}] Eval complete.")
        return True
    else:
        logger.error(f"[{config_id} seed={seed}] Eval incomplete.")
        return False


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis() -> bool:
    """Run aggregation and statistical analysis."""
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


def run_full_sweep(
    config_ids: list[str],
    seeds: list[int],
    hf_token: str | None,
    phase: str = "full",
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
    logger.info(f"  Start: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"=" * 60)

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
        # Distribute eval across GPUs
        # Strategy: iterate configs, for each config run seeds in parallel
        # across available GPUs (up to num_gpus at a time).
        for config_id in config_ids:
            if _shutdown_requested:
                logger.warning("Shutdown requested. Exiting.")
                all_complete = False
                break

            # Skip eval if training isn't done (unless it's baseline C00)
            if config_id != "C00" and not is_training_complete(config_id):
                logger.warning(f"[{config_id}] Training not complete, "
                               f"skipping eval.")
                all_complete = False
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"  EVALUATING: {config_id} ({len(seeds)} seeds)")
            logger.info(f"{'='*60}")

            remaining_seeds = [
                s for s in seeds
                if f"{config_id}_seed{s}" not in state["completed_evals"]
                and not is_eval_complete(config_id, s)
            ]

            if not remaining_seeds:
                logger.info(f"[{config_id}] All evals complete, skipping.")
                continue

            logger.info(f"[{config_id}] {len(remaining_seeds)}/{len(seeds)} "
                        f"seeds remaining.")

            # Run seeds in parallel batches across GPUs
            batch_size = min(num_gpus, len(remaining_seeds))
            for batch_start in range(0, len(remaining_seeds), batch_size):
                if _shutdown_requested:
                    all_complete = False
                    break

                batch = remaining_seeds[batch_start:batch_start + batch_size]
                processes = []

                for gpu_id, seed in enumerate(batch):
                    task = f"eval_{config_id}_seed{seed}"
                    state["current_task"] = task
                    save_state(state)

                    # For sequential simplicity (and crash safety), run one
                    # eval at a time per GPU, cycling through GPUs.
                    # Full parallelism is handled by the outer loop.
                    success = eval_config_seed(config_id, seed, hf_token, gpu_id)

                    if success:
                        eval_key = f"{config_id}_seed{seed}"
                        if eval_key not in state["completed_evals"]:
                            state["completed_evals"].append(eval_key)
                        save_state(state)
                    else:
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

        success = run_analysis()
        if success:
            state["completed_analysis"] = True
            save_state(state)
        else:
            all_complete = False

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

    # If --seed is given, only eval that seed
    if args.seed is not None and args.phase == "eval":
        seeds = [args.seed]

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    logger.info(f"Log file: {log_file}")
    logger.info(f"State file: {STATE_FILE}")

    success = run_full_sweep(config_ids, seeds, hf_token, phase=args.phase)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
