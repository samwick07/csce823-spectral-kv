#!/usr/bin/env python3
"""Reconstruct the C00 Phase 2 W&B run from training logs.

The original C00 Phase 2 run was corrupted on the W&B server:
  - Training ran in two segments (killed at step 753, resumed from checkpoint-700)
  - The deterministic run ID was reused with resume="allow", causing step axis corruption
  - Server-side history shows only ~9 malformed data points instead of 94

This script reconstructs the full run from two data sources:
  1. output.log (first segment): 75 entries, steps 10-750, epoch 0.05-3.99
  2. orchestrator log (resumed segment): 24 entries, epoch 3.78-5.0
     - 5 overlap entries (epoch 3.78-3.99) are skipped to avoid duplication
     - 19 new entries (epoch 4.04-5.0, steps 760-940) are appended

Total: 94 clean data points (steps 10, 20, ..., 940) re-logged into a fresh run.

Usage:
    cd /workspaces/csce823-spectral-kv
    .venv/bin/python3 scripts/fix_wandb_phase2.py

Safe to run while training is in progress — only touches the C00 Phase 2 run,
which has a different deterministic ID than any live run.
"""

import ast
import os
import re
import sys
from pathlib import Path

WANDB_PROJECT = "csce823-spectral-kv"
RUN_ID = "C00_phase2_longalpaca_clean"
RUN_NAME = "C00 Phase 2 LongAlpaca (clean re-log)"

# Data sources (relative to project root)
OUTPUT_LOG = "wandb/wandb/run-20260822_195015-C00_phase2_longalpaca-2/files/output.log"
ORCHESTRATOR_LOG = "logs/orchestrator_20260823_021048.log"

# Training config (from experiment_C00.yaml)
MAX_STEPS = 940
LOGGING_STEPS = 10
NUM_EPOCHS = 5
TOTAL_EXPECTED = MAX_STEPS // LOGGING_STEPS  # 94


def parse_output_log(path: str) -> list[dict]:
    """Parse loss dict lines from the first segment's output.log.

    Each line looks like:
      {'loss': 2.4944, 'grad_norm': 2.281, 'learning_rate': 9e-07, 'epoch': 0.05}
    """
    metrics = []
    text = Path(path).read_text()
    pattern = re.compile(r"\{'loss':[^}]+\}")
    for i, m in enumerate(pattern.finditer(text)):
        d = ast.literal_eval(m.group())
        step = (i + 1) * LOGGING_STEPS
        d["_step"] = step
        metrics.append(d)
    return metrics


def parse_orchestrator_log(path: str, skip_epoch_below: float) -> list[dict]:
    """Parse loss dict lines from the resumed segment's orchestrator log.

    Filters out train_runtime summary lines and wandb summary lines.
    Skips entries with epoch <= skip_epoch_below (overlap with first segment).
    """
    metrics = []
    text = Path(path).read_text()
    # Match lines that look like loss dicts (have 'loss' key, not 'train_loss' or 'train/loss')
    pattern = re.compile(r"\{'loss':\s+[\d.]+,[^}]*'epoch':\s+([\d.]+)\}")
    for m in pattern.finditer(text):
        full_match = m.group()
        d = ast.literal_eval(full_match)
        epoch = d["epoch"]
        if epoch <= skip_epoch_below:
            continue  # skip overlap
        # Compute step from epoch: step = epoch / NUM_EPOCHS * MAX_STEPS
        step = round(epoch / NUM_EPOCHS * MAX_STEPS / LOGGING_STEPS) * LOGGING_STEPS
        d["_step"] = step
        metrics.append(d)
    return metrics


def load_env():
    """Load credentials from /workspaces/.env.spectral."""
    env_path = Path("/workspaces/.env.spectral")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            os.environ[k] = v


def fix_wandb_run(metrics: list[dict]) -> None:
    load_env()
    import wandb
    from wandb.apis.public import Api

    api = Api()
    entity = api.default_entity

    # 1. Delete corrupted runs with the same name (both clean and original IDs)
    for old_id in ["C00_phase2_longalpaca", RUN_ID]:
        run_path = f"{entity}/{WANDB_PROJECT}/{old_id}"
        try:
            old_run = api.run(run_path)
            print(f"Deleting corrupted run: {old_run.url}")
            old_run.delete()
            print("  Deleted.")
        except Exception:
            pass  # may not exist

    # 2. Create a fresh run
    config = {
        "config_id": "C00",
        "variant_name": "baseline",
        "transform_type": "none",
        "filter_type": "none",
        "gamma": 1.0,
        "phase": "phase2_longalpaca",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_steps": MAX_STEPS,
        "longalpaca_epochs": NUM_EPOCHS,
        "longalpaca_context_length": 16384,
        "batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "max_seq_len": 16384,
        "deepspeed": "zero2_4gpu_longctx",
        "dataset": "Yukang/LongAlpaca-12k",
        "dataset_rows": 12000,
        "num_gpus": 4,
        "gpu": "NVIDIA H200",
        "note": "Re-logged from training logs. Original run was corrupted by "
                "kill+resume causing step axis collision (resume=allow appended "
                "resumed steps 710-940 on top of existing 10-750).",
    }

    run = wandb.init(
        id=RUN_ID,
        project=WANDB_PROJECT,
        entity=entity,
        name=RUN_NAME,
        config=config,
        tags=["baseline", "gamma_1.0", "none_none", "phase2_longalpaca", "re-logged"],
        notes="CSCE 823: Baseline (C00) - Phase 2 LongAlpaca SFT (re-logged from training logs)",
        resume="never",
        dir=str(Path.cwd() / "wandb"),
    )

    # 3. Re-log all data points
    print(f"Re-logging {len(metrics)} data points...")
    for m in metrics:
        step = m.pop("_step")
        # Rename keys to match HF Trainer W&B callback naming
        logged = {}
        for k, v in m.items():
            logged[f"train/{k}"] = v
        logged["train/epoch"] = m.get("epoch", 0)
        wandb.log(logged, step=step)

    # 4. Log final summary metrics
    if metrics:
        last = metrics[-1]
        wandb.summary["train/loss"] = last.get("loss", 0)
        wandb.summary["train/epoch"] = last.get("epoch", 0)

    # 5. Mark as finished
    wandb.finish(exit_code=0)
    print(f"Done. Run: https://wandb.ai/{entity}/{WANDB_PROJECT}/runs/{RUN_ID}")


def main():
    root = Path.cwd()

    # Check data sources exist
    output_log_path = root / OUTPUT_LOG
    orch_log_path = root / ORCHESTRATOR_LOG

    if not output_log_path.exists():
        print(f"ERROR: {OUTPUT_LOG} not found", file=sys.stderr)
        sys.exit(1)
    if not orch_log_path.exists():
        print(f"ERROR: {ORCHESTRATOR_LOG} not found", file=sys.stderr)
        sys.exit(1)

    # Parse first segment (steps 10-750)
    seg1 = parse_output_log(str(output_log_path))
    print(f"Segment 1 (output.log): {len(seg1)} entries")
    if seg1:
        print(f"  First: step={seg1[0]['_step']} epoch={seg1[0]['epoch']} loss={seg1[0]['loss']}")
        print(f"  Last:  step={seg1[-1]['_step']} epoch={seg1[-1]['epoch']} loss={seg1[-1]['loss']}")

    # Parse resumed segment (skip overlap with segment 1)
    # Segment 1 ends at epoch 3.99 (step 750). Resumed entries at epoch <= 3.99 overlap.
    skip_below = seg1[-1]["epoch"] if seg1 else 0
    seg2 = parse_orchestrator_log(str(orch_log_path), skip_epoch_below=skip_below)
    print(f"\nSegment 2 (orchestrator, after overlap skip): {len(seg2)} entries")
    if seg2:
        print(f"  First: step={seg2[0]['_step']} epoch={seg2[0]['epoch']} loss={seg2[0]['loss']}")
        print(f"  Last:  step={seg2[-1]['_step']} epoch={seg2[-1]['epoch']} loss={seg2[-1]['loss']}")

    # Combine
    all_metrics = seg1 + seg2
    print(f"\nTotal: {len(all_metrics)} entries (expected {TOTAL_EXPECTED})")

    if len(all_metrics) != TOTAL_EXPECTED:
        print(f"WARNING: Expected {TOTAL_EXPECTED} data points, got {len(all_metrics)}", file=sys.stderr)

    # Verify step sequence
    steps = [m["_step"] for m in all_metrics]
    expected_steps = list(range(LOGGING_STEPS, MAX_STEPS + LOGGING_STEPS, LOGGING_STEPS))
    if steps != expected_steps:
        print("WARNING: Step sequence mismatch!", file=sys.stderr)
        print(f"  Expected: {expected_steps[:5]}...{expected_steps[-5:]}", file=sys.stderr)
        print(f"  Got:      {steps[:5]}...{steps[-5:]}", file=sys.stderr)
        # Find gaps
        missing = set(expected_steps) - set(steps)
        if missing:
            print(f"  Missing steps: {sorted(missing)}", file=sys.stderr)
        extra = set(steps) - set(expected_steps)
        if extra:
            print(f"  Extra steps: {sorted(extra)}", file=sys.stderr)
    else:
        print("Step sequence verified: 10, 20, ..., 940 ✓")

    # Confirm before proceeding
    print(f"\nReady to re-log {len(all_metrics)} points to W&B project '{WANDB_PROJECT}'")
    print(f"Run ID: {RUN_ID}")
    print(f"Run name: {RUN_NAME}")
    print()

    fix_wandb_run(all_metrics)


if __name__ == "__main__":
    main()
