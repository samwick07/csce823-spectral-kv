#!/usr/bin/env python3
"""Fix the C00 Phase 1 W&B run to faithfully represent the actual training.

The original Phase 1 run (steps 0→1000, loss 2.12→2.06) was logged correctly,
but orchestrator retries re-ran steps 500→1000 twice more (lexicographic
checkpoint sort bug, now fixed). W&B's resume="allow" mode appended those
redundant steps to the same run, corrupting the run metadata (step count,
runtime) even though the final per-step values are correct (overwritten).

This script:
  1. Reads the 100 clean loss entries from c00_phase1_clean_metrics.txt
     (extracted from the original successful training section of the log).
  2. Deletes the corrupted online W&B run C00_phase1_redpajama.
  3. Creates a fresh run with the same deterministic ID.
  4. Re-logs the 100 data points at steps 10, 20, ..., 1000.
  5. Marks the run as finished.

Usage:
    WANDB_API_KEY=... python scripts/fix_wandb_phase1.py

The script is safe to run while Phase 2 training is in progress — it only
touches the Phase 1 run, which has a different deterministic ID.
"""

import ast
import os
import re
import sys
from pathlib import Path

WANDB_PROJECT = "csce823-spectral-kv"
RUN_ID = "C00_phase1_redpajama_r"
RUN_NAME = "C00 Phase 1 RedPajama (clean re-log)"
METRICS_FILE = "c00_phase1_clean_metrics.txt"


def parse_metrics(path: str) -> list[dict]:
    """Parse the Python-dict-formatted loss lines from the training log."""
    lines = Path(path).read_text().strip().split("\n")
    metrics = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            d = ast.literal_eval(line)
            d["step"] = (i + 1) * 10  # logging_steps=10, so step 10, 20, ..., 1000
            metrics.append(d)
        except (ValueError, SyntaxError) as e:
            print(f"WARNING: Could not parse line {i+1}: {e}", file=sys.stderr)
    return metrics


def load_env():
    """Load credentials from /workspaces/.env.spectral."""
    from pathlib import Path
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

    # 1. Delete the corrupted run
    entity = api.default_entity
    run_path = f"{entity}/{WANDB_PROJECT}/{RUN_ID}"

    try:
        old_run = api.run(run_path)
        print(f"Deleting corrupted run: {old_run.url}")
        old_run.delete()
        print("Deleted.")
    except Exception as e:
        print(f"Could not delete old run (may not exist): {e}")

    # 2. Create a fresh run with the same deterministic ID
    config = {
        "config_id": "C00",
        "variant_name": "baseline",
        "transform_type": "none",
        "filter_type": "none",
        "gamma": 1.0,
        "phase": "phase1_redpajama",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_rank": 8,
        "max_steps": 1000,
        "batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "max_seq_len": 2048,
        "deepspeed": "zero2_4gpu",
        "optimizer": "AdamW (torch_adam=true)",
        "note": "Re-logged from clean training log. Original run was corrupted "
                "by orchestrator retries (lexicographic checkpoint sort bug, "
                "now fixed with numeric sort).",
    }

    run = wandb.init(
        id=RUN_ID,
        project=WANDB_PROJECT,
        entity=entity,
        name=RUN_NAME,
        config=config,
        tags=["baseline", "gamma_1.0", "none_none", "phase1_redpajama", "re-logged"],
        notes="CSCE 823: Baseline (C00) - Phase 1 RedPajama CPT (re-logged from clean data)",
        resume="never",
        dir=str(Path.cwd() / "wandb"),
    )

    # 3. Re-log the 100 clean data points (train/ prefix to match HF callback)
    print(f"Re-logging {len(metrics)} data points...")
    for m in metrics:
        step = m.pop("step")
        logged = {f"train/{k}": v for k, v in m.items()}
        wandb.log(logged, step=step)

    # 4. Set summary metrics
    if metrics:
        wandb.summary["train/loss"] = metrics[-1].get("loss", 0)

    # 4. Mark as finished
    wandb.finish(exit_code=0)
    print(f"Done. Run: https://wandb.ai/{entity}/{WANDB_PROJECT}/runs/{RUN_ID}")


def main():
    metrics_path = Path.cwd() / METRICS_FILE
    if not metrics_path.exists():
        print(f"ERROR: {METRICS_FILE} not found in {Path.cwd()}", file=sys.stderr)
        print("Run this script from the project root.", file=sys.stderr)
        sys.exit(1)

    metrics = parse_metrics(str(metrics_path))
    if len(metrics) != 100:
        print(f"WARNING: Expected 100 data points, got {len(metrics)}", file=sys.stderr)

    print(f"Parsed {len(metrics)} clean Phase 1 metrics")
    print(f"  First: {metrics[0]}")
    print(f"  Last:  {metrics[-1]}")

    fix_wandb_run(metrics)


if __name__ == "__main__":
    main()
