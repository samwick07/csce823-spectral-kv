#!/usr/bin/env python3
"""Reconstruct C01 Phase 1 and Phase 2 W&B runs from orchestrator logs.

The original C01 runs landed in the wrong W&B project ('huggingface') because
the HF WandbCallback defaulted to that project when WANDB_PROJECT was not set.
The merged run 99ah9gh2 in the 'huggingface' project contains both phases with
a corrupted step axis (Phase 2 steps restart from 10 at the phase boundary).

This script reconstructs two clean runs from the orchestrator log:
  1. C01_phase1_redpajama: 100 entries, steps 10-1000 (RedPajama CPT)
  2. C01_phase2_longalpaca: 94 entries, steps 10-940 (LongAlpaca SFT)

Usage:
    cd /workspaces/csce823-spectral-kv
    .venv/bin/python3 scripts/fix_wandb_C01.py

Safe to run while training is in progress — only touches C01 runs.
"""

import ast
import os
import re
import sys
from pathlib import Path

WANDB_PROJECT = "csce823-spectral-kv"
ORCHESTRATOR_LOG = "logs/orchestrator_20260823_041331.log"

# Phase boundaries (line numbers in orchestrator log)
PHASE1_START_LINE = 182   # "Starting Phase 1 training"
PHASE1_END_LINE = 1422    # "Phase 1 checkpoint saved"
PHASE2_START_LINE = 1565  # "Starting Phase 2 training"
C01_COMPLETE_LINE = 2783  # "[C01] Training complete."

# Training config
MAX_STEPS_P1 = 1000
MAX_STEPS_P2 = 940
LOGGING_STEPS = 10


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


def parse_loss_entries(text: str, phase: str) -> list[dict]:
    """Parse loss dict lines from orchestrator log text.
    
    Filters out train_runtime summary lines (which have 'train_loss' not 'loss').
    Returns list of dicts with loss, grad_norm, learning_rate, epoch.
    """
    metrics = []
    # Match dicts that have 'loss' key (not 'train_loss')
    pattern = re.compile(r"\{'loss':\s+[\d.]+,[^}]*'epoch':\s+([\d.]+)\}")
    for i, m in enumerate(pattern.finditer(text)):
        d = ast.literal_eval(m.group())
        step = (i + 1) * LOGGING_STEPS
        d["_step"] = step
        metrics.append(d)
    return metrics


def fix_wandb_runs(phase1_metrics: list[dict], phase2_metrics: list[dict]) -> None:
    import wandb
    from wandb.apis.public import Api

    load_env()
    api = Api()
    entity = api.default_entity

    # C01 config
    base_config = {
        "config_id": "C01",
        "variant_name": "dct_fixed_g050",
        "transform_type": "dct",
        "filter_type": "fixed",
        "gamma": 0.5,
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 1e-5,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "num_gpus": 4,
        "batch_size_per_gpu": 8,
        "gpu": "NVIDIA H200",
        "max_seq_len": 16384,
        "deepspeed": "zero2_4gpu",
    }

    # ---- Phase 1: RedPajama CPT ----
    p1_config = dict(base_config)
    p1_config.update({
        "phase": "phase1_redpajama",
        "max_steps": MAX_STEPS_P1,
        "redpajama_steps": 1000,
        "redpajama_batch_size": 64,
        "dataset": "RedPajama",
        "note": "Re-logged from orchestrator log. Original run landed in "
                "'huggingface' project due to missing WANDB_PROJECT env var.",
    })

    p1_run_id = "C01_phase1_redpajama_clean"
    p1_run_name = "C01 Phase 1 RedPajama (clean re-log)"

    # Delete any existing runs with this ID
    for old_id in [p1_run_id]:
        try:
            old = api.run(f"{entity}/{WANDB_PROJECT}/{old_id}")
            print(f"  Deleting existing: {old.url}")
            old.delete()
        except Exception:
            pass

    print(f"\n--- Phase 1: {len(phase1_metrics)} entries ---")
    run = wandb.init(
        id=p1_run_id,
        project=WANDB_PROJECT,
        entity=entity,
        name=p1_run_name,
        config=p1_config,
        tags=["dct", "fixed", "gamma_0.5", "phase1_redpajama", "re-logged"],
        notes="CSCE 823: C01 (DCT fixed gamma=0.5) - Phase 1 RedPajama CPT (re-logged)",
        resume="never",
        dir=str(Path.cwd() / "wandb"),
    )
    for m in phase1_metrics:
        step = m.pop("_step")
        logged = {f"train/{k}": v for k, v in m.items()}
        wandb.log(logged, step=step)
    if phase1_metrics:
        wandb.summary["train/loss"] = phase1_metrics[-1].get("loss", 0)
    wandb.finish(exit_code=0)
    print(f"  Done: https://wandb.ai/{entity}/{WANDB_PROJECT}/runs/{p1_run_id}")

    # ---- Phase 2: LongAlpaca SFT ----
    p2_config = dict(base_config)
    p2_config.update({
        "phase": "phase2_longalpaca",
        "max_steps": MAX_STEPS_P2,
        "longalpaca_epochs": 5,
        "longalpaca_context_length": 16384,
        "gradient_accumulation_steps": 16,
        "dataset": "Yukang/LongAlpaca-12k",
        "dataset_rows": 12000,
        "deepspeed": "zero2_4gpu_longctx",
        "note": "Re-logged from orchestrator log. Original run landed in "
                "'huggingface' project due to missing WANDB_PROJECT env var.",
    })

    p2_run_id = "C01_phase2_longalpaca_clean"
    p2_run_name = "C01 Phase 2 LongAlpaca (clean re-log)"

    for old_id in [p2_run_id]:
        try:
            old = api.run(f"{entity}/{WANDB_PROJECT}/{old_id}")
            print(f"  Deleting existing: {old.url}")
            old.delete()
        except Exception:
            pass

    print(f"\n--- Phase 2: {len(phase2_metrics)} entries ---")
    run = wandb.init(
        id=p2_run_id,
        project=WANDB_PROJECT,
        entity=entity,
        name=p2_run_name,
        config=p2_config,
        tags=["dct", "fixed", "gamma_0.5", "phase2_longalpaca", "re-logged"],
        notes="CSCE 823: C01 (DCT fixed gamma=0.5) - Phase 2 LongAlpaca SFT (re-logged)",
        resume="never",
        dir=str(Path.cwd() / "wandb"),
    )
    for m in phase2_metrics:
        step = m.pop("_step")
        logged = {f"train/{k}": v for k, v in m.items()}
        wandb.log(logged, step=step)
    if phase2_metrics:
        wandb.summary["train/loss"] = phase2_metrics[-1].get("loss", 0)
    wandb.finish(exit_code=0)
    print(f"  Done: https://wandb.ai/{entity}/{WANDB_PROJECT}/runs/{p2_run_id}")


def main():
    root = Path.cwd()
    log_path = root / ORCHESTRATOR_LOG
    if not log_path.exists():
        print(f"ERROR: {ORCHESTRATOR_LOG} not found", file=sys.stderr)
        sys.exit(1)

    full_text = log_path.read_text()
    lines = full_text.split("\n")

    # Extract Phase 1 section
    p1_text = "\n".join(lines[PHASE1_START_LINE - 1 : PHASE1_END_LINE])
    p1_metrics = parse_loss_entries(p1_text, "phase1")
    print(f"Phase 1 (RedPajama CPT): {len(p1_metrics)} entries")
    if p1_metrics:
        print(f"  First: step={p1_metrics[0]['_step']} loss={p1_metrics[0]['loss']}")
        print(f"  Last:  step={p1_metrics[-1]['_step']} loss={p1_metrics[-1]['loss']}")

    # Extract Phase 2 section
    p2_text = "\n".join(lines[PHASE2_START_LINE - 1 : C01_COMPLETE_LINE])
    p2_metrics = parse_loss_entries(p2_text, "phase2")
    print(f"\nPhase 2 (LongAlpaca SFT): {len(p2_metrics)} entries")
    if p2_metrics:
        print(f"  First: step={p2_metrics[0]['_step']} loss={p2_metrics[0]['loss']}")
        print(f"  Last:  step={p2_metrics[-1]['_step']} loss={p2_metrics[-1]['loss']}")

    # Verify
    expected_p1 = MAX_STEPS_P1 // LOGGING_STEPS  # 100
    expected_p2 = MAX_STEPS_P2 // LOGGING_STEPS  # 94
    print(f"\nExpected: P1={expected_p1}, P2={expected_p2}")
    if len(p1_metrics) != expected_p1:
        print(f"WARNING: P1 expected {expected_p1}, got {len(p1_metrics)}", file=sys.stderr)
    if len(p2_metrics) != expected_p2:
        print(f"WARNING: P2 expected {expected_p2}, got {len(p2_metrics)}", file=sys.stderr)

    # Verify step sequences
    p1_steps = [m["_step"] for m in p1_metrics]
    p2_steps = [m["_step"] for m in p2_metrics]
    if p1_steps == list(range(LOGGING_STEPS, MAX_STEPS_P1 + LOGGING_STEPS, LOGGING_STEPS)):
        print("P1 step sequence: 10, 20, ..., 1000 ✓")
    else:
        print(f"P1 step sequence MISMATCH: {p1_steps[:5]}...{p1_steps[-5:]}")
    if p2_steps == list(range(LOGGING_STEPS, MAX_STEPS_P2 + LOGGING_STEPS, LOGGING_STEPS)):
        print("P2 step sequence: 10, 20, ..., 940 ✓")
    else:
        print(f"P2 step sequence MISMATCH: {p2_steps[:5]}...{p2_steps[-5:]}")

    print(f"\nReady to re-log to W&B project '{WANDB_PROJECT}'")
    fix_wandb_runs(p1_metrics, p2_metrics)


if __name__ == "__main__":
    main()
