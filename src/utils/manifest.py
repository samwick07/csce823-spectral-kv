"""Experiment provenance manifest generator.

Captures the full runtime environment, git state, and experiment
configuration for reproducibility. Required for the paper's
"Experimental Setup" section and reproducibility appendix.

Output: results/manifest.json
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _get_git_info() -> dict:
    """Get current git commit hash, branch, and status."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": True, "remote": "unknown"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=10,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=10,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=10,
        )
        info["dirty"] = bool(result.stdout.strip())
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=10,
        )
        if result.returncode == 0:
            info["remote"] = result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git info: {e}")
    return info


def _get_python_info() -> dict:
    """Get Python version and key package versions."""
    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            key_packages = [
                "torch", "transformers", "peft", "deepspeed", "datasets",
                "accelerate", "wandb", "scipy", "pandas", "numpy",
                "scikit-posthocs", "statsmodels", "matplotlib", "seaborn",
                "lm-eval", "lm_eval", "pytest",
            ]
            for pkg in packages:
                name = pkg.get("name", "").lower()
                if name in [k.lower() for k in key_packages]:
                    info["packages"][pkg["name"]] = pkg["version"]
    except Exception as e:
        logger.warning(f"Failed to get package info: {e}")
    return info


def _get_gpu_info() -> list[dict]:
    """Get GPU model, count, driver, and CUDA version."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total": parts[2],
                        "driver_version": parts[3],
                    })
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
    # CUDA version via nvcc or torch
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    gpus.append({"cuda_version": line.strip()})
    except Exception:
        pass
    return gpus


def _get_experiment_config() -> dict:
    """Summarize the experiment matrix and run parameters."""
    try:
        from src.stats.experiment_matrix import EXPERIMENT_MATRIX
        configs = []
        for cfg in EXPERIMENT_MATRIX:
            configs.append({
                "config_id": cfg.config_id,
                "transform": cfg.transform,
                "filter": cfg.filter,
                "gamma": cfg.gamma,
                "variant": cfg.variant_name,
                "is_baseline": cfg.is_baseline,
            })
        return {
            "num_configs": len(configs),
            "num_seeds": 30,
            "total_eval_runs": len(configs) * 30,
            "configs": configs,
        }
    except Exception as e:
        logger.warning(f"Failed to get experiment config: {e}")
        return {"error": str(e)}


def _get_orchestrator_state() -> dict:
    """Read the orchestrator state file for run statistics."""
    state_file = RESULTS_DIR / "orchestrator_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            return {
                "start_time": state.get("start_time"),
                "last_update": state.get("last_update"),
                "crash_count": state.get("crash_count", 0),
                "completed_training": state.get("completed_training", []),
                "completed_evals_count": len(state.get("completed_evals", [])),
                "completed_analysis": state.get("completed_analysis", False),
            }
        except Exception as e:
            logger.warning(f"Failed to read orchestrator state: {e}")
    return {"error": "State file not found"}


def generate_manifest() -> dict:
    """Generate the full experiment manifest.

    Captures: timestamp, git state, Python/packages, GPUs, experiment
    config, orchestrator state. Writes to results/manifest.json.
    """
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "spectral-kv",
        "title": "Spectral KV-Cache Compression via Complex FFT with "
                 "Learnable Frequency Selection: An Ablation Study",
        "git": _get_git_info(),
        "python": _get_python_info(),
        "gpus": _get_gpu_info(),
        "experiment": _get_experiment_config(),
        "orchestrator_state": _get_orchestrator_state(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    manifest = generate_manifest()
    print(json.dumps(manifest, indent=2, default=str))
