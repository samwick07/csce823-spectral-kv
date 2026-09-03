"""DeepSpeed ZeRO-2 configuration for multi-GPU training.

Optimized for 8x NVIDIA H200 (141GB HBM3e each).
"""

from __future__ import annotations

import json
from pathlib import Path


def get_deepspeed_config(
    num_gpus: int = 8,
    batch_size_per_gpu: int = 8,
    grad_accum_steps: int = 1,
    bf16: bool = True,
    gradient_clipping: float = 1.0,
) -> dict:
    """Return DeepSpeed ZeRO-2 config optimized for H200 GPUs.

    Args:
        num_gpus: Number of GPUs (default 8 for AFIT CCR cluster).
        batch_size_per_gpu: Micro-batch size per GPU.
        grad_accum_steps: Gradient accumulation steps.
        bf16: Use bfloat16 (H200 native support).
        gradient_clipping: Max gradient norm.

    Returns:
        DeepSpeed config dictionary.
    """
    config = {
        "bf16": {
            "enabled": bf16,
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": grad_accum_steps,
        "gradient_clipping": gradient_clipping,
        "train_batch_size": batch_size_per_gpu * num_gpus * grad_accum_steps,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }

    # Add optimizer (AdamW with cosine schedule)
    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    }

    # Add scheduler (linear warmup + cosine decay)
    config["scheduler"] = {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-6,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 100,
            "warmup_type": "linear",
        },
    }

    return config


def save_deepspeed_config(
    path: str | Path,
    num_gpus: int = 8,
    batch_size_per_gpu: int = 8,
    grad_accum_steps: int = 1,
) -> Path:
    """Save DeepSpeed config to a JSON file.

    Args:
        path: Output file path.
        num_gpus: Number of GPUs.
        batch_size_per_gpu: Micro-batch size per GPU.
        grad_accum_steps: Gradient accumulation steps.

    Returns:
        Path to the saved config file.
    """
    config = get_deepspeed_config(
        num_gpus=num_gpus,
        batch_size_per_gpu=batch_size_per_gpu,
        grad_accum_steps=grad_accum_steps,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path
