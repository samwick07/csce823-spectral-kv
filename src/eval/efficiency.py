"""Efficiency measurement: memory, latency, compression overhead."""

from __future__ import annotations

import logging
import time

import torch

from .metrics import EfficiencyMetrics, compute_efficiency_metrics

logger = logging.getLogger(__name__)


def measure_efficiency(
    model: torch.nn.Module,
    tokenizer,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    generate_length: int = 128,
    device: str = "cuda",
    baseline_time: float | None = None,
) -> EfficiencyMetrics:
    """Measure efficiency metrics for a compressed model.

    Args:
        model: The language model (possibly with spectral compression).
        tokenizer: Tokenizer.
        prompt: Prompt text for generation.
        generate_length: Number of tokens to generate.
        device: Device to run on.
        baseline_time: Decoding time of the uncompressed baseline (for overhead calc).

    Returns:
        EfficiencyMetrics with measured values.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    metrics = compute_efficiency_metrics(
        model=model,
        input_ids=input_ids,
        generate_length=generate_length,
        device=device,
    )

    # Compute compression overhead relative to baseline
    if baseline_time is not None and baseline_time > 0:
        metrics.compression_overhead_pct = (
            (metrics.total_decode_time_s - baseline_time) / baseline_time * 100
        )

    logger.info(
        f"Efficiency: memory={metrics.peak_kv_memory_gb:.2f}GB, "
        f"latency={metrics.decoding_latency_ms_per_token:.1f}ms/tok, "
        f"overhead={metrics.compression_overhead_pct:.1f}%"
    )

    return metrics
