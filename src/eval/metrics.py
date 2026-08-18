"""Perplexity and efficiency metric computation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class EfficiencyMetrics:
    """Container for efficiency measurements."""

    peak_kv_memory_gb: float = 0.0
    decoding_latency_ms_per_token: float = 0.0
    compression_overhead_pct: float = 0.0
    total_decode_time_s: float = 0.0
    num_tokens_generated: int = 0


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity from model logits and labels.

    Uses standard cross-entropy loss and converts to perplexity: PPL = exp(loss).

    Args:
        logits: Model output logits [batch, seq_len, vocab_size].
        labels: Ground truth token IDs [batch, seq_len].
        ignore_index: Label value to ignore in loss computation.

    Returns:
        Perplexity value (float).
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    return math.exp(loss.item())


def compute_sliding_window_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    window_size: int = 256,
    stride: int | None = None,
    device: str = "cuda",
) -> float:
    """Compute perplexity using a sliding window approach.

    Following the protocol: sliding window of 256 tokens.
    This evaluates the model's perplexity on long texts by processing
    overlapping windows and averaging the negative log-likelihood.

    Args:
        model: The language model.
        input_ids: Token IDs [1, seq_len] or [seq_len].
        window_size: Size of the sliding window (default 256).
        stride: Step size between windows. Defaults to window_size (no overlap).
        device: Device to run on.

    Returns:
        Average perplexity across all windows.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if stride is None:
        stride = window_size

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    seq_len = input_ids.shape[1]

    with torch.no_grad():
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window = input_ids[:, start:end].to(device)

            outputs = model(window)
            logits = outputs.logits

            # Compute loss for this window
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = window[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += (end - start - 1)

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def compute_efficiency_metrics(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    generate_length: int = 128,
    device: str = "cuda",
) -> EfficiencyMetrics:
    """Measure efficiency metrics: memory, latency, overhead.

    Args:
        model: The language model.
        input_ids: Prompt token IDs [1, prompt_len].
        generate_length: Number of tokens to generate for latency measurement.
        device: Device to run on.

    Returns:
        EfficiencyMetrics with measured values.
    """
    model.eval()
    metrics = EfficiencyMetrics()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # Measure generation latency
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids.to(device),
            max_new_tokens=generate_length,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    metrics.total_decode_time_s = end_time - start_time
    metrics.num_tokens_generated = generate_length
    metrics.decoding_latency_ms_per_token = (
        (metrics.total_decode_time_s / generate_length) * 1000
    )

    # Peak memory
    peak_memory = torch.cuda.max_memory_allocated(device)
    metrics.peak_kv_memory_gb = peak_memory / (1024**3)

    return metrics
