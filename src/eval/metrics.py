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
    seq_len: int = 2048,
) -> float:
    """Compute perplexity using seq_len-sized chunks with sliding windows within.

    Following FreqKV's evaluation protocol: process the text in seq_len-sized
    chunks (not window_size-sized). This ensures compression is active during
    evaluation when seq_len > cache_size. Within each chunk, loss is computed
    on window_size sliding windows.

    Args:
        model: The language model.
        input_ids: Token IDs [1, seq_len] or [seq_len].
        window_size: Size of the sliding window for loss computation (default 256).
        stride: Step size between windows. Defaults to window_size (no overlap).
        device: Device to run on.
        seq_len: Chunk size for processing. Must be >= cache_size for compression
                 to activate. Default 2048.

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

    full_len = input_ids.shape[1]

    # Import here to avoid circular import
    try:
        from ..spectral.attention import reset_all_caches
        has_spectral = True
    except ImportError:
        has_spectral = False

    with torch.no_grad():
        # Process in seq_len-sized chunks
        for chunk_start in range(0, full_len - window_size + 1, seq_len):
            chunk_end = min(chunk_start + seq_len, full_len)
            chunk = input_ids[:, chunk_start:chunk_end].to(device)

            if chunk.shape[1] < window_size:
                continue

            # Reset spectral caches between chunks
            if has_spectral:
                reset_all_caches(model)

            # Forward pass on the full chunk (compression active if seq_len > cache_size)
            outputs = model(chunk)
            logits = outputs.logits

            # Compute loss on sliding windows within the chunk
            for win_start in range(0, chunk.shape[1] - window_size + 1, stride):
                win_end = win_start + window_size

                shift_logits = logits[:, win_start:win_end - 1, :].contiguous()
                shift_labels = chunk[:, win_start + 1:win_end, :].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )

                total_loss += loss.item()
                total_tokens += (win_end - win_start - 1)

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def compute_efficiency_metrics(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    generate_length: int = 128,
    device: str = "cuda",
    past_key_value=None,
) -> EfficiencyMetrics:
    """Measure efficiency metrics: memory, latency, overhead.

    Args:
        model: The language model.
        input_ids: Prompt token IDs [1, prompt_len].
        generate_length: Number of tokens to generate for latency measurement.
        device: Device to run on.
        past_key_value: Optional DynamicCache for KV caching during generation.

    Returns:
        EfficiencyMetrics with measured values.
    """
    model.eval()
    metrics = EfficiencyMetrics()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # Reset cache if provided
    if past_key_value is not None:
        past_key_value.reset()

    # Measure generation latency
    start_time = time.perf_counter()

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=generate_length,
            do_sample=False,
            use_cache=True,
        )
        if past_key_value is not None:
            gen_kwargs["past_key_value"] = past_key_value
        outputs = model.generate(
            input_ids.to(device),
            **gen_kwargs,
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
