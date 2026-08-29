"""Spectral KV Compressor: stateless frequency-domain KV compression.

Following the FreqKV paradigm (arXiv:2505.00570, ICLR 2026), extended with
learnable spectral filtering and complex FFT support.

The compressor is a stateless utility that holds the spectral transform
(DCT or FFT) and optional learnable filter. It does NOT store KV state —
KV state is managed by standard HF DynamicCache during generation, and
by the forward pass during training.

Compression flow (per FreqKV's dct_compress):
  1. Transform K/V to spectral domain.
  2. Apply learnable filter (if present) on the FULL spectrum.
  3. Truncate to compress_len coefficients.
  4. Inverse transform to spatial domain at length compress_len.
  5. Scale by sqrt(compress_len / original_len).

The compressed output has compress_len positions (NOT original_len with
lossy reconstruction). This is the key difference from the old SpectralKVCache
which reconstructed to full length, causing the causality violation.

The compressor is registered as a submodule named "spectral_cache" on each
attention layer, preserving LoRA modules_to_save compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn

from .transform import SpectralTransform, DCTTransform, FFTTransform
from .filter import SpectralFilter, FixedLowPassFilter, LearnableSpectralFilter

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for spectral KV-cache compression.

    Attributes:
        transform_type: "dct", "fft", or "none" (baseline).
        filter_type: "fixed", "learnable", or "none".
        gamma: Compression ratio (0 < gamma <= 1). Fraction of spectral
               coefficients retained. 1.0 = no compression.
        max_seq_len: Maximum sequence length for spectral resolution.
        init_sharpness: Initial sharpness of learnable filter's low-pass.
        init_offset: Initial frequency offset of learnable filter's cutoff.
        sink_size: Number of initial tokens kept uncompressed (attention sink).
        recent_size: Number of recent tokens kept uncompressed (sliding window).
        cache_size: Maximum KV cache size. Compression triggers when the cache
                    exceeds this size. Should be <= training seq_len for
                    compression to activate.
        use_flash_attn: Whether to use FlashAttention-2 for chunk attention.
    """
    transform_type: str = "fft"
    filter_type: str = "learnable"
    gamma: float = 0.22
    max_seq_len: int = 16384
    init_sharpness: float = 10.0
    init_offset: float = 0.0
    sink_size: int = 4
    recent_size: int = 8
    cache_size: int = 8192
    use_flash_attn: bool = True

    @property
    def is_baseline(self) -> bool:
        return self.transform_type == "none" or self.gamma >= 1.0

    @property
    def variant_name(self) -> str:
        if self.is_baseline:
            return "baseline"
        return f"{self.transform_type}_{self.filter_type}"

    @property
    def fft_span(self) -> int:
        """Number of tokens eligible for compression (cache_size minus sink and recent)."""
        return self.cache_size - self.sink_size - self.recent_size

    @property
    def fft_size(self) -> int:
        """Compressed size (number of spectral coefficients retained)."""
        return int(self.fft_span * self.gamma)

    @property
    def chunk_size(self) -> int:
        """Number of new tokens per chunk after the first cache_size tokens."""
        return self.fft_span - self.fft_size


class SpectralKVCompressor(nn.Module):
    """Stateless spectral KV compressor for a single attention layer.

    Holds the spectral transform and optional learnable filter. Provides a
    compress() method that delegates to the transform. Does NOT store KV state.

    The module is registered as "spectral_cache" on each attention layer
    for LoRA modules_to_save compatibility.

    Compression flow:
      compress(x, compress_len) -> compressed_x:
        spatial [B, H, N, D] -> transform -> filter -> truncate ->
        inverse -> scale -> spatial [B, H, L, D]

    where L = compress_len < N. The compressed tensor has fewer positions
    representing the low-frequency structure of the original.
    """

    def __init__(
        self,
        config: CompressionConfig,
        num_kv_heads: int,
        head_dim: int,
    ):
        """
        Args:
            config: Compression configuration.
            num_kv_heads: Number of KV heads (may differ from query heads with GQA).
            head_dim: Dimension of each attention head.
        """
        super().__init__()

        self.config = config
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        if config.is_baseline:
            # Baseline: no spectral processing
            self.transform: Optional[SpectralTransform] = None
            self.filter: Optional[SpectralFilter] = None
        else:
            # Build the spectral transform
            if config.transform_type == "dct":
                self.transform = DCTTransform(dim=-2)
            elif config.transform_type == "fft":
                self.transform = FFTTransform(dim=-2)
            else:
                raise ValueError(f"Unknown transform_type: {config.transform_type}")

            # Build the spectral filter
            max_spectral_len = self.transform.spectral_len(config.max_seq_len)

            if config.filter_type == "fixed":
                self.filter = FixedLowPassFilter()
            elif config.filter_type == "learnable":
                self.filter = LearnableSpectralFilter(
                    num_heads=num_kv_heads,
                    max_spectral_len=max_spectral_len,
                    init_sharpness=config.init_sharpness,
                    init_offset=config.init_offset,
                )
            else:
                raise ValueError(f"Unknown filter_type: {config.filter_type}")

        # Statistics for logging
        self.compression_stats = {
            "total_compress_calls": 0,
            "total_spatial_elements": 0,
            "total_compressed_elements": 0,
        }

    def compress(
        self,
        x: torch.Tensor,
        compress_len: int,
    ) -> torch.Tensor:
        """Compress a KV tensor from length N to length compress_len.

        Delegates to the spectral transform's compress() method, passing
        the learnable filter (if any) as the filter_fn.

        Args:
            x: Input tensor [B, num_kv_heads, N, head_dim].
            compress_len: Target compressed length L (L < N).

        Returns:
            Compressed tensor [B, num_kv_heads, L, head_dim].
        """
        if self.transform is None:
            # Baseline: no compression
            return x

        # Create filter callable if we have a learnable filter
        filter_fn = None
        if self.filter is not None:
            filter_fn = self.filter.forward

        result = self.transform.compress(
            x, compress_len=compress_len, filter_fn=filter_fn, gamma=self.config.gamma
        )

        self.compression_stats["total_compress_calls"] += 1
        self.compression_stats["total_spatial_elements"] += x.shape[-2] * self.num_kv_heads * self.head_dim
        self.compression_stats["total_compressed_elements"] += result.shape[-2] * self.num_kv_heads * self.head_dim

        return result

    def get_compression_ratio(self) -> float:
        """Return the configured compression ratio (gamma)."""
        return self.config.gamma if not self.config.is_baseline else 1.0

    def get_learnable_parameters(self) -> list[nn.Parameter]:
        """Return learnable parameters from the filter (for LoRA modules_to_save)."""
        if self.filter is not None and hasattr(self.filter, 'filter_logits'):
            return [self.filter.filter_logits]
        return []

    def get_filter_mask(self) -> Optional[torch.Tensor]:
        """Return the current filter mask (sigmoid of logits) for analysis.

        Returns:
            Tensor of shape [num_heads, max_spectral_len] with values in [0, 1],
            or None if no learnable filter.
        """
        if self.filter is not None and hasattr(self.filter, 'get_mask'):
            return self.filter.get_mask()
        return None

    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self.compression_stats = {
            "total_compress_calls": 0,
            "total_spatial_elements": 0,
            "total_compressed_elements": 0,
        }
